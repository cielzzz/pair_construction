#!/usr/bin/env python
"""Keep selected GPUs busy during low-utilization pipeline stages.

This is a process-level guard for Qizhi jobs that reserve multiple GPUs while
some stages are CPU-bound or use only one GPU. It intentionally does no useful
work: each worker runs repeated CUDA matrix multiplies on one GPU until the
parent process receives SIGTERM/SIGINT.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import signal
import sys
import time
from typing import Iterable


def parse_device_index(value: str | None) -> int | None:
    if not value:
        return None
    value = value.strip()
    if value.startswith("cuda:"):
        value = value.split(":", 1)[1]
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_gpu_list(spec: str, count: int, exclude: int | None) -> list[int]:
    spec = (spec or "auto").strip()
    if spec.lower() == "auto":
        gpus = list(range(count))
        if exclude is not None and len(gpus) > 1:
            gpus = [gpu for gpu in gpus if gpu != exclude]
        return gpus

    values: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            values.extend(range(int(left), int(right) + 1))
        else:
            values.append(int(part))
    return [gpu for gpu in values if 0 <= gpu < count]


def dtype_from_name(torch_mod, name: str):
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch_mod.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch_mod.float16
    if name in {"fp32", "float32"}:
        return torch_mod.float32
    raise ValueError(f"unsupported dtype: {name}")


def reserve_memory(torch_mod, device, reserve_mib: int, dtype) -> list:
    if reserve_mib <= 0:
        return []
    bytes_per_elem = torch_mod.tensor([], dtype=dtype).element_size()
    elems = max(1, reserve_mib * 1024 * 1024 // bytes_per_elem)
    block_elems = 256 * 1024 * 1024 // bytes_per_elem
    blocks = []
    remaining = elems
    while remaining > 0:
        n = min(block_elems, remaining)
        blocks.append(torch_mod.empty((n,), device=device, dtype=dtype))
        remaining -= n
    return blocks


def worker(
    gpu: int,
    stop_event,
    matrix_size: int,
    dtype_name: str,
    active_ms: int,
    idle_ms: int,
    reserve_mib: int,
    log_every_sec: int,
) -> None:
    import torch

    signal.signal(signal.SIGTERM, lambda _sig, _frame: stop_event.set())
    signal.signal(signal.SIGINT, lambda _sig, _frame: stop_event.set())

    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")
    try:
        dtype = dtype_from_name(torch, dtype_name)
        a = torch.randn((matrix_size, matrix_size), device=device, dtype=dtype)
        b = torch.randn((matrix_size, matrix_size), device=device, dtype=dtype)
    except Exception as exc:
        if dtype_name.lower() in {"bf16", "bfloat16"}:
            print(f"[gpu-guard:{gpu}] bfloat16 init failed, falling back to float16: {exc}", flush=True)
            dtype = torch.float16
            a = torch.randn((matrix_size, matrix_size), device=device, dtype=dtype)
            b = torch.randn((matrix_size, matrix_size), device=device, dtype=dtype)
        else:
            raise

    c = torch.empty_like(a)
    reserved = reserve_memory(torch, device, reserve_mib, dtype)
    if reserved:
        print(f"[gpu-guard:{gpu}] reserved_mib={reserve_mib}", flush=True)

    torch.cuda.synchronize(device)
    active_sec = max(0.0, active_ms / 1000.0)
    idle_sec = max(0.0, idle_ms / 1000.0)
    iters = 0
    last_log = time.monotonic()
    print(
        f"[gpu-guard:{gpu}] start matrix={matrix_size} dtype={dtype} "
        f"active_ms={active_ms} idle_ms={idle_ms}",
        flush=True,
    )

    while not stop_event.is_set():
        active_until = time.monotonic() + active_sec
        while not stop_event.is_set() and (active_sec == 0.0 or time.monotonic() < active_until):
            torch.mm(a, b, out=c)
            a, c = c, a
            iters += 1
        torch.cuda.synchronize(device)
        now = time.monotonic()
        if log_every_sec > 0 and now - last_log >= log_every_sec:
            print(f"[gpu-guard:{gpu}] alive iters={iters}", flush=True)
            last_log = now
        if idle_sec > 0:
            stop_event.wait(idle_sec)

    torch.cuda.synchronize(device)
    print(f"[gpu-guard:{gpu}] stop iters={iters}", flush=True)


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default=os.environ.get("GPU_GUARD_GPUS", "auto"))
    ap.add_argument("--exclude-device", default=os.environ.get("GPU_GUARD_EXCLUDE_DEVICE", ""))
    ap.add_argument("--matrix-size", type=int, default=int(os.environ.get("GPU_GUARD_MATRIX_SIZE", "8192")))
    ap.add_argument("--dtype", default=os.environ.get("GPU_GUARD_DTYPE", "bfloat16"))
    ap.add_argument("--active-ms", type=int, default=int(os.environ.get("GPU_GUARD_ACTIVE_MS", "900")))
    ap.add_argument("--idle-ms", type=int, default=int(os.environ.get("GPU_GUARD_IDLE_MS", "150")))
    ap.add_argument("--reserve-mib", type=int, default=int(os.environ.get("GPU_GUARD_RESERVE_MIB", "0")))
    ap.add_argument("--log-every-sec", type=int, default=int(os.environ.get("GPU_GUARD_LOG_EVERY_SEC", "60")))
    args = ap.parse_args(argv)

    try:
        import torch
    except Exception as exc:
        print(f"[gpu-guard] torch import failed: {exc}", file=sys.stderr, flush=True)
        return 2

    if not torch.cuda.is_available():
        print("[gpu-guard] CUDA is not available; guard disabled", flush=True)
        return 0

    count = torch.cuda.device_count()
    exclude = parse_device_index(args.exclude_device)
    gpus = parse_gpu_list(args.gpus, count, exclude)
    if not gpus:
        print(f"[gpu-guard] no GPUs selected from spec={args.gpus!r} count={count}", flush=True)
        return 0

    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()

    def handle_signal(_sig, _frame):
        stop_event.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    print(
        f"[gpu-guard] selected_gpus={gpus} exclude={exclude} pid={os.getpid()}",
        flush=True,
    )
    procs = [
        ctx.Process(
            target=worker,
            args=(
                gpu,
                stop_event,
                args.matrix_size,
                args.dtype,
                args.active_ms,
                args.idle_ms,
                args.reserve_mib,
                args.log_every_sec,
            ),
            daemon=False,
        )
        for gpu in gpus
    ]
    for proc in procs:
        proc.start()

    try:
        while not stop_event.is_set():
            alive = [proc.is_alive() for proc in procs]
            if not any(alive):
                break
            time.sleep(1.0)
    finally:
        stop_event.set()
        for proc in procs:
            proc.join(timeout=10)
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        for proc in procs:
            proc.join(timeout=5)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
