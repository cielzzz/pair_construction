from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Iterator

import soundfile as sf
import torch
import torchaudio


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str | Path, rows: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs-jsonl", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-jsonl", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int, default=3072)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--kv-cache-dtype", default=None)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=0)
    parser.add_argument("--preprocess-cache-size", type=int, default=512)
    parser.add_argument("--prepare-workers", type=int, default=1)
    parser.add_argument("--audio-condition-build-workers", type=int, default=0)
    parser.add_argument("--disable-audio-condition-item-parallel", action="store_true")
    parser.add_argument("--audio-condition-cache-dir", default="")
    parser.add_argument("--disable-audio-condition-cache-writeback", action="store_true")
    parser.add_argument("--cosyvoice-dtype", default="bfloat16")
    parser.add_argument("--no-cosyvoice-cuda-graph", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--max-runtime-sec", type=int, default=0)
    parser.add_argument("--batch-metrics-tsv", default="")
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--duration-bucketing", action="store_true")
    parser.add_argument("--duration-bucket-window", type=int, default=0)
    parser.add_argument("--async-write-workers", type=int, default=0)
    parser.add_argument("--async-vocoder-workers", type=int, default=0)
    parser.add_argument("--prepare-breakdown", action="store_true")
    parser.add_argument("--next-batch-prefetch", action="store_true")
    parser.add_argument("--prefetch-depth", type=int, default=1)
    parser.add_argument("--use-async-engine", action="store_true")
    parser.add_argument("--stream-vocode", action="store_true")
    return parser.parse_args()


def batched(items: list[dict], batch_size: int) -> Iterator[list[dict]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def patch_torchaudio_load() -> None:
    original_load = torchaudio.load

    def safe_load(src, *args, **kwargs):
        try:
            return original_load(src, *args, **kwargs)
        except Exception:
            audio_np, sample_rate = sf.read(src, always_2d=True)
            audio = torch.from_numpy(audio_np.T).float()
            return audio, sample_rate

    torchaudio.load = safe_load


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_report_row(job: dict, out_path: Path, output_sr: int) -> dict:
    edit_type = job["metadata"]["edit_type"]
    edit_info = job["metadata"].get("edit_info", "")
    generated_text = job.get("generated_text", "")

    prompt = edit_type
    if edit_info:
        prompt = f"{edit_type}:{edit_info}"
    elif generated_text:
        prompt = f"{edit_type}:{generated_text}"

    return {
        "job_id": job["job_id"],
        "instruction": prompt,
        "text1": job["text1"],
        "audio1": job["audio1"],
        "text2": generated_text or job["text1"],
        "metadata": {
            **job["metadata"],
            "delta_tag": job["metadata"].get("edit_tag", "unknown"),
            "model": "step_audio_editx",
        },
        "one_stage": {
            "backend": "step_audio_editx",
            "prompt": prompt,
            "audio2": str(out_path),
            "result": {
                "sample_rate": output_sr,
            },
        },
    }


def load_completed_job_ids(report_path: Path) -> set[str]:
    completed_job_ids: set[str] = set()
    if not report_path.exists():
        return completed_job_ids

    with report_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            job_id = row.get("job_id")
            if job_id:
                completed_job_ids.add(job_id)
    return completed_job_ids


def load_audio_duration_seconds(audio_path: str) -> float:
    try:
        return float(sf.info(audio_path).duration)
    except Exception:
        return 0.0


def apply_duration_bucketing(
    jobs: list[dict],
    *,
    batch_size: int,
    bucket_window: int,
) -> list[dict]:
    annotated = [
        (load_audio_duration_seconds(job["audio1"]), idx, job)
        for idx, job in enumerate(jobs)
    ]
    if not annotated:
        return []

    window_size = bucket_window if bucket_window > 0 else len(annotated)
    window_size = max(window_size, batch_size)

    ordered: list[dict] = []
    for start in range(0, len(annotated), window_size):
        window = annotated[start : start + window_size]
        window.sort(key=lambda item: (item[0], item[1]))
        ordered.extend(job for _, _, job in window)
    return ordered


def ensure_tsv_header(path: Path, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    path.write_text(header + "\n", encoding="utf-8")


def append_tsv_row(path: Path, values: list[object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\t".join(str(value) for value in values) + "\n")


def write_audio_file(out_path: Path, audio_np, output_sr: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(
        f".{out_path.stem}.tmp.{os.getpid()}.{time.time_ns()}{out_path.suffix}"
    )
    last_exc: Exception | None = None
    for attempt in range(1, 6):
        try:
            sf.write(str(tmp_path), audio_np, output_sr, format="WAV")
            tmp_path.replace(out_path)
            return
        except Exception as exc:
            last_exc = exc
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            if attempt < 5:
                time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"failed to write wav after 5 attempts: {out_path}") from last_exc


def build_batch_execution_plan(
    *,
    job_batch: list[dict],
    output_dir: Path,
    force_rerun: bool,
) -> dict[str, object]:
    batch_rows: list[dict | None] = []
    batch_requests: list[dict] = []
    batch_targets: list[tuple[int, dict, Path]] = []
    reused_existing = 0

    for job in job_batch:
        out_path = output_dir / job["output_relpath"] / "one_stage_audio2.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not force_rerun:
            output_sr = int(sf.info(str(out_path)).samplerate)
            batch_rows.append(build_report_row(job, out_path, output_sr))
            reused_existing += 1
            continue

        batch_rows.append(None)
        batch_requests.append(
            {
                "prompt_wav_path": job["audio1"],
                "prompt_text": job["text1"],
                "edit_type": job["metadata"]["edit_type"],
                "edit_info": job["metadata"].get("edit_info") or None,
                "target_text": job.get("generated_text") or None,
            }
        )
        batch_targets.append((len(batch_rows) - 1, job, out_path))

    return {
        "job_batch": job_batch,
        "batch_rows": batch_rows,
        "batch_requests": batch_requests,
        "batch_targets": batch_targets,
        "reused_existing": reused_existing,
    }


def prepare_batch_requests_with_metrics(model, batch_requests: list[dict]) -> tuple[list[dict], float, dict[str, object]]:
    prepare_start = time.perf_counter()
    prepared_items = model.prepare_batch_requests(batch_requests)
    prepare_sec = time.perf_counter() - prepare_start
    get_prepare_breakdown = getattr(model, "get_last_prepare_breakdown", None)
    prepare_breakdown = get_prepare_breakdown() if callable(get_prepare_breakdown) else {}
    return prepared_items, prepare_sec, (prepare_breakdown or {})


def finalize_nonstream_batch(
    *,
    model,
    prepared_items: list[dict],
    output_ids_batch: list[torch.Tensor],
    batch_targets: list[tuple[int, dict, Path]],
    batch_rows: list[dict | None],
    writer_pool: ThreadPoolExecutor | None,
) -> dict[str, object]:
    vocoder_start = time.perf_counter()
    batch_outputs = [
        model.synthesize_prepared(prepared, output_ids)
        for prepared, output_ids in zip(prepared_items, output_ids_batch)
    ]
    vocoder_sec = time.perf_counter() - vocoder_start

    pending_write_futures: list[tuple[Future | None, str]] = []
    finalized_rows: list[dict] = []
    for (row_index, job, out_path), (output_audio, output_sr) in zip(batch_targets, batch_outputs):
        audio_np = output_audio.detach().cpu().squeeze(0).numpy()
        row = build_report_row(job, out_path, output_sr)
        batch_rows[row_index] = row
        finalized_rows.append(row)
        if writer_pool is None:
            write_audio_file(out_path, audio_np, output_sr)
        else:
            future = writer_pool.submit(write_audio_file, out_path, audio_np, output_sr)
            pending_write_futures.append((future, str(out_path)))

    write_start = time.perf_counter()
    for future, out_path_str in pending_write_futures:
        try:
            future.result()
        except Exception as exc:
            raise RuntimeError(f"failed to write wav: {out_path_str}") from exc
    write_sec = time.perf_counter() - write_start

    finalized_rows = [row for row in batch_rows if row is not None]
    return {
        "batch_rows": batch_rows,
        "finalized_rows": finalized_rows,
        "vocoder_sec": vocoder_sec,
        "write_sec": write_sec,
        "generated": len(batch_targets),
    }


def main() -> None:
    args = parse_args()
    patch_torchaudio_load()
    jobs = load_jsonl(args.jobs_jsonl)
    if args.limit > 0:
        jobs = jobs[: args.limit]

    output_dir = Path(args.output_dir)
    report_path = Path(args.report_jsonl or str(output_dir / "paired_report.jsonl"))
    completed_job_ids = load_completed_job_ids(report_path)
    pending_jobs = [job for job in jobs if job["job_id"] not in completed_job_ids]
    batch_size = max(args.batch_size, 1)
    if args.duration_bucketing:
        pending_jobs = apply_duration_bucketing(
            pending_jobs,
            batch_size=batch_size,
            bucket_window=args.duration_bucket_window,
        )

    batch_metrics_path = Path(args.batch_metrics_tsv or str(output_dir / "batch_metrics.tsv"))
    summary_path = Path(args.summary_json or str(output_dir / "run_summary.json"))
    startup_error_path = output_dir / "startup_error.txt"
    ensure_tsv_header(
        batch_metrics_path,
        "batch_id\tbatch_jobs\tgenerated\treused\tprepare_sec\tgenerate_sec\tvocoder_sec\twrite_sec\tbatch_wall_sec\tprocessed_cumulative\tprepare_request_cache_hits\tprepare_audio_condition_mem_hits\tprepare_audio_condition_disk_hits\tprepare_audio_condition_misses\tprepare_audio_condition_cache_lookup_sec\tprepare_audio_condition_load_sec\tprepare_audio_condition_speech_feat_sec\tprepare_audio_condition_build_sec\tprepare_audio_condition_spk_embedding_sec\tprepare_audio_condition_wav2token_sec\tprepare_audio_condition_pack_sec\tprepare_audio_condition_cache_writeback_sec\tprepare_prompt_encode_sec",
    )

    print(
        f"loaded {len(jobs)} jobs, completed_in_report={len(completed_job_ids)}, pending={len(pending_jobs)}"
    )
    if args.inspect_only:
        print(
            f"inspect_only jobs={len(jobs)} completed={len(completed_job_ids)} pending={len(pending_jobs)} report={report_path.resolve()}"
        )
        return
    if not pending_jobs:
        print(f"nothing to do; report already complete -> {report_path.resolve()}")
        return

    processed = 0
    reused = 0
    generated = 0
    last_logged = 0
    wall_start = time.perf_counter()
    started_at = utc_now_iso()
    stop_reason = "completed"
    failed = False
    error_message = ""
    prepare_total_sec = 0.0
    generate_total_sec = 0.0
    vocoder_total_sec = 0.0
    write_total_sec = 0.0
    prepare_request_cache_hits_total = 0
    prepare_audio_condition_mem_hits_total = 0
    prepare_audio_condition_disk_hits_total = 0
    prepare_audio_condition_misses_total = 0
    prepare_audio_condition_cache_lookup_total_sec = 0.0
    prepare_audio_condition_load_total_sec = 0.0
    prepare_audio_condition_speech_feat_total_sec = 0.0
    prepare_audio_condition_build_total_sec = 0.0
    prepare_audio_condition_spk_embedding_total_sec = 0.0
    prepare_audio_condition_wav2token_total_sec = 0.0
    prepare_audio_condition_pack_total_sec = 0.0
    prepare_audio_condition_cache_writeback_total_sec = 0.0
    prepare_prompt_encode_total_sec = 0.0
    async_write_workers = max(args.async_write_workers, 0)
    async_vocoder_workers = max(args.async_vocoder_workers, 0)
    prefetch_depth = max(args.prefetch_depth, 0)
    next_batch_prefetch_enabled = bool(args.next_batch_prefetch) and prefetch_depth > 0
    prefetch_wait_total_sec = 0.0
    prefetched_batches = 0
    prefetch_hit_batches = 0
    writer_pool = None
    prefetch_executor = None
    vocoder_executor = None
    job_batches = list(batched(pending_jobs, batch_size))
    prefetched_plans: dict[int, dict[str, object]] = {}
    prefetched_futures: dict[int, Future] = {}
    pending_vocoder_batches: dict[int, dict[str, object]] = {}
    next_vocoder_flush_batch_id = 1
    model = None

    try:
        repo_dir = Path(args.repo_dir).resolve()
        sys.path.insert(0, str(repo_dir))

        from tokenizer import StepAudioTokenizer
        from tts import StepAudioTTS

        tokenizer = StepAudioTokenizer(args.tokenizer_dir, model_source="local")
        model = StepAudioTTS(
            args.model_dir,
            tokenizer,
            model_source="local",
            tensor_parallel_size=max(1, args.tensor_parallel_size),
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            enforce_eager=args.enforce_eager,
            dtype=args.dtype,
            kv_cache_dtype=args.kv_cache_dtype,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens or None,
            cosyvoice_dtype=args.cosyvoice_dtype,
            cosyvoice_cuda_graph=not args.no_cosyvoice_cuda_graph,
            use_async_engine=args.use_async_engine,
            stream_vocode=args.stream_vocode,
            preprocess_cache_size=max(args.preprocess_cache_size, 0),
            prepare_workers=max(args.prepare_workers, 1),
            audio_condition_build_workers=(
                max(args.audio_condition_build_workers, 1)
                if args.audio_condition_build_workers > 0
                else max(args.prepare_workers, 1)
            ),
            audio_condition_item_parallel=not args.disable_audio_condition_item_parallel,
            audio_condition_cache_dir=args.audio_condition_cache_dir or None,
            audio_condition_cache_writeback=not args.disable_audio_condition_cache_writeback,
            enable_prepare_breakdown=args.prepare_breakdown,
        )
        writer_pool = (
            ThreadPoolExecutor(max_workers=async_write_workers, thread_name_prefix="editx-write")
            if async_write_workers > 0
            else None
        )
        prefetch_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="editx-prefetch")
            if next_batch_prefetch_enabled
            else None
        )
        vocoder_executor = (
            ThreadPoolExecutor(max_workers=async_vocoder_workers, thread_name_prefix="editx-vocoder")
            if async_vocoder_workers > 0 and not bool(args.use_async_engine and args.stream_vocode)
            else None
        )

        def flush_vocoder_batches(*, block_for_next: bool) -> None:
            nonlocal next_vocoder_flush_batch_id
            nonlocal processed, generated, last_logged
            nonlocal vocoder_total_sec, write_total_sec
            nonlocal prepare_total_sec, generate_total_sec
            nonlocal prepare_request_cache_hits_total
            nonlocal prepare_audio_condition_mem_hits_total
            nonlocal prepare_audio_condition_disk_hits_total
            nonlocal prepare_audio_condition_misses_total
            nonlocal prepare_audio_condition_cache_lookup_total_sec
            nonlocal prepare_audio_condition_load_total_sec
            nonlocal prepare_audio_condition_speech_feat_total_sec
            nonlocal prepare_audio_condition_build_total_sec
            nonlocal prepare_audio_condition_spk_embedding_total_sec
            nonlocal prepare_audio_condition_wav2token_total_sec
            nonlocal prepare_audio_condition_pack_total_sec
            nonlocal prepare_audio_condition_cache_writeback_total_sec
            nonlocal prepare_prompt_encode_total_sec

            while next_vocoder_flush_batch_id in pending_vocoder_batches:
                payload = pending_vocoder_batches[next_vocoder_flush_batch_id]
                future = payload["future"]
                if not block_for_next and not future.done():
                    break
                result = future.result()
                finalized_rows = result["finalized_rows"]
                expected_batch_jobs = payload["batch_jobs"]
                if len(finalized_rows) != expected_batch_jobs:
                    raise RuntimeError(
                        f"incomplete batch rows: expected={expected_batch_jobs} finalized={len(finalized_rows)} batch_id={next_vocoder_flush_batch_id}"
                    )
                append_jsonl(report_path, finalized_rows)
                vocoder_sec = float(result["vocoder_sec"])
                write_sec = float(result["write_sec"])
                prepare_sec = float(payload["prepare_sec"])
                generate_sec = float(payload["generate_sec"])
                prepare_breakdown = payload["prepare_breakdown"]
                processed += int(result["generated"])
                generated += int(result["generated"])
                prepare_total_sec += prepare_sec
                generate_total_sec += generate_sec
                vocoder_total_sec += vocoder_sec
                write_total_sec += write_sec
                prepare_request_cache_hits = int(prepare_breakdown.get("prepare_request_cache_hits", 0))
                prepare_audio_condition_mem_hits = int(prepare_breakdown.get("prepare_audio_condition_mem_hits", 0))
                prepare_audio_condition_disk_hits = int(prepare_breakdown.get("prepare_audio_condition_disk_hits", 0))
                prepare_audio_condition_misses = int(prepare_breakdown.get("prepare_audio_condition_misses", 0))
                prepare_audio_condition_cache_lookup_sec = float(prepare_breakdown.get("prepare_audio_condition_cache_lookup_sec", 0.0))
                prepare_audio_condition_load_sec = float(prepare_breakdown.get("prepare_audio_condition_load_sec", 0.0))
                prepare_audio_condition_speech_feat_sec = float(prepare_breakdown.get("prepare_audio_condition_speech_feat_sec", 0.0))
                prepare_audio_condition_build_sec = float(prepare_breakdown.get("prepare_audio_condition_build_sec", 0.0))
                prepare_audio_condition_spk_embedding_sec = float(prepare_breakdown.get("prepare_audio_condition_spk_embedding_sec", 0.0))
                prepare_audio_condition_wav2token_sec = float(prepare_breakdown.get("prepare_audio_condition_wav2token_sec", 0.0))
                prepare_audio_condition_pack_sec = float(prepare_breakdown.get("prepare_audio_condition_pack_sec", 0.0))
                prepare_audio_condition_cache_writeback_sec = float(prepare_breakdown.get("prepare_audio_condition_cache_writeback_sec", 0.0))
                prepare_prompt_encode_sec = float(prepare_breakdown.get("prepare_prompt_encode_sec", 0.0))
                prepare_request_cache_hits_total += prepare_request_cache_hits
                prepare_audio_condition_mem_hits_total += prepare_audio_condition_mem_hits
                prepare_audio_condition_disk_hits_total += prepare_audio_condition_disk_hits
                prepare_audio_condition_misses_total += prepare_audio_condition_misses
                prepare_audio_condition_cache_lookup_total_sec += prepare_audio_condition_cache_lookup_sec
                prepare_audio_condition_load_total_sec += prepare_audio_condition_load_sec
                prepare_audio_condition_speech_feat_total_sec += prepare_audio_condition_speech_feat_sec
                prepare_audio_condition_build_total_sec += prepare_audio_condition_build_sec
                prepare_audio_condition_spk_embedding_total_sec += prepare_audio_condition_spk_embedding_sec
                prepare_audio_condition_wav2token_total_sec += prepare_audio_condition_wav2token_sec
                prepare_audio_condition_pack_total_sec += prepare_audio_condition_pack_sec
                prepare_audio_condition_cache_writeback_total_sec += prepare_audio_condition_cache_writeback_sec
                prepare_prompt_encode_total_sec += prepare_prompt_encode_sec
                batch_wall_sec = time.perf_counter() - float(payload["batch_wall_start"])
                append_tsv_row(
                    batch_metrics_path,
                    [
                        next_vocoder_flush_batch_id,
                        expected_batch_jobs,
                        int(result["generated"]),
                        int(payload["reused_count"]),
                        f"{prepare_sec:.6f}",
                        f"{generate_sec:.6f}",
                        f"{vocoder_sec:.6f}",
                        f"{write_sec:.6f}",
                        f"{batch_wall_sec:.6f}",
                        processed,
                        prepare_request_cache_hits,
                        prepare_audio_condition_mem_hits,
                        prepare_audio_condition_disk_hits,
                        prepare_audio_condition_misses,
                        f"{prepare_audio_condition_cache_lookup_sec:.6f}",
                        f"{prepare_audio_condition_load_sec:.6f}",
                        f"{prepare_audio_condition_speech_feat_sec:.6f}",
                        f"{prepare_audio_condition_build_sec:.6f}",
                        f"{prepare_audio_condition_spk_embedding_sec:.6f}",
                        f"{prepare_audio_condition_wav2token_sec:.6f}",
                        f"{prepare_audio_condition_pack_sec:.6f}",
                        f"{prepare_audio_condition_cache_writeback_sec:.6f}",
                        f"{prepare_prompt_encode_sec:.6f}",
                    ],
                )
                if (
                    next_vocoder_flush_batch_id == 1
                    or processed == len(pending_jobs)
                    or processed - last_logged >= max(50, batch_size * 4)
                ):
                    print(
                        f"progress batch={next_vocoder_flush_batch_id} processed={processed}/{len(pending_jobs)} generated={generated} reused={reused}"
                    )
                    last_logged = processed
                pending_vocoder_batches.pop(next_vocoder_flush_batch_id, None)
                next_vocoder_flush_batch_id += 1
                block_for_next = False

        for batch_id, job_batch in enumerate(job_batches, start=1):
            flush_vocoder_batches(block_for_next=False)
            if vocoder_executor is not None and len(pending_vocoder_batches) >= async_vocoder_workers:
                flush_vocoder_batches(block_for_next=True)

            if args.max_runtime_sec > 0 and processed > 0:
                elapsed_before_batch = time.perf_counter() - wall_start
                if elapsed_before_batch >= args.max_runtime_sec:
                    stop_reason = "max_runtime_reached"
                    break

            batch_idx0 = batch_id - 1
            if batch_idx0 in prefetched_plans:
                batch_plan = prefetched_plans[batch_idx0]
            else:
                batch_plan = build_batch_execution_plan(
                    job_batch=job_batch,
                    output_dir=output_dir,
                    force_rerun=args.force_rerun,
                )

            batch_rows = batch_plan["batch_rows"]
            batch_requests = batch_plan["batch_requests"]
            batch_targets = batch_plan["batch_targets"]
            prepare_sec = 0.0
            generate_sec = 0.0
            vocoder_sec = 0.0
            write_sec = 0.0
            prepare_breakdown: dict[str, object] = {}
            batch_wall_start = time.perf_counter()
            prefetch_wait_sec = 0.0
            reused += int(batch_plan["reused_existing"])
            processed += int(batch_plan["reused_existing"])

            if batch_requests:
                if batch_idx0 in prefetched_futures:
                    wait_start = time.perf_counter()
                    prepared_items, prepare_sec, prepare_breakdown = prefetched_futures.pop(batch_idx0).result()
                    prefetch_wait_sec = time.perf_counter() - wait_start
                    prefetch_wait_total_sec += prefetch_wait_sec
                    prefetch_hit_batches += 1
                else:
                    prepared_items, prepare_sec, prepare_breakdown = prepare_batch_requests_with_metrics(
                        model, batch_requests
                    )

                prefetched_plans.pop(batch_idx0, None)

                if next_batch_prefetch_enabled and prefetch_executor is not None:
                    upper = min(len(job_batches), batch_idx0 + 1 + prefetch_depth)
                    for next_idx0 in range(batch_idx0 + 1, upper):
                        if next_idx0 in prefetched_plans:
                            continue
                        next_plan = build_batch_execution_plan(
                            job_batch=job_batches[next_idx0],
                            output_dir=output_dir,
                            force_rerun=args.force_rerun,
                        )
                        prefetched_plans[next_idx0] = next_plan
                        next_batch_requests = next_plan["batch_requests"]
                        if next_batch_requests:
                            prefetched_futures[next_idx0] = prefetch_executor.submit(
                                prepare_batch_requests_with_metrics,
                                model,
                                next_batch_requests,
                            )
                            prefetched_batches += 1

                use_stream_vocode = bool(args.use_async_engine and args.stream_vocode)
                if use_stream_vocode:
                    finalized_rows = []
                    generate_start = time.perf_counter()
                    batch_outputs = model.generate_prepared_batch(prepared_items)
                    generate_and_vocode_sec = time.perf_counter() - generate_start
                    generate_sec = generate_and_vocode_sec
                    vocoder_sec = 0.0
                    if len(batch_outputs) != len(batch_targets):
                        raise RuntimeError(
                            f"batch result size mismatch: requests={len(batch_targets)} outputs={len(batch_outputs)}"
                        )
                    output_iter = (
                        (target, output)
                        for target, output in zip(batch_targets, batch_outputs)
                    )
                elif vocoder_executor is None:
                    finalized_rows = []
                    generate_start = time.perf_counter()
                    output_ids_batch = model.generate_prepared_output_ids_batch(prepared_items)
                    generate_sec = time.perf_counter() - generate_start
                    if len(output_ids_batch) != len(batch_targets):
                        raise RuntimeError(
                            f"batch result size mismatch: requests={len(batch_targets)} outputs={len(output_ids_batch)}"
                        )
                    finalize_result = finalize_nonstream_batch(
                        model=model,
                        prepared_items=prepared_items,
                        output_ids_batch=output_ids_batch,
                        batch_targets=batch_targets,
                        batch_rows=batch_rows,
                        writer_pool=writer_pool,
                    )
                    vocoder_sec = float(finalize_result["vocoder_sec"])
                    write_sec = float(finalize_result["write_sec"])
                    finalized_rows = finalize_result["finalized_rows"]
                    generated += int(finalize_result["generated"])
                    processed += int(finalize_result["generated"])
                    output_iter = ()
                else:
                    generate_start = time.perf_counter()
                    output_ids_batch = model.generate_prepared_output_ids_batch(prepared_items)
                    generate_sec = time.perf_counter() - generate_start
                    if len(output_ids_batch) != len(batch_targets):
                        raise RuntimeError(
                            f"batch result size mismatch: requests={len(batch_targets)} outputs={len(output_ids_batch)}"
                        )
                    pending_vocoder_batches[batch_id] = {
                        "future": vocoder_executor.submit(
                            finalize_nonstream_batch,
                            model=model,
                            prepared_items=prepared_items,
                            output_ids_batch=output_ids_batch,
                            batch_targets=batch_targets,
                            batch_rows=batch_rows,
                            writer_pool=writer_pool,
                        ),
                        "batch_jobs": len(job_batch),
                        "reused_count": len(job_batch) - len(batch_targets),
                        "prepare_sec": prepare_sec,
                        "generate_sec": generate_sec,
                        "prepare_breakdown": prepare_breakdown,
                        "batch_wall_start": batch_wall_start,
                    }
                    output_iter = (
                        ()
                    )

                for (row_index, job, out_path), (output_audio, output_sr) in output_iter:
                    audio_np = output_audio.detach().cpu().squeeze(0).numpy()
                    row = build_report_row(job, out_path, output_sr)
                    batch_rows[row_index] = row
                    finalized_rows.append(row)
                    if writer_pool is None:
                        write_audio_file(out_path, audio_np, output_sr)
                    else:
                        future = writer_pool.submit(write_audio_file, out_path, audio_np, output_sr)
                        future.result()
                    generated += 1
                    processed += 1

                if use_stream_vocode or vocoder_executor is None:
                    finalized_rows = [row for row in batch_rows if row is not None]
                    if len(finalized_rows) != len(job_batch):
                        raise RuntimeError(
                            f"incomplete batch rows: expected={len(job_batch)} finalized={len(finalized_rows)} batch_id={batch_id}"
                        )
                    write_start = time.perf_counter()
                    append_jsonl(report_path, finalized_rows)
                    write_sec = time.perf_counter() - write_start
            else:
                write_start = time.perf_counter()
                finalized_rows = [row for row in batch_rows if row is not None]
                if len(finalized_rows) != len(job_batch):
                    raise RuntimeError(
                        f"incomplete batch rows: expected={len(job_batch)} finalized={len(finalized_rows)} batch_id={batch_id}"
                    )
                append_jsonl(report_path, finalized_rows)
                write_sec = time.perf_counter() - write_start

            if vocoder_executor is None or not batch_requests:
                prepare_total_sec += prepare_sec
                generate_total_sec += generate_sec
                vocoder_total_sec += vocoder_sec
                write_total_sec += write_sec
                prepare_request_cache_hits = int(prepare_breakdown.get("prepare_request_cache_hits", 0))
                prepare_audio_condition_mem_hits = int(prepare_breakdown.get("prepare_audio_condition_mem_hits", 0))
                prepare_audio_condition_disk_hits = int(prepare_breakdown.get("prepare_audio_condition_disk_hits", 0))
                prepare_audio_condition_misses = int(prepare_breakdown.get("prepare_audio_condition_misses", 0))
                prepare_audio_condition_cache_lookup_sec = float(prepare_breakdown.get("prepare_audio_condition_cache_lookup_sec", 0.0))
                prepare_audio_condition_load_sec = float(prepare_breakdown.get("prepare_audio_condition_load_sec", 0.0))
                prepare_audio_condition_speech_feat_sec = float(prepare_breakdown.get("prepare_audio_condition_speech_feat_sec", 0.0))
                prepare_audio_condition_build_sec = float(prepare_breakdown.get("prepare_audio_condition_build_sec", 0.0))
                prepare_audio_condition_spk_embedding_sec = float(prepare_breakdown.get("prepare_audio_condition_spk_embedding_sec", 0.0))
                prepare_audio_condition_wav2token_sec = float(prepare_breakdown.get("prepare_audio_condition_wav2token_sec", 0.0))
                prepare_audio_condition_pack_sec = float(prepare_breakdown.get("prepare_audio_condition_pack_sec", 0.0))
                prepare_audio_condition_cache_writeback_sec = float(prepare_breakdown.get("prepare_audio_condition_cache_writeback_sec", 0.0))
                prepare_prompt_encode_sec = float(prepare_breakdown.get("prepare_prompt_encode_sec", 0.0))
                prepare_request_cache_hits_total += prepare_request_cache_hits
                prepare_audio_condition_mem_hits_total += prepare_audio_condition_mem_hits
                prepare_audio_condition_disk_hits_total += prepare_audio_condition_disk_hits
                prepare_audio_condition_misses_total += prepare_audio_condition_misses
                prepare_audio_condition_cache_lookup_total_sec += prepare_audio_condition_cache_lookup_sec
                prepare_audio_condition_load_total_sec += prepare_audio_condition_load_sec
                prepare_audio_condition_speech_feat_total_sec += prepare_audio_condition_speech_feat_sec
                prepare_audio_condition_build_total_sec += prepare_audio_condition_build_sec
                prepare_audio_condition_spk_embedding_total_sec += prepare_audio_condition_spk_embedding_sec
                prepare_audio_condition_wav2token_total_sec += prepare_audio_condition_wav2token_sec
                prepare_audio_condition_pack_total_sec += prepare_audio_condition_pack_sec
                prepare_audio_condition_cache_writeback_total_sec += prepare_audio_condition_cache_writeback_sec
                prepare_prompt_encode_total_sec += prepare_prompt_encode_sec
                batch_wall_sec = time.perf_counter() - batch_wall_start
                append_tsv_row(
                    batch_metrics_path,
                    [
                        batch_id,
                        len(job_batch),
                        len(batch_targets),
                        len(job_batch) - len(batch_targets),
                        f"{prepare_sec:.6f}",
                        f"{generate_sec:.6f}",
                        f"{vocoder_sec:.6f}",
                        f"{write_sec:.6f}",
                        f"{batch_wall_sec:.6f}",
                        processed,
                        prepare_request_cache_hits,
                        prepare_audio_condition_mem_hits,
                        prepare_audio_condition_disk_hits,
                        prepare_audio_condition_misses,
                        f"{prepare_audio_condition_cache_lookup_sec:.6f}",
                        f"{prepare_audio_condition_load_sec:.6f}",
                        f"{prepare_audio_condition_speech_feat_sec:.6f}",
                        f"{prepare_audio_condition_build_sec:.6f}",
                        f"{prepare_audio_condition_spk_embedding_sec:.6f}",
                        f"{prepare_audio_condition_wav2token_sec:.6f}",
                        f"{prepare_audio_condition_pack_sec:.6f}",
                        f"{prepare_audio_condition_cache_writeback_sec:.6f}",
                        f"{prepare_prompt_encode_sec:.6f}",
                    ],
                )

                if (
                    batch_id == 1
                    or processed == len(pending_jobs)
                    or processed - last_logged >= max(50, batch_size * 4)
                ):
                    print(
                        f"progress batch={batch_id} processed={processed}/{len(pending_jobs)} generated={generated} reused={reused}"
                    )
                    last_logged = processed
        flush_vocoder_batches(block_for_next=True)
    except Exception as exc:
        failed = True
        stop_reason = "failed"
        error_message = str(exc)
        startup_error_path.parent.mkdir(parents=True, exist_ok=True)
        startup_error_path.write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        if prefetch_executor is not None:
            prefetch_executor.shutdown(wait=True)
        if vocoder_executor is not None:
            vocoder_executor.shutdown(wait=True)
        if writer_pool is not None:
            writer_pool.shutdown(wait=True)
        close = getattr(model, "close", None) if model is not None else None
        if callable(close):
            close()

        runtime_sec = time.perf_counter() - wall_start
        summary = {
            "started_at": started_at,
            "ended_at": utc_now_iso(),
            "status": "failed" if failed else stop_reason,
            "stop_reason": stop_reason,
            "error_message": error_message,
            "jobs_total": len(jobs),
            "completed_in_report": len(completed_job_ids),
            "pending_at_start": len(pending_jobs),
            "processed": processed,
            "generated": generated,
            "reused": reused,
            "batch_size": batch_size,
            "prepare_workers": max(args.prepare_workers, 1),
            "audio_condition_build_workers": (
                max(args.audio_condition_build_workers, 1)
                if args.audio_condition_build_workers > 0
                else max(args.prepare_workers, 1)
            ),
            "audio_condition_item_parallel": not args.disable_audio_condition_item_parallel,
            "duration_bucketing": bool(args.duration_bucketing),
            "duration_bucket_window": int(args.duration_bucket_window),
            "async_write_workers": async_write_workers,
            "async_vocoder_workers": async_vocoder_workers,
            "next_batch_prefetch": next_batch_prefetch_enabled,
            "prefetch_depth": prefetch_depth,
            "use_async_engine": bool(args.use_async_engine),
            "stream_vocode": bool(args.stream_vocode),
            "prefetched_batches": prefetched_batches,
            "prefetch_hit_batches": prefetch_hit_batches,
            "prefetch_wait_total_sec": prefetch_wait_total_sec,
            "max_runtime_sec": int(args.max_runtime_sec),
            "runtime_sec": runtime_sec,
            "prepare_total_sec": prepare_total_sec,
            "generate_total_sec": generate_total_sec,
            "vocoder_total_sec": vocoder_total_sec,
            "write_total_sec": write_total_sec,
            "prepare_request_cache_hits_total": prepare_request_cache_hits_total,
            "prepare_audio_condition_mem_hits_total": prepare_audio_condition_mem_hits_total,
            "prepare_audio_condition_disk_hits_total": prepare_audio_condition_disk_hits_total,
            "prepare_audio_condition_misses_total": prepare_audio_condition_misses_total,
            "prepare_audio_condition_cache_lookup_total_sec": prepare_audio_condition_cache_lookup_total_sec,
            "prepare_audio_condition_load_total_sec": prepare_audio_condition_load_total_sec,
            "prepare_audio_condition_speech_feat_total_sec": prepare_audio_condition_speech_feat_total_sec,
            "prepare_audio_condition_build_total_sec": prepare_audio_condition_build_total_sec,
            "prepare_audio_condition_spk_embedding_total_sec": prepare_audio_condition_spk_embedding_total_sec,
            "prepare_audio_condition_wav2token_total_sec": prepare_audio_condition_wav2token_total_sec,
            "prepare_audio_condition_pack_total_sec": prepare_audio_condition_pack_total_sec,
            "prepare_audio_condition_cache_writeback_total_sec": prepare_audio_condition_cache_writeback_total_sec,
            "prepare_prompt_encode_total_sec": prepare_prompt_encode_total_sec,
            "throughput_processed_per_sec": (processed / runtime_sec) if runtime_sec > 0 else 0.0,
            "throughput_processed_per_hour": (processed * 3600.0 / runtime_sec) if runtime_sec > 0 else 0.0,
            "throughput_generated_per_sec": (generated / runtime_sec) if runtime_sec > 0 else 0.0,
            "throughput_generated_per_hour": (generated * 3600.0 / runtime_sec) if runtime_sec > 0 else 0.0,
            "report_jsonl": str(report_path.resolve()),
            "batch_metrics_tsv": str(batch_metrics_path.resolve()),
            "startup_error_txt": str(startup_error_path.resolve()),
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"finished processed={processed} generated={generated} reused={reused} -> {report_path.resolve()}")


if __name__ == "__main__":
    main()
