#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torchaudio

from _common import iter_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Seed-VC v1 on prepared I jobs.")
    parser.add_argument("--jobs-jsonl", required=True)
    parser.add_argument("--results-jsonl", required=True)
    parser.add_argument("--seed-vc-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--diffusion-steps", type=int, default=25)
    parser.add_argument("--length-adjust", type=float, default=1.0)
    parser.add_argument("--inference-cfg-rate", type=float, default=0.7)
    parser.add_argument("--fp16", type=str, default="true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--show-model-output", action="store_true", help="Show per-utterance Seed-VC/tqdm output.")
    parser.add_argument("--summary-json", default="")
    return parser.parse_args()


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_seedvc(seed_vc_dir: Path, args: argparse.Namespace) -> tuple[Any, ...]:
    seed_vc_dir = seed_vc_dir.resolve()
    if str(seed_vc_dir) not in sys.path:
        sys.path.insert(0, str(seed_vc_dir))
    os.chdir(seed_vc_dir)
    import inference  # type: ignore

    model_args = SimpleNamespace(
        fp16=str_to_bool(args.fp16),
        f0_condition=False,
        auto_f0_adjust=False,
        semi_tone_shift=0,
        checkpoint=args.checkpoint,
        config=args.config,
    )
    loaded = inference.load_models(model_args)
    return (inference, *loaded)


def same_existing(path: str) -> dict[str, Any] | None:
    out = Path(path)
    if not out.exists():
        return None
    try:
        info = torchaudio.info(str(out))
        return {
            "ok": True,
            "audio": str(out),
            "sample_rate": int(info.sample_rate),
            "num_frames": int(info.num_frames),
            "reused": True,
        }
    except Exception:
        return {"ok": True, "audio": str(out), "reused": True}


@torch.no_grad()
def convert_one(
    inference: Any,
    loaded: tuple[Any, ...],
    source_path: str,
    timbre_path: str,
    output_path: str,
    *,
    diffusion_steps: int,
    length_adjust: float,
    inference_cfg_rate: float,
) -> dict[str, Any]:
    import librosa

    model, semantic_fn, _f0_fn, vocoder_fn, campplus_model, mel_fn, mel_fn_args = loaded
    device = inference.device
    sr = int(mel_fn_args["sampling_rate"])
    hop_length = 256
    max_context_window = sr // hop_length * 30
    overlap_frame_len = 16
    overlap_wave_len = overlap_frame_len * hop_length

    source_audio = librosa.load(source_path, sr=sr)[0]
    ref_audio = librosa.load(timbre_path, sr=sr)[0]
    if source_audio.size == 0:
        raise ValueError(f"empty source audio: {source_path}")
    if ref_audio.size == 0:
        raise ValueError(f"empty timbre reference audio: {timbre_path}")

    source_tensor = torch.tensor(source_audio).unsqueeze(0).float().to(device)
    ref_tensor = torch.tensor(ref_audio[: sr * 25]).unsqueeze(0).float().to(device)

    converted_waves_16k = torchaudio.functional.resample(source_tensor, sr, 16000)
    if converted_waves_16k.size(-1) <= 16000 * 30:
        source_semantic = semantic_fn(converted_waves_16k)
    else:
        overlapping_time = 5
        semantic_chunks = []
        buffer = None
        traversed_time = 0
        while traversed_time < converted_waves_16k.size(-1):
            if buffer is None:
                chunk = converted_waves_16k[:, traversed_time : traversed_time + 16000 * 30]
            else:
                chunk = torch.cat(
                    [buffer, converted_waves_16k[:, traversed_time : traversed_time + 16000 * (30 - overlapping_time)]],
                    dim=-1,
                )
            chunk_semantic = semantic_fn(chunk)
            if traversed_time == 0:
                semantic_chunks.append(chunk_semantic)
            else:
                semantic_chunks.append(chunk_semantic[:, 50 * overlapping_time :])
            buffer = chunk[:, -16000 * overlapping_time :]
            traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
        source_semantic = torch.cat(semantic_chunks, dim=1)

    ref_waves_16k = torchaudio.functional.resample(ref_tensor, sr, 16000)
    ref_semantic = semantic_fn(ref_waves_16k)
    source_mel = mel_fn(source_tensor.to(device).float())
    ref_mel = mel_fn(ref_tensor.to(device).float())

    target_lengths = torch.LongTensor([int(source_mel.size(2) * length_adjust)]).to(source_mel.device)
    ref_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)
    feat = torchaudio.compliance.kaldi.fbank(ref_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    style = campplus_model(feat.unsqueeze(0))

    cond, _, _, _, _ = model.length_regulator(source_semantic, ylens=target_lengths, n_quantizers=3, f0=None)
    prompt_condition, _, _, _, _ = model.length_regulator(ref_semantic, ylens=ref_lengths, n_quantizers=3, f0=None)

    max_source_window = max_context_window - ref_mel.size(2)
    if max_source_window <= overlap_frame_len + 1:
        raise ValueError("timbre reference is too long for Seed-VC context window")

    processed_frames = 0
    generated_wave_chunks = []
    previous_chunk = None
    start = time.time()
    while processed_frames < cond.size(1):
        chunk_cond = cond[:, processed_frames : processed_frames + max_source_window]
        is_last_chunk = processed_frames + max_source_window >= cond.size(1)
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
        with torch.autocast(device_type=device.type, dtype=torch.float16 if inference.fp16 else torch.float32):
            vc_target = model.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(ref_mel.device),
                ref_mel,
                style,
                None,
                diffusion_steps,
                inference_cfg_rate=inference_cfg_rate,
            )
            vc_target = vc_target[:, :, ref_mel.size(-1) :]
        vc_wave = vocoder_fn(vc_target.float()).squeeze()[None, :]
        if processed_frames == 0:
            if is_last_chunk:
                generated_wave_chunks.append(vc_wave[0].cpu().numpy())
                break
            generated_wave_chunks.append(vc_wave[0, :-overlap_wave_len].cpu().numpy())
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
        elif is_last_chunk:
            assert previous_chunk is not None
            generated_wave_chunks.append(inference.crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len))
            processed_frames += vc_target.size(2) - overlap_frame_len
            break
        else:
            assert previous_chunk is not None
            generated_wave_chunks.append(
                inference.crossfade(previous_chunk.cpu().numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(), overlap_wave_len)
            )
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len

    output_wave = np.concatenate(generated_wave_chunks)
    output_tensor = torch.tensor(output_wave)[None, :].float().cpu()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out), output_tensor, sr)
    elapsed = time.time() - start
    return {
        "ok": True,
        "audio": str(out),
        "sample_rate": sr,
        "num_frames": int(output_tensor.size(-1)),
        "duration_sec": float(output_tensor.size(-1) / sr),
        "elapsed_sec": float(elapsed),
        "rtf": float(elapsed / max(1e-6, output_tensor.size(-1) / sr)),
        "reused": False,
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


@contextlib.contextmanager
def silence_output(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def main() -> None:
    args = parse_args()
    seed_vc_dir = Path(args.seed_vc_dir).expanduser().resolve()
    results_path = Path(args.results_jsonl).expanduser().resolve()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if results_path.exists():
        results_path.unlink()

    jobs = list(iter_jsonl(args.jobs_jsonl))
    if args.max_jobs and args.max_jobs > 0:
        jobs = jobs[: args.max_jobs]

    inference, *loaded_values = load_seedvc(seed_vc_dir, args)
    loaded = tuple(loaded_values)
    counts = {"ok": 0, "failed": 0, "reused": 0}
    started = time.time()

    for idx, job in enumerate(jobs, 1):
        output_audio = job["output_audio"]
        row = {
            "job_id": job.get("job_id"),
            "prosody_ref_audio": job.get("prosody_ref_audio"),
            "timbre_ref_audio": job.get("timbre_ref_audio"),
            "output_audio": output_audio,
            "backend": "seed_vc_v1_zero_shot_voice_conversion",
            "diffusion_steps": args.diffusion_steps,
            "length_adjust": args.length_adjust,
            "inference_cfg_rate": args.inference_cfg_rate,
        }
        try:
            reused = same_existing(output_audio) if args.skip_existing else None
            if reused:
                result = reused
            else:
                with silence_output(not args.show_model_output):
                    result = convert_one(
                        inference,
                        loaded,
                        job["prosody_ref_audio"],
                        job["timbre_ref_audio"],
                        output_audio,
                        diffusion_steps=args.diffusion_steps,
                        length_adjust=args.length_adjust,
                        inference_cfg_rate=args.inference_cfg_rate,
                    )
            row.update(result)
            counts["ok"] += 1
            if result.get("reused"):
                counts["reused"] += 1
        except Exception as exc:
            row.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            counts["failed"] += 1
            if args.fail_fast:
                append_jsonl(results_path, row)
                raise
        append_jsonl(results_path, row)
        if idx % 10 == 0 or idx == len(jobs):
            elapsed = time.time() - started
            print(
                f"[seedvc] {idx}/{len(jobs)} ok={counts['ok']} failed={counts['failed']} reused={counts['reused']} elapsed={elapsed:.1f}s",
                flush=True,
            )

    summary = {
        "jobs_jsonl": str(Path(args.jobs_jsonl).resolve()),
        "results_jsonl": str(results_path),
        "seed_vc_dir": str(seed_vc_dir),
        "jobs_requested": len(jobs),
        "counts": counts,
        "diffusion_steps": args.diffusion_steps,
        "length_adjust": args.length_adjust,
        "inference_cfg_rate": args.inference_cfg_rate,
        "fp16": str_to_bool(args.fp16),
        "elapsed_sec": time.time() - started,
    }
    summary_path = Path(args.summary_json).expanduser().resolve() if args.summary_json else results_path.with_suffix(".summary.json")
    write_json(summary_path, summary)
    print(f"[seedvc] results -> {results_path}")
    print(f"[seedvc] summary -> {summary_path}")
    if counts["failed"]:
        raise SystemExit(f"Seed-VC finished with {counts['failed']} failed jobs")


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
