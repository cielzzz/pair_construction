#!/usr/bin/env python3
"""Run multiple Ming-UniAudio-Edit cases after loading the model once."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
import torchaudio


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_audio_16k(path: Path) -> tuple[torch.Tensor, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000
    return wav.contiguous(), sr


def patch_torchaudio_save() -> None:
    def save_with_soundfile(path, src, sample_rate, *args, **kwargs):  # noqa: ANN001
        wav = src.detach().cpu().float()
        if wav.ndim == 2:
            wav = wav.T
        sf.write(str(path), wav.numpy(), int(sample_rate))

    torchaudio.save = save_with_soundfile  # type: ignore[assignment]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--summary-jsonl", required=True)
    parser.add_argument("--ming-repo", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1895)
    parser.add_argument("--no-cot", action="store_true")
    parser.add_argument("--audio-input-mode", choices=["tensor", "path"], default="tensor")
    args = parser.parse_args()

    base_dir = Path.cwd().resolve()
    cases_path = Path(args.cases).resolve()
    summary_path = Path(args.summary_jsonl).resolve()
    ming_repo = Path(args.ming_repo).resolve()
    model_path = Path(args.model_path).resolve()
    sys.path.insert(0, str(ming_repo))
    os.chdir(ming_repo)
    patch_torchaudio_save()

    from cookbooks.test import MingAudio, seed_everything  # noqa: PLC0415

    seed_everything(args.seed)
    load_start = time.time()
    model = MingAudio(str(model_path), device=args.device)
    load_elapsed = time.time() - load_start

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, case in enumerate(iter_jsonl(cases_path), 1):
        audio_path = Path(case["audio"])
        if not audio_path.is_absolute():
            audio_path = base_dir / audio_path
        audio_path = audio_path.resolve()
        output_wav = Path(case["output_wav"])
        if not output_wav.is_absolute():
            output_wav = base_dir / output_wav
        output_wav = output_wav.resolve()
        metrics_json = Path(case["metrics_json"])
        if not metrics_json.is_absolute():
            metrics_json = base_dir / metrics_json
        metrics_json = metrics_json.resolve()
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        metrics_json.parent.mkdir(parents=True, exist_ok=True)

        audio, sr = load_audio_16k(audio_path)
        if args.audio_input_mode == "path":
            audio_item = {"type": "audio", "audio": str(audio_path), "target_sample_rate": 16000}
        else:
            audio_item = {"type": "audio", "audio": audio, "sample_rate": sr, "target_sample_rate": sr}
        messages = [
            {
                "role": "HUMAN",
                "content": [
                    audio_item,
                    {"type": "text", "text": case["instruction"]},
                ],
            },
        ]

        run_start = time.time()
        ok = True
        error = None
        edited_text = None
        edited_speech_shape = None
        try:
            text = model.processor.apply_chat_template(messages, add_generation_prompt=True)
            image_inputs, video_inputs, audio_inputs = model.processor.process_vision_info(messages)
            inputs = model.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                audios=audio_inputs,
                return_tensors="pt",
            ).to(args.device)
            if not args.no_cot:
                ans = torch.tensor([model.tokenizer.encode("<answer>")], device=inputs["input_ids"].device)
                inputs["input_ids"] = torch.cat([inputs["input_ids"], ans], dim=1)
                inputs["attention_mask"] = torch.ones(
                    inputs["input_ids"].shape,
                    dtype=inputs["attention_mask"].dtype,
                    device=inputs["input_ids"].device,
                )
            for key in inputs.keys():
                if key in {"pixel_values", "pixel_values_videos", "audio_feats"}:
                    inputs[key] = inputs[key].to(dtype=torch.bfloat16)

            edited_speech, edited_text = model.model.generate_edit(
                **inputs,
                tokenizer=model.tokenizer,
                output_wav_path=str(output_wav),
            )
            edited_speech_shape = list(edited_speech.shape)
        except Exception as exc:  # keep batch progress for later cases
            ok = False
            error = f"{type(exc).__name__}: {exc}"

        elapsed = time.time() - run_start
        metrics = {
            "ok": ok,
            "case_id": case["case_id"],
            "language": case.get("language"),
            "prompt_type": case.get("prompt_type"),
            "python": sys.executable,
            "model_path": str(model_path),
            "audio": str(audio_path),
            "source_text": case.get("source_text"),
            "target_text": case.get("target_text"),
            "output_wav": str(output_wav),
            "instruction": case["instruction"],
            "device": args.device,
            "seed": args.seed,
            "use_cot": not args.no_cot,
            "audio_input_mode": args.audio_input_mode,
            "load_elapsed_s": round(load_elapsed, 3),
            "elapsed_s": round(elapsed, 3),
            "edited_text": edited_text,
            "error": error,
            "output_exists": output_wav.exists(),
            "output_size": output_wav.stat().st_size if output_wav.exists() else 0,
            "edited_speech_shape": edited_speech_shape,
            "runtime_config": "model config attention, soundfile audio input support, fixed attention_mask device, soundfile save patch",
        }
        metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        rows.append(metrics)
        print(f"[{idx}] {case['case_id']} ok={ok} elapsed={elapsed:.1f}s output={output_wav}", flush=True)

    with summary_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
