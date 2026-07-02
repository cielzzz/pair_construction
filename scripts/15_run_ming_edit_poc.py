#!/usr/bin/env python3
"""Run one Ming-UniAudio-Edit PoC without modifying the upstream Ming repo."""

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


def _load_audio_16k(path: Path) -> tuple[torch.Tensor, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000
    return wav.contiguous(), sr


def _patch_torchaudio_save() -> None:
    def save_with_soundfile(path, src, sample_rate, *args, **kwargs):  # noqa: ANN001
        wav = src.detach().cpu().float()
        if wav.ndim == 2:
            wav = wav.T
        sf.write(str(path), wav.numpy(), int(sample_rate))

    torchaudio.save = save_with_soundfile  # type: ignore[assignment]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ming-repo", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output-wav", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1895)
    parser.add_argument("--no-cot", action="store_true")
    parser.add_argument("--audio-input-mode", choices=["tensor", "path"], default="tensor")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    ming_repo = Path(args.ming_repo).resolve()
    model_path = Path(args.model_path).resolve()
    audio_path = Path(args.audio).resolve()
    output_wav = Path(args.output_wav).resolve()
    metrics_json = Path(args.metrics_json).resolve()
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    metrics_json.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(ming_repo))
    os.chdir(ming_repo)
    _patch_torchaudio_save()

    from cookbooks.test import MingAudio, seed_everything  # noqa: PLC0415

    seed_everything(args.seed)
    audio, sr = _load_audio_16k(audio_path)
    load_start = time.time()
    model = MingAudio(str(model_path), device=args.device)
    load_elapsed = time.time() - load_start

    if args.audio_input_mode == "path":
        audio_item = {"type": "audio", "audio": str(audio_path), "target_sample_rate": 16000}
    else:
        audio_item = {"type": "audio", "audio": audio, "sample_rate": sr, "target_sample_rate": sr}

    messages = [
        {
            "role": "HUMAN",
            "content": [
                audio_item,
                {"type": "text", "text": args.instruction},
            ],
        },
    ]

    run_start = time.time()
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
    elapsed = time.time() - run_start

    metrics = {
        "ok": True,
        "python": sys.executable,
        "model_path": str(model_path),
        "ming_repo": str(ming_repo),
        "audio": str(audio_path),
        "output_wav": str(output_wav),
        "instruction": args.instruction,
        "device": args.device,
        "seed": args.seed,
        "use_cot": not args.no_cot,
        "audio_input_mode": args.audio_input_mode,
        "runtime_config": "model config attention, soundfile audio input support, fixed attention_mask device, soundfile save patch",
        "audio_shape": list(audio.shape),
        "audio_sr": sr,
        "load_elapsed_s": round(load_elapsed, 3),
        "elapsed_s": round(elapsed, 3),
        "edited_text": edited_text,
        "output_exists": output_wav.exists(),
        "output_size": output_wav.stat().st_size if output_wav.exists() else 0,
        "edited_speech_shape": list(edited_speech.shape),
    }
    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
