#!/usr/bin/env python
"""qwen-tts voice clone 10 句 demo
读 jobs.jsonl → 对每行调 Qwen3TTSModel.generate_voice_clone → 写 outputs/<lang>_<idx>.wav
"""
import json, os, sys
from pathlib import Path
import torch
import soundfile as sf

HERE = Path(__file__).resolve().parent
JOBS = HERE / "jobs.jsonl"
OUT_DIR = HERE / "outputs"
OUT_DIR.mkdir(exist_ok=True)

print("Loading Qwen3-TTS-12Hz-1.7B-Base ...")
from qwen_tts import Qwen3TTSModel
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)
print("Model ready.")

ok = fail = 0
results = []
for line in JOBS.open():
    j = json.loads(line)
    out_path = OUT_DIR / f"{j['lang']}_{j['idx']:06d}.wav"
    if out_path.exists():
        print(f"[skip] exists: {out_path.name}")
        results.append({**j, "out_wav": str(out_path), "ok": True})
        ok += 1
        continue
    try:
        wavs, sr = model.generate_voice_clone(
            text=j["target_text"],
            language=j["language"],
            ref_audio=j["ref_audio"],
            ref_text=j["ref_text"],
        )
        sf.write(str(out_path), wavs[0], sr)
        print(f"[ok] {out_path.name} ({len(wavs[0])/sr:.2f}s @ {sr}Hz)")
        results.append({**j, "out_wav": str(out_path), "sample_rate": sr, "ok": True})
        ok += 1
    except Exception as e:
        print(f"[FAIL] {j['lang']}_{j['idx']}: {type(e).__name__}: {e}", file=sys.stderr)
        results.append({**j, "ok": False, "err": str(e)})
        fail += 1

# 写 results
(HERE / "results.jsonl").write_text(
    "\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n"
)
print(f"\n=== 完成：ok={ok} fail={fail}  → {OUT_DIR} ===")
