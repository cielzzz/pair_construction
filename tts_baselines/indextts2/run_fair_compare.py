#!/usr/bin/env python
"""IndexTTS-2 公平对比版：spk_prompt=original_audio + text=ref_text
（与 MOSS-TTS 在 vcdata 阶段做的事完全一致）

输出：tts_baselines/indextts2/outputs_fair/{tag}.wav
"""
import json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOBS = HERE.parent / "jobs_fair_compare.jsonl"
OUT = HERE / "outputs_fair"
OUT.mkdir(exist_ok=True)
CKPT = HERE / "checkpoints"

print("Loading IndexTTS-2 ...")
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(
    cfg_path=str(CKPT / "config.yaml"),
    model_dir=str(CKPT),
    use_fp16=True,
    use_cuda_kernel=False,
)
print("Model ready.")

ok = fail = 0
for line in JOBS.open():
    j = json.loads(line)
    out_wav = OUT / f"{j['tag']}.wav"
    if out_wav.exists():
        print(f"[skip] {out_wav.name}")
        ok += 1; continue
    t0 = time.perf_counter()
    try:
        tts.infer(
            spk_audio_prompt=j["original_audio"],   # 同 MOSS-TTS：用 original 当音色 prompt
            text=j["ref_text"],                      # 同 MOSS-TTS：合成 ref_text
            output_path=str(out_wav),
            emo_alpha=1.0,
            verbose=False,
        )
        print(f"[ok] {out_wav.name}  {time.perf_counter()-t0:.1f}s")
        ok += 1
    except Exception as e:
        print(f"[FAIL] {j['tag']}: {type(e).__name__}: {e}", file=sys.stderr)
        fail += 1

print(f"\n=== 完成：ok={ok} fail={fail}  → {OUT} ===")
