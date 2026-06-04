#!/usr/bin/env python
"""IndexTTS-2 voice clone 推 jobs.jsonl 里的 10 句

API：tts.infer(spk_audio_prompt=ref, text=target_text, output_path=...)
默认 emo_audio_prompt=None 时 IndexTTS-2 用 spk 同时做"音色+情绪"源（保留 ref 情绪），
正是 D-clone "高表现→高表现 同情绪换文本" 需要的行为。

用法（必须用 uv run，因为 IndexTTS-2 依赖独立 env）：
  cd .../tts_baselines/indextts2/repo && uv run python ../run_inference.py
"""
import json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent       # tts_baselines/indextts2/
JOBS = HERE.parent / "jobs.jsonl"             # tts_baselines/jobs.jsonl
OUT = HERE / "outputs"
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
results = []
for line in JOBS.open():
    j = json.loads(line)
    out_wav = OUT / f"{j['tag']}.wav"
    if out_wav.exists():
        print(f"[skip] exists: {out_wav.name}")
        results.append({**j, "out_wav": str(out_wav), "ok": True})
        ok += 1; continue
    t0 = time.perf_counter()
    try:
        tts.infer(
            spk_audio_prompt=j["ref_audio"],
            text=j["target_text"],
            output_path=str(out_wav),
            emo_alpha=1.0,                # 完全跟 ref 的情绪
            verbose=False,
        )
        dt = time.perf_counter() - t0
        print(f"[ok] {out_wav.name}  {dt:.1f}s")
        results.append({**j, "out_wav": str(out_wav), "infer_sec": round(dt, 1), "ok": True})
        ok += 1
    except Exception as e:
        print(f"[FAIL] {j['tag']}: {type(e).__name__}: {e}", file=sys.stderr)
        results.append({**j, "ok": False, "err": str(e)})
        fail += 1

(HERE / "results.jsonl").write_text(
    "\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n"
)
print(f"\n=== 完成：ok={ok} fail={fail}  → {OUT} ===")
