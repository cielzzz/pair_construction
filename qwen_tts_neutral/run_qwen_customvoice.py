#!/usr/bin/env python
"""验证 CustomVoice 默认中性输出（用 Vivian 预设音色，不传 instruct）"""
import json, sys
from pathlib import Path
import torch
import soundfile as sf

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs_customvoice"
OUT_DIR.mkdir(exist_ok=True)

# 选 3 句（中 2 + 英 1）做对比
TEXTS = [
    ("zh_044", "Chinese", "要剿灭国民党的特务斗争手法呢。"),
    ("zh_056", "Chinese", "那有什么关系呢？反正他早就恨死我了。"),
    ("en_048", "English", "What is it?"),
]

print("Loading Qwen3-TTS-12Hz-1.7B-CustomVoice ...")
from qwen_tts import Qwen3TTSModel
# 本地路径优先（避免重新下载）
LOCAL = "/inspire/hdd/project/embodied-multimodality/public/downloaded_ckpts/Qwen3-TTS-12Hz-1.7B-CustomVoice"
model = Qwen3TTSModel.from_pretrained(
    LOCAL,
    device_map="cuda:0",
    dtype=torch.bfloat16,
)
print("Model ready.")

for tag, lang, text in TEXTS:
    out_path = OUT_DIR / f"{tag}_vivian_default.wav"
    try:
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=lang,
            speaker="Vivian",      # 9 预设之一
            # instruct 故意不传 → 验证默认中性
        )
        sf.write(str(out_path), wavs[0], sr)
        print(f"[ok] {out_path.name} ({len(wavs[0])/sr:.2f}s @ {sr}Hz)  text={text}")
    except Exception as e:
        print(f"[FAIL] {tag}: {type(e).__name__}: {e}", file=sys.stderr)
print("Done.")
