from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_fallback_alignment_on_sample_a(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    mod = load_script("extract_alignment_13", "scripts/13_extract_alignment.py")
    sr = 16000
    wav = np.zeros(sr, dtype=np.float32)
    audio = tmp_path / "sample_a.wav"
    sf.write(audio, wav, sr)

    out = mod.extract_alignment(audio, "明天下午开会", backend="mock")

    assert len(out["words"]) >= 4
    assert out["words"][0]["text"] == "明天"
    assert 0 <= out["words"][0]["start_ms"] < out["words"][0]["end_ms"]
    assert sum(w["syllables"] for w in out["words"]) == len("明天下午开会")
    assert out["total_duration_ms"] == 1000
