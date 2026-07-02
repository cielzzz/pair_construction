from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def sample_alignment():
    return {
        "audio_id": "sample.wav",
        "asr_text": "明天下午两点我们开会",
        "words": [
            {"text": "明天", "start_ms": 0, "end_ms": 320},
            {"text": "下午", "start_ms": 320, "end_ms": 650},
            {"text": "两点", "start_ms": 650, "end_ms": 980},
            {"text": "我", "start_ms": 980, "end_ms": 1080},
            {"text": "们", "start_ms": 1080, "end_ms": 1180},
            {"text": "开", "start_ms": 1180, "end_ms": 1280},
            {"text": "会", "start_ms": 1280, "end_ms": 1380},
        ],
    }


def test_anchor_word_selects_first_hit(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    mod = load_script("select_span_14", "scripts/14_select_span.py")
    out = mod.select_spans(sample_alignment(), {"mode": "anchor_word", "params": {"anchor_word": "明天"}})
    assert out["selection_mode"] == "anchor_word"
    assert out["edit_spans"][0]["start_ms"] == 0
    assert out["edit_spans"][0]["end_ms"] == 320
    assert out["edit_spans"][0]["anchor_word_indices"] == [0]


def test_regex_maps_text_match_to_word_range(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    mod = load_script("select_span_14_regex", "scripts/14_select_span.py")
    out = mod.select_spans(sample_alignment(), {"mode": "regex", "params": {"regex": "两点我们"}})
    span = out["edit_spans"][0]
    assert span["anchor_word_indices"] == [2, 3, 4]
    assert span["start_ms"] == 650
    assert span["end_ms"] == 1180
