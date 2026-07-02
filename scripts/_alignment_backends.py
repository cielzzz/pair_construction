"""Alignment backends for B1 local edit tooling.

The public entry point is :func:`extract_alignment`. Heavy ASR backends are
optional: when a backend is unavailable or only returns sentence-level text,
the module falls back to deterministic reference-text timing so downstream span
selection can still be exercised in M1 PoC tests.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf


PUNCT_RE = re.compile(r"[\s，。！？、；：,.!?;:\"'“”‘’（）()【】\[\]《》<>]+")
COMMON_ZH_WORDS = (
    "明天",
    "后天",
    "今天",
    "昨天",
    "上午",
    "下午",
    "晚上",
    "早上",
    "两点",
    "三点",
)


@dataclass
class BackendResult:
    asr_text: str | None
    words: list[dict[str, Any]] | None
    backend_detail: str | None = None
    elapsed_ms: float | None = None
    error: str | None = None


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    return PUNCT_RE.sub("", text)


def audio_duration_ms(audio_path: str | Path) -> int:
    info = sf.info(str(audio_path))
    if info.frames <= 0 or info.samplerate <= 0:
        return 0
    return int(round(info.frames * 1000.0 / info.samplerate))


def syllable_count(text: str) -> int:
    zh = re.findall(r"[\u4e00-\u9fff]", text)
    if zh:
        return len(zh)
    return max(1, len(re.findall(r"[A-Za-z0-9]+", text)) or len(text))


def segment_text(text: str) -> list[str]:
    """Segment text for local-edit spans.

    This intentionally keeps common editable anchors such as "明天" intact, then
    falls back to single Chinese characters for the remaining text. That gives
    stable timing for word-level edits without requiring jieba in tests.
    """

    cleaned = clean_text(text)
    words: list[str] = []
    i = 0
    while i < len(cleaned):
        matched = None
        for word in COMMON_ZH_WORDS:
            if cleaned.startswith(word, i):
                matched = word
                break
        if matched:
            words.append(matched)
            i += len(matched)
            continue
        ch = cleaned[i]
        if re.match(r"[A-Za-z0-9]", ch):
            j = i + 1
            while j < len(cleaned) and re.match(r"[A-Za-z0-9]", cleaned[j]):
                j += 1
            words.append(cleaned[i:j])
            i = j
        else:
            words.append(ch)
            i += 1
    return [w for w in words if w]


def distribute_words_over_duration(
    words: list[str],
    duration_ms: int,
    *,
    confidence: float | None,
) -> list[dict[str, Any]]:
    if not words:
        return []
    total_units = sum(syllable_count(w) for w in words) or len(words)
    cursor = 0
    out = []
    for idx, word in enumerate(words):
        units = syllable_count(word)
        if idx == len(words) - 1:
            end = duration_ms
        else:
            end = int(round((cursor + units) * duration_ms / total_units))
        start = int(round(cursor * duration_ms / total_units))
        if end <= start:
            end = start + 1
        out.append(
            {
                "text": word,
                "start_ms": start,
                "end_ms": end,
                "syllables": units,
                "confidence": confidence,
            }
        )
        cursor += units
    return out


def pauses_from_words(words: list[dict[str, Any]], pause_threshold_ms: int) -> list[dict[str, Any]]:
    pauses = []
    for idx, (left, right) in enumerate(zip(words, words[1:])):
        gap = int(right["start_ms"]) - int(left["end_ms"])
        if gap >= pause_threshold_ms:
            pauses.append(
                {
                    "after_word_idx": idx,
                    "start_ms": int(left["end_ms"]),
                    "duration_ms": gap,
                }
            )
    return pauses


def fallback_alignment(audio_path: str | Path, ref_text: str, *, confidence: float | None = None) -> BackendResult:
    duration = audio_duration_ms(audio_path)
    words = distribute_words_over_duration(segment_text(ref_text), duration, confidence=confidence)
    return BackendResult(asr_text=clean_text(ref_text), words=words, backend_detail="fallback_ref_text_equal_duration")


def run_qwen_sentence_asr(audio_path: str | Path, *, language: str = "zh") -> BackendResult:
    helper = Path(__file__).resolve().parent / "qwen_asr_batch.py"
    model = os.environ.get(
        "QWEN_ASR_MODEL",
        "/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/checkpoint/qwen-asr-1_7b",
    )
    python_bin = os.environ.get(
        "QWEN_ASR_PYTHON",
        "/inspire/ssd/project/embodied-multimodality/public/xyzhang/qwen3_asr_overlay/run_qwen3_asr.sh",
    )
    device = os.environ.get("QWEN_ASR_DEVICE", "cuda:0")
    start = time.time()
    try:
        with tempfile.TemporaryDirectory(prefix="b1_qwen_asr_") as tmp:
            manifest = Path(tmp) / "manifest.jsonl"
            output = Path(tmp) / "asr.jsonl"
            manifest.write_text(
                json.dumps({"uid": str(audio_path), "audio": str(audio_path), "language": language}, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            cmd = [
                python_bin,
                str(helper),
                "--input",
                str(manifest),
                "--output",
                str(output),
                "--model",
                model,
                "--device",
                device,
                "--batch-size",
                "1",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=int(os.environ.get("B1_QWEN_TIMEOUT", "300")))
            elapsed = (time.time() - start) * 1000.0
            if proc.returncode != 0:
                return BackendResult(None, None, "qwen_sentence", elapsed, proc.stderr.strip() or proc.stdout.strip())
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not rows or not rows[0].get("ok"):
                return BackendResult(None, None, "qwen_sentence", elapsed, rows[0].get("error") if rows else "empty output")
            return BackendResult(rows[0].get("text"), None, "qwen_sentence_no_timestamps", elapsed)
    except Exception as exc:
        return BackendResult(None, None, "qwen_sentence", (time.time() - start) * 1000.0, f"{type(exc).__name__}: {exc}")


def run_funasr_paraformer(audio_path: str | Path, model: str | None = None) -> BackendResult:
    start = time.time()
    try:
        from funasr import AutoModel

        model_name = model or os.environ.get(
            "B1_FUNASR_MODEL",
            "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vc_edit/models/Step-Audio-Tokenizer/dengcunqin/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online",
        )
        vad_model = os.environ.get("B1_FUNASR_VAD_MODEL", "")
        kwargs: dict[str, Any] = {"model": model_name}
        if vad_model:
            kwargs["vad_model"] = vad_model
        auto_model = AutoModel(**kwargs)
        result = auto_model.generate(input=str(audio_path), sentence_timestamp=True)
        elapsed = (time.time() - start) * 1000.0
        row = result[0] if isinstance(result, list) and result else result
        asr_text = row.get("text") if isinstance(row, dict) else None
        timestamps = row.get("timestamp") if isinstance(row, dict) else None
        if not timestamps:
            return BackendResult(asr_text, None, "funasr_paraformer_no_timestamps", elapsed)
        chars = list(clean_text(asr_text))
        words = []
        for ch, ts in zip(chars, timestamps):
            if not isinstance(ts, (list, tuple)) or len(ts) < 2:
                continue
            words.append(
                {
                    "text": ch,
                    "start_ms": int(ts[0]),
                    "end_ms": int(ts[1]),
                    "syllables": syllable_count(ch),
                    "confidence": None,
                }
            )
        return BackendResult(asr_text, words, "funasr_paraformer", elapsed)
    except Exception as exc:
        return BackendResult(None, None, "funasr_paraformer", (time.time() - start) * 1000.0, f"{type(exc).__name__}: {exc}")


def run_whisperx(audio_path: str | Path) -> BackendResult:
    start = time.time()
    try:
        import whisperx  # type: ignore  # noqa: F401

        return BackendResult(None, None, "whisperx_not_implemented", (time.time() - start) * 1000.0, "whisperx import succeeded but local runner is not wired")
    except Exception as exc:
        return BackendResult(None, None, "whisperx", (time.time() - start) * 1000.0, f"{type(exc).__name__}: {exc}")


def extract_alignment(
    audio_path: str | Path,
    ref_text: str,
    *,
    backend: str = "paraformer",
    pause_threshold_ms: int = 100,
) -> dict[str, Any]:
    audio_path = Path(audio_path)
    duration = audio_duration_ms(audio_path)
    backend = backend.lower().strip()
    if backend == "paraformer":
        result = run_funasr_paraformer(audio_path)
    elif backend == "qwen":
        result = run_qwen_sentence_asr(audio_path)
    elif backend == "whisperx":
        result = run_whisperx(audio_path)
    elif backend in {"fallback", "mock"}:
        result = fallback_alignment(audio_path, ref_text, confidence=1.0)
    else:
        raise ValueError(f"unsupported backend: {backend}")

    words = result.words
    backend_detail = result.backend_detail
    if not words:
        fallback = fallback_alignment(audio_path, ref_text, confidence=None)
        words = fallback.words or []
        backend_detail = f"{backend_detail or backend}; fallback={fallback.backend_detail}"

    asr_text = result.asr_text or clean_text(ref_text)
    return {
        "audio_id": audio_path.name,
        "audio": str(audio_path),
        "asr_text": asr_text,
        "ref_text": ref_text,
        "backend": backend,
        "backend_detail": backend_detail,
        "backend_error": result.error,
        "backend_elapsed_ms": result.elapsed_ms,
        "words": words,
        "pauses": pauses_from_words(words, pause_threshold_ms),
        "total_duration_ms": duration,
    }
