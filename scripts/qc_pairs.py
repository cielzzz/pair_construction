#!/usr/bin/env python
"""Quality gate for final pair jsonl outputs.

What it checks:
1. Prefer speaker-sim enriched inputs from pairs/scored/*.jsonl when present.
2. Pair-type semantic gate (B/C/D/H1/H2/H3 family).
3. DNSMOS (reused from emotion/per_file_dual.csv via EmotionTable).
4. Audio health: duration, silence ratio, trailing silence, clipping, tail activity.
5. ASR-vs-text consistency (optional, default on) through a pluggable Qwen-ASR helper.
6. Truncation suspicion: low ASR match + abrupt tail / too dense speech.
7. Speaker similarity gate is applied here, instead of relying on *_filtered.jsonl.
8. Repetition/loop/stutter suspicion from ASR text.

Input:
  --pair-root /abs/path/to/pair_outputs/<lang>/<split>

Outputs:
  <pair_root>/quality_gate/
    summary.md
    summary.json
    <pair_name>_qc.csv
    <pair_name>_qc.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from _emotion_lookup import EmotionTable
from _utils import load_config


DEFAULT_QWEN_ASR_PYTHON = "/inspire/ssd/project/embodied-multimodality/public/xyzhang/qwen3_asr_overlay/run_qwen3_asr.sh"
DEFAULT_QWEN_ASR_MODEL = "/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/checkpoint/qwen-asr-1_7b"
SENSEVOICE_TAG_RE = re.compile(r"<\|[^>]+\|>")
SPACE_RE = re.compile(r"\s+")

SPEAKER_SIM_CFG_KEY = {
    "A": "a",
    "B": "bc",
    "C": "bc",
    "C_mixed": "c_mixed",
    "D": "d",
    "D_st": "d_st",
    "D_cross_emo": "d_cross_emo",
    "Genre": "genre",
    "Genre_conv": "genre_conv",
    "H1": "h1",
    "H2": "h2",
    "I": "i",
    "J_fast": "j_fast",
    "J_slow": "j_slow",
}

PROSODY_TRANSFER_PAIR_TYPES = {"I"}
SPEED_PAIR_TYPES = {"J_fast", "J_slow"}


def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def log(msg: str) -> None:
    print(msg, flush=True)


def write_progress(path: Path, **payload: Any) -> None:
    payload.setdefault("ok", 1)
    write_json(path, payload)


def coerce_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def first_nonempty(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def merge_emotion_summary(row_summary: Any, table_summary: dict[str, Any]) -> dict[str, Any]:
    merged = dict(table_summary or {})
    if isinstance(row_summary, dict):
        for key, value in row_summary.items():
            if value not in (None, ""):
                merged[key] = value
    return merged


def coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def mean_or_none(values: list[float | int | None]) -> float | None:
    kept = [float(v) for v in values if v is not None]
    if not kept:
        return None
    return float(np.mean(kept))


def infer_lang(text: str | None) -> str:
    if not text:
        return "auto"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    return "en"


def clean_asr_text(text: str | None) -> str:
    if not text:
        return ""
    text = SENSEVOICE_TAG_RE.sub(" ", text)
    text = text.replace("▁", " ")
    text = unicodedata.normalize("NFKC", text)
    return SPACE_RE.sub(" ", text).strip()


def normalize_text(text: str | None, lang: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    if lang == "zh":
        kept = []
        for ch in text:
            cat = unicodedata.category(ch)
            if re.match(r"[\u4e00-\u9fffA-Za-z0-9]", ch):
                kept.append(ch.lower())
            elif cat.startswith("L") or cat.startswith("N"):
                kept.append(ch.lower())
        return "".join(kept)
    text = text.lower()
    text = text.replace("’", "'")
    tokens = re.findall(r"[a-z0-9']+", text)
    return " ".join(tokens)


def text_units(text: str | None, lang: str) -> int:
    norm = normalize_text(text, lang)
    if not norm:
        return 0
    if lang == "zh":
        return len(norm)
    return len([t for t in norm.split() if t])


def text_match_score(ref_text: str | None, hyp_text: str | None, lang: str) -> float | None:
    ref = normalize_text(ref_text, lang)
    hyp = normalize_text(hyp_text, lang)
    if not ref or not hyp:
        return None
    if lang == "zh":
        err = edit_error_rate(list(ref), list(hyp))
    else:
        err = edit_error_rate(ref.split(), hyp.split())
    return max(0.0, 1.0 - float(err))


def edit_error_rate(ref_seq: list, hyp_seq: list) -> float:
    if not ref_seq:
        return 0.0 if not hyp_seq else 1.0
    dp = np.zeros((len(ref_seq) + 1, len(hyp_seq) + 1), dtype=np.int32)
    dp[:, 0] = np.arange(len(ref_seq) + 1)
    dp[0, :] = np.arange(len(hyp_seq) + 1)
    for i in range(1, len(ref_seq) + 1):
        for j in range(1, len(hyp_seq) + 1):
            cost = 0 if ref_seq[i - 1] == hyp_seq[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost,
            )
    return float(dp[len(ref_seq), len(hyp_seq)] / max(1, len(ref_seq)))


def load_audio_stats(path: str) -> dict[str, Any]:
    try:
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        if wav.size == 0:
            return {"load_ok": False, "error": "empty_audio"}
        peak = float(np.max(np.abs(wav)))
        global_rms = float(np.sqrt(np.mean(np.square(wav))) + 1e-12)
        frame = max(1, int(sr * 0.02))
        pad = (-len(wav)) % frame
        wav_pad = np.pad(wav, (0, pad)) if pad else wav
        frames = wav_pad.reshape(-1, frame)
        rms = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)
        ref_amp = float(np.percentile(np.abs(wav), 95))
        silence_th = max(1e-4, ref_amp * 0.02)
        silent = rms <= silence_th
        leading = 0
        for v in silent:
            if v:
                leading += 1
            else:
                break
        trailing = 0
        for v in silent[::-1]:
            if v:
                trailing += 1
            else:
                break
        tail_samples = max(1, int(sr * 0.2))
        tail_rms = float(np.sqrt(np.mean(np.square(wav[-tail_samples:]))) + 1e-12)
        duration_sec = float(len(wav) / sr)
        return {
            "load_ok": True,
            "sr": int(sr),
            "duration_sec": duration_sec,
            "peak_abs": peak,
            "global_rms": global_rms,
            "clip_ratio": float(np.mean(np.abs(wav) >= 0.999)),
            "silence_ratio": float(np.mean(silent)),
            "leading_silence_sec": float(leading * frame / sr),
            "trailing_silence_sec": float(trailing * frame / sr),
            "tail_rms_ratio": float(tail_rms / global_rms),
        }
    except Exception as exc:
        return {"load_ok": False, "error": f"{type(exc).__name__}: {exc}"}


def chunked(seq: list[dict], size: int):
    if size <= 0:
        size = len(seq) or 1
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def discover_pair_files(pair_dir: Path, pair_type: str, prefer_scored: bool) -> list[Path]:
    if pair_type != "all":
        requested = [x.strip() for x in pair_type.split(",") if x.strip()]
        files = []
        for item in requested:
            raw_path = pair_dir / f"{item}.jsonl"
            scored_path = pair_dir / "scored" / f"{item}.jsonl"
            if prefer_scored and scored_path.exists():
                files.append(scored_path)
                continue
            files.extend([p for p in (raw_path, scored_path) if p.exists()][:1])
        return files

    files = []
    for raw_path in sorted(pair_dir.glob("*.jsonl")):
        stem = raw_path.stem
        if stem.endswith("_filtered") or stem.endswith("_bakfilt"):
            continue
        chosen = raw_path
        scored_path = pair_dir / "scored" / raw_path.name
        if prefer_scored and scored_path.exists():
            chosen = scored_path
        files.append(chosen)
    return files


def build_summary_md(summary: dict[str, Any]) -> str:
    pair_root = summary.get("pair_root")
    md_lines = [
        "# Pair QC Summary",
        "",
        f"- pair_root: `{pair_root}`",
        f"- prefer_scored: `{summary.get('prefer_scored')}`",
        f"- fail_on_speaker_sim: `{summary.get('fail_on_speaker_sim')}`",
        f"- fail_on_semantic: `{summary.get('fail_on_semantic')}`",
        f"- asr_source: `{summary.get('asr_source')}`",
        f"- asr_backend_error: `{summary.get('asr_backend_error')}`",
        f"- emotion_asr_hits: `{summary.get('emotion_asr_hits')}`",
        f"- qwen_fallback_count: `{summary.get('qwen_fallback_count')}`",
        "",
    ]
    for pair_name, file_summary in summary.get("files", {}).items():
        n = file_summary.get("count")
        md_lines += [
            f"## {pair_name}",
            f"- count: `{n}`",
            f"- hard_pass: `{file_summary.get('hard_pass')}/{n}`",
            f"- qc_pass: `{file_summary.get('qc_pass')}/{n}`",
            f"- speaker_sim_min: `{file_summary.get('speaker_sim_min')}`",
            f"- avg_speaker_sim: `{file_summary.get('avg_speaker_sim')}`",
            f"- semantic_fail: `{file_summary.get('semantic_fail')}`",
            f"- prosody_fail: `{file_summary.get('prosody_fail')}`",
            f"- avg_prosody_duration_ratio: `{file_summary.get('avg_prosody_duration_ratio')}`",
            f"- avg_prosody_pause_pattern_jaccard: `{file_summary.get('avg_prosody_pause_pattern_jaccard')}`",
            f"- ref_truncation_suspect: `{file_summary.get('ref_truncation_suspect')}`",
            f"- truncation_suspect: `{file_summary.get('truncation_suspect')}`",
            f"- top_flags: `{dict(Counter(file_summary.get('flag_counts', {})).most_common(10))}`",
            "",
        ]
    return "\n".join(md_lines)


def canonical_pair_output_name(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_filtered"):
        return stem[:-len("_filtered")]
    if stem.endswith("_bakfilt"):
        return stem[:-len("_bakfilt")]
    return stem


def cleanup_old_qc_outputs(out_dir: Path, pair_name: str) -> None:
    csv_dir = out_dir / "csv"
    for name in (
        f"{pair_name}__qc.csv",
        f"{pair_name}__qc.jsonl",
        f"{pair_name}_qc.csv",
        f"{pair_name}_qc.jsonl",
    ):
        path = out_dir / name
        if path.exists():
            path.unlink()
        csv_path = csv_dir / name
        if csv_path.exists():
            csv_path.unlink()


def load_emotion_table(emotion_dir: Path) -> EmotionTable:
    et = EmotionTable()
    et.load_csv(emotion_dir / "per_file_dual.csv")
    et.load_per_pair_for_src(emotion_dir / "per_pair.csv")
    et.load_all_link_mappings(emotion_dir)
    return et


def run_qwen_asr(
    records: list[dict],
    helper_path: Path,
    python_bin: str,
    model_path: str,
    device: str,
    batch_size: int,
    chunk_size: int,
) -> tuple[dict[str, dict], str | None]:
    if not records:
        return {}, None
    with tempfile.TemporaryDirectory(prefix="pair_qc_asr_") as td:
        result_map = {}
        backend_errs: list[str] = []
        chunks = list(chunked(records, chunk_size))
        total = len(chunks)
        log(
            f"[qc] start qwen_asr_fallback count={len(records)} "
            f"chunks={total} batch_size={batch_size} device={device}"
        )
        for idx, chunk_records in enumerate(chunks, start=1):
            manifest = Path(td) / f"manifest_{idx:04d}.jsonl"
            output = Path(td) / f"asr_{idx:04d}.jsonl"
            write_jsonl(manifest, chunk_records)
            cmd = [
                python_bin,
                str(helper_path),
                "--input",
                str(manifest),
                "--output",
                str(output),
                "--model",
                model_path,
                "--device",
                device,
                "--batch-size",
                str(batch_size),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            chunk_err = None
            if proc.returncode != 0:
                chunk_err = f"helper_exit_{proc.returncode}: {proc.stderr.strip() or proc.stdout.strip()}"
            if not output.exists():
                chunk_err = chunk_err or "asr_output_missing"
                backend_errs.append(f"chunk_{idx}: {chunk_err}")
                log(f"[qc] qwen_asr chunk {idx}/{total} failed err={chunk_err}")
                continue
            chunk_rows = list(iter_jsonl(output))
            ok_count = 0
            err_count = 0
            first_error = None
            for row in chunk_rows:
                result_map[row["uid"]] = row
                if row.get("ok"):
                    ok_count += 1
                else:
                    err_count += 1
                    first_error = first_error or row.get("error")
            if chunk_err:
                backend_errs.append(f"chunk_{idx}: {chunk_err}")
            elif first_error and err_count == len(chunk_rows):
                backend_errs.append(f"chunk_{idx}: {first_error}")
            log(
                f"[qc] qwen_asr chunk {idx}/{total} done "
                f"rows={len(chunk_rows)} ok={ok_count} err={err_count}"
            )
        backend_err = "; ".join(backend_errs[:8]) if backend_errs else None
        return result_map, backend_err


def asr_from_emotion(audio_path: str, emotion_table: EmotionTable) -> dict[str, Any] | None:
    rec = emotion_table.get(audio_path)
    if not rec:
        return None
    text = clean_asr_text(rec.get("sv_raw") or rec.get("asr_text") or rec.get("text"))
    if not text:
        return None
    return {
        "uid": audio_path,
        "ok": 1,
        "text": text,
        "language": infer_lang(text),
        "source": "sensevoice",
    }


def speaker_sim_min_for_pair_type(cfg: dict, pair_type: str) -> float | None:
    cfg_key = SPEAKER_SIM_CFG_KEY.get(pair_type)
    if not cfg_key:
        return None
    section = cfg.get(cfg_key, {})
    for key in ("timbre_speaker_sim_min_wavlm", "speaker_sim_min_wavlm", "speaker_sim_min", "sim_min"):
        value = coerce_float(section.get(key))
        if value is not None and value > 0:
            return value
    return None


def resolve_speaker_sim(row: dict[str, Any], pair_type: str | None = None) -> float | None:
    keys = []
    if pair_type in PROSODY_TRANSFER_PAIR_TYPES:
        keys.extend(
            (
                "timbre_ref_vs_tgt_speaker_sim_wavlm",
                "timbre_ref_vs_tgt_speaker_sim",
            )
        )
    else:
        keys.extend(
            (
                "ref_vs_tgt_speaker_sim_wavlm",
                "ref_vs_tgt_speaker_sim",
                "speaker_similarity",
                "speaker_sim",
            )
        )
    for key in keys:
        value = coerce_float(row.get(key))
        if value is not None:
            return value
    return None


def emotion_top1_label(emo: dict[str, Any] | None) -> str | None:
    if not emo:
        return None
    for key in ("top1_label", "sv_label", "label"):
        value = emo.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return None


def emotion_neutral_prob(emo: dict[str, Any] | None) -> float | None:
    value = coerce_float((emo or {}).get("P_neutral"))
    if value is not None:
        return value
    label = emotion_top1_label(emo)
    if label is None:
        return None
    return 1.0 if label == "neutral" else 0.0


def is_neutral_emotion(emo: dict[str, Any] | None) -> bool | None:
    value = emotion_neutral_prob(emo)
    if value is None:
        return None
    return value >= 0.5


def normalized_same_text(ref_text: str | None, tgt_text: str | None) -> bool | None:
    if not ref_text or not tgt_text:
        return None
    lang = "zh" if infer_lang(ref_text) == "zh" or infer_lang(tgt_text) == "zh" else "en"
    return normalize_text(ref_text, lang) == normalize_text(tgt_text, lang)


def normalized_same_audio(ref_audio: str | None, tgt_audio: str | None) -> bool | None:
    if not ref_audio or not tgt_audio:
        return None
    use_realpath = os.environ.get("QC_AUDIO_IDENTITY_REALPATH")
    if use_realpath is None:
        use_realpath = os.environ.get("EMOTION_LOOKUP_REALPATH")
    if str(use_realpath or "1").strip().lower() in {"0", "false", "no", "off"}:
        return ref_audio == tgt_audio
    try:
        return os.path.realpath(ref_audio) == os.path.realpath(tgt_audio)
    except OSError:
        return ref_audio == tgt_audio


def normalized_tokens(text: str | None, lang: str) -> list[str]:
    norm = normalize_text(text, lang)
    if not norm:
        return []
    if lang == "zh":
        return list(norm)
    return [tok for tok in norm.split() if tok]


def max_consecutive_run(tokens: list[str]) -> int:
    if not tokens:
        return 0
    best = 1
    cur = 1
    for idx in range(1, len(tokens)):
        if tokens[idx] == tokens[idx - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def max_consecutive_ngram_repeat(tokens: list[str], max_ngram: int = 4) -> tuple[int, int, tuple[str, ...]]:
    if len(tokens) < 4:
        return 0, 1, ()
    best_n = 0
    best_repeat = 1
    best_pattern: tuple[str, ...] = ()
    upper = min(max_ngram, max(2, len(tokens) // 2))
    for ngram_size in range(2, upper + 1):
        for start in range(0, len(tokens) - ngram_size + 1):
            pattern = tuple(tokens[start : start + ngram_size])
            repeat = 1
            cursor = start + ngram_size
            while cursor + ngram_size <= len(tokens) and tuple(tokens[cursor : cursor + ngram_size]) == pattern:
                repeat += 1
                cursor += ngram_size
            if repeat > best_repeat:
                best_n = ngram_size
                best_repeat = repeat
                best_pattern = pattern
    return best_n, best_repeat, best_pattern


def detect_repetition_flags(
    expected_text: str | None,
    hyp_text: str | None,
    lang: str,
    side: str,
    *,
    token_run_threshold: int,
    ngram_repeat_threshold: int,
) -> tuple[list[str], int, int]:
    exp_tokens = normalized_tokens(expected_text, lang)
    hyp_tokens = normalized_tokens(hyp_text, lang)
    if not hyp_tokens:
        return [], 0, 1

    flags: list[str] = []
    exp_run = max_consecutive_run(exp_tokens)
    hyp_run = max_consecutive_run(hyp_tokens)
    if hyp_run >= token_run_threshold and hyp_run >= exp_run + 2:
        flags.append(f"{side}_asr_stutter_suspect")

    _, exp_loop_repeat, _ = max_consecutive_ngram_repeat(exp_tokens)
    _, hyp_loop_repeat, _ = max_consecutive_ngram_repeat(hyp_tokens)
    if hyp_loop_repeat >= ngram_repeat_threshold and hyp_loop_repeat >= exp_loop_repeat + 1:
        flags.append(f"{side}_asr_loop_suspect")

    return flags, hyp_run, hyp_loop_repeat


def evaluate_pair_semantics(
    row: dict[str, Any],
    pair_name: str,
    ref_emo: dict[str, Any] | None,
    tgt_emo: dict[str, Any] | None,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    ref_text = row.get("reference_text")
    tgt_text = row.get("target_text")
    ref_audio = row.get("reference_audio")
    tgt_audio = row.get("target_audio")

    same_text = normalized_same_text(ref_text, tgt_text)
    same_audio = normalized_same_audio(ref_audio, tgt_audio)
    ref_label = emotion_top1_label(ref_emo)
    tgt_label = emotion_top1_label(tgt_emo)
    ref_neutral = is_neutral_emotion(ref_emo)
    tgt_neutral = is_neutral_emotion(tgt_emo)
    ref_p_neutral = emotion_neutral_prob(ref_emo)
    tgt_p_neutral = emotion_neutral_prob(tgt_emo)

    semantic_flags: list[str] = []
    checked = False

    def require_same_text() -> None:
        if same_text is None:
            semantic_flags.append("semantic_missing_text")
        elif not same_text:
            semantic_flags.append("semantic_text_mismatch")

    if pair_name == "B":
        checked = True
        require_same_text()
        if ref_neutral is not True:
            semantic_flags.append("semantic_ref_not_neutral")
        if tgt_neutral is not False:
            semantic_flags.append("semantic_tgt_not_non_neutral")
    elif pair_name in {"C", "C_mixed"}:
        checked = True
        if ref_neutral is not False:
            semantic_flags.append("semantic_ref_not_non_neutral")
        if tgt_neutral is not True:
            semantic_flags.append("semantic_tgt_not_neutral")
    elif pair_name in {"D", "D_st"}:
        checked = True
        if ref_neutral is not False:
            semantic_flags.append("semantic_ref_not_non_neutral")
        if tgt_neutral is not False:
            semantic_flags.append("semantic_tgt_not_non_neutral")
        if ref_label is None or tgt_label is None:
            semantic_flags.append("semantic_missing_top1_label")
        elif ref_label != tgt_label:
            semantic_flags.append("semantic_top1_not_same")
    elif pair_name == "D_cross_emo":
        checked = True
        if ref_neutral is not False:
            semantic_flags.append("semantic_ref_not_non_neutral")
        if tgt_neutral is not False:
            semantic_flags.append("semantic_tgt_not_non_neutral")
        if ref_label is None or tgt_label is None:
            semantic_flags.append("semantic_missing_top1_label")
        elif ref_label == tgt_label:
            semantic_flags.append("semantic_top1_not_cross")
    elif pair_name == "H1":
        checked = True
        if ref_label is None or tgt_label is None:
            semantic_flags.append("semantic_missing_top1_label")
        elif ref_label != tgt_label:
            semantic_flags.append("semantic_top1_not_same")
    elif pair_name == "H2":
        checked = True
        require_same_text()
        if ref_neutral is not True:
            semantic_flags.append("semantic_ref_not_neutral")
        if tgt_neutral is not True:
            semantic_flags.append("semantic_tgt_not_neutral")
        h2_cfg = cfg.get("h2", {}) if isinstance(cfg, dict) else {}
        if h2_cfg.get("require_target_more_neutral", True):
            margin = coerce_float(h2_cfg.get("target_more_neutral_margin")) or 0.0
            if ref_p_neutral is None or tgt_p_neutral is None:
                semantic_flags.append("semantic_missing_neutral_prob")
            elif tgt_p_neutral + 1e-8 < ref_p_neutral + margin:
                semantic_flags.append("semantic_tgt_not_more_neutral")
    elif pair_name in {"Genre", "Genre_conv"}:
        checked = True
        require_same_text()
        if same_audio is None:
            semantic_flags.append("semantic_missing_audio_identity")
        elif same_audio:
            semantic_flags.append("semantic_same_audio")
    elif pair_name == "H3":
        checked = True
        if same_audio is None:
            semantic_flags.append("semantic_missing_audio_identity")
        elif same_audio:
            semantic_flags.append("semantic_same_audio")
        if same_text is True:
            semantic_flags.append("semantic_same_text")
    elif pair_name in SPEED_PAIR_TYPES:
        checked = True
        require_same_text()
        if same_audio is None:
            semantic_flags.append("semantic_missing_audio_identity")
        elif same_audio:
            semantic_flags.append("semantic_same_audio")
    elif pair_name in PROSODY_TRANSFER_PAIR_TYPES:
        checked = True
        require_same_text()
        if same_audio is None:
            semantic_flags.append("semantic_missing_audio_identity")
        elif same_audio:
            semantic_flags.append("semantic_same_audio")
        timbre_audio = row.get("timbre_ref_audio")
        if not timbre_audio:
            semantic_flags.append("semantic_missing_timbre_ref_audio")
        prosody_audio = row.get("prosody_ref_audio") or ref_audio
        if normalized_same_audio(prosody_audio, timbre_audio) is True:
            semantic_flags.append("semantic_prosody_timbre_same_audio")

    semantic_ok = None if not checked else (len(semantic_flags) == 0)
    return {
        "semantic_ok": semantic_ok,
        "semantic_flags": semantic_flags,
        "same_text": same_text,
        "same_audio": same_audio,
        "ref_top1_label": ref_label,
        "tgt_top1_label": tgt_label,
        "ref_p_neutral": ref_p_neutral,
        "tgt_p_neutral": tgt_p_neutral,
    }


def nested_metric(metrics: dict[str, Any], *path: str) -> float | None:
    value: Any = metrics
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return coerce_float(value)


def evaluate_prosody_metrics(row: dict[str, Any], pair_name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    checked = pair_name in SPEED_PAIR_TYPES or pair_name in PROSODY_TRANSFER_PAIR_TYPES
    flags: list[str] = []
    metrics = row.get("prosody_metrics")
    if not checked:
        return {
            "prosody_ok": None,
            "prosody_flags": flags,
            "duration_ratio": None,
            "pause_pattern_jaccard": None,
            "energy_contour_corr": None,
            "f0_contour_corr": None,
            "prosody_similarity_proxy": None,
            "speed_direction_pass": None,
        }

    if row.get("prosody_metrics_error"):
        flags.append("prosody_metrics_error")
    if not isinstance(metrics, dict):
        flags.append("prosody_metrics_missing")
        return {
            "prosody_ok": False,
            "prosody_flags": flags,
            "duration_ratio": None,
            "pause_pattern_jaccard": None,
            "energy_contour_corr": None,
            "f0_contour_corr": None,
            "prosody_similarity_proxy": None,
            "speed_direction_pass": None,
        }

    duration_ratio = nested_metric(metrics, "duration_ratio_tgt_over_ref")
    pause_pattern_jaccard = nested_metric(metrics, "pause_pattern_jaccard")
    energy_contour_corr = nested_metric(metrics, "energy_contour_corr")
    f0_contour_corr = nested_metric(metrics, "f0_contour_corr")
    prosody_similarity_proxy = nested_metric(metrics, "prosody_similarity_proxy")
    speed_direction_pass = coerce_float(metrics.get("speed_direction_pass"))

    section = cfg.get(SPEAKER_SIM_CFG_KEY.get(pair_name, ""), {}) if isinstance(cfg, dict) else {}
    if pair_name in SPEED_PAIR_TYPES:
        if coerce_bool(section.get("require_speed_direction_pass"), True) and speed_direction_pass != 1:
            flags.append("speed_direction_fail")

    if pair_name in PROSODY_TRANSFER_PAIR_TYPES:
        min_ratio = coerce_float(section.get("duration_ratio_min"))
        max_ratio = coerce_float(section.get("duration_ratio_max"))
        min_pause = coerce_float(section.get("pause_pattern_jaccard_min"))
        min_energy = coerce_float(section.get("energy_contour_corr_min"))
        min_proxy = coerce_float(section.get("prosody_similarity_proxy_min"))
        if duration_ratio is None:
            flags.append("prosody_duration_ratio_missing")
        else:
            if min_ratio is not None and duration_ratio < min_ratio:
                flags.append("prosody_duration_ratio_low")
            if max_ratio is not None and duration_ratio > max_ratio:
                flags.append("prosody_duration_ratio_high")
        if min_pause is not None:
            if pause_pattern_jaccard is None:
                flags.append("prosody_pause_pattern_missing")
            elif pause_pattern_jaccard < min_pause:
                flags.append("prosody_pause_pattern_low")
        if min_energy is not None:
            if energy_contour_corr is None:
                flags.append("prosody_energy_corr_missing")
            elif energy_contour_corr < min_energy:
                flags.append("prosody_energy_corr_low")
        if min_proxy is not None:
            if prosody_similarity_proxy is None:
                flags.append("prosody_similarity_missing")
            elif prosody_similarity_proxy < min_proxy:
                flags.append("prosody_similarity_low")

    return {
        "prosody_ok": len(flags) == 0,
        "prosody_flags": flags,
        "duration_ratio": duration_ratio,
        "pause_pattern_jaccard": pause_pattern_jaccard,
        "energy_contour_corr": energy_contour_corr,
        "f0_contour_corr": f0_contour_corr,
        "prosody_similarity_proxy": prosody_similarity_proxy,
        "speed_direction_pass": speed_direction_pass,
    }


def flag_and_score_pair(
    row: dict,
    cfg: dict[str, Any],
    emotion_table: EmotionTable,
    audio_stats: dict[str, dict],
    asr_map: dict[str, dict],
    thresholds: dict[str, float],
    *,
    pair_name: str,
    speaker_sim_min: float | None,
    fail_on_speaker_sim: bool,
    fail_on_semantic: bool,
) -> dict[str, Any]:
    ref_audio = row.get("reference_audio")
    tgt_audio = row.get("target_audio")
    ref_text = row.get("reference_text")
    tgt_text = row.get("target_text")
    ref_lang = infer_lang(ref_text)
    tgt_lang = infer_lang(tgt_text)

    ref_stats = audio_stats.get(ref_audio, {"load_ok": False, "error": "missing_stats"})
    tgt_stats = audio_stats.get(tgt_audio, {"load_ok": False, "error": "missing_stats"})
    ref_emo = merge_emotion_summary(row.get("ref_emotion"), emotion_table.emotion_summary(ref_audio))
    tgt_emo = merge_emotion_summary(row.get("tgt_emotion"), emotion_table.emotion_summary(tgt_audio))

    ref_asr = asr_map.get(ref_audio, {})
    tgt_asr = asr_map.get(tgt_audio, {})
    ref_asr_text = ref_asr.get("text")
    tgt_asr_text = tgt_asr.get("text")
    ref_asr_score = text_match_score(ref_text, ref_asr_text, ref_lang)
    tgt_asr_score = text_match_score(tgt_text, tgt_asr_text, tgt_lang)

    flags: list[str] = []
    if not ref_stats.get("load_ok"):
        flags.append("ref_audio_load_failed")
    if not tgt_stats.get("load_ok"):
        flags.append("tgt_audio_load_failed")

    ref_dnsmos = coerce_float(first_nonempty(ref_emo.get("dnsmos_ovrl"), row.get("ref_dnsmos_ovrl")))
    tgt_dnsmos = coerce_float(first_nonempty(tgt_emo.get("dnsmos_ovrl"), row.get("tgt_dnsmos_ovrl")))
    ref_dnsmos_sig = coerce_float(first_nonempty(ref_emo.get("dnsmos_sig"), row.get("ref_dnsmos_sig")))
    tgt_dnsmos_sig = coerce_float(first_nonempty(tgt_emo.get("dnsmos_sig"), row.get("tgt_dnsmos_sig")))
    ref_dnsmos_bak = coerce_float(first_nonempty(ref_emo.get("dnsmos_bak"), row.get("ref_dnsmos_bak")))
    tgt_dnsmos_bak = coerce_float(first_nonempty(tgt_emo.get("dnsmos_bak"), row.get("tgt_dnsmos_bak")))
    ref_top1_prob = coerce_float(first_nonempty(ref_emo.get("top1_prob"), row.get("ref_top1_prob")))
    tgt_top1_prob = coerce_float(first_nonempty(tgt_emo.get("top1_prob"), row.get("tgt_top1_prob")))
    ref_sv_label = first_nonempty(ref_emo.get("sv_label"), row.get("ref_sv_label"))
    tgt_sv_label = first_nonempty(tgt_emo.get("sv_label"), row.get("tgt_sv_label"))
    if ref_dnsmos is not None and ref_dnsmos < thresholds["dnsmos_min"]:
        flags.append("ref_dnsmos_low")
    if tgt_dnsmos is not None and tgt_dnsmos < thresholds["dnsmos_min"]:
        flags.append("tgt_dnsmos_low")

    for side, stats in (("ref", ref_stats), ("tgt", tgt_stats)):
        if not stats.get("load_ok"):
            continue
        if stats["duration_sec"] < thresholds["min_duration_sec"]:
            flags.append(f"{side}_too_short")
        if stats["clip_ratio"] > thresholds["max_clip_ratio"]:
            flags.append(f"{side}_clipping")
        if stats["silence_ratio"] > thresholds["max_silence_ratio"]:
            flags.append(f"{side}_high_silence_ratio")
        if stats["trailing_silence_sec"] > thresholds["max_trailing_silence_sec"]:
            flags.append(f"{side}_long_trailing_silence")

    if ref_asr and not ref_asr.get("ok"):
        flags.append("ref_asr_error")
    if tgt_asr and not tgt_asr.get("ok"):
        flags.append("tgt_asr_error")
    if ref_asr_score is not None and ref_asr_score < thresholds["ref_asr_min"]:
        flags.append("ref_asr_mismatch")
    if tgt_asr_score is not None and tgt_asr_score < thresholds["tgt_asr_min"]:
        flags.append("tgt_asr_mismatch")

    ref_units = text_units(ref_text, ref_lang)
    tgt_units = text_units(tgt_text, tgt_lang)
    ref_units_per_sec = None
    tgt_units_per_sec = None
    if ref_stats.get("load_ok") and ref_stats["duration_sec"] > 0:
        ref_units_per_sec = ref_units / ref_stats["duration_sec"]
        max_rate = thresholds["zh_max_units_per_sec"] if ref_lang == "zh" else thresholds["en_max_units_per_sec"]
        if ref_units_per_sec > max_rate:
            flags.append("ref_text_density_high")
    if tgt_stats.get("load_ok") and tgt_stats["duration_sec"] > 0:
        tgt_units_per_sec = tgt_units / tgt_stats["duration_sec"]
        max_rate = thresholds["zh_max_units_per_sec"] if tgt_lang == "zh" else thresholds["en_max_units_per_sec"]
        if tgt_units_per_sec > max_rate:
            flags.append("tgt_text_density_high")

    ref_repetition_flags, ref_max_repeat_run, ref_max_loop_repeat = detect_repetition_flags(
        ref_text,
        ref_asr_text,
        ref_lang,
        "ref",
        token_run_threshold=int(thresholds["repeat_token_run_threshold"]),
        ngram_repeat_threshold=int(thresholds["repeat_ngram_repeat_threshold"]),
    )
    tgt_repetition_flags, tgt_max_repeat_run, tgt_max_loop_repeat = detect_repetition_flags(
        tgt_text,
        tgt_asr_text,
        tgt_lang,
        "tgt",
        token_run_threshold=int(thresholds["repeat_token_run_threshold"]),
        ngram_repeat_threshold=int(thresholds["repeat_ngram_repeat_threshold"]),
    )
    flags.extend(ref_repetition_flags)
    flags.extend(tgt_repetition_flags)

    ref_truncation_suspect = False
    truncation_suspect = False
    if ref_stats.get("load_ok") and ref_asr_score is not None and ref_asr_score < thresholds["ref_asr_min"]:
        if (
            ref_stats["trailing_silence_sec"] < thresholds["truncation_tail_silence_max"]
            and ref_stats["tail_rms_ratio"] > thresholds["truncation_tail_rms_ratio_min"]
        ) or ("ref_text_density_high" in flags):
            ref_truncation_suspect = True
            flags.append("ref_truncation_suspect")
    if tgt_stats.get("load_ok") and tgt_asr_score is not None and tgt_asr_score < thresholds["tgt_asr_min"]:
        if (
            tgt_stats["trailing_silence_sec"] < thresholds["truncation_tail_silence_max"]
            and tgt_stats["tail_rms_ratio"] > thresholds["truncation_tail_rms_ratio_min"]
        ) or ("tgt_text_density_high" in flags):
            truncation_suspect = True
            flags.append("tgt_truncation_suspect")

    semantic = evaluate_pair_semantics(row, pair_name, ref_emo, tgt_emo, cfg)
    flags.extend(semantic["semantic_flags"])

    prosody = evaluate_prosody_metrics(row, pair_name, cfg)
    flags.extend(prosody["prosody_flags"])

    speaker_sim = resolve_speaker_sim(row, pair_name)
    speaker_sim_ok = None
    if speaker_sim_min is not None:
        if speaker_sim is None:
            speaker_sim_ok = False
            flags.append("speaker_sim_missing")
        elif speaker_sim < speaker_sim_min:
            speaker_sim_ok = False
            flags.append("speaker_sim_low")
        else:
            speaker_sim_ok = True

    hard_fail_set = {
        "ref_audio_load_failed",
        "ref_asr_error",
        "ref_asr_mismatch",
        "ref_truncation_suspect",
        "ref_clipping",
        "ref_too_short",
        "ref_asr_stutter_suspect",
        "ref_asr_loop_suspect",
        "tgt_audio_load_failed",
        "tgt_dnsmos_low",
        "tgt_asr_error",
        "tgt_asr_mismatch",
        "tgt_truncation_suspect",
        "tgt_clipping",
        "tgt_too_short",
        "tgt_asr_stutter_suspect",
        "tgt_asr_loop_suspect",
        "prosody_metrics_missing",
        "prosody_metrics_error",
        "prosody_duration_ratio_missing",
        "prosody_duration_ratio_low",
        "prosody_duration_ratio_high",
        "prosody_pause_pattern_missing",
        "prosody_pause_pattern_low",
        "prosody_energy_corr_missing",
        "prosody_energy_corr_low",
        "prosody_similarity_missing",
        "prosody_similarity_low",
        "speed_direction_fail",
    }
    hard_fail_flags = [f for f in flags if f in hard_fail_set]
    soft_fail_flags = [f for f in flags if f not in hard_fail_set]
    hard_pass = len(hard_fail_flags) == 0
    qc_pass = hard_pass
    if fail_on_speaker_sim and speaker_sim_min is not None and speaker_sim_ok is not True:
        qc_pass = False
    if fail_on_semantic and semantic["semantic_ok"] is False:
        qc_pass = False

    return {
        "pair_id": row.get("pair_id"),
        "pair_type": row.get("pair_type") or pair_name,
        "reference_audio": ref_audio,
        "target_audio": tgt_audio,
        "reference_text": ref_text,
        "target_text": tgt_text,
        "ref_dnsmos_ovrl": ref_dnsmos,
        "tgt_dnsmos_ovrl": tgt_dnsmos,
        "ref_dnsmos_sig": ref_dnsmos_sig,
        "tgt_dnsmos_sig": tgt_dnsmos_sig,
        "ref_dnsmos_bak": ref_dnsmos_bak,
        "tgt_dnsmos_bak": tgt_dnsmos_bak,
        "ref_speaker_sim_wavlm": speaker_sim,
        "speaker_sim_min": speaker_sim_min,
        "speaker_sim_ok": speaker_sim_ok,
        "ref_duration_sec": ref_stats.get("duration_sec"),
        "tgt_duration_sec": tgt_stats.get("duration_sec"),
        "ref_silence_ratio": ref_stats.get("silence_ratio"),
        "tgt_silence_ratio": tgt_stats.get("silence_ratio"),
        "ref_trailing_silence_sec": ref_stats.get("trailing_silence_sec"),
        "tgt_trailing_silence_sec": tgt_stats.get("trailing_silence_sec"),
        "ref_clip_ratio": ref_stats.get("clip_ratio"),
        "tgt_clip_ratio": tgt_stats.get("clip_ratio"),
        "ref_tail_rms_ratio": ref_stats.get("tail_rms_ratio"),
        "tgt_tail_rms_ratio": tgt_stats.get("tail_rms_ratio"),
        "ref_asr_text": ref_asr_text,
        "tgt_asr_text": tgt_asr_text,
        "ref_asr_score": ref_asr_score,
        "tgt_asr_score": tgt_asr_score,
        "ref_text_units_per_sec": ref_units_per_sec,
        "tgt_text_units_per_sec": tgt_units_per_sec,
        "ref_truncation_suspect": int(ref_truncation_suspect),
        "truncation_suspect": int(truncation_suspect),
        "semantic_ok": None if semantic["semantic_ok"] is None else int(semantic["semantic_ok"]),
        "semantic_flags": semantic["semantic_flags"],
        "prosody_ok": None if prosody["prosody_ok"] is None else int(prosody["prosody_ok"]),
        "prosody_flags": prosody["prosody_flags"],
        "prosody_duration_ratio_tgt_over_ref": prosody["duration_ratio"],
        "prosody_pause_pattern_jaccard": prosody["pause_pattern_jaccard"],
        "prosody_energy_contour_corr": prosody["energy_contour_corr"],
        "prosody_f0_contour_corr": prosody["f0_contour_corr"],
        "prosody_similarity_proxy": prosody["prosody_similarity_proxy"],
        "speed_direction_pass": prosody["speed_direction_pass"],
        "same_text": None if semantic["same_text"] is None else int(semantic["same_text"]),
        "same_audio": None if semantic["same_audio"] is None else int(semantic["same_audio"]),
        "ref_top1": semantic["ref_top1_label"],
        "tgt_top1": semantic["tgt_top1_label"],
        "ref_top1_label": semantic["ref_top1_label"],
        "tgt_top1_label": semantic["tgt_top1_label"],
        "ref_top1_prob": ref_top1_prob,
        "tgt_top1_prob": tgt_top1_prob,
        "ref_p_neutral": semantic["ref_p_neutral"],
        "tgt_p_neutral": semantic["tgt_p_neutral"],
        "ref_sv_label": ref_sv_label,
        "tgt_sv_label": tgt_sv_label,
        "ref_asr_repeat_run": ref_max_repeat_run,
        "tgt_asr_repeat_run": tgt_max_repeat_run,
        "ref_asr_loop_repeat": ref_max_loop_repeat,
        "tgt_asr_loop_repeat": tgt_max_loop_repeat,
        "hard_fail_flags": hard_fail_flags,
        "soft_fail_flags": soft_fail_flags,
        "hard_pass": int(hard_pass),
        "qc_pass": int(qc_pass),
        "flags": flags,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-root", required=True, help=".../pair_outputs/<lang>/<split>")
    ap.add_argument("--config", default=None)
    ap.add_argument("--pair-type", default="all", help="A/B/.../Genre_conv/I/J_fast/J_slow/all")
    ap.add_argument("--prefer-scored", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--prefer-filtered", dest="prefer_scored", action=argparse.BooleanOptionalAction, help=argparse.SUPPRESS)
    ap.add_argument("--merge-summary", action="store_true", help="Merge refreshed pair_type results into existing summary.json instead of replacing other pair types.")
    ap.add_argument("--fail-on-speaker-sim", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--fail-on-semantic", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--disable-asr", action="store_true")
    ap.add_argument("--asr-source", choices=["auto", "sensevoice", "qwen"], default="qwen")
    ap.add_argument("--asr-python", default=os.environ.get("QWEN_ASR_PYTHON", DEFAULT_QWEN_ASR_PYTHON))
    ap.add_argument("--asr-model", default=os.environ.get("QWEN_ASR_MODEL", DEFAULT_QWEN_ASR_MODEL))
    ap.add_argument("--asr-device", default=os.environ.get("QWEN_ASR_DEVICE", "cuda:0"))
    ap.add_argument("--asr-batch-size", type=int, default=32)
    ap.add_argument("--asr-chunk-size", type=int, default=8192)
    ap.add_argument("--dnsmos-min", type=float, default=2.5)
    ap.add_argument("--ref-asr-min", type=float, default=0.55)
    ap.add_argument("--tgt-asr-min", type=float, default=0.65)
    ap.add_argument("--min-duration-sec", type=float, default=0.8)
    ap.add_argument("--max-clip-ratio", type=float, default=1e-4)
    ap.add_argument("--max-silence-ratio", type=float, default=0.35)
    ap.add_argument("--max-trailing-silence-sec", type=float, default=1.2)
    ap.add_argument("--truncation-tail-silence-max", type=float, default=0.03)
    ap.add_argument("--truncation-tail-rms-ratio-min", type=float, default=0.35)
    ap.add_argument("--repeat-token-run-threshold", type=int, default=4)
    ap.add_argument("--repeat-ngram-repeat-threshold", type=int, default=3)
    ap.add_argument("--zh-max-units-per-sec", type=float, default=8.5)
    ap.add_argument("--en-max-units-per-sec", type=float, default=4.5)
    args = ap.parse_args()

    cfg = load_config(args.config)
    pair_root = Path(args.pair_root)
    pair_dir = pair_root / "pairs"
    emotion_dir = pair_root / "emotion"
    out_dir = pair_root / "quality_gate"
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "_progress.json"
    write_progress(progress_path, phase="discover_files", pair_root=str(pair_root))

    files = discover_pair_files(pair_dir, args.pair_type, args.prefer_scored)
    if not files:
        raise SystemExit(f"no pair files found under {pair_dir} for pair_type={args.pair_type}")
    log(f"[qc] pair_root={pair_root}")
    log(f"[qc] input_files={len(files)} prefer_scored={args.prefer_scored}")

    emotion_table = load_emotion_table(emotion_dir)

    all_rows = []
    unique_audio: dict[str, str] = {}
    for path in files:
        for row in iter_jsonl(path):
            all_rows.append((path, row))
            if row.get("reference_audio"):
                unique_audio.setdefault(row["reference_audio"], infer_lang(row.get("reference_text")))
            if row.get("target_audio"):
                unique_audio.setdefault(row["target_audio"], infer_lang(row.get("target_text")))
    log(f"[qc] loaded_rows={len(all_rows)} unique_audio={len(unique_audio)}")
    write_progress(progress_path, phase="audio_stats", row_count=len(all_rows), unique_audio=len(unique_audio))

    audio_stats = {}
    for idx, audio in enumerate(unique_audio, start=1):
        audio_stats[audio] = load_audio_stats(audio)
        if idx % 5000 == 0 or idx == len(unique_audio):
            log(f"[qc] audio_stats {idx}/{len(unique_audio)}")

    asr_map: dict[str, dict] = {}
    asr_backend_error = None
    emotion_asr_hits = 0
    qwen_fallback_count = 0
    if not args.disable_asr:
        write_progress(progress_path, phase="asr_prepare")
        manifest = []
        for audio, lang in unique_audio.items():
            cached = None
            if args.asr_source in ("auto", "sensevoice"):
                cached = asr_from_emotion(audio, emotion_table)
            if cached is not None:
                asr_map[audio] = cached
                emotion_asr_hits += 1
                continue
            if args.asr_source != "sensevoice":
                manifest.append({"uid": audio, "audio": audio, "language": lang})
        qwen_fallback_count = len(manifest)
        log(
            f"[qc] asr_source={args.asr_source} "
            f"emotion_cache_hits={emotion_asr_hits} qwen_fallback={qwen_fallback_count}"
        )
        write_progress(
            progress_path,
            phase="asr",
            emotion_asr_hits=emotion_asr_hits,
            qwen_fallback=qwen_fallback_count,
        )
        if manifest:
            helper_path = Path(__file__).resolve().parent / "qwen_asr_batch.py"
            fallback_map, asr_backend_error = run_qwen_asr(
                manifest,
                helper_path=helper_path,
                python_bin=args.asr_python,
                model_path=args.asr_model,
                device=args.asr_device,
                batch_size=args.asr_batch_size,
                chunk_size=args.asr_chunk_size,
            )
            asr_map.update(fallback_map)
    else:
        log("[qc] ASR disabled")

    thresholds = {
        "dnsmos_min": args.dnsmos_min,
        "ref_asr_min": args.ref_asr_min,
        "tgt_asr_min": args.tgt_asr_min,
        "min_duration_sec": args.min_duration_sec,
        "max_clip_ratio": args.max_clip_ratio,
        "max_silence_ratio": args.max_silence_ratio,
        "max_trailing_silence_sec": args.max_trailing_silence_sec,
        "truncation_tail_silence_max": args.truncation_tail_silence_max,
        "truncation_tail_rms_ratio_min": args.truncation_tail_rms_ratio_min,
        "repeat_token_run_threshold": float(args.repeat_token_run_threshold),
        "repeat_ngram_repeat_threshold": float(args.repeat_ngram_repeat_threshold),
        "zh_max_units_per_sec": args.zh_max_units_per_sec,
        "en_max_units_per_sec": args.en_max_units_per_sec,
    }

    summary = {
        "pair_root": str(pair_root),
        "pair_type": args.pair_type,
        "prefer_scored": args.prefer_scored,
        "fail_on_speaker_sim": int(args.fail_on_speaker_sim),
        "fail_on_semantic": int(args.fail_on_semantic),
        "asr_enabled": int(not args.disable_asr),
        "asr_source": args.asr_source,
        "asr_backend_error": asr_backend_error,
        "emotion_asr_hits": emotion_asr_hits,
        "qwen_fallback_count": qwen_fallback_count,
        "thresholds": thresholds,
        "files": {},
    }
    md_lines = [
        "# Pair QC Summary",
        "",
        f"- pair_root: `{pair_root}`",
        f"- prefer_scored: `{args.prefer_scored}`",
        f"- fail_on_speaker_sim: `{args.fail_on_speaker_sim}`",
        f"- fail_on_semantic: `{args.fail_on_semantic}`",
        f"- asr_source: `{args.asr_source}`",
        f"- asr_backend_error: `{asr_backend_error}`",
        f"- emotion_asr_hits: `{emotion_asr_hits}`",
        f"- qwen_fallback_count: `{qwen_fallback_count}`",
        "",
    ]

    for pair_file in files:
        pair_name = canonical_pair_output_name(pair_file)
        log(f"[qc] scoring pair_type={pair_name}")
        write_progress(progress_path, phase="score_pairs", pair_name=pair_name)
        speaker_sim_min = speaker_sim_min_for_pair_type(cfg, pair_name)
        rows = [r for p, r in all_rows if p == pair_file]
        qc_metrics_rows = [
            flag_and_score_pair(
                r,
                cfg,
                emotion_table,
                audio_stats,
                asr_map,
                thresholds,
                pair_name=pair_name,
                speaker_sim_min=speaker_sim_min,
                fail_on_speaker_sim=args.fail_on_speaker_sim,
                fail_on_semantic=args.fail_on_semantic,
            )
            for r in rows
        ]
        qc_rows = []
        for orig_row, qc_row in zip(rows, qc_metrics_rows):
            merged = dict(orig_row)
            merged.update(qc_row)
            qc_rows.append(merged)

        cleanup_old_qc_outputs(out_dir, pair_name)
        csv_dir = out_dir / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / f"{pair_name}_qc.csv"
        jsonl_path = out_dir / f"{pair_name}_qc.jsonl"
        if qc_rows:
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                fieldnames = [
                    "pair_id",
                    "pair_type",
                    "reference_audio",
                    "target_audio",
                    "reference_text",
                    "target_text",
                    "ref_dnsmos_ovrl",
                    "tgt_dnsmos_ovrl",
                    "ref_dnsmos_sig",
                    "tgt_dnsmos_sig",
                    "ref_dnsmos_bak",
                    "tgt_dnsmos_bak",
                    "ref_speaker_sim_wavlm",
                    "speaker_sim_min",
                    "speaker_sim_ok",
                    "ref_duration_sec",
                    "tgt_duration_sec",
                    "ref_silence_ratio",
                    "tgt_silence_ratio",
                    "ref_trailing_silence_sec",
                    "tgt_trailing_silence_sec",
                    "ref_clip_ratio",
                    "tgt_clip_ratio",
                    "ref_tail_rms_ratio",
                    "tgt_tail_rms_ratio",
                    "ref_asr_text",
                    "tgt_asr_text",
                    "ref_asr_score",
                    "tgt_asr_score",
                    "ref_text_units_per_sec",
                    "tgt_text_units_per_sec",
                    "ref_truncation_suspect",
                    "truncation_suspect",
                    "semantic_ok",
                    "semantic_flags",
                    "prosody_ok",
                    "prosody_flags",
                    "prosody_duration_ratio_tgt_over_ref",
                    "prosody_pause_pattern_jaccard",
                    "prosody_energy_contour_corr",
                    "prosody_f0_contour_corr",
                    "prosody_similarity_proxy",
                    "speed_direction_pass",
                    "same_text",
                    "same_audio",
                    "ref_top1",
                    "tgt_top1",
                    "ref_top1_label",
                    "tgt_top1_label",
                    "ref_top1_prob",
                    "tgt_top1_prob",
                    "ref_p_neutral",
                    "tgt_p_neutral",
                    "ref_sv_label",
                    "tgt_sv_label",
                    "ref_asr_repeat_run",
                    "tgt_asr_repeat_run",
                    "ref_asr_loop_repeat",
                    "tgt_asr_loop_repeat",
                    "hard_pass",
                    "qc_pass",
                    "hard_fail_flags",
                    "soft_fail_flags",
                    "flags",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in qc_metrics_rows:
                    flat = dict(row)
                    flat["semantic_flags"] = "|".join(row["semantic_flags"])
                    flat["prosody_flags"] = "|".join(row["prosody_flags"])
                    flat["hard_fail_flags"] = "|".join(row["hard_fail_flags"])
                    flat["soft_fail_flags"] = "|".join(row["soft_fail_flags"])
                    flat["flags"] = "|".join(row["flags"])
                    writer.writerow(flat)
            write_jsonl(jsonl_path, qc_rows)

        flag_counter = Counter()
        for row in qc_rows:
            flag_counter.update(row["flags"])
        n = len(qc_rows)
        hard_passed = sum(int(row["hard_pass"]) for row in qc_rows)
        qc_passed = sum(int(row["qc_pass"]) for row in qc_rows)
        trunc = sum(int(row["truncation_suspect"]) for row in qc_rows)
        ref_trunc = sum(int(row["ref_truncation_suspect"]) for row in qc_rows)
        semantic_fail = sum(1 for row in qc_rows if row["semantic_ok"] == 0)
        prosody_fail = sum(1 for row in qc_rows if row["prosody_ok"] == 0)
        file_summary = {
            "count": n,
            "hard_pass": hard_passed,
            "hard_pass_rate": (hard_passed / n) if n else None,
            "qc_pass": qc_passed,
            "qc_pass_rate": (qc_passed / n) if n else None,
            "speaker_sim_min": speaker_sim_min,
            "avg_speaker_sim": mean_or_none([row["ref_speaker_sim_wavlm"] for row in qc_rows]),
            "truncation_suspect": trunc,
            "ref_truncation_suspect": ref_trunc,
            "semantic_fail": semantic_fail,
            "prosody_fail": prosody_fail,
            "avg_ref_asr_score": mean_or_none([row["ref_asr_score"] for row in qc_rows]),
            "avg_tgt_asr_score": mean_or_none([row["tgt_asr_score"] for row in qc_rows]),
            "avg_prosody_duration_ratio": mean_or_none([row["prosody_duration_ratio_tgt_over_ref"] for row in qc_rows]),
            "avg_prosody_pause_pattern_jaccard": mean_or_none([row["prosody_pause_pattern_jaccard"] for row in qc_rows]),
            "avg_prosody_energy_contour_corr": mean_or_none([row["prosody_energy_contour_corr"] for row in qc_rows]),
            "avg_prosody_similarity_proxy": mean_or_none([row["prosody_similarity_proxy"] for row in qc_rows]),
            "flag_counts": dict(flag_counter),
            "csv": str(csv_path),
            "jsonl": str(jsonl_path),
        }
        summary["files"][pair_name] = file_summary
        md_lines += [
            f"## {pair_name}",
            f"- count: `{n}`",
            f"- hard_pass: `{hard_passed}/{n}`",
            f"- qc_pass: `{qc_passed}/{n}`",
            f"- speaker_sim_min: `{speaker_sim_min}`",
            f"- avg_speaker_sim: `{file_summary['avg_speaker_sim']}`",
            f"- semantic_fail: `{semantic_fail}`",
            f"- prosody_fail: `{prosody_fail}`",
            f"- avg_prosody_duration_ratio: `{file_summary['avg_prosody_duration_ratio']}`",
            f"- avg_prosody_pause_pattern_jaccard: `{file_summary['avg_prosody_pause_pattern_jaccard']}`",
            f"- ref_truncation_suspect: `{ref_trunc}`",
            f"- truncation_suspect: `{trunc}`",
            f"- top_flags: `{dict(flag_counter.most_common(10))}`",
            "",
        ]
        log(f"[qc] done pair_type={pair_name} count={n} qc_pass={qc_passed}/{n}")

    summary_path = out_dir / "summary.json"
    if args.merge_summary and args.pair_type != "all" and summary_path.exists():
        existing_summary = read_json(summary_path)
        existing_files = existing_summary.setdefault("files", {})
        existing_files.update(summary["files"])
        existing_summary.update(
            {
                "pair_root": str(pair_root),
                "pair_type": existing_summary.get("pair_type", "all"),
                "prefer_scored": args.prefer_scored,
                "fail_on_speaker_sim": int(args.fail_on_speaker_sim),
                "fail_on_semantic": int(args.fail_on_semantic),
                "asr_enabled": int(not args.disable_asr),
                "asr_source": args.asr_source,
                "asr_backend_error": asr_backend_error,
                "emotion_asr_hits": emotion_asr_hits,
                "qwen_fallback_count": qwen_fallback_count,
                "thresholds": thresholds,
                "partial_refresh": {
                    "pair_types": [canonical_pair_output_name(p) for p in files],
                    "reason": "merge_summary",
                    "emotion_asr_hits": emotion_asr_hits,
                    "qwen_fallback_count": qwen_fallback_count,
                },
            }
        )
        summary = existing_summary
        md_text = build_summary_md(summary)
    else:
        md_text = "\n".join(md_lines)
    write_json(summary_path, summary)
    (out_dir / "summary.md").write_text(md_text, encoding="utf-8")
    write_progress(progress_path, phase="done", summary_json=str(out_dir / "summary.json"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
