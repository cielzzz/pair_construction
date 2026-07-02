#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from _common import iter_jsonl, load_config, text_units, write_json, write_jsonl

try:
    import librosa
except Exception:  # pragma: no cover
    librosa = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add prosody metrics to pair jsonl.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--config", default=None)
    parser.add_argument("--mode", choices=("speed_edit", "prosody_transfer", "generic"), default="generic")
    return parser.parse_args()


def corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 3 or b.size < 3:
        return None
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return None
    aa = a[mask]
    bb = b[mask]
    if float(np.std(aa)) < 1e-8 or float(np.std(bb)) < 1e-8:
        return None
    return float(np.corrcoef(aa, bb)[0, 1])


def resize(values: np.ndarray, points: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return np.full(points, np.nan, dtype=np.float32)
    if values.size == points:
        return values
    xs = np.linspace(0.0, 1.0, num=values.size)
    xt = np.linspace(0.0, 1.0, num=points)
    finite = np.isfinite(values)
    if int(finite.sum()) < 2:
        return np.full(points, np.nan, dtype=np.float32)
    return np.interp(xt, xs[finite], values[finite]).astype(np.float32)


def pause_regions(silent: np.ndarray, hop_sec: float, min_pause_sec: float) -> list[tuple[float, float]]:
    regions: list[tuple[float, float]] = []
    start = None
    for idx, value in enumerate(silent):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            end = idx
            dur = (end - start) * hop_sec
            if dur >= min_pause_sec:
                regions.append((start * hop_sec, end * hop_sec))
            start = None
    if start is not None:
        end = len(silent)
        dur = (end - start) * hop_sec
        if dur >= min_pause_sec:
            regions.append((start * hop_sec, end * hop_sec))
    return regions


def load_mono(path: str) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    return wav.astype(np.float32), int(sr)


def audio_features(path: str, cfg: dict[str, Any]) -> dict[str, Any]:
    metrics_cfg = cfg.get("prosody_metrics", {})
    frame_ms = float(metrics_cfg.get("frame_ms", 20))
    hop_ms = float(metrics_cfg.get("hop_ms", 10))
    min_pause_sec = float(metrics_cfg.get("min_pause_sec", 0.15))
    contour_points = int(metrics_cfg.get("contour_points", 200))

    wav, sr = load_mono(path)
    if wav.size == 0:
        raise ValueError(f"empty audio: {path}")
    duration_sec = float(wav.size / sr)
    frame = max(1, int(sr * frame_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))

    rms = []
    for start in range(0, max(1, wav.size - frame + 1), hop):
        chunk = wav[start : start + frame]
        if chunk.size < frame:
            chunk = np.pad(chunk, (0, frame - chunk.size))
        rms.append(float(np.sqrt(np.mean(np.square(chunk)) + 1e-12)))
    rms_arr = np.asarray(rms, dtype=np.float32)
    amp95 = float(np.percentile(np.abs(wav), 95))
    silence_th = max(1e-4, amp95 * 0.02)
    silent = rms_arr <= silence_th
    pauses = pause_regions(silent, hop / sr, min_pause_sec)
    energy_contour = resize(np.log(rms_arr + 1e-6), contour_points)
    pause_contour = resize(silent.astype(np.float32), contour_points)

    f0_contour = np.full(contour_points, np.nan, dtype=np.float32)
    voiced_ratio = None
    if librosa is not None:
        y, lsr = librosa.load(path, sr=16000, mono=True)
        f0 = librosa.yin(
            y,
            fmin=float(metrics_cfg.get("f0_min_hz", 50)),
            fmax=float(metrics_cfg.get("f0_max_hz", 500)),
            sr=lsr,
            frame_length=1024,
            hop_length=256,
        )
        frame_energy = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
        energy_th = max(1e-5, float(np.percentile(frame_energy, 30)))
        voiced = (frame_energy > energy_th) & np.isfinite(f0)
        f0 = np.where(voiced, f0, np.nan)
        voiced_ratio = float(np.mean(voiced)) if voiced.size else None
        if np.isfinite(f0).sum() >= 2:
            lf0 = np.log(f0)
            f0_contour = resize(lf0, contour_points)

    pause_durs = [end - start for start, end in pauses]
    return {
        "path": path,
        "load_ok": True,
        "sample_rate": sr,
        "duration_sec": duration_sec,
        "rms_mean": float(np.mean(rms_arr)),
        "rms_std": float(np.std(rms_arr)),
        "silence_ratio": float(np.mean(silent)),
        "pause_count": len(pauses),
        "pause_total_sec": float(sum(pause_durs)),
        "pause_mean_sec": float(np.mean(pause_durs)) if pause_durs else 0.0,
        "pause_max_sec": float(max(pause_durs)) if pause_durs else 0.0,
        "voiced_ratio": voiced_ratio,
        "_energy_contour": energy_contour,
        "_pause_contour": pause_contour,
        "_f0_contour": f0_contour,
    }


def strip_private(feat: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in feat.items() if not k.startswith("_")}


def ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or abs(den) < 1e-12:
        return None
    value = float(num / den)
    if math.isfinite(value):
        return value
    return None


def speed_pass(tag: str | None, duration_ratio: float | None, cfg: dict[str, Any]) -> int | None:
    if tag is None or duration_ratio is None:
        return None
    rule = cfg.get("speed_edit", {}).get("expected_duration_ratio", {}).get(tag)
    if not rule:
        return None
    if "max" in rule:
        return int(duration_ratio <= float(rule["max"]))
    if "min" in rule:
        return int(duration_ratio >= float(rule["min"]))
    return None


def add_metrics(row: dict[str, Any], cfg: dict[str, Any], mode: str) -> dict[str, Any]:
    ref_audio = row.get("reference_audio") or row.get("prosody_ref_audio")
    tgt_audio = row.get("target_audio")
    if not ref_audio or not tgt_audio:
        row["prosody_metrics_error"] = "missing reference_audio/prosody_ref_audio or target_audio"
        return row
    try:
        ref = audio_features(ref_audio, cfg)
        tgt = audio_features(tgt_audio, cfg)
        units = text_units(row.get("target_text") or row.get("reference_text"))
        duration_ratio = ratio(tgt["duration_sec"], ref["duration_sec"])
        ref_units_per_sec = ratio(float(units), ref["duration_sec"])
        tgt_units_per_sec = ratio(float(units), tgt["duration_sec"])
        pause_jaccard_den = np.maximum(ref["_pause_contour"], tgt["_pause_contour"]).sum()
        pause_jaccard = None
        if pause_jaccard_den > 0:
            pause_jaccard = float(np.minimum(ref["_pause_contour"], tgt["_pause_contour"]).sum() / pause_jaccard_den)

        metrics = {
            "mode": mode,
            "text_units": units,
            "reference": strip_private(ref),
            "target": strip_private(tgt),
            "duration_ratio_tgt_over_ref": duration_ratio,
            "units_per_sec_ref": ref_units_per_sec,
            "units_per_sec_tgt": tgt_units_per_sec,
            "units_per_sec_ratio_tgt_over_ref": ratio(tgt_units_per_sec, ref_units_per_sec),
            "pause_count_delta": int(tgt["pause_count"] - ref["pause_count"]),
            "pause_ratio_delta": float(tgt["silence_ratio"] - ref["silence_ratio"]),
            "pause_pattern_jaccard": pause_jaccard,
            "energy_contour_corr": corr(ref["_energy_contour"], tgt["_energy_contour"]),
            "f0_contour_corr": corr(ref["_f0_contour"], tgt["_f0_contour"]),
        }
        tag = row.get("source_edit_tag") or row.get("source_edit")
        metrics["speed_direction_pass"] = speed_pass(tag, duration_ratio, cfg) if mode == "speed_edit" else None
        sim_parts = [
            value
            for value in (
                metrics.get("pause_pattern_jaccard"),
                metrics.get("energy_contour_corr"),
                metrics.get("f0_contour_corr"),
            )
            if value is not None
        ]
        metrics["prosody_similarity_proxy"] = float(np.mean(sim_parts)) if sim_parts else None
        row["prosody_metrics"] = metrics
    except Exception as exc:
        row["prosody_metrics_error"] = f"{type(exc).__name__}: {exc}"
    return row


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_rows = [add_metrics(row, cfg, args.mode) for row in iter_jsonl(args.input_jsonl)]
    n = write_jsonl(args.output_jsonl, out_rows)

    ok_rows = [r for r in out_rows if "prosody_metrics" in r]
    summary = {
        "input": str(Path(args.input_jsonl).resolve()),
        "output": str(Path(args.output_jsonl).resolve()),
        "rows": n,
        "metric_ok": len(ok_rows),
        "metric_error": n - len(ok_rows),
        "duration_ratio_mean": mean([r["prosody_metrics"]["duration_ratio_tgt_over_ref"] for r in ok_rows if r["prosody_metrics"].get("duration_ratio_tgt_over_ref") is not None]),
        "pause_pattern_jaccard_mean": mean([r["prosody_metrics"]["pause_pattern_jaccard"] for r in ok_rows if r["prosody_metrics"].get("pause_pattern_jaccard") is not None]),
        "energy_contour_corr_mean": mean([r["prosody_metrics"]["energy_contour_corr"] for r in ok_rows if r["prosody_metrics"].get("energy_contour_corr") is not None]),
        "f0_contour_corr_mean": mean([r["prosody_metrics"]["f0_contour_corr"] for r in ok_rows if r["prosody_metrics"].get("f0_contour_corr") is not None]),
        "speed_direction_pass_rate": mean([r["prosody_metrics"]["speed_direction_pass"] for r in ok_rows if r["prosody_metrics"].get("speed_direction_pass") is not None]),
    }
    if args.summary_json:
        write_json(args.summary_json, summary)
    print(f"added prosody metrics rows={n} ok={len(ok_rows)} -> {Path(args.output_jsonl).resolve()}")
    if args.summary_json:
        print(f"summary -> {Path(args.summary_json).resolve()}")


if __name__ == "__main__":
    main()
