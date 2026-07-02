#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np


PROJECTS_ROOT = Path(__file__).resolve().parents[2]
VCDATA_ROOT = PROJECTS_ROOT / "vcdata_construction"
if str(VCDATA_ROOT) not in sys.path:
    sys.path.insert(0, str(VCDATA_ROOT))

from speaker_similarity import SpeakerSimilarity  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect probable multi-speaker audio and optionally filter manifest rows."
    )
    parser.add_argument("--audio", help="Analyze a single audio file.")
    parser.add_argument("--input-jsonl", help="Analyze rows from a JSONL manifest.")
    parser.add_argument("--kept-jsonl", help="Write kept rows here when --input-jsonl is used.")
    parser.add_argument("--flagged-jsonl", help="Write flagged rows here when --input-jsonl is used.")
    parser.add_argument("--report-jsonl", help="Write per-row analysis here.")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N rows.")
    parser.add_argument("--audio-field", default="local_path")
    parser.add_argument("--id-field", default="object_name")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda:0")
    parser.add_argument("--top-db", type=float, default=28.0)
    parser.add_argument("--min-silence-sec", type=float, default=0.20)
    parser.add_argument("--min-speech-sec", type=float, default=0.35)
    parser.add_argument("--analysis-window-sec", type=float, default=1.60)
    parser.add_argument("--analysis-hop-sec", type=float, default=1.00)
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--cluster-threshold", type=float, default=0.72)
    parser.add_argument("--outlier-threshold", type=float, default=0.66)
    parser.add_argument("--secondary-cluster-min-sec", type=float, default=0.80)
    parser.add_argument("--likely-median-threshold", type=float, default=0.50)
    parser.add_argument("--likely-max-threshold", type=float, default=0.72)
    parser.add_argument("--possible-median-threshold", type=float, default=0.62)
    parser.add_argument("--possible-max-threshold", type=float, default=0.82)
    parser.add_argument(
        "--drop-labels",
        default="likely_multi_speaker",
        help="Comma-separated decision labels to treat as filtered when --input-jsonl is used.",
    )
    return parser.parse_args()


@dataclass
class AudioWindow:
    start_sec: float
    end_sec: float
    duration_sec: float
    audio: np.ndarray


def resolve_device(raw: str) -> str:
    if raw != "auto":
        return raw
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def load_audio(path: str) -> tuple[np.ndarray, int]:
    audio, sr = librosa.load(path, sr=None, mono=True)
    if audio.size == 0:
        raise ValueError(f"empty audio: {path}")
    return audio.astype(np.float32, copy=False), sr


def merge_intervals(
    intervals: np.ndarray, min_gap_samples: int, min_speech_samples: int
) -> list[tuple[int, int]]:
    if len(intervals) == 0:
        return []
    merged: list[tuple[int, int]] = []
    cur_start, cur_end = int(intervals[0][0]), int(intervals[0][1])
    for start, end in intervals[1:]:
        start_i = int(start)
        end_i = int(end)
        if start_i - cur_end <= min_gap_samples:
            cur_end = max(cur_end, end_i)
            continue
        if cur_end - cur_start >= min_speech_samples:
            merged.append((cur_start, cur_end))
        cur_start, cur_end = start_i, end_i
    if cur_end - cur_start >= min_speech_samples:
        merged.append((cur_start, cur_end))
    return merged


def detect_speech_intervals(
    audio: np.ndarray,
    sr: int,
    top_db: float,
    min_silence_sec: float,
    min_speech_sec: float,
) -> list[tuple[int, int]]:
    intervals = librosa.effects.split(
        audio,
        top_db=top_db,
        frame_length=max(2048, int(sr * 0.064)),
        hop_length=max(512, int(sr * 0.016)),
    )
    return merge_intervals(
        intervals=intervals,
        min_gap_samples=int(sr * min_silence_sec),
        min_speech_samples=int(sr * min_speech_sec),
    )


def make_windows(
    audio: np.ndarray,
    sr: int,
    intervals: list[tuple[int, int]],
    window_sec: float,
    hop_sec: float,
    max_windows: int,
) -> list[AudioWindow]:
    window_samples = max(int(sr * window_sec), 1)
    hop_samples = max(int(sr * hop_sec), 1)
    windows: list[AudioWindow] = []

    for start, end in intervals:
        seg = audio[start:end]
        seg_len = len(seg)
        if seg_len <= 0:
            continue

        if seg_len <= window_samples:
            center = (start + end) // 2
            left = max(0, center - window_samples // 2)
            right = min(len(audio), left + window_samples)
            left = max(0, right - window_samples)
            padded = np.zeros(window_samples, dtype=np.float32)
            clip = audio[left:right]
            padded[: len(clip)] = clip
            windows.append(
                AudioWindow(
                    start_sec=left / sr,
                    end_sec=right / sr,
                    duration_sec=(right - left) / sr,
                    audio=padded,
                )
            )
        else:
            max_offset = seg_len - window_samples
            offsets = list(range(0, max_offset + 1, hop_samples))
            if offsets[-1] != max_offset:
                offsets.append(max_offset)
            for offset in offsets:
                s = start + offset
                e = s + window_samples
                windows.append(
                    AudioWindow(
                        start_sec=s / sr,
                        end_sec=e / sr,
                        duration_sec=window_sec,
                        audio=audio[s:e],
                    )
                )

        if len(windows) >= max_windows:
            break

    return windows[:max_windows]


def cosine_similarity_matrix(embs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    safe = np.clip(norms, 1e-12, None)
    normed = embs / safe
    return normed @ normed.T


def pairwise_stats(sim_matrix: np.ndarray) -> dict[str, float | None]:
    n = sim_matrix.shape[0]
    if n < 2:
        return {
            "min": None,
            "p25": None,
            "median": None,
            "mean": None,
            "max": None,
        }
    vals = sim_matrix[np.triu_indices(n, k=1)]
    return {
        "min": float(vals.min()),
        "p25": float(np.quantile(vals, 0.25)),
        "median": float(np.median(vals)),
        "mean": float(vals.mean()),
        "max": float(vals.max()),
    }


def cluster_similarity(sim_matrix: np.ndarray, a: list[int], b: list[int]) -> float:
    vals = [sim_matrix[i, j] for i in a for j in b]
    return float(np.mean(vals))


def agglomerative_clusters(sim_matrix: np.ndarray, threshold: float) -> list[list[int]]:
    clusters = [[i] for i in range(sim_matrix.shape[0])]
    if len(clusters) <= 1:
        return clusters

    while True:
        best_pair: tuple[int, int] | None = None
        best_sim = -1.0
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                sim = cluster_similarity(sim_matrix, clusters[i], clusters[j])
                if sim > best_sim:
                    best_sim = sim
                    best_pair = (i, j)
        if best_pair is None or best_sim < threshold:
            break
        i, j = best_pair
        merged = clusters[i] + clusters[j]
        clusters = [
            cluster
            for idx, cluster in enumerate(clusters)
            if idx not in {i, j}
        ]
        clusters.append(merged)
    return sorted(clusters, key=len, reverse=True)


def window_cluster_durations(windows: list[AudioWindow], clusters: list[list[int]]) -> list[float]:
    durations: list[float] = []
    for cluster in clusters:
        durations.append(float(sum(windows[i].duration_sec for i in cluster)))
    return durations


def metadata_speaker_hint(record: dict) -> dict[str, object]:
    reasons: list[str] = []

    def scan_text(text: str | None, source: str) -> None:
        if not text:
            return
        lowered = text.lower()
        tokens = [
            "multiple male voices",
            "multiple female voices",
            "multiple voices",
            "man and woman",
            "woman and man",
            "two speakers",
            "two voices",
            "chorus",
            "duet",
        ]
        if any(token in lowered for token in tokens):
            reasons.append(source)

    caption = record.get("caption_result")
    if isinstance(caption, list):
        if len(caption) >= 2:
            reasons.append("caption_result:list>=2")
        for item in caption:
            if isinstance(item, dict):
                scan_text(item.get("gender"), "caption_result.gender")
                scan_text(item.get("summary"), "caption_result.summary")
    elif isinstance(caption, dict):
        scan_text(caption.get("gender"), "caption_result.gender")
        scan_text(caption.get("summary"), "caption_result.summary")

    for key in ["qwen3-omni-checkpoint-65000", "whisperd_asr_results"]:
        value = record.get(key)
        if isinstance(value, dict):
            scan_text(value.get("gender"), f"{key}.gender")
            scan_text(value.get("summary"), f"{key}.summary")
            if value.get("num_speakers") not in (None, "", 1):
                reasons.append(f"{key}.num_speakers")

    return {
        "metadata_multi_speaker_hint": bool(reasons),
        "metadata_multi_speaker_reasons": reasons,
    }


def analyze_audio(
    audio_path: str,
    encoder: SpeakerSimilarity,
    top_db: float,
    min_silence_sec: float,
    min_speech_sec: float,
    analysis_window_sec: float,
    analysis_hop_sec: float,
    max_windows: int,
    cluster_threshold: float,
    outlier_threshold: float,
    secondary_cluster_min_sec: float,
    likely_median_threshold: float,
    likely_max_threshold: float,
    possible_median_threshold: float,
    possible_max_threshold: float,
) -> dict[str, object]:
    audio, sr = load_audio(audio_path)
    intervals = detect_speech_intervals(
        audio=audio,
        sr=sr,
        top_db=top_db,
        min_silence_sec=min_silence_sec,
        min_speech_sec=min_speech_sec,
    )
    windows = make_windows(
        audio=audio,
        sr=sr,
        intervals=intervals,
        window_sec=analysis_window_sec,
        hop_sec=analysis_hop_sec,
        max_windows=max_windows,
    )
    total_speech_sec = float(sum((end - start) / sr for start, end in intervals))

    if len(windows) == 0:
        return {
            "decision": "insufficient_speech",
            "reason": "no speech windows after silence split",
            "duration_sec": len(audio) / sr,
            "speech_intervals": [],
            "speech_total_sec": total_speech_sec,
            "analysis_windows": [],
            "estimated_clusters": 0,
            "cluster_durations_sec": [],
            "pairwise_similarity": pairwise_stats(np.eye(1, dtype=np.float32)),
        }

    embs = encoder.embed_batch([w.audio for w in windows], sr=sr)
    sim_matrix = cosine_similarity_matrix(embs)
    stats = pairwise_stats(sim_matrix)
    clusters = agglomerative_clusters(sim_matrix, threshold=cluster_threshold)
    cluster_durations = window_cluster_durations(windows, clusters)

    medoid_idx = 0
    medoid_scores = sim_matrix.mean(axis=1)
    medoid_idx = int(np.argmax(medoid_scores))
    ref_sims = sim_matrix[medoid_idx].tolist()
    outlier_count = int(sum(1 for idx, sim in enumerate(ref_sims) if idx != medoid_idx and sim < outlier_threshold))

    secondary_duration = cluster_durations[1] if len(cluster_durations) > 1 else 0.0
    median_sim = stats["median"] if stats["median"] is not None else 1.0
    max_sim = stats["max"] if stats["max"] is not None else 1.0
    if len(windows) < 2:
        decision = "insufficient_windows"
        reason = "need at least 2 analysis windows"
    elif median_sim < likely_median_threshold and max_sim < likely_max_threshold:
        decision = "likely_multi_speaker"
        reason = "pairwise speaker similarity stays low across windows"
    elif (
        median_sim < possible_median_threshold
        or (len(clusters) >= 2 and secondary_duration >= secondary_cluster_min_sec and max_sim < possible_max_threshold)
        or outlier_count >= 2
    ):
        decision = "possible_multi_speaker"
        reason = "embedding consistency is weaker than a clean single-speaker clip"
    else:
        decision = "single_speaker"
        reason = "window embeddings form one stable cluster"

    return {
        "decision": decision,
        "reason": reason,
        "duration_sec": float(len(audio) / sr),
        "speech_intervals": [
            {"start_sec": round(start / sr, 3), "end_sec": round(end / sr, 3)}
            for start, end in intervals
        ],
        "speech_total_sec": total_speech_sec,
        "analysis_windows": [
            {
                "start_sec": round(window.start_sec, 3),
                "end_sec": round(window.end_sec, 3),
                "duration_sec": round(window.duration_sec, 3),
            }
            for window in windows
        ],
        "estimated_clusters": len(clusters),
        "cluster_durations_sec": [round(x, 3) for x in cluster_durations],
        "medoid_window_index": medoid_idx,
        "medoid_similarities": [round(float(x), 4) for x in ref_sims],
        "outlier_threshold": outlier_threshold,
        "outlier_count": outlier_count,
        "cluster_threshold": cluster_threshold,
        "pairwise_similarity": {
            key: (round(value, 4) if value is not None else None)
            for key, value in stats.items()
        },
        "decision_thresholds": {
            "likely_median_threshold": likely_median_threshold,
            "likely_max_threshold": likely_max_threshold,
            "possible_median_threshold": possible_median_threshold,
            "possible_max_threshold": possible_max_threshold,
        },
    }


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_manifest(args: argparse.Namespace, encoder: SpeakerSimilarity) -> int:
    input_path = Path(args.input_jsonl)
    kept_rows: list[dict] = []
    flagged_rows: list[dict] = []
    report_rows: list[dict] = []
    drop_labels = {item.strip() for item in args.drop_labels.split(",") if item.strip()}

    for idx, row in enumerate(iter_jsonl(input_path), 1):
        if args.limit and idx > args.limit:
            break
        audio_path = row[args.audio_field]
        analysis = analyze_audio(
            audio_path=audio_path,
            encoder=encoder,
            top_db=args.top_db,
            min_silence_sec=args.min_silence_sec,
            min_speech_sec=args.min_speech_sec,
            analysis_window_sec=args.analysis_window_sec,
            analysis_hop_sec=args.analysis_hop_sec,
            max_windows=args.max_windows,
            cluster_threshold=args.cluster_threshold,
            outlier_threshold=args.outlier_threshold,
            secondary_cluster_min_sec=args.secondary_cluster_min_sec,
            likely_median_threshold=args.likely_median_threshold,
            likely_max_threshold=args.likely_max_threshold,
            possible_median_threshold=args.possible_median_threshold,
            possible_max_threshold=args.possible_max_threshold,
        )
        analysis.update(metadata_speaker_hint(row))
        row_with_analysis = dict(row)
        row_with_analysis["multi_speaker_probe"] = analysis

        report_rows.append(
            {
                "row_index": idx,
                "row_id": row.get(args.id_field, f"row_{idx:06d}"),
                "audio_path": audio_path,
                **analysis,
            }
        )

        if analysis["decision"] in drop_labels or analysis["metadata_multi_speaker_hint"]:
            flagged_rows.append(row_with_analysis)
        else:
            kept_rows.append(row_with_analysis)

    if args.kept_jsonl:
        write_jsonl(Path(args.kept_jsonl), kept_rows)
    if args.flagged_jsonl:
        write_jsonl(Path(args.flagged_jsonl), flagged_rows)
    if args.report_jsonl:
        write_jsonl(Path(args.report_jsonl), report_rows)

    print(
        json.dumps(
            {
                "input_jsonl": str(input_path),
                "processed_rows": len(report_rows),
                "kept_rows": len(kept_rows),
                "flagged_rows": len(flagged_rows),
                "drop_labels": sorted(drop_labels),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main() -> int:
    args = parse_args()
    if not args.audio and not args.input_jsonl:
        raise SystemExit("either --audio or --input-jsonl is required")

    device = resolve_device(args.device)
    encoder = SpeakerSimilarity(device=device)

    if args.audio:
        result = analyze_audio(
            audio_path=args.audio,
            encoder=encoder,
            top_db=args.top_db,
            min_silence_sec=args.min_silence_sec,
            min_speech_sec=args.min_speech_sec,
            analysis_window_sec=args.analysis_window_sec,
            analysis_hop_sec=args.analysis_hop_sec,
            max_windows=args.max_windows,
            cluster_threshold=args.cluster_threshold,
            outlier_threshold=args.outlier_threshold,
            secondary_cluster_min_sec=args.secondary_cluster_min_sec,
            likely_median_threshold=args.likely_median_threshold,
            likely_max_threshold=args.likely_max_threshold,
            possible_median_threshold=args.possible_median_threshold,
            possible_max_threshold=args.possible_max_threshold,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    return run_manifest(args, encoder)


if __name__ == "__main__":
    raise SystemExit(main())
