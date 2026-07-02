#!/usr/bin/env python
"""11b: 用 WavLM-Large + ECAPA-TDNN (SeedTTSEval) 重算 ref↔tgt sim 并写入 scored jsonl

CAM++ funasr 模型对 MOSS-TTS / editx 合成的"工具特征"不太敏感，给虚高分。
WavLM-L 是 SeedTTSEval 标准、对细微音色变化更敏感，更贴近人耳判断。

本步骤只补充 ref_vs_tgt_speaker_sim_wavlm 分数；最终阈值和 gate 在 qc_pairs.py 中执行。
要求 moss_ttsd_sglang env（emotion env 缺 transformers + WavLM）。
"""
import argparse, json, sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import load_config, iter_jsonl, pair_path, scored_pair_path, cache_path
from _speaker_sim_cache import CachedSpeakerSimilarity
VCDATA_CODE_ROOT = os.environ.get(
    "VCDATA_CODE_ROOT",
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/code/vcdata_construction",
)
sys.path.insert(0, VCDATA_CODE_ROOT)

PAIR_TYPES = (
    "A",
    "B",
    "C",
    "C_mixed",
    "D",
    "D_st",
    "D_cross_emo",
    "Genre",
    "Genre_conv",
    "H1",
    "H2",
    "H3",
    "I",
    "J_fast",
    "J_slow",
)


def write_jsonl_atomic(path: Path, rows: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    n = 0
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    tmp_path.replace(path)
    return n


def row_key(row: dict, idx: int) -> str:
    return str(row.get("pair_id") or idx)


def wavlm_field_for_pair_type(pair_type: str) -> str:
    if pair_type == "I":
        return "timbre_ref_vs_tgt_speaker_sim_wavlm"
    return "ref_vs_tgt_speaker_sim_wavlm"


def row_has_wavlm_sim(row: dict, pair_type: str) -> bool:
    value = row.get(wavlm_field_for_pair_type(pair_type))
    return value not in ("", None)


def row_wavlm_sim(row: dict, pair_type: str) -> float | None:
    value = row.get(wavlm_field_for_pair_type(pair_type))
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_rows_for_pair_type(cfg: dict, split: str, pair_type: str) -> tuple[list[dict], Path, Path]:
    raw_path = pair_path(cfg, split, f"{pair_type}.jsonl")
    scored_path = scored_pair_path(cfg, split, f"{pair_type}.jsonl")
    raw_rows = list(iter_jsonl(raw_path))
    if not scored_path.exists():
        return raw_rows, raw_path, scored_path

    scored_rows = list(iter_jsonl(scored_path))
    if len(scored_rows) == len(raw_rows):
        print(f"  {pair_type}: resume source=scored rows={len(scored_rows)}")
        return scored_rows, scored_path, scored_path

    rows = [dict(row) for row in raw_rows]
    raw_index = {row_key(row, idx): idx for idx, row in enumerate(raw_rows)}
    merged = 0
    for idx, scored in enumerate(scored_rows):
        key = row_key(scored, idx)
        raw_idx = raw_index.get(key)
        if raw_idx is None:
            continue
        rows[raw_idx].update(scored)
        merged += 1
    print(
        f"  {pair_type}: resume source=raw+partial_scored "
        f"raw={len(raw_rows)} scored={len(scored_rows)} merged={merged}"
    )
    return rows, raw_path, scored_path


def flush_progress(scored_path: Path, rows: list[dict], cached_ss: CachedSpeakerSimilarity, pair_type: str, i: int) -> None:
    write_jsonl_atomic(scored_path, rows)
    cached_ss.save()
    stats = cached_ss.stats()
    print(
        f"  {pair_type}: flushed at {i}/{len(rows)} "
        f"cache_hits={stats['cache_hits']} cache_misses={stats['cache_misses']}",
        flush=True,
    )


def add_speaker_sim_for_row(row, pair_type, cached_ss):
    if pair_type == "I":
        timbre_audio = row.get("timbre_ref_audio")
        target_audio = row.get("target_audio")
        if timbre_audio and target_audio:
            row["timbre_ref_vs_tgt_speaker_sim_wavlm"] = cached_ss.compute_similarity_files(timbre_audio, target_audio)
        ref_audio = row.get("reference_audio") or row.get("prosody_ref_audio")
        if ref_audio and target_audio:
            row["ref_vs_tgt_speaker_sim_wavlm"] = cached_ss.compute_similarity_files(ref_audio, target_audio)
        return row.get("timbre_ref_vs_tgt_speaker_sim_wavlm")

    row["ref_vs_tgt_speaker_sim_wavlm"] = cached_ss.compute_similarity_files(row["reference_audio"], row["target_audio"])
    return row["ref_vs_tgt_speaker_sim_wavlm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pair-type", default="all", help="all or comma-separated pair types")
    ap.add_argument("--flush-every", type=int, default=int(os.environ.get("WAVLM_FLUSH_EVERY", "500")))
    args = ap.parse_args()
    cfg = load_config(args.config)

    from speaker_similarity import SpeakerSimilarity
    MODELS = os.environ.get("VCDATA_MODELS_DIR", f"{VCDATA_CODE_ROOT}/models")
    ss = SpeakerSimilarity(
        device=args.device,
        checkpoint=f"{MODELS}/wavlm_large_finetune.pth",
        seed_tts_eval_root=f"{MODELS}/seed-tts-eval",
        wavlm_dir=f"{MODELS}/wavlm-large",
    )
    cached_ss = CachedSpeakerSimilarity(ss, cache_path(cfg, args.split, "wavlm_embeddings.pkl"))
    print("[11b] WavLM-Large + ECAPA-TDNN loaded")

    if args.pair_type == "all":
        pair_types = PAIR_TYPES
    else:
        requested = tuple(x.strip() for x in args.pair_type.split(",") if x.strip())
        unknown = [x for x in requested if x not in PAIR_TYPES]
        if unknown:
            raise SystemExit(f"unknown pair_type(s): {unknown}; supported={PAIR_TYPES}")
        pair_types = requested

    summary = []
    for pt in pair_types:
        raw_path = pair_path(cfg, args.split, f"{pt}.jsonl")
        if not raw_path.exists():
            continue
        rows, source_path, scored_path = load_rows_for_pair_type(cfg, args.split, pt)
        sim_values = []
        skipped = processed = failed = 0
        for i, r in enumerate(rows, 1):
            existing_sim = row_wavlm_sim(r, pt)
            if existing_sim is not None:
                sim_values.append(existing_sim)
                skipped += 1
                if i % 20 == 0:
                    print(f"  {pt} [{i}/{len(rows)}] skipped={skipped}")
                continue
            try:
                sim = add_speaker_sim_for_row(r, pt, cached_ss)
                processed += 1
            except Exception as ex:
                print(f"  [warn] {pt}#{i}: {ex}")
                sim = None
                failed += 1
            if sim is not None:
                sim_values.append(float(sim))
            if i % 20 == 0:
                print(f"  {pt} [{i}/{len(rows)}] processed={processed} skipped={skipped} failed={failed}")
            if args.flush_every > 0 and i % args.flush_every == 0:
                flush_progress(scored_path, rows, cached_ss, pt, i)
        flush_progress(scored_path, rows, cached_ss, pt, len(rows))
        avg_sim = sum(sim_values) / len(sim_values) if sim_values else None
        summary.append((pt, len(rows), avg_sim, processed, skipped, failed, source_path, scored_path))

    cached_ss.save()
    stats = cached_ss.stats()

    print(f"\n=== {args.split} WavLM-L 统计汇总 ===")
    print(f"{'type':<10} {'orig':<6} {'avg_sim':<10} {'new':<6} {'skip':<6} {'fail':<6} {'scored':<24}")
    for pt, n, avg_sim, processed, skipped, failed, _source, scored in summary:
        avg = f"{avg_sim:.4f}" if avg_sim is not None else "—"
        print(f"{pt:<10} {n:<6} {avg:<10} {processed:<6} {skipped:<6} {failed:<6} {scored.name:<24}")
    print(f"[11b] embedding cache hits={stats['cache_hits']} misses={stats['cache_misses']}")


if __name__ == "__main__":
    main()
