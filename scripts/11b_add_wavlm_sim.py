#!/usr/bin/env python
"""11b: 用 WavLM-Large + ECAPA-TDNN (SeedTTSEval) 重算 ref↔tgt sim 并覆盖 _filtered.jsonl

CAM++ funasr 模型对 MOSS-TTS / editx 合成的"工具特征"不太敏感，给虚高分。
WavLM-L 是 SeedTTSEval 标准、对细微音色变化更敏感，更贴近人耳判断。

cfg 字段：
  bc.speaker_sim_min_wavlm: 0.75    应用于 B + C
  d_st.speaker_sim_min_wavlm: 0.75  应用于 D_st

要求 moss_ttsd_sglang env（emotion env 缺 transformers + WavLM）。
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import load_config, iter_jsonl, write_jsonl, pair_path
sys.path.insert(0, '/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vcdata_construction')

SIM_KEY = {"B": "bc", "C": "bc", "C_mixed": "c_mixed", "D": "d", "D_st": "d_st", "D_cross_emo": "d_cross_emo", "Genre": "genre"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    cfg = load_config(args.config)

    from speaker_similarity import SpeakerSimilarity
    MODELS = "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vcdata_construction/models"
    ss = SpeakerSimilarity(
        device=args.device,
        checkpoint=f"{MODELS}/wavlm_large_finetune.pth",
        seed_tts_eval_root=f"{MODELS}/seed-tts-eval",
        wavlm_dir=f"{MODELS}/wavlm-large",
    )
    print("[11b] WavLM-Large + ECAPA-TDNN loaded")

    summary = []
    for pt, sk in SIM_KEY.items():
        pp = pair_path(cfg, args.split, f"{pt}.jsonl")
        if not pp.exists():
            continue
        sim_min = cfg.get(sk, {}).get("speaker_sim_min_wavlm", 0.0) if sk else 0.0
        rows = list(iter_jsonl(pp))
        kept = []
        for i, r in enumerate(rows, 1):
            try:
                e1 = ss.embed_from_file(r["reference_audio"])
                e2 = ss.embed_from_file(r["target_audio"])
                sim = ss.compute_similarity(e1, e2)
            except Exception as ex:
                print(f"  [warn] {pt}#{i}: {ex}")
                sim = None
            r["ref_vs_tgt_speaker_sim_wavlm"] = sim
            if sim is not None and sim >= sim_min:
                kept.append(r)
            if i % 20 == 0:
                print(f"  {pt} [{i}/{len(rows)}]  sim_min={sim_min}")
        write_jsonl(pp, rows)
        if sim_min > 0:
            fp = pp.with_name(f"{pt}_filtered.jsonl")
            write_jsonl(fp, kept)
            summary.append((pt, len(rows), len(kept), sim_min, fp))
        else:
            summary.append((pt, len(rows), len(kept), sim_min, None))

    print(f"\n=== {args.split} WavLM-L 过滤汇总 ===")
    print(f"{'type':<10} {'orig':<6} {'sim_min':<10} {'kept':<6} {'kept%':<6}")
    for pt, n, k, sm, _ in summary:
        kp = f"{100*k/n:.0f}%" if n else "—"
        print(f"{pt:<10} {n:<6} {sm:<10} {k:<6} {kp:<6}")


if __name__ == "__main__":
    main()
