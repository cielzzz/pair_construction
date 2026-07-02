[English](README.md) | [简体中文](README_zh.md)

# pair_construction

> A **read-only upstream, produce pairs** pipeline that turns voice-cloning + voice-editing outputs into 12 mainline categories plus the I/J prosody categories of `(reference_audio, instruction, target_audio)` training pairs for instruction-conditioned TTS.

This project does **not** train models, does **not** generate any new audio, and does **not** modify upstream artifacts. It only reads, normalizes, joins, filters and emits jsonl.

---

## 1. Purpose

Instruction-conditioned TTS (e.g. "speak more calmly", "convert to news-broadcast style", "keep the same emotion but read this new text") needs paired training data where the same speaker timbre appears in two takes that differ along a controlled axis.

Upstream provides two such axes for free:

- **`vcdata_construction/`** — MOSS-TTS clones a reference speaker into many alternative renderings (different text, same voice).
- **`vc_edit/`** — StepFun EditX applies style / emotion edits to a reference audio (same voice, different delivery).

This repo joins those two streams, attaches emotion scores, and emits 15 supported pair output names covering: neutralization, re-energization, style conversion, cross-emotion conversion, identity controls, cross-speaker negatives, prosody transfer, and speed control — all conditioned on natural-language instructions.

---

## 2. Pair categories

| Type | reference | target | edit source | Purpose |
|---|---|---|---|---|
| **A** | original_audio | ref_audio (same voice, new text) | vcdata only | Voice-clone baseline; only text varies |
| **B** | edited_audio (neutralized) | ref_audio (expressive) | `style_radio` (zh) / `style_chat` (en) | Plain → expressive |
| **C** | ref_audio (expressive) | edited_audio (neutralized) | same as B | Expressive → plain |
| **C_mixed** | original_audio (real human, expressive) | edited_audio (synthetic, neutralized) | same as B | Cross real/synth, expressive → plain |
| **D** | ref_audio (expressive) | edited_audio (expressive) | non-neutralizing tags | Same emotion, different text |
| **D_st** | ref_audio (expressive) | edited_audio (expressive) | same as D, same text | Same emotion, same text (EditX side-output subset) |
| **D_cross_emo** | ref_audio (vcdata, emotion X) | original_audio (real human, emotion Y) | n/a (vcdata only) | Cross-emotion conversion — same speaker (clone vs real) but different emotion category |
| **Genre** | ref_audio | edited_audio (genre-converted) | zh: `[news, chat]` / en: `[news, radio]` | Genre / delivery-style conversion, same text |
| **Genre_conv** | edited_audio (genre A) | edited_audio (genre B) | paired EditX genre tags | Genre A -> genre B conversion, same speaker and same text |
| **H1** | original_audio | ref_audio | A subset with high emotion-cosine | Zero-change control ("keep as-is") |
| **H2** | ref_audio (already neutral) | edited_audio (more neutral) | neutralizer tag | Neutral -> more neutral control |
| **H3** | any A reference | random cross-row reference | A with cross-row shuffle | Cross-speaker negative |
| **I** | prosody_ref_audio (also aliased as reference_audio) | SeedVC output using timbre_ref_audio voice | SeedVC prosody transfer | Preserve the prosody reference's speech rate, pauses, rhythm, emphasis, and intonation while using the timbre reference speaker |
| **J_fast** | original/reference audio | Step-Audio-EditX speed edit | `speed_faster` / `speed_more_faster` | Same speaker and same text, faster speech |
| **J_slow** | original/reference audio | Step-Audio-EditX speed edit | `speed_slower` / `speed_more_slower` | Same speaker and same text, slower speech |

The **neutralizing edit tag differs by language**: `style_radio` is the strongest neutralizer for Chinese; `style_chat` is the strongest for English. Genre's whitelist is the complement set, so B/C/H2 (which need a neutralizer) and Genre (which should not start from already-neutral) never share a tag.

`I`, `J_fast`, and `J_slow` are the I/J prosody categories. They are generated after the regular pair/QC stage by `scripts/run_run03_prosody_speed_pairs.sh`, share the same scoring/QC infrastructure, and are included by default in the Qizhi runner via `RUN_IJ_ON_QZ=1`.

---

## 3. Pipeline

```
                    upstream (read-only)
   ┌──────────────────────────┐    ┌─────────────────────┐
   │  vcdata_construction     │    │  vc_edit (StepFun)  │
   │  MOSS-TTS clones         │    │  EditX style/emo    │
   │  ref_audio per row       │    │  edits ref_audio    │
   └────────────┬─────────────┘    └──────────┬──────────┘
                │                              │
                ▼                              ▼
       01 build_vcdata_base           02 build_editx_base
                │                              │
                └──────────┬───────────────────┘
                           │
                03 join_editx_with_vcdata
                           │
                           ▼
                  joined_editx.jsonl
                           │
                04 run_emotion_eval (emotion2vec + SenseVoice + DNSMOS)
                           │
                           ▼
         emotion/per_file_dual.csv + per_pair.csv
                           │
   ┌────────┬────────┬─────┴────┬─────────┬─────────┬────────┐
   ▼        ▼        ▼          ▼         ▼         ▼        ▼
  05 A   06 B   07 C / 07b C_mixed   07c D / 07d D_st / 07f D_cross_emo
  08 H1     09 H2     10 H3     07e Genre / Genre_conv
                           │
                           ▼
            11b add_wavlm_sim   (WavLM-L + ECAPA-TDNN re-score for supported pair types, writes pairs/scored/*.jsonl)
                           │
                           ▼
            12 filter_dnsmos_bak (anti-electronic-tone, optional apply)
                           │
                           ▼
                13 qc_pairs (final quality gate, writes quality_gate/)
                           │
                           ▼
                  outputs/<split>/pairs/*.jsonl
```

The I/J branch runs after the regular pair stage when enabled:

```
source manifest
   ├── J_fast / J_slow: prepare Step-Audio-EditX speed jobs -> collect speed pairs -> add prosody metrics
   └── I: prepare SeedVC prosody-transfer jobs -> run SeedVC -> collect I pairs -> add prosody metrics
        -> add generated-audio metrics -> add WavLM speaker similarity -> qc_pairs
```

---

## 4. Environments

Four conda envs are used (paths in chain scripts and `04_run_emotion_eval.sh`):

| Env | Used by | Why separate |
|---|---|---|
| `moss-tts` | vcdata `stage1_generate.py` | MOSS-TTS dependencies (PyTorch + custom audio decoder) |
| `step_audio_editx` | `run_step_editx.py` (vLLM) | StepFun EditX dependencies (vLLM, custom kernels) |
| `emotion` | `01–10`, `04` main, `12` | pair_construction core + emotion2vec + SenseVoice |
| `moss_ttsd_sglang` | `11b_add_wavlm_sim.py`, DNSMOS | WavLM-L, ECAPA-TDNN, ONNX runtime |

You only need the upstream envs (`moss-tts`, `step_audio_editx`) if you are also running vcdata / EditX. If you start `from_vcdata`, only `emotion` + `moss_ttsd_sglang` are required.

---

## 5. Quickstart

### 5.0 Unified entrypoint

If you want `vcdata/edit` on QZ and `pair` locally, use:

```bash
cd /inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction

# submit vcdata + edit to QZ
bash run_pipeline_interface.sh submit-qz

# after upstream finishes, construct pairs locally from a run_root
bash run_pipeline_interface.sh pair-local zh zh_slim_0001 configs/default.yaml cuda:0 \
  /path/to/run_root
```

### 5.1 Smoke test (single GPU, single split)

End-to-end on 200 sentences locally:

```bash
SPLIT=smoke_zh200_0603
PC_ROOT=/path/to/pair_construction
EMOPY=/path/to/envs/emotion/bin/python
WAVLMPY=/path/to/envs/moss_ttsd_sglang/bin/python

cd $PC_ROOT
$EMOPY scripts/01_build_vcdata_base.py --split $SPLIT
$EMOPY scripts/02_build_editx_base.py --split $SPLIT
$EMOPY scripts/03_join_editx_with_vcdata.py --split $SPLIT
bash   scripts/04_run_emotion_eval.sh $SPLIT cuda:0
for s in 05_construct_A 06_construct_B 07_construct_C 07b_construct_C_mixed \
         07c_construct_D 07d_construct_D_st 07e_construct_genre 07f_construct_D_cross_emo \
         07e_construct_genre_conv \
         08_construct_H1 09_construct_H2 10_construct_H3; do
    $EMOPY scripts/${s}.py --split $SPLIT
done
$WAVLMPY scripts/11b_add_wavlm_sim.py --split $SPLIT
$EMOPY  scripts/12_filter_dnsmos_bak.py --split $SPLIT
# final QC is run automatically by scripts/run_pairs_local.sh;
# if you drive steps manually, run qc_pairs.py once over the completed split root.
```

For English, switch `--config configs/default_en.yaml` (or `export PAIR_CONFIG=configs/default_en.yaml`).

### 5.2 Full data (distributed via OpenI batch + local sweep)

```bash
# 1) Submit MOSS-TTS stage1 to OpenI batch cluster
sh runs/run_zh_full.sh                  # writes to vcdata_construction/outputs/.../zh/split_*
# (wait for batch to finish, monitored on OpenI)

# 2) Submit EditX stage to OpenI
sh runs/run_zh_from_vcdata.sh            # default RUN_MODE=submit
# (wait for batch)

# 3) Locally walk all splits and run emotion + pair construction
RUN_MODE=after_editx sh runs/run_zh_from_vcdata.sh
```

For English, replace `zh` with `en`.

### 5.3 Verified smoke results

Last verified end-to-end run on 200 sentences:

| Lang | A | B | C | C_mixed | D | D_st | D_cross_emo | Genre | H1 | H2 | H3 | Total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| zh | 200 | 43 | 43 | 81 | 41 | 26 | 19 | 400 | 85 | 138 | 400 | 1476 |
| en | 200 | 45 | 45 | 74 | 25 | 34 | 51 | 400 | 40 | 115 | 200 | 1429 |

Numbers above are raw pair counts before the final QC gate. Final accepted / rejected counts are written to `outputs/<split>/quality_gate/summary.json`.

### 5.4 Verified I/J validation results

Latest completed I/J validation run:

```text
outputs/mtd_pass_nonmulti_primary_le_0p3_zh0004_en0004_ij_qz_20260621_run01
```

QC pass counts:

| Pair type | zh_slim_0004 | en_slim_0004 | Note |
|---|---:|---:|---|
| I | 10,000 -> 6,866 | 10,000 -> 4,794 | SeedVC generated all requested rows; no missing result/audio in the validation run |
| J_fast | 7,375 -> 2,032 | 7,482 -> 1,910 | Low speed-direction pass rate, about 31%; main failure is `speed_direction_fail` |
| J_slow | 7,375 -> 6,358 | 7,482 -> 5,780 | Stable; speed-direction pass rate is about 95% zh / 94% en |

Operational conclusion: `I` and `J_slow` are usable after QC; `J_fast` is integrated but lower-yield and should be tuned if high retained volume is required.

---

## 6. Configuration

Two yaml configs, identical structure, language-specific thresholds:

- `configs/default.yaml` — Chinese (default)
- `configs/default_en.yaml` — English (looser thresholds because English emotion distributions are flatter on the eval models)

Select via `export PAIR_CONFIG=configs/default_en.yaml` or `--config configs/default_en.yaml`.

Key knob categories:

| Group | Effect |
|---|---|
| `paths.*` | upstream `vcdata_root`, `emotion_eval_root`, downstream `outputs_root` |
| `editx.edit_tags` | which EditX tags to consume (default: 3 tags per language) |
| `bc.edit_whitelist` | which tag's edited audio is the "neutral" side of B/C |
| `genre.edit_tag_whitelist` | which tags drive Genre (must exclude neutralizer tag) |
| `bc / d / d_st / c_mixed / d_cross_emo / genre.speaker_sim_min_wavlm` | per-type WavLM-L speaker-similarity floor |
| `h1.cosine_min` | "expression unchanged" threshold for the H1 control |
| `h2.mode` | default `ref_to_edited`; legacy `self` / `neighbor` only for backward-compatible ablations |
| `h2.ref_neutral_min / h2.p_neutral_min / h2.target_more_neutral_margin` | H2's ref/tgt neutral floors and the minimum "target more neutral" margin |
| `dnsmos_bak_filter.apply` | toggle the optional anti-electronic-tone post-filter |

## 6.1 I/J prosody routes

This repo carries two additional pair-generation routes under `scripts/` and `configs/prosody_routes.yaml`. They are additive to the regular A-H/Genre pipeline and are now part of the Qizhi batch flow by default.

- `J_fast` / `J_slow`: `01_prepare_step_speed_jobs.py -> run_step_editx_local.py -> 02_collect_step_speed_pairs.py -> 03_add_prosody_metrics.py`, with launchers `run_speed_pipeline.sh` and `run_zh_en_slim500_speed.sh`.
- `I` (SeedVC prosody transfer): `07_prepare_prosody_no_timbre_seedvc_jobs.py -> 08_run_seedvc_jobs.py -> 09_collect_seedvc_prosody_no_timbre_pairs.py -> 03_add_prosody_metrics.py`, with launcher `run_seedvc_prosody_no_timbre_slim500.sh`.
- `run_run03_prosody_speed_pairs.sh` writes `I.jsonl`, `J_fast.jsonl`, and `J_slow.jsonl` into the normal split `pairs/` layout, then optionally refreshes generated-audio metrics, WavLM speaker similarity, and QC.
- When QC is enabled, `04c_add_pair_audio_metrics.py` first evaluates missing generated-audio emotion/SenseVoice/DNSMOS metrics and merges them into `emotion/per_file_dual.csv`.
- In the Qizhi batch submitter/runner, `RUN_IJ_ON_QZ` defaults to `1`, so I/J are included by default after the regular pair/QC stage. Set `RUN_IJ_ON_QZ=0` only when you intentionally want to skip them.
- Local-only supporting docs live under `docs/prosody_routes.md` and `docs/prosody_no_timbre_model_routes.md`; those `docs/` files are not part of the GitHub sync.
- The old DSP-based prosody-no-timbre prototype is intentionally not synced into this repo.

## 6.2 B1 local edit (M1 spike)

B1 local content editing is currently an M1 spike only. It does not batch-produce data and it does not change the main 12 + I/J pair pipeline.

- New tooling:
  - `scripts/13_extract_alignment.py`: ASR/alignment wrapper for `paraformer`, `qwen`, `whisperx`, with explicit fallback timing when no backend returns word timestamps.
  - `scripts/14_select_span.py`: selects edit spans from alignment JSON via `anchor_word`, `filler_words`, `regex`, or `manual_span`.
  - `scripts/15_run_ming_edit_poc.py`: reproducible Ming-UniAudio-Edit PoC runner that keeps Ming upstream code unchanged and applies only runtime input/device/save workarounds.
- PoC outputs live under `pairs/poc/`, including `pairs/poc/B1_poc.jsonl`.
- Ming env: `/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/ming-uniaudio-edit`, created after sourcing `/inspire/ssd/project/embodied-multimodality/public/xyzhang/activate_conda.sh`.
- M1 result: the alignment/span plumbing runs, but current Step-Audio-EditX has no native span API. Ming-UniAudio passes the B1.1 `明天 -> 后天` automated text/speaker checks after installing `flash_attn` and using `pairs/poc/ming_model_flash`; continue with small-sample validation before batch production.
- Decision docs:
  - [docs/B1_alignment_spike_result.md](/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/docs/B1_alignment_spike_result.md)
  - [docs/B1_editx_api_check.md](/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/docs/B1_editx_api_check.md)
  - [docs/B1_ming_uniaudio_check.md](/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/docs/B1_ming_uniaudio_check.md)
  - [docs/B1_route_comparison.md](/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/docs/B1_route_comparison.md)

---

## 7. Output schema

Every pair jsonl line:

```json
{
  "pair_id": "split_0000:B:000123",
  "pair_type": "A | B | C | C-mixed | D | D-st | D_cross_emo | Genre | Genre_conv | H1 | H2 | H3 | I | J_fast | J_slow",
  "reference_audio": "/path/to/ref.wav",
  "reference_text": "...",
  "target_audio": "/path/to/tgt.wav",
  "target_text": "...",
  "instruction": "Speak with more expression",
  "source_edit_tag": "style_radio | style_chat | style_news | null",
  "ref_emotion": {
    "top1_label": "neutral", "top1_prob": 0.99, "P_neutral": 0.99,
    "sv_label": "neutral", "dnsmos_ovrl": 3.87
  },
  "tgt_emotion": { "...": "same shape" },
  "ref_vs_tgt_speaker_sim_wavlm": 0.79,
  "ref_dnsmos_bak": 4.21,
  "tgt_dnsmos_bak": 3.95,
  "meta": { "split": "...", "source_row_index": 123 }
}
```

QC outputs use side-prefixed metrics such as `ref_top1`/`tgt_top1`
(short aliases), `ref_top1_label`, `tgt_top1_label`, `ref_p_neutral`,
`tgt_p_neutral`, `ref_sv_label`, `tgt_sv_label`, `ref_dnsmos_ovrl`,
`tgt_dnsmos_ovrl`, `ref_dnsmos_sig`, `tgt_dnsmos_sig`, `ref_dnsmos_bak`,
and `tgt_dnsmos_bak`.

I/J rows add prosody-specific fields:

- `I`: `prosody_ref_audio`, `prosody_ref_text`, `timbre_ref_audio`, `timbre_ref_text`, and `timbre_ref_vs_tgt_speaker_sim_wavlm`. `reference_audio` / `reference_text` are aliases for the prosody reference to stay compatible with shared QC code.
- `J_fast` / `J_slow`: `prosody_metrics.duration_ratio_tgt_over_ref`, `prosody_metrics.speed_direction_pass`, and the same `ref_vs_tgt_speaker_sim_wavlm` field as the regular pair types.

---

## 8. Project structure

```
pair_construction/
├── README.md / README_zh.md
├── configs/
│   ├── default.yaml           # zh
│   └── default_en.yaml        # en
├── scripts/
│   ├── 01_build_vcdata_base.py
│   ├── 02_build_editx_base.py
│   ├── 03_join_editx_with_vcdata.py
│   ├── 04_run_emotion_eval.sh
│   ├── 04b_add_dnsmos.py
│   ├── 05_construct_A.py
│   ├── 06_construct_B.py
│   ├── 07_construct_C.py
│   ├── 07b_construct_C_mixed.py
│   ├── 07c_construct_D.py
│   ├── 07d_construct_D_st.py
│   ├── 07e_construct_genre.py
│   ├── 07e_construct_genre_conv.py
│   ├── 07f_construct_D_cross_emo.py
│   ├── 07_prepare_prosody_no_timbre_seedvc_jobs.py
│   ├── 08_run_seedvc_jobs.py
│   ├── 09_collect_seedvc_prosody_no_timbre_pairs.py
│   ├── 01_prepare_step_speed_jobs.py
│   ├── 02_collect_step_speed_pairs.py
│   ├── 03_add_prosody_metrics.py
│   ├── 08_construct_H1.py
│   ├── 09_construct_H2.py
│   ├── 10_construct_H3.py
│   ├── 11b_add_wavlm_sim.py
│   ├── 12_filter_dnsmos_bak.py
│   ├── _utils.py / _emotion_lookup.py / _dnsmos.py
│   └── qc_pairs.py / quality_check.py / compare_edit_modes.py
├── runs/                       # production wrappers
│   ├── run_zh_full.sh          # zh full: stage1 + editx + pair
│   ├── run_zh_from_vcdata.sh   # zh skip stage1, start from editx
│   ├── run_en_full.sh
│   └── run_en_from_vcdata.sh
├── run_all.sh                  # single-split orchestrator (stages 3–5)
├── run_e2e.sh                  # full or from_vcdata mode
└── submit_editx_batch_h200.sh  # OpenI batch submitter for EditX
```

---

## 9. What this project does NOT do

- Does not retrain any model (MOSS-TTS, EditX, emotion2vec, SenseVoice, WavLM-L)
- Does not regenerate any upstream audio
- Does not mutate `vcdata_construction/outputs/` or `vc_edit/.../paired_report.jsonl`
- Does not re-rank speaker_similarity from raw embeddings (vcdata's argmax is already the chosen ref)

It only does: **read → normalize → join → score → filter → emit pairs**.
