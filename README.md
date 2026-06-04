[English](README.md) | [简体中文](README_zh.md)

# pair_construction

> A **read-only upstream, produce pairs** pipeline that turns voice-cloning + voice-editing outputs into 11 categories of `(reference_audio, instruction, target_audio)` training pairs for instruction-conditioned TTS.

This project does **not** train models, does **not** generate any new audio, and does **not** modify upstream artifacts. It only reads, normalizes, joins, filters and emits jsonl.

---

## 1. Purpose

Instruction-conditioned TTS (e.g. "speak more calmly", "convert to news-broadcast style", "keep the same emotion but read this new text") needs paired training data where the same speaker timbre appears in two takes that differ along a controlled axis.

Upstream provides two such axes for free:

- **`vcdata_construction/`** — MOSS-TTS clones a reference speaker into many alternative renderings (different text, same voice).
- **`vc_edit/`** — StepFun EditX applies style / emotion edits to a reference audio (same voice, different delivery).

This repo joins those two streams, attaches emotion scores, and emits 11 categories of pairs covering: neutralization, re-energization, style conversion, cross-emotion conversion, identity controls, and cross-speaker negatives — all conditioned on natural-language instructions.

---

## 2. Pair categories (11 types)

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
| **H1** | original_audio | ref_audio | A subset with high emotion-cosine | Zero-change control ("keep as-is") |
| **H2** | edited_audio (neutralized) | self or neutral neighbor | neutralizer tag | Already-satisfied control ("be more neutral") |
| **H3** | any A reference | random cross-row reference | A with cross-row shuffle | Cross-speaker negative |

The **neutralizing edit tag differs by language**: `style_radio` is the strongest neutralizer for Chinese; `style_chat` is the strongest for English. Genre's whitelist is the complement set, so B/C/H2 (which need a neutralizer) and Genre (which should not start from already-neutral) never share a tag.

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
  08 H1     09 H2     10 H3     07e Genre
                           │
                           ▼
            11b add_wavlm_sim   (WavLM-L + ECAPA-TDNN re-score, emits *_filtered.jsonl)
                           │
                           ▼
            12 filter_dnsmos_bak (anti-electronic-tone, optional apply)
                           │
                           ▼
                  outputs/<split>/pairs/*.jsonl
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
         08_construct_H1 09_construct_H2 10_construct_H3; do
    $EMOPY scripts/${s}.py --split $SPLIT
done
$WAVLMPY scripts/11b_add_wavlm_sim.py --split $SPLIT
$EMOPY  scripts/12_filter_dnsmos_bak.py --split $SPLIT
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

Numbers above are pre-`_filtered`. WavLM-L speaker-sim filtered counts: see `outputs/<split>/pairs/*_filtered.jsonl`.

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
| `h2.mode` | `self` (any neutral edited as both sides) or `neighbor` (paired with another neutral) |
| `dnsmos_bak_filter.apply` | toggle the optional anti-electronic-tone post-filter |

### 6.1 How "neutral" is decided (P_neutral semantics)

There is **no single global threshold** for "neutral". Emotion is judged in two layers:

1. **Hard condition** `top1_label == "neutral"` — emotion2vec's nine-class winner is `neutral`.
2. **Soft condition** `P_neutral` — the actual neutral-class probability (continuous, 0–1).

Each pair type uses different lower / upper bounds, biased by language:

| Knob | zh | en | Meaning |
|---|---|---|---|
| `bc.edited_neutral_min` | **0.7** | **0.3** | B/C neutral side must reach ≥ this |
| `bc.ref_neutral_max` | 0.95 | 0.95 | B/C expressive side must stay ≤ this (else too flat) |
| `c_mixed.ref_neutral_max` | 0.95 | 0.95 | Same as above for C_mixed |
| `d.ref_neutral_max` / `tgt_neutral_max` | 0.95 | 0.95 | Both sides of D must be expressive (≤ 0.95) |
| `d_st.*neutral_max` | 0.95 | 0.95 | Same for D_st |
| **`d_cross_emo.*neutral_max`** | **0.5** | **0.5** | Cross-emotion requires both sides to be very far from neutral |
| `h2.p_neutral_min` | **0.9** | **0.5** | H2 reference must be high-confidence neutral |

**Why zh vs en differ**: emotion2vec is trained on Chinese, so on Chinese audio it yields sharp distributions (a clearly-neutralized sample easily reaches `P_neutral ≥ 0.9`). The same model on English yields flatter distributions — the strongest neutralizer (`style_chat`) tops out around `P_neutral ≈ 0.3-0.5`. English thresholds are uniformly relaxed.

**Worked example** for a row with `P_neutral = 0.024`:
- The audio is **almost certainly non-neutral** (only 2.4% of the neutral class).
- ❌ Rejected as B/C neutral side (needs ≥ 0.7 / 0.3).
- ✅ Accepted as B/C expressive side (allowed ≤ 0.95).
- ✅ Accepted as D / D_cross_emo side (D needs ≤ 0.95; D_cross_emo needs ≤ 0.5).

**Independent signal** `sv_label`: SenseVoice runs a separate emotion classifier. `bc.edited_sv_must_be_neutral` can force dual-model consensus, but it is currently `false` everywhere because en dual-model agreement rate is only ~10%.

---

## 7. Output schema

Every pair jsonl line:

```json
{
  "pair_id": "split_0000:B:000123",
  "pair_type": "B | C | C-mixed | D | D-st | D_cross_emo | Genre | H1 | H2 | H3 | A",
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
│   ├── 07f_construct_D_cross_emo.py
│   ├── 08_construct_H1.py
│   ├── 09_construct_H2.py
│   ├── 10_construct_H3.py
│   ├── 11b_add_wavlm_sim.py
│   ├── 12_filter_dnsmos_bak.py
│   ├── _utils.py / _emotion_lookup.py / _dnsmos.py
│   └── quality_check.py / compare_edit_modes.py
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

## 9. Web dashboard (Streamlit)

A 3-page Streamlit app under `app/` browses pair data, monitors growth, and compares data sources.

```
app/
├── app.py                       # entry point
├── index_builder.py             # scans outputs/<split>/pairs/*.jsonl → index.parquet
├── raw_scanner.py               # scans upstream raw split_*.jsonl → raw_source.parquet + duration_cache.parquet
├── loader.py                    # cached loaders shared by all pages
├── start.sh                     # convenience launcher (uses kxhuang tts env)
└── pages/
    ├── 1_📊_Dashboard.py        # KPI cards, source/lang/type distributions, retention rate
    ├── 2_🔍_Browser.py          # multi-dim filtering, per-pair detail with audio playback
    └── 3_📈_Source_compare.py   # cross-source comparison (each new data source = a new column)
```

### Build indices and launch (on the GPU host)

```bash
# 1) Pre-aggregate pair-side data
python app/index_builder.py

# 2) Pre-aggregate upstream raw data (gives "total hours" + duration cache)
python app/raw_scanner.py \
  --add instruction_0.1_enzh:zh:/inspire/.../kxhuang/instructtts_data/instruction_0.1_enzh/zh \
  --add instruction_0.1_enzh:en:/inspire/.../kxhuang/instructtts_data/instruction_0.1_enzh/en

# 3) Re-run index_builder so it can join duration_cache → per-split pair hours
python app/index_builder.py

# 4) Launch
bash app/start.sh        # default port 8501
```

### Local browser access (SSH tunnel)

```bash
# From your laptop
ssh -L 8501:localhost:8501 <gpu_host>
# Then open http://localhost:8501
```

### Cross-filterable tags (Browser page)

source, language, split, pair_type, is_filtered, source_edit_tag,
ref_emotion.top1, tgt_emotion.top1, sim_wavlm range, ref_dnsmos_bak range,
tgt_dnsmos_bak range, instruction keyword, ref/tgt text keyword.

---

## 10. What this project does NOT do

- Does not retrain any model (MOSS-TTS, EditX, emotion2vec, SenseVoice, WavLM-L)
- Does not regenerate any upstream audio
- Does not mutate `vcdata_construction/outputs/` or `vc_edit/.../paired_report.jsonl`
- Does not re-rank speaker_similarity from raw embeddings (vcdata's argmax is already the chosen ref)

It only does: **read → normalize → join → score → filter → emit pairs**.
