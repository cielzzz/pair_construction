[English](README.md) | [简体中文](README_zh.md)

# pair_construction

> **只读上游、产出 pair** 的流水线 —— 把音色克隆 + 语音编辑两条上游输出，转成 12 个主线类别 + I/J 韵律类别的 `(参考音频, 指令, 目标音频)` 训练对，用于指令条件 TTS 训练。

本项目**不训练模型**、**不生成新音频**、**不改上游产物**。只做：读 → 标准化 → 关联 → 过滤 → 输出 jsonl。

---

## 1. 项目目的

指令条件 TTS（如 "说得更平静一点"、"换成新闻播报风格"、"用同样的情绪读这段新文本"）需要这样的训练对：同一个音色出现在两条沿某个受控轴变化的录音上。

上游免费提供了两条轴：

- **`vcdata_construction/`**：MOSS-TTS 克隆参考说话人，生成同音色 + 不同文本的合成 ref_audio。
- **`vc_edit/`**：StepFun EditX 把参考音频做风格 / 情绪编辑，得到同音色 + 不同表现的 edited_audio。

本仓库把两路打通，挂上情绪打分，支持 15 个 pair 输出名，覆盖：中性化、加情绪、风格转换、跨情绪转换、零变化对照、跨 speaker 负样本、韵律迁移和语速控制 —— 全部带自然语言指令。

---

## 2. Pair 类别定义

| 类型 | reference | target | 编辑来源 | 用途 |
|---|---|---|---|---|
| **A** | original_audio（真人） | ref_audio（同音色、新文本） | 仅 vcdata | 声纹克隆基线，只换文本 |
| **B** | edited_audio（中性化） | ref_audio（有表现力） | zh: `style_radio` / en: `style_chat` | 平淡 → 高表现 |
| **C** | ref_audio（有表现力） | edited_audio（中性化） | 同 B | 高表现 → 平淡 |
| **C_mixed** | original_audio（真人，有表现力） | edited_audio（合成，中性化） | 同 B | 跨真人/合成，高表现 → 平淡 |
| **D** | ref_audio（有表现力） | edited_audio（有表现力） | 非中性化 tag | 同情绪，不同文本 |
| **D_st** | ref_audio（有表现力） | edited_audio（有表现力） | 同 D，同文本 | 同情绪、同文本（EditX 旁路子集） |
| **D_cross_emo** | ref_audio（vcdata，情绪 X） | original_audio（真人，情绪 Y） | 不用 editx（仅 vcdata） | 跨情绪转换 —— 同 speaker（clone vs 真人）、跨情绪类别 |
| **Genre** | ref_audio | edited_audio（风格转换） | zh: `[news, chat]` / en: `[news, radio]` | 风格 / 播报方式转换，同文本 |
| **Genre_conv** | edited_audio（风格 A） | edited_audio（风格 B） | 成对的 EditX 风格 tag | 风格 A -> 风格 B 转换，同 speaker、同文本 |
| **H1** | original_audio | ref_audio | A 的 emotion cosine 极高子集 | 零变化对照（"保持原样"） |
| **H2** | ref_audio（已较中性） | edited_audio（更中性） | 中性化 tag | 中性 -> 更中性对照 |
| **H3** | 任一 A 的 ref | 跨行随机 ref | A 跨行重组 | 跨 speaker 负样本 |
| **I** | prosody_ref_audio（同时作为 reference_audio 别名） | 使用 timbre_ref_audio 音色的 SeedVC 输出 | SeedVC 韵律迁移 | 保留 prosody reference 的语速、停顿、节奏、重音和语调，同时使用 timbre reference 的说话人音色 |
| **J_fast** | 原始 / 参考音频 | Step-Audio-EditX 语速编辑输出 | `speed_faster` / `speed_more_faster` | 同 speaker、同文本，语速更快 |
| **J_slow** | 原始 / 参考音频 | Step-Audio-EditX 语速编辑输出 | `speed_slower` / `speed_more_slower` | 同 speaker、同文本，语速更慢 |

**中性化 tag 因语言而异**：中文用 `style_radio`（中性化效果最强），英文用 `style_chat`（英文模型上最强）。Genre 的白名单是补集 —— B/C/H2（需要中性化）和 Genre（不能从已中性出发）永远不会共享 tag。

`I`、`J_fast`、`J_slow` 是 I/J 韵律类别。它们由 `scripts/run_run03_prosody_speed_pairs.sh` 在常规 pair/QC 阶段之后生成，复用同一套打分和 QC 基础设施，并且在启智 runner 中通过 `RUN_IJ_ON_QZ=1` 默认开启。

---

## 3. 流水线

```
                    上游（只读）
   ┌──────────────────────────┐    ┌─────────────────────┐
   │  vcdata_construction     │    │  vc_edit (StepFun)  │
   │  MOSS-TTS 克隆音色        │    │  EditX 风格/情绪编辑 │
   │  每行产 ref_audio         │    │  改 ref_audio       │
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
            11b add_wavlm_sim   （对支持的 pair 类型做 WavLM-L + ECAPA-TDNN 重打分，写入 pairs/scored/*.jsonl）
                           │
                           ▼
            12 filter_dnsmos_bak （可选电音过滤）
                           │
                           ▼
                13 qc_pairs （最终质量门，产出 quality_gate/）
                           │
                           ▼
                  outputs/<split>/pairs/*.jsonl
```

启用时，I/J 分支在常规 pair 阶段之后继续运行：

```
source manifest
   ├── J_fast / J_slow: 准备 Step-Audio-EditX 语速任务 -> 收集语速 pair -> 加 prosody metrics
   └── I: 准备 SeedVC 韵律迁移任务 -> 跑 SeedVC -> 收集 I pair -> 加 prosody metrics
        -> 补生成音频指标 -> 加 WavLM speaker similarity -> qc_pairs
```

---

## 4. 环境

用 4 个 conda env（路径在 chain 脚本和 `04_run_emotion_eval.sh` 里）：

| Env | 谁用 | 为什么单独 |
|---|---|---|
| `moss-tts` | vcdata 的 `stage1_generate.py` | MOSS-TTS 依赖（PyTorch + 定制 audio decoder） |
| `step_audio_editx` | `run_step_editx.py`（vLLM） | StepFun EditX 依赖（vLLM + 定制 kernel） |
| `emotion` | `01–10`、`04` 主体、`12` | pair_construction 主体 + emotion2vec + SenseVoice |
| `moss_ttsd_sglang` | `11b_add_wavlm_sim.py`、DNSMOS | WavLM-L、ECAPA-TDNN、ONNX runtime |

如果你**从 `from_vcdata` 起跑**（vcdata 已存在），就只需要 `emotion` + `moss_ttsd_sglang` 两个 env。

---

## 5. 快速开始

### 5.0 统一入口

如果你想把 `vcdata/edit` 放到启智跑、`pair` 放到本地跑，直接用：

```bash
cd /inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction

# 提交 vcdata + edit 到启智
bash run_pipeline_interface.sh submit-qz

# 上游跑完后，本地从指定 run_root 构造 pair
bash run_pipeline_interface.sh pair-local zh zh_slim_0001 configs/default.yaml cuda:0 \
  /path/to/run_root
```

### 5.1 Smoke 测试（单卡单 split）

200 句端到端：

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
```

英文换 `--config configs/default_en.yaml`（或 `export PAIR_CONFIG=configs/default_en.yaml`）。

### 5.2 全量数据（启智批量 + 本地遍历）

```bash
# 1) MOSS-TTS stage1 提交到启智批量集群
sh runs/run_zh_full.sh                  # 输出在 vcdata_construction/outputs/.../zh/split_*
# （在启智上等批量跑完）

# 2) EditX 阶段提交到启智
sh runs/run_zh_from_vcdata.sh            # 默认 RUN_MODE=submit
# （等批量跑完）

# 3) 本地遍历所有 split，跑 emotion + pair 构造
RUN_MODE=after_editx sh runs/run_zh_from_vcdata.sh
```

英文把 `zh` 换成 `en`。

### 5.3 已验证的 smoke 结果

最近一次 200 句端到端：

| 语言 | A | B | C | C_mixed | D | D_st | D_cross_emo | Genre | H1 | H2 | H3 | 合计 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| zh | 200 | 43 | 43 | 81 | 41 | 26 | 19 | 400 | 85 | 138 | 400 | 1476 |
| en | 200 | 45 | 45 | 74 | 25 | 34 | 51 | 400 | 40 | 115 | 200 | 1429 |

以上是最终 QC 之前的原始 pair 行数。最终通过 / 拒绝统计见 `outputs/<split>/quality_gate/summary.json`。

### 5.4 已验证的 I/J 结果

最近一次完成的 I/J 验证 run：

```text
outputs/mtd_pass_nonmulti_primary_le_0p3_zh0004_en0004_ij_qz_20260621_run01
```

QC 通过数：

| Pair type | zh_slim_0004 | en_slim_0004 | 备注 |
|---|---:|---:|---|
| I | 10,000 -> 6,866 | 10,000 -> 4,794 | SeedVC 生成了全部请求行；该验证 run 中无缺失结果 / 音频 |
| J_fast | 7,375 -> 2,032 | 7,482 -> 1,910 | 语速方向通过率较低，约 31%；主要失败原因是 `speed_direction_fail` |
| J_slow | 7,375 -> 6,358 | 7,482 -> 5,780 | 稳定；语速方向通过率约 zh 95% / en 94% |

操作备注：`I` 和 `J_slow` 经过 QC 后可用；`J_fast` 已接入但留存率较低，如果需要高产出量，需要继续调 Step-Audio-EditX 语速 prompt / 生成策略或方向阈值。

---

## 6. 配置

两份 yaml，结构相同，阈值随语言不同：

- `configs/default.yaml` — 中文（默认）
- `configs/default_en.yaml` — 英文（阈值整体放宽，因为评估模型在英文上情绪分布更平坦）

切换：`export PAIR_CONFIG=configs/default_en.yaml` 或加 `--config configs/default_en.yaml`。

关键参数分组：

| 分组 | 作用 |
|---|---|
| `paths.*` | 上游 `vcdata_root`、`emotion_eval_root`、下游 `outputs_root` |
| `editx.edit_tags` | 消费哪几个 EditX tag（每语言默认 3 个） |
| `bc.edit_whitelist` | B/C 用哪个 tag 的 edited 当中性侧 |
| `genre.edit_tag_whitelist` | Genre 用哪些 tag（必须排除中性化 tag） |
| `bc / d / d_st / c_mixed / d_cross_emo / genre.speaker_sim_min_wavlm` | 每类 WavLM-L 说话人相似度下限 |
| `h1.cosine_min` | H1 "表达几乎不变" 阈值 |
| `h2.mode` | 默认 `ref_to_edited`；`self` / `neighbor` 仅保留给旧实验兼容 |
| `h2.ref_neutral_min / h2.p_neutral_min / h2.target_more_neutral_margin` | H2 的 ref/tgt 中性阈值，以及“target 更中性”最小 margin |
| `dnsmos_bak_filter.apply` | 是否打开可选的反电音过滤 |

## 6.1 I/J 韵律路线

本仓库额外带了两条 pair 生成路线，入口在 `scripts/` 和 `configs/prosody_routes.yaml`。它们增量接入常规 A-H/Genre 流水线，并且现在默认纳入启智批量流程。

- `J_fast` / `J_slow`：`01_prepare_step_speed_jobs.py -> run_step_editx_local.py -> 02_collect_step_speed_pairs.py -> 03_add_prosody_metrics.py`，启动脚本是 `run_speed_pipeline.sh` 和 `run_zh_en_slim500_speed.sh`。
- `I` (SeedVC prosody transfer)：`07_prepare_prosody_no_timbre_seedvc_jobs.py -> 08_run_seedvc_jobs.py -> 09_collect_seedvc_prosody_no_timbre_pairs.py -> 03_add_prosody_metrics.py`，启动脚本是 `run_seedvc_prosody_no_timbre_slim500.sh`。
- `run_run03_prosody_speed_pairs.sh` 会把 `I.jsonl`、`J_fast.jsonl`、`J_slow.jsonl` 写入标准 split 的 `pairs/` 目录，然后按需刷新生成音频指标、WavLM speaker similarity 和 QC。
- 开启 QC 时，会先运行 `04c_add_pair_audio_metrics.py`，补评新生成音频缺失的 emotion/SenseVoice/DNSMOS 指标并合并回 `emotion/per_file_dual.csv`。
- 启智批量提交/runner 中，`RUN_IJ_ON_QZ` 默认是 `1`，因此常规 pair/QC 之后会默认继续跑 I/J。只有明确要跳过 I/J 时才传 `RUN_IJ_ON_QZ=0`。
- 本地配套说明文档在 `docs/prosody_routes.md` 和 `docs/prosody_no_timbre_model_routes.md`；这些 `docs/` 文件不参与 GitHub 同步。
- 旧的 DSP 版 prosody-no-timbre 原型这次不再同步到本仓库。

---

## 7. 输出 schema

每行 pair jsonl：

```json
{
  "pair_id": "split_0000:B:000123",
  "pair_type": "A | B | C | C-mixed | D | D-st | D_cross_emo | Genre | Genre_conv | H1 | H2 | H3 | I | J_fast | J_slow",
  "reference_audio": "/path/to/ref.wav",
  "reference_text": "...",
  "target_audio": "/path/to/tgt.wav",
  "target_text": "...",
  "instruction": "用更有情绪、更有起伏的方式朗读",
  "source_edit_tag": "style_radio | style_chat | style_news | null",
  "ref_emotion": {
    "top1_label": "neutral", "top1_prob": 0.99, "P_neutral": 0.99,
    "sv_label": "neutral", "dnsmos_ovrl": 3.87
  },
  "tgt_emotion": { "...": "同上结构" },
  "ref_vs_tgt_speaker_sim_wavlm": 0.79,
  "ref_dnsmos_bak": 4.21,
  "tgt_dnsmos_bak": 3.95,
  "meta": { "split": "...", "source_row_index": 123 }
}
```

QC 输出使用带 ref/tgt 前缀的标准指标字段，例如 `ref_top1`/`tgt_top1`
（短别名）、`ref_top1_label`、`tgt_top1_label`、`ref_p_neutral`、
`tgt_p_neutral`、`ref_sv_label`、`tgt_sv_label`、`ref_dnsmos_ovrl`、
`tgt_dnsmos_ovrl`、`ref_dnsmos_sig`、`tgt_dnsmos_sig`、`ref_dnsmos_bak`、
`tgt_dnsmos_bak`。

I/J 行会额外带韵律相关字段：

- `I`：`prosody_ref_audio`、`prosody_ref_text`、`timbre_ref_audio`、`timbre_ref_text`、`timbre_ref_vs_tgt_speaker_sim_wavlm`。其中 `reference_audio` / `reference_text` 是 prosody reference 的别名，用于兼容共享 QC 代码。
- `J_fast` / `J_slow`：`prosody_metrics.duration_ratio_tgt_over_ref`、`prosody_metrics.speed_direction_pass`，以及和常规类别相同的 `ref_vs_tgt_speaker_sim_wavlm`。

---

## 8. 目录结构

```
pair_construction/
├── README.md / README_zh.md
├── configs/
│   ├── default.yaml           # 中文
│   └── default_en.yaml        # 英文
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
├── runs/                       # 生产 wrapper
│   ├── run_zh_full.sh          # 中文全跑：stage1 + editx + pair
│   ├── run_zh_from_vcdata.sh   # 跳 stage1，从 editx 起跑
│   ├── run_en_full.sh
│   └── run_en_from_vcdata.sh
├── run_all.sh                  # 单 split orchestrator（阶段 3–5）
├── run_e2e.sh                  # full 或 from_vcdata 模式
└── submit_editx_batch_h200.sh  # EditX 启智批量提交器
```

---

## 9. 本项目不做什么

- 不重训任何模型（MOSS-TTS、EditX、emotion2vec、SenseVoice、WavLM-L）
- 不重新生成任何上游音频
- 不修改 `vcdata_construction/outputs/` 或 `vc_edit/.../paired_report.jsonl`
- 不从原始 embedding 重排 speaker_similarity（vcdata 的 argmax 已是选定 ref）

只做：**读 → 标准化 → 关联 → 打分 → 过滤 → 输出 pair**。
