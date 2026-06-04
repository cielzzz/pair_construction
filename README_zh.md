[English](README.md) | [简体中文](README_zh.md)

# pair_construction

> **只读上游、产出 pair** 的流水线 —— 把音色克隆 + 语音编辑两条上游输出，转成 11 类 `(参考音频, 指令, 目标音频)` 训练对，用于指令条件 TTS 训练。

本项目**不训练模型**、**不生成新音频**、**不改上游产物**。只做：读 → 标准化 → 关联 → 过滤 → 输出 jsonl。

---

## 1. 项目目的

指令条件 TTS（如 "说得更平静一点"、"换成新闻播报风格"、"用同样的情绪读这段新文本"）需要这样的训练对：同一个音色出现在两条沿某个受控轴变化的录音上。

上游免费提供了两条轴：

- **`vcdata_construction/`**：MOSS-TTS 克隆参考说话人，生成同音色 + 不同文本的合成 ref_audio。
- **`vc_edit/`**：StepFun EditX 把参考音频做风格 / 情绪编辑，得到同音色 + 不同表现的 edited_audio。

本仓库把两路打通，挂上情绪打分，输出 11 类 pair，覆盖：中性化、加情绪、风格转换、跨情绪转换、零变化对照、跨 speaker 负样本 —— 全部带自然语言指令。

---

## 2. 11 类 pair 定义

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
| **H1** | original_audio | ref_audio | A 的 emotion cosine 极高子集 | 零变化对照（"保持原样"） |
| **H2** | edited_audio（已中性化） | 自身或中性近邻 | 中性化 tag | 已满足指令对照（"再中性一点"） |
| **H3** | 任一 A 的 ref | 跨行随机 ref | A 跨行重组 | 跨 speaker 负样本 |

**中性化 tag 因语言而异**：中文用 `style_radio`（中性化效果最强），英文用 `style_chat`（英文模型上最强）。Genre 的白名单是补集 —— B/C/H2（需要中性化）和 Genre（不能从已中性出发）永远不会共享 tag。

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
  08 H1     09 H2     10 H3     07e Genre
                           │
                           ▼
            11b add_wavlm_sim   （WavLM-L + ECAPA-TDNN 重打分，产出 *_filtered.jsonl）
                           │
                           ▼
            12 filter_dnsmos_bak （可选电音过滤）
                           │
                           ▼
                  outputs/<split>/pairs/*.jsonl
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

以上是 `_filtered` 之前的行数。WavLM-L speaker-sim 过滤后的数量见 `outputs/<split>/pairs/*_filtered.jsonl`。

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
| `h2.mode` | `self`（同一中性 edited 两端复用）或 `neighbor`（配另一中性近邻） |
| `dnsmos_bak_filter.apply` | 是否打开可选的反电音过滤 |

### 6.1 怎么判定"中性"（P_neutral 含义）

**没有一个全局阈值**说"P_neutral 大于多少就是 neutral"。判定分两层：

1. **硬条件** `top1_label == "neutral"` —— emotion2vec 九类里 neutral 是最大概率类
2. **软条件** `P_neutral` —— neutral 类的具体概率值（0–1 连续）

不同 pair 类用不同上下限，且语言不同阈值不同：

| 阈值字段 | zh | en | 含义 |
|---|---|---|---|
| `bc.edited_neutral_min` | **0.7** | **0.3** | B/C 中性侧（edited_audio）**下限**：≥ 此值才算够中性 |
| `bc.ref_neutral_max` | 0.95 | 0.95 | B/C 高表现侧（ref_audio）**上限**：≤ 此值才算真有表现力（否则太中性） |
| `c_mixed.ref_neutral_max` | 0.95 | 0.95 | C_mixed 同上 |
| `d.ref_neutral_max` / `tgt_neutral_max` | 0.95 | 0.95 | D 双侧都要有表现力（≤ 0.95） |
| `d_st.*neutral_max` | 0.95 | 0.95 | D_st 同 D |
| **`d_cross_emo.*neutral_max`** | **0.5** | **0.5** | 跨情绪要求**双侧极端非中性** |
| `h2.p_neutral_min` | **0.9** | **0.5** | H2 reference 必须**高置信中性** |

**为什么 zh/en 阈值不同**：emotion2vec 是中文母语模型。中文音频上分布锐利（一条中性化样本能轻松到 `P_neutral ≥ 0.9`），同样模型用到英文上分布扁平，最强中性化 tag（`style_chat`）也只能到 `P_neutral ≈ 0.3-0.5`。英文阈值整体**放宽**才能拿到样本。

**例子**：一条 `P_neutral = 0.024` 的音频：
- 模型几乎肯定它**不是中性**（只有 2.4% 是 neutral 类）
- ❌ 不能进 B/C 中性侧（需要 ≥ 0.7 / 0.3）
- ✅ 可以进 B/C 高表现侧（允许 ≤ 0.95）
- ✅ 可以进 D / D_cross_emo 双侧（D 需 ≤ 0.95；D_cross_emo 需 ≤ 0.5）

**独立信号 `sv_label`**：SenseVoice 是另一个独立情绪分类器。`bc.edited_sv_must_be_neutral` 可以强制两个模型共识，但目前全设为 false —— en 上两模型一致率只 ~10%，强同意会卡掉绝大多数样本。

---

## 7. 输出 schema

每行 pair jsonl：

```json
{
  "pair_id": "split_0000:B:000123",
  "pair_type": "B | C | C-mixed | D | D-st | D_cross_emo | Genre | H1 | H2 | H3 | A",
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
│   ├── 07f_construct_D_cross_emo.py
│   ├── 08_construct_H1.py
│   ├── 09_construct_H2.py
│   ├── 10_construct_H3.py
│   ├── 11b_add_wavlm_sim.py
│   ├── 12_filter_dnsmos_bak.py
│   ├── _utils.py / _emotion_lookup.py / _dnsmos.py
│   └── quality_check.py / compare_edit_modes.py
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

## 9. Web 看板（Streamlit）

`app/` 下有一个 3 页 Streamlit 应用，浏览 pair 数据、监控增长、对比数据源。

```
app/
├── app.py                       # 入口
├── index_builder.py             # 扫 outputs/<split>/pairs/*.jsonl → index.parquet
├── raw_scanner.py               # 扫上游 raw split_*.jsonl → raw_source.parquet + duration_cache.parquet
├── loader.py                    # 共享缓存加载器
├── start.sh                     # 启动脚本（用 kxhuang tts env）
└── pages/
    ├── 1_📊_Dashboard.py        # KPI 卡片 / source-lang-type 分布 / 留存率
    ├── 2_🔍_Browser.py          # 多维过滤 / 单 pair 详情 + 音频对比
    └── 3_📈_Source_compare.py   # 多数据源横向对比（每加一个新源就有一列）
```

### 建索引 + 启动（在 GPU 服务器上）

```bash
# 1) 先把 pair 侧聚合
python app/index_builder.py

# 2) 再扫上游 raw（得到"总小时数" + duration 缓存）
python app/raw_scanner.py \
  --add instruction_0.1_enzh:zh:/inspire/.../kxhuang/instructtts_data/instruction_0.1_enzh/zh \
  --add instruction_0.1_enzh:en:/inspire/.../kxhuang/instructtts_data/instruction_0.1_enzh/en

# 3) 再跑一次 index_builder，让 pair 索引拿到 duration → 算出 per-split 小时数
python app/index_builder.py

# 4) 启动
bash app/start.sh        # 默认端口 8501
```

### 本机浏览器访问（SSH 端口转发）

```bash
# 本机另开 terminal
ssh -L 8501:localhost:8501 <gpu_host>
# 浏览器打开 http://localhost:8501
```

### Browser 页支持的交叉过滤维度

source / language / split / pair_type / is_filtered / source_edit_tag /
ref_emotion.top1 / tgt_emotion.top1 / sim_wavlm 范围 / ref_dnsmos_bak 范围 /
tgt_dnsmos_bak 范围 / instruction 关键词 / ref+tgt 文本关键词。

---

## 10. 本项目不做什么

- 不重训任何模型（MOSS-TTS、EditX、emotion2vec、SenseVoice、WavLM-L）
- 不重新生成任何上游音频
- 不修改 `vcdata_construction/outputs/` 或 `vc_edit/.../paired_report.jsonl`
- 不从原始 embedding 重排 speaker_similarity（vcdata 的 argmax 已是选定 ref）

只做：**读 → 标准化 → 关联 → 打分 → 过滤 → 输出 pair**。
