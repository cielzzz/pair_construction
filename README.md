# pair_construction

一个**只读上游、自产 pair**的流水线，给 TTS 大模型的"指令生成"场景产训练对（参考音频 + instruction → 新音频）。

上游：
- `vcdata_construction/` — MOSS-TTS 克隆音色合成 ref_audio（不动）
- `vc_edit/` + 各 split 下 `stepaudio_<edit_tag>_split_XXXX_all_qzrun/` — StepFun EditX 编辑 ref_audio（不动）

下游产物：
```
outputs/<split>/pairs/
├── A.jsonl          只换文本，表达基本不变（vcdata 全量）
├── B_clean.jsonl    平淡 reference → 高表现 target
├── C_clean.jsonl    高表现 reference → 中性/平淡 target
├── H1.jsonl         零变化对照（表达几乎不变的精筛子集）
├── H2.jsonl         已满足指令对照（已中性，再叫"更中性"）
└── H3.jsonl         跨 speaker 负样本
```

---

## 1. 口径定义（**项目唯一权威**）

| 编号 | 类型 | reference | target | instruction 示例 | 数据源 |
|---|---|---|---|---|---|
| **A** | 只换文本，表达基本不变 | 真实 original_audio | vcdata ref_audio (同音色不同文本) | "保持参考音色，只换文本"（后标注，可选） | vcdata 全量 |
| **B-clean** | 平淡 → 高表现 | edited_audio (中性化) | ref_audio (高表现) | "说得更有感染力 / 更有起伏 / 更像旁白" | joined_editx |
| **C-clean** | 高表现 → 平淡 | ref_audio (高表现) | edited_audio (中性化) | "平静一点 / 不要这么夸张 / 去掉情绪起伏" | joined_editx |
| **H1** | 零变化对照 | original_audio | ref_audio | "保持原样 / 不要改变表达方式" | A 中 cos≥0.97 子集 |
| **H2** | 已满足指令对照 | 已中性 edited_audio | 同 / 同属性近邻 | "更中性 / 去掉情绪起伏" | joined_editx 高置信中性 |
| **H3** | 跨 speaker 负样本 | A 中任一 ref | 跨行 random ref | "（负样本，不应学习）" | A 跨行配对 |

**关键澄清：**
- A 等同于 vcdata 全量；不再拆 A0/A 两层。
- vcdata 阶段 16 候选 argmax 选出的 ref_audio **不一定都是高表现**——所以 B/C 类用 `ref_neutral_max` 过滤掉本身就平淡的 ref。
- H1 是 A 的子集（emotion cosine 极高），不另起一类零碎子标签。
- H2 当前只覆盖 neutral 方向（editx 只验过中性化），未来扩 happy/angry 时再加。

---

## 2. 数据流

```
vcdata_construction/.../split_XXXX/                 vc_edit (stepfun-editx)
  ├── manifest_shard{0..15}.jsonl   ┐                │
  └── merged.stepaudio_input.all.jsonl ─► 01 build_vcdata_base.py
                                          │            │
                                          ▼            ▼
                                  vcdata_base.jsonl   stepaudio_style_radio_split_XXXX_all_qzrun/paired_report.jsonl
                                          │            │
                                          │   02 build_editx_base.py
                                          │            │
                                          │            ▼
                                          │   editx_base.jsonl
                                          │            │
                                          └──► 03 join_editx_with_vcdata.py
                                                       │
                                                       ▼
                                              joined_editx.jsonl
                                                       │
                              04 run_emotion_eval.sh (调 emotion_eval；对 original/ref/edited 全跑)
                                                       │
                                                       ▼
                                              emotion/per_file_dual.csv  + per_pair.csv
                                                       │
        ┌────────────────────────┬──────────────────────┼─────────────────────┬─────────────┬─────────────┐
        ▼                        ▼                      ▼                     ▼             ▼             ▼
05 construct_A.py     06 B_clean.py            07 C_clean.py          08 H1.py        09 H2.py     10 H3.py
        │                        │                      │                     │             │             │
        ▼                        ▼                      ▼                     ▼             ▼             ▼
   A.jsonl                B_clean.jsonl          C_clean.jsonl          H1.jsonl       H2.jsonl      H3.jsonl
```

---

## 3. 一键执行

```bash
bash run_all.sh split_0000             # 全跑 (含 emotion eval)
bash run_all.sh split_0000 --reuse-qzrun  # 复用 split_0000 已有 emotion 结果
bash run_all.sh split_0000 --skip-emotion # 假设 per_file_dual.csv 已就位
```

或分步：
```bash
SPLIT=split_0000
python scripts/01_build_vcdata_base.py --split $SPLIT
python scripts/02_build_editx_base.py  --split $SPLIT
python scripts/03_join_editx_with_vcdata.py --split $SPLIT
bash   scripts/04_run_emotion_eval.sh $SPLIT
python scripts/05_construct_A.py        --split $SPLIT
python scripts/06_construct_B_clean.py  --split $SPLIT
python scripts/07_construct_C_clean.py  --split $SPLIT
python scripts/08_construct_H1.py       --split $SPLIT
python scripts/09_construct_H2.py       --split $SPLIT
python scripts/10_construct_H3.py       --split $SPLIT
```

---

## 4. 输出 schema（所有 pair jsonl 通用）

```json
{
  "pair_id": "split_0000:A:000123",
  "pair_type": "A | B-clean | C-clean | H1 | H2 | H3",
  "reference_audio": "...",
  "reference_text": "...",
  "target_audio": "...",
  "target_text": "...",
  "instruction": "...",
  "source_edit": "style_radio | null",
  "speaker_similarity": 0.91,
  "ref_emotion": {"top1_label": "...", "top1_prob": 0.7, "P_neutral": 0.05, "sv_label": "..."},
  "tgt_emotion": {"top1_label": "...", "top1_prob": 0.7, "P_neutral": 0.05, "sv_label": "..."},
  "meta": {"split": "split_0000", "source_row_index": 123}
}
```

---

## 5. 配置

所有阈值集中在 `configs/default.yaml`，改 yaml 不需改代码。关键项：
- `a.sim_min` —— A 类 speaker_similarity 门槛（默认 0.80，可选）
- `bc_clean.edit_whitelist` —— 默认 `[style_radio]`（emotion_eval 验证最优）
- `bc_clean.edited_neutral_min` —— edited 中性化下限（0.70）
- `bc_clean.ref_neutral_max` —— ref "高表现" 上限（0.50）
- `h1.cosine_min` —— H1 表达几乎不变阈值（0.97）
- `h2.p_neutral_min` —— H2 reference 必须高置信中性（0.90）

---

## 6. 不做什么

- ❌ 不重算 vcdata 的 best_similarity（直接读 manifest）
- ❌ 不调 stepfun-editx 模型（直接读 paired_report.jsonl）
- ❌ 不重新训练 emotion2vec / SenseVoice（调 `emotion_eval/scripts/`）
- ❌ 不修改任何上游脚本

仅做：**读 → 标准化 → 标注 → 按规则过滤 → 输出 pair**。
