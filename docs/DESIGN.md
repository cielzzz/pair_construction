# pair_construction 设计文档

> 项目唯一权威。实现细节看脚本注释，配置看 `configs/default.yaml`。

---

## 0. 术语

| 概念 | 含义 |
|---|---|
| `original_audio` | kxhuang 真实音频（vcdata 的输入） |
| `ref_audio` | MOSS-TTS 合成、从 16 候选挑出的音色相似音频（vcdata 输出） |
| `edited_audio` | 把 `ref_audio` 喂给 stepfun-editx 得到的编辑音频 |

**关键事实：**
- `ref_audio` **不一定**是高表现。16 候选取 argmax(音色相似度)，所选音频本身可能就平淡。
- 因此 B/C 类需要用 emotion 模型对 ref_audio 做二次过滤。

---

## 1. 上游事实

### 1.1 vcdata
- 位置：`vcdata_construction/outputs/instruction_0.1_enzh/zh/split_XXXX/`
- 单元：每 split 16 个 `manifest_shard{0..15}.jsonl`，已合并为 `merged.stepaudio_input.all.jsonl`
- 关键字段：`original_idx`, `original_audio_path`, `original_text`, `ref_audio_path`, `ref_text`, `best_similarity`, `flag`, `caption_result`

### 1.2 editx
- 位置：`vcdata_construction/.../split_XXXX/stepaudio_<edit_tag>_split_XXXX_all_qzrun/paired_report.jsonl`
- 6 种 edit 模式：`emotion_remove`, `emotion_coldness`, `style_chat`, `style_news`, `style_radio`, `style_remove`
- 关键字段：`audio1`（== ref_audio）, `text1` (== ref_text), `one_stage.audio2`（== edited_audio）, `metadata.source_row_index`（== vcdata `original_idx`）, `metadata.edit_tag`

### 1.3 emotion_eval 已验结论
- **`style_radio` 中性化最稳**（mean P(neutral)=0.702，BOTH agree 68.6%）
- `emotion_coldness` / `emotion_remove` 中性化效果几乎为 0
- 排名：`style_radio > style_news > style_chat > style_remove ≫ emotion_*`
- 默认 whitelist 只用 `style_radio`

---

## 2. 中间表

### 2.1 vcdata_base.jsonl
每行：
```json
{
  "sample_id": "split_0000:000000",
  "split": "split_0000",
  "original_idx": 0,
  "original_audio": "...",
  "original_text": "...",
  "ref_audio": "...",
  "ref_text": "...",
  "speaker_similarity": 0.91,
  "flag": "OK | LOW_SIM | ...",
  "duration": 3.7,
  "caption_summary": "...",
  "caption_gender": "Male",
  "zh_summary": "..."
}
```

### 2.2 editx_base.jsonl
```json
{
  "sample_id": "split_0000:style_radio:000000",
  "split": "split_0000",
  "source_row_index": 0,
  "edit_tag": "style_radio",
  "edit_type": "style",
  "edit_info": "radio",
  "input_audio": "...",
  "edited_audio": "...",
  "instruction": "style:radio",
  "original_audio_from_meta": "..."
}
```

### 2.3 joined_editx.jsonl
vcdata_base ⋈ editx_base on (split, original_idx == source_row_index)。

---

## 3. emotion 评估

对 3 个音频池跑评分：
- `original_audio` 池（vcdata_base 提取，symlink 转码后评分）
- `ref_audio` 池（`<split>/ref_audio/`）
- `edited_audio` 池（每个 whitelist 的 edit_tag 一个目录）

复用 `emotion_eval/scripts/score_neutrality.py` + `sensevoice_score.py`。

产物：`outputs/<split>/emotion/per_file_dual.csv`（path → 9 类概率 + top1 + sv_label）。

reuse-qzrun 模式下：直接 symlink `emotion_eval/outputs/zh_split_0000_qzrun/{per_file_dual,per_pair,per_row}.csv`，但 **不含 original_audio 评分**（A 与 H1 在该模式下产出受限）。

---

## 4. 各类 pair 构造规则

### 4.1 A.jsonl —— 只换文本，表达基本不变
- 来源：vcdata_base 全量
- 过滤：`speaker_similarity >= configs.a.sim_min`（默认 0.80，可选门槛）
- reference = original_audio, target = ref_audio
- text 两边不同（original_text vs ref_text）
- instruction = "保持参考音色，只换文本"（后标注，模型训练时可选不参与生成）

### 4.2 B_clean.jsonl —— 平淡 → 高表现
- 来源：joined_editx，仅 `edit_tag ∈ configs.bc_clean.edit_whitelist`（默认 `[style_radio]`）
- 过滤：
  - `P_neutral(edited) >= configs.bc_clean.edited_neutral_min`（默认 0.70）
  - `P_neutral(ref) <= configs.bc_clean.ref_neutral_max`（默认 0.50）—— 保证 ref 是"高表现"
  - 可选 `sv_label(edited) == "neutral"`
- reference = edited_audio, target = ref_audio
- text 两边一致（ref_text）
- instruction 从 `configs.b_clean.instruction_pool` 随机取

### 4.3 C_clean.jsonl —— 高表现 → 平淡
- 来源：同 B_clean
- 过滤：同 B_clean（注意 ref 仍要"高表现"才进入 pair）
- reference = ref_audio, target = edited_audio
- text 两边一致（ref_text）
- instruction 从 `configs.c_clean.instruction_pool` 随机取

### 4.4 H1.jsonl —— 零变化对照
- 来源：vcdata_base
- 过滤：
  - `speaker_similarity >= configs.h1.sim_min`（默认 0.85）
  - `emotion(original).top1 == emotion(ref).top1`
  - `cosine9(emotion(original), emotion(ref)) >= configs.h1.cosine_min`（默认 0.97）
- reference = original_audio, target = ref_audio
- instruction = "保持原样，不要改变表达方式"

### 4.5 H2.jsonl —— 已满足指令对照
- 来源：joined_editx，仅 `configs.h2.source_edit_tag`（默认 style_radio）
- reference 必须 `P_neutral >= 0.9` 且 `sv_label == "neutral"`
- 两种 target 模式：
  - `mode: self`（默认）—— target = reference 自身
  - `mode: neighbor` —— 从同属性池 sample 不同 row
- instruction 从 `configs.h2.instruction_pool` 随机取

### 4.6 H3.jsonl —— 跨 speaker 负样本
- 来源：vcdata_base（同 A 的 sim_min 门槛）
- 构造：对每个 anchor，从其他 row 随机 sample；可选 `require_gender_mismatch: true`
- reference / target 一对的 source_row_index 必须不同
- 显式标记 `is_negative: true`

---

## 5. 命名约定

- `sample_id`：`<split>:<original_idx 6位>` 或 `<split>:<edit_tag>:<original_idx 6位>`
- `pair_id`：`<split>:<pair_type>:<6位序号>`

---

## 6. 未来扩展（不在 v1 范围）

- B-mixed / C-mixed（跨文本版本）
- H2-happy / H2-angry（需要高置信 happy/angry 数据池）
- 多 split 全局采样去重（v1 是 per-split 独立）
