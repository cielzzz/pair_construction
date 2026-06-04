# pair_construction 在 Taxonomy v3 中的归位（周一对齐文档）

整理日期：2026-06-02
依据：PM 给的 Taxonomy v3 (`01_Taxonomy_Master.xlsx`，359 节点，10 条 routing rules)
基线数据：split_demo zh，100 句 vcdata，WavLM-L SeedTTSEval sim 阈值 0.80
Demo 路径：`outputs/split_demo/worklog_samples/taxonomy_demo/<节点名>/`

---

## 一、总览：PM 8 类 + 我们 9 类 pair 对应表

| Taxonomy 节点 | 节点描述 | A/B 分支 | 我们对应的 pair 类 | 是否实现 | 100 句留存 | 未实现部分的后续方向 |
|---------------|---------|---------|------------------|---------|-----------|---------------------|
| **A1.4.1** Reference + Independent Instruction | A 分支：reference 给 trait + 独立 instruction 控制 target | A | **整个 pair_construction 范式**（所有类都遵循）| ✓ 范式层 | — | — |
| **A1.3.4.1** Emotion Transfer | A 分支：reference 的情绪迁移到 target_text | A | **D 类**（跨文本 同情绪）| ✓ | 19 | — |
| **A1.3.4.2** Affect Intensity Matching | A 分支：对齐情感强度，**可跨情绪类别** | A | D 类的同情绪子集已有；**跨情绪强度对齐未实现** | ⚠ 部分 | 19 部分 | 新增 D_cross_emo 类：允许 ref top1≠tgt top1，但 P_neu 都低 + sim≥0.80 |
| **A1.3.9.4** Avoid Source Content Leakage | A 分支：跨 speaker 负样本，防 leak | A | **H3 类** | ✓ | 100 | — |
| **B2.3.1** Emotion Conversion | B 分支：编辑 waveform 改情绪 | B | **B 类**（中→高）+ **C 类**（高→中）| ✓ | 2 + 2 = 4 | 留存量太少，靠 multi-attempt 或换 baseline 提质 |
| **B2.3.2** Genre Conversion | B 分支：编辑 waveform 改 genre（电台/新闻/客服等）| B | **D_st 类**（editx 漏改情绪只改 genre 的子集）；**纯 genre A↔B 互转未实现** | ⚠ 部分 | 3 | editx 跑 6 个 tag，配 (style_radio_edited, style_news_edited) |
| **B2.5.2** Preserve Content Change Emotion | B 分支：保 content + 改 emotion（B2.3.1 的强约束子集） | B | **B 类 + C 类**（同文本同 speaker 自动满足）| ✓ | 4 | — |
| **B2.1.5** Speaker Identity Preservation with Cleanup | B 分支：保 speaker + cleanup 杂质（DNSMOS） | B | **B/C/D_st 类的 WavLM-L sim 过滤层**（不是单一类，是适用所有 B 分支的过滤约束）| ✓ 过滤层 | 应用于 B/C/D_st | 加 DNSMOS_BAK 阈值挡电音；加 funASR-WER 防文本 leak |

**边界 case（不在 PM 8 类）**：

| 我们的类 | 状态 | 建议 |
|----------|------|------|
| **A 类**（跨文本 timbre clone） | 归 A1.3.1.1 Single-Reference Timbre Cloning（taxonomy 有节点，PM 没单列）| 作为 baseline 保留 |
| **C_mixed**（跨文本 + 去情绪 + 真人 ref） | PM 8 类无对应 | 向 PM 澄清：归 A1.4.1（reference + 去情绪 instruction），还是新建 A1.3.4.5 "Emotion Removal" |
| **H1**（emo_cos≥0.97 保表达不变）| 归 A1.3.1.1 + A1.3.4.x | 弱训练信号，可暂留 |
| **H2**（self-ref 中性 baseline）| PM 8 类无对应，taxonomy v3 也没 self-ref 节点 | 建议不当训练数据用，或作为 baseline 单独处理 |

---

## 二、Taxonomy v3 顶层架构（一图秒懂）

```
A  New Speech Performance Generation       (从无到有生成，不修改 waveform)
├── A1.3  Reference-Conditioned             (reference 给 trait)
│   ├── A1.3.4  Emotion / Affect Reference
│   │   ├── A1.3.4.1  Emotion Transfer       ← 我们 D 类
│   │   └── A1.3.4.2  Affect Intensity Matching  ← 我们 D 子集
│   └── A1.3.9  Leakage-Controlled
│       └── A1.3.9.4  Avoid Source Content Leakage  ← 我们 H3 类
└── A1.4  Composite (reference + independent instruction)
    └── A1.4.1  Reference + Independent Instruction  ← 整个 pair_construction 范式

B  Existing Speech Transformation          (修改 existing waveform)
└── B2  Global Speech Conversion
    ├── B2.1.5  Speaker Identity Preservation with Cleanup  ← 我们 sim 过滤层
    ├── B2.3  Emotion and Style Conversion
    │   ├── B2.3.1  Emotion Conversion       ← 我们 B/C 类
    │   └── B2.3.2  Genre Conversion         ← 我们 D_st 类
    └── B2.5  Decoupled Global Conversion
        └── B2.5.2  Preserve Content Change Emotion  ← 我们 B/C 强约束子集
```

**核心区分**（routing rule R2）：
- 训练任务是「修改/转换 existing waveform」→ **B 分支**
- 训练任务是「从 reference 提 trait + 重新合成」→ **A 分支**

---

## 三、各 Taxonomy 节点详述

### A1.4.1 Reference plus Independent Instruction

**对应我们**：整个 pair_construction 范式（所有 9 类 pair 都遵循这个 IO 形态）

**Taxonomy 定义**：reference-derived trait + non-reference instruction 共同决定 target。

**具体管线**：

```
1) 输入：reference_audio + instruction + target_text
2) 模型从 reference 提取 trait（音色 / 情绪 / 韵律 / 风格 etc.）
3) 按 instruction 控制 target 的表达
4) 输出：target_audio（按 target_text 内容 + instruction 指定的表达 + reference trait）
```

**训练样本格式（与 PM 对齐）**：
```json
{
  "original_wav":  "<reference 音频路径>",
  "original_text": "<reference 文本>",
  "target_text":   "<目标文本>",
  "instruction":   "<指令>"
}
```

**Demo wav**：不单独打包，请参考下面任一类（C / D / D_st 等都是 A1.4.1 的实例）

**补充信息**：
- 这是**元级范式**，所有具体训练任务（B2.3.1 / A1.3.4.1 等）都是它的实例
- routing rule R7 + R8 决定一个具体任务归到 A1.3 还是 A1.4：只 role-binding 留 A1.3，有独立 modification instruction 归 A1.4

---

### B2.3.1 Emotion Conversion（PM 优先级 1）

**对应我们**：**B 类**（中性 → 高表现，同文本）+ **C 类**（高表现 → 中性，同文本）

**Taxonomy 定义**：B 分支，对 existing speech 做 emotion conversion，保留要求的 content / traits。

**具体管线**：

```
B 类（中→高）:
1) original_wav (真人高表现) → MOSS-TTS 16 候选 clone → ref_audio
2) ref_audio → stepfun-editx (zh: style_radio) → edited_audio（中性化）
3) emotion eval: 用 emotion2vec_plus_large 算 9 类概率
4) 过滤:
   - reference (edited) top1 == neutral（中性侧硬条件）
   - target (ref_audio) top1 != neutral（高表现侧硬条件）
   - WavLM-L speaker_sim ≥ 0.80（音色一致）
5) 配对: reference = edited_audio, target = ref_audio
6) instruction (临时手写 transformation 风格):
   "从中性平淡的语气，转换为 {tgt_emotion} 的语气，保持音色相似度不变"

C 类（高→中）: 同上反向，reference = ref_audio, target = edited_audio
```

**Demo wav**：`outputs/split_demo/worklog_samples/taxonomy_demo/B_2_3_1_mid_to_high/` 和 `C_B2_3_1_high_to_mid/`

| 类 | NN | ref top1/P_neu | tgt top1/P_neu | sim_wavlm | reference_text（= target_text 同文本）|
|----|----|----------------|----------------|-----------|----------------------------------------|
| B #01 | 01 | neutral / 0.91 | happy / 0.06 | **0.842** | 是的，掌握了核能，到了信息时。 |
| B #02 | 02 | neutral / 1.00 | sad / 0.00 | 0.805 | 我单纯觉得那个灯亮好看。 |
| B #03 | 03 | neutral / 0.90 | happy / 0.01 | 0.799 | 行，那你给我修吧，你在这待着，我过去看一下。 |
| C #01 | 01 | happy / 0.06 | neutral / 0.91 | **0.842** | 是的，掌握了核能，到了信息时。 |
| C #02 | 02 | sad / 0.00 | neutral / 1.00 | 0.805 | 我单纯觉得那个灯亮好看。 |
| C #03 | 03 | happy / 0.01 | neutral / 0.90 | 0.799 | 行，那你给我修吧，你在这待着，我过去看一下。 |

注：B 类和 C 类是同一对 (edited, ref_audio) 反向使用，sim 数值完全相同。

**补充信息**：
- split_demo 100 句留存：B/C 各 **2 句**（0.80 阈值下；0.75 阈值下各 10 句）
- 推全量 10000 句估算：B/C 各 ~200 句（0.80）/ ~1000 句（0.75）
- **PM 下周最高优先级 = 规模化 B 类**（中性→高表现）
- 待优化：用 editx multi-attempt（同 input 跑 N 次取 sim 最高）能提留存率从 7% → 30%+

---

### B2.5.2 Preserve Content but Change Emotion

**对应我们**：**B 类 + C 类的真子集**（B2.5.2 是 B2.3.1 + 严格 preserve content 约束）

**Taxonomy 定义**：在 B2.3.1 基础上多了"严格保留 content（同文本同 speaker）"约束。

**具体管线**：与 B2.3.1 完全相同（因为我们 B/C 类已经同时满足同文本同 speaker）

**Demo wav**：同 B2.3.1（B 类和 C 类的 6 个样本都同时属于 B2.3.1 和 B2.5.2）

**补充信息**：
- B2.5.2 是 B2.3.1 的强约束子集，不需要单独构造数据
- 区别 B2.5.2 vs B2.3.1：B2.3.1 可以"改文本同时改情绪"，B2.5.2 强制"只改情绪不改文本"
- 我们 B/C 类天然都满足 B2.5.2

---

### B2.3.2 Genre Conversion

**对应我们**：**D_st 类**（editx 改 genre 但漏改情绪的子集）；**纯 genre 互转未实现**

**Taxonomy 定义**：B 分支，对 existing speech 做 genre conversion（电台/新闻/客服/旁白等风格方向），保留要求的 content / traits。

**具体管线（当前 D_st 的"曲线救国"实现）**：

```
1) original_wav → MOSS-TTS clone → ref_audio (保情绪)
2) ref_audio → stepfun-editx (style_radio) → edited_audio
3) 筛选 editx 漏改情绪的样本:
   - ref.top1 == edited.top1（情绪类别一致）
   - 9 维 emotion cosine ≥ 0.95（强度也一致）
   - ref/edited 都非 neutral
4) 这些样本相当于："editx 改了 genre 但保了 emotion"
5) 配对: reference = ref_audio, target = edited_audio（B 范式：reference 是 substrate）
6) WavLM-L sim ≥ 0.80
7) instruction: "保持当前情绪和音色，换个语调/播报风格说同样的话"
```

**Demo wav**：`outputs/split_demo/worklog_samples/taxonomy_demo/D_st_B2_3_2_genre_conversion/`

| NN | ref top1/P_neu | tgt top1/P_neu | emo_cos | sim_wavlm | reference_text（= target_text 同文本）|
|----|----------------|----------------|---------|-----------|----------------------------------------|
| 01 | surprised / 0.00 | surprised / 0.15 | 0.985 | **0.853** | 笑笑，算了吧，今天总算是有情无险。 |
| 02 | sad / 0.00 | sad / 0.00 | 1.000 | 0.821 | 谢谢您对我的关心，您就别管我。 |
| 03 | angry / 0.00 | angry / 0.01 | 1.000 | 0.809 | 何大人事儿不是了了吗？这么着急找下官有什么吩咐啊？ |

**补充信息**：
- split_demo 100 句留存：**3 句**（0.80 阈值；0.75 阈值下 5 句）
- 当前是 D_st 类的"伪 B2.3.2"实现，严格说 D_st 是「editx 漏改情绪」的副产品
- **真正的 B2.3.2 未实现**：需要 editx 跑全 6 个 tag，配 (style_radio_edited, style_news_edited) 这种 genre A ↔ genre B 互转 pair
- 后续方向：editx 改造 6 tag 全跑 + 加 genre A ↔ B 配对脚本

---

### A1.3.4.1 Emotion Transfer

**对应我们**：**D 类**（高表现 → 高表现，同情绪，跨文本）

**Taxonomy 定义**：A 分支，用 role-bound reference audio 为 emotion transfer——reference 的情绪直接迁移到从 target_text 重新合成的音频。

**具体管线**：

```
1) original_wav (真人高表现) → MOSS-TTS 16 候选 clone → ref_audio (保情绪)
2) 利用 vcdata 已天然保留情绪的样本（MOSS-TTS clone 时 best_similarity 会自然带情绪）
3) 筛选:
   - ref.top1 == original.top1（同情绪）
   - 9 维 emotion cosine ≥ 0.95（强度一致）
   - 双侧 top1 != neutral（双高表现）
4) 配对: reference = ref_audio (高表现合成), target = original_audio (真人同情绪)
5) 不卡 sim（跨真人/合成 by design，sim 上限只到 0.66）
6) instruction: "保持当前的 {emotion} 情绪和音色，朗读以下新文本"
```

**Demo wav**：`outputs/split_demo/worklog_samples/taxonomy_demo/D_A1_3_4_1_emotion_transfer/`

| NN | ref top1/P_neu | tgt top1/P_neu | emo_cos | sim_wavlm | reference_text → target_text（跨文本）|
|----|----------------|----------------|---------|-----------|----------------------------------------|
| 01 | fearful / 0.00 | fearful / 0.00 | 0.987 | 0.809 | 行，那你给我修吧... → 我的腿是王妃治好的，但是现在王妃生病... |
| 02 | angry / 0.00 | angry / 0.00 | 1.000 | 0.795 | 怪不得呢，原来是兄弟... → 像你这样为虎作伥的人... |
| 03 | happy / 0.00 | happy / 0.01 | 1.000 | 0.737 | 哎何伟行长你好啊... → 你自己在那边辛苦了... |

**补充信息**：
- split_demo 100 句留存：**19 句**（不卡 sim，所以是稳定产出）
- 推全量 10000 句估算：~1900 句
- 这是当前 pair_construction 中**最稳定高产出**的"高表现 ↔ 高表现"类
- sim 在 0.5~0.8 区间是物理上限（clone vs real），不应卡 0.80

---

### A1.3.4.2 Affect Intensity Matching

**对应我们**：**D 类的同情绪子集（已有）+ 跨情绪强度对齐（未实现）**

**Taxonomy 定义**：A 分支，对齐 reference 和 target 的情感强度。**可以跨情绪类别**（如 ref=angry tgt=happy 但强度都强）。

**当前实现部分**：
- D 类中 emo_cos ≥ 0.95 的样本 = 同情绪类别 + 强度对齐（A1.3.4.1 的强子集，同时也满足 A1.3.4.2）

**未实现部分（跨情绪强度对齐）**：
- 当前 D 类要求 same_top1 = true，**直接砍掉了"angry → happy 同 speaker 同强度"这种 pair**

**具体管线（未来 D_cross_emo 类）**：

```
1) MOSS-TTS clone 得 ref_audio
2) 筛选:
   - ref.top1 != tgt.top1（允许跨情绪类别）
   - ref.P_neu < 0.3 且 tgt.P_neu < 0.3（双侧都高强度）
   - WavLM-L sim ≥ 0.80（同 speaker）
3) 不卡 emo_cos（因为不要求同情绪）
4) 配对: reference = ref_audio, target = original_audio (跨情绪 同强度)
5) instruction: "保持音色，从 {ref_emotion} 转换为 {tgt_emotion}，强度保持一致"
```

**Demo wav**：同 D 类（当前 D 19 句都属于同情绪子集；跨情绪样本待新增脚本产出）

**补充信息**：
- 实现成本：低（改 D 类配置加 `allow_cross_emo: true` 选项）
- 训练价值：高（"保留音色但改变情绪类别"比同情绪更稀缺、更难，模型学到后泛化能力强）
- 建议优先级：**仅次于 PM 优先级 1 的 B 类规模化**

---

### A1.3.9.4 Avoid Source Content Leakage

**对应我们**：**H3 类**（跨 speaker 负样本）

**Taxonomy 定义**：A 分支，使用 reference 时**避免模型复制 reference 的原文本或错误 speaker 内容**。

**具体管线**：

```
1) 拿到 ref_audio (speaker A) 集合
2) 配对: reference = ref_audio (speaker A), target = 另一个 row 的 ref_audio (speaker B)
3) 文本可能相同也可能不同（target_text 来自 target 那一行）
4) instruction: "请根据指令生成新内容，不要复制参考音频的文本"（或类似防 leak 指令）
5) 训练目标: 让模型学会忽略 reference 的内容，只取音色/风格 trait
```

**Demo wav**：`outputs/split_demo/worklog_samples/taxonomy_demo/H3_A1_3_9_4_avoid_leakage/`

| NN | reference_audio | target_audio（跨 speaker）| reference_text → target_text |
|----|----------------|--------------------------|------------------------------|
| 01 | `000000_ref.wav` | `000081_ref.wav` | 我也搞不清楚，找个机会去问问他。 → 我也搞不清楚，找个机会去问问他。（同文本但跨 speaker）|
| 02 | `000001_ref.wav` | `000014_ref.wav` | 原则啊这个苦肉计... → 龙舟刑侦大队梁英找你了解一下情况。 |
| 03 | `000002_ref.wav` | `000003_ref.wav` | 都动筷子呀... → 行，那你给我修吧... |

**补充信息**：
- split_demo 100 句留存：**100 句**（所有 ref_audio 都参与跨 speaker 配对）
- 推全量 10000 句估算：~10000 句
- 这是**负样本类**，模型训练时应该学到"用 reference 音色但忽略其文本"
- taxonomy v3 的 A1.3.9 整个家族（A1.3.9.1~7）都是 leakage 控制，我们只覆盖了 A1.3.9.4

---

### B2.1.5 Speaker Identity Preservation with Cleanup

**对应我们**：**所有 B 分支类（B/C/D_st）的 WavLM-L sim 过滤层**（不是单一 pair 类，是元级过滤约束）

**Taxonomy 定义**：B 分支，对 existing speech 做转换的同时保 speaker identity，**并 cleanup 杂质**。

**具体管线（作为过滤层应用）**：

```
对所有 B 分支类（B/C/D_st）的 pair：
1) 算 WavLM-L + ECAPA-TDNN (SeedTTSEval) speaker_sim
2) 卡阈值: sim >= 0.80 (zh) / 0.55 (en)
3) 不过线的 pair 在 _filtered.jsonl 里被排除
4) 等同于强制要求"editx 改造后 speaker identity 没崩"
```

**Demo wav**：不单独打包（应用于所有 B 分支类的 demo）

**补充信息**：
- 当前已实现：scripts/11b_add_wavlm_sim.py
- WavLM-L SeedTTSEval 是 industry standard 的 speaker sim
- 之前用 CAM++ funasr 在 MOSS-TTS 合成音上虚高 ~0.10，已弃用
- 待补：**cleanup 部分**当前只做了 speaker preservation，**没做"去电音 / 去 artifact"**（PM 提的"音频比较电"问题）
- 后续方向：加 DNSMOS_BAK 阈值 + funASR-SquimMOS 二次校验

---

## 四、边界 case 说明

### C_mixed 类（不在 PM 8 类）

**物理结构**：reference = 真人 original_audio（高表现），target = 同 speaker 别条 ref_audio 经 editx 中性化的 edited_audio，**跨文本**。

**边界判断**：
- 不是 B 分支：跨文本，target 不是 reference 的 edited 产物
- 不是 A1.3.4.1：不是迁移 reference 的情绪，反而是**去除**
- 最接近 A1.4.1（reference + 独立 instruction "去除情绪"）

**建议**：向 PM 澄清归 A1.4.1 还是新建 A1.3.4.5 "Emotion Removal"。

**Demo wav**：`outputs/split_demo/worklog_samples/taxonomy_demo/C_mixed_boundary/`

| NN | ref top1/P_neu | tgt top1/P_neu | sim_wavlm | reference_text → target_text |
|----|----------------|----------------|-----------|------------------------------|
| 01 | angry / 0.00 | neutral / 1.00 | 0.706 | 10万块啊，我这儿啊正好有10万块... → 你那笔延期费愁的怎么样了？... |
| 02 | disgusted / 0.29 | neutral / 1.00 | 0.699 | 如果要是贵公司不能答应我的条件... → 何大人事儿不是了了吗？... |
| 03 | surprised / 0.07 | neutral / 1.00 | 0.675 | 喝点汤，暖暖胃又何妨... → 龙舟刑侦大队梁英找你... |

split_demo 100 句留存：**43 句**（不卡 sim，因为跨真人/合成 by design sim 偏低）

### H2 类（self-reference baseline）

**物理结构**：reference = edited_audio（中性 self pool 中的一段），target = 同一段 edited_audio。reference 和 target 是**同一个文件**。

**边界判断**：taxonomy v3 没有专门的 self-reference 节点。

**建议**：不当训练数据用（避免模型学"复制 input"），或作为 A1.4.1 退化 case 处理（instruction = "保持原样不变"）。

---

## 五、taxonomy v3 中可扩展但未做的相关节点

| 节点 | 描述 | 实施成本 | 训练价值 | 建议优先级 |
|------|------|---------|---------|-----------|
| A1.3.4.4 Emotion-Preserving Cross-Language Generation | 用 ref 情绪生成另一语种 | 中（需 zh→en vcdata）| 高 | 等英文 baseline 换完后做 |
| A1.3.5.x Accent / Dialect Reference | 用 ref 给口音/方言 | 中 | 中 | 后排 |
| A1.3.7.x Paralinguistic Reference | 笑声 / 气音 / 哭腔 / 叹气 | 高（需特殊数据收集）| 中 | 后排 |
| **B2.2.x Prosody Conversion** | editx 改语速 / 节奏 / 停顿 | 低（editx 可能支持）| 高 | **PM 提的 "Prosody Control"，优先做** |
| B2.4.x Accent Conversion | 改口音 / 方言 | 中 | 中 | 后排 |
| **B2.5.4 Preserve Timbre Remove Recording Artifacts** | 去噪同时保音色（cleanup）| 中 | 高 | **PM 提的"电音"问题对接** |
| A1.4.2 Timbre Cloning + Timbre Design | "用这个音色但年轻一点" | 高 | 高 | 等 baseline 换 IndexTTS-2 后 |
| A1.4.3 Reference + Local Performance Instruction | "用这个 ref，但第 3 个词强调" | 高 | 高 | 等 caption 模型接入后 |

---

## 六、周一对齐建议

### 6.1 PM 8 类已覆盖情况

- ✓ 完全实现：**5/8**（A1.4.1 / A1.3.4.1 / A1.3.9.4 / B2.3.1 / B2.5.2 / B2.1.5 = 6 个，其中 A1.4.1 和 B2.1.5 是元级）
- ⚠ 部分实现：**2/8**（A1.3.4.2 缺跨情绪强度；B2.3.2 缺纯 genre 互转）
- 边界 case：C_mixed 不在 PM 8 类，H2 也不在 taxonomy

### 6.2 需 PM 拍板的 5 个问题

1. **C_mixed 怎么归**（A1.4.1 / 新建 A1.3.4.5 / 不要 / 其他）
2. **H2 是否当训练数据用**（taxonomy 没节点）
3. **A1.3.4.2 是否优先补"跨情绪强度对齐"子任务**（实施成本低，训练价值高）
4. **B2.3.2 是否补"纯 genre 互转"**（需 editx 跑全 6 tag，GPU 成本高）
5. **B2.5.4（去电音）是否立项**（对应 PM 提的"电音"问题）

### 6.3 一句话总结

我们 9 类 pair 大致覆盖 PM 8 类中的 6 个，2 个边界 case 待澄清，未来可在 taxonomy v3 框架下扩展 8 个相关节点（优先 B2.2.x 韵律转换 + B2.5.4 去电音 + A1.3.4.2 跨情绪强度对齐）。
