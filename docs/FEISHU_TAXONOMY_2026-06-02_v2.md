# pair_construction 在 Taxonomy v3 中的归位（周一对齐文档）

整理日期：2026-06-02
依据：Taxonomy v3 (`01_Taxonomy_Master.xlsx`，359 节点，10 条 routing rules)
基线数据：split_demo zh，100 句 vcdata，WavLM-L SeedTTSEval sim 阈值 0.80
Demo 路径：`outputs/split_demo/worklog_samples/taxonomy_demo/<节点名>/`

---

## 一、总览：建议的 8 类 + 我们 9 类 pair 对应表

| Taxonomy 节点 | 节点描述 | A/B 分支 | 我们对应的 pair 类 | 是否实现 | 100 句留存 | 未实现部分的后续方向 |
|---------------|---------|---------|------------------|---------|-----------|---------------------|
| **A1.4.1** Reference + Independent Instruction | reference 给 trait + 独立 instruction 改造（如"用这音色但年轻一点"） | A | **A 类**（基础音色克隆，宽松解读）| ✓ | 100 | 严格按 taxonomy E4 应归 A1.3.1.1（role-binding only）；A 类当前 instruction 只 role-binding 无独立 modification，未来可加 timbre design 类 instruction 严格匹配 A1.4.1 |
| **A1.3.4.1** Emotion Transfer | reference 的情绪迁移到 target_text | A | **D 类**（高表现→高表现 同情绪 跨文本）| ✓ | 19 | — |
| **A1.3.4.2** Affect Intensity Matching | 对齐情感强度，**可跨情绪类别** | A | **D_st 类**（高表现→高表现 同文本，当前同情绪子集） | ⚠ 部分 | 3 | 当前只支持同情绪（happy→happy），不支持跨情绪（happy→angry 同强度同 speaker）；新增 D_st_cross_emo |
| **A1.3.9.4** Avoid Source Content Leakage | 跨 speaker 负样本，防 leak | A | **H3 类** | ✓ | 100 | — |
| **B2.3.1** Emotion Conversion | 编辑 waveform 改情绪（**跨文本**形式）| B | **C_mixed 类**（高→中 跨文本） | ✓ | 43 | 当前不卡 sim（跨真人/合成 by design）；可能要补反向（中→高 跨文本）|
| **B2.3.2** Genre Conversion | 改 genre（电台/新闻/客服/旁白等风格互转）| B | **新增 Genre_conv 类**（待实现）| ❌ | 0 | editx 跑全 6 tag + 配 (style_radio, style_news 等) genre A↔B 互转 + WavLM-L sim 筛选 |
| **B2.5.2** Preserve Content Change Emotion | 保 content（**同文本同 speaker**）+ 改 emotion | B | **B 类**（中→高 同文本）+ **C 类**（高→中 同文本）| ✓ | 2 + 2 = 4 | 0.80 阈值下留存极少，靠 multi-attempt 或换 baseline 提质 |
| **B2.1.5** Speaker Identity Preservation with Cleanup | 编辑后保 speaker + cleanup 杂质 | B | **B/C/D_st 的 WavLM-L sim 过滤层**（不是单一类）| ✓ 过滤层 | 应用于 B/C/D_st | cleanup 部分缺：加 DNSMOS_BAK 阈值挡电音 + funASR-WER 防文本 leak |

**8 类之外的边界 case**：

| 我们的类 | 状态 | 建议 |
|----------|------|------|
| **A 类**（跨文本 timbre clone） | 宽松解读归 **A1.4.1**（项目所有 pair 都遵循 reference + instruction 范式）；严格 taxonomy 应归 **A1.3.1.1** Single-Reference Timbre Cloning（role-binding only）| 文档中已并入 A1.4.1 节 |
| **H1 类**（emo_cos≥0.97 零变化）| 归 **A1.3.1.1** 的强约束子集（保表达 + 保音色）| 弱训练信号，可暂留 |
| **H2 类**（self-ref 中性 baseline）| **taxonomy v3 无对应节点** | 建议不当训练数据用，或作 A1.x 退化 case（instruction = "保持原样"）|

---

## 二、Taxonomy v3 顶层架构

```
A  New Speech Performance Generation       (从无到有生成，不修改 waveform)
├── A1.3  Reference-Conditioned             (reference 给 trait)
│   ├── A1.3.1  Timbre Reference
│   │   └── A1.3.1.1  Single-Reference Timbre Cloning   ← 我们 A 类 / H1 类
│   ├── A1.3.4  Emotion / Affect Reference
│   │   ├── A1.3.4.1  Emotion Transfer                   ← 我们 D 类
│   │   └── A1.3.4.2  Affect Intensity Matching          ← 我们 D_st 类（同情绪子集）
│   └── A1.3.9  Leakage-Controlled
│       └── A1.3.9.4  Avoid Source Content Leakage       ← 我们 H3 类
└── A1.4  Composite (reference + independent instruction)
    └── A1.4.1  Reference + Independent Instruction      ← 当前无对应

B  Existing Speech Transformation          (修改 existing waveform)
└── B2  Global Speech Conversion
    ├── B2.1.5  Speaker Identity Preservation w/ Cleanup ← 我们 sim 过滤层
    ├── B2.3  Emotion and Style Conversion
    │   ├── B2.3.1  Emotion Conversion (跨文本)          ← 我们 C_mixed 类
    │   └── B2.3.2  Genre Conversion                     ← 待新增 Genre_conv 类
    └── B2.5  Decoupled Global Conversion
        └── B2.5.2  Preserve Content Change Emotion (同文本) ← 我们 B / C 类
```

**核心区分**（routing rule R2）：
- 训练任务是「修改/转换 existing waveform」→ **B 分支**
- 训练任务是「从 reference 提 trait + 重新合成」→ **A 分支**

**B2.3.1 vs B2.5.2 的核心区分**：
- B2.3.1：改情绪本身，可跨文本（→ C_mixed）
- B2.5.2：在 B2.3.1 基础上额外强制"同文本同 speaker"（→ B / C）

---

## 三、各 Taxonomy 节点详述

### A1.4.1 Reference plus Independent Instruction

**对应我们**：**A 类**（跨文本基础音色克隆）—— **宽松解读**

**Taxonomy 严格定义**：reference 给 trait + **非 role-binding 的独立 instruction** 共同决定 target。典型例子（taxonomy boundary E5）："用这段音色，但说得**更年轻一点**"。

**Routing 边界（taxonomy E4 vs E5）**：
- A1.3.1：instruction 只指定 reference 用于什么（"use this audio as the speaker voice"）→ pure role-binding
- A1.4.1：instruction **超出 reference 用法**，独立修改 output 属性（"use this voice but make it younger"）→ independent modification

**我们 A 类的实际情况**：
- instruction："保持参考音色，只换文本，表达方式基本不变"
- "保持参考音色" = role-binding（严格说归 A1.3.1）
- "表达方式基本不变" = preservation（不是 modification）
- **严格按 taxonomy E4，A 类应归 A1.3.1.1 Single-Reference Timbre Cloning**

**宽松解读**：项目层面所有 pair 都遵循 "reference + instruction → target" 复合范式，可把 A 类作为 A1.4.1 的 baseline 退化（instruction 是 trivial 的 "保持原样"）。本文档采用此解读以让建议 8 类全覆盖。

**具体管线**：

```
1) original_wav (真人，跨文本来源) → MOSS-TTS 16 候选 voice clone → ref_audio
2) 配对: reference = original_wav, target = ref_audio
3) 跨文本: reference_text ≠ target_text
4) 无 emotion / sim 过滤（全集）
5) instruction: "保持参考音色，只换文本，表达方式基本不变"
```

**Demo wav**：暂未单独打包 A 类（因为 A 类是无过滤全集，前 3 句即可代表）。`samples.jsonl` 可从 `outputs/split_demo/pairs/A.jsonl` 取头 3 行。

**补充信息**：
- split_demo 100 句留存：**100 句**（A 类无任何过滤，全集保留）
- 当前 instruction 是 role-binding + preservation 风格
- **严格 A1.4.1 训练任务（带独立 modification instruction）当前无对应**
- 后续方向：构造 "用这个音色但 {更年轻 / 更悲伤 / 更慢}" 类 instruction，新建 A1.4.1_design 类
- 实施成本：高（需要新数据 + 新 instruction 类型 + caption 模型支持多维度 attribute）

---

### B2.5.2 Preserve Content Change Emotion（建议优先级 1）

**对应我们**：**B 类**（中性 → 高表现，**同文本**）+ **C 类**（高表现 → 中性，**同文本**）

**Taxonomy 定义**：B 分支，保 content（同文本同 speaker）+ 改 emotion。是 B2.3.1 的"严格保 content"子集。

**具体管线**：

```
B 类（中→高）:
1) original_wav (真人高表现) → MOSS-TTS 16 候选 clone → ref_audio (保留高表现)
2) ref_audio → stepfun-editx (zh: style_radio) → edited_audio（中性化）
3) emotion eval: emotion2vec_plus_large 算 9 类概率 + sensevoice 交叉验证
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

注：目录名前缀写的是 B2.3.1，文档归类按修正后是 B2.5.2，不影响实际 wav 内容

| 类 | NN | ref top1/P_neu | tgt top1/P_neu | sim_wavlm | reference_text（= target_text 同文本）|
|----|----|----------------|----------------|-----------|----------------------------------------|
| B #01 | 01 | neutral / 0.91 | happy / 0.06 | **0.842** | 是的，掌握了核能，到了信息时。 |
| B #02 | 02 | neutral / 1.00 | sad / 0.00 | 0.805 | 我单纯觉得那个灯亮好看。 |
| B #03 | 03 | neutral / 0.90 | happy / 0.01 | 0.799 | 行，那你给我修吧，你在这待着，我过去看一下。 |
| C #01 | 01 | happy / 0.06 | neutral / 0.91 | **0.842** | 是的，掌握了核能，到了信息时。 |
| C #02 | 02 | sad / 0.00 | neutral / 1.00 | 0.805 | 我单纯觉得那个灯亮好看。 |
| C #03 | 03 | happy / 0.01 | neutral / 0.90 | 0.799 | 行，那你给我修吧，你在这待着，我过去看一下。 |

注：B 类和 C 类是同一对 (edited, ref_audio) 反向使用，sim 数值完全相同

**补充信息**：
- split_demo 100 句留存：B/C 各 **2 句**（0.80 阈值；0.75 阈值下各 10 句）
- 推全量 10000 句估算：B/C 各 ~200 句（0.80）/ ~1000 句（0.75）
- **建议下周最高优先级 = 规模化 B 类（中性→高表现）**
- 待优化：editx multi-attempt（同 input 跑 N 次取 sim 最高）能提留存率从 7% → 30%+

---

### B2.3.1 Emotion Conversion（跨文本形式）

**对应我们**：**C_mixed 类**（真人高表现 → 中性合成，**跨文本**）

**Taxonomy 定义**：B 分支，编辑 existing waveform 改情绪。与 B2.5.2 的区别是 **不强制保留 content**（可跨文本）。

**具体管线**：

```
1) original_wav (真人高表现) → 同 speaker 别条 ref_audio (vcdata clone)
2) ref_audio → editx style_radio → edited_audio (中性化)
3) 过滤:
   - reference (original_wav, 真人) top1 != neutral（高表现硬条件）
   - target (edited_audio, 合成) top1 == neutral
   - **不卡 sim**（跨真人/合成 by design，WavLM-L 物理上限只到 ~0.7）
4) 配对: reference = original_wav, target = edited_audio (别条同 speaker)
5) instruction: "从真人 {ref_emotion} 的语气，转换为中性平淡的合成语音，朗读新文本，保持说话人音色"
```

**Demo wav**：`outputs/split_demo/worklog_samples/taxonomy_demo/C_mixed_boundary/`

| NN | ref top1/P_neu | tgt top1/P_neu | sim_wavlm | reference_text → target_text（跨文本）|
|----|----------------|----------------|-----------|----------------------------------------|
| 01 | angry / 0.00 | neutral / 1.00 | 0.706 | 10万块啊，我这儿啊正好有10万块... → 你那笔延期费愁的怎么样了？... |
| 02 | disgusted / 0.29 | neutral / 1.00 | 0.699 | 如果要是贵公司不能答应我的条件... → 何大人事儿不是了了吗？... |
| 03 | surprised / 0.07 | neutral / 1.00 | 0.675 | 喝点汤，暖暖胃又何妨... → 龙舟刑侦大队梁英找你... |

**补充信息**：
- split_demo 100 句留存：**43 句**（不卡 sim 所以稳定产出）
- 推全量 10000 句估算：~4300 句
- sim 偏低是 by design（真人 vs 合成，物理上限），不该硬卡
- 后续可补反向（中→高 跨文本），但 vcdata 中性 original 不多，量级可能很少

---

### B2.3.2 Genre Conversion（新增类，待实现）

**对应我们**：**新增 Genre_conv 类**（当前无）

**Taxonomy 定义**：B 分支，对 existing waveform 做 genre 风格转换（电台/新闻/客服/旁白等）。

**具体管线（待实现）**：

```
1) original_wav → MOSS-TTS clone → ref_audio
2) ref_audio → editx 跑全 6 个 tag:
   - style_radio (电台风格)
   - style_news (新闻播报)
   - style_chat (客服/聊天)
   - style_remove (去 style)
   - emotion_coldness (冷漠)
   - emotion_remove (去情绪)
3) 配对 (style_A_edited, style_B_edited) 任意两两组合作为 genre 互转 pair
   例如: (style_radio, style_news) - 电台 ↔ 新闻
        (style_radio, style_chat) - 电台 ↔ 客服
4) 过滤:
   - WavLM-L speaker_sim ≥ 0.80（同 speaker 硬条件）
   - 不卡 emotion（两个 genre 都是平淡风格，emotion 都偏 neutral）
5) instruction: "保持音色，从电台播报风格转换为新闻播报风格" / "从 {genre_A} 转为 {genre_B}"
```

**Demo wav**：当前无（待新增脚本产出）

**补充信息**：
- 实现成本：中（editx 跑全 6 tag = 6× 当前 GPU 时间；配对脚本 1 天）
- 训练价值：中-高（专家级 genre 控制能力，对 instruction-TTS 多样性有用）
- 当前 D_st 类（之前误归 B2.3.2）实际上是 editx 漏改情绪的副产品，**严格说是 A1.3.4.2 的同情绪子集**（见下节）
- 建议立项：需 editx GPU 预算 + 配对脚本

---

### A1.3.4.1 Emotion Transfer

**对应我们**：**D 类**（高表现 → 高表现，**同情绪**，**跨文本**）

**Taxonomy 定义**：A 分支，用 role-bound reference audio 为 emotion transfer——reference 的情绪迁移到从 target_text 重新合成的音频。

**具体管线**：

```
1) original_wav (真人高表现) → MOSS-TTS 16 候选 clone → ref_audio (保情绪)
2) 利用 vcdata 已天然保留情绪的样本（MOSS-TTS clone 时 best_similarity 自然带情绪）
3) 筛选:
   - ref.top1 == original.top1（同情绪类别）
   - 9 维 emotion cosine ≥ 0.95（强度一致）
   - 双侧 top1 != neutral（双高表现）
4) 配对: reference = ref_audio (高表现合成), target = original_audio (真人同情绪)
5) **不卡 sim**（跨真人/合成 by design）
6) instruction: "保持当前 {emotion} 情绪和音色，朗读以下新文本"
```

**Demo wav**：`outputs/split_demo/worklog_samples/taxonomy_demo/D_A1_3_4_1_emotion_transfer/`

| NN | ref top1/P_neu | tgt top1/P_neu | emo_cos | sim_wavlm | reference_text → target_text（跨文本）|
|----|----------------|----------------|---------|-----------|----------------------------------------|
| 01 | fearful / 0.00 | fearful / 0.00 | 0.987 | 0.809 | 行，那你给我修吧... → 我的腿是王妃治好的，但是现在王妃生病... |
| 02 | angry / 0.00 | angry / 0.00 | 1.000 | 0.795 | 怪不得呢，原来是兄弟... → 像你这样为虎作伥的人... |
| 03 | happy / 0.00 | happy / 0.01 | 1.000 | 0.737 | 哎何伟行长你好啊... → 你自己在那边辛苦了... |

**补充信息**：
- split_demo 100 句留存：**19 句**（不卡 sim 所以稳定产出）
- 推全量 10000 句估算：~1900 句
- 这是当前 pair_construction 中**最稳定高产出**的"高表现 ↔ 高表现"类
- sim 在 0.5~0.8 区间是物理上限（clone vs real），不应卡 0.80

---

### A1.3.4.2 Affect Intensity Matching

**对应我们**：**D_st 类**（高表现 → 高表现，**同文本**，**当前只支持同情绪**）

**Taxonomy 定义**：A 分支，对齐 reference 和 target 的情感强度。**可以跨情绪类别**（如 ref=angry tgt=happy 但强度都强）。

**当前实现部分（D_st 同情绪子集）**：

```
1) MOSS-TTS clone 得 ref_audio (高表现 保情绪)
2) ref_audio → editx style_radio → edited_audio
3) 筛选 editx 漏改情绪的样本:
   - ref.top1 == edited.top1（情绪类别一致）
   - 9 维 emotion cosine ≥ 0.95（强度一致）
   - 双侧 top1 != neutral（双高表现）
4) 配对: reference = ref_audio, target = edited_audio (同文本)
5) WavLM-L sim ≥ 0.80
6) instruction: "保持当前 {emotion} 情绪和音色，换个语调说这句话"
```

**Demo wav**：`outputs/split_demo/worklog_samples/taxonomy_demo/D_st_B2_3_2_genre_conversion/`（目录名是旧的 B2.3.2，内容不变）

| NN | ref top1/P_neu | tgt top1/P_neu | emo_cos | sim_wavlm | reference_text（= target_text 同文本）|
|----|----------------|----------------|---------|-----------|----------------------------------------|
| 01 | surprised / 0.00 | surprised / 0.15 | 0.985 | **0.853** | 笑笑，算了吧，今天总算是有情无险。 |
| 02 | sad / 0.00 | sad / 0.00 | 1.000 | 0.821 | 谢谢您对我的关心，您就别管我。 |
| 03 | angry / 0.00 | angry / 0.01 | 1.000 | 0.809 | 何大人事儿不是了了吗？这么着急找下官有什么吩咐啊？ |

**未实现部分（跨情绪强度对齐）**：

A1.3.4.2 的真正定义允许 **ref top1 ≠ tgt top1 但强度一致**——比如 happy → angry 同 speaker 同高强度，当前 D_st 配置 `same_top1=true` 直接砍掉了这种 pair。

**新增 D_st_cross_emo 子任务的实现方案**：

```
1) MOSS-TTS clone 得 ref_audio
2) 筛选:
   - ref.top1 != tgt.top1（允许跨情绪类别）
   - ref.P_neu < 0.3 且 tgt.P_neu < 0.3（**双侧高强度**）
   - WavLM-L sim ≥ 0.80（同 speaker）
3) 不卡 emo_cos（因为不要求同情绪类别）
4) 配对: reference = ref_audio, target = 同 speaker 别 emotion 的 edited
5) instruction: "保持音色，从 {ref_emotion} 转换为 {tgt_emotion}，情感强度保持一致"
```

**补充信息**：
- 当前同情绪子集：split_demo 100 句留存 **3 句**（0.80 阈值）
- 跨情绪子集未实现：实施成本低（改 D_st 配置加 `allow_cross_emo: true`，1 天落地）
- 训练价值：高（"保留音色但改变情绪类别"比同情绪稀缺、更难学）
- **建议优先级仅次于 B 类规模化**

---

### A1.3.9.4 Avoid Source Content Leakage

**对应我们**：**H3 类**（跨 speaker 负样本）

**Taxonomy 定义**：A 分支，使用 reference 时**避免模型复制 reference 的原文本或错误 speaker 内容**。

**具体管线**：

```
1) 拿到 ref_audio (speaker A) 集合
2) 配对: reference = ref_audio (speaker A), target = 另一个 row 的 ref_audio (speaker B)
3) 文本可能相同也可能不同（target_text 来自 target 那一行）
4) instruction: "请根据指令生成新内容，不要复制参考音频的文本"
5) 训练目标: 让模型学会忽略 reference 的内容，只取音色 / 风格 trait
```

**Demo wav**：`outputs/split_demo/worklog_samples/taxonomy_demo/H3_A1_3_9_4_avoid_leakage/`

| NN | reference_audio | target_audio（跨 speaker）| reference_text → target_text |
|----|----------------|--------------------------|------------------------------|
| 01 | `000000_ref.wav` | `000081_ref.wav` | 我也搞不清楚，找个机会去问问他。 → 同文本但跨 speaker |
| 02 | `000001_ref.wav` | `000014_ref.wav` | 原则啊这个苦肉计... → 龙舟刑侦大队梁英找你了解一下情况。 |
| 03 | `000002_ref.wav` | `000003_ref.wav` | 都动筷子呀... → 行，那你给我修吧... |

**补充信息**：
- split_demo 100 句留存：**100 句**（所有 ref_audio 都参与跨 speaker 配对）
- 推全量 10000 句估算：~10000 句
- 这是**负样本类**，模型训练时学到"用 reference 音色但忽略其文本"
- taxonomy v3 的 A1.3.9 整个家族（A1.3.9.1~7）都是 leakage 控制，当前只覆盖 A1.3.9.4

---

### B2.1.5 Speaker Identity Preservation with Cleanup

**对应我们**：**所有 B 分支类（B/C/D_st）的 WavLM-L sim 过滤层**（元级过滤约束）

**Taxonomy 定义**：B 分支，对 existing speech 做转换的同时保 speaker identity，**并 cleanup 杂质**。

**具体管线（作为过滤层应用）**：

```
对所有 B 分支类（B / C / D_st）的 pair：
1) 算 WavLM-L + ECAPA-TDNN (SeedTTSEval) speaker_sim
2) 卡阈值: sim ≥ 0.80 (zh) / 0.55 (en)
3) 不过线的 pair 在 _filtered.jsonl 里被排除
4) 等同于强制要求"editx 改造后 speaker identity 没崩"
```

**Demo wav**：不单独打包（应用于所有 B 分支类的 demo）

**补充信息**：
- 当前已实现：scripts/11b_add_wavlm_sim.py（WavLM-L SeedTTSEval 标准）
- 之前用 CAM++ funasr 在 MOSS-TTS 合成音上虚高 ~0.10，已弃用
- 待补：**cleanup 部分**当前只做了 speaker preservation，**没做"去电音 / 去 artifact"**
- 后续方向：加 DNSMOS_BAK 阈值 + funASR-SquimMOS 二次校验 + funASR-WER 防文本 leak

---

## 四、taxonomy v3 中可扩展但未做的相关节点

| 节点 | 描述 | 实施成本 | 训练价值 | 建议优先级 |
|------|------|---------|---------|-----------|
| A1.3.4.4 Emotion-Preserving Cross-Language Generation | 用 ref 情绪生成另一语种 | 中（需 zh→en vcdata）| 高 | 等英文 baseline 换完后做 |
| A1.3.5.x Accent / Dialect Reference | 用 ref 给口音/方言 | 中 | 中 | 后排 |
| A1.3.7.x Paralinguistic Reference | 笑声 / 气音 / 哭腔 / 叹气 | 高（需特殊数据收集）| 中 | 后排 |
| **B2.2.x Prosody Conversion** | editx 改语速 / 节奏 / 停顿 | 低（editx 可能支持）| 高 | **优先做，对应 prosody control 需求** |
| B2.4.x Accent Conversion | 改口音 / 方言 | 中 | 中 | 后排 |
| **B2.5.4 Preserve Timbre Remove Recording Artifacts** | 去噪同时保音色（cleanup）| 中 | 高 | **对应"电音"问题，优先做** |
| A1.4.2 Timbre Cloning + Timbre Design | "用这个音色但年轻一点" | 高 | 高 | 等 baseline 换 IndexTTS-2 后 |
| A1.4.3 Reference + Local Performance Instruction | "用这个 ref，但第 3 个词强调" | 高 | 高 | 等 caption 模型接入后 |

---

## 五、周一对齐建议

### 5.1 8 类已覆盖情况

- ✓ 完全实现：**5/8**（A1.4.1 宽松解读对应 A 类 / A1.3.4.1 / A1.3.9.4 / B2.5.2 / B2.1.5）
- ⚠ 部分实现：**3/8**（A1.3.4.2 缺跨情绪强度对齐；B2.3.1 当前只有 C_mixed 高→中方向，可能要补反向；B2.3.2 需新增 Genre_conv 类）
- ❌ 未实现：**0/8**
- 严格 taxonomy 视角：A1.4.1 当前只是 baseline 退化（role-binding only），严格的 A1.4.1 训练任务（带独立 modification instruction）未实现
- 边界 case：H1 严格说归 A1.3.1.1 子集；H2 在 taxonomy 无对应

### 5.2 需澄清的 5 个问题

1. **严格 A1.4.1 是否优先构造**：当前 A 类是宽松解读对应 A1.4.1，严格 A1.4.1（reference + 独立 modification instruction，如"用这音色但年轻一点"）尚未实现，是否立项？
2. **A1.3.4.2 跨情绪强度对齐**（D_st_cross_emo）是否优先补？实施成本低，训练价值高
3. **B2.3.2 Genre_conv 新类是否立项**？需 editx 跑全 6 tag，GPU 成本高
4. **B2.5.4（去电音）是否立项**？对应"电音"问题
5. **H2 是否当训练数据用**？taxonomy 没节点

### 5.3 一句话总结

我们 9 类 pair 大致覆盖建议 8 类中的 5 个（含 A1.4.1 宽松解读）+ 3 个部分实现 + 2 个边界 case（H1 / H2），未来可在 taxonomy v3 框架下扩展 8 个相关节点（优先 B2.2.x 韵律转换 + B2.5.4 去电音 + A1.3.4.2 跨情绪强度对齐 + B2.3.2 Genre_conv 新类）。
