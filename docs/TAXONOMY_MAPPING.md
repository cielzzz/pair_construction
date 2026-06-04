# pair_construction 在 PM Taxonomy v3 中的精确归位

整理日期：2026-06-02
依据：`01_Taxonomy_Master.xlsx` v3.0（2026-05-16，359 节点，10 条 routing rules，18 个 boundary examples）
重要：取代我此前的"分类有重叠"诊断（理解错了）

---

## 0. 关键认知修正（之前的错误）

我之前以为 PM 给的 8 个分类有 3 对重叠（B2.3.1 vs B2.5.2 / A1.3.4.1 vs A1.3.4.2 / B2.3.1 vs B2.3.2），**这是错的**。

读完 taxonomy v3 后明确：

| 节点 | 实际位置 | 关键约束（vs 兄弟）|
|------|---------|--------------------|
| **B2.3.1** Emotion Conversion | B2.3 Emotion and Style Conversion 下 | 改情绪本身（不强制 preserve content）|
| **B2.5.2** Preserve Content but Change Emotion | B2.5 **Decoupled** Global Conversion 下 | 比 B2.3.1 多了"严格 preserve content"约束（同文本 + 同 speaker）|
| **A1.3.4.1** Emotion Transfer | A1.3.4 Emotion/Affect Reference 下 | reference 的情绪直接迁移到 target |
| **A1.3.4.2** Affect Intensity Matching | 同上兄弟节点 | 对齐强度，**可跨情绪类别**（angry→happy 但强度一致）|
| **B2.3.1** vs **B2.3.2** | 同 B2.3 下兄弟 | 一个改情绪、一个改 genre，**不同维度** |

它们**不是重叠，是 taxonomy 树上严格定义的兄弟节点**，各自有不同的 constraint。

---

## 1. Taxonomy v3 顶层架构

```
S (root)
├── A  New Speech Performance Generation              ← 不修改 existing waveform，从 text/reference 重新生成
│   ├── A1  Single-Stream / Single-Speaker
│   │   ├── A1.1  Text-only (无 reference 无 instruction)
│   │   ├── A1.2  Instruction-only (无 reference)
│   │   ├── A1.3  Reference-Conditioned (reference 给某种 trait)
│   │   │   ├── A1.3.1  Timbre Reference            ← 我们的 A / H1
│   │   │   ├── A1.3.2  Rhythm / Prosody Reference
│   │   │   ├── A1.3.3  Style / Genre Reference
│   │   │   ├── A1.3.4  Emotion / Affect Reference  ← 我们的 D
│   │   │   ├── A1.3.5  Accent / Dialect Reference
│   │   │   ├── A1.3.6  Pronunciation Reference
│   │   │   ├── A1.3.7  Paralinguistic Reference
│   │   │   ├── A1.3.8  Multi-Reference Role-Separated
│   │   │   └── A1.3.9  Leakage-Controlled Reference  ← 我们的 H3
│   │   └── A1.4  Composite (reference + independent instruction)
│   │       └── A1.4.1  Reference + Independent Instruction ← 整个 pair_construction 范式
│   └── A2  Multi-Speaker / Cast (我们暂未做)
├── B  Existing Speech Transformation                  ← 修改/转换 existing waveform
│   ├── B1  Local Edit (局部编辑)
│   ├── B2  Global Speech Conversion (全局转换)        ← 我们的 B / C / D_st
│   │   ├── B2.1  Voice / Timbre Conversion
│   │   │   └── B2.1.5  Speaker Identity Preservation with Cleanup  ← 我们的 sim 过滤层
│   │   ├── B2.2  Prosody Conversion
│   │   ├── B2.3  Emotion and Style Conversion
│   │   │   ├── B2.3.1  Emotion Conversion           ← 我们的 B / C
│   │   │   └── B2.3.2  Genre Conversion             ← 我们的 D_st（实质）
│   │   ├── B2.4  Accent Conversion
│   │   └── B2.5  Decoupled Global Conversion
│   │       └── B2.5.2  Preserve Content Change Emotion  ← 我们 B / C 的强约束子集
│   └── B3/B4/B5  其他
└── C  External Timeline / Media Sync (dubbing, lip-sync, 我们暂未做)
```

## 2. Routing 决策树（10 条规则的精简版）

按优先级从高到低判定一个任务归到哪：

| 优先级 | 条件 | 归到 | 例子 |
|--------|------|------|------|
| R1 | 任务必须符合外部 timeline / 媒体（字幕/口型/视频）| **C** | 视频配音对口型 |
| **R2** | 任务 **修改 / 转换 existing speech waveform** | **B** | 把已有片段换情绪 |
| R3 | 新生成涉及多 speaker / 对话结构 | A2 | 主播+嘉宾对谈 |
| R4 | 单流 + 无 reference + 无 instruction | A1.1 | 朗读文章 |
| R5 | 单流 + 仅 instruction（无 reference）| A1.2 | "慢一点更抱歉地说" |
| **R6** | 单流 + reference **只做 trait 绑定** | **A1.3** | "用这段音色读" |
| **R7** | 单流 + reference + **独立 instruction** | **A1.4** | "用这段音色但说得更年轻" |
| R8 | 决定 A1.3 还是 A1.4 的细则 | 同上 | role-binding 留 A1.3；独立修改进 A1.4 |
| R9/R10 | streaming/latency/alignment 不是 taxonomy 节点 | metadata | 单独记录 |

**核心判别**：
- 训练数据是「修改已有 waveform」→ B
- 训练数据是「从 reference 提 trait + 重新生成」→ A1.3 / A1.4

---

## 3. 我们 9 类 pair 在 taxonomy 中的精确归位

注意：**同一份物理 pair 数据 (reference_audio, target_audio) 在不同训练范式下可以归到 A 或 B**，下表是按训练目标（PM 优先级 1 是 B 分支编辑任务）来定的：

| 我们的类 | 文本关系 | reference 角色 | target 来源 | Taxonomy 节点 | 数据管线 |
|----------|---------|---------------|------------|---------------|----------|
| **A** | 跨 | trait source (timbre) | MOSS-TTS 合成 | **A1.3.1.1** Single-Reference Timbre Cloning | 1) original_wav → MOSS-TTS 克隆 → ref_audio<br>2) 配对 (original, ref)，跨文本 |
| **B** | 同 | substrate（中性→改成高表现）| MOSS-TTS 合成的 ref_audio | **B2.3.1** Emotion Conversion + **B2.5.2** Preserve Content Change Emotion | 1) original → MOSS-TTS → ref_audio (高表现)<br>2) ref → editx style_radio → edited (中性)<br>3) 配对 (edited→ref)，B 分支：reference 是要被改的 substrate |
| **C** | 同 | substrate（高→改成中性）| editx style_radio 输出 | **B2.3.1** + **B2.5.2** | 同 B 但反向：(ref→edited) |
| **C_mixed** | 跨 | trait source（真人情绪给 → 但要去掉）| editx 中性合成 | ⚠ **不在 PM 8 类**：跨文本 + 改 emotion + 真人 ref，最接近 A1.3.4.x 但语义反向（不是迁移而是去除）| 1) original (真人高表现) + 同 speaker 别条 ref_audio (vcdata clone)<br>2) ref → editx → edited (中性)<br>3) 配对 (real_original, 别条 edited) |
| **D** | 跨 | trait source (情绪) | 真人 original | **A1.3.4.1** Emotion Transfer ★ | 1) MOSS-TTS 克隆得高表现 ref_audio (保情绪)<br>2) 配对 (ref → original)，让 reference 的情绪迁移到 target_text |
| **D_st** | 同 | substrate（保情绪改 genre）| editx 改造但情绪未变 | **B2.3.2** Genre Conversion（实质：editx 漏改情绪只改了 genre 的样本子集）| 1) ref → editx → edited<br>2) 筛 ref.top1 == edited.top1（情绪未变）<br>3) 视为"genre 改变了 + emotion 保持"的 B2.3.2 case |
| **H1** | 跨 | trait source (timbre + 情绪)| MOSS-TTS clone | **A1.3.1.1** + 部分 A1.3.4.1（保表达全维度）| 1) MOSS-TTS clone 后 emo_cos≥0.97（零变化）<br>2) 配对 (original, ref) |
| **H2** | 同 | self-reference baseline | edited self | ⚠ **没有专门 taxonomy 节点**（self-ref 是 baseline，可能作 A1.x 退化 case 训练）| editx 中性 self pool 同一段两次使用 |
| **H3** | 跨 | trait source 但模型应避免复制 | 另一 speaker 的 ref | **A1.3.9.4** Avoid Source Content Leakage ★ | 1) ref_audio (speaker A)<br>2) 配对 (A, speaker B 的 ref)，作为模型防 leak 训练的负样本 |

## 4. PM 8 类对应矩阵（修正版）

| PM 节点 | Taxonomy 描述 | 我们对应的 pair 类 | split_demo 100 句留存（WavLM-L≥0.80）| 状态 |
|---------|--------------|---------------------|----------------------------------|------|
| **A1.4.1** Reference + Independent Instruction | A 分支范式（reference 给 trait + 独立 instruction）| **整个 pair_construction 范式**（所有类都遵循）| 范式层 | ✓ 范式 |
| **A1.3.4.1** Emotion Transfer | reference 的情绪迁移到 target_text | **D 类** | 19 | ✓ |
| **A1.3.4.2** Affect Intensity Matching | 对齐情感强度（可跨情绪类别）| **D 类的强度严约束子集**（emo_cos≥0.95）；跨情绪强度对齐未实现 | 19 部分 | ⚠ 部分 |
| **A1.3.9.4** Avoid Source Content Leakage | 防止 reference 内容/speaker 泄露到 target | **H3 类**（跨 speaker 负样本）| 100 | ✓ |
| **B2.3.1** Emotion Conversion | B 分支：编辑已有 waveform 改情绪 | **B + C 类**（中↔高 双向）| 2 + 2 = 4 | ✓ |
| **B2.3.2** Genre Conversion | B 分支：编辑已有 waveform 改 genre | **D_st 类**（editx 漏改情绪只改 genre 的样本子集）；纯 genre A↔B 互转未实现 | 3 部分 | ⚠ 部分 |
| **B2.5.2** Preserve Content Change Emotion | B 分支：保 content + 改 emotion（B2.3.1 强约束子集）| **B + C 类**（同文本同 speaker 自动满足）| 4 | ✓ |
| **B2.1.5** Speaker Identity Preservation with Cleanup | B 分支：保 speaker + cleanup 杂质 | **所有 B/C/D_st 类的 WavLM-L sim 过滤层**（不是单一 pair 类，是适用所有 B 分支类的过滤约束） | 应用于 B/C/D_st | ✓ 过滤层 |

## 5. 边界 / 待澄清的 case

### 5.1 C_mixed 在 PM 8 类里没有对应

我们的 C_mixed 类：reference = 真人 original_audio（高表现），target = 同 speaker 别条 ref_audio 经 editx 中性化的 edited_audio，**跨文本**。

边界判断：
- 不是 B 分支：跨文本，target 不是 reference 的 edited 产物
- 不是 A1.3.4.1（Emotion Transfer）：不是把 ref 的情绪迁移到 target，反而是**去除** ref 的情绪
- 最像 A1.4.1 的子任务（reference + 独立 instruction "去除情绪"），但 PM 没单列

**建议向 PM 澄清**：C_mixed 应归到 A1.4.1（"用真人 reference 给音色 + 独立 instruction 要求中性化"），还是建议 taxonomy 加一个新节点（如 A1.3.4.5 "Emotion Removal"）？

### 5.2 H2（self-reference）在 taxonomy 里没节点

H2 = 同一段 edited 中性音频既作 reference 也作 target，用于训练"什么都不要改"。taxonomy v3 里没有 self-reference 专门节点（可能 PM 当作 trivial baseline）。

**建议**：要么不当训练数据用，要么作为 A1.x 退化 case 单独处理（input = "保持原样"指令）。

### 5.3 D_st 是 B2.3.2 还是 B2.3.3？

D_st 数据是「editx 改造时 genre 改了但 emotion 没改」的子集。严格说：
- 如果把它当训练目标 = "改 genre 同时保 emotion"，对应 **B2.3.2 Genre Conversion** + **A1.3.4.x preserve emotion 约束**
- B2.3.3 Character-Performance Conversion 也可能适配（如果把 editx 改造看成"角色表演"层面）

**建议**：归 B2.3.2，加注 "with emotion preservation" 约束。

---

## 6. PM 8 类没单列但我们已有 / 应有的相关节点

### 已有
- **A1.3.1.1** Single-Reference Timbre Cloning：A 类、H1 类的真正归属
- **B2.1.5** 作为过滤层使用，不是独立类

### 还能开发的（taxonomy 里有 collectable 但我们没做）
| Taxonomy 节点 | 描述 | 实施成本 | 训练价值 |
|---------------|------|---------|---------|
| **A1.3.4.4** Emotion-Preserving Cross-Language Generation | 用 reference 情绪生成另一语种的 target | 中（需中→英 vcdata）| 高 |
| **A1.3.5.x** Accent / Dialect Reference | 用 reference 给口音/方言 | 中 | 中 |
| **A1.3.7.x** Paralinguistic Reference | 用 reference 给笑声 / 气音 / 哭腔 / 叹气 | 高（需特殊数据收集）| 中 |
| **B2.2.x** Prosody Conversion | editx 改语速 / 节奏 / 停顿 | 低（editx 可能支持）| 高（PM 提的 "Prosody Control"）|
| **B2.4.x** Accent Conversion | 改口音 / 方言 | 中 | 中 |
| **B2.5.4** Preserve Timbre but Remove Recording Artifacts | 去噪同时保音色（cleanup）| 中 | 高（PM 提的"电音"问题对接）|
| **A1.4.2** Timbre Cloning + Timbre Design | "用这个音色但年轻一点" | 高（需要 timbre 编辑能力）| 高 |
| **A1.4.3** Reference + Local Performance Instruction | "这段语调用这个 ref，但第 3 个词强调一下" | 高 | 高 |

---

## 7. 一句话总结

PM 给的 8 个 collectable 节点是 taxonomy v3 中 **A 分支 4 个 + B 分支 4 个**的精挑选；它们**层级清晰、不重叠**。我们 9 类 pair 大致 1-1 映射到 8 类中的 6 个（**A1.3.4.1 / A1.3.4.2 / A1.3.9.4 / B2.3.1 / B2.3.2 / B2.5.2**），加上 A1.4.1（范式）和 B2.1.5（过滤层）作为元级别约束。

**C_mixed 是边界 case，需要向 PM 澄清是否归 A1.4.1 或新建节点**。

未实现的 PM 8 类已基本覆盖；未来扩展可优先做 taxonomy 里的 **A1.3.4.4（情绪 + 跨语言）、B2.2.x（韵律转换）、B2.5.4（去电音保音色）、A1.4.2（音色 + 设计）**。

