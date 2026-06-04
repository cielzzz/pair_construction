# pair_construction 分类体系审视与重组建议

整理日期：2026-05-29
背景：PM 给了 8 个分类（A1.4.1 / B2.3.1 / B2.3.2 / B2.5.2 / A1.3.4.1 / A1.3.4.2 / A1.3.9.4 / B2.1.5）；本文档梳理这 8 个分类的逻辑问题，补充漏掉的 7 个训练信号，并给出重组方案。

---

## 0. TL;DR

PM 8 个分类存在 3 类问题：
1. 4 个不同层级（范式 / 任务 / 约束 / 过滤）被拍扁到一维，颗粒度不一
2. 3 对内容明显重叠（B2.3.1 vs B2.5.2；A1.3.4.1 vs A1.3.4.2；B2.3.1 vs B2.3.2）
3. B2.3.2 Genre Conversion 在我们当前工程实践里和 B2.3.1 Emotion Conversion 无法切分

漏掉的 7 个重要训练信号：N1 一对多 / N2 跨情绪保音色 / N3 多属性控制 / N4 Self-Reference / N5 韵律控制 / N6 长短不对称 / N7 强度梯度。

建议方案：保留 PM 8 类但拆分重叠 + 把范式/过滤层抽出来 + 新增 7 项。

---

## 1. PM 分类的 3 个问题

### 1.1 问题 1：4 个不同层级混在一维

| 层级 | 含义 | PM 8 类中对应 |
|------|------|---------------|
| 范式层 | 模型 IO 接口形态（reference + instruction → target） | A1.4.1 |
| 任务层 | 模型要学的具体能力 | B2.3.1 / B2.3.2 / B2.5.2 / A1.3.4.1 / A1.3.4.2 |
| 约束层 | 数据 pair 的物理性质（同/跨文本，同/跨 speaker） | （隐含在描述里）|
| 过滤层 | 用哪些指标卡数据质量 | A1.3.9.4 / B2.1.5 |

**问题**：A1.4.1 是范式（所有 pair 都满足），B2.1.5 是过滤层（应用于其他类之上），二者不该与具体数据类别并列。

### 1.2 问题 2：3 对重叠

| 重叠的两类 | 重叠点 | 细微区别 | 实操判断 |
|-----------|--------|---------|---------|
| **B2.3.1 vs B2.5.2** | 都是"编辑情绪" | B2.5.2 额外要求保留文本+说话人；B2.3.1 没明说 | 我们 B/C 类同时满足两者，**B2.5.2 是 B2.3.1 的子集**，没必要并列 |
| **A1.3.4.1 vs A1.3.4.2** | 都是"情绪相关的 target 约束" | A1.3.4.1 = 同类别+同强度（严约束）；A1.3.4.2 = 强度对齐可跨类别（松约束） | **A1.3.4.1 是 A1.3.4.2 的子集**，二者应该是同一分类的两个严格度档位 |
| **B2.3.1 vs B2.3.2** | 都用 editx 改造 | B2.3.1 = 改情绪；B2.3.2 = 改风格 | 我们 zh 用 style_radio 做"中性化"，实际是 Genre Conversion 顺带产生 Emotion Conversion 效果，**工程上不可切分**（见 §1.3）|

### 1.3 问题 3：B2.3.2 Genre Conversion 的歧义（关键）

editx 6 个 tag 真实语义分两类：

| editx tag | 真实意图 | 副作用 |
|-----------|----------|--------|
| `style_remove` / `emotion_remove` | 真正的去除情绪/风格（emotion neutralization） | 输出偏中性平淡 |
| `style_radio` / `style_news` / `style_chat` | Genre Conversion（转电台/新闻/聊天播报风格） | 这几种播报风格本身就平淡，**顺带让情绪也变中性** |
| `emotion_coldness` | Emotion Conversion（转冷漠/疏离） | 不去除情绪，只换情绪 |

我们 zh 用 `style_radio` 作为 B/C 类的中性化方案，**严格说是用 Genre Conversion 实现了 Emotion Conversion 的效果**。

如果要做"纯 Genre Conversion"（不带情绪变化）：
- ref = style_radio_edited，tgt = style_news_edited（两个都是非中性 genre 互转）
- 当前未实现

---

## 2. PM 漏掉的 7 个重要训练信号

| 编号 | 名字 | 物理含义 | 训练价值 | 当前状态 |
|------|------|----------|----------|----------|
| **N1** | One-to-Many（一对多） | 同一段 ref + N 个不同 instruction → N 个不同 target | 模型学到 instruction 的真正作用（避免死记 ref→tgt 映射） | 现有 pair N=1，未实现 |
| **N2** | Cross-Emotion Same-Speaker | ref top1=angry, tgt top1=happy，双高表现 + sim 高 | 学"保留音色但改变情绪类别"，比同情绪更稀缺更难 | D 类 same_top1=true 直接砍掉，未实现 |
| **N3** | Multi-Attribute Control | instruction "用愤怒+电台风格+稍快" → tgt 同时满足多属性 | 多维度控制能力 | instruction_pool 单维度，未实现 |
| **N4** | Self-Reference / Identity | ref == tgt（完全不变换） | 学"什么都不改"，避免对所有输入都加工 | 部分实现（H1 / H2） |
| **N5** | Prosody Control | instruction "速度更慢" / "音调更高" | 控制非情绪的可调参数 | 未实现 |
| **N6** | Long-Reference Short-Target / 反向 | ref 30s 长段, tgt 3s 短段（或反向） | 长程一致性 + 内容截取 | ref/tgt 时长接近，未实现 |
| **N7** | Conditional Style Strength | instruction "稍微开心一点" vs "非常开心" | 学情感强度的"程度词"控制 | 当前 binary 高/中性，未实现 |

---

## 3. 重组方案 A：按 4 层级重构（彻底）

```
范式层  : 所有 pair 默认遵循（不单列）
          - reference + instruction → target

任务层  : T1 Emotion Conversion (B↔C 双向, B2.3.1)
          T2 Emotion Transfer (target 继承 ref 情绪, A1.3.4.1)
          T3 Genre Conversion (风格互转, B2.3.2)
          T4 Cross-Emotion Identity (跨情绪保音色, N2)
          T5 One-to-Many (一对多, N1)
          T6 Self-Reference (不变换, N4)
          T7 Prosody Control (N5)
          T8 Style Strength Gradient (N7)
          T9 Multi-Attribute (N3)

约束层  : C1 同文本 / C2 跨文本
          C3 同 speaker / C4 跨 speaker
          C5 同时长 / C6 长短不对称 (N6)

过滤层  : F1 WavLM-L sim ≥ X (= B2.1.5)
          F2 WER 一致性（防 text leak，部分对应 A1.3.9.4）
          F3 DNSMOS 阈值（防电音）
          F4 跨 speaker 负样本采样
```

每个 pair 类 = (任务 × 约束) + 适用过滤

**优点**：清晰，没有重叠
**缺点**：需要 PM 接受新命名体系

---

## 4. 重组方案 B：保留 PM 8 类 + 轻量修订（推荐）

| PM 原分类 | 修订建议 |
|-----------|---------|
| **A1.4.1** | 标为"范式层"，不当数据类别 |
| **B2.3.1 Emotion Conversion** | 拆为 **B2.3.1a 中性→高表现** + **B2.3.1b 高表现→中性**（双向不应合并） |
| **B2.3.2 Genre Conversion** | 注明：严格定义需要 genre A↔genre B 互转，与 B2.3.1 在工程上难切分；当前 zh 用 style_radio 实际是混合 |
| **B2.5.2 Preserve Content Change Emotion** | 合并到 B2.3.1（是其"同 speaker 同文本"子集，没必要单列） |
| **A1.3.4.1 Emotion Transfer** | 拆为 **A1.3.4.1a 同情绪同文本** + **A1.3.4.1b 同情绪跨文本** |
| **A1.3.4.2 Affect Intensity Matching** | 改名"跨情绪强度对齐"，明确允许 top1 不同但 P_neu 都低（对应 N2）|
| **A1.3.9.4** | 标为"过滤层"（负样本采样方法），不当数据类别 |
| **B2.1.5** | 标为"过滤层"（speaker_sim 阈值），不当数据类别 |
| **+ 新增 N1~N7** | 漏掉的 7 项训练信号 |

**优点**：尊重 PM 原命名 + 补全
**缺点**：拆分后类别变多

---

## 5. 当前 9 类 pair 在新体系下的归类

| 我的现有类 | 严格归类（方案 B）| 训练价值 |
|-----------|------------------|----------|
| A | 范式 baseline（A1.4.1）| 低 |
| **B** | **B2.3.1a**（中→高 同文本）| 高（PM 优先级 1）|
| **C** | **B2.3.1b**（高→中 同文本）| 高 |
| **C_mixed** | **B2.3.1b**（高→中 跨文本）+ 真人 ref | 中 |
| **D** | **A1.3.4.1b**（同情绪 跨文本）| 高 |
| **D_st** | **A1.3.4.1a**（同情绪 同文本）| 中 |
| H1 | **N4** Self-Reference（保持原样） | 低 |
| H2 | **N4** Self-Reference（中性 self pool）| 低 |
| H3 | 过滤层 F4（跨 speaker 负样本）+ A1.3.9.4 防 leak | 低 |
| **D_cross_emo**（待加）| **A1.3.4.2 / N2**（跨情绪强度对齐）| 高 ← 未实现 |
| **B_one2many**（待加）| **N1**（同 ref 多 instruction）| 高 ← 未实现 |
| **Genre_pair**（待加）| **B2.3.2**（真 genre 互转）| 中 ← 未实现 |
| **B_strength**（待加）| **N7**（情感强度梯度）| 中 ← 未实现 |
| **B_prosody**（待加）| **N5**（语速/音高控制）| 中 ← 未实现 |

---

## 6. 当前 100 句留存（zh，WavLM-L ≥ 0.80）

| 类别 | n | 推 10000 句估算 | 备注 |
|------|---|-----------------|------|
| A | 100 | ~10000 | baseline |
| B (B2.3.1a) | **2** | ~200 | 0.80 阈值后量极少 |
| C (B2.3.1b) | **2** | ~200 | 同上 |
| C_mixed (B2.3.1b 跨文本) | 43 | ~4300 | 不卡 sim |
| D (A1.3.4.1b) | 19 | ~1900 | 不卡 sim |
| D_st (A1.3.4.1a) | **3** | ~300 | 同源派生 |
| H1 (N4) | 46 | ~4600 | 保持原样 |
| H2 (N4) | 72 | ~7200 | 中性 self pool |
| H3 (F4) | 100 | ~10000 | 跨 speaker 负样本 |

---

## 7. 周一对齐建议

### 7.1 需 PM 拍板的分类体系问题

1. 是否接受方案 B（保留 8 类 + 拆分 + 补 N1~N7）？还是方案 A（彻底按层级重构）？
2. B2.3.1 是否拆为 B2.3.1a/b 双向？
3. A1.3.4.1 是否拆为 A1.3.4.1a/b 同/跨文本？
4. A1.3.4.2 是否改名为"跨情绪强度对齐"，并明确允许 top1 不同？
5. B2.3.2 Genre Conversion 是否单独立项做（需要 editx 跑全 6 tag + 配对）？

### 7.2 漏掉的 N1~N7 优先级

| 编号 | 实施成本 | 训练价值 | 建议优先级 |
|------|---------|---------|-----------|
| **N1** One-to-Many | 低（同 ref 多 instruction 配对）| 高 | **优先** |
| **N2** Cross-Emotion Same-Speaker | 低（改 D 类配置加 cross_emo 标志）| 高 | **优先** |
| N3 Multi-Attribute | 中（instruction 模板 + caption 模型）| 中 | 等 caption 模型 |
| N4 Self-Reference | 已有 H1/H2，无新工作 | 低 | 已实现 |
| N5 Prosody Control | 高（需要专门 prosody-aware TTS）| 中 | 后排 |
| N6 Long/Short Asymmetry | 低（取消时长约束 + 多采样）| 中 | 接 maoer 长样本后做 |
| N7 Strength Gradient | 中（需要标注强度等级）| 中 | 接 caption 模型后做 |

### 7.3 一句话总结

> PM 8 个分类是好起点，但需要：拆分重叠的 3 对、把范式/过滤层抽出来不当数据类别、补全漏掉的 7 项训练信号。当前我们已实现 6/8 PM 类别 + 1 项 N4，**优先补 N1（一对多）和 N2（跨情绪保音色）**，这两个是 PM 体系空白且训练价值最高。

