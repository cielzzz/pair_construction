# qwen_tts_neutral — 调研结论

> 任务：验证"qwen-tts 克隆任意音频时默认输出中性化"假设  
> 时间：2026-05-28  
> 结果：**假设不成立**。pair_construction 不接入 qwen-tts，C/C-mixed 继续用 stepfun-editx。  
> 详见 [docs/WORKLOG_2026-05-27.md §13.4](../docs/WORKLOG_2026-05-27.md)

---

## 1. 官方四个变体的明确定位（引自官方文档）

来源：[QwenLM/Qwen3-TTS GitHub README](https://github.com/QwenLM/Qwen3-TTS) + 各模型的 HuggingFace 卡片

| 变体 | 官方原文 | 接 ref_audio？ | 预设音色 | 默认行为 |
|---|---|---|---|---|
| **Base** | _"Base model capable of 3-second rapid voice clone from user audio input"_ | ✅ | 无 | 保留 ref 情绪 |
| **CustomVoice** | _"Provides style control over target timbres via user instructions; supports 9 premium timbres"_ | ❌ | **9 个**：Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee | 跟 instruction；不传时倾向中性 |
| **VoiceDesign** | _"Performs voice design based on user-provided descriptions"_ | ❌ | 无（按文字描述设计新音色） | 跟描述 |
| **VoiceEditing**（未发布）| README 未列出；Discussion #231 提到 _"this will happen with the Qwen3-TTS-25Hz-1.7B-VoiceEditing model"_ | ✅ | TBD | 可编辑情绪 |

### 1.1 Base：明确说明默认保留 ref 情绪

Base 模型卡引文：

> "The Base model's emotion is **derived from the reference audio, not from separate instructions**."  
> "Default mode (`x_vector_only_mode=False`): Uses both speaker embedding AND text transcript for higher quality cloning with **better style/emotion preservation**."

Base 不支持 instruction，emotion 完全由 ref 决定。

### 1.2 CustomVoice：明确不接 ref，只能用 9 预设

CustomVoice 模型卡 + GitHub Discussion #231 互证：

> "supports 9 premium timbres"（不是 voice clone）  
> "even when passing an instruction parameter to the Base model, it doesn't do anything as instructed; it only clones the voice as it is."

Discussion 也明确：emotion-customizable cloning 要等 **`Qwen3-TTS-25Hz-1.7B-VoiceEditing`**（未发布）。

### 1.3 VoiceDesign：文字描述生成新音色，不接 ref

> "Performs voice design based on user-provided descriptions"

跟我们项目"基于 vcdata speaker 音色"的需求不匹配（它不接 ref_audio）。

---

## 2. 实测（10+3 句）

### 2.1 Base — voice clone 10 句（中5+英5）

样本：从 split_demo / split_demo_en 选最高表现力 ref_audio（P_neutral=0.000，top1=happy/angry）。

| 维度 | 结果 |
|---|---|
| 10 句 qwen Base 输出 P_neutral 均值 | ≈ **0.003**（跟 ref 0.000 一致，没中性化）|
| top1 标签 | angry / surprised / happy（保留 ref 情绪）|
| 英文短句 stability bug | `en_000001`("I'm getting so hot too.") 输出 **655 秒** 废音频；`en_000051`("Ow! Ow! Ow!") 输出 30 秒 |

**结论**：Base 是高保真克隆，**保留 ref 情绪**，不做中性化。与官方文档一致。

### 2.2 CustomVoice — Vivian 默认 3 句（无 instruct）

| wav | text | CustomVoice P_neu | top1 | sv |
|---|---|---|---|---|
| zh_044 | "要剿灭国民党的特务斗争手法呢。" | **1.000** ✅ | neutral | neutral |
| zh_056 | "那有什么关系呢？反正他早就恨死我了。" | **0.000** ❌ | surprised | angry |
| en_048 | "What is it?" | **0.000** ❌ | angry | emo_unknown |

**结论**：CustomVoice 不传 instruct 时，**只在中性语义文本下默认中性**。带情绪文本（"恨死我了" / "What is it?"）会跟着文本情感走。即"默认中性"不是绝对的。

且 CustomVoice 不接 ref_audio，音色固定 9 预设之一，**不能保留 vcdata speaker 音色**。

---

## 3. 项目最终决策

| 决策 | 理由 |
|---|---|
| ❌ 不把 qwen-tts 加进 C/C-mixed 生产 whitelist | 三个变体都没"任意 ref → 中性"能力 |
| ❌ 不用 CustomVoice 9 预设音色生成数据 | 丢失 vcdata speaker 音色，与项目目标不匹配 |
| ✅ C/C-mixed 继续用 **stepfun-editx**（zh: style_radio, en: style_chat） | 唯一能做"任意 ref → 中性"的可用方案 |
| ⏳ 等 `Qwen3-TTS-25Hz-1.7B-VoiceEditing` 发布后重测 | 唯一计划支持"clone + 情感编辑"的变体 |

---

## 4. 产物清单

```
qwen_tts_neutral/
├── README.md                          本文件
├── jobs.jsonl                          10 句源数据（中5+英5）
├── run_qwen_clone.py                   Base 推理脚本
├── run_qwen_customvoice.py             CustomVoice 推理脚本
├── run.log / run_customvoice.log       推理日志
├── outputs/                            Base 10 句产物（en_000001.wav 是 655s 废文件，可删）
└── outputs_customvoice/                CustomVoice 3 句 Vivian 默认产物
```

拉到本地：
```bash
scp -P 2222 -r root@localhost:/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/qwen_tts_neutral ~/Downloads/
```

---

## 5. 参考资料

- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
- [Qwen3-TTS-12Hz-1.7B-Base 卡片](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
- [Qwen3-TTS-12Hz-1.7B-CustomVoice 卡片](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
- [Qwen3-TTS-12Hz-1.7B-VoiceDesign 卡片](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)
- [Discussion #231: How to customize emotion in cloned voice](https://github.com/QwenLM/Qwen3-TTS/discussions/231)
