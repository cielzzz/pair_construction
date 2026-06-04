# IndexTTS-2 vs MOSS-TTS 公平对比

任务：`spk_prompt = original_audio`, `text = ref_text`（与 MOSS-TTS 在 vcdata 阶段做的事一致）

指标：
- **speaker_sim**：CAM++ cosine(output, original_audio)，越大音色越接近原 speaker
- **PESQ / STOI / SI-SDR**：torchaudio SQUIM_OBJECTIVE，reference-free 客观音频质量分（PESQ 1~4.5 越高越好）
- **P_neu**：emotion2vec 输出 P(neutral)，仅参考

## 中文 split_demo（n=10）

| 指标 | MOSS-TTS | IndexTTS-2 | 优势方 |
|---|---:|---:|---|
| **speaker_sim ↑** | 0.735 | 0.792 | IndexTTS-2 |
| **PESQ ↑** | 3.11 | 2.99 | MOSS-TTS |
| **STOI ↑** | 0.973 | 0.953 | MOSS-TTS |
| **SI-SDR ↑** | 19.34 | 19.76 | IndexTTS-2 |
| P_neu | 0.397 | 0.423 | - |

## 英文 split_demo_en（n=10）

| 指标 | MOSS-TTS | IndexTTS-2 | 优势方 |
|---|---:|---:|---|
| **speaker_sim ↑** | 0.466 | 0.655 | IndexTTS-2 |
| **PESQ ↑** | 2.39 | 2.13 | MOSS-TTS |
| **STOI ↑** | 0.775 | 0.806 | IndexTTS-2 |
| **SI-SDR ↑** | 5.63 | 9.06 | IndexTTS-2 |
| P_neu | 0.018 | 0.001 | - |
