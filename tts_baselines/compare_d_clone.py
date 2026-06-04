#!/usr/bin/env python
"""IndexTTS-2 vs MOSS-TTS 公平对比（中10+英10）

任务：spk_prompt = original_audio + text = ref_text
- MOSS-TTS 输出 = vcdata 已生产的 ref_audio
- IndexTTS-2 输出 = tts_baselines/indextts2/outputs_fair/<tag>.wav

三个指标：
  1. speaker_sim = CAM++ cosine(output, original_audio)   越高音色越像
  2. emotion: top1 label / P_neutral（emotion2vec + sv）
  3. SQUIM_OBJECTIVE: PESQ (越高越好) + STOI (越高越好) + SI-SDR (作 quality 代理)
     用 torchaudio 自带；DNSMOS 同类指标，reference-free 客观分。

输出：
  tts_baselines/d_clone_compare/per_pair.csv
  tts_baselines/d_clone_compare/summary.md
"""
import json, csv, re, sys
from pathlib import Path
import numpy as np
import torch
import torchaudio
import soundfile as sf

PC = Path('/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction')
JOBS = PC / 'tts_baselines/jobs_fair_compare.jsonl'
INDEX_OUT = PC / 'tts_baselines/indextts2/outputs_fair'
REPORT_DIR = PC / 'tts_baselines/d_clone_compare'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading evaluation models ...")
from funasr import AutoModel
emo = AutoModel(model='iic/emotion2vec_plus_large', hub='ms', device='cuda:0', disable_update=True)
sv  = AutoModel(model='iic/SenseVoiceSmall', hub='ms', trust_remote_code=True, device='cuda:0', disable_update=True)
cam = AutoModel(model='iic/speech_campplus_sv_zh-cn_16k-common', hub='ms', device='cuda:0', disable_update=True)

# SQUIM (PyTorch 原生客观音频质量评分；输出 STOI / PESQ / SI-SDR)
from torchaudio.pipelines import SQUIM_OBJECTIVE
squim = SQUIM_OBJECTIVE.get_model().to('cuda:0').eval()
SQUIM_SR = SQUIM_OBJECTIVE.sample_rate   # 16000
print("All models ready.")


def norm_label(raw):
    m = re.search(r'/([a-zA-Z<>]+)', raw)
    return m.group(1).lower() if m else raw


def emo_summary(wav_path):
    r = emo.generate(str(wav_path), granularity='utterance', extract_embedding=False, disable_pbar=True)[0]
    labels = [norm_label(x) for x in r['labels']]
    scores = list(map(float, r['scores']))
    p = dict(zip(labels, scores))
    top1_i = scores.index(max(scores))
    sv_r = sv.generate(input=str(wav_path), language='auto', use_itn=False, ban_emo_unk=False, disable_pbar=True)
    m = re.search(r'<\|([A-Z_]+)\|>', sv_r[0]['text'] if sv_r else '')
    return {
        'top1': labels[top1_i],
        'P_neu': round(p.get('neutral', 0), 3),
        'sv': m.group(1).lower() if m else None,
    }


def emb(path):
    r = cam.generate(input=str(path), disable_pbar=True)
    e = r[0]['spk_embedding']
    if isinstance(e, torch.Tensor): e = e.detach().cpu().numpy().flatten()
    return np.asarray(e).flatten()


def cam_sim(p1, p2):
    e1, e2 = emb(p1), emb(p2)
    return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-9))


def squim_score(wav_path):
    """返回 (STOI, PESQ, SI-SDR)，越高越好。"""
    wav, sr = torchaudio.load(str(wav_path))
    if wav.shape[0] > 1: wav = wav[:1]
    if sr != SQUIM_SR:
        wav = torchaudio.functional.resample(wav, sr, SQUIM_SR)
    with torch.no_grad():
        stoi, pesq, si_sdr = squim(wav.to('cuda:0'))
    return float(stoi.item()), float(pesq.item()), float(si_sdr.item())


# ── 跑评估 ──
rows = []
for line in JOBS.open():
    j = json.loads(line)
    tag = j['tag']
    moss_wav = j['moss_output']          # vcdata 已有的 ref_audio
    idx_wav = INDEX_OUT / f'{tag}.wav'
    if not idx_wav.exists():
        print(f'[skip] no IndexTTS-2 output for {tag}')
        continue
    print(f'─ {tag}')
    rec = {'tag': tag, 'lang': j['lang']}
    # 原音色参考
    orig_audio = j['original_audio']
    for system, wav in [('moss', moss_wav), ('indextts2', idx_wav)]:
        try:
            sim = cam_sim(orig_audio, wav)
            es = emo_summary(wav)
            stoi, pesq, sisdr = squim_score(wav)
        except Exception as e:
            print(f'  {system} eval failed: {e}')
            sim, es, stoi, pesq, sisdr = None, {'top1':None,'P_neu':None,'sv':None}, None, None, None
        rec[f'{system}_sim'] = sim
        rec[f'{system}_top1'] = es['top1']
        rec[f'{system}_P_neu'] = es['P_neu']
        rec[f'{system}_sv'] = es['sv']
        rec[f'{system}_stoi'] = stoi
        rec[f'{system}_pesq'] = pesq
        rec[f'{system}_sisdr'] = sisdr
    rows.append(rec)
    print(f'  moss      sim={rec["moss_sim"]:.3f} top1={rec["moss_top1"]:<10} PESQ={rec["moss_pesq"]:.2f} STOI={rec["moss_stoi"]:.2f}')
    print(f'  indextts2 sim={rec["indextts2_sim"]:.3f} top1={rec["indextts2_top1"]:<10} PESQ={rec["indextts2_pesq"]:.2f} STOI={rec["indextts2_stoi"]:.2f}')


# ── 写 per-pair CSV ──
csv_path = REPORT_DIR / 'per_pair.csv'
with csv_path.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f'\n→ {csv_path}')


# ── 汇总 markdown ──
def mean(vals): return sum(v for v in vals if v is not None) / max(1, len([v for v in vals if v is not None]))

def block(rs):
    return {
        'n': len(rs),
        'moss_sim': mean([r['moss_sim'] for r in rs]),
        'idx_sim':  mean([r['indextts2_sim'] for r in rs]),
        'moss_pesq': mean([r['moss_pesq'] for r in rs]),
        'idx_pesq':  mean([r['indextts2_pesq'] for r in rs]),
        'moss_stoi': mean([r['moss_stoi'] for r in rs]),
        'idx_stoi':  mean([r['indextts2_stoi'] for r in rs]),
        'moss_sisdr': mean([r['moss_sisdr'] for r in rs]),
        'idx_sisdr':  mean([r['indextts2_sisdr'] for r in rs]),
        'moss_neu':  mean([r['moss_P_neu'] for r in rs]),
        'idx_neu':   mean([r['indextts2_P_neu'] for r in rs]),
    }

lines = ['# IndexTTS-2 vs MOSS-TTS 公平对比', '',
         '任务：`spk_prompt = original_audio`, `text = ref_text`（与 MOSS-TTS 在 vcdata 阶段做的事一致）', '',
         '指标：',
         '- **speaker_sim**：CAM++ cosine(output, original_audio)，越大音色越接近原 speaker',
         '- **PESQ / STOI / SI-SDR**：torchaudio SQUIM_OBJECTIVE，reference-free 客观音频质量分（PESQ 1~4.5 越高越好）',
         '- **P_neu**：emotion2vec 输出 P(neutral)，仅参考', '']

for lang, name in [('zh', '中文 split_demo'), ('en', '英文 split_demo_en')]:
    rs = [r for r in rows if r['lang'] == lang]
    if not rs: continue
    b = block(rs)
    lines += [f'## {name}（n={b["n"]}）', '',
              '| 指标 | MOSS-TTS | IndexTTS-2 | 优势方 |',
              '|---|---:|---:|---|',
              f'| **speaker_sim ↑** | {b["moss_sim"]:.3f} | {b["idx_sim"]:.3f} | {"IndexTTS-2" if b["idx_sim"]>b["moss_sim"] else "MOSS-TTS"} |',
              f'| **PESQ ↑** | {b["moss_pesq"]:.2f} | {b["idx_pesq"]:.2f} | {"IndexTTS-2" if b["idx_pesq"]>b["moss_pesq"] else "MOSS-TTS"} |',
              f'| **STOI ↑** | {b["moss_stoi"]:.3f} | {b["idx_stoi"]:.3f} | {"IndexTTS-2" if b["idx_stoi"]>b["moss_stoi"] else "MOSS-TTS"} |',
              f'| **SI-SDR ↑** | {b["moss_sisdr"]:.2f} | {b["idx_sisdr"]:.2f} | {"IndexTTS-2" if b["idx_sisdr"]>b["moss_sisdr"] else "MOSS-TTS"} |',
              f'| P_neu | {b["moss_neu"]:.3f} | {b["idx_neu"]:.3f} | - |', '']

md = REPORT_DIR / 'summary.md'
md.write_text('\n'.join(lines), encoding='utf-8')
print(f'→ {md}')
print()
print('\n'.join(lines))
