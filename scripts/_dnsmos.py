"""Microsoft DNSMOS P.835 (sig_bak_ovr.onnx) 精简推理。

参考 https://github.com/microsoft/DNS-Challenge/blob/master/DNSMOS/dnsmos_local.py
保留核心逻辑：raw audio 16kHz → 9.01s 滑窗 → 每段 3 分数 → 平均。
省略 P808 模型和 polynomial calibration（用 raw 分数即可，跟 vcdata 阶段口径对齐）。
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

import numpy as np

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01

_SESSION = None
_INPUT_NAME = None


def _load_session(onnx_path: str):
    global _SESSION, _INPUT_NAME
    if _SESSION is not None:
        return _SESSION, _INPUT_NAME
    import onnxruntime as ort
    sess_options = ort.SessionOptions()
    threads = os.environ.get("DNSMOS_ORT_THREADS") or os.environ.get("ORT_INTRA_OP_NUM_THREADS")
    if threads:
        try:
            n_threads = int(threads)
            if n_threads > 0:
                sess_options.intra_op_num_threads = n_threads
                sess_options.inter_op_num_threads = 1
        except ValueError:
            pass
    providers = []
    try:
        providers.append("CUDAExecutionProvider")
    except Exception:
        pass
    providers.append("CPUExecutionProvider")
    _SESSION = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
    _INPUT_NAME = _SESSION.get_inputs()[0].name
    return _SESSION, _INPUT_NAME


def compute_dnsmos(audio_path: str, onnx_path: Optional[str] = None) -> Optional[dict]:
    """返回 {'OVRL': float, 'SIG': float, 'BAK': float, 'n_segments': int} 或 None。

    OVRL = 综合 MOS（1~5，越高越好），与 vcdata 原始 jsonl 里的 dnsmos 字段同口径（raw 分数）。
    """
    import librosa
    onnx_path = onnx_path or os.environ.get("DNSMOS_ONNX") or (
        "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/"
        "projects/pair_construction/tts_baselines/_dnsmos/sig_bak_ovr.onnx"
    )
    if not Path(onnx_path).exists():
        return None
    try:
        audio, sr = librosa.load(audio_path, sr=SAMPLING_RATE)
    except Exception:
        return None
    if audio.size == 0:
        return None

    sess, in_name = _load_session(onnx_path)
    len_samples = int(INPUT_LENGTH * SAMPLING_RATE)   # 144160

    # 太短：repeat-pad 到 9.01s
    if len(audio) < len_samples:
        n_rep = int(np.ceil(len_samples / max(1, len(audio))))
        audio = np.tile(audio, n_rep)[:len_samples]

    # 滑窗 9.01s, hop 1s
    hop = SAMPLING_RATE  # 1s
    n_segs = max(1, int(np.floor(len(audio) / hop) - int(INPUT_LENGTH) + 1))
    sig_s, bak_s, ovr_s = [], [], []
    for i in range(n_segs):
        start = i * hop
        seg = audio[start:start + len_samples]
        if len(seg) < len_samples:
            break
        seg = np.asarray(seg, dtype=np.float32).reshape(1, -1)
        out = sess.run(None, {in_name: seg})[0]
        sig_s.append(float(out[0][0]))
        bak_s.append(float(out[0][1]))
        ovr_s.append(float(out[0][2]))

    if not ovr_s:
        return None
    return {
        "OVRL": round(float(np.mean(ovr_s)), 3),
        "SIG":  round(float(np.mean(sig_s)), 3),
        "BAK":  round(float(np.mean(bak_s)), 3),
        "n_segments": len(ovr_s),
    }


if __name__ == "__main__":
    import sys, json
    for p in sys.argv[1:]:
        print(p, json.dumps(compute_dnsmos(p), ensure_ascii=False))
