"""加载 emotion CSV 并提供按音频路径的查询。

支持两种来源（同时载入也可）：
1. per_file_dual.csv —— 由 04 合并产出，path/wav 列做 key
2. per_pair.csv      —— qzrun pipeline 的对比表，src/out 都是路径，
                         src_neu/edt_neu 是 P_neutral，src_top1/edt_top1 是 top1
                         用它能补上 ref_audio 的 emotion（reuse-qzrun 模式必需）
"""
from __future__ import annotations
import csv
import math
import os
from pathlib import Path
from typing import Optional

NINE_CLASSES = ["angry", "disgusted", "fearful", "happy", "neutral",
                "other", "sad", "surprised", "unk"]


class EmotionTable:
    def __init__(self):
        self._by_path: dict[str, dict] = {}

    # --- 通用记录注册 ----------------------------------------------------
    def _register(self, path: str, rec: dict) -> None:
        if not path:
            return
        self._by_path[path] = rec
        try:
            real = os.path.realpath(path)
            if real != path:
                self._by_path.setdefault(real, rec)
        except OSError:
            pass

    # --- 加载主表 per_file_dual.csv -------------------------------------
    def load_csv(self, csv_path: Path) -> int:
        if not csv_path.exists():
            return 0
        n = 0
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rec = dict(row)
                for k in NINE_CLASSES + ["neutral_score", "non_neutral_score",
                                         "non_neutral", "top1_prob"]:
                    if k in rec and rec[k] not in ("", None):
                        try:
                            rec[k] = float(rec[k])
                        except ValueError:
                            pass
                p = rec.get("wav") or rec.get("path")
                self._register(p, rec)
                n += 1
        return n

    # --- 加载 _links_original/_mapping.csv：把 link_wav 的 rec 别名到 original_audio
    def load_link_mapping(self, csv_path: Path) -> int:
        if not csv_path.exists():
            return 0
        n = 0
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                link = row.get("link_wav")
                orig = row.get("original_audio")
                if not link or not orig:
                    continue
                rec = self._by_path.get(link)
                if rec is None:
                    continue
                self._register(orig, rec)
                n += 1
        return n

    # --- 加载 per_pair.csv，把 src 注册成简化记录 ----------------------
    def load_per_pair_for_src(self, csv_path: Path) -> int:
        """per_pair.csv 列：group,row,src,out,src_neu,edt_neu,delta_neu,src_top1,edt_top1,..."""
        if not csv_path.exists():
            return 0
        n = 0
        seen = set()
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                src = row.get("src")
                if not src or src in seen:
                    continue
                seen.add(src)
                try:
                    p_neu = float(row.get("src_neu") or 0.0)
                except ValueError:
                    p_neu = 0.0
                rec = {
                    "wav": src,
                    "neutral": p_neu,
                    "top1_label": row.get("src_top1"),
                    "top1_prob": None,
                    "sv_label": None,
                    "_source": "per_pair.src",
                }
                if src not in self._by_path:
                    self._register(src, rec)
                    n += 1
        return n

    # --- 查询 ------------------------------------------------------------
    def get(self, audio_path: str) -> Optional[dict]:
        if not audio_path:
            return None
        rec = self._by_path.get(audio_path)
        if rec:
            return rec
        try:
            return self._by_path.get(os.path.realpath(audio_path))
        except OSError:
            return None

    @staticmethod
    def cosine9(a: dict | None, b: dict | None) -> Optional[float]:
        if a is None or b is None:
            return None
        va = [_safe_float(a.get(k)) for k in NINE_CLASSES]
        vb = [_safe_float(b.get(k)) for k in NINE_CLASSES]
        na = math.sqrt(sum(x * x for x in va))
        nb = math.sqrt(sum(x * x for x in vb))
        if na == 0 or nb == 0:
            return None
        dot = sum(x * y for x, y in zip(va, vb))
        return dot / (na * nb)

    def emotion_summary(self, audio_path: str) -> dict:
        rec = self.get(audio_path)
        if rec is None:
            return {"top1_label": None, "top1_prob": None,
                    "P_neutral": None, "sv_label": None,
                    "dnsmos_ovrl": None}
        return {
            "top1_label": rec.get("top1_label"),
            "top1_prob": _safe_float(rec.get("top1_prob")),
            "P_neutral": _safe_float(rec.get("neutral")),
            "sv_label": rec.get("sv_label"),
            "dnsmos_ovrl": _safe_float(rec.get("dnsmos_ovrl")) if rec.get("dnsmos_ovrl") not in ("", None) else None,
        }

    def dnsmos_ovrl(self, audio_path: str):
        rec = self.get(audio_path)
        if rec is None:
            return None
        v = rec.get("dnsmos_ovrl")
        if v in ("", None):
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None


def _safe_float(x) -> float:
    if x is None or x == "":
        return 0.0
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0
