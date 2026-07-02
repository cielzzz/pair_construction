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
NUMERIC_FIELDS = NINE_CLASSES + [
    "neutral_score",
    "non_neutral_score",
    "non_neutral",
    "top1_prob",
    "sv_is_neutral",
    "dnsmos_ovrl",
    "dnsmos_sig",
    "dnsmos_bak",
]
SV_LABELS = set(NINE_CLASSES) - {"unk"}


class EmotionTable:
    def __init__(self):
        self._by_path: dict[str, dict] = {}
        self._register_realpath = os.environ.get("EMOTION_LOOKUP_REALPATH", "1").lower() not in (
            "0",
            "false",
            "no",
        )

    # --- 通用记录注册 ----------------------------------------------------
    def _register(self, path: str, rec: dict) -> None:
        if not path:
            return
        self._by_path[path] = rec
        if not self._register_realpath:
            return
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
                for k in NUMERIC_FIELDS:
                    if k in rec and rec[k] not in ("", None):
                        try:
                            rec[k] = float(rec[k])
                        except ValueError:
                            pass
                for p in (
                    rec.get("wav") or rec.get("path"),
                    rec.get("source_audio"),
                    rec.get("audio_path"),
                    rec.get("target_audio"),
                    rec.get("original_audio"),
                    rec.get("edited_audio"),
                    rec.get("audio"),
                ):
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
                target = (
                    row.get("original_audio")
                    or row.get("edited_audio")
                    or row.get("target_audio")
                    or row.get("audio_path")
                    or row.get("audio")
                )
                if not link or not target:
                    continue
                rec = self._by_path.get(link)
                if rec is None:
                    continue
                self._register(target, rec)
                n += 1
        return n

    def load_all_link_mappings(self, emotion_root: Path) -> int:
        if not emotion_root.exists():
            return 0
        total = 0
        for csv_path in sorted(emotion_root.glob("_links*/_mapping.csv")):
            total += self.load_link_mapping(csv_path)
        return total

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
        if not self._register_realpath:
            return None
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
                    "dnsmos_ovrl": None, "dnsmos_sig": None,
                    "dnsmos_bak": None}
        return {
            "top1_label": rec.get("top1_label"),
            "top1_prob": _optional_float(rec.get("top1_prob")),
            "P_neutral": _optional_float(rec.get("neutral")),
            "sv_label": _sensevoice_label(rec),
            "dnsmos_ovrl": _optional_float(rec.get("dnsmos_ovrl")),
            "dnsmos_sig": _optional_float(rec.get("dnsmos_sig")),
            "dnsmos_bak": _optional_float(rec.get("dnsmos_bak")),
        }

    def dnsmos_value(self, audio_path: str, key: str):
        rec = self.get(audio_path)
        if rec is None:
            return None
        v = rec.get(key)
        if v in ("", None):
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def dnsmos_ovrl(self, audio_path: str):
        return self.dnsmos_value(audio_path, "dnsmos_ovrl")

    def dnsmos_sig(self, audio_path: str):
        return self.dnsmos_value(audio_path, "dnsmos_sig")

    def dnsmos_bak(self, audio_path: str):
        return self.dnsmos_value(audio_path, "dnsmos_bak")


def _safe_float(x) -> float:
    if x is None or x == "":
        return 0.0
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _optional_float(x) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _sensevoice_label(rec: dict) -> str | None:
    label = rec.get("sv_label")
    if isinstance(label, str) and label.strip():
        return label.strip().lower()
    raw = rec.get("sv_raw")
    if not isinstance(raw, str):
        return None
    for token in raw.split("<|"):
        tag = token.split("|>", 1)[0].strip().lower()
        if tag in SV_LABELS:
            return tag
    return None
