"""通用工具：jsonl I/O、配置加载、id 生成、路径解析"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Iterator, Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(cfg_path: Path | str | None = None) -> dict:
    """优先级：显式传参 > 环境变量 PAIR_CONFIG > configs/default.yaml

    paths.vcdata_root 可由环境变量 VCDATA_ROOT 覆盖（用于 from_vcdata 模式
    切换不同上游数据源，如自己跑的 xyzhang 输出 vs kxhuang 已跑好的输出）。
    """
    import os
    if cfg_path is None:
        env = os.environ.get("PAIR_CONFIG")
        cfg_path = Path(env) if env else PROJECT_ROOT / "configs" / "default.yaml"
    else:
        cfg_path = Path(cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # env 覆盖 vcdata_root（保留其它 paths 不变）
    vc_root = os.environ.get("VCDATA_ROOT")
    if vc_root:
        cfg.setdefault("paths", {})["vcdata_root"] = vc_root
    return cfg


def iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            yield json.loads(ln)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def split_idx_of(split_name: str) -> int:
    m = re.match(r"split_(\d+)$", split_name)
    if not m:
        raise ValueError(f"bad split name: {split_name}")
    return int(m.group(1))


def split_dir(cfg: dict, split: str) -> Path:
    return Path(cfg["paths"]["vcdata_root"]) / split


def project_split_root(cfg: dict, split: str) -> Path:
    return Path(cfg["paths"]["outputs_root"]) / split


def intermediate_path(cfg: dict, split: str, name: str) -> Path:
    return project_split_root(cfg, split) / "intermediate" / name


def emotion_path(cfg: dict, split: str, name: str) -> Path:
    return project_split_root(cfg, split) / "emotion" / name


def pair_path(cfg: dict, split: str, name: str) -> Path:
    return project_split_root(cfg, split) / "pairs" / name


def make_pair_id(split: str, pair_type: str, idx: int) -> str:
    return f"{split}:{pair_type}:{idx:06d}"


def make_sample_id_vc(split: str, original_idx: int) -> str:
    return f"{split}:{original_idx:06d}"


def make_sample_id_editx(split: str, edit_tag: str, source_row_index: int) -> str:
    return f"{split}:{edit_tag}:{source_row_index:06d}"
