"""通用工具：jsonl I/O、配置加载、id 生成、路径解析"""
from __future__ import annotations
import json
import os
import re
from pathlib import Path
from typing import Iterator, Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROW_REF_RE = re.compile(r"(?P<row>\d+)_ref\.wav$")
ROW_DIR_RE = re.compile(r"row_(?P<row>\d+)$")


def load_config(cfg_path: Path | str | None = None) -> dict:
    """优先级：显式传参 > 环境变量 PAIR_CONFIG > configs/default.yaml

    可由环境变量覆盖的路径：
      - paths.vcdata_root       <- VCDATA_ROOT
      - paths.emotion_eval_root <- EMOTION_EVAL_ROOT
      - paths.outputs_root      <- PAIR_OUTPUTS_ROOT
    """
    if cfg_path is None:
        env = os.environ.get("PAIR_CONFIG")
        cfg_path = Path(env) if env else PROJECT_ROOT / "configs" / "default.yaml"
    else:
        cfg_path = Path(cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("paths", {})
    if os.environ.get("VCDATA_ROOT"):
        cfg["paths"]["vcdata_root"] = os.environ["VCDATA_ROOT"]
    if os.environ.get("EMOTION_EVAL_ROOT"):
        cfg["paths"]["emotion_eval_root"] = os.environ["EMOTION_EVAL_ROOT"]
    if os.environ.get("PAIR_OUTPUTS_ROOT"):
        cfg["paths"]["outputs_root"] = os.environ["PAIR_OUTPUTS_ROOT"]
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


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)


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


def cache_path(cfg: dict, split: str, name: str) -> Path:
    return project_split_root(cfg, split) / ".cache" / name


def step_stamp_path(cfg: dict, split: str, step: str) -> Path:
    return project_split_root(cfg, split) / ".steps" / f"{step}.json"


def scored_pair_path(cfg: dict, split: str, name: str) -> Path:
    return pair_path(cfg, split, f"scored/{name}")


def filtered_pair_path(cfg: dict, split: str, name: str) -> Path:
    return pair_path(cfg, split, f"filtered/{name}")


def prefer_local_ref_audio(split_root: Path, row_index: int, fallback: str | None) -> str | None:
    local = split_root / "ref_audio" / f"{row_index:06d}_ref.wav"
    if local.exists():
        return str(local)
    return fallback


def prefer_local_edit_audio(
    split_root: Path,
    split: str,
    edit_tag: str,
    row_index: int,
    fallback: str | None,
) -> str | None:
    row_dir = f"row_{row_index:06d}"
    preferred_roots = [
        split_root / f"stepaudio_{edit_tag}_{split}_all",
        split_root / f"stepaudio_{edit_tag}_{split}_all_qzrun",
    ]
    candidate_patterns = [
        f"multi_engine/engine*/{edit_tag}/{row_dir}/one_stage_audio2.wav",
        f"{edit_tag}/{row_dir}/one_stage_audio2.wav",
        f"**/{edit_tag}/{row_dir}/one_stage_audio2.wav",
        f"**/{row_dir}/one_stage_audio2.wav",
    ]
    for root in preferred_roots:
        if not root.exists():
            continue
        for pat in candidate_patterns:
            matches = sorted(root.glob(pat))
            if matches:
                return str(matches[0])
    return fallback


def infer_edit_tag_from_run_dir(run_dir_name: str, split: str) -> str | None:
    prefix = "stepaudio_"
    marker = f"_{split}_"
    if not run_dir_name.startswith(prefix) or marker not in run_dir_name:
        return None
    remainder = run_dir_name[len(prefix):]
    return remainder.split(marker, 1)[0]


def build_split_audio_index(
    split_root: Path,
    split: str,
    edit_tags: Iterable[str] | None = None,
) -> dict:
    tag_filter = set(edit_tags or [])
    index = {
        "split": split,
        "split_root": str(split_root),
        "ref_audio_by_row": {},
        "edit_audio_by_tag_row": {},
        "paired_reports": {},
    }

    ref_dir = split_root / "ref_audio"
    if ref_dir.exists():
        for audio_path in sorted(ref_dir.glob("*_ref.wav")):
            match = ROW_REF_RE.match(audio_path.name)
            if not match:
                continue
            index["ref_audio_by_row"][match.group("row")] = str(audio_path)

    for run_dir in sorted(split_root.glob(f"stepaudio_*_{split}_*")):
        if not run_dir.is_dir():
            continue
        edit_tag = infer_edit_tag_from_run_dir(run_dir.name, split)
        if not edit_tag:
            continue
        if tag_filter and edit_tag not in tag_filter:
            continue

        report_path = run_dir / "paired_report.jsonl"
        if report_path.exists():
            index["paired_reports"].setdefault(edit_tag, str(report_path))

        tag_map = index["edit_audio_by_tag_row"].setdefault(edit_tag, {})
        candidate_patterns = [
            f"multi_engine/engine*/{edit_tag}/row_*/one_stage_audio2.wav",
            f"{edit_tag}/row_*/one_stage_audio2.wav",
            "**/row_*/one_stage_audio2.wav",
        ]
        for pattern in candidate_patterns:
            for audio_path in sorted(run_dir.glob(pattern)):
                row_match = ROW_DIR_RE.match(audio_path.parent.name)
                if not row_match:
                    continue
                row = row_match.group("row")
                tag_map.setdefault(row, str(audio_path))

    return index


def load_or_build_audio_index(
    cfg: dict,
    split: str,
    edit_tags: Iterable[str] | None = None,
    force_rebuild: bool = False,
) -> dict:
    index_path = intermediate_path(cfg, split, "audio_index.json")
    if not force_rebuild:
        cached = read_json(index_path)
        if cached:
            return cached
    index = build_split_audio_index(split_dir(cfg, split), split, edit_tags=edit_tags)
    write_json(index_path, index)
    return index


def resolve_ref_audio_from_index(index: dict | None, row_index: int, fallback: str | None) -> str | None:
    if index:
        resolved = index.get("ref_audio_by_row", {}).get(str(row_index))
        if resolved:
            return resolved
    return fallback


def resolve_edit_audio_from_index(
    index: dict | None,
    edit_tag: str,
    row_index: int,
    fallback: str | None,
) -> str | None:
    if index:
        resolved = index.get("edit_audio_by_tag_row", {}).get(edit_tag, {}).get(str(row_index))
        if resolved:
            return resolved
    return fallback


def base_pair_jsonl_paths(cfg: dict, split: str) -> list[Path]:
    pair_dir = pair_path(cfg, split, "")
    if not pair_dir.exists():
        return []
    rows = []
    for jsonl_path in sorted(pair_dir.glob("*.jsonl")):
        name = jsonl_path.name
        if "_filtered" in name or "_bakfilt" in name:
            continue
        rows.append(jsonl_path)
    return rows


def preferred_pair_input_path(cfg: dict, split: str, name: str) -> Path:
    scored = scored_pair_path(cfg, split, name)
    if scored.exists():
        return scored
    return pair_path(cfg, split, name)


def make_pair_id(split: str, pair_type: str, idx: int) -> str:
    return f"{split}:{pair_type}:{idx:06d}"


def make_sample_id_vc(split: str, original_idx: int) -> str:
    return f"{split}:{original_idx:06d}"


def make_sample_id_editx(split: str, edit_tag: str, source_row_index: int) -> str:
    return f"{split}:{edit_tag}:{source_row_index:06d}"
