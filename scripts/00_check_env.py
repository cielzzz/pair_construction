#!/usr/bin/env python
from __future__ import annotations

import importlib
from pathlib import Path

from _common import load_config


def check_path(label: str, path: str) -> None:
    p = Path(path)
    status = "OK" if p.exists() else "MISSING"
    print(f"{label:<28} {status:<8} {p}")


def check_module(name: str) -> None:
    try:
        importlib.import_module(name)
        print(f"python module {name:<14} OK")
    except Exception as exc:
        print(f"python module {name:<14} MISSING  {type(exc).__name__}: {exc}")


def main() -> None:
    cfg = load_config()
    paths = cfg["paths"]
    step = cfg["step_audio_editx"]
    py = cfg["python"]

    print("== paths ==")
    for key in (
        "project_root",
        "existing_pair_construction",
        "vc_edit_framework",
        "vc_edit_root",
        "outputs_root",
        "default_source_jsonl",
    ):
        check_path(key, paths[key])

    print("\n== Step-Audio-EditX ==")
    for key in ("model_dir", "tokenizer_dir", "repo_dir", "run_script"):
        check_path(key, step[key])

    print("\n== python ==")
    check_path("analysis_python", py["analysis_python"])
    check_path("conda_bin", py["conda_bin"])
    print(f"step_editx_env             {py['step_editx_env']}")

    print("\n== modules in current python ==")
    for name in ("yaml", "numpy", "soundfile", "scipy", "librosa", "torch", "torchaudio"):
        check_module(name)


if __name__ == "__main__":
    main()
