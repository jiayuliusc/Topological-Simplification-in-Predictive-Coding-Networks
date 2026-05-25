"""Project path configuration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "local_config.json"


@dataclass(frozen=True)
class ProjectConfig:
    root_dir: Path
    data_dir: Path
    results_dir: Path
    pcx_dir: Path | None = None
    pcx2_dir: Path | None = None
    ripser_plusplus_dir: Path | None = None
    use_pcx2: bool = False


def _read_file_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open("r") as f:
        return json.load(f)


def _path(value: str | None) -> Path | None:
    if value in (None, ""):
        return None
    return Path(value).expanduser()


def load_config() -> ProjectConfig:
    file_config = _read_file_config()
    root_dir = _path(os.getenv("TDL_ROOT_DIR") or file_config.get("root_dir")) or PROJECT_ROOT
    data_dir = _path(os.getenv("TDL_DATA_DIR") or file_config.get("data_dir")) or root_dir / "data"
    results_dir = _path(os.getenv("TDL_RESULTS_DIR") or file_config.get("results_dir")) or root_dir / "results"
    use_pcx2_raw = os.getenv("TDL_USE_PCX2", file_config.get("use_pcx2", False))

    return ProjectConfig(
        root_dir=root_dir,
        data_dir=data_dir,
        results_dir=results_dir,
        pcx_dir=_path(os.getenv("TDL_PCX_DIR") or file_config.get("pcx_dir")),
        pcx2_dir=_path(os.getenv("TDL_PCX2_DIR") or file_config.get("pcx2_dir")),
        ripser_plusplus_dir=_path(os.getenv("TDL_RIPSER_PLUSPLUS_DIR") or file_config.get("ripser_plusplus_dir")),
        use_pcx2=str(use_pcx2_raw).lower() in {"1", "true", "yes", "on"},
    )


CONFIG = load_config()


def dataset_results_dir(dataset: str) -> Path:
    return CONFIG.results_dir / dataset
