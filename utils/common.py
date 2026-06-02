import csv
import datetime
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import yaml

from utils.io import ensure_outdir, save_text

# Default config path for load_config() when no path is provided
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


def _resolve_config_path(path: str | None) -> Path:
    if path is None:
        return _DEFAULT_CONFIG_PATH
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    return config_path


def load_config(path: str | None = None) -> Tuple[Dict[str, Any], Path]:
    config_path = _resolve_config_path(path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg, config_path.parent


def read_prompts(path: str, base_dir: Path) -> List[str]:
    prompt_path = Path(path)
    if not prompt_path.is_absolute():
        prompt_path = base_dir / prompt_path
    with open(prompt_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines


def prepare_run_directory(output_root: str, base_dir: Path, run_dir_name: str | None = None) -> str:
    output_path = Path(output_root)
    if not output_path.is_absolute():
        output_path = base_dir / output_path
    if run_dir_name is None:
        run_dir_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_outdir(os.path.join(str(output_path), run_dir_name))
    return run_dir


def dump_run_config(cfg: Dict[str, Any], run_dir: str) -> None:
    serializable = deepcopy(cfg)
    serializable.pop("_config_dir", None)
    save_text(json.dumps(serializable, indent=2, ensure_ascii=False), os.path.join(run_dir, "config.json"))


def iter_prompt_seed_combinations(prompts: Iterable[str], seeds: Iterable[int]) -> Iterator[Tuple[str, int]]:
    for prompt in prompts:
        for seed in seeds:
            yield prompt, seed


def resolve_path(path: str | Path, base_dir: Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = base_dir / p
    return p


def _has_private_sample_dirs(root: Path) -> bool:
    if not root.exists() or not root.is_dir():
        return False

    for child in root.iterdir():
        if not child.is_dir():
            continue
        if child.name.isdigit():
            return True
        try:
            if any(grandchild.is_dir() and grandchild.name.isdigit() for grandchild in child.iterdir()):
                return True
        except OSError:
            continue
    return False


def describe_public_release_dataset(cfg: Dict[str, Any], config_dir: Path) -> Optional[str]:
    paths_cfg = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    mode = str(data_cfg.get("mode", "auto")).strip().lower()
    if mode not in {"auto", "public_manifest"}:
        return None

    project_root = Path(config_dir).parent
    dataset_root = resolve_path(paths_cfg.get("dataset_root", "datasets"), project_root)
    if _has_private_sample_dirs(dataset_root):
        return None

    manifest_default = dataset_root / "benchmark_prompts.csv"
    manifest_path = resolve_path(data_cfg.get("prompt_manifest", manifest_default), project_root)
    if not manifest_path.exists() or not manifest_path.is_file():
        return None

    prompt_col = str(data_cfg.get("prompt_column", "prompt"))
    scene_col = str(data_cfg.get("scene_column", "scene"))
    sample_col = str(data_cfg.get("sample_id_column", "sample_id"))
    group_col = str(data_cfg.get("dataset_group_column", "dataset_group"))
    prompt_id_col = str(data_cfg.get("prompt_id_column", "prompt_id"))

    row_count = 0
    unique_prompts = set()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_count += 1
            unique_prompts.add(
                (
                    str(row.get(group_col, "")).strip(),
                    str(row.get(scene_col, "")).strip(),
                    str(row.get(sample_col, "")).strip(),
                    str(row.get(prompt_id_col, "")).strip(),
                    str(row.get(prompt_col, "")).strip(),
                )
            )

    return (
        f"Detected public prompt manifest only: {manifest_path} "
        f"({len(unique_prompts)} unique prompts, {row_count} CSV rows). "
        f"Full generation still requires private sample folders with one input image, prompt.txt, "
        f"and processed/ assets."
    )
