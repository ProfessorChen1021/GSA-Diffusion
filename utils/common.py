import datetime
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple

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
