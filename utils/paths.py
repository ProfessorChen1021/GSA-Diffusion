from __future__ import annotations

import os
from pathlib import Path


def _default_hf_cache_root() -> Path:
    env_root = os.environ.get("GSA_HF_CACHE_ROOT")
    if env_root:
        return Path(env_root).expanduser()

    env_hub = os.environ.get("HF_HUB_CACHE")
    if env_hub:
        return Path(env_hub).expanduser()

    env_home = os.environ.get("HF_HOME")
    if env_home:
        return Path(env_home).expanduser() / "hub"

    return Path.home() / ".cache" / "huggingface" / "hub"


HF_CACHE_ROOT = _default_hf_cache_root().resolve()
HF_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_CLIPSCORE_SNAPSHOT = (
    HF_CACHE_ROOT
    / "models--openai--clip-vit-base-patch16"
    / "snapshots"
    / "57c216476eefef5ab752ec549e440a49ae4ae5f3"
)


def resolve_model_identifier(model_name: str) -> str:
    model_path = Path(model_name).expanduser()
    if model_path.exists():
        return str(model_path.resolve())
    return model_name

