"""
Standalone SDXL test script for comparing baseline vs ADI under the same
sampling setup.

Typical usage:
    py -3 run_in_SDXL.py --sample-dir "path\\to\\sample"

The sample directory is expected to follow the repository's processed layout:
    <sample_dir>/
      prompt.txt
      processed/
        entities.json
        masks/
          <phrase_entity>.png

In the public paper-release version of this repository, `datasets/` may only
contain `benchmark_prompts.csv`. In that case the script prints a note and
exits early, because full SDXL generation still requires the private reference
images and processed assets.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import inspect
import json
import math
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

from methods.adi_attention import ADICrossAttnProcessor
from src.sd15 import SCHED_MAP
from utils.common import describe_public_release_dataset, dump_run_config, load_config
from utils.image import list_images, make_canny_image
from utils.io import ensure_outdir, save_image, save_json, save_text
from utils.paths import HF_CACHE_ROOT, resolve_model_identifier
from utils.seed import make_torch_generator, seed_everything
from utils.timers import Timer
from utils.visualization import save_entity_attention_maps


DEFAULT_SDXL_PATH = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_SDXL_CONFIG = "config/sdxl.yaml"
DEFAULT_DATASET_ROOT = Path("datasets")
DEFAULT_NEGATIVE_PROMPT = "low quality, blurry, unfinished, cartoon, 3d model, bad anatomy"
DEFAULT_POSITIVE_SUFFIX = "photorealistic, 4k, highly detailed, cinematic lighting"


def _resolve_config_path(config_dir: Path, path_like: str) -> str:
    path = Path(path_like)
    if not path.is_absolute():
        path = (config_dir.parent / path).resolve()
    return str(path)


def _cli_flags(argv: Sequence[str]) -> Set[str]:
    flags: Set[str] = set()
    for token in argv:
        if token.startswith("--"):
            flags.add(token.split("=", 1)[0])
    return flags


def _flag_missing(cli_flags: Set[str], *names: str) -> bool:
    return not any(name in cli_flags for name in names)


@dataclass
class SDXLADIContext:
    entities: List[Dict[str, Any]]
    enable_global: bool = True
    injection_enabled: bool = True
    injection_strength: float = 1.0
    current_step_idx: int = 0
    capture_steps: Set[int] = field(default_factory=set)
    attn_records: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SamplePaths:
    dataset_name: str
    sample_id: str
    root: Path
    raw_image: Path
    raw_prompt: Path
    processed_dir: Path
    canny_path: Path
    entities_json: Path
    masks_dir: Path


def apply_config_defaults(
    args: argparse.Namespace,
    argv: Sequence[str],
) -> Tuple[argparse.Namespace, Dict[str, Any], Path]:
    cfg, config_dir = load_config(args.config)
    cli_flags = _cli_flags(argv)

    paths_cfg = cfg.get("paths", {})
    params_cfg = cfg.get("params", {})
    controlnet_cfg = cfg.get("controlnet", {})
    prompt_cfg = cfg.get("prompt_config", {})
    injection_cfg = cfg.get("injection", {})
    adi_cfg = cfg.get("adi", {})

    if _flag_missing(cli_flags, "--dataset-root") and "dataset_root" in paths_cfg:
        args.dataset_root = _resolve_config_path(config_dir, str(paths_cfg["dataset_root"]))
    else:
        args.dataset_root = str(Path(args.dataset_root).resolve())

    if _flag_missing(cli_flags, "--output-root") and "output_root" in paths_cfg:
        args.output_root = _resolve_config_path(config_dir, str(paths_cfg["output_root"]))

    if _flag_missing(cli_flags, "--width") and "width" in params_cfg:
        args.width = int(params_cfg["width"])
    if _flag_missing(cli_flags, "--height") and "height" in params_cfg:
        args.height = int(params_cfg["height"])
    if _flag_missing(cli_flags, "--steps") and "num_inference_steps" in params_cfg:
        args.steps = int(params_cfg["num_inference_steps"])
    if _flag_missing(cli_flags, "--guidance-scale") and "guidance_scale" in params_cfg:
        args.guidance_scale = float(params_cfg["guidance_scale"])
    if _flag_missing(cli_flags, "--scheduler") and "scheduler" in params_cfg:
        args.scheduler = str(params_cfg["scheduler"])
    if _flag_missing(cli_flags, "--dtype") and "dtype" in params_cfg:
        args.dtype = str(params_cfg["dtype"])
    if _flag_missing(cli_flags, "--seeds") and "seeds" in params_cfg:
        args.seeds = [int(x) for x in params_cfg["seeds"]]
    if _flag_missing(cli_flags, "--max-samples") and "max_samples" in params_cfg:
        args.max_samples = params_cfg["max_samples"]
    if _flag_missing(cli_flags, "--enable-vae-slicing", "--no-enable-vae-slicing"):
        args.enable_vae_slicing = bool(params_cfg.get("enable_vae_slicing", args.enable_vae_slicing))
    if _flag_missing(cli_flags, "--enable-xformers", "--no-enable-xformers"):
        args.enable_xformers = bool(params_cfg.get("enable_xformers", args.enable_xformers))
    if _flag_missing(cli_flags, "--pair-baseline", "--no-pair-baseline"):
        args.pair_baseline = bool(params_cfg.get("pair_baseline", args.pair_baseline))

    if _flag_missing(cli_flags, "--controlnet-model-path") and "model_path" in controlnet_cfg:
        args.controlnet_model_path = str(controlnet_cfg["model_path"])
    if _flag_missing(cli_flags, "--conditioning-scale") and "conditioning_scale" in controlnet_cfg:
        args.conditioning_scale = float(controlnet_cfg["conditioning_scale"])
    if _flag_missing(cli_flags, "--guess-mode", "--no-guess-mode"):
        args.guess_mode = bool(controlnet_cfg.get("guess_mode", args.guess_mode))
    if _flag_missing(cli_flags, "--canny-low-threshold") and "canny_low_threshold" in controlnet_cfg:
        args.canny_low_threshold = int(controlnet_cfg["canny_low_threshold"])
    if _flag_missing(cli_flags, "--canny-high-threshold") and "canny_high_threshold" in controlnet_cfg:
        args.canny_high_threshold = int(controlnet_cfg["canny_high_threshold"])

    if _flag_missing(cli_flags, "--negative-prompt"):
        args.negative_prompt = str(prompt_cfg.get("negative_prompt", args.negative_prompt))
    if _flag_missing(cli_flags, "--positive-suffix"):
        args.positive_suffix = str(prompt_cfg.get("positive_suffix", args.positive_suffix))

    if _flag_missing(cli_flags, "--adi-ratio") and "adi_ratio" in injection_cfg:
        args.adi_ratio = float(injection_cfg["adi_ratio"])
    if _flag_missing(cli_flags, "--controlnet-ratio") and "controlnet_ratio" in injection_cfg:
        args.controlnet_ratio = float(injection_cfg["controlnet_ratio"])
    if _flag_missing(cli_flags, "--adi-linear-decay", "--no-adi-linear-decay"):
        args.adi_linear_decay = bool(injection_cfg.get("adi_linear_decay", args.adi_linear_decay))

    if _flag_missing(cli_flags, "--alpha-g") and "alpha_g" in adi_cfg:
        args.alpha_g = float(adi_cfg["alpha_g"])
    if _flag_missing(cli_flags, "--alpha-e") and "alpha_e" in adi_cfg:
        args.alpha_e = float(adi_cfg["alpha_e"])
    if _flag_missing(cli_flags, "--sharpen-t") and "sharpen_t" in adi_cfg:
        args.sharpen_t = float(adi_cfg["sharpen_t"])
    if _flag_missing(cli_flags, "--enable-self-iso", "--no-enable-self-iso"):
        args.enable_self_iso = bool(adi_cfg.get("enable_self_iso", args.enable_self_iso))
    if _flag_missing(cli_flags, "--enable-global", "--no-enable-global"):
        args.enable_global = bool(adi_cfg.get("enable_global", args.enable_global))
    if _flag_missing(cli_flags, "--self-iso-max-side") and "self_iso_max_side" in adi_cfg:
        args.self_iso_max_side = int(adi_cfg["self_iso_max_side"])

    return args, cfg, config_dir


def build_effective_run_config(args: argparse.Namespace, cfg: Dict[str, Any]) -> Dict[str, Any]:
    effective = deepcopy(cfg)
    effective.setdefault("paths", {})
    effective.setdefault("params", {})
    effective.setdefault("controlnet", {})
    effective.setdefault("prompt_config", {})
    effective.setdefault("injection", {})
    effective.setdefault("adi", {})
    effective["engine"] = "sdxl"

    effective["paths"]["base_model_path"] = args.base_model_path
    effective["paths"]["dataset_root"] = args.dataset_root
    effective["paths"]["output_root"] = args.output_root

    effective["params"]["width"] = args.width
    effective["params"]["height"] = args.height
    effective["params"]["num_inference_steps"] = args.steps
    effective["params"]["guidance_scale"] = args.guidance_scale
    effective["params"]["scheduler"] = args.scheduler
    effective["params"]["dtype"] = args.dtype
    effective["params"]["enable_vae_slicing"] = args.enable_vae_slicing
    effective["params"]["enable_xformers"] = args.enable_xformers
    effective["params"]["pair_baseline"] = bool(getattr(args, "pair_baseline", False))
    effective["params"]["seeds"] = list(args.seeds)
    effective["params"]["max_samples"] = args.max_samples

    effective["prompt_config"]["negative_prompt"] = args.negative_prompt
    effective["prompt_config"]["positive_suffix"] = args.positive_suffix

    effective["controlnet"]["model_path"] = args.controlnet_model_path
    effective["controlnet"]["conditioning_scale"] = args.conditioning_scale
    effective["controlnet"]["guess_mode"] = args.guess_mode
    effective["controlnet"]["canny_low_threshold"] = args.canny_low_threshold
    effective["controlnet"]["canny_high_threshold"] = args.canny_high_threshold

    effective["injection"]["controlnet_ratio"] = args.controlnet_ratio
    effective["injection"]["adi_ratio"] = args.adi_ratio
    effective["injection"]["adi_linear_decay"] = args.adi_linear_decay

    effective["adi"]["alpha_g"] = args.alpha_g
    effective["adi"]["alpha_e"] = args.alpha_e
    effective["adi"]["sharpen_t"] = args.sharpen_t
    effective["adi"]["enable_self_iso"] = args.enable_self_iso
    effective["adi"]["enable_global"] = args.enable_global
    effective["adi"]["self_iso_max_side"] = args.self_iso_max_side

    return effective


class SDXLADICrossAttnProcessor(ADICrossAttnProcessor):
    """
    Reuses the repository's ADI implementation and only adds:
    1. a more permissive __call__ signature for SDXL/diffusers variants
    2. a guard that skips mask-based self-isolation on very large feature maps
       to avoid SDXL OOM on 1024px inference.
    """

    def __init__(self, *args, self_iso_max_side: int = 32, **kwargs):
        super().__init__(*args, **kwargs)
        self.self_iso_max_side = int(self_iso_max_side)
        self._reported_large_self_iso: Set[Tuple[str, int]] = set()

    def _should_skip_self_iso(self, hidden_states: torch.Tensor) -> bool:
        n_tokens = int(hidden_states.shape[1])
        side = int(math.sqrt(n_tokens))
        if side * side != n_tokens:
            return True

        if self.self_iso_max_side > 0 and side > self.self_iso_max_side:
            key = (self.layer_name, side)
            if key not in self._reported_large_self_iso:
                print(
                    f"[ADI][SDXL] Skip self isolation at {self.layer_name} "
                    f"for {side}x{side} feature map (limit={self.self_iso_max_side})."
                )
                self._reported_large_self_iso.add(key)
            return True
        return False

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        if attention_mask is None:
            attention_mask = kwargs.get("encoder_attention_mask", None)

        self._debug_call_idx += 1
        if self._debug_call_idx % 1000 == 0:
            print(f"[ADI Running] Processing Layer, Self-Attn={encoder_hidden_states is None}")

        strength = float(getattr(self.ctx, "injection_strength", 1.0))
        if (hasattr(self.ctx, "injection_enabled") and not self.ctx.injection_enabled) or strength <= 1e-4:
            return self._vanilla_attn(attn, hidden_states, encoder_hidden_states, attention_mask)

        is_self_attn = (encoder_hidden_states is None) or (
            encoder_hidden_states.shape[1] == hidden_states.shape[1]
        )

        if is_self_attn:
            if not self.enable_self_iso or self._should_skip_self_iso(hidden_states):
                return self._vanilla_attn(attn, hidden_states, encoder_hidden_states, attention_mask)

            b_full = hidden_states.shape[0]
            if b_full % 2 == 0:
                h_u, h_c = hidden_states.chunk(2)
                out_u = self._vanilla_attn(attn, h_u)
                out_c_iso = self._masked_self_attn(attn, h_c)
                out_c_vanilla = self._vanilla_attn(attn, h_c)
                out_c = out_c_iso * strength + out_c_vanilla * (1.0 - strength)
                return torch.cat([out_u, out_c], dim=0)

            iso = self._masked_self_attn(attn, hidden_states)
            vanilla = self._vanilla_attn(attn, hidden_states)
            return iso * strength + vanilla * (1.0 - strength)

        b_full = hidden_states.shape[0]
        if b_full % 2 == 0:
            h_u, h_c = hidden_states.chunk(2)
            e_u, e_c = encoder_hidden_states.chunk(2)

            out_u = self._vanilla_attn(attn, h_u, e_u, attention_mask)
            out_c_adi = self._adi_cross_attn(attn, h_c, e_c)
            out_c_vanilla = self._vanilla_attn(attn, h_c, e_c, attention_mask)
            out_c = out_c_adi * strength + out_c_vanilla * (1.0 - strength)
            return torch.cat([out_u, out_c], dim=0)

        adi = self._adi_cross_attn(attn, hidden_states, encoder_hidden_states)
        vanilla = self._vanilla_attn(attn, hidden_states, encoder_hidden_states, attention_mask)
        return adi * strength + vanilla * (1.0 - strength)


def clamp01(value: Any) -> float:
    try:
        value_f = float(value)
    except Exception:
        return 1.0
    return max(0.0, min(1.0, value_f))


def inject_steps(num_inference_steps: int, ratio: float) -> int:
    return int(math.ceil(int(num_inference_steps) * clamp01(ratio)))


def append_positive_suffix(text: str, suffix: str) -> str:
    suffix = suffix.strip()
    text = text.strip()
    if not suffix:
        return text
    return f"{text}, {suffix}" if text else suffix


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def find_mask_path(masks_dir: Path, phrase_entity: str) -> Path:
    candidates = [
        masks_dir / f"{phrase_entity}.png",
        masks_dir / f"{phrase_entity}.jpg",
        masks_dir / f"{phrase_entity}.jpeg",
    ]
    normalized = phrase_entity.replace("/", "_").replace("\\", "_")
    if normalized != phrase_entity:
        candidates.extend(
            [
                masks_dir / f"{normalized}.png",
                masks_dir / f"{normalized}.jpg",
                masks_dir / f"{normalized}.jpeg",
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Mask not found for entity '{phrase_entity}' under {masks_dir}")


def load_mask_tensor(mask_path: Path) -> torch.Tensor:
    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask, dtype=np.float32) / 255.0
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
    return mask_tensor


def is_valid_sample_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "prompt.txt").exists()
        and (path / "processed" / "entities.json").exists()
        and (path / "processed" / "masks").is_dir()
    )


def build_sample_paths(sample_dir: Path, dataset_name: str) -> SamplePaths:
    images = list_images(sample_dir)
    if len(images) != 1:
        raise RuntimeError(f"[{dataset_name}/{sample_dir.name}] Expected exactly one image in {sample_dir}, found {len(images)}")

    processed_dir = sample_dir / "processed"
    return SamplePaths(
        dataset_name=dataset_name,
        sample_id=sample_dir.name,
        root=sample_dir,
        raw_image=images[0],
        raw_prompt=sample_dir / "prompt.txt",
        processed_dir=processed_dir,
        canny_path=processed_dir / "canny.png",
        entities_json=processed_dir / "entities.json",
        masks_dir=processed_dir / "masks",
    )


def discover_dataset_samples(
    root: Path,
    max_samples: Optional[Any] = None,
    sample_id: Optional[str] = None,
) -> List[SamplePaths]:
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    sample_id = str(sample_id).strip() if sample_id is not None else None

    if is_valid_sample_dir(root):
        dataset_name = root.parent.name if root.parent.name and not root.parent.name.isdigit() else ""
        samples = [build_sample_paths(root, dataset_name=dataset_name)]
        if sample_id and samples[0].sample_id != sample_id:
            raise RuntimeError(f"Requested sample_id={sample_id}, but resolved sample is {samples[0].sample_id}")
        return samples

    candidate_sets: List[Tuple[Path, List[Path]]] = []
    for ds_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda x: x.name):
        numeric_children = [c for c in ds_dir.iterdir() if c.is_dir() and c.name.isdigit()]
        if numeric_children:
            numeric_children = sorted(numeric_children, key=lambda x: int(x.name))
            candidate_sets.append((ds_dir, numeric_children))

    candidate_parents = {ds_dir for ds_dir, _ in candidate_sets}
    direct_numeric = [
        p for p in root.iterdir() if p.is_dir() and p.name.isdigit() and p not in candidate_parents
    ]
    direct_numeric = sorted(direct_numeric, key=lambda x: int(x.name))

    samples_flat: List[Tuple[str, Path]] = []
    for ds_dir, child_dirs in candidate_sets:
        for child in child_dirs:
            samples_flat.append((ds_dir.name, child))

    direct_numeric_dataset_name = root.name if direct_numeric else ""
    samples_flat.extend([(direct_numeric_dataset_name, p) for p in direct_numeric])

    if sample_id:
        samples_flat = [(ds, p) for ds, p in samples_flat if p.name == sample_id]

    if not samples_flat:
        child_dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda x: x.name)
        preview_dirs = [p.name for p in child_dirs[:10]]
        public_manifest = root / "benchmark_prompts.csv"
        hints: List[str] = [
            f"No valid samples found under {root}.",
            "Expected one of these layouts:",
            "  1. <dataset_root>/<dataset_name>/<sample_id>/prompt.txt",
            "     with <dataset_root>/<dataset_name>/<sample_id>/processed/entities.json and processed/masks/",
            "  2. <sample_dir>/prompt.txt with <sample_dir>/processed/entities.json and processed/masks/",
        ]
        if public_manifest.exists():
            hints.extend(
                [
                    "",
                    f"Detected public prompt manifest: {public_manifest}",
                    "This release includes prompt metadata, but not the private reference images and processed assets",
                    "required for the full SDXL + ControlNet workflow.",
                ]
            )
        if preview_dirs:
            hints.append(f"Top-level directories under {root}: {preview_dirs}")
        else:
            hints.append(f"{root} is empty or contains no subdirectories.")
        raise RuntimeError("\n".join(hints))

    samples_flat.sort(key=lambda x: (x[0], int(x[1].name)))
    if max_samples is not None:
        if isinstance(max_samples, int):
            samples_flat = samples_flat[: max(0, int(max_samples))]
        elif isinstance(max_samples, (list, tuple)) and len(max_samples) == 2:
            if not all(isinstance(x, (int, float)) for x in max_samples):
                raise ValueError("max_samples list must contain two numeric values, e.g., [13,78] or [0,0]")
            start = int(max_samples[0])
            end = int(max_samples[1])
            if not (start == 0 and end == 0):
                if start > end:
                    start, end = end, start
                samples_flat = [(ds, p) for ds, p in samples_flat if start <= int(p.name) <= end]
        else:
            raise ValueError("max_samples must be None, int, or a two-element list/tuple like [start, end]")

    samples = [build_sample_paths(path, dataset_name=dataset_name) for dataset_name, path in samples_flat]
    return samples


def resolve_single_input_sample(
    sample_dir: Optional[Path],
    prompt_text: str,
    entities_path: Path,
    masks_dir: Path,
) -> Tuple[Optional[Path], str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not entities_path.exists():
        raise FileNotFoundError(f"entities.json not found: {entities_path}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"masks directory not found: {masks_dir}")

    entities_raw = json.loads(entities_path.read_text(encoding="utf-8"))
    if not isinstance(entities_raw, list) or not entities_raw:
        raise ValueError(f"entities.json must be a non-empty list: {entities_path}")

    entities: List[Dict[str, Any]] = []
    manifest: List[Dict[str, Any]] = []

    for item in entities_raw:
        if not isinstance(item, dict):
            continue

        phrase_entity = str(item.get("phrase_entity", "")).strip()
        phrase_ent_attr = str(item.get("phrase_ent_attr", "")).strip() or phrase_entity
        if not phrase_entity:
            continue

        mask_path = find_mask_path(masks_dir, phrase_entity)
        entity = dict(item)
        entity["phrase_entity"] = phrase_entity
        entity["phrase_ent_attr"] = phrase_ent_attr
        entity["mask_path"] = str(mask_path)
        entity["mask_tensor"] = load_mask_tensor(mask_path)
        entities.append(entity)

        manifest.append(
            {
                "phrase_entity": phrase_entity,
                "phrase_ent_attr": phrase_ent_attr,
                "mask_path": str(mask_path),
            }
        )

    if not entities:
        raise ValueError(f"No valid entities loaded from: {entities_path}")

    return sample_dir, prompt_text, entities, manifest


def load_inputs_from_sample(sample: SamplePaths) -> Tuple[Path, str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not sample.raw_prompt.exists():
        raise FileNotFoundError(f"prompt.txt not found: {sample.raw_prompt}")
    return resolve_single_input_sample(
        sample_dir=sample.root,
        prompt_text=read_text_file(sample.raw_prompt),
        entities_path=sample.entities_json,
        masks_dir=sample.masks_dir,
    )


def resolve_conditioning_image(
    args: argparse.Namespace,
    sample: Optional[SamplePaths] = None,
    conditioning_image_path: Optional[Path] = None,
) -> Image.Image:
    if conditioning_image_path is not None:
        image = Image.open(conditioning_image_path).convert("RGB")
        if image.size != (args.width, args.height):
            image = image.resize((args.width, args.height), Image.Resampling.BILINEAR)
        return image

    if sample is None:
        raise ValueError("A conditioning image is required for SDXL + ControlNet.")

    if sample.canny_path.exists():
        image = Image.open(sample.canny_path).convert("RGB")
        if image.size == (args.width, args.height):
            return image

    return make_canny_image(
        path=sample.raw_image,
        width=args.width,
        height=args.height,
        low_threshold=args.canny_low_threshold,
        high_threshold=args.canny_high_threshold,
    )


def maybe_add_watermarker_flag(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    init_sig = inspect.signature(StableDiffusionXLControlNetPipeline.__init__)
    if "add_watermarker" in init_sig.parameters:
        kwargs["add_watermarker"] = False
    return kwargs


def load_sdxl_pipeline(args: argparse.Namespace):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this script.")
    if not args.controlnet_model_path:
        raise RuntimeError(
            "An SDXL ControlNet model path is required. "
            "Set controlnet.model_path in config or pass --controlnet-model-path."
        )

    resolved_model_path = resolve_model_identifier(args.base_model_path)
    resolved_controlnet_path = resolve_model_identifier(args.controlnet_model_path)
    if args.scheduler not in SCHED_MAP:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")

    torch_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    from_pretrained_kwargs: Dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "cache_dir": str(HF_CACHE_ROOT),
    }
    if torch_dtype == torch.float16:
        from_pretrained_kwargs["variant"] = "fp16"
        from_pretrained_kwargs["use_safetensors"] = True
    maybe_add_watermarker_flag(from_pretrained_kwargs)

    controlnet_load_kwargs: Dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "cache_dir": str(HF_CACHE_ROOT),
    }
    if torch_dtype == torch.float16:
        controlnet_load_kwargs["variant"] = "fp16"
        controlnet_load_kwargs["use_safetensors"] = True

    controlnet = ControlNetModel.from_pretrained(
        resolved_controlnet_path,
        **controlnet_load_kwargs,
    )
    controlnet_cross_attention_dim = getattr(controlnet.config, "cross_attention_dim", None)
    if controlnet_cross_attention_dim is not None and int(controlnet_cross_attention_dim) != 2048:
        raise RuntimeError(
            "The configured ControlNet is not an SDXL ControlNet. "
            f"Expected cross_attention_dim=2048, got {controlnet_cross_attention_dim}. "
            f"Current path: {resolved_controlnet_path}"
        )

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        resolved_model_path,
        controlnet=controlnet,
        **from_pretrained_kwargs,
    )
    pipe.scheduler = SCHED_MAP[args.scheduler].from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    if args.enable_vae_slicing:
        pipe.enable_vae_slicing()

    if args.enable_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as exc:
            print(f"[SDXL] xformers not enabled: {exc}")

    return pipe


def build_token_mask(pipe: StableDiffusionXLControlNetPipeline, text: str) -> torch.Tensor:
    tok1 = pipe.tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=pipe.tokenizer.model_max_length,
    )
    tok2 = pipe.tokenizer_2(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=pipe.tokenizer_2.model_max_length,
    )
    attn_mask = torch.maximum(tok1["attention_mask"], tok2["attention_mask"])
    return attn_mask


def encode_local_prompt(pipe: StableDiffusionXLControlNetPipeline, text: str, device: torch.device) -> torch.Tensor:
    encoded = pipe.encode_prompt(
        prompt=text,
        prompt_2=text,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    if not isinstance(encoded, tuple) or not encoded:
        raise RuntimeError("StableDiffusionXLControlNetPipeline.encode_prompt returned an unexpected value.")
    prompt_embeds = encoded[0]
    return prompt_embeds


def build_sdxl_adi_context(
    pipe: StableDiffusionXLControlNetPipeline,
    entities: Sequence[Dict[str, Any]],
    positive_suffix: str,
    enable_global: bool,
) -> SDXLADIContext:
    device = pipe._execution_device if hasattr(pipe, "_execution_device") else torch.device("cuda")
    dtype = pipe.unet.dtype
    encoded_entities: List[Dict[str, Any]] = []

    with torch.inference_mode():
        for entity in entities:
            item = dict(entity)
            local_text = append_positive_suffix(
                item.get("phrase_ent_attr") or item.get("phrase_entity", ""),
                positive_suffix,
            )
            local_embedding = encode_local_prompt(pipe, local_text, device=device).to(device=device, dtype=dtype)
            local_attn_mask = build_token_mask(pipe, local_text).to(device)

            item["local_embedding"] = local_embedding
            item["local_attn_mask"] = local_attn_mask
            item["token_mask"] = local_attn_mask[0].detach().cpu().bool()
            item["mask_tensor"] = item["mask_tensor"].to(device=device, dtype=dtype)
            encoded_entities.append(item)

    return SDXLADIContext(entities=encoded_entities, enable_global=enable_global)


def collect_target_attention_modules(
    pipe: StableDiffusionXLControlNetPipeline,
) -> List[Tuple[str, Any]]:
    targets: List[Tuple[str, Any]] = []
    for name, module in pipe.unet.named_modules():
        if ("up_blocks" in name or "mid_block" in name) and ("attn1" in name or "attn2" in name):
            if hasattr(module, "set_processor") and hasattr(module, "processor"):
                targets.append((name, module))

    if not targets:
        raise RuntimeError("No target attention layers found in SDXL UNet.")

    print(f"[ADI][SDXL] Found {len(targets)} target attention layers.")
    return targets


def store_original_processors(targets: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    return {name: module.processor for name, module in targets}


def restore_original_processors(targets: Sequence[Tuple[str, Any]], originals: Dict[str, Any]) -> None:
    for name, module in targets:
        module.set_processor(originals[name])


def attach_adi_processors(
    targets: Sequence[Tuple[str, Any]],
    ctx: SDXLADIContext,
    args: argparse.Namespace,
) -> None:
    for name, module in targets:
        module.set_processor(
            SDXLADICrossAttnProcessor(
                ctx,
                layer_name=name,
                alpha_g=args.alpha_g,
                alpha_e=args.alpha_e,
                sharpen_t=args.sharpen_t,
                enable_self_iso=args.enable_self_iso,
                self_iso_max_side=args.self_iso_max_side,
            )
        )


def prepare_capture_steps(num_inference_steps: int, enabled: bool) -> Set[int]:
    if not enabled:
        return set()

    capture_steps: Set[int] = set()
    for ratio in (0.4, 0.7):
        index = int(num_inference_steps * ratio)
        if 0 <= index < num_inference_steps:
            capture_steps.add(index)
    return capture_steps


def reset_context_runtime(ctx: SDXLADIContext, args: argparse.Namespace) -> int:
    adi_steps = inject_steps(args.steps, args.adi_ratio)
    ctx.injection_enabled = adi_steps > 0
    ctx.injection_strength = 1.0 if adi_steps > 0 else 0.0
    ctx.current_step_idx = 0
    ctx.capture_steps = prepare_capture_steps(args.steps, not args.disable_attn_maps)
    ctx.attn_records = []
    return adi_steps


def update_context_after_step(
    ctx: SDXLADIContext,
    next_step: int,
    num_inference_steps: int,
    adi_steps: int,
    linear_decay: bool,
) -> None:
    if adi_steps <= 0:
        strength = 0.0
    elif next_step <= adi_steps:
        strength = 1.0
    else:
        strength = 0.0

    if linear_decay and num_inference_steps > 0 and adi_steps > 0 and next_step > adi_steps:
        remain = num_inference_steps - adi_steps
        if remain > 0:
            decay_ratio = (next_step - adi_steps) / remain
            strength = max(0.0, 1.0 - decay_ratio)
        else:
            strength = 0.0

    ctx.injection_strength = strength
    ctx.injection_enabled = strength > 1e-4
    ctx.current_step_idx = next_step


def build_pipeline_call_kwargs(
    pipe: StableDiffusionXLControlNetPipeline,
    ctx: Optional[SDXLADIContext],
    args: argparse.Namespace,
    adi_steps: int,
    step_hook=None,
) -> Dict[str, Any]:
    call_sig = inspect.signature(pipe.__call__)
    kwargs: Dict[str, Any] = {}

    if "callback_on_step_end" in call_sig.parameters:
        def _cb_on_step_end(_pipe, step_idx, timestep, callback_kwargs):
            if ctx is not None:
                update_context_after_step(
                    ctx=ctx,
                    next_step=int(step_idx) + 1,
                    num_inference_steps=args.steps,
                    adi_steps=adi_steps,
                    linear_decay=args.adi_linear_decay,
                )
            if step_hook is not None:
                step_hook(int(step_idx), int(timestep), callback_kwargs.get("latents"))
            return callback_kwargs

        if ctx is not None or step_hook is not None:
            kwargs["callback_on_step_end"] = _cb_on_step_end
        if (ctx is not None or step_hook is not None) and "callback_on_step_end_tensor_inputs" in call_sig.parameters:
            kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
        return kwargs

    if "callback" in call_sig.parameters:
        def _cb(step_idx, timestep, latents):
            if ctx is not None:
                update_context_after_step(
                    ctx=ctx,
                    next_step=int(step_idx) + 1,
                    num_inference_steps=args.steps,
                    adi_steps=adi_steps,
                    linear_decay=args.adi_linear_decay,
                )
            if step_hook is not None:
                step_hook(int(step_idx), int(timestep), latents)

        if ctx is not None or step_hook is not None:
            kwargs["callback"] = _cb
        if (ctx is not None or step_hook is not None) and "callback_steps" in call_sig.parameters:
            kwargs["callback_steps"] = 1
        return kwargs

    raise RuntimeError("This diffusers version does not expose a supported step callback API.")


def _tensor_image_to_pil(image_tensor: torch.Tensor) -> Tuple[Image.Image, bool]:
    image_tensor = image_tensor.detach().float().cpu()
    had_nonfinite = not torch.isfinite(image_tensor).all()
    if had_nonfinite:
        image_tensor = torch.nan_to_num(image_tensor, nan=0.0, posinf=1.0, neginf=0.0)
    image_tensor = image_tensor.clamp(0.0, 1.0)

    if image_tensor.ndim == 3 and image_tensor.shape[0] in {1, 3, 4}:
        image_np = image_tensor.permute(1, 2, 0).numpy()
    elif image_tensor.ndim == 3 and image_tensor.shape[-1] in {1, 3, 4}:
        image_np = image_tensor.numpy()
    else:
        raise RuntimeError(f"Unexpected image tensor shape from pipeline: {tuple(image_tensor.shape)}")

    if image_np.shape[-1] == 1:
        image_np = np.repeat(image_np, 3, axis=-1)
    elif image_np.shape[-1] == 4:
        image_np = image_np[..., :3]

    image_np = np.clip(image_np, 0.0, 1.0)
    image_np = np.uint8(np.round(image_np * 255.0))
    return Image.fromarray(image_np), had_nonfinite


@torch.inference_mode()
def generate_one(
    pipe,
    prompt: str,
    negative_prompt: str,
    conditioning_image: Image.Image,
    seed: int,
    args: argparse.Namespace,
    ctx: Optional[SDXLADIContext] = None,
    adi_steps: int = 0,
    step_hook=None,
) -> Tuple[Image.Image, Dict[str, Any]]:
    generator = make_torch_generator(seed, device="cuda")

    call_kwargs = build_pipeline_call_kwargs(pipe, ctx, args, adi_steps, step_hook=step_hook)
    controlnet_end_ratio = max(0.0, min(1.0, float(args.controlnet_ratio)))
    conditioning_scale = float(args.conditioning_scale)
    if controlnet_end_ratio <= 0.0:
        conditioning_scale = 0.0
        controlnet_end_ratio = 1e-3

    out = pipe(
        prompt=prompt,
        prompt_2=prompt,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt,
        image=conditioning_image,
        controlnet_conditioning_scale=conditioning_scale,
        guess_mode=args.guess_mode,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        control_guidance_start=0.0,
        control_guidance_end=controlnet_end_ratio,
        output_type="pt",
        **call_kwargs,
    )

    image, had_nonfinite = _tensor_image_to_pil(out.images[0])
    extra = {
        "nsfw": getattr(out, "nsfw_content_detected", None),
        "nan_sanitized": had_nonfinite,
    }
    if had_nonfinite:
        print(f"[SDXL] Non-finite image tensor detected for seed={seed}; sanitized before saving.")
    return image, extra


def run_generation_timed(
    pipe,
    prompt: str,
    negative_prompt: str,
    conditioning_image: Image.Image,
    seed: int,
    args: argparse.Namespace,
    ctx: Optional[SDXLADIContext] = None,
    adi_steps: int = 0,
    step_hook=None,
) -> Tuple[Image.Image, Dict[str, Any], float, Optional[float]]:
    timer = Timer()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    image, extra = generate_one(
        pipe=pipe,
        prompt=prompt,
        negative_prompt=negative_prompt,
        conditioning_image=conditioning_image,
        seed=seed,
        args=args,
        ctx=ctx,
        adi_steps=adi_steps,
        step_hook=step_hook,
    )
    elapsed = timer.elapsed()
    peak_mem = None
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return image, extra, elapsed, peak_mem


def save_comparison_image(baseline_img: Image.Image, adi_img: Image.Image, path: Path) -> None:
    width = baseline_img.width + adi_img.width
    height = max(baseline_img.height, adi_img.height)
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    canvas.paste(baseline_img, (0, 0))
    canvas.paste(adi_img, (baseline_img.width, 0))
    canvas.save(path)


def save_runtime_summaries(
    seed_dir: Path,
    baseline_elapsed: float,
    adi_elapsed: float,
    baseline_peak_mem: Optional[float],
    adi_peak_mem: Optional[float],
) -> None:
    save_text(
        f"baseline={baseline_elapsed:.3f}s\nadi={adi_elapsed:.3f}s",
        seed_dir / "time.txt",
    )
    if baseline_peak_mem is not None or adi_peak_mem is not None:
        baseline_text = "n/a" if baseline_peak_mem is None else f"{baseline_peak_mem:.3f} GB"
        adi_text = "n/a" if adi_peak_mem is None else f"{adi_peak_mem:.3f} GB"
        save_text(
            f"baseline={baseline_text}\nadi={adi_text}",
            seed_dir / "gpu_mem_gb.txt",
        )


def save_single_runtime_summary(
    seed_dir: Path,
    elapsed: float,
    peak_mem: Optional[float],
) -> None:
    save_text(f"{elapsed:.3f}s", seed_dir / "time.txt")
    if peak_mem is not None:
        save_text(f"{peak_mem:.3f} GB", seed_dir / "gpu_mem_gb.txt")


def release_cuda_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def build_output_root(args: argparse.Namespace, sample_count: int, seed_count: int, run_tag: str) -> Path:
    now = dt.datetime.now()
    date_part = now.strftime("%y%m%d")
    time_part = now.strftime("%H%M")
    output_root = Path(args.output_root) if args.output_root else Path("outputs")
    if sample_count > 0:
        run_dir_name = f"{sample_count}samples{seed_count}seeds_{date_part}_{time_part}"
    else:
        run_dir_name = f"{run_tag}_{date_part}_{time_part}"
    run_dir = output_root / run_dir_name
    ensure_outdir(str(run_dir))
    return run_dir.resolve()


def build_seed_meta(
    args: argparse.Namespace,
    sample_dir: Optional[Path],
    canny_image_path: Optional[str],
    prompt_raw: str,
    prompt_used: str,
    seed: int,
    adi_steps: int,
    controlnet_steps: int,
    baseline_extra: Dict[str, Any],
    adi_extra: Dict[str, Any],
    entities_manifest: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "sample_dir": str(sample_dir) if sample_dir is not None else None,
        "canny_image": canny_image_path,
        "base_model_path": resolve_model_identifier(args.base_model_path),
        "controlnet_model_path": resolve_model_identifier(args.controlnet_model_path),
        "seed": int(seed),
        "prompt_raw": prompt_raw,
        "prompt_used": prompt_used,
        "negative_prompt": args.negative_prompt,
        "width": args.width,
        "height": args.height,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "scheduler": args.scheduler,
        "dtype": args.dtype,
        "enable_xformers": args.enable_xformers,
        "enable_vae_slicing": args.enable_vae_slicing,
        "controlnet_conditioning_scale": args.conditioning_scale,
        "controlnet_guess_mode": args.guess_mode,
        "pair_baseline": bool(getattr(args, "pair_baseline", False)),
        "controlnet_inject_ratio": args.controlnet_ratio,
        "controlnet_inject_steps": controlnet_steps,
        "canny_low_threshold": args.canny_low_threshold,
        "canny_high_threshold": args.canny_high_threshold,
        "adi": {
            "enabled": adi_steps > 0,
            "adi_ratio": args.adi_ratio,
            "adi_steps": adi_steps,
            "adi_linear_decay": args.adi_linear_decay,
            "alpha_g": args.alpha_g,
            "alpha_e": args.alpha_e,
            "sharpen_t": args.sharpen_t,
            "enable_self_iso": args.enable_self_iso,
            "self_iso_max_side": args.self_iso_max_side,
            "enable_global": args.enable_global,
        },
        "baseline_extra": baseline_extra,
        "adi_extra": adi_extra,
        "entities": entities_manifest,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SDXL pipeline with config-driven ControlNet/ADI injection.")
    parser.add_argument("--config", default=DEFAULT_SDXL_CONFIG, help="Path to config yaml. Defaults to config/sdxl.yaml.")
    parser.add_argument("--base-model-path", default=DEFAULT_SDXL_PATH, help="Local path or HF id for SDXL base model.")
    parser.add_argument("--controlnet-model-path", default=None, help="Local path or HF id for SDXL ControlNet model.")
    parser.add_argument("--sample-dir", default=None, help="Sample dir containing prompt.txt and processed/ assets.")
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_DATASET_ROOT),
        help="Dataset root used for auto-discovery or together with --sample-rel.",
    )
    parser.add_argument(
        "--sample-rel",
        default=None,
        help="Sample path relative to --dataset-root, for example: two_entity/horse_cow_grassland/1",
    )
    parser.add_argument(
        "--sample-id",
        default=None,
        help="Optional leaf sample id used when --sample-dir or --sample-rel points to an upper directory.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Limit the number of discovered samples.")
    parser.add_argument("--prompt", default=None, help="Prompt text. Used directly or overrides sample-dir/prompt.txt.")
    parser.add_argument("--entities-json", default=None, help="Path to entities.json.")
    parser.add_argument("--masks-dir", default=None, help="Directory that stores entity masks.")
    parser.add_argument("--conditioning-image", default=None, help="Manual mode ControlNet conditioning image path.")
    parser.add_argument("--output-root", default=None, help="Directory for all outputs.")

    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--scheduler", choices=sorted(SCHED_MAP.keys()), default="euler_a")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--seeds", nargs="+", type=int, default=[23])

    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--positive-suffix", default=DEFAULT_POSITIVE_SUFFIX)

    parser.add_argument("--conditioning-scale", type=float, default=1.0)
    parser.add_argument("--guess-mode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--canny-low-threshold", type=int, default=100)
    parser.add_argument("--canny-high-threshold", type=int, default=200)
    parser.add_argument("--controlnet-ratio", type=float, default=1.0)

    parser.add_argument("--adi-ratio", type=float, default=0.85)
    parser.add_argument("--adi-linear-decay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--alpha-g", type=float, default=0.5)
    parser.add_argument("--alpha-e", type=float, default=1.2)
    parser.add_argument("--sharpen-t", type=float, default=1.0)
    parser.add_argument("--enable-self-iso", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-global", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--self-iso-max-side",
        type=int,
        default=32,
        help="Skip self-attention isolation on feature maps larger than this side length.",
    )

    parser.add_argument("--enable-xformers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-vae-slicing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-attn-maps", action="store_true", help="Do not save captured entity attention maps.")
    parser.add_argument(
        "--pair-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run a matched baseline and ADI pair for each seed. Disabled by default to mirror the original pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args, cfg, _config_dir = apply_config_defaults(args, sys.argv[1:])
    seed_everything(0)
    controlnet_steps = inject_steps(args.steps, args.controlnet_ratio)
    manual_mode = bool(args.prompt and args.entities_json and args.masks_dir)
    public_release_note = describe_public_release_dataset(cfg, _config_dir)
    manual_payload: Optional[Tuple[Optional[Path], str, List[Dict[str, Any]], List[Dict[str, Any]]]] = None
    samples: List[SamplePaths] = []

    if manual_mode:
        manual_payload = resolve_single_input_sample(
            sample_dir=None,
            prompt_text=args.prompt.strip(),
            entities_path=Path(args.entities_json).resolve(),
            masks_dir=Path(args.masks_dir).resolve(),
        )
        run_tag = "manual"
    else:
        if public_release_note and not args.sample_dir and not args.sample_rel:
            print(public_release_note)
            return
        if args.sample_dir:
            discovery_root = Path(args.sample_dir).resolve()
        elif args.sample_rel:
            discovery_root = (Path(args.dataset_root).resolve() / args.sample_rel).resolve()
        else:
            discovery_root = Path(args.dataset_root).resolve()

        samples = discover_dataset_samples(
            root=discovery_root,
            max_samples=args.max_samples,
            sample_id=args.sample_id,
        )
        run_tag = discovery_root.name

    sample_count = 0 if manual_mode else len(samples)
    seed_count = len(args.seeds)
    run_dir = build_output_root(args, sample_count=sample_count, seed_count=seed_count, run_tag=run_tag)
    dump_run_config(build_effective_run_config(args, cfg), str(run_dir))
    print(f"[SDXL] Output dir: {run_dir}")
    print(f"[SDXL] Loading model from: {resolve_model_identifier(args.base_model_path)}")

    if not manual_mode:
        print(f"[SDXL] Discovered {len(samples)} sample(s).")

    method_mod = importlib.import_module(cfg.get("method", "methods.baseline"))
    method = method_mod.Method(**cfg.get("method_kwargs", {}))
    step_hook = getattr(method, "on_step", None)
    if not callable(step_hook):
        step_hook = None

    pipe = load_sdxl_pipeline(args)
    targets = collect_target_attention_modules(pipe)
    original_processors = store_original_processors(targets)

    try:
        if manual_mode:
            conditioning_image_path = Path(args.conditioning_image).resolve() if args.conditioning_image else None
            conditioning_image = resolve_conditioning_image(
                args=args,
                sample=None,
                conditioning_image_path=conditioning_image_path,
            )
            sample_dir, prompt_raw, entities, entities_manifest = manual_payload  # type: ignore[misc]
            prompt_used = append_positive_suffix(prompt_raw, args.positive_suffix)
            save_text(prompt_raw, run_dir / "prompt_raw.txt")
            save_text(prompt_used, run_dir / "prompt_used.txt")
            save_json(entities_manifest, run_dir / "entities_used.json")

            adi_steps_cfg = inject_steps(args.steps, args.adi_ratio)
            ctx = None
            if adi_steps_cfg > 0:
                ctx = build_sdxl_adi_context(
                    pipe=pipe,
                    entities=entities,
                    positive_suffix=args.positive_suffix,
                    enable_global=args.enable_global,
                )

            for seed in args.seeds:
                seed_dir = Path(ensure_outdir(str(run_dir / f"seed_{seed}")))
                restore_original_processors(targets, original_processors)
                base_meta = build_seed_meta(
                    args=args,
                    sample_dir=sample_dir,
                    canny_image_path=str(conditioning_image_path) if conditioning_image_path is not None else None,
                    prompt_raw=prompt_raw,
                    prompt_used=prompt_used,
                    seed=seed,
                    adi_steps=adi_steps_cfg,
                    controlnet_steps=controlnet_steps,
                    baseline_extra={},
                    adi_extra={},
                    entities_manifest=entities_manifest,
                )

                if args.pair_baseline:
                    print(f"[SDXL][Manual][Seed {seed}] Running baseline ...")
                    method.before_sampling(pipe, base_meta)
                    baseline_img, baseline_extra, baseline_elapsed, baseline_peak_mem = run_generation_timed(
                        pipe=pipe,
                        prompt=prompt_used,
                        negative_prompt=args.negative_prompt,
                        conditioning_image=conditioning_image,
                        seed=seed,
                        args=args,
                        step_hook=step_hook,
                    )
                    method.after_sampling(pipe, base_meta, baseline_extra)
                    save_image(baseline_img, seed_dir / "baseline.png")

                    if adi_steps_cfg > 0 and ctx is not None:
                        adi_steps = reset_context_runtime(ctx, args)
                        attach_adi_processors(targets, ctx, args)

                        print(f"[SDXL][Manual][Seed {seed}] Running ADI ...")
                        method.before_sampling(pipe, base_meta)
                        adi_img, adi_extra, adi_elapsed, adi_peak_mem = run_generation_timed(
                            pipe=pipe,
                            prompt=prompt_used,
                            negative_prompt=args.negative_prompt,
                            conditioning_image=conditioning_image,
                            seed=seed,
                            args=args,
                            ctx=ctx,
                            adi_steps=adi_steps,
                            step_hook=step_hook,
                        )
                        method.after_sampling(pipe, base_meta, adi_extra)
                        adi_path = seed_dir / "adi.png"
                        save_image(adi_img, adi_path)
                        save_image(adi_img, seed_dir / "result.png")
                        save_image(conditioning_image, seed_dir / "canny.png")
                        save_comparison_image(baseline_img, adi_img, seed_dir / "comparison.png")
                        save_runtime_summaries(seed_dir, baseline_elapsed, adi_elapsed, baseline_peak_mem, adi_peak_mem)

                        if ctx.attn_records:
                            attn_dir = seed_dir / "adi_attn_maps"
                            save_entity_attention_maps(ctx.attn_records, adi_path, attn_dir)
                            ctx.attn_records = []

                        meta = build_seed_meta(
                            args=args,
                            sample_dir=sample_dir,
                            canny_image_path=str(conditioning_image_path) if conditioning_image_path is not None else None,
                            prompt_raw=prompt_raw,
                            prompt_used=prompt_used,
                            seed=seed,
                            adi_steps=adi_steps,
                            controlnet_steps=controlnet_steps,
                            baseline_extra=baseline_extra,
                            adi_extra=adi_extra,
                            entities_manifest=entities_manifest,
                        )
                        meta["result_mode"] = "adi"
                        meta["baseline_time_sec"] = baseline_elapsed
                        meta["adi_time_sec"] = adi_elapsed
                        meta["baseline_peak_mem_gb"] = baseline_peak_mem
                        meta["adi_peak_mem_gb"] = adi_peak_mem
                    else:
                        print(f"[SDXL][Manual][Seed {seed}] ADI disabled (adi_ratio=0); skipping ADI pass.")
                        save_image(baseline_img, seed_dir / "result.png")
                        save_image(conditioning_image, seed_dir / "canny.png")
                        save_single_runtime_summary(seed_dir, baseline_elapsed, baseline_peak_mem)

                        meta = build_seed_meta(
                            args=args,
                            sample_dir=sample_dir,
                            canny_image_path=str(conditioning_image_path) if conditioning_image_path is not None else None,
                            prompt_raw=prompt_raw,
                            prompt_used=prompt_used,
                            seed=seed,
                            adi_steps=0,
                            controlnet_steps=controlnet_steps,
                            baseline_extra=baseline_extra,
                            adi_extra={},
                            entities_manifest=entities_manifest,
                        )
                        meta["result_mode"] = "baseline"
                        meta["result_time_sec"] = baseline_elapsed
                        meta["result_peak_mem_gb"] = baseline_peak_mem
                else:
                    if adi_steps_cfg > 0 and ctx is not None:
                        adi_steps = reset_context_runtime(ctx, args)
                        attach_adi_processors(targets, ctx, args)
                        run_label = "ADI"
                        result_ctx = ctx
                        result_adi_steps = adi_steps
                        print(f"[SDXL][Manual][Seed {seed}] Running ADI ...")
                    else:
                        run_label = "baseline"
                        result_ctx = None
                        result_adi_steps = 0
                        print(f"[SDXL][Manual][Seed {seed}] Running baseline ...")

                    method.before_sampling(pipe, base_meta)
                    result_img, result_extra, result_elapsed, result_peak_mem = run_generation_timed(
                        pipe=pipe,
                        prompt=prompt_used,
                        negative_prompt=args.negative_prompt,
                        conditioning_image=conditioning_image,
                        seed=seed,
                        args=args,
                        ctx=result_ctx,
                        adi_steps=result_adi_steps,
                        step_hook=step_hook,
                    )
                    method.after_sampling(pipe, base_meta, result_extra)
                    save_image(result_img, seed_dir / "result.png")
                    canny_out_path = seed_dir / "canny.png"
                    save_image(conditioning_image, canny_out_path)
                    save_single_runtime_summary(seed_dir, result_elapsed, result_peak_mem)

                    if result_ctx is not None and result_ctx.attn_records:
                        attn_dir = seed_dir / "attn_maps"
                        save_entity_attention_maps(result_ctx.attn_records, canny_out_path, attn_dir)
                        result_ctx.attn_records = []

                    meta = build_seed_meta(
                        args=args,
                        sample_dir=sample_dir,
                        canny_image_path=str(conditioning_image_path) if conditioning_image_path is not None else None,
                        prompt_raw=prompt_raw,
                        prompt_used=prompt_used,
                        seed=seed,
                        adi_steps=result_adi_steps,
                        controlnet_steps=controlnet_steps,
                        baseline_extra=result_extra if result_adi_steps <= 0 else {},
                        adi_extra=result_extra if result_adi_steps > 0 else {},
                        entities_manifest=entities_manifest,
                    )
                    meta["result_mode"] = run_label.lower()
                    meta["result_time_sec"] = result_elapsed
                    meta["result_peak_mem_gb"] = result_peak_mem

                save_json(meta, seed_dir / "meta.json")
                restore_original_processors(targets, original_processors)
                release_cuda_memory()
        else:
            for sample in samples:
                sample_label = f"{sample.dataset_name}/{sample.sample_id}" if sample.dataset_name else sample.sample_id
                print(f"[SDXL] === Sample {sample_label} ===")
                sample_dir, prompt_raw, entities, entities_manifest = load_inputs_from_sample(sample)
                conditioning_image = resolve_conditioning_image(args=args, sample=sample)
                prompt_used = append_positive_suffix(prompt_raw, args.positive_suffix)

                sample_out_dir = run_dir / sample.dataset_name / sample.sample_id if sample.dataset_name else run_dir / sample.sample_id
                ensure_outdir(str(sample_out_dir))
                save_text(prompt_raw, sample_out_dir / "prompt_raw.txt")
                save_text(prompt_used, sample_out_dir / "prompt_used.txt")
                save_json(entities_manifest, sample_out_dir / "entities_used.json")

                adi_steps_cfg = inject_steps(args.steps, args.adi_ratio)
                ctx = None
                if adi_steps_cfg > 0:
                    ctx = build_sdxl_adi_context(
                        pipe=pipe,
                        entities=entities,
                        positive_suffix=args.positive_suffix,
                        enable_global=args.enable_global,
                    )

                for seed in args.seeds:
                    seed_dir = Path(ensure_outdir(str(sample_out_dir / f"seed_{seed}")))
                    restore_original_processors(targets, original_processors)
                    base_meta = build_seed_meta(
                        args=args,
                        sample_dir=sample_dir,
                        canny_image_path=str(sample.canny_path) if sample.canny_path.exists() else None,
                        prompt_raw=prompt_raw,
                        prompt_used=prompt_used,
                        seed=seed,
                        adi_steps=adi_steps_cfg,
                        controlnet_steps=controlnet_steps,
                        baseline_extra={},
                        adi_extra={},
                        entities_manifest=entities_manifest,
                    )
                    base_meta["dataset_name"] = sample.dataset_name
                    base_meta["sample_id"] = sample.sample_id
                    base_meta["raw_image"] = str(sample.raw_image)

                    if args.pair_baseline:
                        print(f"[SDXL][{sample_label}][Seed {seed}] Running baseline ...")
                        method.before_sampling(pipe, base_meta)
                        baseline_img, baseline_extra, baseline_elapsed, baseline_peak_mem = run_generation_timed(
                            pipe=pipe,
                            prompt=prompt_used,
                            negative_prompt=args.negative_prompt,
                            conditioning_image=conditioning_image,
                            seed=seed,
                            args=args,
                            step_hook=step_hook,
                        )
                        method.after_sampling(pipe, base_meta, baseline_extra)
                        save_image(baseline_img, seed_dir / "baseline.png")

                        if adi_steps_cfg > 0 and ctx is not None:
                            adi_steps = reset_context_runtime(ctx, args)
                            attach_adi_processors(targets, ctx, args)

                            print(f"[SDXL][{sample_label}][Seed {seed}] Running ADI ...")
                            method.before_sampling(pipe, base_meta)
                            adi_img, adi_extra, adi_elapsed, adi_peak_mem = run_generation_timed(
                                pipe=pipe,
                                prompt=prompt_used,
                                negative_prompt=args.negative_prompt,
                                conditioning_image=conditioning_image,
                                seed=seed,
                                args=args,
                                ctx=ctx,
                                adi_steps=adi_steps,
                                step_hook=step_hook,
                            )
                            method.after_sampling(pipe, base_meta, adi_extra)
                            adi_path = seed_dir / "adi.png"
                            save_image(adi_img, adi_path)
                            save_image(adi_img, seed_dir / "result.png")
                            save_image(conditioning_image, seed_dir / "canny.png")
                            save_comparison_image(baseline_img, adi_img, seed_dir / "comparison.png")
                            save_runtime_summaries(seed_dir, baseline_elapsed, adi_elapsed, baseline_peak_mem, adi_peak_mem)

                            if ctx.attn_records:
                                attn_dir = seed_dir / "adi_attn_maps"
                                save_entity_attention_maps(ctx.attn_records, adi_path, attn_dir)
                                ctx.attn_records = []

                            meta = build_seed_meta(
                                args=args,
                                sample_dir=sample_dir,
                                canny_image_path=str(sample.canny_path) if sample.canny_path.exists() else None,
                                prompt_raw=prompt_raw,
                                prompt_used=prompt_used,
                                seed=seed,
                                adi_steps=adi_steps,
                                controlnet_steps=controlnet_steps,
                                baseline_extra=baseline_extra,
                                adi_extra=adi_extra,
                                entities_manifest=entities_manifest,
                            )
                            meta["result_mode"] = "adi"
                            meta["baseline_time_sec"] = baseline_elapsed
                            meta["adi_time_sec"] = adi_elapsed
                            meta["baseline_peak_mem_gb"] = baseline_peak_mem
                            meta["adi_peak_mem_gb"] = adi_peak_mem
                        else:
                            print(f"[SDXL][{sample_label}][Seed {seed}] ADI disabled (adi_ratio=0); skipping ADI pass.")
                            save_image(baseline_img, seed_dir / "result.png")
                            save_image(conditioning_image, seed_dir / "canny.png")
                            save_single_runtime_summary(seed_dir, baseline_elapsed, baseline_peak_mem)

                            meta = build_seed_meta(
                                args=args,
                                sample_dir=sample_dir,
                                canny_image_path=str(sample.canny_path) if sample.canny_path.exists() else None,
                                prompt_raw=prompt_raw,
                                prompt_used=prompt_used,
                                seed=seed,
                                adi_steps=0,
                                controlnet_steps=controlnet_steps,
                                baseline_extra=baseline_extra,
                                adi_extra={},
                                entities_manifest=entities_manifest,
                            )
                            meta["result_mode"] = "baseline"
                            meta["result_time_sec"] = baseline_elapsed
                            meta["result_peak_mem_gb"] = baseline_peak_mem
                    else:
                        if adi_steps_cfg > 0 and ctx is not None:
                            adi_steps = reset_context_runtime(ctx, args)
                            attach_adi_processors(targets, ctx, args)
                            run_label = "ADI"
                            result_ctx = ctx
                            result_adi_steps = adi_steps
                            print(f"[SDXL][{sample_label}][Seed {seed}] Running ADI ...")
                        else:
                            run_label = "baseline"
                            result_ctx = None
                            result_adi_steps = 0
                            print(f"[SDXL][{sample_label}][Seed {seed}] Running baseline ...")

                        method.before_sampling(pipe, base_meta)
                        result_img, result_extra, result_elapsed, result_peak_mem = run_generation_timed(
                            pipe=pipe,
                            prompt=prompt_used,
                            negative_prompt=args.negative_prompt,
                            conditioning_image=conditioning_image,
                            seed=seed,
                            args=args,
                            ctx=result_ctx,
                            adi_steps=result_adi_steps,
                            step_hook=step_hook,
                        )
                        method.after_sampling(pipe, base_meta, result_extra)
                        save_image(result_img, seed_dir / "result.png")
                        save_image(conditioning_image, seed_dir / "canny.png")
                        save_single_runtime_summary(seed_dir, result_elapsed, result_peak_mem)

                        if result_ctx is not None and result_ctx.attn_records:
                            attn_dir = seed_dir / "attn_maps"
                            save_entity_attention_maps(result_ctx.attn_records, sample.canny_path, attn_dir)
                            result_ctx.attn_records = []

                        meta = build_seed_meta(
                            args=args,
                            sample_dir=sample_dir,
                            canny_image_path=str(sample.canny_path) if sample.canny_path.exists() else None,
                            prompt_raw=prompt_raw,
                            prompt_used=prompt_used,
                            seed=seed,
                            adi_steps=result_adi_steps,
                            controlnet_steps=controlnet_steps,
                            baseline_extra=result_extra if result_adi_steps <= 0 else {},
                            adi_extra=result_extra if result_adi_steps > 0 else {},
                            entities_manifest=entities_manifest,
                        )
                        meta["result_mode"] = run_label.lower()
                        meta["result_time_sec"] = result_elapsed
                        meta["result_peak_mem_gb"] = result_peak_mem

                    meta["dataset_name"] = sample.dataset_name
                    meta["sample_id"] = sample.sample_id
                    meta["raw_image"] = str(sample.raw_image)
                    save_json(meta, seed_dir / "meta.json")
                    restore_original_processors(targets, original_processors)
                    release_cuda_memory()
    finally:
        restore_original_processors(targets, original_processors)

    print("[SDXL] Done.")


if __name__ == "__main__":
    main()





