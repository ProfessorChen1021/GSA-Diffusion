"""
Run preprocessing and generation for the SD1.5 experiments.

Each sample folder should contain an input image and a prompt. The pipeline
fills `processed/` on demand with parsed entities, masks, and a Canny map,
then generates outputs for each configured seed.
"""

from __future__ import annotations

import csv
import importlib
import json
import os
import math
import datetime
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from src.controlnet_sd15 import SD15ControlNetPipeline
from methods.adi_attention import ADIContext, ADICrossAttnProcessor
from src.planner import PromptEntityPlanner, PlannerConfig
from src.grounder import generate_entity_masks

from utils.image import list_images, make_canny_image, relative_to
from utils.io import ensure_outdir, save_image, save_text
from utils.seed import make_torch_generator, seed_everything
from utils.timers import Timer
from utils.paths import resolve_model_identifier
from utils.common import dump_run_config
from utils.visualization import save_entity_attention_maps


LLMFn = Callable[[str], str]


@dataclass
class SamplePaths:
    """Resolved paths for one sample folder."""
    dataset_name: str
    sample_id: str
    root: Path
    raw_image: Path
    raw_prompt: Path
    processed_dir: Path
    canny_path: Path
    entities_json: Path
    masks_dir: Path


class GSADiffusionPipeline:
    """Tie together prompt parsing, mask generation, and image sampling."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        config_dir: Path | str,
        llm_fn: Optional[LLMFn] = None,
    ) -> None:
        self.cfg = cfg
        self.config_dir = Path(config_dir)

        paths_cfg = cfg["paths"]
        data_cfg = cfg.get("data", {})
        params_cfg = cfg["params"]
        controlnet_cfg = cfg["controlnet"]
        grounder_cfg = cfg.get("grounder", {})
        prompt_cfg = cfg.get("prompt_config", {})
        injection_cfg = cfg.get("injection", {})
        adi_cfg = cfg.get("adi", {})

        # ----------------- Paths -----------------
        self.base_model_path = resolve_model_identifier(paths_cfg["base_model_path"])
        self.dataset_root = self._resolve_path(paths_cfg["dataset_root"])
        self.output_root = self._resolve_path(paths_cfg["output_root"])
        self.dataset_mode = str(data_cfg.get("mode", "auto")).strip().lower()
        default_manifest = self.dataset_root / "benchmark_prompts.csv"
        self.prompt_manifest_path = self._resolve_path(data_cfg.get("prompt_manifest", default_manifest))
        self.prompt_manifest_prompt_col = str(data_cfg.get("prompt_column", "prompt"))
        self.prompt_manifest_scene_col = str(data_cfg.get("scene_column", "scene"))
        self.prompt_manifest_sample_col = str(data_cfg.get("sample_id_column", "sample_id"))
        self.prompt_manifest_group_col = str(data_cfg.get("dataset_group_column", "dataset_group"))
        self.prompt_manifest_id_col = str(data_cfg.get("prompt_id_column", "prompt_id"))

        # ----------------- Generation parameters -----------------
        self.width = int(params_cfg["width"])
        self.height = int(params_cfg["height"])
        self.num_inference_steps = int(params_cfg["num_inference_steps"])
        self.guidance_scale = float(params_cfg["guidance_scale"])
        self.scheduler_name = params_cfg["scheduler"]
        self.dtype = params_cfg.get("dtype", "float16")
        self.enable_vae_slicing = bool(params_cfg.get("enable_vae_slicing", True))
        self.enable_xformers = bool(params_cfg.get("enable_xformers", True))
        self.seeds: List[int] = list(params_cfg.get("seeds", [42]))
        self.max_samples: Optional[int] = params_cfg.get("max_samples")

        # ----------------- ControlNet (Canny) -----------------
        self.controlnet_model_path = resolve_model_identifier(controlnet_cfg["model_path"])
        self.cn_scale = float(controlnet_cfg.get("conditioning_scale", 1.0))
        self.cn_guess = bool(controlnet_cfg.get("guess_mode", False))
        self.canny_low = int(controlnet_cfg.get("canny_low_threshold", 100))
        self.canny_high = int(controlnet_cfg.get("canny_high_threshold", 200))

        # ----------------- Grounded-SAM -----------------
        self.dino_id = resolve_model_identifier(
            grounder_cfg.get("dino_id", "IDEA-Research/grounding-dino-base")
        )
        self.sam_id = resolve_model_identifier(
            grounder_cfg.get("sam_id", "facebook/sam-vit-base")
        )
        self.box_threshold = float(grounder_cfg.get("box_threshold", 0.35))
        self.text_threshold = float(grounder_cfg.get("text_threshold", 0.25))
        self.grounder_use_attr_prompt = bool(grounder_cfg.get("use_attr_prompt", False))

        # ----------------- Prompt / negative prompt -----------------
        self.negative_prompt = prompt_cfg.get("negative_prompt", "")
        self.prompt_positive_suffix = prompt_cfg.get("positive_suffix", "")

        # ----------------- Injection ratios (0~1) -----------------
        self.controlnet_ratio = self._clamp01(injection_cfg.get("controlnet_ratio", 1.0))
        self.adi_ratio = self._clamp01(injection_cfg.get("adi_ratio", 1.0))
        self.adi_linear_decay = bool(injection_cfg.get("adi_linear_decay", True))
        # ----------------- ADI weights -----------------
        self.adi_alpha_g = float(adi_cfg.get("alpha_g", 0.2))
        self.adi_alpha_e = float(adi_cfg.get("alpha_e", 1.2))
        self.adi_sharpen_t = float(adi_cfg.get("sharpen_t", 1.0))
        self.adi_enable_self_iso = bool(adi_cfg.get("enable_self_iso", True))
        self.adi_enable_global = bool(adi_cfg.get("enable_global", True))

        # ----------------- Dynamic method loading -----------------
        method_mod = importlib.import_module(cfg["method"])
        method_kwargs = cfg.get("method_kwargs", {})
        self.method = method_mod.Method(**method_kwargs)

        # ----------------- SD15 + ControlNet pipeline -----------------
        print("[Pipeline] Loading SD1.5 and ControlNet...")
        self.pipe = SD15ControlNetPipeline(
            base_model_path=self.base_model_path,
            controlnet_model_path=self.controlnet_model_path,
            scheduler_name=self.scheduler_name,
            dtype=self.dtype,
            enable_vae_slicing=self.enable_vae_slicing,
            enable_xformers=self.enable_xformers,
        )

        if hasattr(self.pipe, "device"):
            self.device = self.pipe.device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.timer = Timer()
        seed_everything(0)

        # ----------------- LLM planner -----------------
        planner_cfg_dict = cfg.get("planner", {})
        planner_cfg = PlannerConfig(mode=planner_cfg_dict.get("mode", "rule"))
        print(f"[Pipeline] Planner mode = {planner_cfg.mode}")
        if planner_cfg.mode == "llm" and llm_fn is None:
            raise RuntimeError("cfg.planner.mode='llm' but GSADiffusionPipeline did not receive llm_fn")
        self.planner = PromptEntityPlanner(cfg=planner_cfg, llm_fn=llm_fn)

        # Global sample counter (used for output naming).
        self.sample_counter = 0

    # ===================== Main entry =====================

    def run_dataset(self, outputs_root: Optional[Path] = None) -> None:
        samples = self._discover_samples(max_samples=self.max_samples)
        sample_count = len(samples)
        seed_count = len(self.seeds)
        selection_info = getattr(self, "_last_sample_selection_info", "")

        if outputs_root is None:
            now = datetime.datetime.now()
            date_part = now.strftime("%y%m%d")
            time_part = now.strftime("%H%M")
            run_dir_name = f"{sample_count}samples{seed_count}seeds_{date_part}_{time_part}"
            run_dir = ensure_outdir(os.path.join(str(self.output_root), run_dir_name))
            outputs_root = Path(run_dir)
            print(f"[Pipeline] Saving this run to: {outputs_root}")
        else:
            outputs_root = Path(outputs_root)

        config_dump_path = outputs_root / "config.json"
        if not config_dump_path.exists():
            dump_run_config(self.cfg, str(outputs_root))

        print(f"[Pipeline] Found {sample_count} samples under {self.dataset_root}")
        if selection_info:
            print(f"[Pipeline] Sample selection: {selection_info}")

        for sp in samples:
            tag = f"{sp.dataset_name}/{sp.sample_id}" if sp.dataset_name else sp.sample_id
            print(f"\n[Pipeline] === Sample {tag} ===")
            entities, prompt = self._preprocess_sample(sp, overwrite=False)
            self._generate_for_sample(sp, prompt, entities, outputs_root)

        print("[Pipeline] Finished.")

    # ===================== Paths / sample management =====================

    def _resolve_path(self, p: str) -> Path:
        path = Path(p)
        if not path.is_absolute():
            # Use the parent of config.yaml as project root.
            path = (self.config_dir.parent / path).resolve()
        return path

    def _clamp01(self, v: Any) -> float:
        try:
            f = float(v)
        except Exception:
            return 1.0
        return max(0.0, min(1.0, f))

    def _inject_steps(self, ratio: float) -> int:
        ratio = self._clamp01(ratio)
        return int(math.ceil(self.num_inference_steps * ratio))

    def _append_positive_suffix(self, text: str) -> str:
        suffix = self.prompt_positive_suffix.strip()
        return f"{text}, {suffix}" if suffix else text

    def _enumerate_duplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Auto-enumerate duplicate entities: cat, cat, cat -> cat_1, cat_2, cat_3
        Keep original base name in phrase_entity_base.
        """
        totals: Dict[str, int] = defaultdict(int)
        for ent in entities:
            base = str(ent.get("phrase_entity", "")).strip()
            if base:
                totals[base] += 1

        running: Dict[str, int] = defaultdict(int)
        enumerated: List[Dict[str, Any]] = []
        for ent in entities:
            base = str(ent.get("phrase_entity", "")).strip()
            if not base:
                # Skip malformed entries.
                continue
            running[base] += 1
            use_suffix = totals[base] > 1
            new_name = f"{base}_{running[base]}" if use_suffix else base

            new_ent = dict(ent)
            new_ent["phrase_entity_base"] = base
            new_ent["phrase_entity"] = new_name
            enumerated.append(new_ent)

        return enumerated

    def _build_grounder_targets(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert entities into GroundingDINO/SAM detection targets:
            Use phrase_ent_attr (including color/appearance attributes) to disambiguate same-name instances.
            Use enumerated phrase_entity names to keep mask save/load consistent.
        """
        groups: Dict[str, List[str]] = {}
        for ent in entities:
            name = ent.get("phrase_entity")
            if not name:
                continue
            base_prompt = ent.get("phrase_entity_base") or name
            prompt = ent.get("phrase_ent_attr") if getattr(self, "grounder_use_attr_prompt", False) else base_prompt
            groups.setdefault(str(prompt), []).append(str(name))

        targets = [{"prompt": prompt, "names": names} for prompt, names in groups.items()]
        return targets

    def _discover_samples(self, max_samples: Optional[Any] = None) -> List[SamplePaths]:
        root = self.dataset_root
        if not root.exists():
            raise FileNotFoundError(f"Dataset root not found: {root}")

        # Prefer datasets/<desc>/<digit>/ layout; fallback to legacy datasets/<digit>/ if needed.
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
        samples_flat.extend([("", p) for p in direct_numeric])

        if not samples_flat:
            manifest_message = self._build_public_manifest_message()
            if manifest_message:
                raise RuntimeError(manifest_message)
            raise RuntimeError(f"No valid samples found under {root}")

        samples_flat.sort(key=lambda x: (x[0], int(x[1].name)))

        selection_info = "all"
        # Selection modes: None -> all; int -> first N; [start,end] -> inclusive range filter.
        if max_samples is not None:
            if isinstance(max_samples, int):
                n = max(0, max_samples)
                selection_info = f"first {n}"
                samples_flat = samples_flat[:n]
            elif isinstance(max_samples, (list, tuple)) and len(max_samples) == 2:
                if not all(isinstance(x, (int, float)) for x in max_samples):
                    raise ValueError("max_samples list must contain two numeric values, e.g., [13,78] or [0,0]")
                start = int(max_samples[0])
                end = int(max_samples[1])
                if start == 0 and end == 0:
                    selection_info = "all (range [0,0])"
                else:
                    if start > end:
                        start, end = end, start
                    selection_info = f"range [{start},{end}]"
                    samples_flat = [(ds, p) for ds, p in samples_flat if start <= int(p.name) <= end]
            else:
                raise ValueError("max_samples must be None, int, or a two-element list/tuple like [start, end]")

        samples: List[SamplePaths] = []
        for dataset_name, d in samples_flat:
            sid = d.name
            tag = f"{dataset_name}/{sid}" if dataset_name else sid
            imgs = list_images(d)
            if len(imgs) != 1:
                raise RuntimeError(f"[{tag}] Expect exactly one image in {d}, found {len(imgs)}")
            raw_image = imgs[0]
            raw_prompt = d / "prompt.txt"
            if not raw_prompt.exists():
                raise FileNotFoundError(f"[{tag}] Missing prompt.txt in {d}")

            processed_dir = d / "processed"
            canny_path = processed_dir / "canny.png"
            entities_json = processed_dir / "entities.json"
            masks_dir = processed_dir / "masks"

            samples.append(
                SamplePaths(
                    dataset_name=dataset_name,
                    sample_id=sid,
                    root=d,
                    raw_image=raw_image,
                    raw_prompt=raw_prompt,
                    processed_dir=processed_dir,
                    canny_path=canny_path,
                    entities_json=entities_json,
                    masks_dir=masks_dir,
                )
            )

        dataset_count = len({s.dataset_name or "<root>" for s in samples})
        self._last_sample_selection_info = f"{selection_info}, {dataset_count} dataset folders -> {len(samples)} selected"
        return samples

    def _build_public_manifest_message(self) -> Optional[str]:
        manifest_path = self.prompt_manifest_path
        if self.dataset_mode not in {"auto", "public_manifest"}:
            return None
        if not manifest_path.exists() or not manifest_path.is_file():
            return None

        row_count = 0
        unique_prompts = set()
        try:
            with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_count += 1
                    unique_prompts.add(
                        (
                            str(row.get(self.prompt_manifest_group_col, "")).strip(),
                            str(row.get(self.prompt_manifest_scene_col, "")).strip(),
                            str(row.get(self.prompt_manifest_sample_col, "")).strip(),
                            str(row.get(self.prompt_manifest_id_col, "")).strip(),
                            str(row.get(self.prompt_manifest_prompt_col, "")).strip(),
                        )
                    )
        except Exception:
            return (
                f"No sample folders were found under {self.dataset_root}, and the public prompt manifest "
                f"could not be read from {manifest_path}."
            )

        return (
            f"No sample folders were found under {self.dataset_root}. "
            f"This public release currently ships a prompt manifest only: {manifest_path} "
            f"({len(unique_prompts)} unique prompts, {row_count} CSV rows). "
            f"Full generation still needs sample folders with one input image, prompt.txt, and processed/ assets."
        )

    # ===================== Preprocess: Step1~3 =====================

    def _load_prompt_text(self, path: Path) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def _save_masks_as_png(self, entities: List[Dict[str, Any]], masks_dir: Path) -> None:
        """
        Save masks as PNG using entities[*]["phrase_entity"] -> mask_tensor mapping.
        """
        ensure_outdir(str(masks_dir))
        for ent in entities:
            name = ent["phrase_entity"]
            mask = ent.get("mask_tensor", None)
            if mask is None:
                continue
            m = mask.detach().cpu().float()
            if m.ndim == 4:
                m = m[0, 0]
            arr = (m.clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            img = Image.fromarray(arr)
            img.save(masks_dir / f"{name}.png")

    def _load_masks_from_png(self, entities: List[Dict[str, Any]], masks_dir: Path) -> None:
        """
        Load PNG masks from masks_dir back into entities[*]["mask_tensor"].
        """
        if not masks_dir.exists():
            return
        cache: Dict[str, torch.Tensor] = {}
        for p in masks_dir.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            name = p.stem
            img = Image.open(p).convert("L")
            arr = np.array(img, dtype=np.float32) / 255.0
            t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            cache[name] = t

        for ent in entities:
            name = ent["phrase_entity"]
            ent["mask_tensor"] = cache.get(name, None)

    def _save_entities_json(self, entities: List[Dict[str, Any]], path: Path) -> None:
        """
        Persist text-only entity fields (do not serialize mask_tensor).
        """
        serializable: List[Dict[str, Any]] = []
        for ent in entities:
            serializable.append(
                {
                    "phrase_entity": ent["phrase_entity"],
                    "phrase_ent_attr": ent["phrase_ent_attr"],
                    "phrase_entity_base": ent.get("phrase_entity_base", ent["phrase_entity"]),
                }
            )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

    def _load_entities_json(self, path: Path) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"entities.json must be a list, got {type(data)}")

        entities: List[Dict[str, Any]] = []
        for idx, ent in enumerate(data):
            if not isinstance(ent, dict):
                raise ValueError(f"Entity #{idx} in entities.json is not an object.")
            phrase_entity = ent.get("phrase_entity")
            phrase_ent_attr = ent.get("phrase_ent_attr", phrase_entity)
            phrase_entity_base = ent.get("phrase_entity_base", phrase_entity)
            if not isinstance(phrase_entity, str):
                raise ValueError(f"Entity #{idx} missing valid 'phrase_entity'.")
            if not isinstance(phrase_ent_attr, str):
                raise ValueError(f"Entity #{idx} 'phrase_ent_attr' must be str.")
            if phrase_entity_base is None:
                phrase_entity_base = phrase_entity
            entities.append(
                {
                    "phrase_entity": phrase_entity,
                    "phrase_ent_attr": phrase_ent_attr,
                    "phrase_entity_base": phrase_entity_base,
                    "mask_tensor": None,
                }
            )
        return entities

    def _preprocess_sample(
        self,
        sp: SamplePaths,
        overwrite: bool = False,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Step 1: planner/LLM parses prompt -> entities (phrase_entity / phrase_ent_attr)
        Step 2: Grounded-SAM fills entities[*]["mask_tensor"]
        Step 3: Canny -> canny.png

        Reuse processed cache by default unless overwrite=True.
        """
        prompt_text = self._load_prompt_text(sp.raw_prompt)
        tag = f"{sp.dataset_name}/{sp.sample_id}" if sp.dataset_name else sp.sample_id

        # Load directly when cache exists.
        if sp.processed_dir.exists() and not overwrite:
            print(f"[{tag}] Using cached processed/ ...")
            if sp.entities_json.exists():
                entities = self._load_entities_json(sp.entities_json)
            else:
                entities = []

            self._load_masks_from_png(entities, sp.masks_dir)
            if sp.canny_path.exists():
                return entities, prompt_text
            # If canny.png is missing, rerun only Step 3.

            ensure_outdir(str(sp.processed_dir))
            canny_img = make_canny_image(
                path=sp.raw_image,
                width=self.width,
                height=self.height,
                low_threshold=self.canny_low,
                high_threshold=self.canny_high,
            )
            canny_img.save(sp.canny_path)
            return entities, prompt_text

        # No cache or overwrite requested -> recompute all preprocess steps.
        ensure_outdir(str(sp.processed_dir))
        ensure_outdir(str(sp.masks_dir))

        # Step 1: Planner (rule / LLM)
        print(f"[{tag}] Planner running ...")
        entities = self.planner.plan(prompt_text, sp.root)
        entities = self._enumerate_duplicate_entities(entities)
        self._save_entities_json(entities, sp.entities_json)

        # Step 2: Grounded-SAM
        grounder_targets = self._build_grounder_targets(entities)
        entity_names = [n for t in grounder_targets for n in t["names"]]
        print(f"[{tag}] Grounded-SAM entities: {entity_names}")

        self.timer.reset()
        masks_dict = generate_entity_masks(
            image_path=str(sp.raw_image),
            entity_targets=grounder_targets,
            dino_id=self.dino_id,
            sam_id=self.sam_id,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )
        print(f"[{tag}] Grounded-SAM done in {self.timer.elapsed():.3f}s")

        for e in entities:
            name = e["phrase_entity"]
            e["mask_tensor"] = masks_dict.get(name, None)

        self._save_masks_as_png(entities, sp.masks_dir)

        # Step 3: Canny
        canny_img = make_canny_image(
            path=sp.raw_image,
            width=self.width,
            height=self.height,
            low_threshold=self.canny_low,
            high_threshold=self.canny_high,
        )
        canny_img.save(sp.canny_path)

        return entities, prompt_text

    # ===================== ADI build & hook =====================

    def _build_adi_context(self, entities: List[Dict[str, Any]]) -> ADIContext:
        """
        entities already match ADIContext expected structure:
            - phrase_entity
            - phrase_ent_attr
            - mask_tensor
        """
        style = self.prompt_positive_suffix.strip()
        styled_entities: List[Dict[str, Any]] = []
        for ent in entities:
            e = dict(ent)
            if style:
                base_attr = e.get("phrase_ent_attr") or e.get("phrase_entity", "")
                e["phrase_ent_attr"] = f"{base_attr}, {style}" if base_attr else style
            styled_entities.append(e)

        ctx = ADIContext(
            entities=styled_entities,
            tokenizer=self.pipe.pipe.tokenizer,
            text_encoder=self.pipe.pipe.text_encoder,
            device=str(self.device),
            dtype=self.pipe.dtype if hasattr(self.pipe, "dtype") else torch.float16,
            enable_global=self.adi_enable_global,
        )
        return ctx

    def _attach_adi(self, ctx: ADIContext) -> None:
        targets: List[Tuple[str, Any]] = []
        for name, module in self.pipe.pipe.unet.named_modules():
            if ("up_blocks" in name or "mid_block" in name) and ("attn1" in name or "attn2" in name):
                targets.append((name, module))

        print(f"[ADI] Hooking into {len(targets)} cross-attention layers...")
        for name, module in targets:
            if hasattr(module, "set_processor"):
                proc = ADICrossAttnProcessor(
                    ctx,
                    layer_name=name,
                    alpha_g=self.adi_alpha_g,
                    alpha_e=self.adi_alpha_e,
                    sharpen_t=self.adi_sharpen_t,
                    enable_self_iso=self.adi_enable_self_iso,
                )
                module.set_processor(proc)

    # ===================== Generation =====================

    def _generate_for_sample(
        self,
        sp: SamplePaths,
        prompt: str,
        entities: List[Dict[str, Any]],
        outputs_root: Path,
    ) -> None:
        """
        Full generation flow for one sample:
          - Build ADIContext and attach ADI processors
          - Call pipe.generate for all seeds
        Each (sample, seed) pair writes to its own output folder for reproducibility.
        """
        # Append global style suffix to the raw prompt.
        prompt_with_style = self._append_positive_suffix(prompt)

        # ADI
        ctx = self._build_adi_context(entities)
        self._attach_adi(ctx)
        adi_steps = self._inject_steps(self.adi_ratio)
        controlnet_steps = self._inject_steps(self.controlnet_ratio)
        ctx.injection_enabled = adi_steps > 0
        ctx.injection_strength = 1.0 if adi_steps > 0 else 0.0
        ctx.current_step_idx = 0
        # Capture attention maps at 0.4T and 0.7T (floor index).
        capture_set = set()
        for ratio in (0.4, 0.7):
            idx = int(self.num_inference_steps * ratio)
            if 0 <= idx < self.num_inference_steps:
                capture_set.add(idx)
        ctx.capture_steps = capture_set
        ctx.attn_records = []

        base_dirs = {Path.cwd(), self.dataset_root, sp.root}
        rel_input = relative_to(sp.raw_image, base_dirs)
        rel_canny = relative_to(sp.canny_path, base_dirs)

        base_step_cb = getattr(self.method, "on_step", None)

        def step_callback(step_idx: int, timestep: int, latents) -> None:
            # Callback runs at step end; use step_idx+1 to control next-step ADI state/strength.
            next_step = step_idx + 1

            # 1) Base strength decision
            if adi_steps <= 0:
                strength = 0.0
            elif next_step <= adi_steps:
                strength = 1.0  # Full strength before cutoff step
            else:
                strength = 0.0  # Default off after cutoff

            # 2) Linear decay: start from adi_steps and decay to 0 afterwards.
            if self.adi_linear_decay and self.num_inference_steps > 0 and adi_steps > 0:
                if next_step > adi_steps:
                    remain = self.num_inference_steps - adi_steps
                    if remain > 0:
                        decay_ratio = (next_step - adi_steps) / remain
                        strength = max(0.0, 1.0 - decay_ratio)
                    else:
                        strength = 0.0

            # 3) Apply state
            ctx.injection_strength = strength
            ctx.injection_enabled = strength > 1e-4  # Tiny threshold avoids unnecessary overhead
            ctx.current_step_idx = next_step

            if base_step_cb is not None:
                base_step_cb(step_idx, timestep, latents)

        sample_dir = ensure_outdir(os.path.join(str(outputs_root), sp.dataset_name, sp.sample_id))

        for seed in self.seeds:
            self.sample_counter += 1
            seed_dir = ensure_outdir(os.path.join(sample_dir, f"seed_{seed}"))

            meta = {
                "dataset_name": sp.dataset_name,
                "sample_id": sp.sample_id,
                "sample_index": self.sample_counter,
                "prompt": prompt_with_style,
                "prompt_raw": prompt,
                "negative_prompt": self.negative_prompt,
                "seed": seed,
                "width": self.width,
                "height": self.height,
                "num_inference_steps": self.num_inference_steps,
                "guidance_scale": self.guidance_scale,
                "input_image": rel_input,
                "canny_image": rel_canny,
                "canny_low_threshold": self.canny_low,
                "canny_high_threshold": self.canny_high,
                "controlnet_conditioning_scale": self.cn_scale,
                "controlnet_guess_mode": self.cn_guess,
                "controlnet_inject_ratio": self.controlnet_ratio,
                "adi_inject_ratio": self.adi_ratio,
                "controlnet_inject_steps": controlnet_steps,
                "adi_inject_steps": adi_steps,
                "adi_linear_decay": self.adi_linear_decay,
                "adi_enable_self_iso": self.adi_enable_self_iso,
            }
            save_text(
                json.dumps(meta, indent=2, ensure_ascii=False),
                os.path.join(seed_dir, "meta.json"),
            )

            self.method.before_sampling(self.pipe, meta)
            ctx.injection_enabled = adi_steps > 0

            generator = make_torch_generator(seed)

            self.timer.reset()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            canny_img = Image.open(sp.canny_path).convert("RGB")

            image, extra = self.pipe.generate(
                prompt=prompt_with_style,
                negative_prompt=self.negative_prompt,
                width=self.width,
                height=self.height,
                steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                conditioning_image=canny_img,
                conditioning_scale=self.cn_scale,
                guess_mode=self.cn_guess,
                generator=generator,
                step_callback=step_callback,
                controlnet_end_ratio=self.controlnet_ratio,
            )
            elapsed = self.timer.elapsed()

            self.method.after_sampling(self.pipe, meta, extra)

            save_image(image, os.path.join(seed_dir, "result.png"))
            save_image(canny_img, os.path.join(seed_dir, "canny.png"))
            save_text(f"{elapsed:.3f}s", os.path.join(seed_dir, "time.txt"))

            if ctx.attn_records:
                vis_dir = Path(seed_dir) / "attn_maps"
                save_entity_attention_maps(ctx.attn_records, sp.canny_path, vis_dir)
                ctx.attn_records = []

            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
                with open(os.path.join(seed_dir, "gpu_mem_gb.txt"), "w", encoding="utf-8") as f:
                    f.write(f"{peak:.3f} GB")






