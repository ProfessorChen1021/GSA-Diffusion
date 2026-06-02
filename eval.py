"""
Evaluation script: masked CLIP scoring.
Fixed for nested dataset structures (for example, category folders under dataset roots).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Quick local defaults.

# Experiment output root
TARGET_ROOT = "outputs"

# Method folders to compare
TARGET_METHODS = [
    # Leave empty to auto-discover all immediate method folders under TARGET_ROOT.
]

# Dataset roots
# Add extra dataset roots if your masks are stored outside the default datasets/ folder.
# Include all parent roots that contain <dataset>/<sample>/processed/masks.
EXTRA_DATASET_ROOTS = [
    "datasets",
]

# CLIP model path or model id
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

# Other parameters
TARGET_DEVICE = "cuda"
OUTPUT_CSV = "eval_result_nested_fix.csv"
MASK_THRESH = 128

try:
    from utils.common import load_config, resolve_path
except ImportError:
    def load_config(path): return {}, Path(".")
    def resolve_path(path, base): return Path(path)

@dataclass
class EntityInfo:
    name: str
    attr: str

@dataclass
class SampleAssets:
    dataset: str
    sample_id: str
    masks: Dict[str, Path]
    entities: List[EntityInfo]

@dataclass
class SampleRecord:
    method: str
    dataset: str
    sample_id: str
    seed: int
    prompt_raw: str
    image_path: Path

def parse_args_bypassed() -> argparse.Namespace:
    print("[Eval] Using built-in defaults.")
    ns = argparse.Namespace()
    ns.root = TARGET_ROOT
    ns.methods = TARGET_METHODS
    ns.config = "config/default.yaml"
    ns.dataset_roots = EXTRA_DATASET_ROOTS
    ns.clip_model = CLIP_MODEL_ID
    ns.device = TARGET_DEVICE
    ns.mask_thresh = MASK_THRESH
    ns.output_csv = OUTPUT_CSV
    ns.output_json = OUTPUT_CSV.replace(".csv", ".json")
    ns.verbose = True
    return ns

def _dedup_paths(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    uniq: List[Path] = []
    for p in paths:
        key = p.resolve()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(key)
    return uniq

def resolve_dataset_roots(cfg: dict, cfg_dir: Path, extra: Optional[Sequence[str]]) -> List[Path]:
    candidates: List[Path] = []
    if extra:
        for p in extra:
            pp = Path(p)
            candidates.append(pp)

    ds_default = Path("datasets")
    if ds_default.exists():
        candidates.append(ds_default.resolve())

    valid = []
    for p in candidates:
        if p.exists():
            valid.append(p)
        else:
            # Skip missing paths silently to avoid noisy logs.
            pass

    return _dedup_paths(valid)

def build_mask_index(dataset_roots: List[Path], verbose: bool = False) -> Dict[Tuple[str, str], SampleAssets]:
    index: Dict[Tuple[str, str], SampleAssets] = {}

    print(f"\n[Dataset Scan] Scanning {len(dataset_roots)} root paths...")

    for root in dataset_roots:
        # print(f" -> Checking root: {root.name}")

        # ds_dir corresponds to the dataset folder name (for example, "car bus bike").
        subdirs = [d for d in root.iterdir() if d.is_dir()]

        for ds_dir in subdirs:
            # Check whether this is a valid dataset folder (must contain sample subfolders).
            sample_dirs = [d for d in ds_dir.iterdir() if d.is_dir()]
            valid_samples = 0

            for sample_dir in sample_dirs:
                # Expected structure: root/dataset/sample/processed/masks
                masks_dir = sample_dir / "processed" / "masks"
                entities_path = sample_dir / "processed" / "entities.json"

                if not masks_dir.exists() or not entities_path.exists():
                    continue

                # Load entities
                try:
                    with open(entities_path, "r", encoding="utf-8") as f:
                        entities_raw = json.load(f)
                except Exception:
                    continue

                if not isinstance(entities_raw, list): continue

                mask_paths = {
                    p.stem: p for p in masks_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                }

                entities: List[EntityInfo] = []
                for ent in entities_raw:
                    if not isinstance(ent, dict): continue
                    name = ent.get("phrase_entity")
                    attr = ent.get("phrase_ent_attr") or name
                    if not name: continue
                    entities.append(EntityInfo(name=name, attr=attr or name))

                if not entities or not mask_paths:
                    continue

                key = (ds_dir.name, sample_dir.name)
                # Keep the first indexed entry for duplicate keys.
                if key not in index:
                    index[key] = SampleAssets(
                        dataset=ds_dir.name,
                        sample_id=sample_dir.name,
                        masks=mask_paths,
                        entities=entities,
                    )
                    valid_samples += 1

            if valid_samples > 0:
                print(f"    [OK] Indexed Dataset: '{ds_dir.name}' ({valid_samples} samples)")

    print(f"[Dataset Scan] Total samples indexed: {len(index)}")
    return index

def parse_methods(root: Path, methods: Optional[Sequence[str]]) -> List[Tuple[str, Path]]:
    parsed: List[Tuple[str, Path]] = []

    # When no explicit list is provided, auto-discover immediate method folders.
    if not methods:
        if not root.exists():
            print(f"[Error] Target root NOT FOUND: {root}")
            return parsed
        discovered = [p for p in sorted(root.iterdir()) if p.is_dir()]
        for p in discovered:
            parsed.append((p.name, p.resolve()))
        print(f"[Scan] Auto-discovered {len(parsed)} method folders under {root}")
        return parsed

    for entry in methods:
        if "=" in entry:
            name, path_str = entry.split("=", 1)
            path = Path(path_str)
        else:
            name, path = entry, Path(entry)

        if not path.is_absolute():
            path = (root / path).resolve()

        if not path.exists():
            print(f"[Error] Method path NOT FOUND: {path}")
        else:
            parsed.append((name, path))

    return parsed

def collect_samples(method: str, method_dir: Path, verbose: bool = False) -> List[SampleRecord]:
    meta_files = sorted(method_dir.rglob("meta.json"))
    records: List[SampleRecord] = []

    if not meta_files:
        print(f"[Warn] No meta.json found in {method_dir}")
        return records

    for meta_path in meta_files:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue

        dataset = str(meta.get("dataset_name", "")).strip()
        sample_id = str(meta.get("sample_id", "")).strip()

        # Path fallback logic
        if not dataset or not sample_id:
            try:
                # Assume layout: .../dataset_name/sample_id/seed_xx/meta.json
                parts = meta_path.parts
                sample_id = parts[-3]
                dataset = parts[-4]
            except:
                pass

        seed = meta.get("seed", -1)
        prompt_raw = str(meta.get("prompt_raw", "")).strip()
        image_path = meta_path.parent / "result.png"

        if image_path.exists() and dataset and sample_id:
            records.append(
                SampleRecord(
                    method=method,
                    dataset=dataset,
                    sample_id=sample_id,
                    seed=seed,
                    prompt_raw=prompt_raw,
                    image_path=image_path,
                )
            )

    if verbose:
        print(f"[Scan] {method}: Found {len(records)} images.")
    return records

def load_mask(path: Path) -> Image.Image:
    return Image.open(path).convert("L")

def apply_mask(image: Image.Image, mask: Image.Image, thresh: int) -> Image.Image:
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.NEAREST)
    mask_np = (np.array(mask) >= thresh).astype(np.uint8)
    img_np = np.array(image.convert("RGB"))
    masked = img_np * mask_np[..., None]
    return Image.fromarray(masked)

class ClipScorer:
    def __init__(self, model_path: str, device: str):
        print(f"[Init] Loading CLIP from: {model_path}")
        try:
            device_obj = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
            self.device = device_obj
            try:
                self.model = CLIPModel.from_pretrained(model_path, local_files_only=True).to(self.device)
                self.processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
            except Exception:
                self.model = CLIPModel.from_pretrained(model_path, local_files_only=False).to(self.device)
                self.processor = CLIPProcessor.from_pretrained(model_path, local_files_only=False)
            self.model.eval()
        except Exception as e:
            print(f"[Fatal Error] CLIP load failed: {e}")
            sys.exit(1)
        self.text_cache: Dict[str, torch.Tensor] = {}

    def encode_text(self, text: str) -> torch.Tensor:
        if text in self.text_cache:
            return self.text_cache[text]
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            feat = self.model.get_text_features(**inputs)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        self.text_cache[text] = feat
        return feat

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            feat = self.model.get_image_features(**inputs)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat

    def score(self, image: Image.Image, text: str) -> float:
        img_feat = self.encode_image(image)
        txt_feat = self.encode_text(text)
        return float(torch.mm(img_feat, txt_feat.T).item())

def evaluate_records(records: List[SampleRecord], mask_index: Dict, scorer: ClipScorer, mask_thresh: int, verbose: bool) -> List[dict]:
    results = []
    total = len(records)
    print(f"\nEvaluating {total} samples...")

    missing_datasets = set()

    for i, rec in enumerate(records):
        if i % 20 == 0: print(f"Processing {i}/{total}...", end="\r")

        key = (rec.dataset, rec.sample_id)
        assets = mask_index.get(key)

        if not assets:
            missing_datasets.add(rec.dataset)
            continue

        try:
            img = Image.open(rec.image_path).convert("RGB")
        except:
            continue

        for ent in assets.entities:
            mask_path = assets.masks.get(ent.name)
            if not mask_path: continue

            masked_img = apply_mask(img, load_mask(mask_path), mask_thresh)

            r_score = scorer.score(masked_img, ent.attr)

            leak_targets = [e.attr for e in assets.entities if e.name != ent.name]
            leak_scores = [scorer.score(masked_img, t) for t in leak_targets]
            l_score = float(np.mean(leak_scores)) if leak_scores else None

            results.append({
                "method": rec.method,
                "dataset": rec.dataset,
                "prompt_id": rec.sample_id,
                "seed": rec.seed,
                "entity": ent.name,
                "text_r": ent.attr,
                "text_l": "|".join(leak_targets),
                "r_clip": r_score,
                "l_clip": l_score,
                "image_path": str(rec.image_path)
            })

    if missing_datasets:
        print(f"\n[Warn] Skipped samples because Masks were not found for datasets: {list(missing_datasets)[:5]}...")
        print("Note: If output dataset folder names do not match mask dataset folder names, evaluation will skip those samples.")
        print("Rename one side or provide a mapping layer before evaluation.")

    return results

def print_summary(rows: List[dict]):
    from collections import defaultdict
    stats = defaultdict(lambda: {"r": [], "l": []})
    for r in rows:
        stats[r["method"]]["r"].append(r["r_clip"])
        if r["l_clip"] is not None:
            stats[r["method"]]["l"].append(r["l_clip"])

    print("\n" + "="*40)
    print("           METRIC SUMMARY")
    print("="*40)
    for method, d in sorted(stats.items()):
        r_mean = np.mean(d["r"]) if d["r"] else 0
        l_mean = np.mean(d["l"]) if d["l"] else 0
        print(f"[{method}]")
        print(f"  R-CLIP (Alignment): {r_mean:.4f}  (Higher is better)")
        print(f"  L-CLIP (Leakage)  : {l_mean:.4f}  (Lower is better)")
        print("-" * 40)

def main():
    args = parse_args_bypassed()

    # 1) Build mask index
    dataset_roots = resolve_dataset_roots({}, Path("."), args.dataset_roots)
    mask_index = build_mask_index(dataset_roots, verbose=True)

    if not mask_index:
        print("\n[CRITICAL ERROR] No Mask Data Found!")
        return

    # 2) Parse methods
    root = Path(args.root).resolve()
    method_dirs = parse_methods(root, args.methods)
    if not method_dirs:
        return

    # 3) Load model
    scorer = ClipScorer(args.clip_model, args.device)

    # 4) Evaluate
    all_rows = []
    for name, path in method_dirs:
        recs = collect_samples(name, path, verbose=True)
        rows = evaluate_records(recs, mask_index, scorer, args.mask_thresh, verbose=True)
        all_rows.extend(rows)

    if not all_rows:
        print("\n[Final Error] No scored rows were produced.")
        print("Most likely cause: dataset folder names in outputs and mask roots do not match.")
        print("Please rename folders or add explicit name mapping before running evaluation.")
        return

    csv_path = Path(args.output_csv)
    fieldnames = ["method", "dataset", "prompt_id", "seed", "entity", "text_r", "text_l", "r_clip", "l_clip", "image_path"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved details to: {csv_path.absolute()}")
    print_summary(all_rows)

if __name__ == "__main__":
    main()


