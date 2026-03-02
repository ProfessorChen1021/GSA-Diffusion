# src/grounder.py
# -*- coding: utf-8 -*-
"""
Grounded-SAM helper (Step 2).

generate_entity_masks(
    image_path="path/to/img.png",
    entity_targets=[
        {"prompt": "red bird", "names": ["bird_1", "bird_2"]},
        {"prompt": "blue bird", "names": ["bird_3"]},
    ],
    ...
)

Returns: dict[name] -> mask_tensor (torch[1,1,H,W])
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    SamModel,
    SamProcessor,
)


def _bbox_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two boxes in xyxy format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def _trim_mask_overlaps(masks: Dict[str, torch.Tensor]) -> None:
    """
    For overlapping masks, keep the smaller one intact and subtract the overlap from the larger.
    Modifies masks in place.
    """
    names = list(masks.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            m1 = masks[n1]
            m2 = masks[n2]
            if m1.shape != m2.shape:
                continue

            b1 = m1 > 0.5
            b2 = m2 > 0.5
            inter = b1 & b2
            inter_area = inter.sum().item()
            if inter_area == 0:
                continue

            area1 = b1.sum().item()
            area2 = b2.sum().item()
            if area1 <= 0 or area2 <= 0:
                continue

            # Identify smaller vs larger by mask area
            if area1 <= area2:
                small_name, large_name = n1, n2
            else:
                small_name, large_name = n2, n1

            # Subtract overlap from the larger mask
            new_large = masks[large_name].clone()
            new_large[inter] = 0.0
            masks[large_name] = new_large

            print(f"[Grounder] Trimmed overlap: keep '{small_name}', removed {int(inter_area)} pixels from '{large_name}'.")


def _detect_entity_boxes(
    image: Image.Image,
    entity_prompt: str,
    dino_processor: AutoProcessor,
    dino_model: AutoModelForZeroShotObjectDetection,
    device: str,
    box_threshold: float,
    text_threshold: float,
    top_k: int = 1,
) -> List[torch.Tensor]:
    """
    Run GroundingDINO and return up to top_k boxes (xyxy) sorted by score.
    """
    text_prompt = f"{entity_prompt}."

    inputs = dino_processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dino_model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=device)
    results = dino_processor.image_processor.post_process_object_detection(
        outputs,
        threshold=box_threshold,
        target_sizes=target_sizes,
    )[0]

    boxes = results["boxes"]          # [N,4]
    scores = results["scores"]        # [N]

    if boxes.numel() == 0:
        print(f"[Grounder] No box found for entity prompt '{entity_prompt}'.")
        return []

    keep = scores > text_threshold
    if keep.any():
        boxes = boxes[keep]
        scores = scores[keep]

    if boxes.numel() == 0:
        print(f"[Grounder] No box passed threshold for '{entity_prompt}'.")
        return []

    sorted_idx = torch.argsort(scores, descending=True)
    k = max(1, int(top_k))
    selected = sorted_idx[:k]
    selected_boxes = boxes[selected].detach().cpu()
    return [b for b in selected_boxes]


def generate_entity_masks(
    image_path: str,
    entity_targets: List[Any],
    dino_id: str = "IDEA-Research/grounding-dino-base",
    sam_id: str = "facebook/sam-vit-base",
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
) -> Dict[str, torch.Tensor]:
    """
    Inputs: image path + entity detection requests.
    Outputs: dict[name] -> mask tensor [1,1,H,W].

    entity_targets formats:
      1) ["cat", "rabbit"] -> prompt=name, one mask per name.
      2) [{"prompt": "...", "names": ["bird_1", "bird_2"]}, ...]
         - prompt: text for GroundingDINO (can include attributes to disambiguate)
         - names: names to generate masks for; count decides top_k
         - {"name": "cat"} is treated as names=["cat"], prompt="cat"
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Grounder] Using device: {device}")

    # ----- Load models -----
    print(f"[Grounder] Loading GroundingDINO from: {dino_id}")
    dino_processor = AutoProcessor.from_pretrained(dino_id)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(device)

    print(f"[Grounder] Loading SAM from: {sam_id}")
    sam_processor = SamProcessor.from_pretrained(sam_id)
    sam_model = SamModel.from_pretrained(sam_id).to(device)

    # ----- Read image -----
    image = Image.open(image_path).convert("RGB")

    # ----- Normalize entity_targets -----
    normalized_targets: List[Dict[str, Any]] = []
    if all(isinstance(t, str) for t in entity_targets):
        for name in entity_targets:
            normalized_targets.append({"prompt": str(name), "names": [str(name)]})
    else:
        for idx, t in enumerate(entity_targets):
            if not isinstance(t, dict):
                raise ValueError(f"entity_targets[{idx}] must be str or dict, got {type(t)}")

            names = t.get("names")
            name = t.get("name")
            if names is None:
                if isinstance(name, str):
                    names = [name]
                else:
                    raise ValueError(f"entity_targets[{idx}] missing 'name' or 'names'")
            if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
                raise ValueError(f"entity_targets[{idx}] 'names' must be list[str]")

            prompt = t.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                prompt = name if isinstance(name, str) else names[0]
            normalized_targets.append({"prompt": prompt, "names": names})

    # ----- Run DINO, collect boxes -----
    boxes_for_sam: list[list[float]] = []
    ents_for_sam: list[str] = []

    for target in normalized_targets:
        prompt = target["prompt"]
        names: List[str] = target["names"]

        boxes = _detect_entity_boxes(
            image=image,
            entity_prompt=prompt,
            dino_processor=dino_processor,
            dino_model=dino_model,
            device=device,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            top_k=len(names),
        )
        if not boxes:
            continue

        # If multiple instances are requested but fewer boxes are found, try a relaxed pass.
        if len(boxes) < len(names) and len(names) > 1:
            relaxed_boxes = _detect_entity_boxes(
                image=image,
                entity_prompt=prompt,
                dino_processor=dino_processor,
                dino_model=dino_model,
                device=device,
                box_threshold=max(0.0, box_threshold * 0.8),
                text_threshold=max(0.0, text_threshold * 0.8),
                top_k=len(names) * 2,
            )
            boxes = boxes + relaxed_boxes

        # Deduplicate to favor distinct instances when multiple names are requested.
        if len(names) > 1 and len(boxes) > 1:
            deduped: list[list[float]] = []
            for b in boxes:
                b_list = b.tolist()
                if any(_bbox_iou(prev, b_list) > 0.6 for prev in deduped):
                    continue
                deduped.append(b_list)
            boxes_lists = deduped
        else:
            boxes_lists = [b.tolist() for b in boxes]

        if len(boxes_lists) < len(names):
            print(f"[Grounder] Only {len(boxes_lists)} boxes found for prompt '{prompt}' (requested {len(names)}).")

        for name, box_list in zip(names, boxes_lists):
            boxes_for_sam.append(box_list)
            ents_for_sam.append(name)

    if not boxes_for_sam:
        print("[Grounder] No boxes for any entity, returning empty dict.")
        return {}

    # SAM expects a batch: [[box1, box2, ...]]
    sam_inputs = sam_processor(image, input_boxes=[boxes_for_sam], return_tensors="pt").to(device)
    with torch.no_grad():
        sam_outputs = sam_model(**sam_inputs)

    masks = sam_processor.image_processor.post_process_masks(
        sam_outputs.pred_masks,
        sam_inputs["original_sizes"],
        sam_inputs["reshaped_input_sizes"],
    )
    iou_scores = sam_outputs.iou_scores  # [1, N, 3]

    out_masks: Dict[str, torch.Tensor] = {}
    for i, name in enumerate(ents_for_sam):
        # pick the layer with max IoU
        best_idx = torch.argmax(iou_scores[0, i]).item()
        m = masks[0][i][best_idx]  # [H,W]

        m = m.float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        out_masks[name] = m

    # Remove overlapping regions from larger masks to favor smaller ones.
    _trim_mask_overlaps(out_masks)

    return out_masks
