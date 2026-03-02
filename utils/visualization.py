# utils/visualization.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Dict

import numpy as np
from PIL import Image
import cv2


def _normalize(arr: np.ndarray) -> np.ndarray:
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-8:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def _overlay_heatmap(base_rgb: np.ndarray, heatmap_norm: np.ndarray) -> np.ndarray:
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_norm), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    heatmap_color = heatmap_color.astype(np.float32) / 255.0
    base = base_rgb.astype(np.float32) / 255.0
    alpha = 0.5
    blended = base * (1 - alpha) + heatmap_color * alpha
    blended = np.clip(blended, 0.0, 1.0)
    return np.uint8(blended * 255)


def save_entity_attention_maps(
    records: Iterable[Dict],
    base_image_path: Path,
    save_dir: Path,
) -> None:
    """
    records: list of dicts {step, layer, entity, side, map: Tensor[N]}
    base_image_path: PIL-loadable path (建议用 Canny 图)
    save_dir: 输出目录
    """
    records = list(records)
    if not records:
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    base_img = Image.open(base_image_path).convert("RGB")
    W, H = base_img.size
    base_arr = np.array(base_img)

    for rec in records:
        attn_map = rec["map"]
        side = rec.get("side", None)
        layer = rec.get("layer", "layer")
        entity = rec.get("entity", "entity")
        step = rec.get("step", -1)

        attn_np = attn_map.detach().cpu().float().numpy()
        if side is None:
            side = int(math.sqrt(attn_np.shape[0]))
        attn_np = attn_np.reshape(side, side)
        attn_norm = _normalize(attn_np)
        attn_resized = cv2.resize(attn_norm, (W, H), interpolation=cv2.INTER_CUBIC)

        overlay = _overlay_heatmap(base_arr, attn_resized)

        layer_tag = "mid" if "mid_block" in layer else "up"
        safe_entity = "".join([c for c in entity if c.isalnum()]) or "ent"
        fname = f"step{step:02d}_{layer_tag}_res{side}_{safe_entity}.png"
        out_path = save_dir / fname
        Image.fromarray(overlay).save(out_path)
