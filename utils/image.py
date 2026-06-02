"""Image utilities shared across the pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Set

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTS: Set[str] = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _imread_with_unicode(path: Path):
    """Read an image while staying friendly to Windows Unicode paths."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size:
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            return img

    # Fall back to standard imread when the unicode-safe path is not needed.
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def list_images(image_path: Path) -> List[Path]:
    """Return one image path or all supported images in a directory."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image path not found: {image_path}")

    if image_path.is_file():
        if image_path.suffix.lower() not in IMAGE_EXTS:
            raise RuntimeError(f"Unsupported image extension: {image_path.suffix}")
        return [image_path]

    files = sorted(p for p in image_path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise RuntimeError(f"No images found in {image_path}.")
    return files


def make_canny_image(
    path: Path,
    width: int,
    height: int,
    low_threshold: int,
    high_threshold: int,
) -> Image.Image:
    """Turn a reference image into the Canny map used by ControlNet."""
    img = _imread_with_unicode(path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")

    # A light blur keeps the edge map stable without washing out structure.
    resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    canny = Image.fromarray(edges)
    return canny.convert("RGB")


def relative_to(path: Path, base_dirs: Iterable[Path]) -> str:
    """Prefer a relative path for metadata, otherwise keep the full path."""
    for base in base_dirs:
        try:
            return str(path.relative_to(base))
        except ValueError:
            continue
    return str(path)
