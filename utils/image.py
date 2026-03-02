# src/utils.py
# -*- coding: utf-8 -*-
"""
通用工具：
- 图片加载 / 枚举
- Canny 边缘图生成
- 路径相对化
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Set

import cv2
import numpy as np
from PIL import Image

# 支持的输入图扩展名
IMAGE_EXTS: Set[str] = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _imread_with_unicode(path: Path):
    """
    Read image while handling Unicode paths that cv2.imread may fail on (Windows).
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size:
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            return img

    # Fallback to standard imread (ASCII paths)
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def list_images(image_path: Path) -> List[Path]:
    """
    枚举输入图像。

    参数
    ----
    image_path : Path
        可以是单张图片路径，或者一个目录。

    返回
    ----
    List[Path]
        图片路径列表（按文件名排序）。

    异常
    ----
    FileNotFoundError : image_path 不存在
    RuntimeError      : 目录下没有符合扩展名的图片
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image path not found: {image_path}")

    # 单文件
    if image_path.is_file():
        if image_path.suffix.lower() not in IMAGE_EXTS:
            raise RuntimeError(f"Unsupported image extension: {image_path.suffix}")
        return [image_path]

    # 目录
    files = sorted(
        p for p in image_path.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
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
    """
    参考图 -> Canny 边缘图 (PIL.Image，RGB)

    参数
    ----
    path : Path
        原始参考图路径
    width, height : int
        输出图像分辨率 (一般与你的生成 width/height 一致)
    low_threshold, high_threshold : int
        Canny 低/高阈值

    返回
    ----
    PIL.Image.Image (RGB)
        用于 ControlNet Canny 的 conditioning_image
    """
    img = _imread_with_unicode(path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")

    # resize -> blur -> Canny
    resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    canny = Image.fromarray(edges)  # 单通道
    return canny.convert("RGB")     # ControlNet 一般期望 3 通道


def relative_to(path: Path, base_dirs: Iterable[Path]) -> str:
    """
    将 path 转成相对路径（用于 meta 记录），若无法相对任何 base_dir，则返回绝对/原始路径。

    参数
    ----
    path : Path
    base_dirs : Iterable[Path]
        用于尝试相对化的一组目录，比如：
        {cwd, config_dir, image_root}

    返回
    ----
    str
        相对路径或原路径字符串
    """
    for base in base_dirs:
        try:
            return str(path.relative_to(base))
        except ValueError:
            continue
    return str(path)

