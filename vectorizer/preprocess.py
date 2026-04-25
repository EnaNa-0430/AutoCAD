from __future__ import annotations

from typing import Any, Dict, Tuple

import cv2
import numpy as np


def preprocess_image(img_bgr: np.ndarray, cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    blur_ksize = int(cfg.get("blur_ksize", 3))
    block_size = int(cfg.get("adaptive_block_size", 21))
    c_val = float(cfg.get("adaptive_C", 3))
    morph_kernel = int(cfg.get("morph_kernel", 3))

    if blur_ksize % 2 == 0:
        blur_ksize += 1
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(3, block_size)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c_val,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    meta = {
        "blur_ksize": blur_ksize,
        "adaptive_block_size": block_size,
        "adaptive_C": c_val,
        "morph_kernel": morph_kernel,
        "foreground_ratio": float(np.count_nonzero(cleaned) / cleaned.size),
    }
    return cleaned, meta
