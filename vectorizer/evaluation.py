from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .io_utils import polyline_from_primitives


def draw_overlay(img_bgr: np.ndarray, primitives: List[Dict[str, Any]], color_bgr=(0, 0, 255), alpha=0.7) -> np.ndarray:
    canvas = img_bgr.copy()
    for x1, y1, x2, y2 in polyline_from_primitives(primitives):
        cv2.line(canvas, (x1, y1), (x2, y2), color_bgr, 1, cv2.LINE_AA)
    return cv2.addWeighted(canvas, alpha, img_bgr, 1 - alpha, 0)


def estimate_global_rmse(binary: np.ndarray, primitives: List[Dict[str, Any]]) -> float:
    mask = np.zeros_like(binary)
    for x1, y1, x2, y2 in polyline_from_primitives(primitives):
        cv2.line(mask, (x1, y1), (x2, y2), 255, 1, cv2.LINE_AA)

    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return 0.0
    dt = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    vals = dt[ys, xs].astype(np.float64)
    return float(np.sqrt(np.mean(vals * vals)))


def build_report(
    sample_id: str,
    timings: Dict[str, float],
    primitive_count: int,
    param_total: int,
    global_rmse: float,
    extras: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "sample_id": sample_id,
        "timings_sec": timings,
        "metrics": {
            "global_rmse": global_rmse,
            "primitive_count": primitive_count,
            "param_total": param_total,
        },
        "extras": extras,
    }
