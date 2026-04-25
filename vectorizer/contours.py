from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .io_utils import contour_to_xy, is_closed, points_to_list


def _retrieval_mode(name: str) -> int:
    if name.lower() == "tree":
        return cv2.RETR_TREE
    if name.lower() == "list":
        return cv2.RETR_LIST
    return cv2.RETR_EXTERNAL


def extract_contours(binary: np.ndarray, cfg: Dict[str, Any]) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    t1 = int(cfg.get("canny_t1", 40))
    t2 = int(cfg.get("canny_t2", 120))
    min_len = int(cfg.get("min_contour_len", 30))

    edges = cv2.Canny(binary, t1, t2)
    contours, _ = cv2.findContours(binary, _retrieval_mode(str(cfg.get("retrieval_mode", "external"))), cv2.CHAIN_APPROX_NONE)

    contour_items: List[Dict[str, Any]] = []
    lengths: List[int] = []
    for idx, c in enumerate(contours):
        pts = contour_to_xy(c)
        if len(pts) < min_len:
            continue
        contour_items.append(
            {
                "contour_id": len(contour_items),
                "source_contour_id": idx,
                "closed": bool(is_closed(pts) or cv2.contourArea(c) > 1.0),
                "points": points_to_list(pts),
            }
        )
        lengths.append(int(len(pts)))

    meta = {
        "canny_t1": t1,
        "canny_t2": t2,
        "min_contour_len": min_len,
        "contour_count": len(contour_items),
        "contour_length_min": int(min(lengths)) if lengths else 0,
        "contour_length_max": int(max(lengths)) if lengths else 0,
        "contour_length_mean": float(np.mean(lengths)) if lengths else 0.0,
    }
    return edges, contour_items, meta
