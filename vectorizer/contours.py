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


def _hierarchy_depths(hierarchy: np.ndarray) -> List[int]:
    # hierarchy row format: [next, prev, first_child, parent]
    depths = [0] * len(hierarchy)
    for i in range(len(hierarchy)):
        d = 0
        p = int(hierarchy[i][3])
        while p != -1:
            d += 1
            p = int(hierarchy[p][3])
            if d > len(hierarchy):
                break
        depths[i] = d
    return depths


def extract_contours(binary: np.ndarray, cfg: Dict[str, Any]) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    t1 = int(cfg.get("canny_t1", 40))
    t2 = int(cfg.get("canny_t2", 120))
    min_len = int(cfg.get("min_contour_len", 30))
    retrieval_name = str(cfg.get("retrieval_mode", "tree")).lower()
    dedupe_nested = bool(cfg.get("dedupe_nested_boundaries", True))
    keep_depth_parity = int(cfg.get("keep_depth_parity", 0))

    edges = cv2.Canny(binary, t1, t2)
    contours, hierarchy = cv2.findContours(binary, _retrieval_mode(retrieval_name), cv2.CHAIN_APPROX_NONE)
    h = hierarchy[0] if hierarchy is not None and len(hierarchy) > 0 else None
    depths = _hierarchy_depths(h) if h is not None else [0] * len(contours)

    contour_items: List[Dict[str, Any]] = []
    lengths: List[int] = []
    for idx, c in enumerate(contours):
        if retrieval_name == "tree" and dedupe_nested and (depths[idx] % 2) != keep_depth_parity:
            # Keep one boundary per stroke while preserving nested objects.
            continue
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
        "retrieval_mode": retrieval_name,
        "dedupe_nested_boundaries": dedupe_nested,
        "keep_depth_parity": keep_depth_parity,
        "contour_count_raw": len(contours),
        "contour_count": len(contour_items),
        "contour_length_min": int(min(lengths)) if lengths else 0,
        "contour_length_max": int(max(lengths)) if lengths else 0,
        "contour_length_mean": float(np.mean(lengths)) if lengths else 0.0,
    }
    return edges, contour_items, meta
