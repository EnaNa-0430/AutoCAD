from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

def _angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 180.0
    cosv = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosv)))


def _pick_corners(points: np.ndarray, closed: bool, angle_th_deg: float, window: int, min_gap: int) -> List[int]:
    n = len(points)
    if n < (2 * window + 1):
        return []

    raw: List[int] = []
    if closed:
        indices = range(n)
    else:
        indices = range(window, n - window)

    for i in indices:
        i_prev = (i - window) % n
        i_next = (i + window) % n
        ang = _angle_deg(points[i_prev] - points[i], points[i_next] - points[i])
        if ang < angle_th_deg:
            raw.append(i)

    if not raw:
        return []

    raw = sorted(raw)
    corners: List[int] = []
    for idx in raw:
        if not corners:
            corners.append(idx)
            continue
        if abs(idx - corners[-1]) >= min_gap:
            corners.append(idx)
    if closed and len(corners) > 1 and (n - corners[-1] + corners[0]) < min_gap:
        corners.pop()
    return corners


def segment_contours(contours: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    min_points = int(cfg.get("min_segment_points", 20))

    segments: List[Dict[str, Any]] = []
    total_breaks = 0
    for contour in contours:
        pts = np.array(contour["points"], dtype=np.float64)
        if len(pts) < min_points:
            continue
        is_parent_closed = bool(contour.get("closed", False))
        seg_id = f"{contour['contour_id']}_{len(segments)}"
        segments.append(
            {
                "segment_id": seg_id,
                "contour_id": contour["contour_id"],
                "closed_parent": is_parent_closed,
                "is_closed": is_parent_closed,
                "point_indices": [],
                "points": pts.tolist(),
            }
        )

    meta = {
        "segment_count": len(segments),
        "avg_breakpoints_per_contour": float(total_breaks / max(len(contours), 1)),
        "corner_angle_deg": cfg.get("corner_angle_deg", 110.0),
        "corner_window": cfg.get("corner_window", 6),
        "corner_min_gap": cfg.get("corner_min_gap", 12),
        "min_segment_points": min_points,
    }
    return segments, meta
