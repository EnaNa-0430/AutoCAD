from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
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


def _closed_indices(start: int, end: int, n: int) -> List[int]:
    if n <= 0:
        return []
    out = [int(start)]
    i = int(start)
    # Walk forward on a circular index ring [0, n).
    while i != int(end):
        i = (i + 1) % n
        out.append(i)
        if len(out) > n + 1:
            break
    return out


def _nearest_point_index(points: np.ndarray, p: np.ndarray) -> int:
    d2 = np.sum((points - p.reshape(1, 2)) ** 2, axis=1)
    return int(np.argmin(d2))


def _polygon_fallback_corners(
    points: np.ndarray,
    min_gap: int,
    eps_ratio: float,
    min_vertices: int,
    max_vertices: int,
) -> List[int]:
    if len(points) < max(6, min_vertices):
        return []
    contour = points.astype(np.float32).reshape(-1, 1, 2)
    arc_len = float(cv2.arcLength(contour, True))
    eps = max(1.0, arc_len * float(eps_ratio))
    approx = cv2.approxPolyDP(contour, epsilon=eps, closed=True).reshape(-1, 2).astype(np.float64)
    if len(approx) < int(min_vertices) or len(approx) > int(max_vertices):
        return []
    mapped = sorted(set(_nearest_point_index(points, v) for v in approx))
    if not mapped:
        return []
    out: List[int] = []
    for idx in mapped:
        if not out or abs(idx - out[-1]) >= min_gap:
            out.append(int(idx))
    n = len(points)
    if len(out) > 1 and (n - out[-1] + out[0]) < min_gap:
        out.pop()
    if len(out) < int(min_vertices):
        return []
    return out


def _pick_stroke_endpoints(points: np.ndarray, corners: List[int]) -> Tuple[int, int]:
    n = len(points)
    if n < 2:
        return 0, 0
    cand = [c for c in corners if 0 <= c < n]
    if len(cand) >= 2:
        best = (cand[0], cand[1])
        best_d2 = -1.0
        for i in range(len(cand)):
            for j in range(i + 1, len(cand)):
                d = points[cand[i]] - points[cand[j]]
                d2 = float(np.dot(d, d))
                if d2 > best_d2:
                    best_d2 = d2
                    best = (cand[i], cand[j])
        return best
    if len(cand) == 1:
        i = cand[0]
        diff = points - points[i].reshape(1, 2)
        j = int(np.argmax(np.sum(diff * diff, axis=1)))
        return i, j
    diff_mat = points.reshape(n, 1, 2) - points.reshape(1, n, 2)
    d2 = np.sum(diff_mat * diff_mat, axis=2)
    flat = int(np.argmax(d2))
    i, j = divmod(flat, n)
    return int(i), int(j)


def _elongation_ratio(points: np.ndarray) -> float:
    if len(points) < 4:
        return 1.0
    c = points - np.mean(points, axis=0, keepdims=True)
    cov = np.cov(c.T)
    vals, _ = np.linalg.eig(cov)
    vals = np.sort(np.abs(np.real(vals)))
    if len(vals) < 2:
        return 1.0
    return float(np.sqrt((vals[-1] + 1e-9) / (vals[0] + 1e-9)))


def _farthest_pair_indices(points: np.ndarray) -> Tuple[int, int]:
    n = len(points)
    if n < 2:
        return 0, 0
    best_i, best_j = 0, 1
    best_d2 = -1.0
    for i in range(n):
        pi = points[i]
        diff = points - pi.reshape(1, 2)
        d2 = np.sum(diff * diff, axis=1)
        j = int(np.argmax(d2))
        if float(d2[j]) > best_d2:
            best_d2 = float(d2[j])
            best_i, best_j = i, j
    return best_i, best_j


def _closed_path(points: np.ndarray, i0: int, i1: int) -> np.ndarray:
    n = len(points)
    idxs = _closed_indices(i0, i1, n)
    return points[np.array(idxs, dtype=np.int32)]


def _path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def _stroke_single_path(points: np.ndarray, min_points: int) -> np.ndarray:
    if len(points) < max(8, min_points):
        return points
    i0, i1 = _farthest_pair_indices(points)
    if i0 == i1:
        return points
    p_a = _closed_path(points, i0, i1)
    p_b = _closed_path(points, i1, i0)[::-1]
    if len(p_a) < 2 or len(p_b) < 2:
        return points
    la = _path_length(p_a)
    lb = _path_length(p_b)
    chosen = p_a if la <= lb else p_b
    if len(chosen) < min_points:
        other = p_b if chosen is p_a else p_a
        if len(other) > len(chosen):
            chosen = other
    return chosen


def segment_contours(contours: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    min_points = int(cfg.get("min_segment_points", 20))
    angle_th = float(cfg.get("corner_angle_deg", 110.0))
    window = int(cfg.get("corner_window", 6))
    min_gap = int(cfg.get("corner_min_gap", 12))
    stroke_like_elongation_min = float(cfg.get("stroke_like_elongation_min", 2.2))
    polygon_fallback_enabled = bool(cfg.get("polygon_fallback_enabled", True))
    polygon_fallback_min_raw_corners = int(cfg.get("polygon_fallback_min_raw_corners", 2))
    polygon_fallback_eps_ratio = float(cfg.get("polygon_fallback_eps_ratio", 0.01))
    polygon_fallback_min_vertices = int(cfg.get("polygon_fallback_min_vertices", 3))
    polygon_fallback_max_vertices = int(cfg.get("polygon_fallback_max_vertices", 6))

    segments: List[Dict[str, Any]] = []
    total_breaks = 0
    polygon_fallback_used = 0
    for contour in contours:
        pts = np.array(contour["points"], dtype=np.float64)
        if len(pts) < min_points:
            continue
        is_parent_closed = bool(contour.get("closed", False))
        n = len(pts)
        corners = _pick_corners(pts, is_parent_closed, angle_th, window, min_gap)
        if (
            is_parent_closed
            and polygon_fallback_enabled
            and polygon_fallback_min_raw_corners <= len(corners) < 3
        ):
            fb = _polygon_fallback_corners(
                pts,
                min_gap=min_gap,
                eps_ratio=polygon_fallback_eps_ratio,
                min_vertices=polygon_fallback_min_vertices,
                max_vertices=polygon_fallback_max_vertices,
            )
            if len(fb) > len(corners):
                corners = fb
                polygon_fallback_used += 1
        total_breaks += len(corners)
        elongation = _elongation_ratio(pts)
        stroke_like_closed = bool(is_parent_closed and len(corners) <= 2 and elongation >= stroke_like_elongation_min)

        emitted = 0
        if stroke_like_closed:
            # Keep as a single open stroke-like segment; tail trimming is handled in bspline fitting.
            e0, e1 = _pick_stroke_endpoints(pts, corners)
            seg_id = f"{contour['contour_id']}_{len(segments)}"
            segments.append(
                {
                    "segment_id": seg_id,
                    "contour_id": contour["contour_id"],
                    "closed_parent": is_parent_closed,
                    "stroke_like": True,
                    "stroke_endpoints": [[float(pts[e0, 0]), float(pts[e0, 1])], [float(pts[e1, 0]), float(pts[e1, 1])]],
                    "is_closed": False,
                    "point_indices": [],
                    "points": pts.tolist(),
                }
            )
            emitted = 1
        elif is_parent_closed and len(corners) >= 2:
            for i in range(len(corners)):
                s = corners[i]
                e = corners[(i + 1) % len(corners)]
                idxs = _closed_indices(s, e, n)
                if len(idxs) < min_points:
                    continue
                seg_id = f"{contour['contour_id']}_{len(segments)}"
                seg_pts = pts[idxs]
                segments.append(
                    {
                        "segment_id": seg_id,
                        "contour_id": contour["contour_id"],
                        "closed_parent": is_parent_closed,
                        "stroke_like": False,
                        "is_closed": False,
                        "point_indices": idxs,
                        "points": seg_pts.tolist(),
                    }
                )
                emitted += 1
        elif (not is_parent_closed) and len(corners) >= 1:
            anchors = [0] + [c for c in corners if 0 < c < (n - 1)] + [n - 1]
            anchors = sorted(set(anchors))
            for a, b in zip(anchors[:-1], anchors[1:]):
                idxs = list(range(int(a), int(b) + 1))
                if len(idxs) < min_points:
                    continue
                seg_id = f"{contour['contour_id']}_{len(segments)}"
                seg_pts = pts[idxs]
                segments.append(
                    {
                        "segment_id": seg_id,
                        "contour_id": contour["contour_id"],
                        "closed_parent": is_parent_closed,
                        "stroke_like": False,
                        "is_closed": False,
                        "point_indices": idxs,
                        "points": seg_pts.tolist(),
                    }
                )
                emitted += 1

        if emitted == 0:
            seg_id = f"{contour['contour_id']}_{len(segments)}"
            segments.append(
                {
                    "segment_id": seg_id,
                    "contour_id": contour["contour_id"],
                    "closed_parent": is_parent_closed,
                    "stroke_like": False,
                    "is_closed": is_parent_closed,
                    "point_indices": [],
                    "points": pts.tolist(),
                }
            )

    meta = {
        "segment_count": len(segments),
        "avg_breakpoints_per_contour": float(total_breaks / max(len(contours), 1)),
        "corner_angle_deg": angle_th,
        "corner_window": window,
        "corner_min_gap": min_gap,
        "stroke_like_elongation_min": stroke_like_elongation_min,
        "polygon_fallback_enabled": polygon_fallback_enabled,
        "polygon_fallback_min_raw_corners": polygon_fallback_min_raw_corners,
        "polygon_fallback_eps_ratio": polygon_fallback_eps_ratio,
        "polygon_fallback_min_vertices": polygon_fallback_min_vertices,
        "polygon_fallback_max_vertices": polygon_fallback_max_vertices,
        "polygon_fallback_used": polygon_fallback_used,
        "min_segment_points": min_points,
    }
    return segments, meta
