from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .io_utils import line_distance


def _all_finite(values: List[float]) -> bool:
    arr = np.array(values, dtype=np.float64)
    return bool(np.all(np.isfinite(arr)))


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


def _point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    den = float(np.dot(ab, ab))
    if den < 1e-10:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / den, 0.0, 1.0))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def _point_to_polyline_distance(p: np.ndarray, poly: np.ndarray, closed: bool = False) -> float:
    if len(poly) < 2:
        return 1e9
    best = 1e9
    for i in range(len(poly) - 1):
        d = _point_to_segment_distance(p, poly[i], poly[i + 1])
        best = min(best, d)
    if closed:
        d = _point_to_segment_distance(p, poly[-1], poly[0])
        best = min(best, d)
    return best


def _fit_line(points: np.ndarray) -> Optional[Dict[str, Any]]:
    if len(points) < 2:
        return None
    line = cv2.fitLine(points.astype(np.float32).reshape(-1, 1, 2), cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = [float(x) for x in line.reshape(-1)]
    if not _all_finite([vx, vy, x0, y0]):
        return None
    d = np.array([vx, vy], dtype=np.float64)
    c0 = np.array([x0, y0], dtype=np.float64)
    t = (points - c0.reshape(1, 2)) @ d.reshape(2, 1)
    t = t.reshape(-1)
    p1 = c0 + d * float(np.min(t))
    p2 = c0 + d * float(np.max(t))
    if np.linalg.norm(p2 - p1) < 1e-8:
        return None
    dists = np.array([line_distance(p, p1, p2) for p in points], dtype=np.float64)
    rmse = float(np.sqrt(np.mean(dists * dists)))
    return {
        "type": "line",
        "params": {"p1": [float(p1[0]), float(p1[1])], "p2": [float(p2[0]), float(p2[1])]},
        "param_count": 2,
        "rmse": rmse,
    }


def _fit_circle(points: np.ndarray) -> Optional[Dict[str, Any]]:
    if len(points) < 3:
        return None
    x = points[:, 0]
    y = points[:, 1]
    a = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    b = x * x + y * y
    try:
        sol, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy, c0 = [float(v) for v in sol]
    r_sq = c0 + cx * cx + cy * cy
    if not np.isfinite(r_sq) or r_sq <= 1e-8:
        return None
    radius = float(np.sqrt(r_sq))
    d = np.linalg.norm(points - np.array([cx, cy], dtype=np.float64), axis=1)
    rmse = float(np.sqrt(np.mean((d - radius) ** 2)))
    if not np.isfinite(rmse):
        return None
    return {
        "type": "circle",
        "params": {"cx": cx, "cy": cy, "r": radius},
        "param_count": 3,
        "rmse": rmse,
    }


def _fit_ellipse(points: np.ndarray) -> Optional[Dict[str, Any]]:
    if len(points) < 5:
        return None
    try:
        p = points.astype(np.float32).reshape(-1, 1, 2)
        (cx, cy), (w, h), angle = cv2.fitEllipse(p)
        if w <= 0 or h <= 0:
            return None
        ax = float(w / 2.0)
        by = float(h / 2.0)
        axis_ratio = min(ax, by) / max(ax, by)
        if axis_ratio < 0.12:
            return None
        theta = np.deg2rad(float(angle))
        if not _all_finite([cx, cy, ax, by, theta]) or ax < 1e-8 or by < 1e-8:
            return None

        def dist(pt: np.ndarray) -> float:
            x = pt[0] - cx
            y = pt[1] - cy
            xr = x * np.cos(theta) + y * np.sin(theta)
            yr = -x * np.sin(theta) + y * np.cos(theta)
            val = ((xr / ax) ** 2 + (yr / by) ** 2) - 1.0
            return float(abs(val) * min(ax, by))

        d = np.array([dist(pt) for pt in points], dtype=np.float64)
        if not np.all(np.isfinite(d)):
            return None
        rmse = float(np.sqrt(np.mean(d * d)))
        if not np.isfinite(rmse):
            return None
        return {
            "type": "ellipse",
            "params": {"cx": float(cx), "cy": float(cy), "ax": ax, "by": by, "angle": float(angle)},
            "param_count": 5,
            "rmse": rmse,
        }
    except cv2.error:
        return None


def _fit_arc(points: np.ndarray) -> Optional[Dict[str, Any]]:
    c = _fit_circle(points)
    if c is None:
        return None
    cx = c["params"]["cx"]
    cy = c["params"]["cy"]
    ang = (np.degrees(np.arctan2(points[:, 1] - cy, points[:, 0] - cx)) + 360.0) % 360.0
    sa = np.sort(ang)
    gaps = np.diff(np.r_[sa, sa[0] + 360.0])
    max_gap_idx = int(np.argmax(gaps))
    max_gap = float(gaps[max_gap_idx])
    span = 360.0 - max_gap
    if span < 15.0 or span > 330.0:
        return None
    start = float(sa[(max_gap_idx + 1) % len(sa)])
    end = float(sa[max_gap_idx])
    return {
        "type": "arc",
        "params": {
            "cx": cx,
            "cy": cy,
            "r": c["params"]["r"],
            "start_angle": start,
            "end_angle": end,
            "span": span,
        },
        "param_count": 4,
        "rmse": c["rmse"],
        "coverage": span / 360.0,
    }


def _fit_bspline(points: np.ndarray, ctrl_points: int, closed: bool) -> Optional[Dict[str, Any]]:
    if len(points) < 4:
        return None
    ctrl_points = int(np.clip(ctrl_points, 4, min(20, len(points))))
    idx = np.linspace(0, len(points) - 1, num=ctrl_points, dtype=int)
    ctrl = points[idx]
    d = np.array([_point_to_polyline_distance(p, ctrl, closed=closed) for p in points], dtype=np.float64)
    rmse = float(np.sqrt(np.mean(d * d)))
    return {
        "type": "bspline",
        "params": {
            "degree": 3,
            "closed": bool(closed),
            "control_points": [[float(x), float(y)] for x, y in ctrl],
        },
        "param_count": int(2 * len(ctrl)),
        "rmse": rmse,
    }


def _model_fit(model: str, points: np.ndarray) -> Optional[Dict[str, Any]]:
    if model == "line":
        return _fit_line(points)
    if model == "circle":
        return _fit_circle(points)
    if model == "ellipse":
        return _fit_ellipse(points)
    if model == "arc":
        return _fit_arc(points)
    return None


def _point_error(model: Dict[str, Any], p: np.ndarray) -> float:
    t = model["type"]
    prm = model["params"]
    if t == "line":
        a = np.array(prm["p1"], dtype=np.float64)
        b = np.array(prm["p2"], dtype=np.float64)
        return line_distance(p, a, b)
    if t in ("circle", "arc"):
        cx, cy, r = prm["cx"], prm["cy"], prm["r"]
        d = np.linalg.norm(p - np.array([cx, cy], dtype=np.float64))
        return float(abs(d - r))
    if t == "ellipse":
        cx, cy = prm["cx"], prm["cy"]
        ax, by = prm["ax"], prm["by"]
        angle = np.deg2rad(prm["angle"])
        if not _all_finite([cx, cy, ax, by, angle]) or ax < 1e-8 or by < 1e-8:
            return 1e9
        x = p[0] - cx
        y = p[1] - cy
        xr = x * np.cos(angle) + y * np.sin(angle)
        yr = -x * np.sin(angle) + y * np.cos(angle)
        val = ((xr / max(ax, 1e-8)) ** 2 + (yr / max(by, 1e-8)) ** 2) - 1.0
        out = float(abs(val) * min(ax, by))
        return out if np.isfinite(out) else 1e9
    if t == "bspline":
        ctrl = np.array(prm.get("control_points", []), dtype=np.float64)
        closed = bool(prm.get("closed", False))
        return _point_to_polyline_distance(p, ctrl, closed=closed)
    return 1e9


def _ransac(model: str, points: np.ndarray, iters: int, inlier_th: float) -> Optional[Dict[str, Any]]:
    min_sample = {"line": 2, "circle": 3, "arc": 3, "ellipse": 5}.get(model, 2)
    if len(points) < min_sample:
        return None

    best: Optional[Dict[str, Any]] = None
    idx_all = list(range(len(points)))
    for _ in range(iters):
        idx = random.sample(idx_all, min_sample)
        m = _model_fit(model, points[idx])
        if m is None:
            continue
        errs = np.array([_point_error(m, p) for p in points], dtype=np.float64)
        if not np.all(np.isfinite(errs)):
            continue
        inliers = errs <= inlier_th
        inlier_ratio = float(np.mean(inliers))
        if np.count_nonzero(inliers) < min_sample:
            continue
        refined = _model_fit(model, points[inliers])
        if refined is None:
            continue
        refined["inlier_ratio"] = inlier_ratio
        refined["inlier_count"] = int(np.count_nonzero(inliers))
        if best is None:
            best = refined
            continue
        if (refined["inlier_ratio"] > best["inlier_ratio"]) or (
            abs(refined["inlier_ratio"] - best["inlier_ratio"]) < 1e-8 and refined["rmse"] < best["rmse"]
        ):
            best = refined
    return best


def fit_segments(segments: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    models = [str(x) for x in cfg.get("models", ["circle", "ellipse", "arc", "line", "bspline"])]
    closed_order = [str(x) for x in cfg.get("closed_model_order", ["circle", "ellipse", "arc", "line"])]
    open_order = [str(x) for x in cfg.get("open_model_order", ["line", "arc", "circle", "ellipse"])]
    ransac_iter = int(cfg.get("ransac_iter", 150))
    inlier_th = float(cfg.get("inlier_threshold", 1.5))
    bspline_ctrl = int(cfg.get("bspline_ctrl_points", 8))
    line_like_elongation = float(cfg.get("line_like_elongation", 8.0))

    out: List[Dict[str, Any]] = []
    fitted_candidates = 0

    for seg in segments:
        points = np.array(seg["points"], dtype=np.float64)
        is_closed = bool(seg.get("is_closed", False) or seg.get("closed_parent", False))
        elongation = _elongation_ratio(points)
        line_like = elongation >= line_like_elongation
        prefer_closed_models = is_closed and not line_like
        order = closed_order if prefer_closed_models else open_order
        candidates: List[Dict[str, Any]] = []
        for model in order:
            if model not in models:
                continue
            m = _ransac(model, points, ransac_iter, inlier_th)
            if m is None:
                continue
            if model == "circle":
                # Estimate angular coverage for circle-vs-arc disambiguation.
                cx = m["params"]["cx"]
                cy = m["params"]["cy"]
                ang = (np.degrees(np.arctan2(points[:, 1] - cy, points[:, 0] - cx)) + 360.0) % 360.0
                sa = np.sort(ang)
                gaps = np.diff(np.r_[sa, sa[0] + 360.0])
                m["coverage"] = float((360.0 - float(np.max(gaps))) / 360.0)
            fitted_candidates += 1
            candidates.append(m)
        if "bspline" in models:
            b = _fit_bspline(points, ctrl_points=bspline_ctrl, closed=prefer_closed_models)
            if b is not None:
                candidates.append(b)
                fitted_candidates += 1
        out.append(
            {
                "segment_id": seg["segment_id"],
                "contour_id": seg["contour_id"],
                "closed_parent": bool(seg.get("closed_parent", False)),
                "is_closed": is_closed,
                "line_like": line_like,
                "elongation": elongation,
                "point_count": len(points),
                "candidates": candidates,
            }
        )

    meta = {
        "segment_count": len(segments),
        "candidate_total": fitted_candidates,
        "ransac_iter": ransac_iter,
        "inlier_threshold": inlier_th,
        "model_order_closed": closed_order,
        "model_order_open": open_order,
    }
    return out, meta
