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


def _circular_chain(points: np.ndarray, i0: int, i1: int) -> np.ndarray:
    n = len(points)
    out = [points[i0 % n]]
    k = i0 % n
    while k != (i1 % n):
        k = (k + 1) % n
        out.append(points[k])
        if len(out) > n + 1:
            break
    return np.array(out, dtype=np.float64)


def _polyline_curviness(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    v1 = points[1:-1] - points[:-2]
    v2 = points[2:] - points[1:-1]
    n1 = np.linalg.norm(v1, axis=1) + 1e-8
    n2 = np.linalg.norm(v2, axis=1) + 1e-8
    cosv = np.sum(v1 * v2, axis=1) / (n1 * n2)
    cosv = np.clip(cosv, -1.0, 1.0)
    ang = np.arccos(cosv)
    return float(np.sum(np.abs(ang)))


def _select_stroke_chain(points: np.ndarray, inlier_ratio: float, endpoint_hint: Optional[np.ndarray] = None) -> np.ndarray:
    if len(points) < 8:
        return points
    if endpoint_hint is not None and endpoint_hint.shape == (2, 2):
        d0 = np.sum((points - endpoint_hint[0].reshape(1, 2)) ** 2, axis=1)
        d1 = np.sum((points - endpoint_hint[1].reshape(1, 2)) ** 2, axis=1)
        i0 = int(np.argmin(d0))
        i1 = int(np.argmin(d1))
    else:
        i0, i1 = _farthest_pair_indices(points)
    if i0 == i1:
        return points
    a = _circular_chain(points, i0, i1)
    b = _circular_chain(points, i1, i0)
    if len(a) < 4:
        return b
    if len(b) < 4:
        return a

    def _monotonicize(chain: np.ndarray, tol: float = 1.0) -> np.ndarray:
        if len(chain) < 4:
            return chain
        s = chain[0]
        e = chain[-1]
        d = e - s
        den = float(np.linalg.norm(d))
        if den < 1e-8:
            return chain
        v = d / den
        proj = (chain - s.reshape(1, 2)) @ v.reshape(2, 1)
        proj = proj.reshape(-1)
        kept = [0]
        last_p = float(proj[0])
        for i in range(1, len(chain) - 1):
            p = float(proj[i])
            # Keep only forward-moving samples to guarantee one-way path.
            if p + tol >= last_p:
                kept.append(i)
                last_p = max(last_p, p)
        kept.append(len(chain) - 1)
        out = chain[np.array(sorted(set(kept)), dtype=np.int32)]
        if len(out) < 4:
            return chain
        return out

    def _chain_score(chain: np.ndarray) -> float:
        m = _monotonicize(chain)
        d = np.array([_point_to_polyline_distance(p, m, closed=False) for p in points], dtype=np.float64)
        keep_n = int(np.clip(round(len(d) * float(inlier_ratio)), 8, len(d)))
        d_use = np.partition(d, keep_n - 1)[:keep_n]
        rmse = float(np.sqrt(np.mean(d_use * d_use)))
        # Penalize reverse movement before monotonicization.
        s = chain[0]
        e = chain[-1]
        dirv = e - s
        den = float(np.linalg.norm(dirv))
        if den < 1e-8:
            return rmse + 1e3
        v = dirv / den
        proj = (chain - s.reshape(1, 2)) @ v.reshape(2, 1)
        proj = proj.reshape(-1)
        back = np.maximum(0.0, proj[:-1] - proj[1:])
        back_penalty = float(np.sum(back))
        return rmse + 0.03 * back_penalty

    best = a if _chain_score(a) <= _chain_score(b) else b
    return _monotonicize(best)


def _resample_open_polyline(points: np.ndarray, m: int) -> np.ndarray:
    if len(points) < 2 or m <= 2:
        return points
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    total = float(s[-1])
    if total < 1e-8:
        return np.repeat(points[:1], m, axis=0)
    t = np.linspace(0.0, total, num=m)
    x = np.interp(t, s, points[:, 0])
    y = np.interp(t, s, points[:, 1])
    return np.column_stack((x, y))


def _smooth_open_polyline(points: np.ndarray, passes: int = 3) -> np.ndarray:
    if len(points) < 5 or passes <= 0:
        return points
    out = points.copy()
    for _ in range(passes):
        nxt = out.copy()
        nxt[1:-1] = (out[:-2] + 2.0 * out[1:-1] + out[2:]) / 4.0
        nxt[0] = out[0]
        nxt[-1] = out[-1]
        out = nxt
    return out


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


def _trim_backtracking_tail(
    ctrl: np.ndarray,
    drop_ratio: float = 0.12,
    drop_abs: float = 20.0,
) -> np.ndarray:
    if len(ctrl) < 4:
        return ctrl
    d = ctrl[-1] - ctrl[0]
    n = float(np.linalg.norm(d))
    if n < 1e-8:
        return ctrl
    proj = (ctrl - ctrl[0]) @ (d / n)
    if float(proj[-1]) < float(proj[0]):
        ctrl = ctrl[::-1]
        d = ctrl[-1] - ctrl[0]
        n = float(np.linalg.norm(d))
        if n < 1e-8:
            return ctrl
        proj = (ctrl - ctrl[0]) @ (d / n)
    pmin = float(np.min(proj))
    pmax = float(np.max(proj))
    pr = pmax - pmin
    if pr < 1e-6:
        return ctrl
    imax = int(np.argmax(proj))
    drop = float(proj[imax] - proj[-1])
    if imax < (len(ctrl) - 1) and drop > max(float(drop_abs), float(drop_ratio) * pr):
        trimmed = ctrl[: imax + 1]
        if len(trimmed) >= 4:
            return trimmed
    return ctrl


def _trim_stroke_head(
    ctrl: np.ndarray,
    head_keep_span_min: float = 30.0,
    head_keep_ratio_min: float = 0.8,
) -> np.ndarray:
    if len(ctrl) < 5:
        return ctrl
    c = ctrl - np.mean(ctrl, axis=0, keepdims=True)
    cov = np.cov(c.T)
    vals, vecs = np.linalg.eig(cov)
    k = int(np.argmax(np.abs(np.real(vals))))
    v = np.real(vecs[:, k]).astype(np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return ctrl
    v = v / n
    proj = ctrl @ v.reshape(2, 1)
    proj = proj.reshape(-1)
    i_min = int(np.argmin(proj))
    if i_min <= 0 or i_min >= len(ctrl) - 2:
        return ctrl
    span_before = float(np.max(proj[: i_min + 1]) - np.min(proj[: i_min + 1]))
    span_after = float(np.max(proj[i_min:]) - np.min(proj[i_min:]))
    if span_after >= max(head_keep_span_min, span_before * head_keep_ratio_min):
        trimmed = ctrl[i_min:]
        if len(trimmed) >= 4:
            return trimmed
    return ctrl


def _cycle_chain(ctrl: np.ndarray, i: int, j: int) -> np.ndarray:
    n = len(ctrl)
    out = [ctrl[i % n]]
    k = i % n
    while k != (j % n):
        k = (k + 1) % n
        out.append(ctrl[k])
        if len(out) > n + 1:
            break
    return np.array(out, dtype=np.float64)


def _best_stroke_chain(
    ctrl_cycle: np.ndarray,
    points: np.ndarray,
    inlier_ratio: float,
) -> np.ndarray:
    n = len(ctrl_cycle)
    if n < 5:
        return ctrl_cycle
    seg_lens = np.linalg.norm(np.diff(np.vstack([ctrl_cycle, ctrl_cycle[0]]), axis=0), axis=1)
    perim = float(np.sum(seg_lens))
    if perim < 1e-6:
        return ctrl_cycle

    best_chain = ctrl_cycle
    best_score = 1e18
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            chain = _cycle_chain(ctrl_cycle, i, j)
            if len(chain) < 4:
                continue
            clen = float(np.sum(np.linalg.norm(np.diff(chain, axis=0), axis=1)))
            frac = clen / perim
            # Keep one side-like open chain, avoid tiny fragments and near-full loops.
            if frac < 0.25 or frac > 0.8:
                continue
            d = np.array([_point_to_polyline_distance(p, chain, closed=False) for p in points], dtype=np.float64)
            keep_n = int(np.clip(round(len(d) * float(inlier_ratio)), 8, len(d)))
            d_use = np.partition(d, keep_n - 1)[:keep_n]
            rmse = float(np.sqrt(np.mean(d_use * d_use)))
            # Prefer larger end-to-end span when rmse similar.
            span = float(np.linalg.norm(chain[-1] - chain[0]))
            score = rmse - 0.002 * span
            if score < best_score:
                best_score = score
                best_chain = chain
    return best_chain


def _trim_stroke_terminal_rebound(
    ctrl: np.ndarray,
    rebound_abs: float = 18.0,
    max_tail_points: int = 3,
) -> np.ndarray:
    if len(ctrl) < 6:
        return ctrl
    c = ctrl - np.mean(ctrl, axis=0, keepdims=True)
    cov = np.cov(c.T)
    vals, vecs = np.linalg.eig(cov)
    k = int(np.argmax(np.abs(np.real(vals))))
    v = np.real(vecs[:, k]).astype(np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return ctrl
    v = v / n
    nvec = np.array([-v[1], v[0]], dtype=np.float64)
    q = ctrl @ nvec.reshape(2, 1)
    q = q.reshape(-1)

    cand = [int(np.argmin(q)), int(np.argmax(q))]
    for idx in cand:
        tail_n = len(ctrl) - 1 - idx
        if idx < 2 or tail_n <= 0 or tail_n > max_tail_points:
            continue
        after = q[idx + 1 :]
        rebound = float(max(np.max(after) - q[idx], q[idx] - np.min(after)))
        if rebound > rebound_abs:
            trimmed = ctrl[: idx + 1]
            if len(trimmed) >= 4:
                return trimmed
    return ctrl


def _stroke_chain_rmse(chain: np.ndarray, points: np.ndarray, inlier_ratio: float) -> float:
    if len(chain) < 2:
        return 1e9
    d = np.array([_point_to_polyline_distance(p, chain, closed=False) for p in points], dtype=np.float64)
    keep_n = int(np.clip(round(len(d) * float(inlier_ratio)), 8, len(d)))
    d_use = np.partition(d, keep_n - 1)[:keep_n]
    return float(np.sqrt(np.mean(d_use * d_use)))


def _trim_stroke_endpoint_spurs(
    ctrl: np.ndarray,
    points: np.ndarray,
    inlier_ratio: float,
    max_drop_each_side: int = 2,
    improve_ratio: float = 0.02,
) -> np.ndarray:
    out = ctrl.copy()
    left_drop = 0
    right_drop = 0
    while len(out) >= 5:
        cur = _stroke_chain_rmse(out, points, inlier_ratio)
        best = out
        best_rmse = cur
        if left_drop < max_drop_each_side and len(out) >= 5:
            cand = out[1:]
            rm = _stroke_chain_rmse(cand, points, inlier_ratio)
            if rm < best_rmse:
                best_rmse = rm
                best = cand
                side = "left"
            else:
                side = ""
        else:
            side = ""
        if right_drop < max_drop_each_side and len(out) >= 5:
            cand = out[:-1]
            rm = _stroke_chain_rmse(cand, points, inlier_ratio)
            if rm < best_rmse:
                best_rmse = rm
                best = cand
                side = "right"
        if best is out or (cur - best_rmse) < (cur * improve_ratio):
            break
        out = best
        if side == "left":
            left_drop += 1
        elif side == "right":
            right_drop += 1
    return out


def _fit_bspline(
    points: np.ndarray,
    ctrl_points: int,
    closed: bool,
    approx_eps_ratio: float = 0.01,
    stroke_like: bool = False,
    backtrack_drop_ratio: float = 0.12,
    backtrack_drop_abs: float = 20.0,
    head_keep_span_min: float = 30.0,
    head_keep_ratio_min: float = 0.8,
    stroke_inlier_ratio: float = 0.55,
    stroke_rebound_abs: float = 18.0,
    stroke_rebound_tail_max_points: int = 3,
    stroke_spur_max_drop_each_side: int = 2,
    stroke_spur_improve_ratio: float = 0.02,
    endpoint_hint: Optional[np.ndarray] = None,
    stroke_smooth_samples: int = 96,
    stroke_smooth_passes: int = 3,
) -> Optional[Dict[str, Any]]:
    if len(points) < 4:
        return None
    fit_points = points
    approx_closed = bool(closed)
    if stroke_like and (not closed):
        fit_points = _select_stroke_chain(points, inlier_ratio=stroke_inlier_ratio, endpoint_hint=endpoint_hint)
        rs_n = int(np.clip(stroke_smooth_samples, 32, 256))
        fit_points = _resample_open_polyline(fit_points, m=rs_n)
        fit_points = _smooth_open_polyline(fit_points, passes=int(np.clip(stroke_smooth_passes, 0, 8)))
        approx_closed = False

    contour = fit_points.astype(np.float32).reshape(-1, 1, 2)
    arc_len = float(cv2.arcLength(contour, bool(approx_closed)))
    eps = max(arc_len * float(approx_eps_ratio), 1.0)
    approx = cv2.approxPolyDP(contour, epsilon=eps, closed=approx_closed).reshape(-1, 2).astype(np.float64)

    if len(approx) < 4:
        ctrl_points = int(np.clip(ctrl_points, 4, min(20, len(fit_points))))
        idx = np.linspace(0, len(fit_points) - 1, num=ctrl_points, dtype=int)
        ctrl = fit_points[idx]
    else:
        ctrl = approx

    if stroke_like and (not closed):
        # Guarantee endpoint correctness for stroke-like curves.
        if len(ctrl) >= 2:
            ctrl[0] = fit_points[0]
            ctrl[-1] = fit_points[-1]

    d = np.array([_point_to_polyline_distance(p, ctrl, closed=closed) for p in fit_points], dtype=np.float64)
    if stroke_like and (not closed):
        keep_n = int(np.clip(round(len(d) * float(stroke_inlier_ratio)), 8, len(d)))
        d_use = np.partition(d, keep_n - 1)[:keep_n]
    else:
        d_use = d
    rmse = float(np.sqrt(np.mean(d_use * d_use)))
    return {
        "type": "bspline",
        "params": {
            "degree": 3,
            "closed": bool(closed),
            "control_points": [[float(x), float(y)] for x, y in ctrl],
            "approx_eps_ratio": float(approx_eps_ratio),
            "stroke_inlier_ratio": float(stroke_inlier_ratio),
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


def _ellipse_bbox_ratio(points: np.ndarray, model: Dict[str, Any]) -> float:
    prm = model.get("params", {})
    ax = float(prm.get("ax", 0.0))
    by = float(prm.get("by", 0.0))
    if ax <= 0.0 or by <= 0.0 or len(points) < 2:
        return 1e9
    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    bw = float(max(bbox_max[0] - bbox_min[0], 1e-6))
    bh = float(max(bbox_max[1] - bbox_min[1], 1e-6))
    return float(max(ax / bw, by / bh))


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
    bspline_approx_eps_ratio = float(cfg.get("bspline_approx_eps_ratio", 0.01))
    bspline_backtrack_drop_ratio = float(cfg.get("bspline_backtrack_drop_ratio", 0.12))
    bspline_backtrack_drop_abs = float(cfg.get("bspline_backtrack_drop_abs", 20.0))
    bspline_head_keep_span_min = float(cfg.get("bspline_head_keep_span_min", 30.0))
    bspline_head_keep_ratio_min = float(cfg.get("bspline_head_keep_ratio_min", 0.8))
    bspline_stroke_inlier_ratio = float(cfg.get("bspline_stroke_inlier_ratio", 0.55))
    bspline_stroke_rebound_abs = float(cfg.get("bspline_stroke_rebound_abs", 18.0))
    bspline_stroke_rebound_tail_max_points = int(cfg.get("bspline_stroke_rebound_tail_max_points", 3))
    bspline_stroke_spur_max_drop_each_side = int(cfg.get("bspline_stroke_spur_max_drop_each_side", 2))
    bspline_stroke_spur_improve_ratio = float(cfg.get("bspline_stroke_spur_improve_ratio", 0.02))
    bspline_stroke_smooth_samples = int(cfg.get("bspline_stroke_smooth_samples", 96))
    bspline_stroke_smooth_passes = int(cfg.get("bspline_stroke_smooth_passes", 3))
    line_like_elongation = float(cfg.get("line_like_elongation", 8.0))
    ellipse_axis_bbox_ratio_max = float(cfg.get("ellipse_axis_bbox_ratio_max", 2.5))

    out: List[Dict[str, Any]] = []
    fitted_candidates = 0

    for seg in segments:
        points = np.array(seg["points"], dtype=np.float64)
        is_closed = bool(seg.get("is_closed", False))
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
            if model == "ellipse":
                # Reject numerically valid but geometrically implausible giant ellipses.
                if _ellipse_bbox_ratio(points, m) > ellipse_axis_bbox_ratio_max:
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
            endpoint_hint = None
            if bool(seg.get("stroke_like", False)):
                ep = seg.get("stroke_endpoints")
                if isinstance(ep, list) and len(ep) == 2:
                    endpoint_hint = np.array(ep, dtype=np.float64)
            b = _fit_bspline(
                points,
                ctrl_points=bspline_ctrl,
                closed=prefer_closed_models,
                approx_eps_ratio=bspline_approx_eps_ratio,
                stroke_like=bool(seg.get("stroke_like", False)),
                backtrack_drop_ratio=bspline_backtrack_drop_ratio,
                backtrack_drop_abs=bspline_backtrack_drop_abs,
                head_keep_span_min=bspline_head_keep_span_min,
                head_keep_ratio_min=bspline_head_keep_ratio_min,
                stroke_inlier_ratio=bspline_stroke_inlier_ratio,
                stroke_rebound_abs=bspline_stroke_rebound_abs,
                stroke_rebound_tail_max_points=bspline_stroke_rebound_tail_max_points,
                stroke_spur_max_drop_each_side=bspline_stroke_spur_max_drop_each_side,
                stroke_spur_improve_ratio=bspline_stroke_spur_improve_ratio,
                endpoint_hint=endpoint_hint,
                stroke_smooth_samples=bspline_stroke_smooth_samples,
                stroke_smooth_passes=bspline_stroke_smooth_passes,
            )
            if b is not None:
                candidates.append(b)
                fitted_candidates += 1
        out.append(
            {
                "segment_id": seg["segment_id"],
                "contour_id": seg["contour_id"],
                "closed_parent": bool(seg.get("closed_parent", False)),
                "stroke_like": bool(seg.get("stroke_like", False)),
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
        "bspline_approx_eps_ratio": bspline_approx_eps_ratio,
        "bspline_backtrack_drop_ratio": bspline_backtrack_drop_ratio,
        "bspline_backtrack_drop_abs": bspline_backtrack_drop_abs,
        "bspline_head_keep_span_min": bspline_head_keep_span_min,
        "bspline_head_keep_ratio_min": bspline_head_keep_ratio_min,
        "bspline_stroke_inlier_ratio": bspline_stroke_inlier_ratio,
        "bspline_stroke_rebound_abs": bspline_stroke_rebound_abs,
        "bspline_stroke_rebound_tail_max_points": bspline_stroke_rebound_tail_max_points,
        "bspline_stroke_spur_max_drop_each_side": bspline_stroke_spur_max_drop_each_side,
        "bspline_stroke_spur_improve_ratio": bspline_stroke_spur_improve_ratio,
        "bspline_stroke_smooth_samples": bspline_stroke_smooth_samples,
        "bspline_stroke_smooth_passes": bspline_stroke_smooth_passes,
        "ellipse_axis_bbox_ratio_max": ellipse_axis_bbox_ratio_max,
    }
    return out, meta
