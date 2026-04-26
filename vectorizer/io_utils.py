from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np

STAGE_DIRS = [
    "00_meta",
    "01_preprocess",
    "02_edges_contours",
    "03_segmentation",
    "04_fitting",
    "05_model_selection",
    "06_topology",
    "07_json_svg",
    "08_visualization",
    "09_report",
]


def make_run_dir(output_root: str | Path) -> Path:
    output_root = Path(output_root)
    run_dir = output_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    for stage in STAGE_DIRS:
        (run_dir / stage).mkdir(parents=True, exist_ok=True)
    return run_dir


def read_image(image_path: str | Path) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    return img


def write_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_image(path: str | Path, image: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), image)


def points_to_list(points: np.ndarray) -> List[List[float]]:
    return [[float(x), float(y)] for x, y in points]


def contour_to_xy(contour: np.ndarray) -> np.ndarray:
    return contour.reshape(-1, 2).astype(np.float64)


def is_closed(points: np.ndarray, eps: float = 3.0) -> bool:
    if len(points) < 3:
        return False
    return float(np.linalg.norm(points[0] - points[-1])) < eps


def rmse_distance(points: np.ndarray, dist_fn) -> float:
    if len(points) == 0:
        return float("inf")
    d = np.array([dist_fn(p) for p in points], dtype=np.float64)
    return float(np.sqrt(np.mean(d * d)))


def line_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    norm = np.linalg.norm(ab)
    if norm < 1e-8:
        return float(np.linalg.norm(p - a))
    return float(abs(np.cross(ab, p - a)) / norm)


def rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    if len(points) < 3:
        return points
    a = points[0]
    b = points[-1]
    dmax = -1.0
    idx = -1
    for i in range(1, len(points) - 1):
        d = line_distance(points[i], a, b)
        if d > dmax:
            dmax = d
            idx = i
    if dmax > epsilon and idx > 0:
        left = rdp(points[: idx + 1], epsilon)
        right = rdp(points[idx:], epsilon)
        return np.vstack((left[:-1], right))
    return np.vstack((a, b))


def _catmull_rom_points(ctrl: np.ndarray, closed: bool, samples_per_span: int = 12) -> np.ndarray:
    if len(ctrl) < 2:
        return ctrl
    pts: List[np.ndarray] = []
    n = len(ctrl)

    def _get(i: int) -> np.ndarray:
        if closed:
            return ctrl[i % n]
        return ctrl[int(np.clip(i, 0, n - 1))]

    span_count = n if closed else (n - 1)
    for i in range(span_count):
        p0 = _get(i - 1)
        p1 = _get(i)
        p2 = _get(i + 1)
        p3 = _get(i + 2)
        for j in range(samples_per_span):
            t = float(j) / float(samples_per_span)
            t2 = t * t
            t3 = t2 * t
            pt = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            pts.append(pt)
    pts.append(ctrl[0] if closed else ctrl[-1])
    return np.array(pts, dtype=np.float64)


def polyline_from_primitives(primitives: Iterable[Dict[str, Any]], step_deg: int = 4) -> List[Tuple[int, int, int, int]]:
    segments: List[Tuple[int, int, int, int]] = []
    for p in primitives:
        t = p["type"]
        if t == "line":
            x1, y1 = p["p1"]
            x2, y2 = p["p2"]
            segments.append((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))))
        elif t == "circle":
            cx, cy = p["center"]
            r = p["radius"]
            pts = []
            for deg in range(0, 361, step_deg):
                rad = np.deg2rad(deg)
                pts.append((cx + r * np.cos(rad), cy + r * np.sin(rad)))
            for i in range(len(pts) - 1):
                x1, y1 = pts[i]
                x2, y2 = pts[i + 1]
                segments.append((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))))
        elif t == "ellipse":
            cx, cy = p["center"]
            ax, by = p["axes"]
            angle = np.deg2rad(p.get("angle", 0.0))
            pts = []
            for deg in range(0, 361, step_deg):
                th = np.deg2rad(deg)
                x = ax * np.cos(th)
                y = by * np.sin(th)
                xr = x * np.cos(angle) - y * np.sin(angle) + cx
                yr = x * np.sin(angle) + y * np.cos(angle) + cy
                pts.append((xr, yr))
            for i in range(len(pts) - 1):
                x1, y1 = pts[i]
                x2, y2 = pts[i + 1]
                segments.append((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))))
        elif t == "arc":
            cx, cy = p["center"]
            r = p["radius"]
            start = p["start_angle"]
            end = p["end_angle"]
            if end < start:
                end += 360.0
            pts = []
            deg = start
            while deg <= end + 1e-6:
                rad = np.deg2rad(deg)
                pts.append((cx + r * np.cos(rad), cy + r * np.sin(rad)))
                deg += step_deg
            if len(pts) >= 2:
                for i in range(len(pts) - 1):
                    x1, y1 = pts[i]
                    x2, y2 = pts[i + 1]
                    segments.append((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))))
        elif t == "bspline":
            ctrl = p.get("control_points", [])
            if len(ctrl) >= 2:
                closed = bool(p.get("closed", False))
                ctrl_np = np.array(ctrl, dtype=np.float64)
                curve = _catmull_rom_points(ctrl_np, closed=closed, samples_per_span=12)
                for i in range(len(curve) - 1):
                    x1, y1 = curve[i]
                    x2, y2 = curve[i + 1]
                    segments.append((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))))
    return segments
