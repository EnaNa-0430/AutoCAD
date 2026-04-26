from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


def _bspline_svg_path(ctrl: List[List[float]], closed: bool) -> str:
    if len(ctrl) < 2:
        return ""

    pts = [[float(x), float(y)] for x, y in ctrl]
    n = len(pts)

    def _get(i: int) -> List[float]:
        if closed:
            return pts[i % n]
        if i < 0:
            return pts[0]
        if i >= n:
            return pts[-1]
        return pts[i]

    if closed:
        start = pts[0]
        spans = n
    else:
        start = pts[0]
        spans = n - 1

    d = [f"M {start[0]:.3f} {start[1]:.3f}"]
    for i in range(spans):
        p0 = _get(i - 1)
        p1 = _get(i)
        p2 = _get(i + 1)
        p3 = _get(i + 2)
        c1x = p1[0] + (p2[0] - p0[0]) / 6.0
        c1y = p1[1] + (p2[1] - p0[1]) / 6.0
        c2x = p2[0] - (p3[0] - p1[0]) / 6.0
        c2y = p2[1] - (p3[1] - p1[1]) / 6.0
        d.append(f"C {c1x:.3f} {c1y:.3f}, {c2x:.3f} {c2y:.3f}, {p2[0]:.3f} {p2[1]:.3f}")
    if closed:
        d.append("Z")
    return " ".join(d)


def graph_to_primitives(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for e in graph.get("edges", []):
        t = e["type"]
        prm = e["params"]
        if t == "line":
            out.append({"type": "line", "p1": prm["p1"], "p2": prm["p2"]})
        elif t == "circle":
            out.append({"type": "circle", "center": [prm["cx"], prm["cy"]], "radius": prm["r"]})
        elif t == "ellipse":
            out.append({"type": "ellipse", "center": [prm["cx"], prm["cy"]], "axes": [prm["ax"], prm["by"]], "angle": prm["angle"]})
        elif t == "arc":
            out.append(
                {
                    "type": "arc",
                    "center": [prm["cx"], prm["cy"]],
                    "radius": prm["r"],
                    "start_angle": prm["start_angle"],
                    "end_angle": prm["end_angle"],
                }
            )
        elif t == "bspline":
            out.append(
                {
                    "type": "bspline",
                    "degree": int(prm.get("degree", 3)),
                    "closed": bool(prm.get("closed", False)),
                    "control_points": prm.get("control_points", []),
                }
            )
        else:
            out.append({"type": "polyline"})
    return out


def export_json_payload(sample_id: str, primitives: List[Dict[str, Any]], global_rmse: float | None) -> Dict[str, Any]:
    param_total = 0
    for p in primitives:
        t = p.get("type")
        if t == "line":
            param_total += 2
        elif t == "circle":
            param_total += 3
        elif t == "ellipse":
            param_total += 5
        elif t == "arc":
            param_total += 4
        elif t == "bspline":
            param_total += int(2 * len(p.get("control_points", [])))
    return {
        "sample_id": sample_id,
        "primitives": primitives,
        "metrics": {
            "global_rmse": global_rmse,
            "primitive_count": len(primitives),
            "param_total": param_total,
        },
    }


def export_svg(path: str | Path, primitives: List[Dict[str, Any]], width: int, height: int) -> None:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<g fill="none" stroke="red" stroke-width="1.5">',
    ]

    for p in primitives:
        t = p.get("type")
        if t == "line":
            x1, y1 = p["p1"]
            x2, y2 = p["p2"]
            parts.append(f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" />')
        elif t == "circle":
            cx, cy = p["center"]
            r = p["radius"]
            parts.append(f'<circle cx="{cx:.3f}" cy="{cy:.3f}" r="{r:.3f}" />')
        elif t == "ellipse":
            cx, cy = p["center"]
            ax, by = p["axes"]
            angle = p.get("angle", 0.0)
            parts.append(
                f'<ellipse cx="{cx:.3f}" cy="{cy:.3f}" rx="{ax:.3f}" ry="{by:.3f}" '
                f'transform="rotate({angle:.3f} {cx:.3f} {cy:.3f})" />'
            )
        elif t == "arc":
            cx, cy = p["center"]
            r = p["radius"]
            start = p["start_angle"]
            end = p["end_angle"]
            import math

            srad = math.radians(start)
            erad = math.radians(end)
            x1 = cx + r * math.cos(srad)
            y1 = cy + r * math.sin(srad)
            x2 = cx + r * math.cos(erad)
            y2 = cy + r * math.sin(erad)
            delta = (end - start) % 360.0
            large = 1 if delta > 180 else 0
            parts.append(
                f'<path d="M {x1:.3f} {y1:.3f} A {r:.3f} {r:.3f} 0 {large} 1 {x2:.3f} {y2:.3f}" />'
            )
        elif t == "bspline":
            ctrl = p.get("control_points", [])
            if len(ctrl) >= 2:
                d = _bspline_svg_path(ctrl, bool(p.get("closed", False)))
                if d:
                    parts.append(f'<path d="{d}" />')

    parts.append("</g></svg>")
    content = "\n".join(parts)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
