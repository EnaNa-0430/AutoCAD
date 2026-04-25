from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def _get_endpoints(model: Dict[str, Any], seg_points: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
    t = model["type"]
    prm = model["params"]
    if t == "line":
        return np.array(prm["p1"], dtype=np.float64), np.array(prm["p2"], dtype=np.float64)
    return np.array(seg_points[0], dtype=np.float64), np.array(seg_points[-1], dtype=np.float64)


def refine_topology(
    selected_models: List[Dict[str, Any]],
    segments: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    snap_eps = float(cfg.get("snap_eps", 2.0))
    close_eps = float(cfg.get("close_eps", 3.0))
    seg_map = {s["segment_id"]: s for s in segments}

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    def find_or_add_node(pt: np.ndarray) -> int:
        for i, n in enumerate(nodes):
            np_pt = np.array(n["xy"], dtype=np.float64)
            if np.linalg.norm(np_pt - pt) <= snap_eps:
                return i
        nodes.append({"xy": [float(pt[0]), float(pt[1])]})
        return len(nodes) - 1

    for m in selected_models:
        seg = seg_map.get(m["segment_id"])
        if seg is None or not seg.get("points"):
            continue
        p_start, p_end = _get_endpoints(m, seg["points"])
        n1 = find_or_add_node(p_start)
        n2 = find_or_add_node(p_end)

        if np.linalg.norm(np.array(nodes[n1]["xy"]) - np.array(nodes[n2]["xy"])) <= close_eps:
            n2 = n1

        edges.append(
            {
                "edge_id": len(edges),
                "segment_id": m["segment_id"],
                "type": m["type"],
                "params": m["params"],
                "from": n1,
                "to": n2,
            }
        )

    graph = {"nodes": nodes, "edges": edges}
    meta = {"node_count": len(nodes), "edge_count": len(edges), "snap_eps": snap_eps, "close_eps": close_eps}
    return graph, meta
