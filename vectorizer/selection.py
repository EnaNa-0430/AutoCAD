from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple


def _score_candidate(cand: Dict[str, Any], criterion: str, n: int, lam: float) -> float:
    rmse = float(cand.get("rmse", 1e9))
    rss = max((rmse * rmse) * max(n, 1), 1e-12)
    k = int(cand.get("param_count", 10))
    criterion = criterion.lower()
    if criterion == "aic":
        return 2.0 * k + n * math.log(rss / max(n, 1))
    if criterion == "bic":
        return k * math.log(max(n, 1)) + n * math.log(rss / max(n, 1))
    return rmse + lam * k


def select_models(
    fit_candidates: List[Dict[str, Any]],
    segments: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    criterion = str(cfg.get("criterion", "bic"))
    rmse_max = float(cfg.get("rmse_max", 2.5))
    rmse_accept_curve = float(cfg.get("rmse_accept_curve", rmse_max))
    prefer_curve_ratio = float(cfg.get("prefer_curve_ratio", 1.2))
    circle_min_coverage = float(cfg.get("circle_min_coverage", 0.85))
    line_like_line_rmse_max = float(cfg.get("line_like_line_rmse_max", rmse_max))
    min_inlier_ratio_default = float(cfg.get("min_inlier_ratio_default", 0.5))
    min_inlier_ratio_line = float(cfg.get("min_inlier_ratio_line", min_inlier_ratio_default))
    min_inlier_ratio_arc = float(cfg.get("min_inlier_ratio_arc", min_inlier_ratio_default))
    min_inlier_ratio_circle = float(cfg.get("min_inlier_ratio_circle", 0.65))
    min_inlier_ratio_ellipse = float(cfg.get("min_inlier_ratio_ellipse", 0.72))
    bspline_prefer_ratio = float(cfg.get("bspline_prefer_ratio", 0.9))
    bspline_prefer_rmse_min = float(cfg.get("bspline_prefer_rmse_min", 0.6))
    bspline_override_rmse_gate = float(cfg.get("bspline_override_rmse_gate", rmse_accept_curve))
    stroke_like_bspline_ratio = float(cfg.get("stroke_like_bspline_ratio", 2.6))
    stroke_like_line_rmse_max = float(cfg.get("stroke_like_line_rmse_max", 0.35))
    arc_prefer_ratio = float(cfg.get("arc_prefer_ratio", 1.15))
    arc_prefer_min_coverage = float(cfg.get("arc_prefer_min_coverage", 0.25))
    arc_prefer_max_coverage = float(cfg.get("arc_prefer_max_coverage", 0.8))
    curve_types = {str(x) for x in cfg.get("curve_types", ["circle", "ellipse", "arc"])}
    lam = float(cfg.get("lambda_complexity", 0.3))
    tie_delta = float(cfg.get("tie_delta", 0.05))

    seg_len = {s["segment_id"]: len(s.get("points", [])) for s in segments}
    selected: List[Dict[str, Any]] = []
    type_counter: Dict[str, int] = {}

    for item in fit_candidates:
        sid = item["segment_id"]
        is_closed = bool(item.get("is_closed", False))
        line_like = bool(item.get("line_like", False))
        stroke_like = bool(item.get("stroke_like", False))
        all_cands = [c for c in item.get("candidates", []) if c.get("rmse") is not None and math.isfinite(float(c.get("rmse", 1e9)))]
        analytic = [c for c in all_cands if c.get("type") != "bspline"]
        bspline = [c for c in all_cands if c.get("type") == "bspline"]
        feasible = [c for c in analytic if float(c.get("rmse", 1e9)) <= rmse_max]
        if feasible:
            filtered = []
            for c in feasible:
                ctype = str(c.get("type", ""))
                inlier_ratio = c.get("inlier_ratio")
                if inlier_ratio is not None:
                    ratio_val = float(inlier_ratio)
                    min_ratio = min_inlier_ratio_default
                    if ctype == "line":
                        min_ratio = min_inlier_ratio_line
                    elif ctype == "arc":
                        min_ratio = min_inlier_ratio_arc
                    elif ctype == "circle":
                        min_ratio = min_inlier_ratio_circle
                    elif ctype == "ellipse":
                        min_ratio = min_inlier_ratio_ellipse
                    if ratio_val < min_ratio:
                        continue
                if c.get("type") == "circle":
                    cov = float(c.get("coverage", 1.0))
                    if cov < circle_min_coverage:
                        continue
                if c.get("type") == "arc":
                    cov = float(c.get("coverage", 0.0))
                    if cov < arc_prefer_min_coverage or cov > 0.85:
                        continue
                filtered.append(c)
            feasible = filtered

        if not feasible and not bspline:
            selected.append(
                {
                    "segment_id": sid,
                    "contour_id": item["contour_id"],
                    "type": "polyline",
                    "params": {},
                    "param_count": 999,
                    "rmse": None,
                    "score": None,
                }
            )
            type_counter["polyline"] = type_counter.get("polyline", 0) + 1
            continue

        n = seg_len.get(sid, 10)
        if feasible:
            if line_like and (not stroke_like):
                line_feasible = [c for c in feasible if c.get("type") == "line" and float(c.get("rmse", 1e9)) <= line_like_line_rmse_max]
                if line_feasible:
                    best = min(line_feasible, key=lambda c: _score_candidate(c, criterion, n, lam))
                    best_score = _score_candidate(best, criterion, n, lam)
                    selected_item = {
                        "segment_id": sid,
                        "contour_id": item["contour_id"],
                        "type": best["type"],
                        "params": best["params"],
                        "param_count": best["param_count"],
                        "rmse": best["rmse"],
                        "score": best_score,
                        "inlier_ratio": best.get("inlier_ratio"),
                        "inlier_count": best.get("inlier_count"),
                    }
                    selected.append(selected_item)
                    t = selected_item["type"]
                    type_counter[t] = type_counter.get(t, 0) + 1
                    continue

            scored = [(c, _score_candidate(c, criterion, n, lam)) for c in feasible]
            scored.sort(key=lambda x: x[1])
            best, best_score = scored[0]

            if len(scored) > 1:
                second, second_score = scored[1]
                if abs(second_score - best_score) <= tie_delta and int(second["param_count"]) < int(best["param_count"]):
                    best, best_score = second, second_score

            line_cands = [c for c in feasible if c.get("type") == "line"]
            curve_cands = [c for c in feasible if c.get("type") in curve_types]
            if is_closed and not line_like:
                # Closed contours should prefer curve primitives.
                if curve_cands:
                    curve_scored = [(c, _score_candidate(c, criterion, n, lam)) for c in curve_cands]
                    curve_scored.sort(key=lambda x: x[1])
                    best, best_score = curve_scored[0]

            arc_cands = [c for c in feasible if c.get("type") == "arc"]
            arc_valid = [
                c
                for c in arc_cands
                if arc_prefer_min_coverage <= float(c.get("coverage", 0.0)) <= arc_prefer_max_coverage
            ]
            if (not line_like) and arc_cands and best.get("type") != "arc":
                if arc_valid:
                    best_arc = min(arc_valid, key=lambda c: float(c["rmse"]))
                else:
                    best_arc = None
                if best_arc is not None and float(best_arc["rmse"]) <= rmse_accept_curve:
                    ratio = arc_prefer_ratio
                    if best.get("type") == "ellipse":
                        ratio = min(ratio, 0.9)
                    if float(best_arc["rmse"]) <= float(best["rmse"]) * ratio:
                        best = best_arc
                        best_score = _score_candidate(best, criterion, n, lam)

            if best.get("type") == "ellipse" and arc_valid:
                low_cov_circle = any(
                    c.get("type") == "circle" and float(c.get("coverage", 1.0)) < circle_min_coverage
                    for c in all_cands
                )
                if low_cov_circle:
                    best_arc = min(arc_valid, key=lambda c: float(c["rmse"]))
                    if float(best_arc["rmse"]) <= float(best["rmse"]) * 1.6:
                        best = best_arc
                        best_score = _score_candidate(best, criterion, n, lam)

            if best.get("type") == "arc":
                strong_circle = [
                    c
                    for c in all_cands
                    if c.get("type") == "circle" and float(c.get("coverage", 0.0)) >= 0.6
                ]
                if strong_circle:
                    best_circle = min(strong_circle, key=lambda c: float(c["rmse"]))
                    if float(best_circle["rmse"]) <= float(best["rmse"]) * 1.25:
                        best = best_circle
                        best_score = _score_candidate(best, criterion, n, lam)
            elif best.get("type") == "line" and line_cands and curve_cands:
                line_best = min(line_cands, key=lambda c: float(c["rmse"]))
                curve_good = [
                    c
                    for c in curve_cands
                    if float(c["rmse"]) <= rmse_accept_curve
                    and float(c["rmse"]) <= float(line_best["rmse"]) * prefer_curve_ratio
                ]
                if curve_good:
                    curve_scored = [(c, _score_candidate(c, criterion, n, lam)) for c in curve_good]
                    curve_scored.sort(key=lambda x: x[1])
                    best, best_score = curve_scored[0]

            if bspline:
                best_bspline = min(bspline, key=lambda c: float(c["rmse"]))
                best_rmse = float(best["rmse"])
                bs_rmse = float(best_bspline["rmse"])
                # Prefer analytic geometry when already accurate enough; use bspline as fallback.
                if (
                    best_rmse >= bspline_prefer_rmse_min
                    and best_rmse > bspline_override_rmse_gate
                    and bs_rmse <= best_rmse * bspline_prefer_ratio
                ):
                    best = best_bspline
                    best_score = _score_candidate(best, criterion, n, lam)
                elif (
                    stroke_like
                    and best.get("type") == "line"
                    and best_rmse > stroke_like_line_rmse_max
                    and bs_rmse <= best_rmse * stroke_like_bspline_ratio
                ):
                    best = best_bspline
                    best_score = _score_candidate(best, criterion, n, lam)
        else:
            best = min(bspline, key=lambda c: float(c["rmse"]))
            best_score = _score_candidate(best, criterion, n, lam)

        selected_item = {
            "segment_id": sid,
            "contour_id": item["contour_id"],
            "type": best["type"],
            "params": best["params"],
            "param_count": best["param_count"],
            "rmse": best["rmse"],
            "score": best_score,
            "inlier_ratio": best.get("inlier_ratio"),
            "inlier_count": best.get("inlier_count"),
        }
        selected.append(selected_item)
        t = selected_item["type"]
        type_counter[t] = type_counter.get(t, 0) + 1

    meta = {
        "criterion": criterion,
        "rmse_max": rmse_max,
        "rmse_accept_curve": rmse_accept_curve,
        "prefer_curve_ratio": prefer_curve_ratio,
        "circle_min_coverage": circle_min_coverage,
        "line_like_line_rmse_max": line_like_line_rmse_max,
        "min_inlier_ratio_default": min_inlier_ratio_default,
        "min_inlier_ratio_line": min_inlier_ratio_line,
        "min_inlier_ratio_arc": min_inlier_ratio_arc,
        "min_inlier_ratio_circle": min_inlier_ratio_circle,
        "min_inlier_ratio_ellipse": min_inlier_ratio_ellipse,
        "bspline_prefer_ratio": bspline_prefer_ratio,
        "bspline_prefer_rmse_min": bspline_prefer_rmse_min,
        "bspline_override_rmse_gate": bspline_override_rmse_gate,
        "stroke_like_bspline_ratio": stroke_like_bspline_ratio,
        "stroke_like_line_rmse_max": stroke_like_line_rmse_max,
        "arc_prefer_ratio": arc_prefer_ratio,
        "arc_prefer_min_coverage": arc_prefer_min_coverage,
        "arc_prefer_max_coverage": arc_prefer_max_coverage,
        "selected_count": len(selected),
        "type_distribution": type_counter,
    }
    return selected, meta
