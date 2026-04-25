from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

from .contours import extract_contours
from .evaluation import build_report, draw_overlay, estimate_global_rmse
from .exporters import export_json_payload, export_svg, graph_to_primitives
from .fitting import fit_segments
from .io_utils import make_run_dir, read_image, write_image, write_json
from .preprocess import preprocess_image
from .segmentation import segment_contours
from .selection import select_models
from .topology import refine_topology


def run_pipeline(image_path: str, cfg: Dict[str, Any], output_root: str) -> Dict[str, Any]:
    image_path = str(image_path)
    sample_id = Path(image_path).stem
    run_dir = make_run_dir(output_root)
    timings: Dict[str, float] = {}

    t0 = time.perf_counter()
    img = read_image(image_path)
    write_json(run_dir / "00_meta" / f"{sample_id}_meta.json", {"image_path": image_path, "sample_id": sample_id})
    timings["load"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    binary, prep_meta = preprocess_image(img, cfg.get("preprocess", {}))
    write_image(run_dir / "01_preprocess" / f"{sample_id}_binary.png", binary)
    write_json(run_dir / "01_preprocess" / f"{sample_id}_preprocess_summary.json", prep_meta)
    timings["m1_preprocess"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    edges, contours, contour_meta = extract_contours(binary, cfg.get("contour", {}))
    write_image(run_dir / "02_edges_contours" / f"{sample_id}_edges.png", edges)
    write_json(
        run_dir / "02_edges_contours" / f"{sample_id}_contours.json",
        {"sample_id": sample_id, "contours": contours},
    )
    write_json(run_dir / "02_edges_contours" / f"{sample_id}_contours_summary.json", contour_meta)
    timings["m2_contours"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    segments, seg_meta = segment_contours(contours, cfg.get("segment", {}))
    write_json(
        run_dir / "03_segmentation" / f"{sample_id}_segments.json",
        {"sample_id": sample_id, "segments": segments},
    )
    seg_preview = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for s in segments:
        pts = np.array(s["points"], dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(
                seg_preview,
                [pts.reshape(-1, 1, 2)],
                bool(s.get("is_closed", False)),
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
    write_image(run_dir / "03_segmentation" / f"{sample_id}_segments_preview.png", seg_preview)
    write_json(run_dir / "03_segmentation" / f"{sample_id}_segments_summary.json", seg_meta)
    timings["m3_segmentation"] = time.perf_counter() - t3

    t4 = time.perf_counter()
    fit_candidates, fit_meta = fit_segments(segments, cfg.get("fit", {}))
    write_json(
        run_dir / "04_fitting" / f"{sample_id}_fit_candidates.json",
        {"sample_id": sample_id, "items": fit_candidates},
    )
    write_json(run_dir / "04_fitting" / f"{sample_id}_fitting_summary.json", fit_meta)
    timings["m4_fitting"] = time.perf_counter() - t4

    t5 = time.perf_counter()
    selected_models, selection_meta = select_models(fit_candidates, segments, cfg.get("selection", {}))
    write_json(
        run_dir / "05_model_selection" / f"{sample_id}_selected_models.json",
        {"sample_id": sample_id, "selected": selected_models},
    )
    write_json(run_dir / "05_model_selection" / f"{sample_id}_selection_summary.json", selection_meta)
    timings["m5_selection"] = time.perf_counter() - t5

    t6 = time.perf_counter()
    graph, topo_meta = refine_topology(selected_models, segments, cfg.get("topology", {}))
    write_json(run_dir / "06_topology" / f"{sample_id}_graph.json", graph)
    write_json(run_dir / "06_topology" / f"{sample_id}_topology_summary.json", topo_meta)
    timings["m6_topology"] = time.perf_counter() - t6

    t7 = time.perf_counter()
    primitives = graph_to_primitives(graph)
    global_rmse = estimate_global_rmse(binary, primitives)
    payload = export_json_payload(sample_id, primitives, global_rmse)
    json_path = run_dir / "07_json_svg" / f"{sample_id}.json"
    svg_path = run_dir / "07_json_svg" / f"{sample_id}.svg"
    write_json(json_path, payload)
    export_svg(svg_path, primitives, width=img.shape[1], height=img.shape[0])
    timings["m7_m8_export"] = time.perf_counter() - t7

    t8 = time.perf_counter()
    eval_cfg = cfg.get("eval", {})
    overlay = draw_overlay(
        img,
        primitives,
        color_bgr=tuple(eval_cfg.get("overlay_color_bgr", [0, 0, 255])),
        alpha=float(eval_cfg.get("overlay_alpha", 0.7)),
    )
    write_image(run_dir / "08_visualization" / f"{sample_id}_overlay.png", overlay)
    report = build_report(
        sample_id=sample_id,
        timings=timings,
        primitive_count=payload["metrics"]["primitive_count"],
        param_total=payload["metrics"]["param_total"],
        global_rmse=global_rmse,
        extras={
            "contour_summary": contour_meta,
            "segment_summary": seg_meta,
            "fit_summary": fit_meta,
            "selection_summary": selection_meta,
            "topology_summary": topo_meta,
        },
    )
    write_json(run_dir / "09_report" / f"{sample_id}_report.json", report)
    timings["m9_eval"] = time.perf_counter() - t8

    return {
        "run_dir": str(run_dir),
        "json_path": str(json_path),
        "svg_path": str(svg_path),
        "overlay_path": str(run_dir / "08_visualization" / f"{sample_id}_overlay.png"),
        "report_path": str(run_dir / "09_report" / f"{sample_id}_report.json"),
        "sample_id": sample_id,
    }
