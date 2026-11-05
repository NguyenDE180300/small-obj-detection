#!/usr/bin/env python3
"""Evaluate Jetson TinyPerson window predictions using merged TinyBenchmark AP."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO


def iou_xyxy(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + ab - inter, 1e-12)


def merge_predictions(predictions: list[dict], corner_gt: dict, windows: list[dict]) -> list[dict]:
    name_by_id = {int(image["id"]): image["file_name"] for image in corner_gt["images"]}
    window_by_name = {row["file_name"]: row for row in windows}
    by_original: dict[int, list[dict]] = defaultdict(list)
    original_id = {image["file_name"]: int(image["id"]) for image in corner_gt.get("old_images", [])}
    if not original_id:
        raise ValueError("Corner annotation metadata has no old_images original-image mapping")

    for det in predictions:
        name = name_by_id.get(int(det["image_id"]))
        if name is None or name not in window_by_name:
            raise ValueError(f"Prediction image id {det['image_id']} is absent from TinyBenchmark window metadata")
        row = window_by_name[name]
        x, y, w, h = map(float, det["bbox"])
        x0, y0, _, _ = map(float, row["corner"])
        by_original[original_id[name]].append({
            "xyxy": [x + x0, y + y0, x + w + x0, y + h + y0],
            "score": float(det["score"]),
        })

    merged = []
    for image_id, candidates in by_original.items():
        candidates.sort(key=lambda x: x["score"], reverse=True)
        kept: list[dict] = []
        while candidates:
            current = candidates.pop(0)
            kept.append(current)
            candidates = [x for x in candidates if iou_xyxy(current["xyxy"], x["xyxy"]) <= 0.5]
        for item in kept:
            x1, y1, x2, y2 = item["xyxy"]
            merged.append({
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "score": item["score"],
            })
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, default=Path(__file__).resolve().parent / "tinyperson_cocoeval.py")
    args = parser.parse_args()

    blob = json.loads(args.predictions.read_text())
    corner_path = args.data_root / "annotations/corner/task/tiny_set_test_sw640_sh512_all.json"
    merged_path = args.data_root / "annotations/task/tiny_set_test_all.json"
    windows_path = args.data_root / "tinybenchmark/test_windows.json"
    corner_gt = json.loads(corner_path.read_text())
    merged_gt = COCO(str(merged_path))
    windows = json.loads(windows_path.read_text())
    detections = merge_predictions(blob["predictions"], corner_gt, windows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged_predictions = args.output.with_name(args.output.stem + "_merged_predictions.json")
    merged_predictions.write_text(json.dumps(detections) + "\n")
    coco_dt = merged_gt.loadRes(str(merged_predictions)) if detections else merged_gt.loadRes([])
    spec = importlib.util.spec_from_file_location("tinyperson_cocoeval", args.evaluator)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load TinyBenchmark evaluator: {args.evaluator}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.Params.EVAL_STRANDARD = "tiny"
    evaluator = module.COCOeval(merged_gt, coco_dt, "bbox", True, True)
    evaluator.params.iouThrs = np.linspace(0.50, 0.75, 6)
    # Old TinyBenchmark uses np.float and a float-valued linspace count.
    old_linspace = module.np.linspace
    had_np_float = hasattr(module.np, "float")
    if not had_np_float:
        module.np.float = float
    def compatible_linspace(start, stop, num=50, *a, **kw):
        if isinstance(num, (float, np.floating)):
            num = int(num)
        return old_linspace(start, stop, num, *a, **kw)
    module.np.linspace = compatible_linspace
    try:
        evaluator.evaluate()
        evaluator.accumulate()
    finally:
        module.np.linspace = old_linspace
        if not had_np_float:
            delattr(module.np, "float")

    precision = evaluator.eval["precision"]
    labels = evaluator.params.areaRngLbl
    def mean_slice(iou_index: int, area_name: str) -> float | None:
        values = np.asarray(precision[iou_index, :, :, labels.index(area_name), -1])
        values = values[values > -1]
        return float(values.mean()) if values.size else None
    all_ap = [mean_slice(i, "all") for i in range(6)]
    all_valid = [x for x in all_ap if x is not None]
    metrics = {
        "dataset": "tinyperson",
        "protocol": "TinyBenchmark merged original-image test; corner windows translated and class-agnostic NMS IoU=0.5",
        "images_merged": len(merged_gt.dataset["images"]),
        "predictions_after_merge_nms": len(detections),
        "AP50": all_ap[0],
        "AP75": all_ap[-1],
        "mAP50-75": float(np.mean(all_valid)) if all_valid else None,
    }
    for area in ("tiny1", "tiny2", "tiny3", "tiny", "small", "medium"):
        if area in labels:
            metrics[f"AP50_{area}"] = mean_slice(0, area)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
