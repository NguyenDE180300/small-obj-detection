"""Aggregate Jetson benchmark JSON files into a COCO-metric CSV report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def coco_metrics(coco_gt: COCO, detections: list[dict]) -> dict[str, float]:
    coco_dt = coco_gt.loadRes(detections)
    values = {}
    for area_name, area in (("all", [0.0, 1e10]), ("small", [0.0, 32.0**2])):
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.maxDets = [1, 10, 300]
        ev.params.iouThrs = np.linspace(0.5, 0.95, 10)
        ev.params.areaRng = [area]
        ev.params.areaRngLbl = [area_name]
        ev.evaluate()
        ev.accumulate()
        precision = ev.eval["precision"][:, :, :, 0, -1]
        valid = precision > -1
        if area_name == "all":
            values["AP50"] = float(precision[0][precision[0] > -1].mean())
            values["mAP50-95"] = float(precision[valid].mean())
            values["AP75"] = float(precision[5][precision[5] > -1].mean())
        else:
            values["AP50-95 small"] = float(precision[valid].mean())
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    coco_gt = COCO(str(args.gt))
    rows = []
    for path in sorted(args.results_dir.glob("*.json")):
        blob = json.loads(path.read_text())
        summary = blob["summary"]
        # Do not mix older end-to-end predict timings into model-only reports.
        if not summary.get("latency_scope", "").startswith("model inference only"):
            continue
        row = {
            "Backend": summary["backend"],
            "Precision": summary["precision"],
            **coco_metrics(coco_gt, blob["predictions"]),
            "Median model latency (ms)": summary["latency_median_ms"],
            "Model FPS": summary["fps_mean"],
            "Median end-to-end predict latency (ms)": summary.get("end_to_end_latency_median_ms"),
            "End-to-end predict FPS": summary.get("end_to_end_fps_median"),
            "GPU load avg (%)": summary["gpu_load_avg_percent"],
            "GPU load peak (%)": summary["gpu_load_peak_percent"],
            "CPU system avg (%)": summary["cpu_system_load_avg_percent"],
            "CPU system peak (%)": summary["cpu_system_load_peak_percent"],
            "CPU process avg (%)": summary["cpu_process_load_avg_percent"],
            "RAM process avg (MiB)": summary["ram_process_avg_mib"],
            "RAM process peak (MiB)": summary["ram_process_peak_mib"],
            "RAM system used avg (MiB)": summary["ram_system_used_avg_mib"],
            "RAM system used peak (MiB)": summary["ram_system_used_peak_mib"],
            "Images": summary["images"],
            "Batch size": summary["batch_size"],
            "Image size": summary["image_size"],
            "Source file": path.name,
        }
        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No benchmark JSON files in {args.results_dir}")
    fields = list(rows[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} benchmark rows to {args.output}")


if __name__ == "__main__":
    main()
