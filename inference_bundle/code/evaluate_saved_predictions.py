"""Compute COCO AP metrics from predictions produced by benchmark_jetson.py."""

from __future__ import annotations

import argparse
import json

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def coco_metrics(gt_path: str, prediction_path: str) -> dict[str, float]:
    coco_gt = COCO(gt_path)
    detections = json.load(open(prediction_path, encoding="utf-8"))["predictions"]
    coco_dt = coco_gt.loadRes(detections)
    result = {}
    for metric, thresholds, area, label in (
        ("all", np.linspace(0.50, 0.95, 10), [0.0, 1e10], "all"),
        ("small", np.linspace(0.50, 0.95, 10), [0.0, 32.0**2], "small"),
    ):
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.maxDets = [1, 10, 300]
        ev.params.iouThrs = thresholds
        ev.params.areaRng = [area]
        ev.params.areaRngLbl = [label]
        ev.evaluate()
        ev.accumulate()
        precision = ev.eval["precision"][:, :, :, 0, -1]
        valid = precision > -1
        if metric == "all":
            result["mAP50_95"] = float(precision[valid].mean())
            result["AP50"] = float(precision[0][precision[0] > -1].mean())
            idx75 = int(np.argmin(np.abs(thresholds - 0.75)))
            result["AP75"] = float(precision[idx75][precision[idx75] > -1].mean())
        else:
            result["AP50_95_small"] = float(precision[valid].mean())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()
    print(json.dumps(coco_metrics(args.gt, args.predictions), indent=2))


if __name__ == "__main__":
    main()
