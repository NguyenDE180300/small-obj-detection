#!/usr/bin/env python3
"""Run a full test-set inference benchmark for one packaged seed-42 checkpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    # Artifacts (manifest, datasets, models, benchmarks) live beside ``code/``.
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("levir_ship", "varroa", "tinyperson"), required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--run-tag", help="Required if the case has duplicate seed-42 checkpoints")
    parser.add_argument("--backend", default="PyTorch .pt")
    parser.add_argument("--precision", choices=("FP32", "FP16", "INT8"), default="FP32")
    parser.add_argument("--model", type=Path, help="Override model path for ONNX/TensorRT artifacts")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--timing-samples", type=int, default=2)
    parser.add_argument("--timing-only", action="store_true", help="Only measure latency/FPS; skip test-set predictions and AP evaluation.")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument(
        "--evaluate-only", action="store_true",
        help="Recompute metrics from an existing benchmark JSON without running inference again.",
    )
    args = parser.parse_args()

    inventory = json.loads((root / "checkpoint_manifest_seed42.json").read_text())
    matches = [
        row for row in inventory["checkpoints"]
        if row["dataset"] == args.dataset and row["case"] == args.case
        and (args.run_tag is None or row["run_tag"] == args.run_tag)
    ]
    if len(matches) != 1:
        found = ", ".join(row["run_tag"] for row in matches) or "none"
        raise SystemExit(
            f"Expected exactly one checkpoint for {args.dataset}/{args.case}; found {len(matches)}. "
            f"Run tags: {found}. Pass --run-tag explicitly."
        )
    data_root = root / "data" / args.dataset
    annotation = data_root / "annotations/seed_42/test.json"
    if not annotation.is_file():
        raise SystemExit("Test assets are not prepared. Run: python prepare_test_data.py --dataset all")

    checkpoint = root / matches[0]["bundle_path"]
    model = args.model.resolve() if args.model else checkpoint
    if not model.is_file():
        raise FileNotFoundError(model)
    output_dir = root / "benchmarks" / args.dataset / args.case / matches[0]["run_tag"]
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{args.backend.replace(' ', '_').replace('.', '')}_{args.precision}.json"

    command = [
        sys.executable, str(root / "code/benchmark_jetson.py"),
        "--model", str(model), "--backend", args.backend, "--precision", args.precision,
        "--test-json", str(annotation), "--image-dir", str(data_root), "--output", str(result_path),
        "--imgsz", str(args.imgsz), "--warmup", str(args.warmup),
        "--iterations", str(args.iterations), "--conf", str(args.conf), "--iou", str(args.iou),
        "--timing-samples", str(args.timing_samples),
        "--dataset", args.dataset,
    ]
    if args.timing_only:
        command.append("--timing-only")
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(root / "code") + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if not args.evaluate_only:
        subprocess.run(command, check=True, env=env)
    elif not result_path.is_file():
        raise FileNotFoundError(f"No saved benchmark to evaluate: {result_path}")

    if args.timing_only:
        print(f"Saved latency-only benchmark: {result_path}", flush=True)
        return

    # Standard COCO metrics are valid for LEVIR/Varroa. TinyPerson must use its
    # merged-original TinyBenchmark evaluator, not the corner-window COCO score.
    if args.dataset != "tinyperson":
        from evaluate_saved_predictions import coco_metrics
        blob = json.loads(result_path.read_text())
        metrics = coco_metrics(str(annotation), str(result_path))
        blob["summary"].update(metrics)
        result_path.write_text(json.dumps(blob, indent=2) + "\n")
    else:
        evaluator = root / "code/evaluate_tinyperson_merged.py"
        subprocess.run([
            sys.executable, str(evaluator), "--predictions", str(result_path),
            "--data-root", str(data_root), "--output", str(output_dir / "tinybenchmark_metrics.json"),
        ], check=True, env=env)
    print(f"Saved benchmark: {result_path}", flush=True)


if __name__ == "__main__":
    main()
