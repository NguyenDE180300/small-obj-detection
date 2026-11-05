#!/usr/bin/env python3
"""Measure the seven inference configurations for every packaged seed-42 checkpoint.

This is a latency-only matrix: batch 1, fixed 640x640, two rotating test images,
50 warmups, 200 measured calls. It does not calculate AP over the test sets.
TensorRT INT8 calibration uses the matching dataset's seed-42 training samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path


MODES = (
    ("PyTorch .pt", "FP32", "pt", None),
    ("PyTorch .pt", "FP16", "pt", None),
    ("TensorRT", "FP32", "engine", "FP32"),
    ("TensorRT", "FP16", "engine", "FP16"),
    ("TensorRT", "INT8", "engine", "INT8"),
    ("ONNX Runtime CPU", "FP32", "onnx", "FP32"),
    ("ONNX Runtime CPU", "FP16", "onnx", "FP16"),
)


def slug(backend: str, precision: str) -> str:
    return backend.lower().replace(" ", "_").replace(".", "") + "_" + precision.lower()


def run(command: list[str], env: dict[str, str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def write_summary_csv(root: Path, matrix_root: Path) -> None:
    rows = []
    for result in sorted(matrix_root.rglob("*.json")):
        try:
            summary = json.loads(result.read_text(encoding="utf-8"))["summary"]
        except (OSError, json.JSONDecodeError, KeyError):
            continue
        rows.append({
            "Dataset": summary.get("dataset"),
            "Case": summary.get("case"),
            "Run tag": summary.get("run_tag"),
            "Backend": summary.get("backend"),
            "Precision": summary.get("precision"),
            "Image size": summary.get("image_size"),
            "Batch size": summary.get("batch_size"),
            "Warmup": summary.get("warmup_iterations"),
            "Measured iterations": summary.get("measured_iterations"),
            "Timing sample images": summary.get("timing_sample_images"),
            "Median model latency (ms)": summary.get("latency_median_ms"),
            "Model FPS from median": summary.get("fps_median"),
            "Mean model latency (ms)": summary.get("latency_mean_ms"),
            "Mean model FPS": summary.get("fps_mean"),
            "P95 model latency (ms)": summary.get("latency_p95_ms"),
            "Median end-to-end predict latency (ms)": summary.get("end_to_end_latency_median_ms"),
            "End-to-end predict FPS from median": summary.get("end_to_end_fps_median"),
            "GPU load avg (%)": summary.get("gpu_load_avg_percent"),
            "CPU load avg (%)": summary.get("cpu_system_load_avg_percent"),
            "Process RAM peak (MiB)": summary.get("ram_process_peak_mib"),
            "Result JSON": str(result.relative_to(root)),
        })
    if not rows:
        return
    output = root / "benchmarks/seed42_latency_matrix_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    code = root / "code"
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--timing-samples", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-checkpoints", type=int, default=0, help="Optional limit for a short matrix smoke test.")
    parser.add_argument("--resume", action="store_true", help="Skip result JSONs that already contain a valid summary.")
    args = parser.parse_args()

    manifest = json.loads((root / "checkpoint_manifest_seed42.json").read_text(encoding="utf-8"))
    checkpoints = manifest["checkpoints"]
    if args.max_checkpoints:
        checkpoints = checkpoints[: args.max_checkpoints]
    matrix_root = root / "benchmarks/full_seed42_7mode_latency"
    engine_root = root / "engines/full_seed42_7mode_latency"
    matrix_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(code) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    failures = root / "benchmarks/seed42_latency_matrix_failures.jsonl"

    print(
        f"Starting seed-42 latency matrix: {len(checkpoints)} checkpoints x {len(MODES)} modes; "
        f"batch=1, imgsz={args.imgsz}, warmup={args.warmup}, iterations={args.iterations}, "
        f"timing_images={args.timing_samples}",
        flush=True,
    )
    for index, row in enumerate(checkpoints, start=1):
        dataset = row["dataset"]
        case = row["case"]
        run_tag = row["run_tag"]
        weight = root / row["bundle_path"]
        data_root = root / "data" / dataset
        test_json = data_root / "annotations/seed_42/test.json"
        model_artifact_root = engine_root / dataset / case / run_tag
        result_dir = matrix_root / dataset / case / run_tag
        result_dir.mkdir(parents=True, exist_ok=True)
        if not weight.is_file() or not test_json.is_file():
            message = f"Missing weights or test annotation for {dataset}/{case}/{run_tag}"
            print("ERROR: " + message, flush=True)
            with failures.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"checkpoint": row, "error": message}) + "\n")
            continue

        print(f"\n[{index}/{len(checkpoints)}] {dataset}/{case}/{run_tag}", flush=True)
        for backend, precision, artifact_type, artifact_precision in MODES:
            result_path = result_dir / f"{slug(backend, precision)}.json"
            if args.resume and result_path.is_file():
                try:
                    old = json.loads(result_path.read_text(encoding="utf-8"))["summary"]
                    if (old.get("latency_scope", "").startswith("model inference only")
                            and old.get("latency_median_ms") and old.get("fps_median")):
                        print(f"SKIP existing {backend} {precision}", flush=True)
                        continue
                except (OSError, json.JSONDecodeError, KeyError):
                    pass

            model_path = weight
            try:
                if artifact_type == "onnx":
                    artifact_dir = model_artifact_root / f"onnx_{artifact_precision.lower()}"
                    model_path = artifact_dir / "model.onnx"
                    report_path = artifact_dir / "build_report.json"
                    if not (model_path.is_file() and report_path.is_file()):
                        run([
                            sys.executable, str(code / "export_onnx_variant.py"),
                            "--weights", str(weight), "--output-dir", str(artifact_dir),
                            "--precision", artifact_precision, "--imgsz", str(args.imgsz),
                            "--batch", "1", "--device", "0",
                        ], env)
                elif artifact_type == "engine":
                    artifact_dir = model_artifact_root / artifact_precision.lower()
                    model_path = artifact_dir / "best.engine"
                    report_path = artifact_dir / "build_report.json"
                    if not (model_path.is_file() and report_path.is_file()):
                        command = [
                            sys.executable, str(code / "export_engine_variant.py"),
                            "--weights", str(weight), "--output-dir", str(artifact_dir),
                            "--precision", artifact_precision, "--imgsz", str(args.imgsz),
                            "--batch", "1",
                        ]
                        if artifact_precision == "INT8":
                            calibration_yaml = root / "int8_calibration_seed42" / dataset / "data.yaml"
                            if not calibration_yaml.is_file():
                                raise FileNotFoundError(f"No {dataset}-specific INT8 calibration YAML: {calibration_yaml}")
                            command.extend(["--data", str(calibration_yaml)])
                        run(command, env)
                if not model_path.is_file():
                    raise FileNotFoundError(model_path)

                run([
                    sys.executable, str(code / "benchmark_jetson.py"),
                    "--model", str(model_path), "--backend", backend,
                    "--precision", precision, "--test-json", str(test_json),
                    "--image-dir", str(data_root), "--output", str(result_path),
                    "--imgsz", str(args.imgsz), "--warmup", str(args.warmup),
                    "--iterations", str(args.iterations), "--timing-samples", str(args.timing_samples),
                    "--timing-only", "--dataset", dataset,
                ], env)
                blob = json.loads(result_path.read_text(encoding="utf-8"))
                blob["summary"].update({"case": case, "run_tag": run_tag, "seed": 42})
                result_path.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
                print(
                    f"DONE {backend}/{precision}: model-only median={blob['summary']['latency_median_ms']:.3f} ms, "
                    f"FPS(median)={blob['summary']['fps_median']:.3f}",
                    flush=True,
                )
            except Exception as exc:
                detail = {
                    "checkpoint": row,
                    "backend": backend,
                    "precision": precision,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                print(f"FAILED {backend}/{precision}: {exc}", flush=True)
                with failures.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(detail) + "\n")
            finally:
                write_summary_csv(root, matrix_root)

    write_summary_csv(root, matrix_root)
    print(f"Latency matrix loop complete. Summary: {root / 'benchmarks/seed42_latency_matrix_summary.csv'}", flush=True)
    if failures.is_file():
        print(f"Failures, if any: {failures}", flush=True)


if __name__ == "__main__":
    main()
