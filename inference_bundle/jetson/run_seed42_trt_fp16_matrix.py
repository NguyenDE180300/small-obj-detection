#!/usr/bin/env python3
"""Benchmark TensorRT FP16 only for every packaged seed-42 checkpoint.

Uses separate engine/result roots so the previous seven-mode matrix remains
intact. The existing Jetson shared timing cache is reused by export_engine_variant.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path


def write_summary(root: Path, matrix_root: Path) -> None:
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
            "Seed": summary.get("seed"),
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
    output = root / "benchmarks/seed42_trt_fp16_deduplicated_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)


def run(command: list[str], env: dict[str, str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    code = root / "code"
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--timing-samples", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    manifest = json.loads((root / "checkpoint_manifest_seed42.json").read_text(encoding="utf-8"))
    checkpoints = manifest["checkpoints"]
    datasets = sorted({row["dataset"] for row in checkpoints})
    if not {"levir_ship", "varroa", "tinyperson"}.issubset(datasets):
        raise RuntimeError(f"Manifest does not cover all three requested datasets: {datasets}")

    matrix_root = root / "benchmarks/full_seed42_trt_fp16_deduplicated"
    engine_root = root / "engines/full_seed42_trt_fp16_deduplicated"
    old_matrix_root = root / "benchmarks/full_seed42_7mode_latency"
    old_engine_root = root / "engines/full_seed42_7mode_latency"
    first_fp16_matrix_root = root / "benchmarks/full_seed42_trt_fp16_only"
    first_fp16_engine_root = root / "engines/full_seed42_trt_fp16_only"
    matrix_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(code) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    failures = root / "benchmarks/seed42_trt_fp16_deduplicated_failures.jsonl"

    print(
        f"Starting deduplicated TensorRT-FP16-only matrix: {len(checkpoints)} checkpoints; "
        f"datasets={','.join(datasets)}; batch=1, imgsz={args.imgsz}, "
        f"warmup={args.warmup}, iterations={args.iterations}, timing_images={args.timing_samples}",
        flush=True,
    )
    for index, row in enumerate(checkpoints, start=1):
        dataset, case, run_tag = row["dataset"], row["case"], row["run_tag"]
        weight = root / row["bundle_path"]
        data_root = root / "data" / dataset
        test_json = data_root / "annotations/seed_42/test.json"
        model_artifact_root = engine_root / dataset / case / run_tag / "fp16"
        result_path = matrix_root / dataset / case / run_tag / "tensorrt_fp16.json"
        model_path = model_artifact_root / "best.engine"

        if result_path.is_file():
            try:
                old = json.loads(result_path.read_text(encoding="utf-8"))["summary"]
                if (old.get("latency_scope", "").startswith("model inference only")
                        and old.get("latency_median_ms") and old.get("fps_median")):
                    print(f"SKIP existing [{index}/{len(checkpoints)}] {dataset}/{case}/{run_tag}", flush=True)
                    continue
            except (OSError, json.JSONDecodeError, KeyError):
                pass

        # Import an already completed FP16 measurement from the old seven-mode
        # matrix, or from the initial FP16-only attempt, instead of measuring it
        # twice. Copy/reuse the matching Jetson-built engine as well when present.
        reused = False
        for source_matrix_root, source_engine_root in (
            (old_matrix_root, old_engine_root),
            (first_fp16_matrix_root, first_fp16_engine_root),
        ):
            source_result = source_matrix_root / dataset / case / run_tag / "tensorrt_fp16.json"
            if not source_result.is_file():
                continue
            try:
                blob = json.loads(source_result.read_text(encoding="utf-8"))
                summary = blob["summary"]
                if (
                    summary.get("backend") != "TensorRT"
                    or summary.get("precision") != "FP16"
                    or not summary.get("latency_scope", "").startswith("model inference only")
                    or not summary.get("latency_median_ms")
                    or not summary.get("fps_median")
                ):
                    continue
            except (OSError, json.JSONDecodeError, KeyError):
                continue

            result_path.parent.mkdir(parents=True, exist_ok=True)
            source_artifact = source_engine_root / dataset / case / run_tag / "fp16"
            model_artifact_root.mkdir(parents=True, exist_ok=True)
            for artifact_name in ("best.engine", "build_report.json"):
                source_artifact_file = source_artifact / artifact_name
                destination = model_artifact_root / artifact_name
                if source_artifact_file.is_file() and not destination.exists():
                    try:
                        os.link(source_artifact_file, destination)
                    except OSError:
                        shutil.copy2(source_artifact_file, destination)
            if model_path.is_file():
                summary["model"] = str(model_path)
            summary.update({"case": case, "run_tag": run_tag, "seed": 42})
            result_path.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
            print(
                f"IMPORT existing TensorRT/FP16 [{index}/{len(checkpoints)}] "
                f"{dataset}/{case}/{run_tag} from {source_matrix_root.name}",
                flush=True,
            )
            reused = True
            write_summary(root, matrix_root)
            break
        if reused:
            continue

        print(f"\n[{index}/{len(checkpoints)}] {dataset}/{case}/{run_tag}", flush=True)
        try:
            result_path.parent.mkdir(parents=True, exist_ok=True)
            if not weight.is_file() or not test_json.is_file():
                raise FileNotFoundError(f"Missing weights or test annotation: {weight} / {test_json}")
            if not (model_path.is_file() and (model_artifact_root / "build_report.json").is_file()):
                run([
                    sys.executable, str(code / "export_engine_variant.py"),
                    "--weights", str(weight), "--output-dir", str(model_artifact_root),
                    "--precision", "FP16", "--imgsz", str(args.imgsz), "--batch", "1",
                ], env)
            if not model_path.is_file():
                raise FileNotFoundError(model_path)
            run([
                sys.executable, str(code / "benchmark_jetson.py"),
                "--model", str(model_path), "--backend", "TensorRT",
                "--precision", "FP16", "--test-json", str(test_json),
                "--image-dir", str(data_root), "--output", str(result_path),
                "--imgsz", str(args.imgsz), "--warmup", str(args.warmup),
                "--iterations", str(args.iterations), "--timing-samples", str(args.timing_samples),
                "--timing-only", "--dataset", dataset,
            ], env)
            blob = json.loads(result_path.read_text(encoding="utf-8"))
            blob["summary"].update({"case": case, "run_tag": run_tag, "seed": 42})
            result_path.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
            print(
                f"DONE TensorRT/FP16: model-only median={blob['summary']['latency_median_ms']:.3f} ms, "
                f"FPS(median)={blob['summary']['fps_median']:.3f}",
                flush=True,
            )
        except Exception as exc:
            detail = {
                "checkpoint": row,
                "backend": "TensorRT",
                "precision": "FP16",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            print(f"FAILED TensorRT/FP16: {exc}", flush=True)
            with failures.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(detail) + "\n")
        finally:
            write_summary(root, matrix_root)

    write_summary(root, matrix_root)
    print(
        f"Deduplicated FP16-only matrix complete. Summary: "
        f"{root / 'benchmarks/seed42_trt_fp16_deduplicated_summary.csv'}",
        flush=True,
    )
    if failures.is_file():
        print(f"Failures, if any: {failures}", flush=True)


if __name__ == "__main__":
    main()
