#!/usr/bin/env python3
"""Run the seed-42 FP16 benchmark matrix on PT, TensorRT, and ONNX Runtime CPU.

Previously completed measurements are imported from the seven-mode and
TensorRT-FP16-only matrices. Only missing/invalid checkpoint-mode pairs run.
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


MODES = (
    ("PyTorch .pt", "FP16", "pt"),
    ("TensorRT", "FP16", "trt"),
    ("ONNX Runtime CPU", "FP16", "onnx_cpu"),
)


def valid_result(path: Path, backend: str) -> dict | None:
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
        summary = blob["summary"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None
    if (
        summary.get("backend") != backend
        or summary.get("precision") != "FP16"
        or not summary.get("latency_scope", "").startswith("model inference only")
        or not summary.get("latency_median_ms")
        or not summary.get("fps_median")
    ):
        return None
    return blob


def write_summary(root: Path, matrix_root: Path) -> None:
    rows = []
    for result in sorted(matrix_root.rglob("*.json")):
        try:
            blob = json.loads(result.read_text(encoding="utf-8"))
            summary = blob["summary"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            continue
        if (
            summary.get("precision") != "FP16"
            or not summary.get("latency_scope", "").startswith("model inference only")
            or not summary.get("latency_median_ms")
            or not summary.get("fps_median")
        ):
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
    output = root / "benchmarks/seed42_fp16_3backend_summary.csv"
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

    checkpoints = json.loads((root / "checkpoint_manifest_seed42.json").read_text(encoding="utf-8"))["checkpoints"]
    matrix_root = root / "benchmarks/full_seed42_fp16_3backend_deduplicated"
    artifact_root = root / "engines/full_seed42_fp16_3backend_deduplicated"
    old_matrix = root / "benchmarks/full_seed42_7mode_latency"
    old_artifacts = root / "engines/full_seed42_7mode_latency"
    trt_matrix = root / "benchmarks/full_seed42_trt_fp16_deduplicated"
    trt_artifacts = root / "engines/full_seed42_trt_fp16_deduplicated"
    first_trt_matrix = root / "benchmarks/full_seed42_trt_fp16_only"
    first_trt_artifacts = root / "engines/full_seed42_trt_fp16_only"
    failures = root / "benchmarks/seed42_fp16_3backend_failures.jsonl"
    matrix_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(code) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    total = len(checkpoints) * len(MODES)
    done = 0
    print(f"FP16 matrix: {len(checkpoints)} checkpoints x {len(MODES)} modes = {total} pairs; "
          f"batch=1, imgsz={args.imgsz}, warmup={args.warmup}, iterations={args.iterations}", flush=True)

    for checkpoint_index, row in enumerate(checkpoints, start=1):
        dataset, case, run_tag = row["dataset"], row["case"], row["run_tag"]
        weight = root / row["bundle_path"]
        data_root = root / "data" / dataset
        test_json = data_root / "annotations/seed_42/test.json"
        result_dir = matrix_root / dataset / case / run_tag
        result_dir.mkdir(parents=True, exist_ok=True)
        per_checkpoint = artifact_root / dataset / case / run_tag

        for backend, precision, mode in MODES:
            done += 1
            slug = {"PyTorch .pt": "pytorch_pt_fp16", "TensorRT": "tensorrt_fp16",
                    "ONNX Runtime CPU": "onnx_runtime_cpu_fp16"}[backend]
            result_path = result_dir / f"{slug}.json"
            current = valid_result(result_path, backend)
            if current:
                print(f"SKIP current [{done}/{total}] {dataset}/{case}/{run_tag} {backend}/FP16", flush=True)
                continue

            # Import valid prior measurements so no checkpoint/mode is measured twice.
            candidates = []
            if backend == "TensorRT":
                candidates.extend([
                    (trt_matrix / dataset / case / run_tag / "tensorrt_fp16.json", trt_artifacts / dataset / case / run_tag / "fp16"),
                    (first_trt_matrix / dataset / case / run_tag / "tensorrt_fp16.json", first_trt_artifacts / dataset / case / run_tag / "fp16"),
                    (old_matrix / dataset / case / run_tag / "tensorrt_fp16.json", old_artifacts / dataset / case / run_tag / "fp16"),
                ])
            else:
                candidates.append((old_matrix / dataset / case / run_tag / f"{slug}.json", old_artifacts / dataset / case / run_tag))
            imported = False
            for source, source_artifact in candidates:
                old_blob = valid_result(source, backend)
                if old_blob is None:
                    continue
                if backend == "TensorRT":
                    source_engine = source_artifact / "best.engine"
                    dest_engine = per_checkpoint / "fp16" / "best.engine"
                    dest_engine.parent.mkdir(parents=True, exist_ok=True)
                    if source_engine.is_file() and not dest_engine.exists():
                        try:
                            os.link(source_engine, dest_engine)
                        except OSError:
                            shutil.copy2(source_engine, dest_engine)
                    if dest_engine.is_file():
                        old_blob["summary"]["model"] = str(dest_engine)
                elif backend == "ONNX Runtime CPU":
                    src_dir = source_artifact / "onnx_fp16"
                    dst_dir = per_checkpoint / "onnx_fp16"
                    if (src_dir / "model.onnx").is_file() and not (dst_dir / "model.onnx").exists():
                        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
                old_blob["summary"].update({"case": case, "run_tag": run_tag, "seed": 42})
                result_path.write_text(json.dumps(old_blob, indent=2) + "\n", encoding="utf-8")
                print(f"IMPORT [{done}/{total}] {dataset}/{case}/{run_tag} {backend}/FP16", flush=True)
                imported = True
                break
            if imported:
                write_summary(root, matrix_root)
                continue

            print(f"\n[{done}/{total}] {dataset}/{case}/{run_tag} {backend}/FP16", flush=True)
            try:
                if not weight.is_file() or not test_json.is_file():
                    raise FileNotFoundError(f"Missing weights or test annotation: {weight} / {test_json}")
                model_path = weight
                if mode == "trt":
                    model_path = per_checkpoint / "fp16" / "best.engine"
                    report = model_path.parent / "build_report.json"
                    if not (model_path.is_file() and report.is_file()):
                        run([sys.executable, str(code / "export_engine_variant.py"),
                             "--weights", str(weight), "--output-dir", str(model_path.parent),
                             "--precision", "FP16", "--imgsz", str(args.imgsz), "--batch", "1"], env)
                elif mode == "onnx_cpu":
                    artifact_dir = per_checkpoint / "onnx_fp16"
                    model_path = artifact_dir / "model.onnx"
                    if not (model_path.is_file() and (artifact_dir / "build_report.json").is_file()):
                        run([sys.executable, str(code / "export_onnx_variant.py"),
                             "--weights", str(weight), "--output-dir", str(artifact_dir),
                             "--precision", "FP16", "--imgsz", str(args.imgsz), "--batch", "1", "--device", "0"], env)
                if not model_path.is_file():
                    raise FileNotFoundError(model_path)
                run([sys.executable, str(code / "benchmark_jetson.py"),
                     "--model", str(model_path), "--backend", backend, "--precision", precision,
                     "--test-json", str(test_json), "--image-dir", str(data_root), "--output", str(result_path),
                     "--imgsz", str(args.imgsz), "--warmup", str(args.warmup),
                     "--iterations", str(args.iterations), "--timing-samples", str(args.timing_samples),
                     "--timing-only", "--dataset", dataset], env)
                blob = json.loads(result_path.read_text(encoding="utf-8"))
                blob["summary"].update({"case": case, "run_tag": run_tag, "seed": 42})
                result_path.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
                print(f"DONE {backend}/FP16: model-only median={blob['summary']['latency_median_ms']:.3f} ms, "
                      f"FPS(median)={blob['summary']['fps_median']:.3f}", flush=True)
            except Exception as exc:
                detail = {"checkpoint": row, "backend": backend, "precision": precision,
                          "error": repr(exc), "traceback": traceback.format_exc()}
                print(f"FAILED {backend}/FP16: {exc}", flush=True)
                with failures.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(detail) + "\n")
            finally:
                write_summary(root, matrix_root)

    write_summary(root, matrix_root)
    print(f"Three-backend FP16 matrix complete: {root / 'benchmarks/seed42_fp16_3backend_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
