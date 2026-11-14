#!/usr/bin/env bash
set -euo pipefail

SERVICE_ROOT="${JETSON_INFERENCE_ROOT:-$HOME/jetson_inference_service}"
VENV_PYTHON="${JETSON_INFERENCE_PYTHON:-$SERVICE_ROOT/.venv/bin/python}"

export MODEL_ROOT="$SERVICE_ROOT/models"
export CUSTOM_CODE_ROOT="$SERVICE_ROOT/custom_code"
export CALIBRATION_ROOT="$SERVICE_ROOT/no_calibration_data"
export INT8_CACHE_ROOT="$SERVICE_ROOT/int8_cache"
export YOLO_CONFIG_DIR="$SERVICE_ROOT/ultralytics_config"
export SUPPORTED_PRECISIONS="fp32,fp16"
export PYTHONPATH="$CUSTOM_CODE_ROOT:$SERVICE_ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
unset JETSON_BACKEND_URL

mkdir -p "$INT8_CACHE_ROOT" "$YOLO_CONFIG_DIR"
exec "$VENV_PYTHON" -m uvicorn --app-dir "$SERVICE_ROOT/code" app:app \
  --host 192.168.55.1 --port 8765 --workers 1
