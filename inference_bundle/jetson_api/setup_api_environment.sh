#!/usr/bin/env bash
set -euo pipefail

SERVICE_ROOT="${JETSON_INFERENCE_ROOT:-$HOME/jetson_inference_service}"
BASE_ENV="${JETSON_BASE_ENV:-$HOME/jetson_orin_batch16/.venv}"
BASE_WHEELHOUSE="$HOME/jetson_orin_batch16/wheelhouse"
API_WHEELHOUSE="$SERVICE_ROOT/wheelhouse"
VENV="$SERVICE_ROOT/.venv"

if [[ ! -x "$BASE_ENV/bin/python" || ! -f "$BASE_WHEELHOUSE/get-pip.py" ]]; then
  echo "Existing verified Jetson Torch environment/wheelhouse was not found." >&2
  exit 1
fi
if [[ ! -d "$API_WHEELHOUSE" ]]; then
  echo "API wheelhouse missing: $API_WHEELHOUSE" >&2
  exit 1
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv --system-site-packages --without-pip "$VENV"
fi
"$VENV/bin/python" "$BASE_WHEELHOUSE/get-pip.py" \
  --no-index --find-links "$BASE_WHEELHOUSE" pip==26.2.2

# Reuse the already validated Torch/TorchVision/CUDA Python packages without
# modifying that environment; FastAPI/Uvicorn and their dependencies stay here.
printf '%s\n' "$BASE_ENV/lib/python3.10/site-packages" \
  > "$VENV/lib/python3.10/site-packages/jetson_torch_runtime.pth"

"$VENV/bin/python" -m pip install --no-index \
  --find-links "$BASE_WHEELHOUSE" --find-links "$API_WHEELHOUSE" \
  fastapi==0.115.14 uvicorn==0.34.3 python-multipart==0.0.20

"$VENV/bin/python" - <<'PY'
import cv2, fastapi, torch, uvicorn
print("FastAPI:", fastapi.__version__)
print("Uvicorn:", uvicorn.__version__)
print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())
print("OpenCV:", cv2.__version__)
PY
