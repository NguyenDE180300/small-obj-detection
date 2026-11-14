#!/usr/bin/env bash
set -euo pipefail

JETSON_HOST="${JETSON_HOST:-ait01@192.168.55.1}"
SERVICE_ROOT="${JETSON_INFERENCE_ROOT:-/home/ait01/jetson_inference_service}"
PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

command -v rsync >/dev/null || { echo "rsync is required on the workstation" >&2; exit 1; }
command -v ssh >/dev/null || { echo "ssh is required on the workstation" >&2; exit 1; }

ssh "$JETSON_HOST" "mkdir -p '$SERVICE_ROOT/code' '$SERVICE_ROOT/custom_code/ultralytics'"
rsync -a "$PACKAGE_ROOT/code/" "$JETSON_HOST:$SERVICE_ROOT/code/"
rsync -a "$PACKAGE_ROOT/custom_code/ultralytics/" "$JETSON_HOST:$SERVICE_ROOT/custom_code/ultralytics/"
rsync -a "$PACKAGE_ROOT/Logo_Trường_Đại_học_FPT.svg.webp" "$JETSON_HOST:$SERVICE_ROOT/"
rsync -a "$PACKAGE_ROOT/start_backend.sh" "$PACKAGE_ROOT/setup_api_environment.sh" "$JETSON_HOST:$SERVICE_ROOT/"

ssh "$JETSON_HOST" "SERVICE_ROOT='$SERVICE_ROOT' bash -s" <<'REMOTE'
set -euo pipefail
cd "$SERVICE_ROOT"
if ! test -x .venv/bin/python; then
  echo "API virtualenv is missing; run setup_api_environment.sh after provisioning its wheelhouses." >&2
  exit 1
fi
if ! test -f models/manifest.json; then
  echo "Checkpoint manifest is missing at $SERVICE_ROOT/models/manifest.json." >&2
  exit 1
fi
if tmux has-session -t inference-api 2>/dev/null; then
  tmux kill-session -t inference-api
fi
tmux new-session -d -s inference-api "cd '$SERVICE_ROOT' && bash start_backend.sh >> backend.log 2>&1"
REMOTE

for attempt in $(seq 1 20); do
  if curl -fsS "http://192.168.55.1:8765/api/health"; then
    printf '\nJetson inference API is healthy.\n'
    exit 0
  fi
  sleep 1
done
echo "Jetson API did not become healthy; inspect backend.log on Jetson." >&2
exit 1
