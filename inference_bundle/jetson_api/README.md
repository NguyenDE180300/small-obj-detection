# Jetson inference API

This package runs the FastAPI inference service on Jetson Orin. The browser UI
stays on the workstation and proxies `/api/health`, `/api/catalog`, and
`/api/infer` to the Jetson over the USB network (`192.168.55.1:8765`).

The API loads only checkpoint paths listed in `models/manifest.json`; model
weights, datasets, API credentials, and wheel files are intentionally not
stored in this Git repository. The deployed Jetson service keeps those in
`~/jetson_inference_service/models` and `~/jetson_inference_service/wheelhouse`.

## Runtime layout

```text
~/jetson_inference_service/
├── code/                    # FastAPI API and static assets
├── custom_code/ultralytics/  # custom YOLO model definitions
├── models/                  # checkpoint tree and manifest.json (not in Git)
├── wheelhouse/              # offline FastAPI wheels (not in Git)
├── .venv/                   # API venv referencing the verified CUDA Torch env
└── start_backend.sh
```

The service advertises FP32 and FP16. It binds only to `192.168.55.1`, not to a
public interface. It uses the existing JetPack PyTorch/CUDA runtime and does not
install or replace system CUDA, TensorRT, or OS packages.

## Deploy/update from the workstation

With the Jetson connected by USB and SSH reachable as `ait01@192.168.55.1`:

```bash
bash inference_bundle/jetson_api/deploy_to_jetson.sh
```

The script updates only the API source, custom Ultralytics code, and launcher;
it does not overwrite checkpoints, the manifest, virtual environments, or
wheelhouses. It restarts the `inference-api` tmux session and checks the API
health endpoint. SSH may prompt for the Jetson account password.

For a fresh service setup, first provision the two offline wheelhouses expected
by `setup_api_environment.sh`, then run it on the Jetson:

```bash
cd ~/jetson_inference_service
bash setup_api_environment.sh
```

The API expects the already-prepared model tree and manifest. To verify it:

```bash
curl http://192.168.55.1:8765/api/health
curl http://192.168.55.1:8765/api/catalog
```

The local UI's Docker compose uses `JETSON_BACKEND_URL=http://192.168.55.1:8765`.
Keep the USB link connected while using inference from the workstation UI.
