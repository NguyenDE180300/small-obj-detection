# Jetson Orin deployment (batch 16)

## Benchmark timing convention

The checkpoint benchmark reports model-only latency/FPS as its primary timing:
`result.speed["inference"]`, i.e. the backend model call profiled by Ultralytics.
It excludes Ultralytics preprocessing and NMS/postprocessing, image-file I/O,
and model loading. Warmup calls are discarded before measurement. For context,
the JSON also records end-to-end `model.predict()` wall latency/FPS separately;
do not mix these two timing scopes in comparisons.

This bundle targets the `aduy_gap_factorized_k15_rgb_saturation`, LEVIR-Ship,
seed-42 checkpoint. It exports an FP16 TensorRT engine on the Jetson itself,
with a dynamic batch profile from 1 through 16 (optimization point 16), then
starts a small Streamlit UI that accepts up to 16 images per request.

## Why build the engine on the Jetson?

TensorRT engines are tied to the TensorRT/runtime and GPU platform that built
them. Do not copy the RTX 3060 `.engine` files to Orin. The included dynamic
ONNX (`model_dynamic_b16.onnx`) was exported from the same checkpoint and
verified to accept batch sizes 1, 8, and 16 on ONNX Runtime CPU; the Jetson
setup script exports the actual FP16 engine locally from `best.pt` so Ultralytics
metadata is embedded and the UI can load it.

## Deploy

1. Copy `jetson_orin_batch16.tar.gz` to the Jetson and extract it.
2. In the extracted folder run:

   ```bash
   chmod +x setup_run.sh
   ./setup_run.sh
   ```

3. Open the URL printed by Streamlit (usually `http://<jetson-ip>:8501`).
   Upload up to 16 images in one request. The TensorRT engine accepts a dynamic
   batch with maximum 16; a smaller final batch is allowed.

Do not expose port 8501 to the public internet; use it on a trusted local network
or put an authenticated reverse proxy in front of it.

The setup script does not flash JetPack, change system packages, or configure
Docker. If the archive contains `wheelhouse/`, pip, PyTorch/TorchVision, and the
UI dependencies are installed offline into the bundle-local virtualenv. This
is intended for a Jetson reachable over USB/SSH but without Internet. Without a
wheelhouse it uses the online package indexes. TensorRT must already be present
from JetPack. If a prerequisite is missing or incompatible, it stops and
prints diagnostics; send that output back before changing system packages.

## Device details to capture if setup stops

```bash
uname -m
cat /etc/nv_tegra_release
python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
python3 -c 'import tensorrt as trt; print(trt.__version__)'
```

The engine is built for 640×640, FP16, batch profile min=1 / opt=16 / max=16.
Batch 16 increases throughput for groups of images, not single-image latency;
actual memory use and throughput must be measured on the specific Orin module.
