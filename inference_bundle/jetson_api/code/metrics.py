"""Per-run inference/resource metrics for the checkpoint UI."""

from __future__ import annotations

import ctypes
import os
import threading
import time
from dataclasses import dataclass

import psutil
import torch

from inference import predict_image


def model_profile(model, image_size: int) -> dict:
    """Count parameters and estimate 2*MACs for one forward at ``image_size``."""
    model_module = getattr(model, "model", None)
    if model_module is None or not callable(getattr(model_module, "parameters", None)):
        # Exported runtimes (e.g. INT8 ONNX) do not expose PyTorch parameters.
        return {"parameters": None, "gflops": None}
    parameters = sum(parameter.numel() for parameter in model_module.parameters())
    try:
        from thop import profile

        # Profiling at 256 avoids a second full-resolution pass. Convolutional
        # work scales with pixel count; this is an estimate, not a latency metric.
        sample_size = min(int(image_size), 256)
        parameter = next(model_module.parameters())
        device = parameter.device
        channels = model.model.yaml.get("ch", 3)
        if not isinstance(channels, int):
            channels = 3
        dtype = parameter.dtype
        sample = torch.zeros(1, channels, sample_size, sample_size, device=device, dtype=dtype)
        with torch.inference_mode():
            macs, _ = profile(model_module.eval(), inputs=(sample,), verbose=False)
        gflops = 2.0 * float(macs) / 1e9 * (image_size / sample_size) ** 2
    except Exception:
        gflops = None
    return {"parameters": parameters, "gflops": gflops}


class _GpuUtilization:
    """Read whole-device GPU utilization from the already-mounted NVML driver."""

    class _Rates(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

    def __init__(self, index: int):
        self.library = ctypes.CDLL("libnvidia-ml.so.1")
        self.library.nvmlInit_v2.restype = ctypes.c_int
        self.library.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        self.library.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
        self.library.nvmlDeviceGetUtilizationRates.argtypes = [ctypes.c_void_p, ctypes.POINTER(self._Rates)]
        self.library.nvmlDeviceGetUtilizationRates.restype = ctypes.c_int
        if self.library.nvmlInit_v2() != 0:
            raise RuntimeError("NVML initialization failed")
        self.handle = ctypes.c_void_p()
        if self.library.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(self.handle)) != 0:
            raise RuntimeError("GPU not available to NVML")

    def sample(self) -> int | None:
        rates = self._Rates()
        if self.library.nvmlDeviceGetUtilizationRates(self.handle, ctypes.byref(rates)) == 0:
            return int(rates.gpu)
        return None


@dataclass
class _GpuSampler:
    gpu: _GpuUtilization

    def __post_init__(self):
        self.samples: list[int] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop_event.is_set():
            value = self.gpu.sample()
            if value is not None:
                self.samples.append(value)
            self.stop_event.wait(0.1)

    def start(self):
        self.thread.start()

    def stop(self) -> int | None:
        self.stop_event.set()
        self.thread.join(timeout=1)
        return max(self.samples) if self.samples else None


def timed_prediction(model, image, *, device: str, **kwargs):
    """Measure a user-visible prediction, excluding checkpoint load/profiling."""
    use_cuda = device != "cpu" and torch.cuda.is_available()
    cuda_device = torch.device(f"cuda:{device}") if use_cuda else None
    sampler = None
    if use_cuda:
        try:
            sampler = _GpuSampler(_GpuUtilization(int(device)))
        except (OSError, RuntimeError, ValueError):
            pass
        torch.cuda.synchronize(cuda_device)
        torch.cuda.reset_peak_memory_stats(cuda_device)

    process = psutil.Process(os.getpid())
    psutil.cpu_percent(interval=None)
    cpu_before = process.cpu_times()
    timing_breakdown = {"preprocess_ms": 0.0, "model_forward_ms": 0.0, "postprocess_ms": 0.0}
    if sampler:
        sampler.start()
    start = time.perf_counter()
    try:
        detections = predict_image(model, image, device=device, timing_breakdown=timing_breakdown, **kwargs)
        if use_cuda:
            torch.cuda.synchronize(cuda_device)
    finally:
        elapsed = max(time.perf_counter() - start, 1e-9)
        gpu_load = sampler.stop() if sampler else None

    cpu_after = process.cpu_times()
    cpu_seconds = (cpu_after.user + cpu_after.system) - (cpu_before.user + cpu_before.system)
    metrics = {
        "inference_ms": elapsed * 1000,
        "fps": 1.0 / elapsed,
        **timing_breakdown,
        "model_forward_fps": (
            1000.0 / timing_breakdown["model_forward_ms"]
            if timing_breakdown["model_forward_ms"] > 0 else None
        ),
        "cpu_load_percent": psutil.cpu_percent(interval=None),
        "process_cpu_percent": 100.0 * cpu_seconds / elapsed,
        "process_ram_mib": process.memory_info().rss / 2**20,
        "gpu_load_percent": gpu_load,
        "gpu_peak_mib": torch.cuda.max_memory_allocated(cuda_device) / 2**20 if use_cuda else None,
        "device_name": torch.cuda.get_device_name(cuda_device) if use_cuda else "CPU",
    }
    return detections, metrics
