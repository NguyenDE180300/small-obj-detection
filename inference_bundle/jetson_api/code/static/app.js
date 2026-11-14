const byId = (id) => document.getElementById(id);
const state = { checkpoints: [], selectedFile: null, objectUrl: null, downloadImageUrl: null, downloadJsonUrl: null, lastCheckpointId: null, device: "cpu" };

function setStatus(message = "", kind = "") {
  const target = byId("status");
  target.textContent = message;
  target.className = `status ${kind}`.trim();
}

function invalidateResult() {
  byId("result-grid").hidden = true;
  byId("details").hidden = true;
}

function formatMetric(value, suffix = "", digits = 2) {
  return value === null || value === undefined || !Number.isFinite(Number(value))
    ? "N/A"
    : `${Number(value).toFixed(digits)}${suffix}`;
}

function matchingCheckpoints() {
  const dataset = byId("dataset-filter").value;
  const family = byId("family-filter").value;
  const search = byId("checkpoint-search").value.trim().toLowerCase();
  return state.checkpoints.filter((item) =>
    (dataset === "all" || item.dataset === dataset) &&
    (family === "all" || item.family === family) &&
    `${item.case} ${item.run_tag} ${item.seed}`.toLowerCase().includes(search)
  );
}

function refreshCheckpointOptions(preferredId = null) {
  const select = byId("checkpoint-select");
  const choices = matchingCheckpoints();
  select.replaceChildren();
  choices.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = `${item.dataset_label} · ${item.case} · ${item.seed} · ${item.run_tag}`;
    select.append(option);
  });
  if (preferredId && choices.some((item) => item.id === preferredId)) select.value = preferredId;
  select.disabled = choices.length === 0;
  if (!choices.length) {
    const option = document.createElement("option");
    option.textContent = "No checkpoints match these filters";
    select.append(option);
  }
  byId("run-inference").disabled = !state.selectedFile || choices.length === 0;
  updateDatasetDefaults();
  return choices;
}

function selectedCheckpoint() {
  return state.checkpoints.find((item) => item.id === byId("checkpoint-select").value) || null;
}

function updateDatasetDefaults() {
  const selected = selectedCheckpoint();
  if (!selected) return;
  if (selected.id !== state.lastCheckpointId) {
    byId("imgsz").value = selected.dataset === "levir_ship" ? 512 : 640;
    state.lastCheckpointId = selected.id;
  }
  const tiled = byId("tiled");
  if (tiled.dataset.userChanged !== "true") tiled.checked = selected.dataset === "tinyperson";
  byId("tile-options").hidden = !tiled.checked;
}

function dataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error || new Error("Could not read the image."));
    reader.readAsDataURL(file);
  });
}

function metric(id, value) { byId(id).textContent = value; }

function populateResult(data, decodeMs, inputUrl, imgsz) {
  const values = data.metrics;
  const annotatedUrl = `data:image/png;base64,${data.annotated_png}`;
  state.annotatedUrl = annotatedUrl;
  byId("input-preview").src = inputUrl;
  byId("output-preview").src = annotatedUrl;
  byId("result-grid").hidden = false;
  byId("details").hidden = false;

  metric("metric-forward", formatMetric(values.model_forward_ms, " ms"));
  metric("metric-model-fps", formatMetric(values.model_forward_fps, "", 2));
  metric("inference-caption", `${values.precision || "FP32"} · ${data.detections.length} detections`);
  metric("metric-ram", formatMetric(values.process_ram_mib, " MiB", 0));
  metric("metric-vram", formatMetric(values.gpu_peak_mib, " MiB", 0));
  metric("metric-gpu", values.gpu_load_percent == null ? "N/A" : `${values.gpu_load_percent}%`);
  metric("metric-cpu", formatMetric(values.process_cpu_percent, "%", 1));

  metric("detail-decode", formatMetric(decodeMs, " ms", 1));
  metric("detail-preprocess", formatMetric(values.preprocess_ms, " ms"));
  metric("detail-input-size", `${data.image_width} × ${data.image_height}`);
  metric("detail-model-size", `${imgsz} × ${imgsz}`);
  metric("detail-postprocess", formatMetric(values.postprocess_ms, " ms"));
  metric("detail-annotation", formatMetric(values.annotation_ms, " ms", 1));
  metric("detail-params", values.parameters == null ? "N/A" : `${(values.parameters / 1e6).toFixed(3)} M`);
  metric("detail-gflops", formatMetric(values.gflops, "", 3));
  metric("detail-e2e", formatMetric(values.inference_ms, " ms", 1));
  metric("detail-pipeline-fps", formatMetric(values.fps, "", 2));
  metric("detail-network", formatMetric(values.ui_gateway_roundtrip_ms, " ms", 1));
  metric("detail-system-cpu", formatMetric(values.cpu_load_percent, "%", 1));
  metric("detail-detections", String(data.detections.length));

  const imageBlob = Uint8Array.from(atob(data.annotated_png), (char) => char.charCodeAt(0));
  if (state.downloadImageUrl) URL.revokeObjectURL(state.downloadImageUrl);
  if (state.downloadJsonUrl) URL.revokeObjectURL(state.downloadJsonUrl);
  const downloadBlob = new Blob([imageBlob], { type: "image/png" });
  state.downloadImageUrl = URL.createObjectURL(downloadBlob);
  state.downloadJsonUrl = URL.createObjectURL(new Blob([JSON.stringify(data.detections, null, 2)], { type: "application/json" }));
  byId("download-image").href = state.downloadImageUrl;
  byId("download-json").href = state.downloadJsonUrl;
  renderDetections(data.detections);
  byId("result-grid").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderDetections(detections) {
  const tbody = byId("detection-rows");
  tbody.replaceChildren();
  detections.forEach((detection) => {
    const row = document.createElement("tr");
    const cells = [detection.class_name, Number(detection.confidence).toFixed(4), ...detection.xyxy.map((x) => Number(x).toFixed(1))];
    cells.forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    });
    tbody.append(row);
  });
}

async function runInference() {
  const checkpoint = selectedCheckpoint();
  if (!state.selectedFile || !checkpoint) return;
  const imgsz = Number(byId("imgsz").value);
  const precision = byId("precision").value;
  const device = state.device;
  if (!Number.isInteger(imgsz) || imgsz < 256 || imgsz > 2048 || imgsz % 32 !== 0) {
    setStatus("Image size must be a multiple of 32 between 256 and 2048.", "error");
    return;
  }
  if (precision === "fp16" && device === "cpu") {
    setStatus("FP16 requires a CUDA device. Choose FP32 for CPU inference.", "error");
    return;
  }

  const button = byId("run-inference");
  button.disabled = true;
  button.textContent = "Running inference…";
  setStatus("Loading checkpoint if needed, then running prediction…", "busy");
  const requestStarted = performance.now();
  const form = new FormData();
  form.append("image_file", state.selectedFile);
  form.append("checkpoint_id", checkpoint.id);
  form.append("imgsz", String(imgsz));
  form.append("conf", byId("confidence").value);
  form.append("iou", byId("nms-iou").value);
  form.append("device", device);
  form.append("precision", precision);
  form.append("tiled", String(byId("tiled").checked));
  form.append("tile_size", byId("tile-size").value);
  form.append("tile_stride", byId("tile-stride").value);

  try {
    const response = await fetch("/api/infer", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
    data.metrics.ui_gateway_roundtrip_ms = performance.now() - requestStarted;
    populateResult(data, Number(data.metrics.input_decode_ms || 0), state.objectUrl, imgsz);
    setStatus("Inference complete.", "success");
  } catch (error) {
    setStatus(error.message || "Inference failed.", "error");
  } finally {
  button.disabled = !state.selectedFile || !selectedCheckpoint();
    button.textContent = "Run inference";
  }
}

function installZoom() {
  const modal = byId("lightbox");
  const image = byId("zoom-image");
  const stage = byId("zoom-stage");
  const label = byId("zoom-level");
  let scale = 1;
  const update = (next, clientX = null, clientY = null) => {
    const rect = image.getBoundingClientRect();
    if (clientX !== null && rect.width && rect.height) {
      const x = Math.max(0, Math.min(rect.width, clientX - rect.left));
      const y = Math.max(0, Math.min(rect.height, clientY - rect.top));
      image.style.transformOrigin = `${(x / rect.width) * 100}% ${(y / rect.height) * 100}%`;
    } else image.style.transformOrigin = "center center";
    scale = Math.max(1, Math.min(6, next));
    image.style.transform = `scale(${scale})`;
    label.textContent = `${Math.round(scale * 100)}%`;
  };
  document.querySelectorAll("[data-zoom-target]").forEach((button) => button.addEventListener("click", () => {
    image.src = byId(button.dataset.zoomTarget).src;
    modal.hidden = false;
    document.body.style.overflow = "hidden";
    update(1);
  }));
  const close = () => { modal.hidden = true; document.body.style.overflow = ""; };
  byId("lightbox-close").addEventListener("click", close);
  modal.addEventListener("click", (event) => { if (event.target === modal) close(); });
  document.addEventListener("keydown", (event) => { if (event.key === "Escape" && !modal.hidden) close(); });
  stage.addEventListener("wheel", (event) => {
    event.preventDefault();
    update(scale * Math.exp(-event.deltaY * 0.0015), event.clientX, event.clientY);
  }, { passive: false });
  byId("zoom-in").addEventListener("click", () => update(scale * 1.25));
  byId("zoom-out").addEventListener("click", () => update(scale / 1.25));
  byId("zoom-reset").addEventListener("click", () => update(1));
}

async function initialize() {
  try {
    const [healthResponse, catalogResponse] = await Promise.all([fetch("/api/health"), fetch("/api/catalog")]);
    if (!healthResponse.ok || !catalogResponse.ok) throw new Error("Could not load the inference service catalog.");
    const catalog = await catalogResponse.json();
    state.checkpoints = catalog.checkpoints;
    if (Array.isArray(catalog.precisions) && catalog.precisions.length) {
      const precisionSelect = byId("precision");
      precisionSelect.replaceChildren();
      catalog.precisions.forEach((precision) => {
        const option = document.createElement("option");
        option.value = precision.id;
        option.textContent = precision.label;
        precisionSelect.append(option);
      });
    }
    byId("checkpoint-count").textContent = String(state.checkpoints.length);
    state.device = catalog.devices.find((device) => device.id !== "cpu")?.id || "cpu";
    refreshCheckpointOptions();
    if (!state.checkpoints.length) setStatus("No downloaded checkpoints found. Run download_checkpoints.py first.", "error");
  } catch (error) {
    setStatus(error.message || "Could not connect to the API.", "error");
  }
}

byId("dataset-filter").addEventListener("change", () => { refreshCheckpointOptions(); invalidateResult(); });
byId("family-filter").addEventListener("change", () => { refreshCheckpointOptions(); invalidateResult(); });
byId("checkpoint-search").addEventListener("input", () => { refreshCheckpointOptions(); invalidateResult(); });
byId("checkpoint-select").addEventListener("change", () => { updateDatasetDefaults(); invalidateResult(); });
byId("imgsz").addEventListener("change", invalidateResult);
byId("confidence").addEventListener("input", (event) => { byId("confidence-value").value = Number(event.target.value).toFixed(2); invalidateResult(); });
byId("nms-iou").addEventListener("input", (event) => { byId("iou-value").value = Number(event.target.value).toFixed(2); invalidateResult(); });
byId("tiled").addEventListener("change", (event) => {
  event.target.dataset.userChanged = "true";
  byId("tile-options").hidden = !event.target.checked;
  invalidateResult();
});
byId("precision").addEventListener("change", (event) => {
  byId("precision-hint").textContent = event.target.value === "int8"
    ? "Hybrid INT8 runs on CPU. First use may export and calibrate the model from the dataset training images."
    : "FP16 requires a CUDA device. FP32 can run on GPU or CPU.";
  invalidateResult();
});
byId("tile-size").addEventListener("change", invalidateResult);
byId("tile-stride").addEventListener("change", invalidateResult);
byId("image-upload").addEventListener("change", (event) => {
  state.selectedFile = event.target.files?.[0] || null;
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.objectUrl = state.selectedFile ? URL.createObjectURL(state.selectedFile) : null;
  byId("run-inference").disabled = !state.selectedFile || !selectedCheckpoint();
  byId("file-meta").textContent = state.selectedFile ? `${state.selectedFile.name} · ${(state.selectedFile.size / 1024).toFixed(1)} KiB` : "Choose an image to begin.";
  byId("result-grid").hidden = true;
  byId("details").hidden = true;
  if (state.selectedFile) byId("input-preview").src = state.objectUrl;
});
byId("run-inference").addEventListener("click", runInference);
byId("sidebar-toggle").addEventListener("click", () => {
  byId("sidebar").classList.add("collapsed");
  document.body.classList.add("sidebar-collapsed");
});
byId("sidebar-open").addEventListener("click", () => {
  byId("sidebar").classList.remove("collapsed");
  document.body.classList.remove("sidebar-collapsed");
});
installZoom();
initialize();
