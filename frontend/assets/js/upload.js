(() => {
  const $ = (id) => document.getElementById(id);
  const input = $("fileInput"), zone = $("dropZone");
  const allowed = new Set(["mp4", "mov", "avi", "mkv", "webm"]);
  let selectedFile = null, previewUrl = null, maxBytes = 500 * 1024 * 1024, maxDuration = 600, onChange = () => {}, onError = () => {};

  function formatBytes(bytes) { return bytes >= 1024 ** 3 ? `${(bytes / 1024 ** 3).toFixed(2)} GB` : `${(bytes / 1024 ** 2).toFixed(1)} MB`; }
  function select(file) {
    onError("");
    if (!file) return;
    const extension = file.name.split(".").pop().toLowerCase();
    if (!allowed.has(extension)) return onError(`Choose a ${[...allowed].join(", ").toUpperCase()} video.`);
    if (file.size > maxBytes) return onError(`This file exceeds the ${formatBytes(maxBytes)} upload limit.`);
    if (!file.type.startsWith("video/") && file.type !== "application/octet-stream") return onError("The selected file does not appear to be a video.");

    selectedFile = file;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    const preview = $("preview");
    preview.src = previewUrl;
    $("fileName").textContent = file.name;
    $("fileInfo").textContent = `${formatBytes(file.size)} · reading duration…`;
    preview.onloadedmetadata = () => {
      const seconds = preview.duration;
      $("fileInfo").textContent = `${formatBytes(file.size)} · ${Number.isFinite(seconds) ? `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}` : "duration unavailable"}`;
      if (Number.isFinite(seconds) && seconds > maxDuration) {
        selectedFile = null;
        onError(`This video is longer than the ${Math.ceil(maxDuration / 60)} minute limit.`);
      }
      onChange(selectedFile);
    };
    $("fileCard").hidden = false;
    onChange(selectedFile);
  }
  function clear() {
    selectedFile = null;
    input.value = "";
    $("fileCard").hidden = true;
    $("preview").removeAttribute("src");
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    onError("");
    onChange(null);
  }

  function init(callbacks) {
    onChange = callbacks.onChange || onChange;
    onError = callbacks.onError || onError;
    zone.addEventListener("click", (event) => { if (event.target.id !== "browseButton") input.click(); });
    zone.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); input.click(); } });
    $("browseButton").addEventListener("click", (event) => { event.stopPropagation(); input.click(); });
    input.addEventListener("change", () => select(input.files[0]));
    $("removeFile").addEventListener("click", clear);
    for (const name of ["dragenter", "dragover"]) zone.addEventListener(name, (event) => { event.preventDefault(); zone.classList.add("dragging"); });
    for (const name of ["dragleave", "drop"]) zone.addEventListener(name, (event) => { event.preventDefault(); zone.classList.remove("dragging"); });
    zone.addEventListener("drop", (event) => select(event.dataTransfer.files[0]));
  }
  function configure(config) {
    if (Number.isFinite(config.max_upload_mb)) maxBytes = config.max_upload_mb * 1024 * 1024;
    if (Number.isFinite(config.max_duration_seconds)) maxDuration = config.max_duration_seconds;
    if (Array.isArray(config.supported_formats)) {
      allowed.clear();
      config.supported_formats.forEach((format) => allowed.add(format.replace(/^\./, "").toLowerCase()));
    }
    zone.dataset.maxBytes = String(maxBytes);
    zone.dataset.maxDuration = String(maxDuration);
    $("fileLimits").innerHTML = `${[...allowed].map((format) => format.toUpperCase()).join(", ")} · up to <b>${config.max_upload_mb || 500} MB</b> · up to <b>${Math.ceil(maxDuration / 60)} min</b>`;
  }
  window.VideoUpload = { init, configure, getFile: () => selectedFile, clear };
})();
