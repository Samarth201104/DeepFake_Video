const API_BASE = window.DEEPFAKE_API_BASE || "";
const $ = (id) => document.getElementById(id);
const input = $("fileInput"), zone = $("dropZone"), analyze = $("analyzeButton");
let selectedFile = null, previewUrl = null;

function formatBytes(bytes) { return bytes >= 1024 ** 3 ? `${(bytes / 1024 ** 3).toFixed(2)} GB` : `${(bytes / 1024 ** 2).toFixed(1)} MB`; }
function showError(message) { $("inlineError").textContent = message; $("inlineError").hidden = !message; }
function clearResult() { $("resultContent").hidden = true; $("emptyResult").hidden = false; $("resultSubtitle").textContent = "Your results will appear here."; }
function chooseFile(file) {
  showError("");
  if (!file) return;
  const supported = /\.(mp4|mov|avi|mkv|webm)$/i.test(file.name);
  if (!supported) return showError("Choose an MP4, MOV, AVI, MKV or WebM video.");
  const maxBytes = Number(zone.dataset.maxBytes || 500 * 1024 * 1024);
  if (file.size > maxBytes) return showError(`This file is larger than the ${formatBytes(maxBytes)} upload limit.`);
  if (!file.type.startsWith("video/") && file.type !== "application/octet-stream") return showError("The selected file does not appear to be a video.");
  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  $("preview").src = previewUrl;
  $("fileName").textContent = file.name;
  $("fileInfo").textContent = `${formatBytes(file.size)} · reading duration…`;
  $("preview").onloadedmetadata = () => {
    const seconds = $("preview").duration;
    $("fileInfo").textContent = `${formatBytes(file.size)} · ${Number.isFinite(seconds) ? `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}` : "duration unavailable"}`;
    const maxDuration = Number(zone.dataset.maxDuration || 600);
    if (Number.isFinite(seconds) && seconds > maxDuration) {
      showError(`This video is longer than the ${Math.ceil(maxDuration / 60)} minute limit.`);
      analyze.disabled = true;
    }
  };
  $("fileCard").hidden = false; analyze.disabled = false; clearResult();
}
function clearFile() {
  selectedFile = null; input.value = ""; analyze.disabled = true; $("fileCard").hidden = true;
  $("preview").removeAttribute("src"); if (previewUrl) URL.revokeObjectURL(previewUrl); previewUrl = null; showError("");
}
zone.addEventListener("click", (event) => { if (event.target.id !== "browseButton") input.click(); });
zone.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); input.click(); } });
$("browseButton").addEventListener("click", (event) => { event.stopPropagation(); input.click(); });
input.addEventListener("change", () => chooseFile(input.files[0]));
$("removeFile").addEventListener("click", clearFile);
$("resetButton").addEventListener("click", () => { clearFile(); clearResult(); zone.scrollIntoView({behavior:"smooth",block:"center"}); });
for (const eventName of ["dragenter", "dragover"]) zone.addEventListener(eventName, (event) => { event.preventDefault(); zone.classList.add("dragging"); });
for (const eventName of ["dragleave", "drop"]) zone.addEventListener(eventName, (event) => { event.preventDefault(); zone.classList.remove("dragging"); });
zone.addEventListener("drop", (event) => chooseFile(event.dataTransfer.files[0]));

function renderResult(data) {
  $("emptyResult").hidden = true; $("resultContent").hidden = false;
  const fake = data.prediction === "Fake";
  $("predictionLabel").textContent = data.prediction;
  $("predictionLabel").style.color = fake ? "var(--danger)" : "var(--accent)";
  $("confidenceValue").textContent = `${Number(data.confidence).toFixed(1)}%`;
  $("scoreRing").style.setProperty("--score", `${Number(data.confidence) * 3.6}deg`);
  $("scoreFill").style.width = `${Math.max(0, Math.min(100, Number(data.fake_score)))}%`;
  $("scoreFill").style.background = fake ? "var(--danger)" : "var(--accent)";
  $("resultExplanation").textContent = fake ? "The model's average face score crossed its Fake classification threshold." : "The model's average face score fell below its Fake classification threshold.";
  $("resultSubtitle").textContent = data.filename || "Analysis complete.";
  $("durationValue").textContent = data.duration_seconds == null ? "Unavailable" : `${data.duration_seconds.toFixed(1)} s`;
  $("framesValue").textContent = data.processed_frames ?? "—"; $("facesValue").textContent = data.faces_analyzed ?? "—";
  $("timeValue").textContent = data.processing_seconds == null ? "—" : `${data.processing_seconds.toFixed(2)} s`;
}
analyze.addEventListener("click", async () => {
  if (!selectedFile) return;
  showError(""); analyze.disabled = true; $("progressState").hidden = false; $("progressState").querySelector("small").textContent = "Sampling frames and running model inference…";
  const body = new FormData(); body.append("file", selectedFile);
  try {
    const response = await fetch(`${API_BASE}/api/detect`, {method:"POST", body});
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `Analysis failed (${response.status}).`);
    if (!data.prediction || !["Real", "Fake"].includes(data.prediction) || !Number.isFinite(Number(data.confidence))) throw new Error("The API returned an incomplete prediction.");
    renderResult(data);
  } catch (error) {
    showError(`${error.message || "Could not reach the analysis API."} Check that the backend is running, then retry.`);
  } finally { $("progressState").hidden = true; analyze.disabled = !selectedFile; }
});

$("themeToggle").addEventListener("click", () => {
  document.documentElement.classList.toggle("light");
  const light = document.documentElement.classList.contains("light");
  try { localStorage.setItem("framecheck-theme", light ? "light" : "dark"); } catch {}
  $("themeToggle").setAttribute("aria-label", `Switch to ${light ? "dark" : "light"} theme`);
});
try { if (localStorage.getItem("framecheck-theme") === "light") document.documentElement.classList.add("light"); } catch {}
$("menuToggle").addEventListener("click", () => { const open = $("navigation").classList.toggle("open"); $("menuToggle").setAttribute("aria-expanded", String(open)); });
$("navigation").addEventListener("click", (event) => { if (event.target.closest("a")) { $("navigation").classList.remove("open"); $("menuToggle").setAttribute("aria-expanded", "false"); } });

fetch(`${API_BASE}/api/health`).then(async (response) => {
  if (!response.ok) throw new Error("API unavailable");
  const config = await response.json();
  $("apiStatus").classList.add("online"); $("apiStatus").querySelector("b").textContent = config.model_configured ? "API ready" : "API online · weights missing";
  if (Number.isFinite(config.max_upload_mb)) { zone.dataset.maxBytes = String(config.max_upload_mb * 1024 * 1024); $("sizeLimit").textContent = `${config.max_upload_mb} MB`; }
  if (Number.isFinite(config.max_duration_seconds)) { zone.dataset.maxDuration = String(config.max_duration_seconds); $("durationLimit").textContent = `${Math.ceil(config.max_duration_seconds / 60)} min`; }
  if (Array.isArray(config.supported_formats)) $("fileLimits").innerHTML = `${config.supported_formats.map((x) => x.replace(".", "").toUpperCase()).join(", ")} · up to <b id="sizeLimit">${config.max_upload_mb} MB</b> · up to <b id="durationLimit">${Math.ceil(config.max_duration_seconds / 60)} min</b>`;
}).catch(() => { $("apiStatus").classList.add("offline"); $("apiStatus").querySelector("b").textContent = "API offline"; });
