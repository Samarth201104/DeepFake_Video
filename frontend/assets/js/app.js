(() => {
  const $ = (id) => document.getElementById(id);
  const apiBase = window.DEEPFAKE_API_BASE || "";
  const analyzeButton = $("analyzeButton");

  window.VideoUpload.init({
    onChange(file) {
      analyzeButton.disabled = !file;
      window.ResultView.clear();
    },
    onError(message) { $("inlineError").textContent = message; $("inlineError").hidden = !message; },
  });

  analyzeButton.addEventListener("click", async () => {
    const file = window.VideoUpload.getFile();
    if (!file) return;
    $("inlineError").hidden = true;
    analyzeButton.disabled = true;
    $("progressState").hidden = false;
    const body = new FormData();
    body.append("file", file);
    try {
      const response = await fetch(`${apiBase}/api/detect`, { method: "POST", body });
      const result = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(result.error || `Analysis failed (${response.status}).`);
      if (!["Real", "Fake"].includes(result.prediction) || !Number.isFinite(Number(result.confidence))) {
        throw new Error("The API returned an incomplete prediction.");
      }
      window.ResultView.render(result);
    } catch (error) {
      const message = error.message || "Analysis failed.";
      $("inlineError").textContent = error instanceof TypeError
        ? `Could not connect to the analysis API. Check that the backend is running, then retry. (${message})`
        : message;
      $("inlineError").hidden = false;
    } finally {
      $("progressState").hidden = true;
      analyzeButton.disabled = !window.VideoUpload.getFile();
    }
  });

  $("resetButton").addEventListener("click", () => {
    window.VideoUpload.clear();
    window.ResultView.clear();
    $("dropZone").scrollIntoView({ behavior: "smooth", block: "center" });
  });

  fetch(`${apiBase}/api/health`).then(async (response) => {
    if (!response.ok) throw new Error("API unavailable");
    const config = await response.json();
    $("apiStatus").classList.add("online");
    $("apiStatus").querySelector("b").textContent = config.model_configured ? "API ready" : "API online · weights missing";
    window.VideoUpload.configure(config);
  }).catch(() => {
    $("apiStatus").classList.add("offline");
    $("apiStatus").querySelector("b").textContent = "API offline";
  });
})();
