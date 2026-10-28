(() => {
  const $ = (id) => document.getElementById(id);
  function clear() {
    $("resultContent").hidden = true;
    $("emptyResult").hidden = false;
    $("resultSubtitle").textContent = "Your results will appear here.";
  }
  function render(data) {
    $("emptyResult").hidden = true;
    $("resultContent").hidden = false;
    const fake = data.prediction === "Fake";
    $("predictionLabel").textContent = data.prediction;
    $("predictionLabel").style.color = fake ? "var(--danger)" : "var(--accent)";
    $("confidenceValue").textContent = `${Number(data.confidence).toFixed(1)}%`;
    $("scoreRing").style.setProperty("--score", `${Number(data.confidence) * 3.6}deg`);
    $("scoreFill").style.width = `${Math.max(0, Math.min(100, Number(data.fake_score)))}%`;
    $("scoreFill").style.background = fake ? "var(--danger)" : "var(--accent)";
    $("resultExplanation").textContent = fake
      ? "The model's average face score crossed its Fake classification threshold."
      : "The model's average face score fell below its Fake classification threshold.";
    $("resultSubtitle").textContent = data.filename || "Analysis complete.";
    $("durationValue").textContent = data.duration_seconds == null ? "Unavailable" : `${data.duration_seconds.toFixed(1)} s`;
    $("framesValue").textContent = data.processed_frames ?? "—";
    $("facesValue").textContent = data.faces_analyzed ?? "—";
    $("timeValue").textContent = data.processing_seconds == null ? "—" : `${data.processing_seconds.toFixed(2)} s`;
  }
  window.ResultView = { clear, render };
})();
