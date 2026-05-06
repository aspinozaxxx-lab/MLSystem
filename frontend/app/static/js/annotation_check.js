const form = document.getElementById("annotation-form");
const progressPanel = document.getElementById("progress-panel");
const resultPanel = document.getElementById("result-panel");
const runIdEl = document.getElementById("run-id");
const resultSummary = document.getElementById("result-summary");
const stageResults = document.getElementById("stage-results");
const sceneTable = document.getElementById("scene-table");
const artifactList = document.getElementById("artifact-list");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  progressPanel.hidden = false;
  resultPanel.hidden = true;
  setStep("upload", "active");
  const data = new FormData(form);
  if (!data.has("allow_inferred_annotation_crs")) data.set("allow_inferred_annotation_crs", "false");
  const response = await fetch("/api/annotation-check", { method: "POST", body: data });
  const payload = await response.json();
  if (!response.ok) {
    showError(payload.error || "Ошибка запуска проверки");
    return;
  }
  runIdEl.textContent = `run_id: ${payload.run_id}`;
  setStep("upload", "done");
  poll(payload.run_id);
});

async function poll(runId) {
  const response = await fetch(`/api/annotation-check/${runId}`);
  const payload = await response.json();
  render(payload);
  if (payload.status === "running" || payload.status === "queued") {
    setTimeout(() => poll(runId), 2500);
  }
}

function render(payload) {
  setStep("inventory_scenes", stageClass(payload, "inventory_scenes"));
  setStep("prepare_dataset", stageClass(payload, "prepare_dataset"));
  if (payload.status === "succeeded") setStep("report", "done");
  if (payload.status === "failed") setStep("report", "failed");
  resultPanel.hidden = false;
  const s = payload.summary || {};
  resultSummary.innerHTML = [
    ["Статус", payload.status],
    ["Сцен в списке", s.total_scenes_requested],
    ["Найдено сцен", s.matched_scenes],
    ["Отсутствует", s.missing_scenes],
    ["Объектов", s.total_objects],
    ["Train сцен", s.train_scenes],
    ["Val сцен", s.val_scenes],
    ["Split", s.split_strategy],
  ].map(([k, v]) => `<div class="summary-item"><span>${escapeHtml(k)}</span><b>${escapeHtml(v ?? "—")}</b></div>`).join("");
  stageResults.innerHTML = (payload.stages || []).map(stage => `
    <div class="stage-box ${escapeHtml(stage.status)}">
      <h3>${escapeHtml(stage.name)}: ${escapeHtml(stage.status)}</h3>
      <p>${escapeHtml(stage.summary || "")}</p>
      ${listBlock("Warnings", stage.warnings, "warn")}
      ${listBlock("Errors", stage.errors, "error")}
    </div>
  `).join("");
  sceneTable.innerHTML = (payload.scene_rows || []).map(row => `
    <tr>
      <td>${escapeHtml(row.scene)}</td>
      <td>${escapeHtml(row.objects ?? "—")}</td>
      <td>${escapeHtml(row.storage_status)}</td>
      <td>${escapeHtml(row.split)}</td>
    </tr>
  `).join("");
  artifactList.innerHTML = (payload.artifacts || []).map(item => `<li><code>${escapeHtml(item)}</code></li>`).join("");
}

function stageClass(payload, stageName) {
  const stage = (payload.stages || []).find(item => item.name === stageName);
  if (!stage || stage.status === "pending") return "";
  if (stage.status === "failed") return "failed";
  if (stage.status === "success" || stage.status === "success_with_warning") return "done";
  return "active";
}

function setStep(name, cls) {
  const item = document.querySelector(`[data-step="${name}"]`);
  if (!item) return;
  item.className = cls || "";
}

function showError(message) {
  resultPanel.hidden = false;
  resultSummary.innerHTML = `<div class="error">${escapeHtml(message)}</div>`;
}

function listBlock(title, rows, cls) {
  if (!rows || rows.length === 0) return "";
  return `<h4>${title}</h4><ul class="${cls}">${rows.map(row => `<li>${escapeHtml(row)}</li>`).join("")}</ul>`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}

