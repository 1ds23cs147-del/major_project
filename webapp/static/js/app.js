/* ============================================================
   SentinelFire — frontend logic
   ============================================================ */

"use strict";

const MODALITIES = ["rgb", "thermal", "nir"];

const state = {
  uploads: {},          // modality -> { file, previewUrl }
  captureId: null,
  decision: null,       // last fused decision
  alerts: [],
  snapshot: null,       // { id, url, score }
};

const $ = (id) => document.getElementById(id);

/* ---------------- Utilities ---------------- */

function formatTime(ts) {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function addAlert(type, title, sub) {
  state.alerts.unshift({ type, title, sub, time: Date.now() / 1000 });
  if (state.alerts.length > 20) state.alerts.pop();
  renderAlerts();
}

function renderAlerts() {
  const list = $("alertsList");
  const empty = $("alertsEmpty");
  if (state.alerts.length === 0) {
    list.innerHTML = "";
    empty.hidden = false;
    return;
  }
  empty.hidden = true;
  list.innerHTML = state.alerts
    .map(
      (a) => `
      <div class="alert-item ${a.type}">
        <div class="alert-item-icon">${a.type === "fire" ? "🔥" : "ℹ️"}</div>
        <div class="alert-item-body">
          <div class="alert-item-title">${a.title}</div>
          <div class="alert-item-sub">${a.sub}</div>
        </div>
        <div class="alert-item-time">${formatTime(a.time)}</div>
      </div>`
    )
    .join("");
}

/* ---------------- Live monitor ---------------- */

async function pollLiveStatus() {
  try {
    const res = await fetch("/api/live/status");
    const data = await res.json();

    $("cameraStatus").textContent = data.available ? "Online" : "Not configured";
    $("cameraStatus").style.color = data.available ? "var(--success)" : "var(--danger)";

    if (data.available) {
      $("fireAlertStatus").textContent = data.fire_detected ? "⚠ FIRE" : "Clear";
      $("fireAlertStatus").style.color = data.fire_detected ? "var(--danger)" : "var(--success)";
      $("liveFireScore").textContent = data.fire_score.toFixed(2);
      $("liveDetections").textContent = data.detections.length;
      $("lastUpdate").textContent = formatTime(Date.now() / 1000);

      const overlay = $("overlayStatus");
      const banner = $("liveAlertBanner");
      if (data.fire_detected) {
        overlay.textContent = "⚠ FIRE ALERT";
        overlay.classList.add("alert");
        banner.hidden = false;
      } else {
        overlay.textContent = "Monitoring";
        overlay.classList.remove("alert");
        banner.hidden = true;
      }

      // Snapshot confirmation flow
      if (data.alert_active && data.snapshot_url) {
        if (!state.snapshot || state.snapshot.id !== data.snapshot_id) {
          state.snapshot = {
            id: data.snapshot_id,
            url: data.snapshot_url,
            score: data.fire_score,
          };
          $("snapshotImg").src = data.snapshot_url;
          $("snapshotScore").textContent = `Fire score: ${data.fire_score.toFixed(2)}`;
          $("snapshotPanel").hidden = false;
        }
      } else if (!data.alert_active) {
        state.snapshot = null;
        $("snapshotPanel").hidden = true;
      }
    }
  } catch (err) {
    $("cameraStatus").textContent = "Unavailable";
    $("cameraStatus").style.color = "var(--danger)";
  }
}

/* ---------------- Camera source ---------------- */

async function applyCameraSource() {
  const input = $("cameraSourceInput");
  const source = input.value.trim();
  if (!source) return;
  try {
    const res = await fetch("/api/camera/source", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Failed to switch camera");
    addAlert("info", "Camera switched", `Source: ${source}`);
    // Force the feed to reload with the new source.
    $("liveFeed").src = "/api/live/feed?t=" + Date.now();
  } catch (err) {
    addAlert("info", "Camera error", err.message);
  }
}

/* ---------------- Upload handling ---------------- */

function setupDropzone(modality) {
  const dropzone = $(`dropzone-${modality}`);
  const input = $(`file-${modality}`);

  dropzone.addEventListener("click", () => input.click());
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length) handleFile(modality, e.dataTransfer.files[0]);
  });
  input.addEventListener("change", () => {
    if (input.files.length) handleFile(modality, input.files[0]);
  });
}

function handleFile(modality, file) {
  if (!file.type.startsWith("image/")) {
    addAlert("info", "Invalid file", `${file.name} is not an image.`);
    return;
  }
  const previewUrl = URL.createObjectURL(file);
  state.uploads[modality] = { file, previewUrl };

  const preview = $(`preview-${modality}`);
  const img = $(`previewImg-${modality}`);
  img.src = previewUrl;
  preview.hidden = false;
  $(`${modality}Status`).textContent = "Ready";
  $(`${modality}Status`).classList.add("ready");
  document.querySelector(`.upload-card[data-modality="${modality}"]`).classList.add("has-image");

  updateAnalyzeBar();
}

function updateAnalyzeBar() {
  const count = Object.keys(state.uploads).length;
  $("uploadCount").textContent = `${count} / 3`;
  $("analyzeBtn").disabled = count < 2;
}

/* ---------------- Analysis & SOS ---------------- */

async function analyzeScene() {
  const btn = $("analyzeBtn");
  btn.disabled = true;
  btn.innerHTML = '<span class="btn-icon">⏳</span> Analyzing…';

  const form = new FormData();
  state.captureId = `cap_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  form.append("capture_id", state.captureId);

  for (const m of MODALITIES) {
    if (state.uploads[m]) form.append(m, state.uploads[m].file);
  }

  try {
    const res = await fetch("/api/upload", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Analysis failed");

    state.decision = data.decision;
    renderDecision(data);
    renderModalityResults(data.modalities);
  } catch (err) {
    addAlert("info", "Analysis error", err.message);
  } finally {
    btn.disabled = Object.keys(state.uploads).length < 2;
    btn.innerHTML = '<span class="btn-icon">🔬</span> Analyze Scene';
  }
}

function renderModalityResults(modalities) {
  for (const m of MODALITIES) {
    const r = modalities[m];
    const el = $(`result-${m}`);
    if (!r) {
      el.textContent = "Not uploaded";
      el.className = "preview-result";
      continue;
    }
    if (r.fire_detected) {
      el.textContent = `🔥 Fire detected (${(r.fire_score * 100).toFixed(1)}%)`;
      el.className = "preview-result fire";
    } else {
      el.textContent = `✓ No fire (${(r.fire_score * 100).toFixed(1)}%)`;
      el.className = "preview-result clear";
    }
  }
}

function renderDecision(data) {
  const d = data.decision;
  const panel = $("decisionPanel");
  panel.hidden = false;

  const isAlert = d.status === "confirmed_fire";
  panel.classList.toggle("alert", isAlert);

  $("decisionIcon").textContent = isAlert ? "🚨" : "✅";
  $("decisionTitle").textContent = isAlert ? "FIRE CONFIRMED" : "No Fire Confirmed";
  $("decisionSub").textContent = d.message;
  $("fusedScore").textContent = d.fused_score.toFixed(2);
  $("modalitiesChecked").textContent = d.modalities_checked.join(", ").toUpperCase();
  $("modalitiesConfirmed").textContent = d.modalities_confirmed.length;

  $("decisionDetail").textContent =
    `SOS allowed: ${d.sos_allowed ? "YES" : "NO"} (requires ≥ 2 confirming modalities)`;

  $("sosBtn").disabled = !d.sos_allowed;

  if (isAlert) {
    addAlert("fire", "FIRE CONFIRMED", `Fused score ${d.fused_score.toFixed(2)} · ${d.modalities_confirmed.length}/3 modalities`);
  }
}

async function sendSos() {
  if (!state.decision || !state.decision.sos_allowed) return;
  const btn = $("sosBtn");
  btn.disabled = true;
  btn.innerHTML = '<span class="btn-icon">⏳</span> Sending…';

  try {
    const res = await fetch("/api/sos", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        capture_id: state.captureId,
        confirmed: state.decision.sos_allowed,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "SOS failed");
    addAlert("fire", "SOS ALERT SENT", `Capture ${state.captureId} · ${formatTime(Date.now() / 1000)}`);
  } catch (err) {
    addAlert("info", "SOS error", err.message);
  } finally {
    btn.disabled = !state.decision.sos_allowed;
    btn.innerHTML = '<span class="btn-icon">🆘</span> Send SOS Alert';
  }
}

function resetScene() {
  for (const m of MODALITIES) {
    if (state.uploads[m]) URL.revokeObjectURL(state.uploads[m].previewUrl);
    delete state.uploads[m];
    $(`preview-${m}`).hidden = true;
    $(`${m}Status`).textContent = "Not uploaded";
    $(`${m}Status`).classList.remove("ready");
    document.querySelector(`.upload-card[data-modality="${m}"]`).classList.remove("has-image");
    $(`file-${m}`).value = "";
  }
  state.decision = null;
  $("decisionPanel").hidden = true;
  $("sosBtn").disabled = true;
  updateAnalyzeBar();
}

/* ---------------- Snapshot confirmation ---------------- */

async function confirmSnapshot() {
  if (!state.snapshot) return;
  const btn = $("confirmBtn");
  btn.disabled = true;
  btn.innerHTML = '<span class="btn-icon">⏳</span> Dispatching…';
  try {
    const res = await fetch("/api/live/confirm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ snapshot_id: state.snapshot.id }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Dispatch failed");
    addAlert("fire", "RESCUE TEAM DISPATCHED", `Snapshot ${state.snapshot.id} · ${formatTime(Date.now() / 1000)}`);
    $("snapshotPanel").hidden = true;
    state.snapshot = null;
  } catch (err) {
    addAlert("info", "Dispatch error", err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span class="btn-icon">🚨</span> Confirm — Dispatch Rescue Team';
  }
}

async function dismissSnapshot() {
  try {
    await fetch("/api/live/dismiss", { method: "POST" });
    $("snapshotPanel").hidden = true;
    state.snapshot = null;
    addAlert("info", "Alert dismissed", "Marked as false alarm.");
  } catch (err) {
    addAlert("info", "Dismiss error", err.message);
  }
}

/* ---------------- Init ---------------- */

function init() {
  MODALITIES.forEach(setupDropzone);
  $("analyzeBtn").addEventListener("click", analyzeScene);
  $("sosBtn").addEventListener("click", sendSos);
  $("resetBtn").addEventListener("click", resetScene);
  $("cameraSourceBtn").addEventListener("click", applyCameraSource);
  $("confirmBtn").addEventListener("click", confirmSnapshot);
  $("dismissBtn").addEventListener("click", dismissSnapshot);

  // Camera source presets
  document.querySelectorAll(".chip-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      $("cameraSourceInput").value = btn.dataset.source;
      applyCameraSource();
    });
  });

  // Scroll-spy for nav links
  const links = document.querySelectorAll(".nav-link");
  const sections = ["live", "upload", "alerts"];
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          links.forEach((l) => l.classList.toggle("active", l.getAttribute("href") === `#${entry.target.id}`));
        }
      });
    },
    { rootMargin: "-40% 0px -55% 0px" }
  );
  sections.forEach((id) => observer.observe(document.getElementById(id)));

  pollLiveStatus();
  setInterval(pollLiveStatus, 2000);
}

document.addEventListener("DOMContentLoaded", init);