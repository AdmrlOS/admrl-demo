"use strict";

(() => {
  const byId = (id) => document.getElementById(id);
  const ui = Object.fromEntries([
    "connection", "connection-label", "backend", "notice", "notice-text", "reconnect",
    "fps", "latency", "plate-count", "accepted-count", "frame", "result-time", "mode",
    "feed-title", "preview", "feed-overlay", "overlay-title", "overlay-detail", "source",
    "reads", "reads-badge", "history", "history-empty", "history-count", "footer-status",
  ].map((id) => [id, byId(id)]));
  const history = new Map();
  let latest = null;
  let lastSequence = null;
  let lastFrameAt = Date.now();
  let lastSuccessfulPoll = 0;
  let failedPolls = 0;
  let feedFailed = false;
  let nextFeedRetryAt = 0;

  const number = (value) => typeof value === "number" && Number.isFinite(value) ? value : null;
  const confidence = (value) => {
    const score = number(value);
    return score === null || score < 0 || score > 1 ? null : score;
  };
  const percent = (value) => {
    const score = confidence(value);
    return score === null ? "Unavailable" : `${(score * 100).toFixed(1)}%`;
  };
  function node(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  }
  function readStatus(accepted) {
    return node("span", `read-status${accepted ? "" : " review"}`, accepted ? "ACCEPTED" : "REVIEW");
  }
  function scoreRow(label, score, kind) {
    const fragment = document.createDocumentFragment();
    const row = node("div", "score-row");
    row.append(node("span", "", label), node("span", "score-value", percent(score)));
    const track = node("div", "score-track");
    const fill = node("div", `score-fill ${kind}`);
    fill.style.width = `${(confidence(score) ?? 0) * 100}%`;
    track.append(fill);
    fragment.append(row, track);
    return fragment;
  }
  function renderReads(plates, state) {
    ui.reads.replaceChildren();
    ui["reads-badge"].textContent = String(plates.length);
    if (!plates.length) {
      const empty = node("div", "empty-state");
      empty.append(node("span", "empty-mark", "▱"));
      empty.append(node("strong", "", state === "loading" ? "Waiting for plate reads" : "No plates in this frame"));
      empty.append(node("p", "", state === "loading" ? "Detected plates and confidence scores appear here." : "The pipeline will show a result when it detects a plate."));
      ui.reads.append(empty);
      return;
    }
    for (const plate of plates) {
      const card = node("article", "plate-card");
      const top = node("div", "plate-top");
      const text = typeof plate.plate === "string" ? plate.plate : typeof plate.text === "string" ? plate.text : "";
      top.append(node("strong", "plate-string", text || "Unreadable"), readStatus(plate.accepted === true));
      card.append(top, scoreRow("OCR confidence", plate.recognition_confidence, "ocr"), scoreRow("Detection confidence", plate.detection_confidence, "detection"));
      ui.reads.append(card);
    }
  }
  function remember(plates) {
    const now = Date.now();
    for (const plate of plates) {
      const text = typeof plate.plate === "string" ? plate.plate : typeof plate.text === "string" ? plate.text : "";
      if (!text.trim()) continue;
      history.delete(text);
      history.set(text, {text, score: confidence(plate.recognition_confidence), accepted: plate.accepted === true, seenAt: now});
    }
    while (history.size > 20) history.delete(history.keys().next().value);
  }
  function renderHistory() {
    ui.history.replaceChildren();
    ui["history-count"].textContent = String(history.size);
    ui["history-empty"].hidden = history.size > 0;
    for (const item of [...history.values()].reverse()) {
      const row = node("tr", "");
      const status = node("td", "");
      status.append(readStatus(item.accepted));
      const seconds = Math.max(0, Math.floor((Date.now() - item.seenAt) / 1000));
      const ago = seconds < 5 ? "Just now" : seconds < 60 ? `${seconds}s ago` : seconds < 3600 ? `${Math.floor(seconds / 60)}m ago` : `${Math.floor(seconds / 3600)}h ago`;
      row.append(node("td", "", item.text), node("td", "", percent(item.score)), status, node("td", "", ago));
      ui.history.append(row);
    }
  }
  function notice(text, style) {
    ui.notice.hidden = !text;
    ui.notice.className = `notice${style ? ` ${style}` : ""}`;
    ui["notice-text"].textContent = text;
  }
  function overlay(title, detail) {
    ui["feed-overlay"].hidden = !title;
    ui["overlay-title"].textContent = title;
    ui["overlay-detail"].textContent = detail;
  }
  function render(data, newFrame) {
    const state = typeof data.status === "string" ? data.status : "loading";
    const mode = typeof data.mode === "string" ? data.mode : "unknown";
    const plates = Array.isArray(data.plates) ? data.plates.filter((plate) => plate && typeof plate === "object") : [];
    const staticImage = mode === "sample" || mode === "image";
    const networkStream = mode === "video" && typeof data.source === "string" && /^(?:rtsp|rtsps|http|https):\/\//i.test(data.source);
    const disconnected = failedPolls >= 2 || Date.now() - lastSuccessfulPoll > 3500;
    const stale = !staticImage && state === "live" && Date.now() - lastFrameAt > 10000;
    const hasFrame = number(data.frame) !== null;
    const sourceNames = {camera: "Webcam", video: networkStream ? "Live stream" : "Video playback", image: "Image preview", sample: "Sample preview"};
    ui["feed-title"].textContent = sourceNames[mode] || "Camera preview";
    ui.source.textContent = typeof data.source === "string" && data.source ? data.source : "Waiting for pipeline";
    ui.backend.textContent = data.backend === "rknn" ? "RK3588 · NPU / RKNN" : data.backend === "onnx" ? "CPU / ONNX" : "Runtime initializing";
    ui.connection.className = `connection ${disconnected ? "disconnected" : "connected"}`;
    ui["connection-label"].textContent = disconnected ? "Disconnected" : "Device connected";

    let badge = "Loading models";
    let badgeStyle = "loading";
    if (state === "error") { badge = "Pipeline error"; badgeStyle = "error"; }
    else if (disconnected) { badge = "Disconnected"; badgeStyle = "disconnected"; }
    else if (stale) { badge = "Waiting for frames"; badgeStyle = "stale"; }
    else if (state === "stopped") { badge = "Stopped"; badgeStyle = "stopped"; }
    else if (mode === "sample" && hasFrame) { badge = "Sample image"; badgeStyle = "sample"; }
    else if (mode === "image" && hasFrame) { badge = "Image result"; badgeStyle = "sample"; }
    else if (state === "live") { badge = mode === "camera" ? "Live webcam" : networkStream ? "Live stream" : "Video playback"; badgeStyle = "live"; }
    ui.mode.textContent = badge;
    ui.mode.className = `state-badge ${badgeStyle}`;

    if (state === "error") {
      const error = typeof data.error === "string" && data.error ? data.error : "The recognition pipeline reported an error.";
      notice(`${error}${disconnected ? " The device is now disconnected." : ""}`, "error");
      overlay("Recognition stopped", "Check the device error above. Any existing reads are from the last processed frame.");
    } else if (disconnected) {
      notice("Connection to the device was interrupted. The last result is retained; it is no longer live.", "error");
      overlay("Device disconnected", "Reconnecting automatically. Results below are from the last successful frame.");
    } else if (state === "loading" || !hasFrame) {
      notice("The device is starting the recognition pipeline. The first frame will appear when it is ready.", "");
      overlay("Starting the pipeline", "Loading models and waiting for the first processed frame.");
    } else if (stale) {
      notice("No new processed frame for over 10 seconds. The preview and reads may be stale.", "warning");
      overlay("Waiting for new frames", "Showing the last processed frame when the connection resumes.");
    } else if (feedFailed) {
      notice("Results are connected, but the image preview was interrupted. Reconnecting the preview…", "warning");
      overlay("Preview interrupted", "Plate results continue updating while the image connection reconnects.");
    } else {
      overlay("", "");
      if (mode === "sample") notice("Bundled sample image · this is a still-image demonstration, with no live webcam feed.", "warning");
      else if (mode === "image") notice("Still-image result · the image and plate reads remain on screen for inspection.", "");
      else if (state === "stopped") notice("The source has stopped. The preview and reads show its last processed frame.", "warning");
      else notice("", "");
    }

    const fps = number(data.fps);
    const latency = number(data.elapsed_ms);
    ui.fps.textContent = staticImage || disconnected || stale || state === "stopped" || state === "error" ? "—" : fps !== null && fps >= 0 ? fps.toFixed(1) : "—";
    ui.latency.textContent = latency !== null && latency >= 0 ? latency.toFixed(1) : "—";
    ui["plate-count"].textContent = hasFrame ? String(plates.length) : "—";
    ui["accepted-count"].textContent = hasFrame ? `${plates.filter((plate) => plate.accepted === true).length} accepted · ${plates.length} detected` : "Waiting for a result";
    ui.frame.textContent = hasFrame ? String(data.frame) : "—";
    const timestamp = typeof data.timestamp === "string" ? new Date(data.timestamp) : null;
    ui["result-time"].textContent = timestamp && !Number.isNaN(timestamp.getTime()) ? `Last result ${timestamp.toLocaleTimeString()}` : "No result yet";
    ui["footer-status"].textContent = `${badge}${disconnected || stale ? " · last result retained" : ""}`;
    if (newFrame) {
      renderReads(plates, state);
      if (hasFrame && (state === "live" || state === "sample")) remember(plates);
    }
    renderHistory();
  }
  function reconnectFeed() {
    feedFailed = false;
    nextFeedRetryAt = Date.now() + 3000;
    ui.preview.src = `/stream.mjpg?reconnect=${Date.now()}`;
    if (latest) render(latest, false);
  }
  ui.preview.addEventListener("error", () => {
    feedFailed = true;
    nextFeedRetryAt = Date.now() + 3000;
    if (latest) render(latest, false);
  });
  ui.preview.addEventListener("load", () => {
    feedFailed = false;
    if (latest) render(latest, false);
  });
  ui.reconnect.addEventListener("click", reconnectFeed);

  async function poll() {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 2500);
    try {
      const response = await fetch("/results", {cache: "no-store", signal: controller.signal});
      if (!response.ok) throw new Error("Device result request failed");
      const data = await response.json();
      if (!data || typeof data !== "object" || !Array.isArray(data.plates)) throw new Error("Unexpected device result");
      failedPolls = 0;
      lastSuccessfulPoll = Date.now();
      const sequence = number(data.sequence);
      const newFrame = latest === null || (sequence !== null && sequence !== lastSequence);
      if (newFrame) lastFrameAt = Date.now();
      if (sequence !== null) lastSequence = sequence;
      latest = data;
      render(data, newFrame);
      if (feedFailed && Date.now() >= nextFeedRetryAt) reconnectFeed();
    } catch (error) {
      failedPolls += 1;
      if (latest) render(latest, false);
      else {
        ui.connection.className = "connection disconnected";
        ui["connection-label"].textContent = "Disconnected";
        ui.mode.textContent = "Disconnected";
        ui.mode.className = "state-badge disconnected";
        notice("Waiting for the device to respond. Reconnecting automatically…", "error");
        overlay("Device unavailable", "Check that the ANPR container is running and its web port is reachable.");
      }
    } finally {
      clearTimeout(timeout);
      setTimeout(poll, 500);
    }
  }
  poll();
})();
