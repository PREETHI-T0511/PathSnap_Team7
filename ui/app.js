(() => {
  const $ = (id) => document.getElementById(id);
  const apiInput = $("apiBase");
  const connectBtn = $("connectBtn");
  const tripSelect = $("tripSelect");
  const predictBtn = $("predictBtn");
  const status = $("status");
  const connection = $("connection");
  const results = $("results");
  const SVG_NS = "http://www.w3.org/2000/svg";
  let apiBase = localStorage.getItem("pathsnapApiBase") || apiInput.value;
  apiInput.value = apiBase;

  function setStatus(message, type = "") {
    status.textContent = message;
    status.className = `status ${type}`.trim();
  }
  function setConnection(message, ok) {
    connection.textContent = message;
    connection.className = `connection ${ok ? "ok" : "bad"}`;
  }
  async function api(path, options = {}) {
    const response = await fetch(`${apiBase.replace(/\/$/, "")}${path}`, {
      ...options,
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || `API error (${response.status})`);
    return data;
  }
  async function connect() {
    apiBase = apiInput.value.trim().replace(/\/$/, "");
    localStorage.setItem("pathsnapApiBase", apiBase);
    connectBtn.disabled = true;
    setStatus("Connecting to the model laptop…");
    try {
      const health = await api("/health");
      if (!health.model_loaded) throw new Error(`API is up, but model is not loaded: ${health.model_error || "check model path"}`);
      const data = await api("/api/trips");
      tripSelect.replaceChildren();
      if (!data.trip_ids || data.trip_ids.length === 0) {
        tripSelect.add(new Option("No evaluator trips available", ""));
        tripSelect.disabled = true;
        predictBtn.disabled = true;
        setStatus("API connected. No evaluator trips are configured on the server yet.", "error");
      } else {
        tripSelect.add(new Option("Select a trip…", ""));
        data.trip_ids.forEach((id) => tripSelect.add(new Option(id, id)));
        tripSelect.disabled = false;
        predictBtn.disabled = false;
        setStatus(`Connected. ${data.trip_ids.length} evaluator trips available.`, "success");
      }
      setConnection("API connected", true);
    } catch (error) {
      setConnection("API unavailable", false);
      setStatus(error.message, "error");
    } finally {
      connectBtn.disabled = false;
    }
  }

  function svgElement(name, attrs = {}) {
    const element = document.createElementNS(SVG_NS, name);
    Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, String(value)));
    return element;
  }
  function renderMap(gps, segments) {
    const svg = $("routeMap");
    svg.replaceChildren();
    const routeCoords = (segments || []).flatMap((segment) => segment || []);
    const all = [...(gps || []), ...routeCoords].filter((p) => Array.isArray(p) && p.length >= 2 && p.every(Number.isFinite));
    if (!all.length) {
      const text = svgElement("text", { x: 450, y: 230, "text-anchor": "middle", fill: "#607184" });
      text.textContent = "No coordinates returned for this trip.";
      svg.appendChild(text);
      return;
    }
    const lats = all.map((p) => p[0]);
    const lons = all.map((p) => p[1]);
    let minLat = Math.min(...lats), maxLat = Math.max(...lats);
    let minLon = Math.min(...lons), maxLon = Math.max(...lons);
    if (maxLat === minLat) { maxLat += 0.001; minLat -= 0.001; }
    if (maxLon === minLon) { maxLon += 0.001; minLon -= 0.001; }
    const pad = 42, width = 900, height = 460;
    const project = ([lat, lon]) => [
      pad + ((lon - minLon) / (maxLon - minLon)) * (width - pad * 2),
      height - pad - ((lat - minLat) / (maxLat - minLat)) * (height - pad * 2),
    ];
    const gpsProjected = (gps || []).map(project);
    if (gpsProjected.length > 1) {
      svg.appendChild(svgElement("polyline", { points: gpsProjected.map((p) => p.join(",")).join(" "), class: "gps-path" }));
    }
    const routeProjected = routeCoords.map(project);
    if (routeProjected.length > 1) {
      svg.appendChild(svgElement("polyline", { points: routeProjected.map((p) => p.join(",")).join(" "), class: "predicted-path" }));
    }
    const stride = Math.max(1, Math.ceil(gpsProjected.length / 1200));
    gpsProjected.filter((_, i) => i % stride === 0).forEach((point) => {
      svg.appendChild(svgElement("circle", { cx: point[0], cy: point[1], r: 3.1, class: "gps-point" }));
    });
    if (gpsProjected.length) {
      svg.appendChild(svgElement("circle", { cx: gpsProjected[0][0], cy: gpsProjected[0][1], r: 6, class: "start-point" }));
      const end = gpsProjected[gpsProjected.length - 1];
      svg.appendChild(svgElement("circle", { cx: end[0], cy: end[1], r: 6, class: "end-point" }));
    }
  }
  function percent(value) {
    return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "Unavailable";
  }

  connectBtn.addEventListener("click", connect);
  tripSelect.addEventListener("change", () => { predictBtn.disabled = !tripSelect.value; });
  predictBtn.addEventListener("click", async () => {
    if (!tripSelect.value) { setStatus("Select an evaluator trip first.", "error"); return; }
    predictBtn.disabled = true;
    setStatus(`Reconstructing ${tripSelect.value}…`);
    try {
      const result = await api("/api/predict", {
        method: "POST",
        body: JSON.stringify({ trip_id: tripSelect.value }),
      });
      const metrics = result.metrics || {};
      $("accuracy").textContent = percent(metrics.top1_accuracy);
      $("scored").textContent = `${metrics.matched_observations || 0} observations scored`;
      $("missRate").textContent = percent(metrics.candidate_miss_rate);
      $("continuity").textContent = percent(metrics.route_continuity);
      $("inference").textContent = `${Number(result.inference_seconds || 0).toFixed(2)} s`;
      $("mapTitle").textContent = `Trip ${result.trip_id}`;
      $("predictionCount").textContent = `${result.prediction_count} road-segment predictions returned. ${metrics.status || ""}`;
      renderMap(result.gps_points || [], result.predicted_segments || []);
      results.classList.remove("hidden");
      setStatus("Route reconstruction complete.", "success");
    } catch (error) {
      setStatus(error.message, "error");
    } finally {
      predictBtn.disabled = !tripSelect.value;
    }
  });
  if (apiBase && apiBase !== "http://127.0.0.1:8000") connect();
})();
