/* ===================================================================
   SmartCart frontend

   >>> API base URL — the ONLY place it is set. No trailing slash. <<<
   Local dev: "http://localhost:8000"
   =================================================================== */
const API_BASE = "https://smartcart-customer-segmentation-xfyh.onrender.com";
/* =================================================================== */

// Render's free tier sleeps after 15 min idle; the first request then takes
// ~50s to wake. Timeout must outlast that; a hint appears while it happens.
const COLD_START_TIMEOUT_MS = 75000;
const WAKE_NOTICE_AFTER_MS = 3500;
const VB_W = 400, VB_H = 356;      // svg viewBox
const CX = 200, CY = 168;          // diamond centre
const D = 104;                     // diamond half-extent for the segment anchors

// Anchor slots, ordered by spend rank: budget-conscious (left) -> premium (right),
// mid ranks split to top and bottom. Labels point outward so they never collide.
const SLOTS = [
  { x: CX - D, y: CY,     align: "end",    lx: -18, ly: 4 },
  { x: CX,     y: CY - D, align: "middle", lx: 0,   ly: -20 },
  { x: CX,     y: CY + D, align: "middle", lx: 0,   ly: 30 },
  { x: CX + D, y: CY,     align: "start",  lx: 18,  ly: 4 },
];

const form = document.getElementById("profile-form");
const card = document.getElementById("result-card");
const svg = document.getElementById("segment-map");
const runBtn = form.querySelector(".run-btn");
const notice = document.getElementById("notice");
const hint = document.getElementById("viz-hint");
const el = {
  kicker: document.getElementById("verdict-kicker"),
  name: document.getElementById("verdict-name"),
  desc: document.getElementById("verdict-desc"),
  drivers: document.getElementById("drivers"),
  driversList: document.getElementById("drivers-list"),
  spread: document.getElementById("spread"),
  details: document.getElementById("details"),
  detailsTable: document.getElementById("details-table"),
  provenance: document.getElementById("provenance"),
  apiUrl: document.getElementById("api-tag-url"),
};

const PROFILE_LABELS = {
  Income: "Income",
  Recency: "Days since last purchase",
  NumDealsPurchases: "Deal purchases",
  NumWebPurchases: "Web purchases",
  NumCatalogPurchases: "Catalog purchases",
  NumStorePurchases: "In-store purchases",
  NumWebVisitsMonth: "Site visits per month",
  Complain: "Complaint rate",
  Response: "Campaign acceptance",
  Age: "Age",
  Customer_Tenure_Days: "Days as a member",
  Total_Spending: "Total spending",
  Total_Children: "Children at home",
};

function formatProfileValue(key, v) {
  if (key === "Income" || key === "Total_Spending") return `$${Math.round(v).toLocaleString()}`;
  if (key === "Complain" || key === "Response") return `${Math.round(v * 100)}%`;
  if (key === "Age" || key === "Recency" || key === "Customer_Tenure_Days") return `${Math.round(v)}`;
  return v.toFixed(1);
}

document.getElementById("api-tag-url").textContent = API_BASE.replace(/^https?:\/\//, "");

let anchorPos = {};      // segment id -> {x, y}
let segmentMeta = {};    // segment id -> {name, description, profile}

/* ---------- number steppers ---------- */
form.querySelectorAll(".stepper").forEach((box) => {
  const input = box.querySelector("input");
  box.querySelectorAll("button").forEach((btn) => {
    btn.addEventListener("click", () => {
      const step = Number(btn.dataset.step);
      const next = (Number(input.value) || 0) + step;
      const min = input.min !== "" ? Number(input.min) : -Infinity;
      const max = input.max !== "" ? Number(input.max) : Infinity;
      input.value = Math.min(max, Math.max(min, next));
    });
  });
});

/* ---------- segment map: layout + render ----------
   Anchors sit left -> right along the spend ladder (budget-conscious to
   premium), staggered vertically so labels never collide. The customer
   marker is the confidence-weighted blend of the anchor positions, so its
   horizontal spot shows where on that ladder the model places them and its
   distance from an anchor shows how sure the call is. */
function shortLabel(name) {
  const trimmed = name
    .replace(/\b(households?|shoppers?|customers?|segment)\b/gi, "")
    .replace(/budget-conscious/gi, "")
    .trim();
  const base = trimmed || name;
  return base.charAt(0).toUpperCase() + base.slice(1);
}

function layoutAnchors(meta) {
  const ids = Object.keys(meta);
  if (ids.length === 0) {
    // placeholder diamond, shown while the backend is still waking
    anchorPos = Object.fromEntries(SLOTS.map((s, i) => [String(i), { ...s, label: "" }]));
    return;
  }
  const spendOf = (id) => meta[id].profile.feature_means.Total_Spending;
  anchorPos = {};
  ids.sort((a, b) => spendOf(a) - spendOf(b)).forEach((id, i) => {
    anchorPos[id] = { ...SLOTS[i % SLOTS.length], label: shortLabel(meta[id].name) };
  });
}

function svgEl(tag, attrs) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
}

function renderMap() {
  svg.innerHTML = "";

  const cap = svgEl("text", { class: "map-axis-label", x: CX, y: VB_H - 6, "text-anchor": "middle" });
  cap.textContent = "position shows the model's pull toward each segment";
  svg.appendChild(cap);

  // tether layer (populated on result)
  svg.appendChild(svgEl("g", { id: "tethers" }));

  // anchors
  Object.entries(anchorPos).forEach(([id, p]) => {
    const g = svgEl("g", { class: "anchor", "data-id": id, transform: `translate(${p.x} ${p.y})` });
    g.appendChild(svgEl("circle", { class: "anchor-ring", r: 13 }));
    g.appendChild(svgEl("circle", { class: "anchor-core", r: 3 }));

    const name = svgEl("text", { class: "anchor-name", x: p.lx, y: p.ly, "text-anchor": p.align });
    name.textContent = p.label;
    g.appendChild(name);

    const pct = svgEl("text", { class: "anchor-pct", x: p.lx, y: p.ly + 13, "text-anchor": p.align });
    g.appendChild(pct);

    svg.appendChild(g);
  });

  // customer marker (starts at centre)
  const marker = svgEl("g", { id: "marker", transform: `translate(${CX} ${CY})` });
  marker.appendChild(svgEl("circle", { class: "marker-halo", r: 16 }));
  marker.appendChild(svgEl("circle", { class: "marker-core", r: 6 }));
  svg.appendChild(marker);
}

function paintResult(res) {
  const byId = Object.fromEntries(res.confidence_by_segment.map((s) => [String(s.segment), s.confidence]));

  // marker position = confidence-weighted blend of the anchors. Weights are
  // sharpened (^1.6) so a decisive call visibly pulls toward its anchor while
  // a near-tie still floats near the centre; the labels keep the true numbers.
  const wsum = Object.values(byId).reduce((s, c) => s + c ** 1.6, 0);
  let mx = 0, my = 0;
  for (const [id, conf] of Object.entries(byId)) {
    const w = conf ** 1.6 / wsum;
    mx += anchorPos[id].x * w;
    my += anchorPos[id].y * w;
  }
  const marker = svg.querySelector("#marker");
  // force reflow so the transition from centre plays
  void marker.getBoundingClientRect();
  marker.setAttribute("transform", `translate(${mx.toFixed(1)} ${my.toFixed(1)})`);

  // tethers + anchor highlight
  const tethers = svg.querySelector("#tethers");
  tethers.innerHTML = "";
  svg.querySelectorAll(".anchor").forEach((g) => {
    const id = g.dataset.id;
    const conf = byId[id] ?? 0;
    const assigned = Number(id) === res.segment;
    g.classList.toggle("is-assigned", assigned);
    g.querySelector(".anchor-ring").setAttribute("r", assigned ? 17 : 13);
    g.querySelector(".anchor-core").setAttribute("r", assigned ? 4 : 3);
    g.querySelector(".anchor-pct").textContent = `${Math.round(conf * 100)}%`;

    const line = svgEl("line", {
      class: "tether",
      x1: mx, y1: my, x2: anchorPos[id].x, y2: anchorPos[id].y,
      "stroke-width": (0.6 + conf * 7).toFixed(2),
    });
    tethers.appendChild(line);
  });
  requestAnimationFrame(() => tethers.querySelectorAll(".tether").forEach((l) => (l.style.opacity = "0.85")));

  // verdict text
  el.kicker.textContent = `Assigned segment · ${Math.round(res.confidence * 100)}% confidence`;
  el.name.textContent = res.segment_name;
  el.desc.textContent = res.description;

  // why this customer -> this segment
  el.driversList.innerHTML = "";
  res.drivers.forEach((d) => {
    const li = document.createElement("li");
    li.innerHTML =
      `<span class="driver-feat">${d.feature}</span>` +
      `<span class="driver-cmp"><strong>${d.customer}</strong> vs ${d.segment_average} avg</span>`;
    el.driversList.appendChild(li);
  });
  el.drivers.hidden = res.drivers.length === 0;

  // full segment profile (collapsed)
  el.detailsTable.innerHTML =
    `<tr><td>Members</td><td>${res.segment_profile.size} · ` +
    `${Math.round(res.segment_profile.share_of_base * 100)}% of base</td></tr>` +
    `<tr><td>Live without a partner</td><td>` +
    `${Math.round(res.segment_profile.share_living_alone * 100)}%</td></tr>` +
    Object.entries(res.segment_profile.feature_means)
      .map(([k, v]) => `<tr><td>${PROFILE_LABELS[k] ?? k}</td><td>${formatProfileValue(k, v)}</td></tr>`)
      .join("");
  el.details.hidden = false;
  el.details.open = false;

  // confidence spread
  el.spread.innerHTML = "";
  res.confidence_by_segment.forEach((s) => {
    const row = document.createElement("div");
    row.className = "spread-row" + (s.segment === res.segment ? " is-assigned" : "");
    row.innerHTML = `
      <span class="spread-name">${s.name}</span>
      <span class="spread-pct">${Math.round(s.confidence * 100)}%</span>
      <span class="spread-track"><span class="spread-fill"></span></span>`;
    el.spread.appendChild(row);
    requestAnimationFrame(() => {
      row.querySelector(".spread-fill").style.width = `${(s.confidence * 100).toFixed(1)}%`;
    });
  });

  el.provenance.hidden = false;
  card.dataset.state = "done";
}

/* ---------- payload + validation ---------- */
function collectPayload() {
  const fd = new FormData(form);
  const numeric = new Set([
    "Year_Birth", "Income", "Kidhome", "Teenhome", "Recency",
    "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts",
    "MntSweetProducts", "MntGoldProds", "NumDealsPurchases", "NumWebPurchases",
    "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth",
    "Complain", "Response",
  ]);
  const payload = {};
  for (const [k, v] of fd.entries()) {
    payload[k] = numeric.has(k) ? Number(v) : v;
  }
  return payload;
}

function findInvalidFields() {
  const bad = [];
  form.querySelectorAll("input, select").forEach((f) => {
    if (f.name && !f.validity.valid) bad.push(f.name);
  });
  return bad;
}

/* ---------- prediction ---------- */
function setNotice(state, html) {
  card.dataset.state = state;
  notice.innerHTML = html;
}

const WAKING_HTML =
  `<strong>Waking the server.</strong> The demo backend sleeps after 15 minutes idle
   on Render's free tier — the first request can take up to a minute. This updates
   automatically when it responds.`;

async function callApi(payload) {
  const ctrl = new AbortController();
  const timeout = setTimeout(() => ctrl.abort(), COLD_START_TIMEOUT_MS);
  const wakeHint = setTimeout(() => setNotice("waking", WAKING_HTML), WAKE_NOTICE_AFTER_MS);
  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: ctrl.signal,
    });
    clearTimeout(timeout);
    clearTimeout(wakeHint);

    if (res.status === 422) {
      const body = await res.json().catch(() => null);
      const items = (body?.detail || [])
        .map((d) => `<li>${(d.loc || []).slice(1).join(".")}: ${d.msg}</li>`)
        .join("");
      setNotice("error", `The API rejected the input.<ul>${items}</ul>`);
      return;
    }
    if (!res.ok) throw new Error(`server ${res.status}`);
    paintResult(await res.json());
  } catch (err) {
    clearTimeout(timeout);
    clearTimeout(wakeHint);
    const msg = err.name === "AbortError"
      ? `No response after ${COLD_START_TIMEOUT_MS / 1000}s. The server may still be waking —
         <button type="button" id="retry-btn">try again</button>.`
      : `Can't reach the API at <strong>${API_BASE}</strong>.
         <button type="button" id="retry-btn">Try again</button>`;
    setNotice("error", msg);
    notice.querySelector("#retry-btn")?.addEventListener("click", () => runPrediction());
  }
}

async function runPrediction() {
  const bad = findInvalidFields();
  if (bad.length) {
    setNotice("error", `Check these fields: <strong>${bad.join(", ")}</strong>`);
    return;
  }
  runBtn.disabled = true;
  runBtn.dataset.loading = "true";
  card.dataset.state = "loading";
  notice.innerHTML = "";
  try {
    await callApi(collectPayload());
  } finally {
    runBtn.disabled = false;
    runBtn.dataset.loading = "false";
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  runPrediction();
});

form.addEventListener("reset", () => {
  setTimeout(() => {
    card.dataset.state = "idle";
    el.kicker.textContent = "Assigned segment";
    el.name.textContent = "—";
    el.desc.textContent = "";
    el.spread.innerHTML = "";
    el.drivers.hidden = true;
    el.details.hidden = true;
    el.provenance.hidden = true;
    notice.innerHTML = "";
    if (svg.querySelector("#marker")) renderMap();
  }, 0);
});

/* ---------- boot ---------- */
(async function init() {
  layoutAnchors({});          // placeholder diamond while /segments loads
  renderMap();

  const ctrl = new AbortController();
  const timeout = setTimeout(() => ctrl.abort(), COLD_START_TIMEOUT_MS);
  const wakeHint = setTimeout(() => {
    hint.textContent = "Waking the server — the free tier sleeps when idle, first load takes up to a minute.";
  }, WAKE_NOTICE_AFTER_MS);
  try {
    const res = await fetch(`${API_BASE}/segments`, { signal: ctrl.signal });
    clearTimeout(timeout);
    clearTimeout(wakeHint);
    if (!res.ok) throw new Error();
    segmentMeta = await res.json();
    layoutAnchors(segmentMeta);
    renderMap();
    hint.textContent = "Fill in the profile, then find the segment.";
  } catch {
    clearTimeout(timeout);
    clearTimeout(wakeHint);
    hint.textContent = "Couldn't reach the backend.";
    card.dataset.state = "error";
    notice.innerHTML =
      `Couldn't load segment data from <strong>${API_BASE}</strong> — it may be asleep.
       <button type="button" id="init-retry">Retry</button>`;
    notice.querySelector("#init-retry")?.addEventListener("click", () => location.reload());
  }
})();
