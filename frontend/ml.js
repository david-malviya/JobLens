/**
 * JobLens — AI/ML Insights Module
 * Handles semantic search, clustering visualization, trend forecasting, resume matching.
 * Uses Chart.js for visualizations (loaded from CDN).
 */

console.log("[ML.js] AI/ML Insights module loaded");
const ML_BASE = "http://127.0.0.1:5000/api/ml";

function mlToken() { return localStorage.getItem("joblens_token"); }
function mlHeaders(json = false) {
  const h = { Authorization: `Bearer ${mlToken()}` };
  if (json) h["Content-Type"] = "application/json";
  return h;
}

/**
 * Shared fetch helper for ML endpoints.
 * Handles auth errors (401 → redirect), non-OK responses, and network errors.
 */
async function mlFetch(url, options = {}) {
  const res = await fetch(url, options);
  if (res.status === 401) {
    // Token expired or invalid — redirect to login
    localStorage.removeItem("joblens_token");
    localStorage.removeItem("joblens_user");
    window.location.href = "auth.html";
    throw new Error("Session expired. Redirecting to login...");
  }
  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.error || `Server error (${res.status})`);
  }
  return res.json();
}

// ═══════════════════════════════════════
// SEMANTIC SEARCH
// ═══════════════════════════════════════
async function runSemanticSearch() {
  const query = document.getElementById("mlSearchInput").value.trim();
  const box = document.getElementById("mlSearchResults");
  if (!query) { box.innerHTML = "<p class='ml-hint'>Enter a query above.</p>"; return; }

  box.innerHTML = "<p class='ml-hint'>Searching...</p>";
  try {
    const data = await mlFetch(`${ML_BASE}/semantic-search`, {
      method: "POST", headers: mlHeaders(true),
      body: JSON.stringify({ query, top_k: 20 }),
    });
    if (!data.results || !data.results.length) {
      box.innerHTML = "<p class='ml-hint'>No semantic matches found.</p>"; return;
    }
    box.innerHTML = `<p class="ml-meta">${data.total} results &middot; Method: ${data.method}</p>` +
      data.results.map(j => `
        <div class="ml-result-row">
          <div class="ml-result-info">
            <strong>${esc(j.job_title)}</strong>
            <span class="ml-result-company">${esc(j.company_name)} ${googleSearchLink(j.company_name)}</span>
            <span class="ml-result-detail">${esc(j.location)} &middot; ${esc(j.employment_type)}</span>
          </div>
          <div class="ml-score">${j.match_score}%</div>
        </div>`).join("");
  } catch (e) { box.innerHTML = "<p class='ml-hint'>Error: " + e.message + "</p>"; }
}

// ═══════════════════════════════════════
// CLUSTERING
// ═══════════════════════════════════════
let clusterChart = null;

async function loadClusters() {
  const box = document.getElementById("clusterInfo");
  box.innerHTML = "<p class='ml-hint'>Loading clusters...</p>";
  try {
    const data = await mlFetch(`${ML_BASE}/clusters`, { headers: mlHeaders() });

    // Validate response structure
    if (!data.points || !Array.isArray(data.points) || data.points.length === 0) {
      box.innerHTML = "<p class='ml-hint'>No cluster data available.</p>";
      return;
    }

    renderClusterChart(data);
    // Cluster legend
    let legend = "<div class='cluster-legend'>";
    for (const [k, name] of Object.entries(data.cluster_names || {})) {
      const color = getClusterColor(parseInt(k));
      legend += `<span class="cluster-tag" style="border-color:${color};color:${color}">● ${name}</span>`;
    }
    legend += "</div>";
    box.innerHTML = `<p class="ml-meta">${data.n_clusters} clusters &middot; ${data.points.length} jobs sampled &middot; PCA variance: ${(data.pca_variance[0]*100).toFixed(1)}% + ${(data.pca_variance[1]*100).toFixed(1)}%</p>${legend}`;
  } catch (e) { box.innerHTML = "<p class='ml-hint'>Error loading clusters: " + e.message + "</p>"; }
}

function renderClusterChart(data) {
  const canvas = document.getElementById("clusterCanvas");
  if (!canvas) { console.error("clusterCanvas element not found"); return; }

  // Check if Chart.js is loaded
  if (typeof Chart === "undefined") {
    console.error("Chart.js not loaded");
    document.getElementById("clusterInfo").innerHTML = "<p class='ml-hint'>Chart library not loaded. Please refresh the page.</p>";
    return;
  }

  const ctx = canvas.getContext("2d");
  if (clusterChart) clusterChart.destroy();

  const datasets = {};
  data.points.forEach(p => {
    if (!datasets[p.cluster]) {
      datasets[p.cluster] = {
        label: (data.cluster_names && data.cluster_names[p.cluster]) || `Cluster ${p.cluster}`,
        data: [], backgroundColor: getClusterColor(p.cluster) + "99",
        borderColor: getClusterColor(p.cluster), borderWidth: 1,
        pointRadius: 3, pointHoverRadius: 6,
      };
    }
    datasets[p.cluster].data.push({ x: p.x, y: p.y, title: p.job_title, company: p.company });
  });

  clusterChart = new Chart(ctx, {
    type: "scatter",
    data: { datasets: Object.values(datasets) },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: true, position: "bottom", labels: { boxWidth: 8, padding: 12, font: { size: 11 } } },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const d = ctx.raw;
              return `${d.title} — ${d.company}`;
            }
          }
        }
      },
      scales: {
        x: { title: { display: true, text: "PCA Component 1" }, grid: { color: "rgba(128,128,128,0.15)" } },
        y: { title: { display: true, text: "PCA Component 2" }, grid: { color: "rgba(128,128,128,0.15)" } },
      },
    },
  });
}

// ═══════════════════════════════════════
// TREND FORECASTING
// ═══════════════════════════════════════
let trendChart = null;

async function loadTrends() {
  const role = document.getElementById("trendRoleInput").value.trim() || "";
  const box = document.getElementById("trendInfo");
  box.innerHTML = "<p class='ml-hint'>Analyzing trends...</p>";
  try {
    const url = role ? `${ML_BASE}/trends?role=${encodeURIComponent(role)}&months=6` : `${ML_BASE}/trends?months=6`;
    const data = await mlFetch(url, { headers: mlHeaders() });

    if (!data.historical || !data.historical.length) {
      box.innerHTML = "<p class='ml-hint'>Not enough data for this role.</p>"; return;
    }
    renderTrendChart(data);
    let info = `<p class="ml-meta">${data.total_jobs} total jobs${data.role ? " for '" + data.role + "'" : ""} &middot; R² = ${data.r_squared} &middot; Polynomial Regression</p>`;
    if (data.top_roles && Object.keys(data.top_roles).length) {
      info += "<div class='trend-roles'><strong>Top roles:</strong> ";
      info += Object.entries(data.top_roles).slice(0, 5).map(([r, c]) => `${esc(r)} (${c})`).join(", ");
      info += "</div>";
    }
    box.innerHTML = info;
  } catch (e) { box.innerHTML = "<p class='ml-hint'>Error loading trends: " + e.message + "</p>"; }
}

function renderTrendChart(data) {
  const canvas = document.getElementById("trendCanvas");
  if (!canvas) { console.error("trendCanvas element not found"); return; }

  // Check if Chart.js is loaded
  if (typeof Chart === "undefined") {
    console.error("Chart.js not loaded");
    document.getElementById("trendInfo").innerHTML = "<p class='ml-hint'>Chart library not loaded. Please refresh the page.</p>";
    return;
  }

  const ctx = canvas.getContext("2d");
  if (trendChart) trendChart.destroy();

  const histLabels = data.historical.map(d => d.month);
  const histCounts = data.historical.map(d => d.count);
  const histTrend = data.historical.map(d => d.trend);
  const foreLabels = data.forecast.map(d => d.month);
  const foreCounts = data.forecast.map(d => d.count);

  const allLabels = [...histLabels, ...foreLabels];
  const actual = [...histCounts, ...Array(foreLabels.length).fill(null)];
  const trend = [...histTrend, ...foreCounts];

  const accentColor = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#4f6ef7';

  trendChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: allLabels,
      datasets: [
        {
          label: "Actual postings",
          data: actual, borderColor: accentColor, backgroundColor: accentColor + "22",
          fill: true, tension: 0.3, pointRadius: 3, borderWidth: 2,
        },
        {
          label: "Trend + Forecast",
          data: trend, borderColor: "#f59e0b", borderDash: [6, 3],
          fill: false, tension: 0.3, pointRadius: 2, borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: "bottom", labels: { boxWidth: 12, padding: 12, font: { size: 11 } } } },
      scales: {
        x: { grid: { color: "rgba(128,128,128,0.15)" } },
        y: { title: { display: true, text: "Job postings" }, grid: { color: "rgba(128,128,128,0.15)" }, beginAtZero: true },
      },
    },
  });
}

// ═══════════════════════════════════════
// RESUME MATCHING
// ═══════════════════════════════════════
async function runResumeMatch() {
  const text = document.getElementById("resumeTextarea").value.trim();
  const box = document.getElementById("resumeResults");
  if (!text || text.length < 20) { box.innerHTML = "<p class='ml-hint'>Paste your resume or skills (at least 20 characters).</p>"; return; }

  box.innerHTML = "<p class='ml-hint'>Matching...</p>";
  try {
    const data = await mlFetch(`${ML_BASE}/resume-match`, {
      method: "POST", headers: mlHeaders(true),
      body: JSON.stringify({ resume_text: text, top_k: 15 }),
    });
    if (!data.results || !data.results.length) {
      box.innerHTML = "<p class='ml-hint'>No matches found. Try adding more details.</p>"; return;
    }
    box.innerHTML = `<p class="ml-meta">${data.total} matches &middot; ${data.method}</p>` +
      data.results.map(j => `
        <div class="ml-result-row">
          <div class="ml-result-info">
            <strong>${esc(j.job_title)}</strong>
            <span class="ml-result-company">${esc(j.company_name)} ${googleSearchLink(j.company_name)}</span>
            <span class="ml-result-detail">${esc(j.location)} &middot; ${esc(j.seniority_level)} &middot; ${esc(j.hiring_status)}</span>
          </div>
          <div class="ml-score ${j.match_score >= 30 ? 'score-high' : j.match_score >= 15 ? 'score-mid' : ''}">${j.match_score}%</div>
        </div>`).join("");
  } catch (e) { box.innerHTML = "<p class='ml-hint'>Error: " + e.message + "</p>"; }
}

// ═══════════════════════════════════════
// UTILITIES
// ═══════════════════════════════════════
const CLUSTER_COLORS = ["#4f6ef7","#22a06b","#f59e0b","#e34850","#8b5cf6","#06b6d4","#ec4899","#84cc16"];
function getClusterColor(i) { return CLUSTER_COLORS[i % CLUSTER_COLORS.length]; }
