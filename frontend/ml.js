/**
 * JobLens — AI/ML Insights Module
 * Handles semantic search, clustering visualization, trend forecasting, resume matching.
 * Uses Chart.js for visualizations (loaded from CDN).
 */

const ML_BASE = "http://127.0.0.1:5000/api/ml";

function mlToken() { return localStorage.getItem("joblens_token"); }
function mlHeaders(json = false) {
  const h = { Authorization: `Bearer ${mlToken()}` };
  if (json) h["Content-Type"] = "application/json";
  return h;
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
    const res = await fetch(`${ML_BASE}/semantic-search`, {
      method: "POST", headers: mlHeaders(true),
      body: JSON.stringify({ query, top_k: 20 }),
    });
    const data = await res.json();
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
    const res = await fetch(`${ML_BASE}/clusters`, { headers: mlHeaders() });
    const data = await res.json();
    renderClusterChart(data);
    // Cluster legend
    let legend = "<div class='cluster-legend'>";
    for (const [k, name] of Object.entries(data.cluster_names)) {
      const color = getClusterColor(parseInt(k));
      legend += `<span class="cluster-tag" style="border-color:${color};color:${color}">● ${name}</span>`;
    }
    legend += "</div>";
    box.innerHTML = `<p class="ml-meta">${data.n_clusters} clusters &middot; ${data.points.length} jobs sampled &middot; PCA variance: ${(data.pca_variance[0]*100).toFixed(1)}% + ${(data.pca_variance[1]*100).toFixed(1)}%</p>${legend}`;
  } catch (e) { box.innerHTML = "<p class='ml-hint'>Error: " + e.message + "</p>"; }
}

function renderClusterChart(data) {
  const ctx = document.getElementById("clusterCanvas").getContext("2d");
  if (clusterChart) clusterChart.destroy();

  const datasets = {};
  data.points.forEach(p => {
    if (!datasets[p.cluster]) {
      datasets[p.cluster] = {
        label: data.cluster_names[p.cluster] || `Cluster ${p.cluster}`,
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
    const res = await fetch(url, { headers: mlHeaders() });
    const data = await res.json();

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
  } catch (e) { box.innerHTML = "<p class='ml-hint'>Error: " + e.message + "</p>"; }
}

function renderTrendChart(data) {
  const ctx = document.getElementById("trendCanvas").getContext("2d");
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
    const res = await fetch(`${ML_BASE}/resume-match`, {
      method: "POST", headers: mlHeaders(true),
      body: JSON.stringify({ resume_text: text, top_k: 15 }),
    });
    const data = await res.json();
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
function esc(s) { if (!s) return ""; const d = document.createElement("div"); d.textContent = s; return d.innerHTML; }
function googleSearchLink(company) {
  if (!company || company === "Not Specified") return "";
  const query = encodeURIComponent(company + " careers");
  return `<a href="https://www.google.com/search?q=${query}" target="_blank" title="Search ${esc(company)} on Google" class="google-link" onclick="event.stopPropagation()">
    <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/><path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/><path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z" fill="#FBBC05"/><path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/></svg>
  </a>`;
}
