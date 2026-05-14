/**
 * JobLens — Main Application
 * Auth guard, theme toggle, API integration, filtering, pagination.
 */

const API_BASE = "http://127.0.0.1:5000/api";

// ── Auth ──
function getToken() { return localStorage.getItem("joblens_token"); }
function getUser() { try { return JSON.parse(localStorage.getItem("joblens_user")); } catch { return null; } }
function logout() {
  localStorage.removeItem("joblens_token");
  localStorage.removeItem("joblens_user");
  window.location.href = "auth.html";
}
function checkAuth() {
  if (!getToken()) { window.location.href = "auth.html"; return false; }
  return true;
}

// ── Theme ──
function getTheme() { return localStorage.getItem("joblens_theme") || "light"; }
function setTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("joblens_theme", theme);
  const btn = document.getElementById("themeToggle");
  if (btn) btn.textContent = theme === "dark" ? "☀️" : "🌙";
}

// ── State ──
let currentPage = 1;
const perPage = 20;
let debounceTimer = null;
const $ = (id) => document.getElementById(id);

// ── Init ──
document.addEventListener("DOMContentLoaded", () => {
  // Apply saved theme
  setTheme(getTheme());

  if (!checkAuth()) return;

  // User profile
  const user = getUser();
  if (user) {
    $("userName").textContent = user.name;
    $("userAvatar").textContent = user.name.charAt(0).toUpperCase();
  }

  // Events
  $("logoutBtn").addEventListener("click", logout);
  $("themeToggle").addEventListener("click", () => {
    setTheme(getTheme() === "dark" ? "light" : "dark");
  });

  $("searchInput").addEventListener("input", () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => { currentPage = 1; loadJobs(); }, 400);
  });

  ["filterLocation","filterFunction","filterSeniority","filterEmployment","filterHiring"].forEach(id => {
    $(id).addEventListener("change", () => { currentPage = 1; loadJobs(); });
  });

  $("clearFilters").addEventListener("click", () => {
    $("searchInput").value = "";
    ["filterLocation","filterFunction","filterSeniority","filterEmployment","filterHiring"].forEach(id => $(id).value = "");
    currentPage = 1;
    loadJobs();
  });

  // Load data
  loadStats();
  loadFilters();
  loadJobs();
});

// ── API ──
async function apiFetch(endpoint) {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    headers: { Authorization: `Bearer ${getToken()}` },
  });
  if (res.status === 401) { logout(); return; }
  if (!res.ok) throw new Error(`API ${res.status}`);
  return res.json();
}

function buildQuery() {
  const p = new URLSearchParams();
  const s = $("searchInput").value.trim();
  if (s) p.set("search", s);
  if ($("filterLocation").value) p.set("location", $("filterLocation").value);
  if ($("filterFunction").value) p.set("job_function", $("filterFunction").value);
  if ($("filterSeniority").value) p.set("seniority", $("filterSeniority").value);
  if ($("filterEmployment").value) p.set("employment_type", $("filterEmployment").value);
  if ($("filterHiring").value) p.set("hiring_status", $("filterHiring").value);
  p.set("page", currentPage);
  p.set("per_page", perPage);
  return p.toString();
}

// ── Stats ──
async function loadStats() {
  try {
    const d = await apiFetch("/jobs/stats");
    animateNum("statJobs", d.total_jobs);
    animateNum("statHiring", d.actively_hiring);
    animateNum("statCompanies", d.total_companies);
    animateNum("statLocations", d.total_locations);
    $("nav-total").textContent = fmt(d.total_jobs);
    $("nav-hiring").textContent = fmt(d.actively_hiring);
  } catch (e) { console.error("Stats:", e); }
}

function animateNum(id, target) {
  const el = $(id); const dur = 1200; const t0 = performance.now();
  (function tick(now) {
    const p = Math.min((now - t0) / dur, 1);
    el.textContent = fmt(Math.floor((1 - Math.pow(1 - p, 3)) * target));
    if (p < 1) requestAnimationFrame(tick);
  })(t0);
}
function fmt(n) { return n.toLocaleString("en-US"); }

// ── Filters ──
async function loadFilters() {
  try {
    const d = await apiFetch("/jobs/filters");
    fillSelect($("filterLocation"), d.locations.slice(0, 100), "Location");
    fillSelect($("filterFunction"), d.job_functions.slice(0, 60), "Function");
    fillSelect($("filterSeniority"), d.seniority_levels, "Seniority");
    fillSelect($("filterEmployment"), d.employment_types, "Type");
    fillSelect($("filterHiring"), d.hiring_statuses, "Status");
  } catch (e) { console.error("Filters:", e); }
}

function fillSelect(el, opts, label) {
  el.innerHTML = `<option value="">${label}</option>`;
  opts.forEach(o => { const op = document.createElement("option"); op.value = o; op.textContent = o; el.appendChild(op); });
}

// ── Jobs ──
async function loadJobs() {
  $("loadingOverlay").classList.add("show");
  try {
    const d = await apiFetch(`/jobs?${buildQuery()}`);
    renderJobs(d.jobs);
    renderPag(d.pagination);
    $("resultsCount").innerHTML = `Showing <strong>${d.jobs.length}</strong> of <strong>${fmt(d.pagination.total)}</strong> results`;
  } catch (e) {
    console.error("Jobs:", e);
    $("jobsGrid").innerHTML = `<div class="empty-state"><h3>Can't reach the server</h3><p>Make sure the Flask backend is running on port 5000.</p></div>`;
    $("resultsCount").textContent = "Connection error";
  }
  $("loadingOverlay").classList.remove("show");
}

function renderJobs(jobs) {
  if (!jobs.length) {
    $("jobsGrid").innerHTML = `<div class="empty-state"><h3>No matching jobs</h3><p>Try broadening your search or removing some filters.</p></div>`;
    return;
  }
  $("jobsGrid").innerHTML = jobs.map(j => `
    <div class="job-row">
      <div class="job-info">
        <h3>${esc(j.job_title)}</h3>
        <div class="job-company">${esc(j.company_name)} ${googleSearchLink(j.company_name)}</div>
        <div class="job-tags">
          <span class="job-tag">${esc(j.location)}</span>
          <span class="job-tag">${esc(j.employment_type)}</span>
          <span class="job-tag">${esc(j.seniority_level)}</span>
        </div>
      </div>
      <div class="job-right">
        <span class="job-status ${j.hiring_status === 'Actively Hiring' ? 'status-active' : 'status-early'}">
          <span class="status-dot"></span>
          ${esc(j.hiring_status)}
        </span>
        <span class="job-date">${j.date || '—'}</span>
      </div>
    </div>`).join("");
}

// ── Pagination ──
function renderPag(pg) {
  if (pg.total_pages <= 1) { $("pagination").innerHTML = ""; return; }
  let h = `<button class="page-btn" ${pg.page <= 1 ? "disabled" : ""} onclick="goToPage(${pg.page-1})">Prev</button>`;
  const max = 5;
  let s = Math.max(1, pg.page - Math.floor(max/2));
  let e = Math.min(pg.total_pages, s + max - 1);
  if (e - s < max - 1) s = Math.max(1, e - max + 1);
  if (s > 1) h += `<button class="page-btn" onclick="goToPage(1)">1</button><span style="color:var(--text-faint);padding:0 2px">…</span>`;
  for (let i = s; i <= e; i++) h += `<button class="page-btn ${i===pg.page?"active":""}" onclick="goToPage(${i})">${i}</button>`;
  if (e < pg.total_pages) h += `<span style="color:var(--text-faint);padding:0 2px">…</span><button class="page-btn" onclick="goToPage(${pg.total_pages})">${pg.total_pages}</button>`;
  h += `<button class="page-btn" ${pg.page>=pg.total_pages?"disabled":""} onclick="goToPage(${pg.page+1})">Next</button>`;
  $("pagination").innerHTML = h;
}
function goToPage(p) {
  currentPage = p; loadJobs();
  document.querySelector(".results-bar").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── Util ──
function esc(s) { if (!s) return ""; const d = document.createElement("div"); d.textContent = s; return d.innerHTML; }
function googleSearchLink(company) {
  if (!company || company === "Not Specified") return "";
  const query = encodeURIComponent(company + " careers");
  return `<a href="https://www.google.com/search?q=${query}" target="_blank" title="Search ${esc(company)} on Google" class="google-link" onclick="event.stopPropagation()">
    <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/><path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/><path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z" fill="#FBBC05"/><path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/></svg>
  </a>`;
}

// ── Page Tab Switching ──
let aiLoaded = false;
function switchPage(page) {
  $("pageJobs").style.display = page === "jobs" ? "block" : "none";
  $("pageAI").style.display = page === "ai" ? "block" : "none";
  $("tabJobs").classList.toggle("active", page === "jobs");
  $("tabAI").classList.toggle("active", page === "ai");

  if (page === "ai" && !aiLoaded) {
    aiLoaded = true;
    loadClusters();
    loadTrends();
  }
}
