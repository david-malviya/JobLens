/**
 * JobLens — Auth Logic
 * Login, registration, token management, theme toggle.
 */

const API_BASE = "http://127.0.0.1:5000/api";

// ── Theme ──
function getTheme() { return localStorage.getItem("joblens_theme") || "light"; }
function setTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("joblens_theme", theme);
  const btn = document.getElementById("themeToggle");
  if (btn) btn.textContent = theme === "dark" ? "☀️" : "🌙";
}

// ── Init ──
document.addEventListener("DOMContentLoaded", () => {
  setTheme(getTheme());

  document.getElementById("themeToggle").addEventListener("click", () => {
    setTheme(getTheme() === "dark" ? "light" : "dark");
  });

  // If already logged in, go to dashboard
  const token = localStorage.getItem("joblens_token");
  if (token) {
    fetch(`${API_BASE}/auth/me`, { headers: { Authorization: `Bearer ${token}` } })
      .then(r => { if (r.ok) window.location.href = "index.html"; })
      .catch(() => { localStorage.removeItem("joblens_token"); localStorage.removeItem("joblens_user"); });
  }
});

// ── Tabs ──
function switchTab(tab) {
  document.getElementById("loginForm").style.display = tab === "login" ? "flex" : "none";
  document.getElementById("registerForm").style.display = tab === "register" ? "flex" : "none";
  document.getElementById("tabLogin").classList.toggle("active", tab === "login");
  document.getElementById("tabRegister").classList.toggle("active", tab === "register");
  hideMsg();
}

function showMsg(text, type = "error") {
  const el = document.getElementById("authMessage");
  el.textContent = text; el.className = `auth-message ${type}`; el.style.display = "block";
}
function hideMsg() {
  const el = document.getElementById("authMessage"); el.style.display = "none";
}

function togglePw(id, btn) {
  const inp = document.getElementById(id);
  inp.type = inp.type === "password" ? "text" : "password";
  btn.textContent = inp.type === "password" ? "👁" : "✕";
}

function setLoading(btnId, on) {
  const btn = document.getElementById(btnId);
  btn.disabled = on;
  btn.querySelector(".btn-text").style.display = on ? "none" : "inline";
  btn.querySelector(".btn-loader").style.display = on ? "inline-block" : "none";
}

// ── Login ──
async function handleLogin(e) {
  e.preventDefault();
  const email = document.getElementById("loginEmail").value.trim();
  const password = document.getElementById("loginPassword").value;
  if (!email || !password) { showMsg("Fill in all fields"); return; }

  setLoading("loginBtn", true);
  try {
    const res = await fetch(`${API_BASE}/auth/login`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    const data = await res.json();
    if (!res.ok) { showMsg(data.error || "Login failed"); setLoading("loginBtn", false); return; }

    localStorage.setItem("joblens_token", data.token);
    localStorage.setItem("joblens_user", JSON.stringify(data.user));
    showMsg("Signed in! Redirecting...", "success");
    setTimeout(() => window.location.href = "index.html", 600);
  } catch {
    showMsg("Can't reach the server"); setLoading("loginBtn", false);
  }
}

// ── Register ──
async function handleRegister(e) {
  e.preventDefault();
  const name = document.getElementById("regName").value.trim();
  const email = document.getElementById("regEmail").value.trim();
  const password = document.getElementById("regPassword").value;
  const confirm = document.getElementById("regConfirm").value;

  if (!name || !email || !password || !confirm) { showMsg("Fill in all fields"); return; }
  if (password !== confirm) { showMsg("Passwords don't match"); return; }
  if (password.length < 6) { showMsg("Password must be at least 6 characters"); return; }

  setLoading("registerBtn", true);
  try {
    const res = await fetch(`${API_BASE}/auth/register`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, email, password }),
    });
    const data = await res.json();
    if (!res.ok) { showMsg(data.error || "Registration failed"); setLoading("registerBtn", false); return; }

    localStorage.setItem("joblens_token", data.token);
    localStorage.setItem("joblens_user", JSON.stringify(data.user));
    showMsg("Account created! Redirecting...", "success");
    setTimeout(() => window.location.href = "index.html", 600);
  } catch {
    showMsg("Can't reach the server"); setLoading("registerBtn", false);
  }
}
