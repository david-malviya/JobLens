/**
 * JobLens — LLM Chat Module
 * Grok-powered AI chat interface for querying the job dataset.
 */

// ── Chat State ──
let chatHistory = [];
let llmReady = false;
let isTyping = false;

// ── Initialize ──
async function initLLMChat() {
  try {
    const res = await fetch(`${API_BASE}/llm/status`, {
      headers: { Authorization: `Bearer ${getToken()}` },
    });
    if (res.status === 401) return;
    const data = await res.json();
    llmReady = data.ready;
    updateLLMStatus(data);
  } catch (e) {
    console.error("LLM Status:", e);
  }
}

function updateLLMStatus(data) {
  const statusEl = document.getElementById("llmStatus");
  const keySection = document.getElementById("llmKeySection");

  if (data.ready) {
    statusEl.innerHTML = `
      <span class="llm-status-dot llm-dot-active"></span>
      <span>Connected to <strong>${data.model || "Grok"}</strong></span>`;
    statusEl.className = "llm-status llm-status-active";
    if (keySection) keySection.style.display = "none";
  } else {
    statusEl.innerHTML = `
      <span class="llm-status-dot llm-dot-inactive"></span>
      <span>API key required</span>`;
    statusEl.className = "llm-status llm-status-inactive";
    if (keySection) keySection.style.display = "flex";
  }
}

// ── Set API Key ──
async function setGeminiKey() {
  const input = document.getElementById("geminiKeyInput");
  const key = input.value.trim();
  if (!key) return;

  const btn = document.getElementById("setKeyBtn");
  btn.disabled = true;
  btn.textContent = "Connecting...";

  try {
    const res = await fetch(`${API_BASE}/llm/set-key`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${getToken()}`,
      },
      body: JSON.stringify({ api_key: key }),
    });
    const data = await res.json();

    if (data.status === "success") {
      llmReady = true;
      updateLLMStatus({ ready: true, model: "grok" });
      input.value = "";
      addSystemMessage("✅ Grok API connected! Ask me anything about the job dataset.");
    } else {
      addSystemMessage("❌ Failed to set API key. Please check and try again.");
    }
  } catch (e) {
    console.error("Set key:", e);
    addSystemMessage("❌ Connection error. Make sure the backend is running.");
  }
  btn.disabled = false;
  btn.textContent = "Connect";
}

// ── Send Message ──
async function sendChatMessage() {
  const input = document.getElementById("chatInput");
  const question = input.value.trim();
  if (!question || isTyping) return;

  // Add user message
  addUserMessage(question);
  input.value = "";
  input.style.height = "auto";

  if (!llmReady) {
    addSystemMessage("⚠️ Please enter your Grok API key first to use the AI chat.");
    return;
  }

  // Show typing indicator
  isTyping = true;
  showTypingIndicator();

  try {
    const res = await fetch(`${API_BASE}/llm/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${getToken()}`,
      },
      body: JSON.stringify({
        question: question,
        history: chatHistory.slice(-6),
      }),
    });

    removeTypingIndicator();
    const data = await res.json();

    if (data.error) {
      addBotMessage("❌ " + data.error);
    } else {
      addBotMessage(data.answer);
      chatHistory.push(
        { role: "user", content: question },
        { role: "assistant", content: data.answer }
      );
    }
  } catch (e) {
    removeTypingIndicator();
    addBotMessage("❌ Failed to reach the server. Make sure the backend is running.");
    console.error("Chat error:", e);
  }

  isTyping = false;
}

// ── Message Rendering ──
function addUserMessage(text) {
  const container = document.getElementById("chatMessages");
  const msgDiv = document.createElement("div");
  msgDiv.className = "chat-message chat-user";
  msgDiv.innerHTML = `
    <div class="chat-bubble chat-bubble-user">
      <div class="chat-text">${escapeHtml(text)}</div>
    </div>
    <div class="chat-avatar chat-avatar-user">${getUserInitial()}</div>`;
  container.appendChild(msgDiv);
  scrollToBottom();
}

function addBotMessage(text) {
  const container = document.getElementById("chatMessages");
  const msgDiv = document.createElement("div");
  msgDiv.className = "chat-message chat-bot";
  msgDiv.innerHTML = `
    <div class="chat-avatar chat-avatar-bot">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
      </svg>
    </div>
    <div class="chat-bubble chat-bubble-bot">
      <div class="chat-text">${renderMarkdown(text)}</div>
    </div>`;
  container.appendChild(msgDiv);
  scrollToBottom();
}

function addSystemMessage(text) {
  const container = document.getElementById("chatMessages");
  const msgDiv = document.createElement("div");
  msgDiv.className = "chat-message chat-system";
  msgDiv.innerHTML = `<div class="chat-system-text">${text}</div>`;
  container.appendChild(msgDiv);
  scrollToBottom();
}

function showTypingIndicator() {
  const container = document.getElementById("chatMessages");
  const indicator = document.createElement("div");
  indicator.className = "chat-message chat-bot";
  indicator.id = "typingIndicator";
  indicator.innerHTML = `
    <div class="chat-avatar chat-avatar-bot">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
      </svg>
    </div>
    <div class="chat-bubble chat-bubble-bot">
      <div class="typing-dots">
        <span></span><span></span><span></span>
      </div>
    </div>`;
  container.appendChild(indicator);
  scrollToBottom();
}

function removeTypingIndicator() {
  const el = document.getElementById("typingIndicator");
  if (el) el.remove();
}

// ── Utility ──
function scrollToBottom() {
  const container = document.getElementById("chatMessages");
  requestAnimationFrame(() => {
    container.scrollTop = container.scrollHeight;
  });
}

function getUserInitial() {
  const user = getUser();
  return user ? user.name.charAt(0).toUpperCase() : "U";
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function renderMarkdown(text) {
  if (!text) return "";
  // Simple markdown rendering
  let html = escapeHtml(text);
  
  // Headers
  html = html.replace(/^### (.+)$/gm, '<h4 class="md-h4">$1</h4>');
  html = html.replace(/^## (.+)$/gm, '<h3 class="md-h3">$1</h3>');
  html = html.replace(/^# (.+)$/gm, '<h2 class="md-h2">$1</h2>');
  
  // Bold and italic
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
  
  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code class="md-code">$1</code>');
  
  // Bullet points
  html = html.replace(/^[-•] (.+)$/gm, '<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul class="md-list">$&</ul>');
  
  // Numbered lists
  html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');
  
  // Line breaks
  html = html.replace(/\n\n/g, '</p><p>');
  html = html.replace(/\n/g, '<br>');
  
  return `<p>${html}</p>`;
}

// ── Quick Questions ──
function askQuickQuestion(question) {
  document.getElementById("chatInput").value = question;
  sendChatMessage();
}

// ── Chat Input Events ──
document.addEventListener("DOMContentLoaded", () => {
  const chatInput = document.getElementById("chatInput");
  if (chatInput) {
    chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendChatMessage();
      }
    });

    // Auto-resize textarea
    chatInput.addEventListener("input", () => {
      chatInput.style.height = "auto";
      chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + "px";
    });
  }

  const keyInput = document.getElementById("geminiKeyInput");
  if (keyInput) {
    keyInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        setGeminiKey();
      }
    });
  }
});

// ── Clear Chat ──
function clearChat() {
  chatHistory = [];
  const container = document.getElementById("chatMessages");
  // Keep only the welcome message
  container.innerHTML = `
    <div class="chat-message chat-bot">
      <div class="chat-avatar chat-avatar-bot">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
        </svg>
      </div>
      <div class="chat-bubble chat-bubble-bot">
        <div class="chat-text">
          <p>👋 <strong>Hello!</strong> I'm JobLens AI, powered by Grok.</p>
          <p>I can analyze the <strong>31,000+ LinkedIn job postings</strong> in this dataset and answer your questions with data-driven insights.</p>
          <p>Try asking me something like:</p>
          <ul class="md-list">
            <li>What are the most in-demand job roles?</li>
            <li>Which companies are hiring the most?</li>
            <li>How does the job market look for data scientists?</li>
          </ul>
        </div>
      </div>
    </div>`;
}
