#pragma once

// Minified single-page HTML, modern CSS and vanilla JS for LlamaNX Web Remote Control.
// Zero external CDNs or dependencies, fast load over Switch LAN Wi-Fi.

static const char INDEX_HTML[] = R"rawliteral(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LlamaNX Remote Control</title>
<style>
:root {
  --bg: #121417; --card: #1b1f24; --border: #2d333b; --text: #e6edf3;
  --subtext: #8b949e; --accent: #2ea043; --accent-hover: #3fb950;
  --danger: #da3633; --neon: #58a6ff;
}
* { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
body { background: var(--bg); color: var(--text); display: flex; height: 100vh; overflow: hidden; }
#sidebar { width: 340px; background: var(--card); border-right: 1px solid var(--border); display: flex; flex-direction: column; padding: 20px; gap: 18px; overflow-y: auto; }
#main { flex: 1; display: flex; flex-direction: column; background: var(--bg); height: 100vh; }
h1 { font-size: 1.25rem; display: flex; align-items: center; gap: 8px; color: var(--neon); }
.badge { background: #388bfd26; color: #58a6ff; border: 1px solid #388bfd4d; font-size: 0.7rem; padding: 2px 6px; border-radius: 4px; }
.section-title { font-size: 0.8rem; font-weight: 600; text-transform: uppercase; color: var(--subtext); letter-spacing: 0.5px; }
.card { background: #161b22; border: 1px solid var(--border); border-radius: 8px; padding: 12px; display: flex; flex-direction: column; gap: 10px; }
label { font-size: 0.82rem; color: var(--subtext); display: flex; justify-content: space-between; }
label span { color: var(--text); font-weight: 600; }
select, input[type=range] { width: 100%; background: #0d1117; border: 1px solid var(--border); color: var(--text); padding: 8px; border-radius: 6px; }
input[type=range] { padding: 0; cursor: pointer; accent-color: var(--neon); }
button { background: var(--accent); color: #fff; border: 0; padding: 9px 14px; border-radius: 6px; font-weight: 600; cursor: pointer; transition: 0.2s; }
button:hover { background: var(--accent-hover); }
button.secondary { background: #21262d; border: 1px solid var(--border); color: var(--text); }
button.secondary:hover { background: #30363d; }
.bar-track { height: 8px; background: #21262d; border-radius: 4px; overflow: hidden; margin-top: 4px; }
.bar-fill { height: 100%; width: 0%; background: var(--neon); transition: width 0.3s; }
#chat-history { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 16px; }
.msg { max-width: 80%; padding: 12px 16px; border-radius: 10px; line-height: 1.5; font-size: 0.95rem; white-space: pre-wrap; word-break: break-word; }
.msg.user { align-self: flex-end; background: #1f6feb; color: #fff; border-bottom-right-radius: 2px; }
.msg.ai { align-self: flex-start; background: #21262d; border: 1px solid var(--border); border-bottom-left-radius: 2px; }
.msg.sys { align-self: center; font-size: 0.8rem; color: var(--subtext); background: transparent; }
#input-bar { border-top: 1px solid var(--border); padding: 16px 24px; display: flex; gap: 12px; background: var(--card); }
#prompt-input { flex: 1; background: #0d1117; border: 1px solid var(--border); border-radius: 8px; color: var(--text); padding: 12px 16px; font-size: 0.95rem; outline: none; }
#prompt-input:focus { border-color: var(--neon); }
@media (max-width: 768px) {
  body { flex-direction: column; }
  #sidebar { width: 100%; height: auto; max-height: 240px; }
}
</style>
</head>
<body>

<div id="sidebar">
  <h1>LlamaNX <span class="badge">Horizon OS</span></h1>

  <div class="card">
    <div class="section-title">Hardware Monitor</div>
    <div>
      <label>Memory: <span id="ram-txt">Loading...</span></label>
      <div class="bar-track"><div id="ram-bar" class="bar-fill"></div></div>
    </div>
    <div style="font-size: 0.8rem; color: var(--subtext); display: flex; justify-content: space-between;">
      <span>IP: <strong id="ip-txt" style="color:var(--text)">...</strong></span>
      <span>Target: <strong>Tegra X1</strong></span>
    </div>
  </div>

  <div class="card">
    <div class="section-title">Model Selection</div>
    <select id="model-select"><option value="">Scanning SD card...</option></select>
    <button id="load-btn" onclick="applyConfig()">Load & Apply Config</button>
  </div>

  <div class="card">
    <div class="section-title">Inference Hyperparameters</div>
    
    <div>
      <label>Context Size (n_ctx): <span id="val-ctx">1024</span></label>
      <input type="range" id="ctx-slider" min="256" max="3072" step="256" value="1024" oninput="document.getElementById('val-ctx').innerText = this.value">
    </div>

    <div>
      <label>Temperature: <span id="val-temp">0.7</span></label>
      <input type="range" id="temp-slider" min="0" max="1.5" step="0.05" value="0.7" oninput="document.getElementById('val-temp').innerText = this.value">
    </div>

    <div>
      <label>Threads: <span id="val-th">3</span></label>
      <input type="range" id="th-slider" min="1" max="4" step="1" value="3" oninput="document.getElementById('val-th').innerText = this.value">
    </div>

    <div>
      <label>Max Tokens: <span id="val-pred">256</span></label>
      <input type="range" id="pred-slider" min="32" max="1024" step="32" value="256" oninput="document.getElementById('val-pred').innerText = this.value">
    </div>
  </div>

  <button class="secondary" onclick="clearChat()">Clear Chat History</button>
</div>

<div id="main">
  <div id="chat-history">
    <div class="msg sys">Welcome to LlamaNX Web Console. Select a model on the left to begin.</div>
  </div>
  <form id="input-bar" onsubmit="sendPrompt(event)">
    <input type="text" id="prompt-input" placeholder="Type a message or instruction to your Nintendo Switch..." autocomplete="off">
    <button type="submit" id="send-btn">Send</button>
  </form>
</div>

<script>
let isGenerating = false;

async function fetchStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    document.getElementById('ram-txt').innerText = `${data.used_ram_mb} MB / ${data.total_ram_mb} MB`;
    const pct = Math.min(100, Math.round((data.used_ram_mb / data.total_ram_mb) * 100));
    document.getElementById('ram-bar').style.width = pct + '%';
    document.getElementById('ip-txt').innerText = data.ip;

    const sel = document.getElementById('model-select');
    if (data.models && data.models.length > 0 && sel.children.length <= 1) {
      sel.innerHTML = '';
      data.models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.path;
        opt.innerText = `${m.name} (${(m.size / 1048576).toFixed(0)}MB)`;
        if (m.path === data.loaded_model) opt.selected = true;
        sel.appendChild(opt);
      });
    }
  } catch (e) {
    console.error('Status fetch failed', e);
  }
}

async function applyConfig() {
  const model = document.getElementById('model-select').value;
  const n_ctx = parseInt(document.getElementById('ctx-slider').value);
  const temp = parseFloat(document.getElementById('temp-slider').value);
  const threads = parseInt(document.getElementById('th-slider').value);
  const n_predict = parseInt(document.getElementById('pred-slider').value);

  const btn = document.getElementById('load-btn');
  btn.disabled = true;
  btn.innerText = 'Loading Model...';

  try {
    const res = await fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, n_ctx, temp, threads, n_predict })
    });
    const reply = await res.json();
    appendMsg('sys', reply.message || 'Config updated');
  } catch (e) {
    appendMsg('sys', 'Failed to update config: ' + e);
  } finally {
    btn.disabled = false;
    btn.innerText = 'Load & Apply Config';
    fetchStatus();
  }
}

function appendMsg(role, text) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.innerText = text;
  document.getElementById('chat-history').appendChild(div);
  document.getElementById('chat-history').scrollTop = document.getElementById('chat-history').scrollHeight;
  return div;
}

function clearChat() {
  document.getElementById('chat-history').innerHTML = '<div class="msg sys">Chat cleared.</div>';
}

async function sendPrompt(e) {
  e.preventDefault();
  const input = document.getElementById('prompt-input');
  const text = input.value.trim();
  if (!text || isGenerating) return;

  input.value = '';
  appendMsg('user', text);
  const aiBubble = appendMsg('ai', '...');
  isGenerating = true;
  document.getElementById('send-btn').disabled = true;

  try {
    const res = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: [{ role: 'user', content: text }],
        temperature: parseFloat(document.getElementById('temp-slider').value),
        max_tokens: parseInt(document.getElementById('pred-slider').value)
      })
    });
    const data = await res.json();
    if (data.choices && data.choices[0] && data.choices[0].message) {
      aiBubble.innerText = data.choices[0].message.content;
    } else if (data.error) {
      aiBubble.innerText = 'Error: ' + data.error;
    } else {
      aiBubble.innerText = JSON.stringify(data);
    }
  } catch (err) {
    aiBubble.innerText = 'Network error: ' + err;
  } finally {
    isGenerating = false;
    document.getElementById('send-btn').disabled = false;
    fetchStatus();
  }
}

setInterval(fetchStatus, 3000);
fetchStatus();
</script>
</body>
</html>)rawliteral";
