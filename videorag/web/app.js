// Winter web client: library management, streaming Q&A, and timestamp-linked citations.
const $ = (sel) => document.querySelector(sel);
const state = { videos: [], current: null, history: [], busy: false, sources: [] };

const fmt = (s) => {
  s = Math.max(0, Math.floor(s || 0));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = String(s % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${sec}` : `${m}:${sec}`;
};
const esc = (t) => String(t).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function toast(message, kind = "err") {
  const t = document.createElement("div");
  t.className = `toast ${kind}`;
  t.textContent = message;
  $("#toasts").appendChild(t);
  setTimeout(() => t.remove(), 5000);
}

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
  return res.status === 204 ? null : res.json();
}

// ---------- library ----------
async function refreshVideos() {
  try { state.videos = await api("/api/videos"); } catch { return; }
  renderLibrary();
  const pending = state.videos.some((v) => v.status === "queued" || v.status === "processing");
  clearTimeout(refreshVideos.t);
  refreshVideos.t = setTimeout(refreshVideos, pending ? 1500 : 8000);
}

function videoMeta(v) {
  if (v.status === "processing" || v.status === "queued") {
    const pct = Math.round((v.progress || 0) * 100);
    return `<span class="busy">${esc(v.status === "queued" ? "queued" : v.stage || "processing")}</span> ${pct}%`;
  }
  if (v.status === "failed") return `<span class="failed">failed</span>`;
  return [v.duration ? fmt(v.duration) : null, v.language ? v.language.toUpperCase() : null].filter(Boolean).join(" · ");
}

function renderLibrary() {
  const list = $("#video-list");
  list.innerHTML = "";
  $("#empty-lib").style.display = state.videos.length ? "none" : "block";
  $("#lib-count").textContent = String(state.videos.length).padStart(2, "0");
  for (const v of state.videos) {
    const li = document.createElement("li");
    li.className = "video-item" + (state.current === v.id ? " active" : "");
    const busy = v.status === "processing" || v.status === "queued";
    li.innerHTML = `
      <div class="thumb">${v.status === "ready" ? `<img src="/api/videos/${v.id}/thumb" alt="" loading="lazy" onerror="this.remove()">` : ""}</div>
      <div class="vi-body">
        <div class="vi-title" title="${esc(v.title)}">${esc(v.title)}</div>
        <div class="vi-meta">${videoMeta(v)}</div>
        ${busy ? `<div class="progress"><div style="width:${Math.round((v.progress || 0) * 100)}%"></div></div>` : ""}
        ${v.error ? `<div class="error-text">${esc(v.error)}</div>` : ""}
      </div>
      <button class="del" title="Delete" aria-label="Delete ${esc(v.title)}">×</button>`;
    li.addEventListener("click", () => v.has_media && playAt(v.id, 0, false));
    li.querySelector(".del").addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!confirm(`Delete "${v.title}"?`)) return;
      try { await api(`/api/videos/${v.id}`, { method: "DELETE" }); } catch (err) { toast(err.message); }
      if (state.current === v.id) {
        state.current = null;
        $("#player").removeAttribute("src");
        $("#player-wrap").classList.remove("has-video");
        $("#timeline").hidden = true;
      }
      refreshVideos();
    });
    list.appendChild(li);
  }
  const scope = $("#scope"), selected = scope.value;
  scope.innerHTML = '<option value="">All recordings</option>' +
    state.videos.filter((v) => v.status === "ready").map((v) => `<option value="${v.id}">${esc(v.title.slice(0, 40))}</option>`).join("");
  scope.value = selected;
}

async function uploadFiles(files) {
  for (const f of files) {
    const fd = new FormData();
    fd.append("file", f);
    try { await api("/api/videos", { method: "POST", body: fd }); } catch (e) { toast(`${f.name}: ${e.message}`); }
    refreshVideos();
  }
}

// ---------- player + timeline ----------
function playAt(videoId, t, autoplay = true) {
  const player = $("#player");
  if (state.current !== videoId) {
    player.src = `/api/videos/${videoId}/media`;
    state.current = videoId;
    renderLibrary();
  }
  $("#player-wrap").classList.add("has-video");
  $("#timeline").hidden = false;
  renderTimeline();
  const seek = () => { player.currentTime = t; if (autoplay) player.play().catch(() => {}); };
  player.readyState >= 1 ? seek() : player.addEventListener("loadedmetadata", seek, { once: true });
}

function duration() {
  const p = $("#player"), v = state.videos.find((x) => x.id === state.current);
  return (isFinite(p.duration) && p.duration) || v?.duration || 0;
}

function renderTimeline() {
  const v = state.videos.find((x) => x.id === state.current);
  if (!v) return;
  $("#tl-title").textContent = v.title;
  const d = duration(), marks = $("#tl-marks");
  marks.innerHTML = "";
  if (!d) return;
  for (const s of state.sources.filter((x) => x.video_id === v.id)) {
    const b = document.createElement("button");
    b.className = "tl-mark";
    b.title = `[${s.n}] ${fmt(s.start)}`;
    b.style.left = `${(s.start / d) * 100}%`;
    b.style.width = `${Math.max(0.4, ((s.end - s.start) / d) * 100)}%`;
    b.innerHTML = `<span>${s.n}</span>`;
    b.addEventListener("click", (e) => { e.stopPropagation(); playAt(s.video_id, Math.max(0, s.start - 1)); });
    marks.appendChild(b);
  }
}

function updatePlayhead() {
  const p = $("#player"), d = duration();
  $("#tl-played").style.width = d ? `${Math.min(100, (p.currentTime / d) * 100)}%` : "0";
  $("#tl-time").textContent = `${fmt(p.currentTime)} / ${fmt(d)}`;
}

// ---------- thread ----------
function renderMarkdown(text, sources) {
  const lines = esc(text).split("\n");
  let html = "", inList = false;
  for (const raw of lines) {
    const line = raw.trim();
    const bullet = line.match(/^[-*•]\s+(.*)/) || line.match(/^\d+\.\s+(.*)/);
    if (bullet) { if (!inList) { html += "<ul>"; inList = true; } html += `<li>${inline(bullet[1])}</li>`; continue; }
    if (inList) { html += "</ul>"; inList = false; }
    if (line) html += `<p>${inline(line)}</p>`;
  }
  if (inList) html += "</ul>";
  return html;

  function inline(s) {
    return s
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*(?!\s)(.+?)\*/g, "$1<em>$2</em>")
      .replace(/(^|\s)_(.+?)_(?=$|[\s.,;:!?)])/g, "$1<em>$2</em>")
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\[(\d+)\]/g, (m, n) => {
        const src = sources.find((x) => x.n === Number(n));
        return src ? `<button class="cite" data-n="${n}" title="${esc(src.video_title)}, ${fmt(src.start)}">${fmt(src.start)}</button>` : m;
      });
  }
}

function sourceList(sources) {
  return `<ol class="sources">${sources.map((s) => `
    <li><button class="source" data-n="${s.n}" type="button">
      <span class="n">${String(s.n).padStart(2, "0")}</span>
      <span class="frame">${s.frame_url ? `<img src="${s.frame_url}" loading="lazy" alt="">` : ""}</span>
      <span class="tc">${fmt(s.start)}</span>
      <span class="txt">${s.kind === "frame" ? `<span class="kind">frame</span>` : ""}<b>${esc(s.video_title)}</b>${esc(s.text)}</span>
    </button></li>`).join("")}</ol>`;
}

function bindCitations(el, sources) {
  el.querySelectorAll("[data-n]").forEach((b) =>
    b.addEventListener("click", () => {
      const s = sources.find((x) => x.n === Number(b.dataset.n));
      if (s) playAt(s.video_id, Math.max(0, s.start - 1));
    }));
}

function addTurn(question) {
  $("#hint")?.remove();
  const turn = document.createElement("article");
  turn.className = "turn";
  turn.innerHTML = `
    <div class="q"><span class="label">Q</span><span>${esc(question)}</span></div>
    <div class="a"><span class="label">A</span><div class="answer typing"></div></div>`;
  $("#chat").appendChild(turn);
  $("#chat").scrollTop = $("#chat").scrollHeight;
  return turn;
}

async function ask(question) {
  if (state.busy) return;
  state.busy = true;
  $("#ask-btn").disabled = true;
  const turn = addTurn(question), out = turn.querySelector(".answer");
  let answer = "", sources = [];
  const scope = $("#scope").value;

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, video_ids: scope ? [scope] : null, history: state.history }),
    });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
    const reader = res.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, i); buf = buf.slice(i + 2);
        const event = (block.match(/^event: (.*)$/m) || [])[1];
        const data = JSON.parse((block.match(/^data: (.*)$/m) || [, "null"])[1]);
        if (event === "sources") sources = data;
        else if (event === "delta") answer += data;
        else if (event === "error") answer += `\n\n_Error: ${data}_`;
        out.innerHTML = renderMarkdown(answer, sources);
        $("#chat").scrollTop = $("#chat").scrollHeight;
      }
    }
  } catch (e) {
    answer += `\n\n_Error: ${e.message}_`;
  }
  out.classList.remove("typing");
  out.innerHTML = renderMarkdown(answer || "_No answer._", sources);
  if (sources.length) turn.querySelector(".a").insertAdjacentHTML("beforeend", sourceList(sources));
  bindCitations(turn, sources);
  state.sources = sources;
  renderTimeline();
  state.history.push({ role: "user", content: question }, { role: "assistant", content: answer });
  state.history = state.history.slice(-6);
  state.busy = false;
  $("#ask-btn").disabled = false;
  $("#chat").scrollTop = $("#chat").scrollHeight;
}

// ---------- wiring ----------
const bin = $("#bin");
$("#file-input").addEventListener("change", (e) => { uploadFiles([...e.target.files]); e.target.value = ""; });
["dragenter", "dragover"].forEach((t) => bin.addEventListener(t, (e) => { e.preventDefault(); bin.classList.add("drag"); }));
bin.addEventListener("dragleave", (e) => { if (!bin.contains(e.relatedTarget)) bin.classList.remove("drag"); });
bin.addEventListener("drop", (e) => { e.preventDefault(); bin.classList.remove("drag"); uploadFiles([...e.dataTransfer.files]); });

$("#url-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const url = $("#url-input").value.trim();
  try {
    await api("/api/videos/url", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url }) });
    $("#url-input").value = "";
    toast("Queued for import", "ok");
  } catch (err) { toast(err.message); }
  refreshVideos();
});

$("#ask-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("#question").value.trim();
  if (q) { $("#question").value = ""; ask(q); }
});
document.querySelectorAll(".examples .link-btn").forEach((c) => c.addEventListener("click", () => ask(c.textContent)));

const player = $("#player");
["timeupdate", "seeked", "loadedmetadata"].forEach((t) => player.addEventListener(t, updatePlayhead));
player.addEventListener("loadedmetadata", renderTimeline);
$("#tl-track").addEventListener("click", (e) => {
  const r = e.currentTarget.getBoundingClientRect(), d = duration();
  if (d) player.currentTime = ((e.clientX - r.left) / r.width) * d;
});

api("/api/health").then((h) => {
  const llm = {
    claude: `<span class="stat llm on" title="Answers are written by Claude"><i></i>Claude <b>${esc(h.llm_model)}</b></span>`,
    ollama: `<span class="stat llm on" title="Answers are written by a local model via Ollama"><i></i>Local <b>${esc(h.llm_model)}</b></span>`,
  }[h.llm_provider] ||
    `<span class="stat llm off" title="Set ANTHROPIC_API_KEY or run a local Ollama model to get written answers. Search still works."><i></i>LLM <b>not connected</b></span>`;
  $("#status").innerHTML = `<span class="stat on"><i></i>Whisper <b>${esc(h.whisper_model)}</b></span>${llm}`;
}).catch(() => {});
refreshVideos();
