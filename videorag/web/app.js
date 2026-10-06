// Video RAG web client: library management, streaming Q&A, and timestamp-linked citations.
const $ = (sel) => document.querySelector(sel);
const state = { videos: [], current: null, history: [], busy: false };

const fmt = (s) => {
  s = Math.max(0, Math.floor(s || 0));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = String(s % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${sec}` : `${m}:${sec}`;
};
const esc = (t) => t.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

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

function renderLibrary() {
  const list = $("#video-list");
  list.innerHTML = "";
  $("#empty-lib").style.display = state.videos.length ? "none" : "block";
  for (const v of state.videos) {
    const li = document.createElement("li");
    li.className = "video-item" + (state.current === v.id ? " active" : "");
    const stage = v.status === "processing" ? `${v.stage}…` : v.status;
    li.innerHTML = `
      <div class="title" title="${esc(v.title)}">${esc(v.title)}</div>
      <div class="meta"><span class="badge ${v.status}">${esc(stage)}</span>
        <span class="muted small">${v.duration ? fmt(v.duration) : ""}${v.language ? " · " + v.language : ""}</span></div>
      ${v.status === "processing" || v.status === "queued" ? `<div class="progress"><div style="width:${Math.round((v.progress || 0) * 100)}%"></div></div>` : ""}
      ${v.error ? `<div class="error-text">${esc(v.error)}</div>` : ""}
      <button class="del" title="Delete">✕</button>`;
    li.addEventListener("click", () => v.has_media && playAt(v.id, 0, false));
    li.querySelector(".del").addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!confirm(`Delete "${v.title}"?`)) return;
      try { await api(`/api/videos/${v.id}`, { method: "DELETE" }); } catch (err) { alert(err.message); }
      if (state.current === v.id) { state.current = null; $("#player").removeAttribute("src"); $("#player-wrap").classList.remove("has-video"); }
      refreshVideos();
    });
    list.appendChild(li);
  }
  const scope = $("#scope"), selected = scope.value;
  scope.innerHTML = '<option value="">All videos</option>' +
    state.videos.filter((v) => v.status === "ready").map((v) => `<option value="${v.id}">${esc(v.title.slice(0, 40))}</option>`).join("");
  scope.value = selected;
}

async function uploadFiles(files) {
  for (const f of files) {
    const fd = new FormData();
    fd.append("file", f);
    try { await api("/api/videos", { method: "POST", body: fd }); } catch (e) { alert(`${f.name}: ${e.message}`); }
    refreshVideos();
  }
}

// ---------- player ----------
function playAt(videoId, t, autoplay = true) {
  const player = $("#player"), v = state.videos.find((x) => x.id === videoId);
  if (state.current !== videoId) {
    player.src = `/api/videos/${videoId}/media`;
    state.current = videoId;
    renderLibrary();
  }
  $("#player-wrap").classList.add("has-video");
  $("#now-playing").textContent = v ? `${v.title} — ${fmt(t)}` : "";
  const seek = () => { player.currentTime = t; if (autoplay) player.play().catch(() => {}); };
  player.readyState >= 1 ? seek() : player.addEventListener("loadedmetadata", seek, { once: true });
}

// ---------- chat ----------
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
        return src ? `<button class="cite" data-n="${n}" title="${esc(src.video_title)} @ ${fmt(src.start)}">${fmt(src.start)}</button>` : m;
      });
  }
}

function sourceCards(sources) {
  return `<div class="sources">${sources.map((s) => `
    <button class="source" data-n="${s.n}">
      ${s.frame_url ? `<img src="${s.frame_url}" loading="lazy" alt="">` : ""}
      <div class="body"><div class="head"><span>[${s.n}] ${fmt(s.start)}</span><span class="muted">${s.kind === "frame" ? "visual" : ""}</span></div>
      <div class="snippet">${esc(s.video_title)} — ${esc(s.text)}</div></div>
    </button>`).join("")}</div>`;
}

function bindCitations(el, sources) {
  el.querySelectorAll("[data-n]").forEach((b) =>
    b.addEventListener("click", () => {
      const s = sources.find((x) => x.n === Number(b.dataset.n));
      if (s) playAt(s.video_id, Math.max(0, s.start - 1));
    }));
}

function addMsg(cls, html) {
  $("#hint")?.remove();
  const div = document.createElement("div");
  div.className = `msg ${cls}`;
  div.innerHTML = html;
  $("#chat").appendChild(div);
  $("#chat").scrollTop = $("#chat").scrollHeight;
  return div;
}

async function ask(question) {
  if (state.busy) return;
  state.busy = true;
  $("#ask-btn").disabled = true;
  addMsg("user", esc(question));
  const bot = addMsg("bot typing", "");
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
        bot.innerHTML = renderMarkdown(answer, sources);
        $("#chat").scrollTop = $("#chat").scrollHeight;
      }
    }
  } catch (e) {
    answer += `\n\n_Error: ${e.message}_`;
  }
  bot.classList.remove("typing");
  bot.innerHTML = renderMarkdown(answer || "_No answer._", sources) + (sources.length ? sourceCards(sources) : "");
  bindCitations(bot, sources);
  state.history.push({ role: "user", content: question }, { role: "assistant", content: answer });
  state.history = state.history.slice(-6);
  state.busy = false;
  $("#ask-btn").disabled = false;
  $("#chat").scrollTop = $("#chat").scrollHeight;
}

// ---------- wiring ----------
const dz = $("#dropzone");
$("#file-input").addEventListener("change", (e) => { uploadFiles([...e.target.files]); e.target.value = ""; });
["dragenter", "dragover"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", (e) => uploadFiles([...e.dataTransfer.files]));

$("#url-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const url = $("#url-input").value.trim();
  try { await api("/api/videos/url", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url }) }); $("#url-input").value = ""; }
  catch (err) { alert(err.message); }
  refreshVideos();
});

$("#ask-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("#question").value.trim();
  if (q) { $("#question").value = ""; ask(q); }
});
document.querySelectorAll(".examples .chip").forEach((c) => c.addEventListener("click", () => ask(c.textContent)));

api("/api/health").then((h) => ($("#model-info").textContent = `${h.claude_model} · whisper ${h.whisper_model}`)).catch(() => {});
refreshVideos();
