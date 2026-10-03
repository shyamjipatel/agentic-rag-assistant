import { api } from "./api.js";
import { element, icon, iconButton, fileIcon, dateLabel, pendingMessage, renderTurn } from "./render.js";

const $ = id => document.getElementById(id);
const state = {
  sessions: [], sessionTotal: 0, sessionOffset: 0, sessionLoading: false,
  documents: [], documentTotal: 0, documentOffset: 0, documentLoading: false,
  activeId: null, turns: [], turnCount: 0, historyLoading: false, historyVersion: 0,
  pending: null, creating: false, failedRequest: null, uploads: [], uploading: false,
  dialogSession: null, dialogAction: null, libraryReady: false, toastTimer: null,
};

function notify(message) {
  clearTimeout(state.toastTimer);
  $("toast").textContent = message;
  $("toast").hidden = false;
  state.toastTimer = setTimeout(() => { $("toast").hidden = true; }, 5000);
}

function showError(id, error) {
  const target = $(id);
  (target.querySelector("span") || target.querySelector("p") || target).textContent = error.message;
  target.hidden = false;
}

function setConnection() {
  const ready = $("sidebar-error").hidden && state.libraryReady;
  $("connection-dot").className = `connection-dot ${ready ? "online" : "offline"}`;
  $("connection-label").textContent = ready ? "Workspace connected" : "Workspace needs attention";
}

function renderSessions() {
  const query = $("session-search").value.trim().toLowerCase();
  const fragment = document.createDocumentFragment();
  for (const session of state.sessions.filter(item => item.title.toLowerCase().includes(query))) {
    const row = element("div", `session-row${session.conversation_id === state.activeId ? " active" : ""}`);
    const button = element("button", "session-select");
    button.type = "button";
    button.title = session.title;
    if (session.conversation_id === state.activeId) button.setAttribute("aria-current", "page");
    button.append(icon("chat"), element("span", "session-name", session.title));
    button.addEventListener("click", () => selectSession(session.conversation_id));
    const actions = element("div", "session-actions");
    actions.append(iconButton("edit", `Rename ${session.title}`, () => openSessionDialog(session, "rename")),
      iconButton("trash", `Delete ${session.title}`, () => openSessionDialog(session, "delete")));
    row.append(button, actions);
    fragment.append(row);
  }
  if (!fragment.childNodes.length) fragment.append(element("p", "muted sidebar-note",
    query ? "No matching conversations in the loaded list." : "Your conversations will appear here."));
  $("session-list").replaceChildren(fragment);
  $("session-count").textContent = state.sessionTotal || "";
  $("more-sessions").hidden = state.sessionOffset >= state.sessionTotal;
  updateTitle();
}

function updateTitle() {
  const session = state.sessions.find(item => item.conversation_id === state.activeId);
  $("chat-title").textContent = session?.title || (state.activeId ? "Conversation" : "New conversation");
  document.title = `${session?.title || "Document workspace"} · Agentic`;
  $("saved-indicator").hidden = !state.activeId || Boolean(state.pending?.id === state.activeId);
}

async function loadSessions(append = false) {
  if (state.sessionLoading) return;
  state.sessionLoading = true;
  $("more-sessions").disabled = true;
  try {
    const offset = append ? state.sessionOffset : 0;
    const data = await api.conversations(offset);
    state.sessions = append ? mergeBy(state.sessions, data.conversations, "conversation_id") : data.conversations;
    state.sessionOffset = offset + data.conversations.length;
    state.sessionTotal = data.total;
    $("sidebar-error").hidden = true;
    renderSessions();
  } catch (error) { showError("sidebar-error", error); }
  finally { state.sessionLoading = false; $("more-sessions").disabled = false; setConnection(); }
}

function mergeBy(previous, next, key) {
  return [...new Map([...previous, ...next].map(item => [item[key], item])).values()];
}

function activeOptions() {
  return { document_id: $("document-scope").value || null, use_tools: $("use-tools").checked };
}

function rememberSelection() {
  const url = new URL(location.href);
  if (state.activeId) url.searchParams.set("conversation", state.activeId);
  else url.searchParams.delete("conversation");
  history.replaceState(null, "", url);
}

function closeSidebar() {
  const restoreFocus = $("sidebar").contains(document.activeElement);
  $("sidebar").classList.remove("open");
  $("sidebar-toggle").setAttribute("aria-expanded", "false");
  $("sidebar-backdrop").hidden = true;
  $("sidebar").inert = window.matchMedia("(max-width: 700px)").matches;
  if (restoreFocus && $("sidebar").inert) $("sidebar-toggle").focus();
}

function renderMessages({ bottom = false } = {}) {
  const fragment = document.createDocumentFragment();
  for (const turn of state.turns) fragment.append(renderTurn(turn, notify));
  if (state.pending?.id === state.activeId) fragment.append(pendingMessage(state.pending.question));
  $("messages").replaceChildren(fragment);
  $("empty-state").hidden = Boolean(state.turns.length || state.pending?.id === state.activeId || state.historyLoading || !$("history-error").hidden);
  $("older-turns").hidden = !state.turns.length || state.turns[0].turn_number <= 1;
  if (bottom) $("messages-scroll").scrollTop = $("messages-scroll").scrollHeight;
  updateComposer();
  updateTitle();
}

function updateComposer() {
  const length = $("question").value.length;
  $("send").disabled = !$("question").value.trim() || Boolean(state.pending) || state.creating || state.historyLoading || !$("history-error").hidden;
  $("send").setAttribute("aria-label", state.pending ? "Answer in progress" : "Send message");
  $("character-count").hidden = length < 1700;
  $("character-count").textContent = `${length}/2000`;
  $("new-chat").disabled = state.creating || Boolean(state.pending && !state.pending.id);
}

function resizeComposer() {
  $("question").style.height = "auto";
  $("question").style.height = `${Math.min($("question").scrollHeight, 156)}px`;
  updateComposer();
}

async function selectSession(id) {
  if (state.pending && !state.pending.id) return;
  state.activeId = id;
  state.turns = [];
  state.turnCount = 0;
  state.failedRequest = null;
  $("request-error").hidden = true;
  $("history-error").hidden = true;
  $("question").value = "";
  resizeComposer();
  closeSidebar();
  rememberSelection();
  renderSessions();
  await loadHistory();
}

async function loadHistory({ older = false } = {}) {
  if (!state.activeId) return;
  const id = state.activeId;
  const version = ++state.historyVersion;
  const scrollHeight = $("messages-scroll").scrollHeight;
  const scrollTop = $("messages-scroll").scrollTop;
  const before = older ? state.turns[0]?.turn_number : undefined;
  state.historyLoading = true;
  $("older-turns").disabled = true;
  $("history-error").hidden = true;
  renderMessages();
  try {
    const data = await api.history(id, before);
    if (id !== state.activeId || version !== state.historyVersion) return;
    state.turns = older ? mergeBy(data.turns, state.turns, "turn_number").sort((a, b) => a.turn_number - b.turn_number) : data.turns;
    state.turnCount = data.turn_count;
    renderMessages({ bottom: !older });
    if (older) $("messages-scroll").scrollTop = scrollTop + $("messages-scroll").scrollHeight - scrollHeight;
  } catch (error) {
    if (id !== state.activeId || version !== state.historyVersion) return;
    showError("history-error", error);
    if (error.status === 404) await loadSessions();
  } finally {
    if (id === state.activeId && version === state.historyVersion) {
      state.historyLoading = false;
      $("older-turns").disabled = false;
      renderMessages();
    }
  }
}

async function newConversation() {
  if (state.creating || (state.pending && !state.pending.id)) return;
  state.creating = true;
  updateComposer();
  try {
    const data = await api.create();
    state.sessions.unshift({ conversation_id: data.conversation_id, title: "New conversation", turn_count: 0 });
    await selectSession(data.conversation_id);
    await loadSessions();
    $("question").focus();
  } catch (error) { notify(error.message); }
  finally { state.creating = false; updateComposer(); }
}

async function sendQuestion(event) {
  event.preventDefault();
  const question = $("question").value.trim();
  if (!question || state.pending || state.creating || state.historyLoading || !$("history-error").hidden) return;
  const options = activeOptions();
  let id = state.activeId;
  state.pending = { id, question };
  state.failedRequest = null;
  $("request-error").hidden = true;
  updateComposer();
  try {
    if (!id) {
      const created = await api.create();
      id = created.conversation_id;
      state.activeId = id;
      state.turns = [];
      state.turnCount = 0;
      state.pending.id = id;
      rememberSelection();
      state.sessions.unshift({ conversation_id: id, title: "New conversation", turn_count: 0 });
      renderSessions();
    }
    const turnCount = state.turnCount;
    $("question").value = "";
    resizeComposer();
    renderMessages({ bottom: true });
    const response = await api.ask({ question, conversation_id: id, ...options });
    if (state.activeId === id) {
      // Ignore any history read that started before this new turn was committed.
      state.historyVersion++;
      state.historyLoading = false;
      state.turns.push({ turn_number: turnCount + 1, created_at: new Date().toISOString(), response });
      state.turnCount = turnCount + 1;
    } else notify("Your answer is saved in its conversation.");
    await loadSessions();
  } catch (error) {
    if (state.activeId === id || !id) {
      state.failedRequest = { question, id, options };
      $("question").value = question;
      resizeComposer();
      const message = error.status === 409 ? "This conversation changed in another request. Reload its history before sending again."
        : error.status === 0 ? `${error.message} Reload the conversation before retrying; the answer may already have been saved.`
        : error.message;
      showError("request-error", new Error(message));
      $("retry-question").hidden = !id;
      $("retry-question").textContent = "Reload history";
      if ([0, 409, 404].includes(error.status) && id) {
        showError("history-error", new Error("Reload this conversation to check its latest saved state."));
      }
    } else notify(`The answer could not be saved: ${error.message}`);
  } finally {
    state.pending = null;
    renderMessages({ bottom: true });
  }
}

async function loadDocuments(append = false) {
  if (state.documentLoading) return;
  state.documentLoading = true;
  $("refresh-library").disabled = true;
  $("more-documents").disabled = true;
  try {
    const offset = append ? state.documentOffset : 0;
    const data = await api.documents(offset);
    const selected = $("document-scope").selectedOptions[0];
    const priorScope = selected?.value ? { id: selected.value, name: selected.textContent } : null;
    state.documents = append ? mergeBy(state.documents, data.documents, "document_id") : data.documents;
    state.documentOffset = offset + data.documents.length;
    state.documentTotal = data.total;
    state.libraryReady = true;
    $("library-error").hidden = true;
    renderDocuments();
    // Keep an explicitly selected document even when it sits beyond this library page.
    if (priorScope && !state.documents.some(doc => doc.document_id === priorScope.id)) {
      $("document-scope").append(new Option(priorScope.name, priorScope.id));
      $("document-scope").value = priorScope.id;
    }
  } catch (error) { state.libraryReady = false; showError("library-error", error); }
  finally {
    state.documentLoading = false;
    $("refresh-library").disabled = false;
    $("more-documents").disabled = false;
    setConnection();
  }
}

function renderDocuments() {
  const fragment = document.createDocumentFragment();
  const selected = $("document-scope").value;
  $("document-scope").replaceChildren(new Option("All documents", ""));
  for (const doc of state.documents) {
    const row = element("div", "document-item");
    const info = element("div", "document-info");
    const filename = element("strong", "", doc.filename);
    filename.title = doc.filename;
    info.append(filename, element("p", "document-meta", `${doc.chunk_count} passages${doc.page_count ? ` · ${doc.page_count} pages` : " · TXT"}`));
    row.title = `Indexed ${dateLabel(doc.indexed_at)}`;
    row.append(fileIcon(doc.filename), info, element("span", "indexed-dot"));
    fragment.append(row);
    $("document-scope").append(new Option(doc.filename, doc.document_id));
  }
  if (!state.documents.length) fragment.append(element("p", "muted", "No documents yet. Upload a file to start building your knowledge base."));
  $("document-list").replaceChildren(fragment);
  if (state.documents.some(doc => doc.document_id === selected)) $("document-scope").value = selected;
  $("document-count").textContent = state.documentTotal;
  $("more-documents").hidden = state.documentOffset >= state.documentTotal;
}

function setLibrary(open) {
  $("library").hidden = !open;
  $("library-toggle").setAttribute("aria-expanded", String(open));
  if (!open && $("library").contains(document.activeElement)) $("library-toggle").focus();
}

function openUpload() {
  if (!$("upload-dialog").open) $("upload-dialog").showModal();
}

function renderUploads() {
  const fragment = document.createDocumentFragment();
  for (const item of state.uploads) {
    const row = element("div", "queue-row");
    const info = element("div", "document-info");
    const statusClass = item.status === "Ready" ? " success" : item.status === "Failed" ? " error" : "";
    info.append(element("strong", "", item.file.name), element("p", `queue-status${statusClass}`, item.detail || item.status));
    row.append(fileIcon(item.file.name), info);
    if (item.status === "Ready") row.append(icon("check"));
    if (item.status === "Failed" && item.retryable) row.append(iconButton("upload", "Retry upload", () => {
      item.status = "Queued";
      item.detail = "Queued";
      processUploads();
    }));
    fragment.append(row);
  }
  $("upload-queue").replaceChildren(fragment);
  const ready = state.uploads.filter(item => item.status === "Ready").length;
  const active = state.uploads.some(item => ["Queued", "Indexing…"].includes(item.status));
  $("upload-summary").textContent = active ? "Indexing documents…" : `${ready} of ${state.uploads.length} files indexed`;
  $("upload-done").textContent = active ? "Continue in background" : "Done";
}

function queueFiles(files) {
  for (const file of Array.from(files).slice(0, 20)) {
    const item = { file, status: "Queued", detail: "Queued", retryable: true };
    if (!/\.(txt|pdf)$/i.test(file.name)) Object.assign(item, { status: "Failed", detail: "Choose a .txt or .pdf file.", retryable: false });
    else if (file.size > 5 * 1024 * 1024) Object.assign(item, { status: "Failed", detail: "File exceeds the 5 MiB limit.", retryable: false });
    else if (!file.size) Object.assign(item, { status: "Failed", detail: "This file is empty.", retryable: false });
    state.uploads.push(item);
  }
  if (files.length > 20) notify("Add up to 20 files at a time. Remaining files were not queued.");
  $("file-input").value = "";
  renderUploads();
  processUploads();
}

async function processUploads() {
  if (state.uploading) return;
  state.uploading = true;
  try {
    let item;
    while ((item = state.uploads.find(entry => entry.status === "Queued"))) {
      item.status = "Indexing…";
      item.detail = "Parsing and indexing…";
      renderUploads();
      try {
        const result = await api.index(item.file);
        item.status = "Ready";
        item.detail = `Indexed · ${result.chunk_count} passages`;
        await loadDocuments();
      } catch (error) {
        item.status = "Failed";
        item.detail = error.message;
        item.retryable = [0, 502, 503, 504].includes(error.status);
      }
      renderUploads();
    }
    const ready = state.uploads.filter(entry => entry.status === "Ready").length;
    notify(ready ? "Documents indexed. You can now ask questions about them." : "Upload finished. Review the file errors to continue.");
  } finally { state.uploading = false; }
}

function openSessionDialog(session, action) {
  state.dialogSession = session;
  state.dialogAction = action;
  const deleting = action === "delete";
  $("session-dialog-title").textContent = deleting ? "Delete conversation?" : "Rename conversation";
  $("session-dialog-description").textContent = deleting ? `“${session.title}” and its saved messages will be permanently deleted. Your indexed documents will stay in the library.` : "Give this conversation a name that is easy to find.";
  $("rename-label").hidden = deleting;
  $("rename-input").required = !deleting;
  $("rename-input").value = session.title;
  $("session-dialog-submit").textContent = deleting ? "Delete conversation" : "Save name";
  $("session-dialog-submit").className = `button ${deleting ? "button-danger" : "button-primary"}`;
  $("session-dialog-error").hidden = true;
  if (state.pending?.id === session.conversation_id && deleting) {
    notify("Wait for this conversation's answer to finish before deleting it.");
    return;
  }
  $("session-dialog").showModal();
  if (!deleting) { $("rename-input").focus(); $("rename-input").select(); }
  else $("session-dialog-cancel").focus();
}

async function submitSessionDialog(event) {
  event.preventDefault();
  const session = state.dialogSession;
  if (!session) return;
  $("session-dialog-submit").disabled = true;
  try {
    if (state.dialogAction === "delete") {
      await api.remove(session.conversation_id);
      if (state.activeId === session.conversation_id) {
        state.activeId = null;
        state.historyVersion++;
        state.historyLoading = false;
        state.turns = [];
        state.turnCount = 0;
        $("question").value = "";
        $("request-error").hidden = true;
        $("history-error").hidden = true;
        rememberSelection();
        renderMessages();
      }
      notify("Conversation deleted");
    } else {
      const summary = await api.rename(session.conversation_id, $("rename-input").value.trim());
      Object.assign(session, summary);
      renderSessions();
      notify("Conversation renamed");
    }
    $("session-dialog").close();
    await loadSessions();
  } catch (error) { showError("session-dialog-error", error); }
  finally { $("session-dialog-submit").disabled = false; }
}

$("new-chat").addEventListener("click", newConversation);
$("session-search").addEventListener("input", renderSessions);
$("more-sessions").addEventListener("click", () => loadSessions(true));
$("retry-sessions").addEventListener("click", () => loadSessions());
$("chat-form").addEventListener("submit", sendQuestion);
$("question").addEventListener("input", resizeComposer);
$("question").addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!$("send").disabled) $("chat-form").requestSubmit();
  }
});
$("retry-history").addEventListener("click", () => loadHistory());
$("older-turns").addEventListener("click", () => loadHistory({ older: true }));
$("retry-question").addEventListener("click", async () => {
  await loadHistory();
  if ($("history-error").hidden) $("request-error").hidden = true;
});
$("refresh-library").addEventListener("click", () => loadDocuments());
$("more-documents").addEventListener("click", () => loadDocuments(true));
$("library-toggle").addEventListener("click", () => setLibrary($("library").hidden));
$("library-close").addEventListener("click", () => setLibrary(false));
$("sidebar-toggle").addEventListener("click", () => {
  const open = $("sidebar").classList.toggle("open");
  $("sidebar").inert = !open;
  $("sidebar-toggle").setAttribute("aria-expanded", String(open));
  $("sidebar-backdrop").hidden = !open;
  if (open) $("new-chat").focus();
});
$("sidebar-backdrop").addEventListener("click", closeSidebar);
document.addEventListener("keydown", event => {
  if (event.key === "Escape") {
    closeSidebar();
    if (!$("upload-dialog").open && !$("session-dialog").open && window.matchMedia("(max-width: 980px)").matches) setLibrary(false);
  }
});
document.querySelectorAll("[data-upload]").forEach(button => button.addEventListener("click", openUpload));
$("upload-close").addEventListener("click", () => $("upload-dialog").close());
$("upload-done").addEventListener("click", () => $("upload-dialog").close());
$("file-input").addEventListener("change", event => queueFiles(event.target.files));
$("drop-zone").addEventListener("keydown", event => {
  if (["Enter", " "].includes(event.key)) { event.preventDefault(); $("file-input").click(); }
});
$("drop-zone").addEventListener("dragover", event => { event.preventDefault(); $("drop-zone").classList.add("dragging"); });
$("drop-zone").addEventListener("dragleave", () => $("drop-zone").classList.remove("dragging"));
$("drop-zone").addEventListener("drop", event => {
  event.preventDefault();
  $("drop-zone").classList.remove("dragging");
  queueFiles(event.dataTransfer.files);
});
window.addEventListener("dragover", event => event.preventDefault());
window.addEventListener("drop", event => event.preventDefault());
document.querySelectorAll(".suggestion").forEach(button => button.addEventListener("click", () => {
  $("question").value = button.dataset.prompt;
  if (button.dataset.tools) $("use-tools").checked = true;
  resizeComposer();
  $("question").focus();
}));
$("session-form").addEventListener("submit", submitSessionDialog);
for (const id of ["session-dialog-close", "session-dialog-cancel"]) $(id).addEventListener("click", () => $("session-dialog").close());

setLibrary(window.matchMedia("(min-width: 981px)").matches);
closeSidebar();
window.matchMedia("(max-width: 700px)").addEventListener("change", closeSidebar);
await Promise.all([loadSessions(), loadDocuments()]);
const initialId = new URL(location.href).searchParams.get("conversation");
if (initialId && /^[0-9a-f-]{36}$/i.test(initialId)) await selectSession(initialId);
else renderMessages();
