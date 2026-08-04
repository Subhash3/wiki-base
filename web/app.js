const API_HOST = window.location.hostname || "localhost";
const API_BASE_URL = `http://${API_HOST}:8000`;
const wikiBaseForm = document.querySelector("#wiki-base-form");
const nameInput = document.querySelector("#wiki-base-name");
const documentsInput = document.querySelector("#wiki-base-documents");
const addButton = document.querySelector("#add-button");
const refreshButton = document.querySelector("#refresh-button");
const messageElement = document.querySelector("#wiki-base-message");
const tableBody = document.querySelector("#wiki-base-table-body");
const wikiBaseSelect = document.querySelector("#wiki-base-select");
const retrievalModeSelect = document.querySelector("#retrieval-mode");
const chatMessages = document.querySelector("#chat-messages");
const chatForm = document.querySelector("#chat-form");
const questionInput = document.querySelector("#chat-question");
const sendButton = document.querySelector("#send-button");

let wikiBases = [];
let isQuerying = false;
const histories = new Map();

function getDocumentCount(wikiBase) {
  if (Array.isArray(wikiBase.documents)) {
    return wikiBase.documents.length;
  }
  return wikiBase.document_count ?? "—";
}

function renderWikiBases() {
  if (wikiBases.length === 0) {
    tableBody.innerHTML = `
      <tr><td colspan="6" class="empty">No wiki bases found.</td></tr>
    `;
  } else {
    tableBody.replaceChildren(
      ...wikiBases.map((wikiBase) => {
        const row = document.createElement("tr");
        row.innerHTML = `
          <td>${escapeHtml(wikiBase.id)}</td>
          <td>${escapeHtml(wikiBase.name)}</td>
          <td>${escapeHtml(String(getDocumentCount(wikiBase)))}</td>
          <td class="status">${escapeHtml(getRetrievalStatus(wikiBase, "lite"))}</td>
          <td class="status">${escapeHtml(getRetrievalStatus(wikiBase, "pro"))}</td>
          <td class="status">${escapeHtml(getRetrievalStatus(wikiBase, "facts"))}</td>
        `;
        return row;
      }),
    );
  }

  const selectedId = wikiBaseSelect.value;
  wikiBaseSelect.innerHTML = '<option value="">Select a wiki base</option>';
  for (const wikiBase of wikiBases) {
    const option = document.createElement("option");
    option.value = wikiBase.id;
    option.textContent = `${wikiBase.name} (Lite: ${getRetrievalStatus(wikiBase, "lite")}, Pro: ${getRetrievalStatus(wikiBase, "pro")}, Facts: ${getRetrievalStatus(wikiBase, "facts")})`;
    wikiBaseSelect.append(option);
  }
  wikiBaseSelect.value = wikiBases.some((item) => item.id === selectedId) ? selectedId : "";
  updateRetrievalModeAvailability();
}

function getRetrievalStatus(wikiBase, mode) {
  return wikiBase.retrieval_statuses?.[mode] ?? "unknown";
}

function isRetrievalReady(status) {
  return status === "ready" || status === "partially_failed";
}

function updateRetrievalModeAvailability() {
  const wikiBase = wikiBases.find((item) => item.id === wikiBaseSelect.value);
  for (const option of retrievalModeSelect.options) {
    const status = wikiBase ? getRetrievalStatus(wikiBase, option.value) : "unknown";
    option.disabled = Boolean(wikiBase) && !isRetrievalReady(status);
  }

  if (retrievalModeSelect.selectedOptions[0]?.disabled) {
    const available = [...retrievalModeSelect.options].find((option) => !option.disabled);
    if (available) {
      retrievalModeSelect.value = available.value;
    }
  }

  const selectedStatus = wikiBase
    ? getRetrievalStatus(wikiBase, retrievalModeSelect.value)
    : "unknown";
  sendButton.disabled =
    isQuerying || !wikiBase || !isRetrievalReady(selectedStatus);
}

function escapeHtml(value) {
  const element = document.createElement("span");
  element.textContent = value;
  return element.innerHTML;
}

async function readError(response) {
  try {
    const body = await response.json();
    return body.error?.message ?? body.detail ?? `Request failed with status ${response.status}.`;
  } catch {
    return `Request failed with status ${response.status}.`;
  }
}

async function fetchWikiBases() {
  const response = await fetch(`${API_BASE_URL}/wiki-bases`);
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  wikiBases = await response.json();
  renderWikiBases();
}

wikiBaseForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  addButton.disabled = true;
  messageElement.classList.remove("error");
  messageElement.textContent = "Uploading documents…";

  const formData = new FormData();
  formData.append("name", nameInput.value.trim());
  for (const file of documentsInput.files) {
    formData.append("documents", file);
  }

  try {
    const response = await fetch(`${API_BASE_URL}/wiki-bases`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      throw new Error(await readError(response));
    }

    const wikiBase = await response.json();
    await fetchWikiBases();
    wikiBaseForm.reset();
    messageElement.textContent = `Queued “${wikiBase.name}” for ingestion.`;
  } catch (error) {
    messageElement.classList.add("error");
    messageElement.textContent = error.message;
  } finally {
    addButton.disabled = false;
  }
});

refreshButton.addEventListener("click", async () => {
  refreshButton.disabled = true;
  messageElement.classList.remove("error");
  messageElement.textContent = "Refreshing…";

  try {
    await fetchWikiBases();
    messageElement.textContent = "Wiki bases refreshed.";
  } catch (error) {
    messageElement.classList.add("error");
    messageElement.textContent = error.message;
  } finally {
    refreshButton.disabled = false;
  }
});

wikiBaseSelect.addEventListener("change", () => {
  updateRetrievalModeAvailability();
  renderChat();
});
retrievalModeSelect.addEventListener("change", updateRetrievalModeAvailability);

function renderChat() {
  const wikiBaseId = wikiBaseSelect.value;
  const history = histories.get(wikiBaseId) ?? [];
  chatMessages.replaceChildren();

  if (!wikiBaseId) {
    chatMessages.innerHTML = '<p class="empty">Select a ready wiki base to start chatting.</p>';
    return;
  }

  if (history.length === 0) {
    chatMessages.innerHTML = '<p class="empty">No messages yet.</p>';
    return;
  }

  for (const message of history) {
    const element = document.createElement("div");
    element.className = `chat-message ${message.role}`;
    element.textContent = message.content;

    if (message.role === "assistant" && message.mode) {
      const mode = document.createElement("span");
      const usedFallback = message.retrievalStrategy === "vector_fallback";
      mode.className = `retrieval-mode ${message.mode}${usedFallback ? " fallback" : ""}`;
      const modeLabel =
        message.mode === "facts" ? "Facts" : message.mode === "pro" ? "Pro" : "Lite";
      mode.textContent = usedFallback
        ? `${modeLabel} · Vector fallback`
        : message.mode === "facts"
          ? "Facts · Graph traversal"
          : message.mode === "pro"
            ? "Pro · PageRank"
            : "Lite · Vector";
      element.prepend(mode);
    }

    if (message.citations?.length) {
      const citations = document.createElement("div");
      citations.className = "citations";
      citations.textContent = message.citations
        .map((citation) => {
          const location = citation.page
            ? `page ${citation.page}`
            : citation.slide
              ? `slide ${citation.slide}`
              : citation.section ?? "source";
          return `${citation.document_name} — ${location}`;
        })
        .join("; ");
      element.append(citations);
    }

    chatMessages.append(element);
  }
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const wikiBaseId = wikiBaseSelect.value;
  const question = questionInput.value.trim();
  const mode = retrievalModeSelect.value;
  if (!wikiBaseId || !question) {
    return;
  }

  const history = histories.get(wikiBaseId) ?? [];
  const requestHistory = history.map(({ role, content }) => ({ role, content }));
  history.push({ role: "user", content: question });
  histories.set(wikiBaseId, history);
  questionInput.value = "";
  isQuerying = true;
  sendButton.disabled = true;
  renderChat();

  try {
    const response = await fetch(`${API_BASE_URL}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        wiki_base_id: wikiBaseId,
        question,
        history: requestHistory,
        limit: 5,
        mode,
      }),
    });
    if (!response.ok) {
      throw new Error(await readError(response));
    }

    const result = await response.json();
    history.push({
      role: "assistant",
      content: result.answer,
      citations: result.citations ?? [],
      mode: result.mode ?? mode,
      retrievalStrategy: result.retrieval_strategy,
    });
  } catch (error) {
    history.push({
      role: "assistant",
      content: `Error: ${error.message}`,
    });
  } finally {
    isQuerying = false;
    updateRetrievalModeAvailability();
    questionInput.focus();
    renderChat();
  }
});

renderWikiBases();
renderChat();
fetchWikiBases().catch((error) => {
  messageElement.classList.add("error");
  messageElement.textContent = `Could not load wiki bases from ${API_BASE_URL}: ${error.message}`;
});
