const API_BASE_URL = "http://localhost:8000";
const STORAGE_KEY = "wiki-base-tester-items";

const wikiBaseForm = document.querySelector("#wiki-base-form");
const nameInput = document.querySelector("#wiki-base-name");
const documentsInput = document.querySelector("#wiki-base-documents");
const addButton = document.querySelector("#add-button");
const refreshButton = document.querySelector("#refresh-button");
const messageElement = document.querySelector("#wiki-base-message");
const tableBody = document.querySelector("#wiki-base-table-body");
const wikiBaseSelect = document.querySelector("#wiki-base-select");
const chatMessages = document.querySelector("#chat-messages");
const chatForm = document.querySelector("#chat-form");
const questionInput = document.querySelector("#chat-question");
const sendButton = document.querySelector("#send-button");

let wikiBases = loadWikiBases();
const histories = new Map();

function loadWikiBases() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) ?? [];
  } catch {
    return [];
  }
}

function saveWikiBases() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(wikiBases));
}

function getDocumentCount(wikiBase) {
  if (Array.isArray(wikiBase.documents)) {
    return wikiBase.documents.length;
  }
  return wikiBase.document_count ?? "—";
}

function renderWikiBases() {
  if (wikiBases.length === 0) {
    tableBody.innerHTML = `
      <tr><td colspan="4" class="empty">No wiki bases created in this browser yet.</td></tr>
    `;
  } else {
    tableBody.replaceChildren(
      ...wikiBases.map((wikiBase) => {
        const row = document.createElement("tr");
        row.innerHTML = `
          <td>${escapeHtml(wikiBase.id)}</td>
          <td>${escapeHtml(wikiBase.name)}</td>
          <td>${escapeHtml(String(getDocumentCount(wikiBase)))}</td>
          <td class="status">${escapeHtml(wikiBase.status ?? "unknown")}</td>
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
    option.textContent = `${wikiBase.name} (${wikiBase.status ?? "unknown"})`;
    wikiBaseSelect.append(option);
  }
  wikiBaseSelect.value = wikiBases.some((item) => item.id === selectedId) ? selectedId : "";
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
    wikiBases = [wikiBase, ...wikiBases.filter((item) => item.id !== wikiBase.id)];
    saveWikiBases();
    renderWikiBases();
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
    const results = await Promise.allSettled(
      wikiBases.map(async (wikiBase) => {
        const response = await fetch(`${API_BASE_URL}/wiki-bases/${wikiBase.id}/status`);
        if (!response.ok) {
          throw new Error(await readError(response));
        }
        return response.json();
      }),
    );

    let failures = 0;
    wikiBases = wikiBases.map((current, index) => {
      const result = results[index];
      if (result.status === "fulfilled") {
        return { ...current, ...result.value };
      }
      failures += 1;
      return current;
    });

    saveWikiBases();
    renderWikiBases();
    messageElement.textContent = failures
      ? `Refresh completed with ${failures} failed request(s).`
      : "Statuses refreshed.";
    messageElement.classList.toggle("error", failures > 0);
  } finally {
    refreshButton.disabled = false;
  }
});

wikiBaseSelect.addEventListener("change", renderChat);

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
  if (!wikiBaseId || !question) {
    return;
  }

  const history = histories.get(wikiBaseId) ?? [];
  const requestHistory = history.map(({ role, content }) => ({ role, content }));
  history.push({ role: "user", content: question });
  histories.set(wikiBaseId, history);
  questionInput.value = "";
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
    });
  } catch (error) {
    history.push({
      role: "assistant",
      content: `Error: ${error.message}`,
    });
  } finally {
    sendButton.disabled = false;
    questionInput.focus();
    renderChat();
  }
});

renderWikiBases();
renderChat();
