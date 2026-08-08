import { createForceGraph3D } from "../src/force-graph-3d.js";

const API_HOST = window.location.hostname || "localhost";
const API_BASE_URL = `http://${API_HOST}:8000`;
const select = document.querySelector("#wiki-base-select");
const canvas = document.querySelector("#graph");
const viewport = document.querySelector("#viewport");
const message = document.querySelector("#message");
const tooltip = document.querySelector("#tooltip");
const stats = document.querySelector("#stats");
const nodePanel = document.querySelector("#node-panel");
const nodeName = document.querySelector("#node-name");
const nodeInfo = document.querySelector("#node-info");
const nodeFacts = document.querySelector("#node-facts");
const context = canvas.getContext("2d");

let layout;
let graph = { nodes: [], links: [] };
let projectedNodes = [];
let projectedLinks = [];
let yaw = -0.35;
let pitch = 0.2;
let zoom = 1;
let panX = 0;
let panY = 0;
let dragStart;
let selectedNodeId;
let nodeRequest = 0;

function showMessage(text, error = false) {
  message.textContent = text;
  message.classList.toggle("error", error);
  message.hidden = !text;
}

async function readJson(response) {
  if (response.ok) return response.json();
  const body = await response.json().catch(() => ({}));
  throw new Error(body.error?.message ?? `Request failed with status ${response.status}.`);
}

function graphData(payload) {
  const nodesById = new Map();
  for (const item of payload.nodes ?? []) {
    nodesById.set(item.id, {
      id: item.id,
      name: item.name,
      kind: item.kind ?? item.type ?? "entity",
      linkCount: 0,
    });
  }
  const links = (payload.edges ?? [])
    .filter((edge) => nodesById.has(edge.source) && nodesById.has(edge.target))
    .map((edge) => ({
      source: edge.source, target: edge.target, label: edge.relation, kind: "fact",
    }));
  for (const edge of payload.synonyms ?? []) links.push({
    source: edge.source, target: edge.target,
    label: `synonym · ${edge.similarity.toFixed(2)}`, kind: "synonym",
  });
  for (const link of links) {
    const source = nodesById.get(link.source);
    const target = nodesById.get(link.target);
    if (!source || !target || isDocumentNode(source) || isDocumentNode(target)) continue;
    source.linkCount += 1;
    target.linkCount += 1;
  }
  const nodes = [...nodesById.values()].map((node) => ({
    ...node,
    // Square-root scaling keeps hubs manageable without collapsing their size
    // into the same hard cap as moderately connected nodes.
    radius: isDocumentNode(node) ? 4 : 3.5 + Math.sqrt(node.linkCount) * 1.5,
  }));
  return { nodes, links };
}

function resize() {
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(viewport.clientWidth * ratio);
  canvas.height = Math.round(viewport.clientHeight * ratio);
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  draw();
}

function project(node) {
  const x = node.x ?? 0, y = node.y ?? 0, z = node.z ?? 0;
  const cosY = Math.cos(yaw), sinY = Math.sin(yaw);
  const cosP = Math.cos(pitch), sinP = Math.sin(pitch);
  const rx = x * cosY - z * sinY;
  const rz = x * sinY + z * cosY;
  const ry = y * cosP - rz * sinP;
  const depth = y * sinP + rz * cosP;
  const focal = 650;
  const scale = zoom * focal / Math.max(180, focal + depth);
  return {
    node,
    x: viewport.clientWidth / 2 + panX + rx * scale,
    y: viewport.clientHeight / 2 + panY + ry * scale,
    depth,
    rx,
    ry,
    scale,
  };
}

function draw() {
  const width = viewport.clientWidth, height = viewport.clientHeight;
  context.clearRect(0, 0, width, height);
  projectedNodes = [];
  projectedLinks = [];
  if (!graph.nodes.length) return;
  const points = new Map(graph.nodes.map((node) => [node, project(node)]));
  for (const link of graph.links) {
    const source = points.get(link.source), target = points.get(link.target);
    if (!source || !target) continue;
    projectedLinks.push({ link, source, target });
    context.beginPath(); context.moveTo(source.x, source.y); context.lineTo(target.x, target.y);
    context.strokeStyle = link.kind === "synonym" ? "#b76be0aa" : "#63779d70";
    context.setLineDash(link.kind === "synonym" ? [4, 5] : []);
    context.lineWidth = link.kind === "synonym" ? 1.2 : 0.8;
    context.stroke();
  }
  context.setLineDash([]);
  projectedNodes = [...points.values()].sort((a, b) => b.depth - a.depth);
  const maxLinkCount = Math.max(1, ...graph.nodes
    .filter((node) => !isDocumentNode(node))
    .map((node) => node.linkCount));
  for (const point of projectedNodes) {
    const radius = screenRadius(point);
    context.beginPath(); context.arc(point.x, point.y, radius, 0, Math.PI * 2);
    context.fillStyle = nodeColor(point.node, maxLinkCount); context.fill();
    if (point.node.id === selectedNodeId) {
      context.beginPath(); context.arc(point.x, point.y, radius + 4, 0, Math.PI * 2);
      context.strokeStyle = "#fff"; context.lineWidth = 1.5; context.stroke();
    }
  }
}

async function loadWikiBases() {
  const bases = await readJson(await fetch(`${API_BASE_URL}/wiki-bases`));
  for (const base of bases) {
    const option = document.createElement("option");
    option.value = base.id;
    option.textContent = base.name;
    option.disabled = !["ready", "partially_failed"].includes(base.retrieval_statuses?.pro);
    select.append(option);
  }
}

async function loadGraph(id) {
  layout?.stop();
  clearNodeSelection();
  graph = { nodes: [], links: [] };
  stats.textContent = "";
  draw();
  if (!id) { showMessage("Select a wiki base to view its graph."); return; }
  showMessage("Loading merged graph…");
  try {
    graph = graphData(await readJson(await fetch(`${API_BASE_URL}/wiki-bases/${id}/graph`)));
    stats.textContent = `${graph.nodes.length.toLocaleString()} nodes · ${graph.links.length.toLocaleString()} links`;
    showMessage(graph.nodes.length ? "" : "This graph has no entities.");
    layout = createForceGraph3D(graph, {
      chargeStrength: -100,
      linkDistance: 42,
      collisionRadius: (node) => node.radius + 1,
      onTick: draw,
      onEnd: draw,
    });
    graph = layout.graph;
  } catch (error) {
    showMessage(error instanceof Error ? error.message : "Could not load graph.", true);
  }
}

select.addEventListener("change", () => loadGraph(select.value));
document.querySelector("#close-panel").addEventListener("click", clearNodeSelection);
document.querySelector("#reset-view").addEventListener("click", () => {
  yaw = -0.35; pitch = 0.2; zoom = 1; panX = 0; panY = 0; draw();
});
canvas.addEventListener("pointerdown", (event) => {
  const bounds = canvas.getBoundingClientRect();
  const x = event.clientX - bounds.left;
  const y = event.clientY - bounds.top;
  const point = findNodeAt(x, y);
  if (event.ctrlKey) {
    dragStart = { type: "orbit", x: event.clientX, y: event.clientY, yaw, pitch };
  } else if (point) {
    selectNode(point.node);
    dragStart = {
      type: "node",
      x: event.clientX,
      y: event.clientY,
      point,
    };
    point.node.fx = point.node.x;
    point.node.fy = point.node.y;
    point.node.fz = point.node.z;
    layout?.simulation.alphaTarget(0.25).restart();
  } else {
    dragStart = { type: "pan", x: event.clientX, y: event.clientY, panX, panY };
  }
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointermove", (event) => {
  if (dragStart?.type === "node") {
    moveNodeInViewPlane(
      dragStart.point,
      event.clientX - dragStart.x,
      event.clientY - dragStart.y,
    );
    draw(); return;
  }
  if (dragStart?.type === "orbit") {
    yaw = dragStart.yaw + (event.clientX - dragStart.x) * 0.008;
    pitch = Math.max(-1.45, Math.min(1.45, dragStart.pitch + (event.clientY - dragStart.y) * 0.008));
    draw(); return;
  }
  if (dragStart?.type === "pan") {
    panX = dragStart.panX + event.clientX - dragStart.x;
    panY = dragStart.panY + event.clientY - dragStart.y;
    draw(); return;
  }
  const bounds = canvas.getBoundingClientRect();
  const x = event.clientX - bounds.left, y = event.clientY - bounds.top;
  const nodeHit = findNodeAt(x, y);
  const edgeHit = nodeHit ? undefined : projectedLinks
    .map((edge) => ({ edge, distance: distanceToSegment(x, y, edge.source, edge.target) }))
    .filter(({ distance }) => distance < 6)
    .sort((first, second) => first.distance - second.distance)[0]?.edge;
  tooltip.hidden = !nodeHit && !edgeHit;
  if (nodeHit || edgeHit) {
    tooltip.textContent = nodeHit
      ? `${nodeHit.node.name} · ${nodeHit.node.linkCount} ${nodeHit.node.linkCount === 1 ? "link" : "links"}`
      : `${nodeNameOf(edgeHit.link.source)} — ${edgeHit.link.label} → ${nodeNameOf(edgeHit.link.target)}`;
    tooltip.style.left = `${x + 14}px`;
    tooltip.style.top = `${y + 14}px`;
  }
});
canvas.addEventListener("pointerup", finishDrag);
canvas.addEventListener("pointercancel", finishDrag);
canvas.addEventListener("pointerleave", () => { tooltip.hidden = true; });
canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  const bounds = canvas.getBoundingClientRect();
  const pointerX = event.clientX - bounds.left;
  const pointerY = event.clientY - bounds.top;
  const nextZoom = Math.max(0.02, Math.min(5, zoom * Math.exp(-event.deltaY * 0.001)));
  const factor = nextZoom / zoom;
  const originX = viewport.clientWidth / 2 + panX;
  const originY = viewport.clientHeight / 2 + panY;
  panX = pointerX - viewport.clientWidth / 2 - factor * (pointerX - originX);
  panY = pointerY - viewport.clientHeight / 2 - factor * (pointerY - originY);
  zoom = nextZoom;
  draw();
}, { passive: false });
window.addEventListener("resize", resize);
window.addEventListener("beforeunload", () => layout?.stop());

resize();
loadWikiBases().catch((error) => showMessage(error.message, true));

function distanceToSegment(x, y, start, end) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  if (dx === 0 && dy === 0) return Math.hypot(x - start.x, y - start.y);
  const amount = Math.max(0, Math.min(1,
    ((x - start.x) * dx + (y - start.y) * dy) / (dx * dx + dy * dy),
  ));
  return Math.hypot(x - (start.x + amount * dx), y - (start.y + amount * dy));
}

function nodeNameOf(node) {
  if (typeof node === "object" && node !== null) return node.name;
  return graph.nodes.find((candidate) => candidate.id === node)?.name ?? node;
}

function findNodeAt(x, y) {
  return [...projectedNodes].reverse().find(
    (point) => Math.hypot(point.x - x, point.y - y) < Math.max(9, screenRadius(point) + 3),
  );
}

function screenRadius(point) {
  // Clamp the camera scale, not the final radius, to preserve degree ratios.
  return point.node.radius * Math.max(0.55, Math.min(2.5, point.scale));
}

function isDocumentNode(node) {
  return String(node.kind).toLowerCase() === "document";
}

function nodeColor(node, maxLinkCount) {
  if (isDocumentNode(node)) return "hsl(220 12% 55%)";
  const intensity = Math.log1p(node.linkCount) / Math.log1p(maxLinkCount);
  const hue = 222 - intensity * 28;
  const saturation = 48 + intensity * 47;
  const lightness = 38 + intensity * 30;
  return `hsl(${hue} ${saturation}% ${lightness}%)`;
}

function moveNodeInViewPlane(point, screenDx, screenDy) {
  const rx = point.rx + screenDx / point.scale;
  const ry = point.ry + screenDy / point.scale;
  const cosP = Math.cos(pitch), sinP = Math.sin(pitch);
  const y = ry * cosP + point.depth * sinP;
  const rz = -ry * sinP + point.depth * cosP;
  const cosY = Math.cos(yaw), sinY = Math.sin(yaw);
  point.node.x = point.node.fx = rx * cosY + rz * sinY;
  point.node.y = point.node.fy = y;
  point.node.z = point.node.fz = -rx * sinY + rz * cosY;
}

function finishDrag() {
  if (dragStart?.type === "node") {
    dragStart.point.node.fx = null;
    dragStart.point.node.fy = null;
    dragStart.point.node.fz = null;
    layout?.simulation.alphaTarget(0);
  }
  dragStart = undefined;
}

async function selectNode(node) {
  selectedNodeId = node.id;
  draw();
  nodePanel.hidden = false;
  nodeName.textContent = node.name;
  nodeInfo.innerHTML = '<p class="loading">Loading node information…</p>';
  nodeFacts.innerHTML = '<p class="loading">Loading facts…</p>';
  const request = ++nodeRequest;
  try {
    const [info, facts] = await Promise.all([
      readJson(await fetch(`${API_BASE_URL}/wiki-bases/${select.value}/graph/nodes/${node.id}`)),
      readJson(await fetch(`${API_BASE_URL}/wiki-bases/${select.value}/graph/nodes/${node.id}/facts`)),
    ]);
    if (request !== nodeRequest) return;
    renderNodeInfo(info);
    renderNodeFacts(facts);
  } catch (error) {
    if (request !== nodeRequest) return;
    const text = escapeHtml(error instanceof Error ? error.message : "Could not load node details.");
    nodeInfo.innerHTML = `<p class="widget-error">${text}</p>`;
    nodeFacts.innerHTML = `<p class="widget-error">${text}</p>`;
  }
}

function clearNodeSelection() {
  selectedNodeId = undefined;
  nodeRequest += 1;
  nodePanel.hidden = true;
  draw();
}

function renderNodeInfo(info) {
  const documents = info.documents.length
    ? `<ul class="document-list">${info.documents.map((document) =>
      `<li>${escapeHtml(document.name)}<small>${document.chunk_count} supporting ${document.chunk_count === 1 ? "chunk" : "chunks"}</small></li>`
    ).join("")}</ul>`
    : '<p class="empty-widget">No source documents available.</p>';
  nodeInfo.innerHTML = `
    <div class="metrics">
      <div class="metric"><strong>${info.link_count}</strong><span>Links</span></div>
      <div class="metric"><strong>${info.fact_count}</strong><span>Facts</span></div>
      <div class="metric"><strong>${info.document_count}</strong><span>Documents</span></div>
    </div>
    <strong>Source documents</strong>${documents}
  `;
}

function renderNodeFacts(result) {
  if (!result.facts.length) {
    nodeFacts.innerHTML = '<p class="empty-widget">No direct facts found.</p>';
    return;
  }
  nodeFacts.innerHTML = `<ul class="fact-list">${result.facts.map((fact) => `
    <li><strong>${escapeHtml(fact.subject)}</strong> ${escapeHtml(fact.relation)} <strong>${escapeHtml(fact.object)}</strong>
      <small>${fact.document_names.length ? escapeHtml(fact.document_names.join(", ")) : "No document name"} · ${fact.evidence_count} ${fact.evidence_count === 1 ? "evidence" : "evidence items"}</small>
    </li>`).join("")}</ul>`;
}

function escapeHtml(value) {
  const element = document.createElement("span");
  element.textContent = String(value);
  return element.innerHTML;
}
