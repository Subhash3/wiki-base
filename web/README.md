# Wiki Base test web app

This is a standalone, dependency-free test client. It is not bundled into or served by the
FastAPI application.

Start any static file server in this directory. For example:

```bash
cd web
python3 -m http.server 8080
```

Then open `http://localhost:8080`.

The client uses the web page's hostname and API port 8000. For example, a page loaded from
`http://127.0.0.1:8080` calls `http://127.0.0.1:8000`.

Wiki bases are loaded from the API. Chat history is sent with each question but is kept only
in memory and is lost when the page is reloaded.

The chat offers three answering modes:

- **Lite** uses cosine-similarity chunk retrieval.
- **Pro** uses Personalized PageRank over the wiki base's ready graphs.
- **Facts** uses bounded graph traversal and ranked canonical facts.

The selected mode is sent as the `mode` field of `POST /query`. Assistant messages show both
the requested mode and the `retrieval_strategy` actually used. A graph request that cannot
produce ranked chunks is labeled as a vector fallback.

Wiki-base responses expose a `retrieval_statuses` map. The table displays Lite, Pro, and
Facts readiness separately, and unavailable modes are disabled for the selected wiki base.

The client expects these API operations:

- `POST /wiki-bases`
- `GET /wiki-bases`
- `GET /wiki-bases/{id}/status`
- `GET /wiki-bases/{id}/graph`
- `GET /wiki-bases/{id}/graph/nodes/{node_id}`
- `GET /wiki-bases/{id}/graph/nodes/{node_id}/facts`
- `POST /query`

The development API accepts browser requests from localhost, `127.0.0.1`, and `0.0.0.0` on
any port. Additional non-local origins can be configured with `WIKI_BASE_CORS_ORIGINS`.

## Reusable 3D graph layout

[`src/force-graph-3d.ts`](src/force-graph-3d.ts) provides a typed, renderer-independent
3D force layout backed by `d3-force-3d`. It clones input data before the simulation mutates
it and supports live updates, reheating, deterministic manual settling, and tick/end hooks.

```ts
import { createForceGraph3D } from "./src/force-graph-3d";

const layout = createForceGraph3D(
  {
    nodes: [{ id: "a" }, { id: "b" }],
    links: [{ source: "a", target: "b" }],
  },
  { onTick: ({ nodes, links }) => renderScene(nodes, links) },
);

// Call when the view is removed.
layout.stop();
```

Install and type-check the module with `npm install` and `npm run check` from `web/`.

Open `/viz/` from the static web server to select a graph-ready wiki base and explore its
merged knowledge graph. Drag to orbit, use the mouse wheel to zoom, and hover over nodes to
see entity names.
