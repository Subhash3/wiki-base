# Wiki Base test web app

This is a standalone, dependency-free test client. It is not bundled into or served by the
FastAPI application.

Start any static file server in this directory. For example:

```bash
cd web
python3 -m http.server 8080
```

Then open `http://localhost:8080`.

The API URL is defined at the top of `app.js` and defaults to `http://localhost:8000`.

Created wiki-base IDs are retained in browser `localStorage` because the API does not yet
provide a list endpoint. Chat history is sent with each question but is kept only in memory
and is lost when the page is reloaded.

The client expects these API operations:

- `POST /wiki-bases`
- `GET /wiki-bases/{id}/status`
- `POST /query`

The default API configuration allows `http://localhost:8080` and
`http://127.0.0.1:8080` through CORS. Change `WIKI_BASE_CORS_ORIGINS` when serving this
client from a different origin.
