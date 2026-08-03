CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS wiki_bases (
    id uuid PRIMARY KEY,
    name text NOT NULL CHECK (length(trim(name)) > 0),
    embedding_model text,
    embedding_dimensions integer CHECK (embedding_dimensions > 0),
    created_at timestamptz NOT NULL DEFAULT now(),
    started_at timestamptz,
    completed_at timestamptz
);

ALTER TABLE wiki_bases DROP COLUMN IF EXISTS status;

CREATE TABLE IF NOT EXISTS documents (
    id uuid PRIMARY KEY,
    wiki_base_id uuid NOT NULL REFERENCES wiki_bases(id) ON DELETE CASCADE,
    name text NOT NULL CHECK (length(trim(name)) > 0),
    media_type text NOT NULL,
    content_checksum text NOT NULL,
    parser_type text,
    parser_version text,
    page_count integer CHECK (page_count >= 0),
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (wiki_base_id, content_checksum)
);

ALTER TABLE documents
    DROP COLUMN IF EXISTS status,
    DROP COLUMN IF EXISTS started_at,
    DROP COLUMN IF EXISTS completed_at,
    DROP COLUMN IF EXISTS error_code,
    DROP COLUMN IF EXISTS error_message;

CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id uuid PRIMARY KEY,
    wiki_base_id uuid NOT NULL REFERENCES wiki_bases(id) ON DELETE CASCADE,
    document_id uuid NOT NULL UNIQUE REFERENCES documents(id) ON DELETE CASCADE,
    status text NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'processing', 'ready', 'failed')),
    staging_reference text,
    attempts integer NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    leased_until timestamptz,
    heartbeat_at timestamptz,
    error_code text,
    error_message text,
    queued_at timestamptz NOT NULL DEFAULT now(),
    started_at timestamptz,
    completed_at timestamptz
);

CREATE TABLE IF NOT EXISTS graph_indexing_jobs (
    document_id uuid PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    status text NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'processing', 'ready', 'failed')),
    error_message text,
    queued_at timestamptz NOT NULL DEFAULT now(),
    started_at timestamptz,
    completed_at timestamptz
);

ALTER TABLE graph_indexing_jobs
    DROP COLUMN IF EXISTS output_path,
    DROP COLUMN IF EXISTS extraction_model,
    DROP COLUMN IF EXISTS index_version;

CREATE TABLE IF NOT EXISTS document_graphs (
    document_id uuid PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    graph jsonb NOT NULL CHECK (jsonb_typeof(graph) = 'object'),
    extraction_model text NOT NULL,
    index_version text NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id uuid PRIMARY KEY,
    wiki_base_id uuid NOT NULL REFERENCES wiki_bases(id) ON DELETE CASCADE,
    document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content text NOT NULL,
    embedding_content text NOT NULL,
    embedding vector(1024),
    ordinal integer NOT NULL CHECK (ordinal >= 0),
    token_count integer NOT NULL CHECK (token_count > 0),
    page_number integer CHECK (page_number > 0),
    slide_number integer CHECK (slide_number > 0),
    section text,
    heading text,
    caption text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(heading, '') || ' ' || content)
    ) STORED,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (document_id, ordinal)
);

-- Keep an existing development database aligned with the selected embedding model.
ALTER TABLE chunks
    ALTER COLUMN embedding TYPE vector(1024)
    USING embedding::vector(1024);

INSERT INTO graph_indexing_jobs (document_id)
SELECT document.id
FROM documents AS document
JOIN ingestion_jobs AS job ON job.document_id = document.id
WHERE job.status = 'ready'
ON CONFLICT (document_id) DO NOTHING;

-- File-backed ready jobs have no database artifact and must be indexed again.
UPDATE graph_indexing_jobs AS job
SET status = 'queued', queued_at = now(), started_at = NULL,
    completed_at = NULL, error_message = NULL
WHERE job.status = 'ready'
  AND NOT EXISTS (
      SELECT 1
      FROM document_graphs AS graph
      WHERE graph.document_id = job.document_id
  );

CREATE INDEX IF NOT EXISTS documents_wiki_base_id_idx ON documents (wiki_base_id);
CREATE INDEX IF NOT EXISTS ingestion_jobs_status_idx ON ingestion_jobs (status, queued_at);
CREATE INDEX IF NOT EXISTS graph_indexing_jobs_status_idx
    ON graph_indexing_jobs (status, queued_at);
CREATE INDEX IF NOT EXISTS chunks_wiki_base_id_idx ON chunks (wiki_base_id);
CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_search_vector_idx ON chunks USING gin (search_vector);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
