CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS wiki_bases (
    id uuid PRIMARY KEY,
    name text NOT NULL CHECK (length(trim(name)) > 0),
    status text NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'processing', 'ready', 'partially_failed', 'failed')),
    embedding_model text,
    embedding_dimensions integer CHECK (embedding_dimensions > 0),
    created_at timestamptz NOT NULL DEFAULT now(),
    started_at timestamptz,
    completed_at timestamptz
);

CREATE TABLE IF NOT EXISTS documents (
    id uuid PRIMARY KEY,
    wiki_base_id uuid NOT NULL REFERENCES wiki_bases(id) ON DELETE CASCADE,
    name text NOT NULL CHECK (length(trim(name)) > 0),
    media_type text NOT NULL,
    content_checksum text NOT NULL,
    status text NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'processing', 'ready', 'failed')),
    parser_type text,
    parser_version text,
    page_count integer CHECK (page_count >= 0),
    error_code text,
    error_message text,
    created_at timestamptz NOT NULL DEFAULT now(),
    started_at timestamptz,
    completed_at timestamptz,
    UNIQUE (wiki_base_id, content_checksum)
);

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

CREATE INDEX IF NOT EXISTS documents_wiki_base_id_idx ON documents (wiki_base_id);
CREATE INDEX IF NOT EXISTS ingestion_jobs_status_idx ON ingestion_jobs (status, queued_at);
CREATE INDEX IF NOT EXISTS chunks_wiki_base_id_idx ON chunks (wiki_base_id);
CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_search_vector_idx ON chunks USING gin (search_vector);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
