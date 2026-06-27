-- LLM copilot document store with pgvector embeddings
-- Embedding dim 384 matches sentence-transformers all-MiniLM-L6-v2

CREATE TABLE IF NOT EXISTS copilot_documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content TEXT NOT NULL,
    embedding vector(384),
    document_type TEXT NOT NULL,  -- 'fraud_ring', 'rule', 'transaction', 'alert'
    source_id TEXT,               -- references the originating entity id
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_copilot_embedding
    ON copilot_documents
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
