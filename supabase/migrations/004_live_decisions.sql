-- Live decision feed for the Realtime dashboard tab.
--
-- The API best-effort inserts one row per scored transaction (fire-and-forget,
-- so /score latency is unaffected). The frontend subscribes to INSERTs on this
-- table via Supabase Realtime and streams them into the Live Feed tab. Rows are
-- non-PII (no card number, IP, or key) — only the decision and coarse features.

CREATE TABLE IF NOT EXISTS live_decisions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trans_id TEXT NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('APPROVE', 'REVIEW', 'DECLINE')),
    fraud_score DOUBLE PRECISION,
    amount DOUBLE PRECISION,
    merchant TEXT,
    category TEXT,
    hour INTEGER,
    layer TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_live_decisions_created_at ON live_decisions(created_at DESC);

-- Stream INSERTs to subscribed browsers. (No-op error if already a member.)
ALTER PUBLICATION supabase_realtime ADD TABLE live_decisions;
