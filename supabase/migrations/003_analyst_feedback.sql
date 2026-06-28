-- Standalone analyst-feedback sink for the live demo.
--
-- The `feedback` table in 001 requires a decision_id FK into a persisted
-- `decisions` row. The deployed demo scores statelessly (no decision is
-- persisted), so it has no decision_id to reference. This table captures the
-- ✓/✗ analyst labels the API's POST /feedback endpoint actually sends, with no
-- FK dependency, so the active-learning queue works end-to-end on free tier.
-- The richer FK-linked `feedback` table remains for a future stateful pipeline.

CREATE TABLE IF NOT EXISTS analyst_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trans_id TEXT NOT NULL,
    decision TEXT,
    fraud_score DOUBLE PRECISION,
    label TEXT NOT NULL CHECK (label IN ('fraud', 'legit')),
    note TEXT,
    model_version TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_analyst_feedback_created_at ON analyst_feedback(created_at);
CREATE INDEX IF NOT EXISTS idx_analyst_feedback_label ON analyst_feedback(label);

-- Non-sensitive demo data; the API writes with the service_role key. Disable
-- RLS so reads work without per-row policies (Supabase enables it by default).
ALTER TABLE analyst_feedback DISABLE ROW LEVEL SECURITY;
