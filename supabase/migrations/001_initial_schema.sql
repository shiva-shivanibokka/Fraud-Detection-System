-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- Raw transaction inputs
CREATE TABLE IF NOT EXISTS transactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    card_number TEXT NOT NULL,
    amount NUMERIC(12,2) NOT NULL,
    merchant TEXT,
    category TEXT,
    state TEXT,
    hour INTEGER CHECK (hour >= 0 AND hour <= 23),
    geo_distance_km NUMERIC(10,2),
    device_id TEXT,
    ip_prefix TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Scoring decisions
CREATE TABLE IF NOT EXISTS decisions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id UUID NOT NULL REFERENCES transactions(id) ON DELETE CASCADE,
    decision TEXT NOT NULL CHECK (decision IN ('APPROVE', 'REVIEW', 'DECLINE')),
    fraud_score NUMERIC(6,4) NOT NULL CHECK (fraud_score >= 0 AND fraud_score <= 1),
    confidence_lower NUMERIC(6,4),
    confidence_upper NUMERIC(6,4),
    shap_reasons JSONB,
    triggered_rules JSONB,
    latency_ms NUMERIC(8,2),
    model_version TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Analyst feedback (drives retraining)
CREATE TABLE IF NOT EXISTS feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    decision_id UUID NOT NULL REFERENCES decisions(id) ON DELETE CASCADE,
    analyst_label TEXT NOT NULL CHECK (analyst_label IN ('fraud', 'not_fraud')),
    analyst_note TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Entity blocklist
CREATE TABLE IF NOT EXISTS blocklist (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_type TEXT NOT NULL,
    entity_value TEXT NOT NULL,
    reason TEXT,
    added_by TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (entity_type, entity_value)
);

-- FP-Growth rule alerts
CREATE TABLE IF NOT EXISTS rule_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    rule_id TEXT NOT NULL,
    transaction_id UUID NOT NULL REFERENCES transactions(id) ON DELETE CASCADE,
    rule_antecedent JSONB NOT NULL,
    confidence NUMERIC(6,4),
    lift NUMERIC(8,4),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for query performance
CREATE INDEX IF NOT EXISTS idx_decisions_transaction_id ON decisions(transaction_id);
CREATE INDEX IF NOT EXISTS idx_decisions_created_at ON decisions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_decision_id ON feedback(decision_id);
CREATE INDEX IF NOT EXISTS idx_blocklist_entity ON blocklist(entity_type, entity_value);
CREATE INDEX IF NOT EXISTS idx_rule_alerts_transaction_id ON rule_alerts(transaction_id);
