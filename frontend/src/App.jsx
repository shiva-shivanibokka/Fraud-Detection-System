import { useState, useEffect, useRef } from "react";
import * as d3 from "d3";
import { createClient } from "@supabase/supabase-js";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const API = import.meta.env.VITE_API_URL || "http://localhost:8000";
const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL || "";
const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY || "";
let _supabase = null;
function getSupabase() {
  if (!SUPABASE_URL || !SUPABASE_ANON_KEY) return null;
  if (!_supabase) _supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
  return _supabase;
}

// ---------------------------------------------------------------------------
// Cold-start handling. The free-tier backend sleeps after ~15 min idle, so the
// first request then takes 30-60s to wake. apiFetch keeps the request open long
// enough to succeed, raises a global "waking up" banner if a call is slow, and
// retries on hard failures (e.g. a brief redeploy window) before giving up.
// ---------------------------------------------------------------------------
let _waking = false;
const _wakeSubs = new Set();
function _setWaking(v) {
  if (v !== _waking) {
    _waking = v;
    _wakeSubs.forEach((f) => f(v));
  }
}
function useWaking() {
  const [w, setW] = useState(_waking);
  useEffect(() => {
    _wakeSubs.add(setW);
    return () => _wakeSubs.delete(setW);
  }, []);
  return w;
}
async function apiFetch(path, options = {}) {
  const RETRIES = 3, SOFT_MS = 4000, HARD_MS = 70000;
  let lastErr;
  for (let attempt = 0; attempt <= RETRIES; attempt++) {
    const ctrl = new AbortController();
    const hard = setTimeout(() => ctrl.abort(), HARD_MS);
    const soft = setTimeout(() => _setWaking(true), SOFT_MS); // banner if slow
    try {
      const res = await fetch(`${API}${path}`, { ...options, signal: ctrl.signal });
      clearTimeout(hard); clearTimeout(soft); _setWaking(false);
      return res;
    } catch (e) {
      clearTimeout(hard); clearTimeout(soft);
      lastErr = e;
      _setWaking(true);
      if (attempt < RETRIES) await new Promise((r) => setTimeout(r, 2000 * (attempt + 1)));
    }
  }
  _setWaking(false);
  throw new Error("The server isn’t responding. The free-tier backend may be waking up — please retry in a moment.");
}

const US_STATES = [
  "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
  "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
  "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
  "VA","WA","WV","WI","WY",
];
const CATEGORIES = [
  "misc_net","grocery_pos","entertainment","gas_transport","misc_pos",
  "grocery_net","shopping_net","shopping_pos","food_dining","personal_care",
  "health_fitness","travel","kids_pets","home",
];

// ---------------------------------------------------------------------------
// Design tokens — light, professional, with color pops
// ---------------------------------------------------------------------------
const C = {
  bg: "#F4F6FB",
  surface: "#FFFFFF",
  border: "#E7EAF3",
  ink: "#161B2E",     // headings
  body: "#46506A",    // body text
  muted: "#646E84",
  faint: "#8A93A6",
  field: "#EEF1F8",
  primary: "#4F46E5", // indigo
  sky: "#0EA5E9",
  violet: "#8B5CF6",
  pink: "#EC4899",
  approve: "#10B981",
  review: "#F59E0B",
  decline: "#F43F5E",
};
const FONT = {
  display: "'Plus Jakarta Sans', system-ui, sans-serif",
  mono: "'JetBrains Mono', monospace",
};
const GRAD = `linear-gradient(135deg, ${C.primary}, ${C.sky})`;
const GRAD_HOT = `linear-gradient(135deg, ${C.violet}, ${C.pink})`;

const decisionColor = (d) =>
  d === "APPROVE" ? C.approve : d === "REVIEW" ? C.review : C.decline;
const scoreGradient = (s) => (s < 0.4 ? C.approve : s < 0.8 ? C.review : C.decline);

// snake_case -> "Title Case" for category/merchant display.
const prettify = (s) => (s || "").replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

// FP-Growth rule items -> plain English (cat_gas_transport -> "category: gas transport").
function humanizeToken(t) {
  if (!t) return "";
  if (t.startsWith("cat_")) return `category: ${prettify(t.slice(4)).toLowerCase()}`;
  if (t.startsWith("state_")) {
    const s = t.slice(6);
    return s === "other" ? "other state" : `state: ${s.toUpperCase()}`;
  }
  const map = {
    hour_night: "nighttime", hour_day: "daytime", weekend: "weekend", weekday: "weekday",
    amt_low: "low amount", amt_medium: "medium amount", amt_high: "high amount",
    geo_far: "far from home", geo_near: "near home",
  };
  return map[t] || t.replace(/_/g, " ");
}

// ---------------------------------------------------------------------------
// Shared styles
// ---------------------------------------------------------------------------
const card = { padding: "26px 28px", marginBottom: 20 };
const label = {
  fontFamily: FONT.mono, fontSize: 12, letterSpacing: "0.1em",
  textTransform: "uppercase", color: C.faint, marginBottom: 8, display: "block", fontWeight: 600,
};
const input = {
  width: "100%", background: C.field, border: `1px solid ${C.border}`,
  borderRadius: 11, padding: "12px 15px", color: C.ink, fontSize: 15,
  boxSizing: "border-box", fontWeight: 500,
};
const btn = (variant = "primary") => {
  const base = {
    border: "none", borderRadius: 11, padding: "12px 24px", fontSize: 15,
    cursor: "pointer", fontWeight: 700, color: "#fff", fontFamily: FONT.display,
    letterSpacing: "0.01em", transition: "transform .12s ease, filter .12s ease",
  };
  const v = {
    primary: { background: GRAD, boxShadow: `0 10px 24px -10px ${C.primary}` },
    hot: { background: GRAD_HOT, boxShadow: `0 10px 24px -10px ${C.pink}` },
    ghost: { background: C.field, border: `1px solid ${C.border}`, color: C.primary },
    danger: { background: C.decline, boxShadow: `0 10px 24px -10px ${C.decline}` },
    success: { background: C.approve, boxShadow: `0 10px 24px -10px ${C.approve}` },
  };
  return { ...base, ...(v[variant] || v.primary) };
};
const h3 = { margin: 0, color: C.ink, fontFamily: FONT.display, fontWeight: 700, fontSize: 22 };
const eyebrow = {
  fontFamily: FONT.mono, fontSize: 12, letterSpacing: "0.12em",
  textTransform: "uppercase", color: C.primary, fontWeight: 600,
};
const sub = { fontSize: 15, color: C.body, marginBottom: 16, lineHeight: 1.6 };

function Card({ children, style, className = "" }) {
  return <div className={`glass fade-up ${className}`} style={{ ...card, ...style }}>{children}</div>;
}
function CardHead({ title, kicker, right }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 14, marginBottom: 16 }}>
      <div>
        {kicker && <div style={{ ...eyebrow, marginBottom: 6 }}>{kicker}</div>}
        <h3 style={h3}>{title}</h3>
      </div>
      {right}
    </div>
  );
}

// Small hover "?" that explains a metric or term in plain language.
function Info({ text }) {
  const [show, setShow] = useState(false);
  return (
    <span style={{ position: "relative", display: "inline-flex", marginLeft: 6, verticalAlign: "middle" }}
      onMouseEnter={() => setShow(true)} onMouseLeave={() => setShow(false)}>
      <span style={{
        width: 16, height: 16, borderRadius: "50%", background: C.field, border: `1px solid ${C.border}`,
        color: C.muted, fontSize: 11, fontWeight: 700, display: "inline-flex", alignItems: "center",
        justifyContent: "center", cursor: "help", fontFamily: FONT.display,
      }}>?</span>
      {show && (
        <span style={{
          position: "absolute", bottom: "150%", left: "50%", transform: "translateX(-50%)",
          width: 240, background: C.ink, color: "#fff", fontFamily: "'Inter', sans-serif",
          fontSize: 12.5, lineHeight: 1.5, fontWeight: 400, letterSpacing: "normal", textTransform: "none",
          padding: "10px 13px", borderRadius: 9, zIndex: 40, boxShadow: "0 12px 28px -8px rgba(0,0,0,0.45)",
        }}>{text}</span>
      )}
    </span>
  );
}

// Per-tab explainer banner: what it is, what's happening, how to use it.
function TabIntro({ title, children }) {
  return (
    <div className="fade-up" style={{
      background: "linear-gradient(135deg, rgba(79,70,229,0.07), rgba(14,165,233,0.06))",
      border: `1px solid ${C.border}`, borderLeft: `4px solid ${C.primary}`,
      borderRadius: 14, padding: "16px 22px", marginBottom: 20,
    }}>
      <div style={{ fontFamily: FONT.display, fontWeight: 700, fontSize: 17, color: C.ink, marginBottom: 4 }}>{title}</div>
      <p style={{ fontSize: 14.5, color: C.body, lineHeight: 1.6, margin: 0 }}>{children}</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// LLM config (BYOK) — provider, model, key live ONLY in this browser's
// localStorage and are sent per-request via X-LLM-* headers; never on the server.
// ---------------------------------------------------------------------------
const LLM_LS = { provider: "fds_llm_provider", model: "fds_llm_model", key: "fds_llm_key" };
function loadLLMConfig() {
  return {
    provider: localStorage.getItem(LLM_LS.provider) || "",
    model: localStorage.getItem(LLM_LS.model) || "",
    key: localStorage.getItem(LLM_LS.key) || "",
  };
}
function saveLLMConfig({ provider, model, key }) {
  localStorage.setItem(LLM_LS.provider, provider);
  localStorage.setItem(LLM_LS.model, model);
  localStorage.setItem(LLM_LS.key, key);
}
function hasLLMConfig() {
  const c = loadLLMConfig();
  return Boolean(c.provider && c.key);
}
function llmHeaders() {
  const c = loadLLMConfig();
  return { "X-LLM-Provider": c.provider, "X-LLM-Model": c.model, "X-LLM-Key": c.key };
}
async function llmPost(path, body) {
  const res = await apiFetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...llmHeaders() },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
  return data;
}
const llmText = { whiteSpace: "pre-wrap", lineHeight: 1.65, fontSize: 15, color: C.ink };
const notice = {
  background: "#FFFFFF", border: `1px solid ${C.border}`, borderRadius: 16,
  padding: "22px 24px", color: C.body, fontSize: 15, lineHeight: 1.65,
  boxShadow: "0 10px 34px -14px rgba(38,50,90,0.22)",
};

// ---------------------------------------------------------------------------
// Confidence (conformal) chip
// ---------------------------------------------------------------------------
function ConfidenceChip({ label: cl }) {
  if (!cl || cl === "unknown") return null;
  const map = {
    confident_fraud: ["Confident · Fraud", C.decline],
    confident_legit: ["Confident · Legit", C.approve],
    uncertain: ["Uncertain · route to review", C.review],
  };
  const [text, color] = map[cl] || [cl, C.muted];
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 8, fontFamily: FONT.mono,
      fontSize: 12.5, letterSpacing: "0.02em", color, background: color + "16",
      border: `1px solid ${color}44`, borderRadius: 999, padding: "6px 13px", fontWeight: 600,
    }}>
      <span style={{ width: 8, height: 8, borderRadius: "50%", background: color }} />
      {text} <span style={{ opacity: 0.7 }}>· 90% coverage</span>
    </span>
  );
}

// ---------------------------------------------------------------------------
// Risk gauge — signature element. Tri-color arc sweeps to the fraud probability.
// ---------------------------------------------------------------------------
function RiskGauge({ score, decision }) {
  const R = 106, sw = 18, cx = 132, cy = 130, W = 264, Hh = 152;
  const arc = `M ${cx - R} ${cy} A ${R} ${R} 0 0 1 ${cx + R} ${cy}`;
  const col = decisionColor(decision);
  const v = Math.min(1, Math.max(0, score));
  const ang = Math.PI * (1 - v);
  const mx = cx + R * Math.cos(ang), my = cy - R * Math.sin(ang);
  return (
    <div style={{ textAlign: "center" }}>
      <svg width={W} height={Hh} viewBox={`0 0 ${W} ${Hh}`} style={{ maxWidth: "100%" }}>
        <defs>
          <linearGradient id="rg-track" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor={C.approve} />
            <stop offset="50%" stopColor={C.review} />
            <stop offset="100%" stopColor={C.decline} />
          </linearGradient>
        </defs>
        <path d={arc} fill="none" stroke="url(#rg-track)" strokeWidth={sw} strokeLinecap="round" opacity={0.32} />
        <path d={arc} fill="none" stroke={col} strokeWidth={sw} strokeLinecap="round"
          pathLength={1} strokeDasharray={`${v} 1`}
          style={{ transition: "stroke-dasharray 0.9s cubic-bezier(.22,1,.36,1)" }} />
        <circle cx={mx} cy={my} r={8} fill={col} stroke="#fff" strokeWidth={3}
          style={{ transition: "cx 0.9s cubic-bezier(.22,1,.36,1), cy 0.9s cubic-bezier(.22,1,.36,1)", filter: `drop-shadow(0 2px 5px ${col}88)` }} />
        <text x={cx - R} y={cy + 22} fill={C.faint} fontFamily={FONT.mono} fontSize={11}>0%</text>
        <text x={cx + R} y={cy + 22} fill={C.faint} fontFamily={FONT.mono} fontSize={11} textAnchor="end">100%</text>
      </svg>
      <div style={{ fontFamily: FONT.mono, fontSize: 44, fontWeight: 700, color: scoreGradient(score), lineHeight: 1, marginTop: -4 }}>
        {(score * 100).toFixed(1)}<span style={{ fontSize: 22 }}>%</span>
      </div>
      <div style={{ ...eyebrow, color: C.faint, marginTop: 5 }}>fraud probability</div>
    </div>
  );
}

function FeedbackButtons({ result }) {
  const [sent, setSent] = useState(null);
  const [err, setErr] = useState(null);
  const send = async (labelVal) => {
    setErr(null);
    try {
      const res = await apiFetch("/feedback", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          trans_id: result.trans_id, decision: result.decision,
          fraud_score: result.fraud_score, label: labelVal,
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setSent(labelVal);
    } catch (e) { setErr(e.message); }
  };
  if (sent) {
    return (
      <div style={{ marginTop: 18, fontSize: 14, color: C.approve, fontFamily: FONT.mono, fontWeight: 600 }}>
        ✓ Logged as <b>{sent === "fraud" ? "confirmed fraud" : "legitimate"}</b> — queued for retraining.
      </div>
    );
  }
  return (
    <div style={{ marginTop: 20, borderTop: `1px solid ${C.border}`, paddingTop: 16 }}>
      <div style={{ ...eyebrow, color: C.faint, marginBottom: 10 }}>analyst feedback</div>
      <div style={{ display: "flex", gap: 10 }}>
        <button style={{ ...btn("danger"), padding: "9px 16px", fontSize: 13.5 }} onClick={() => send("fraud")}>✗ Confirm Fraud</button>
        <button style={{ ...btn("success"), padding: "9px 16px", fontSize: 13.5 }} onClick={() => send("legit")}>✓ Mark Legitimate</button>
      </div>
      {err && <div style={{ marginTop: 8, color: C.decline, fontSize: 13 }}>{err}</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tab — Live Scoring
// ---------------------------------------------------------------------------
const pad2 = (n) => String(n).padStart(2, "0");
function toLocalInput(d) {
  return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}T${pad2(d.getHours())}:${pad2(d.getMinutes())}`;
}
function randCard() { return Array.from({ length: 16 }, () => Math.floor(Math.random() * 10)).join(""); }
function randDevice() { return "dev_" + Math.random().toString(36).slice(2, 10); }
function randIp() { return `${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}`; }

function LiveScoring() {
  const [form, setForm] = useState(() => ({
    amt: "125.00", category: "misc_net", merchant: "Riverside Market",
    state: "CA", geo_distance_km: "42", when: toLocalInput(new Date()),
    cc_num: randCard(), device_id: randDevice(), ip_prefix: randIp(),
  }));
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }));
  const reroll = () => setForm((f) => ({ ...f, cc_num: randCard(), device_id: randDevice(), ip_prefix: randIp() }));

  const submit = async () => {
    setLoading(true); setError(null);
    try {
      const when = new Date(form.when);
      const hour = when.getHours();
      const dow = when.getDay();
      const body = {
        trans_id: crypto.randomUUID?.() ?? Math.random().toString(36).slice(2),
        cc_num: form.cc_num, device_id: form.device_id, ip_prefix: form.ip_prefix,
        merchant: form.merchant, category: form.category,
        amt: parseFloat(form.amt) || 0, hour, day_of_week: dow,
        is_weekend: dow === 0 || dow === 6 ? 1 : 0,
        is_night: hour < 6 || hour >= 22 ? 1 : 0,
        age: 35, geo_distance_km: parseFloat(form.geo_distance_km) || 0,
        city_pop: 150000, state: form.state, gender: "M", timestamp: when.getTime() / 1000,
      };
      const res = await apiFetch("/score", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setResult(await res.json());
    } catch (e) { setError(e.message); } finally { setLoading(false); }
  };

  return (
    <>
      <TabIntro title="Live Scoring — score a transaction in real time">
        This runs one transaction through the 3-layer engine: hard rules → calibrated XGBoost → explanation.
        Fill in the details on the left and click <b>Score Transaction</b>; the right panel shows the verdict,
        fraud probability, the conformal confidence band, and why the model decided that way.
      </TabIntro>
      <div className="grid-2">
        <Card>
          <CardHead kicker="Module 1 · scoring engine" title="Transaction" />
          <div className="grid-fields">
            {[["Amount ($)", "amt", "number"], ["Geo distance (km)", "geo_distance_km", "number"],
              ["Merchant", "merchant", "text"]].map(([lbl, key, type]) => (
              <div key={key} style={key === "merchant" ? { gridColumn: "1 / -1" } : null}>
                <span style={label}>{lbl}</span>
                <input style={input} type={type} value={form[key]} onChange={set(key)} />
              </div>
            ))}
            <div>
              <span style={label}>Category</span>
              <select style={input} value={form.category} onChange={set("category")}>
                {CATEGORIES.map((c) => <option key={c} value={c}>{prettify(c)}</option>)}
              </select>
            </div>
            <div>
              <span style={label}>State</span>
              <select style={input} value={form.state} onChange={set("state")}>
                {US_STATES.map((s) => <option key={s}>{s}</option>)}
              </select>
            </div>
            <div style={{ gridColumn: "1 / -1" }}>
              <span style={label}>Transaction time</span>
              <input style={input} type="datetime-local" value={form.when} onChange={set("when")} />
              <div style={{ fontSize: 12.5, color: C.faint, marginTop: 6, fontFamily: FONT.mono }}>
                derives hour · day-of-week · weekend · night
              </div>
            </div>
          </div>
          <div style={{ marginTop: 16, padding: "14px 16px", background: C.field, borderRadius: 12, fontSize: 13, color: C.muted, fontFamily: FONT.mono }}>
            <div>card&nbsp;&nbsp;<span style={{ color: C.ink }}>{form.cc_num.slice(0, 4)} •••• {form.cc_num.slice(-4)}</span></div>
            <div>device&nbsp;<span style={{ color: C.ink }}>{form.device_id}</span></div>
            <div>ip&nbsp;&nbsp;&nbsp;&nbsp;<span style={{ color: C.ink }}>{form.ip_prefix}.x.x</span></div>
          </div>
          <div style={{ display: "flex", gap: 12, marginTop: 18 }}>
            <button style={btn("primary")} onClick={submit} disabled={loading}>{loading ? "Scoring…" : "Score Transaction"}</button>
            <button style={btn("ghost")} onClick={reroll}>New Identity</button>
          </div>
          {error && <div style={{ marginTop: 14, color: C.decline, fontSize: 14 }}>{error}</div>}
        </Card>

        <Card>
          <CardHead kicker="verdict" title="Decision" />
          {!result ? (
            <div style={{ color: C.faint, textAlign: "center", padding: "76px 0", fontFamily: FONT.mono, fontSize: 14 }}>
              Score a transaction to render a verdict.
            </div>
          ) : (
            <>
              <div className={result.decision === "DECLINE" ? "pulse-red" : ""}
                style={{
                  borderRadius: 16, padding: "12px 0 20px", marginBottom: 16,
                  background: `radial-gradient(120% 90% at 50% 0%, ${decisionColor(result.decision)}14, transparent 70%)`,
                  border: `1px solid ${decisionColor(result.decision)}33`,
                }}>
                <div style={{ textAlign: "center", paddingTop: 10 }}>
                  <span style={{
                    fontFamily: FONT.display, fontWeight: 800, fontSize: 36, letterSpacing: "0.01em",
                    color: decisionColor(result.decision),
                  }}>{result.decision}</span>
                </div>
                <RiskGauge score={result.fraud_score} decision={result.decision} />
                <div style={{ display: "flex", justifyContent: "center", gap: 10, marginTop: 10, flexWrap: "wrap" }}>
                  <ConfidenceChip label={result.confidence_label} />
                </div>
                <div style={{ textAlign: "center", marginTop: 12, fontFamily: FONT.mono, fontSize: 12.5, color: C.faint }}>
                  total {result.latency_ms}ms · model {result.model_latency_ms}ms · layer {result.layer_triggered}
                </div>
              </div>
              {result.reasons?.length > 0 && (
                <div style={{ marginBottom: 14 }}>
                  <div style={{ ...eyebrow, color: C.faint, marginBottom: 10 }}>why this decision</div>
                  {result.reasons.map((r, i) => (
                    <div key={i} style={{ display: "flex", gap: 10, alignItems: "flex-start", marginBottom: 8 }}>
                      <span style={{ color: decisionColor(result.decision), fontWeight: 700 }}>▸</span>
                      <span style={{ fontSize: 14.5, color: C.body, lineHeight: 1.5 }}>{r}</span>
                    </div>
                  ))}
                </div>
              )}
              {result.triggered_rules?.length > 0 && (
                <div>
                  <div style={{ ...eyebrow, color: C.faint, marginBottom: 9 }}>triggered rules</div>
                  {result.triggered_rules.map((rule, i) => (
                    <div key={i} style={{ background: C.field, borderRadius: 9, padding: "8px 12px", marginBottom: 6, fontSize: 13, color: C.review, fontFamily: FONT.mono, fontWeight: 600 }}>
                      IF {(rule.antecedents || []).map(humanizeToken).join(" AND ")} → {(rule.confidence * 100).toFixed(0)}%
                    </div>
                  ))}
                </div>
              )}
              <FeedbackButtons result={result} />
            </>
          )}
        </Card>
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------
// Tab — Fraud Ring Graph (D3 force-directed)
// ---------------------------------------------------------------------------
function FraudRingGraph() {
  const svgRef = useRef(null);
  const [threshold, setThreshold] = useState(0);
  const [hoveredNode, setHoveredNode] = useState(null);
  const [graphData, setGraphData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    apiFetch("/entity-graph").then((r) => r.json()).then(setGraphData).catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!graphData || !svgRef.current) return;
    const sid = (l) => l.source?.id ?? l.source;
    const tid = (l) => l.target?.id ?? l.target;
    // Accept either key ("links" or the pipeline's "edges"); clone so D3's
    // mutations don't corrupt the cached source data across re-renders.
    const allLinks = (graphData.links || graphData.edges || []).map((l) => ({ ...l }));
    // Seeds = entities meeting the fraud-rate threshold; then pull in their
    // direct neighbors so rings render connected instead of as isolated dots.
    const seeds = new Set(
      (graphData.nodes || []).filter((n) => (n.fraud_rate ?? 0) >= threshold).map((n) => n.id)
    );
    const keep = new Set(seeds);
    allLinks.forEach((l) => {
      if (seeds.has(sid(l))) keep.add(tid(l));
      if (seeds.has(tid(l))) keep.add(sid(l));
    });
    const nodes = (graphData.nodes || []).filter((n) => keep.has(n.id)).map((n) => ({ ...n }));
    const links = allLinks.filter((l) => keep.has(sid(l)) && keep.has(tid(l)));
    const W = svgRef.current.parentElement.clientWidth || 700, H = 480;
    d3.select(svgRef.current).selectAll("*").remove();
    const svg = d3.select(svgRef.current).attr("width", W).attr("height", H);
    const g = svg.append("g");
    svg.call(d3.zoom().scaleExtent([0.3, 4]).on("zoom", (e) => g.attr("transform", e.transform)));
    const colorMap = { card: C.sky, device: C.primary, ip: C.violet, merchant: C.approve };
    const sim = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id((d) => d.id).distance(80))
      .force("charge", d3.forceManyBody().strength(-130))
      .force("center", d3.forceCenter(W / 2, H / 2))
      .force("collision", d3.forceCollide(20));
    const link = g.append("g").selectAll("line").data(links).join("line")
      .attr("stroke", "#CBD2E4").attr("stroke-width", 1.5);
    const node = g.append("g").selectAll("circle").data(nodes).join("circle")
      .attr("r", (d) => 6 + Math.sqrt(d.txn_count ?? 1) * 2)
      .attr("fill", (d) => d.is_fraud ? C.decline : (colorMap[d.type] || C.muted))
      .attr("stroke", "#fff").attr("stroke-width", 2).attr("cursor", "pointer")
      .on("mouseover", (_, d) => setHoveredNode(d)).on("mouseout", () => setHoveredNode(null))
      .call(d3.drag()
        .on("start", (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on("drag", (e, d) => { d.fx = e.x; d.fy = e.y; })
        .on("end", (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }));
    sim.on("tick", () => {
      link.attr("x1", (d) => d.source.x).attr("y1", (d) => d.source.y).attr("x2", (d) => d.target.x).attr("y2", (d) => d.target.y);
      node.attr("cx", (d) => d.x).attr("cy", (d) => d.y);
    });
    return () => sim.stop();
  }, [graphData, threshold]);

  return (
    <>
      <TabIntro title="Fraud Ring Graph — find connected fraud networks">
        Cards, devices, IPs, and merchants are linked when they share transactions. Tight clusters of red
        (fraud) nodes are likely organized rings. Drag nodes to explore, scroll to zoom, and raise the
        <b> fraud-rate threshold</b> to hide low-risk entities and surface the rings.
      </TabIntro>
      <Card>
        <CardHead kicker="entity network" title="Fraud Ring Graph"
          right={
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <span style={{ ...label, marginBottom: 0 }}>fraud rate ≥ {(threshold * 100).toFixed(0)}%</span>
              <input type="range" min={0} max={1} step={0.05} value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))} style={{ width: 150 }} />
            </div>
          } />
        <div style={{ display: "flex", gap: 18, marginBottom: 14, fontSize: 13, color: C.muted, flexWrap: "wrap", fontFamily: FONT.mono }}>
          {[["card", C.sky], ["device", C.primary], ["ip", C.violet], ["merchant", C.approve], ["fraud", C.decline]].map(([t, c]) => (
            <div key={t} style={{ display: "flex", gap: 6, alignItems: "center" }}>
              <span style={{ width: 11, height: 11, borderRadius: "50%", background: c }} />{t}
            </div>
          ))}
        </div>
        {error
          ? <div style={{ color: C.decline, padding: 20 }}>Failed to load graph: {error}. Ensure the API is running.</div>
          : <div style={{ position: "relative" }}>
              <svg ref={svgRef} style={{ background: "#FBFCFE", borderRadius: 14, width: "100%", border: `1px solid ${C.border}` }} />
              {hoveredNode && (
                <div className="glass" style={{ position: "absolute", top: 12, left: 12, padding: "12px 16px", fontSize: 13, fontFamily: FONT.mono }}>
                  <div style={{ color: C.primary, fontWeight: 700, marginBottom: 4 }}>{hoveredNode.id}</div>
                  <div style={{ color: C.muted }}>type: {hoveredNode.type}</div>
                  <div style={{ color: C.muted }}>txns: {hoveredNode.txn_count ?? "—"}</div>
                  <div style={{ color: hoveredNode.fraud_rate > 0.3 ? C.decline : C.muted }}>
                    fraud: {((hoveredNode.fraud_rate ?? 0) * 100).toFixed(1)}%
                  </div>
                </div>
              )}
            </div>}
      </Card>
    </>
  );
}

// ---------------------------------------------------------------------------
// Tab — Drift Monitor
// ---------------------------------------------------------------------------
function DriftMonitor() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    apiFetch("/drift").then((r) => r.json()).then(setData).catch((e) => setError(e.message));
  }, []);
  const months = Array.isArray(data) ? data : (data?.months ?? []);
  const intro = (
    <TabIntro title="Drift Monitor — is the model still accurate?">
      Concept drift means fraud patterns shift and a model silently goes stale. This tracks test AUC month by
      month: green segments are healthy, <b>red segments dropped below the 0.85 retrain threshold</b>. The
      dashed line is precision@1%. A run of red is the signal to retrain.
    </TabIntro>
  );
  if (!months.length && !error) return <>{intro}<Card style={{ color: C.faint }}>Loading drift data…</Card></>;
  if (error) return <>{intro}<Card style={{ color: C.decline }}>Failed: {error}</Card></>;

  const W = 760, H = 280, p = { top: 20, right: 26, bottom: 54, left: 52 };
  const iW = W - p.left - p.right, iH = H - p.top - p.bottom;
  const xScale = (i) => (i / (months.length - 1 || 1)) * iW;
  const yScale = (v) => iH - ((v - 0.6) / 0.4) * iH;
  const linePath = (key) => months.map((m, i) => `${i === 0 ? "M" : "L"}${xScale(i).toFixed(1)},${yScale(m[key] ?? 0).toFixed(1)}`).join(" ");
  return (
    <>
      {intro}
      <Card>
        <CardHead kicker="model health" title="Model AUC over time" />
        <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ overflow: "visible" }}>
          <g transform={`translate(${p.left},${p.top})`}>
            {[0.6, 0.7, 0.8, 0.85, 0.9, 1.0].map((v) => (
              <g key={v}>
                <line x1={0} x2={iW} y1={yScale(v)} y2={yScale(v)} stroke="#EDF0F7" />
                <text x={-10} y={yScale(v) + 4} fill={C.faint} fontSize={11} textAnchor="end" fontFamily={FONT.mono}>{v.toFixed(2)}</text>
              </g>
            ))}
            <line x1={0} x2={iW} y1={yScale(0.85)} y2={yScale(0.85)} stroke={C.decline} strokeWidth={1} strokeDasharray="4 3" />
            {months.slice(1).map((m, i) => {
              const prev = months[i], avg = ((m.auc ?? 0) + (prev.auc ?? 0)) / 2;
              return <line key={i} x1={xScale(i)} y1={yScale(prev.auc ?? 0)} x2={xScale(i + 1)} y2={yScale(m.auc ?? 0)} stroke={avg < 0.85 ? C.decline : C.approve} strokeWidth={3} />;
            })}
            <path d={linePath("precision_at_1pct")} fill="none" stroke={C.sky} strokeWidth={2} strokeDasharray="5 3" />
            {months.map((m, i) => (
              <circle key={i} cx={xScale(i)} cy={yScale(m.auc ?? 0)} r={4.5} fill={(m.auc ?? 1) < 0.85 ? C.decline : C.approve} stroke="#fff" strokeWidth={1.5} />
            ))}
            {months.map((m, i) => (
              <text key={i} x={xScale(i)} y={iH + 24} fill={C.faint} fontSize={11} textAnchor="middle" fontFamily={FONT.mono} transform={`rotate(-40,${xScale(i)},${iH + 24})`}>{m.month ?? `M${i + 1}`}</text>
            ))}
          </g>
        </svg>
        <div style={{ display: "flex", gap: 22, marginTop: 10, fontSize: 13, color: C.muted, flexWrap: "wrap", fontFamily: FONT.mono }}>
          <span style={{ display: "flex", gap: 6, alignItems: "center" }}><span style={{ width: 22, height: 3, background: C.approve }} /> AUC ≥ 0.85</span>
          <span style={{ display: "flex", gap: 6, alignItems: "center" }}><span style={{ width: 22, height: 3, background: C.decline }} /> AUC &lt; 0.85</span>
          <span style={{ display: "flex", gap: 6, alignItems: "center" }}><span style={{ width: 22, height: 3, background: C.sky }} /> precision@1%</span>
        </div>
      </Card>
    </>
  );
}

// ---------------------------------------------------------------------------
// Tab — Rule Explorer (scrollable)
// ---------------------------------------------------------------------------
function RuleExplorer() {
  const [rules, setRules] = useState([]);
  const [sortKey, setSortKey] = useState("lift");
  const [error, setError] = useState(null);
  useEffect(() => {
    apiFetch("/fraud-rules").then((r) => r.json()).then((d) => setRules(d.rules ?? [])).catch((e) => setError(e.message));
  }, []);
  const sorted = [...rules].sort((a, b) => (b[sortKey] ?? 0) - (a[sortKey] ?? 0));
  const maxLift = Math.max(1, ...rules.map((r) => r.lift ?? 0));
  const Bar = ({ val, max, color }) => (
    <div style={{ background: C.field, borderRadius: 4, height: 8, width: 110, overflow: "hidden" }}>
      <div style={{ width: `${(val / max) * 100}%`, height: "100%", background: color, borderRadius: 4 }} />
    </div>
  );
  return (
    <>
      <TabIntro title="Rule Explorer — the patterns mined from fraud">
        These are association rules found by FP-Growth: combinations of attributes that co-occur with fraud
        far more than chance. <b>Lift</b> is the strength (3× = three times more likely than random); confidence
        and support measure reliability and frequency. Sort below and scroll the list to browse all rules.
      </TabIntro>
      <Card>
        <CardHead kicker="FP-Growth" title="Fraud Rules"
          right={
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <span style={{ ...label, marginBottom: 0 }}>sort</span>
              {["lift", "confidence", "support"].map((k) => (
                <button key={k} style={{ ...btn(sortKey === k ? "primary" : "ghost"), padding: "7px 13px", fontSize: 13 }}
                  onClick={() => setSortKey(k)}>{k}</button>
              ))}
            </div>
          } />
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 16 }}>
          {[
            ["Antecedent", "The conditions that must all hold for the rule to fire."],
            ["Confidence", "When those conditions hold, how often the transaction is fraud."],
            ["Lift", "How many × more likely fraud is than the baseline rate. Above 1× is predictive; 5× is strong."],
            ["Support", "The share of all fraud cases this rule covers."],
          ].map(([t, dsc]) => (
            <div key={t} style={{ flex: "1 1 190px", background: C.field, border: `1px solid ${C.border}`, borderRadius: 10, padding: "11px 14px" }}>
              <div style={{ fontFamily: FONT.mono, fontSize: 11, letterSpacing: "0.08em", textTransform: "uppercase", color: C.primary, fontWeight: 600, marginBottom: 4 }}>{t}</div>
              <div style={{ fontSize: 12.5, color: C.body, lineHeight: 1.45 }}>{dsc}</div>
            </div>
          ))}
        </div>
        {error ? <div style={{ color: C.decline }}>Failed: {error}</div>
          : rules.length === 0 ? <div style={{ color: C.faint }}>No rules loaded. Run the training pipeline to generate FP-Growth rules.</div>
          : <div style={{ maxHeight: 540, overflowY: "auto", overflowX: "auto", borderRadius: 12, border: `1px solid ${C.border}` }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
                <thead style={{ position: "sticky", top: 0, background: "#fff", zIndex: 1 }}>
                  <tr>{["Antecedent", "Confidence", "Lift", "Support"].map((hd) => (
                    <th key={hd} style={{ ...label, display: "table-cell", marginBottom: 0, textAlign: "left", padding: "12px 14px", borderBottom: `2px solid ${C.border}`, whiteSpace: "nowrap" }}>{hd}</th>
                  ))}</tr>
                </thead>
                <tbody>
                  {sorted.map((rule, i) => (
                    <tr key={i} style={{ borderBottom: `1px solid ${C.border}` }}>
                      <td style={{ padding: "12px 14px", color: C.ink }}>
                        {(rule.antecedents ?? []).map((a) => (
                          <span key={a} style={{ display: "inline-block", background: "#EDF0FB", border: `1px solid ${C.border}`, borderRadius: 6, padding: "3px 9px", marginRight: 5, marginBottom: 3, fontSize: 12.5, fontFamily: FONT.mono, color: C.primary }}>{humanizeToken(a)}</span>
                        ))}
                        {!rule.antecedents?.length && <span style={{ color: C.faint }}>—</span>}
                      </td>
                      <td style={{ padding: "12px 14px" }}>
                        <div style={{ color: C.review, marginBottom: 5, fontFamily: FONT.mono, fontWeight: 600 }}>{((rule.confidence ?? 0) * 100).toFixed(1)}%</div>
                        <Bar val={rule.confidence ?? 0} max={1} color={C.review} />
                      </td>
                      <td style={{ padding: "12px 14px" }}>
                        <div style={{ color: C.pink, marginBottom: 5, fontFamily: FONT.mono, fontWeight: 600 }}>{(rule.lift ?? 0).toFixed(2)}×</div>
                        <Bar val={rule.lift ?? 0} max={maxLift} color={C.pink} />
                      </td>
                      <td style={{ padding: "12px 14px", color: C.muted, fontFamily: FONT.mono }}>{((rule.support ?? 0) * 100).toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>}
      </Card>
    </>
  );
}

// ---------------------------------------------------------------------------
// Tab — GNN Predictions (Elliptic Bitcoin graph)
// ---------------------------------------------------------------------------
function Stat({ label: lab, value, color, info }) {
  return (
    <div style={{ background: C.field, borderRadius: 14, padding: "16px 18px", flex: "1 1 120px", border: `1px solid ${C.border}` }}>
      <div style={{ fontFamily: FONT.mono, fontSize: 28, fontWeight: 700, color: color || C.ink }}>{value}</div>
      <div style={{ ...eyebrow, color: C.faint, marginTop: 4, fontSize: 11, display: "flex", alignItems: "center" }}>
        {lab}{info && <Info text={info} />}
      </div>
    </div>
  );
}

function IllicitTimeline({ timeline }) {
  if (!timeline?.length) return null;
  const W = 760, H = 200, p = { top: 14, right: 16, bottom: 30, left: 40 };
  const iW = W - p.left - p.right, iH = H - p.top - p.bottom;
  const maxV = Math.max(1, ...timeline.map((t) => Math.max(t.actual_illicit, t.predicted_illicit)));
  const x = (i) => (i / (timeline.length - 1 || 1)) * iW;
  const y = (v) => iH - (v / maxV) * iH;
  const path = (key) => timeline.map((t, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(t[key]).toFixed(1)}`).join(" ");
  return (
    <div style={{ marginTop: 8 }}>
      <div style={{ ...eyebrow, color: C.faint, marginBottom: 10 }}>illicit nodes per time-step — actual vs predicted</div>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`}>
        <g transform={`translate(${p.left},${p.top})`}>
          {[0, 0.5, 1].map((f) => (
            <g key={f}>
              <line x1={0} x2={iW} y1={y(maxV * f)} y2={y(maxV * f)} stroke="#EDF0F7" />
              <text x={-8} y={y(maxV * f) + 4} fill={C.faint} fontSize={10} textAnchor="end" fontFamily={FONT.mono}>{Math.round(maxV * f)}</text>
            </g>
          ))}
          <path d={`${path("actual_illicit")} L ${x(timeline.length - 1)},${iH} L 0,${iH} Z`} fill={C.decline + "20"} stroke="none" />
          <path d={path("actual_illicit")} fill="none" stroke={C.decline} strokeWidth={2} />
          <path d={path("predicted_illicit")} fill="none" stroke={C.primary} strokeWidth={2} strokeDasharray="5 3" />
        </g>
      </svg>
      <div style={{ display: "flex", gap: 20, fontSize: 13, color: C.muted, fontFamily: FONT.mono, marginTop: 4 }}>
        <span style={{ display: "flex", gap: 6, alignItems: "center" }}><span style={{ width: 20, height: 3, background: C.decline }} /> actual illicit</span>
        <span style={{ display: "flex", gap: 6, alignItems: "center" }}><span style={{ width: 20, height: 3, background: C.primary }} /> predicted illicit</span>
      </div>
    </div>
  );
}

function GNNSubgraph({ graph }) {
  const svgRef = useRef(null);
  const [hovered, setHovered] = useState(null);
  useEffect(() => {
    if (!graph?.nodes?.length || !svgRef.current) return;
    const nodes = graph.nodes.map((n) => ({ ...n }));
    const idset = new Set(nodes.map((n) => n.id));
    const links = (graph.links || []).filter((l) => idset.has(l.source) && idset.has(l.target)).map((l) => ({ ...l }));
    const W = svgRef.current.parentElement.clientWidth || 700, H = 460;
    d3.select(svgRef.current).selectAll("*").remove();
    const svg = d3.select(svgRef.current).attr("width", W).attr("height", H);
    const g = svg.append("g");
    svg.call(d3.zoom().scaleExtent([0.3, 4]).on("zoom", (e) => g.attr("transform", e.transform)));
    const fill = (n) => (n.label === 1 ? C.decline : n.label === 0 ? C.approve : C.faint);
    const sim = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id((d) => d.id).distance(60))
      .force("charge", d3.forceManyBody().strength(-90))
      .force("center", d3.forceCenter(W / 2, H / 2))
      .force("collision", d3.forceCollide(12));
    const link = g.append("g").selectAll("line").data(links).join("line").attr("stroke", "#CBD2E4").attr("stroke-width", 1);
    const node = g.append("g").selectAll("circle").data(nodes).join("circle")
      .attr("r", (d) => 5 + d.prob * 9).attr("fill", fill).attr("stroke", "#fff").attr("stroke-width", 1.5).attr("cursor", "pointer")
      .on("mouseover", (_, d) => setHovered(d)).on("mouseout", () => setHovered(null))
      .call(d3.drag()
        .on("start", (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on("drag", (e, d) => { d.fx = e.x; d.fy = e.y; })
        .on("end", (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }));
    sim.on("tick", () => {
      link.attr("x1", (d) => d.source.x).attr("y1", (d) => d.source.y).attr("x2", (d) => d.target.x).attr("y2", (d) => d.target.y);
      node.attr("cx", (d) => d.x).attr("cy", (d) => d.y);
    });
    return () => sim.stop();
  }, [graph]);
  return (
    <div style={{ position: "relative" }}>
      <svg ref={svgRef} style={{ background: "#FBFCFE", borderRadius: 14, width: "100%", border: `1px solid ${C.border}` }} />
      {hovered && (
        <div className="glass" style={{ position: "absolute", top: 12, left: 12, padding: "12px 16px", fontSize: 13, fontFamily: FONT.mono }}>
          <div style={{ color: C.primary, fontWeight: 700, marginBottom: 4 }}>node #{hovered.id}</div>
          <div style={{ color: C.muted }}>fraud prob: {(hovered.prob * 100).toFixed(1)}%</div>
          <div style={{ color: C.muted }}>label: {hovered.label === 1 ? "illicit" : hovered.label === 0 ? "licit" : "unknown"}</div>
          <div style={{ color: C.muted }}>time-step: {hovered.step}</div>
        </div>
      )}
    </div>
  );
}

function GNNTab() {
  const [d, setD] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const load = () => {
    setLoading(true); setError(null);
    apiFetch("/graph/elliptic").then((r) => r.json())
      .then((j) => { setD(j); setLoading(false); })
      .catch((e) => { setError(e.message); setLoading(false); });
  };
  useEffect(() => { load(); }, []);
  const intro = (
    <TabIntro title="GNN Predictions — graph fraud on real Bitcoin data">
      Module 2 runs a graph neural network on the Elliptic Bitcoin dataset, classifying transactions as
      illicit or licit from the money-flow network. Below: the model's test metrics, how predicted vs actual
      illicit volume tracks over time, and the riskiest sub-network (node size = predicted fraud probability).
    </TabIntro>
  );
  if (error) return <>{intro}<Card style={{ color: C.decline }}>Failed: {error}</Card></>;
  if (!d) return <>{intro}<Card style={{ color: C.faint }}>Loading predictions…</Card></>;

  const hasData = d.graph?.nodes?.length > 0 || Object.keys(d.metrics || {}).length > 0;
  const m = d.metrics || {}, gs = d.graph_stats || {};
  const pct = (v) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);
  return (
    <>
      {intro}
      {!hasData ? (
        <div style={notice}>
          🧠 <b>Predictions aren’t published yet.</b> Generate them on a GPU box with{" "}
          <code style={{ color: C.primary }}>python -m src.graph_fraud.export_predictions</code>, then upload{" "}
          <code style={{ color: C.primary }}>models/elliptic_graph.json</code> to the Hugging Face model repo so the API can serve them.
        </div>
      ) : (
        <>
          <Card>
            <CardHead kicker={`${d.model || "GNN"} · Elliptic Bitcoin`} title="Test performance (illicit class)"
              right={
                <button style={{ ...btn("ghost"), padding: "9px 16px", fontSize: 13.5 }}
                  onClick={load} disabled={loading}>{loading ? "Refreshing…" : "↻ Refresh"}</button>
              } />
            <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
              <Stat label="ROC-AUC" value={pct(m.auc)} color={C.primary}
                info="Area under the ROC curve — the chance the model ranks a random fraud above a random legit transaction. 50% = guessing, 100% = perfect." />
              <Stat label="Illicit F1" value={pct(m.illicit_f1)} color={C.violet}
                info="The balance of precision and recall for the fraud class (their harmonic mean). A single fair score where both matter." />
              <Stat label="Precision" value={pct(m.illicit_precision)} color={C.sky}
                info="Of the transactions flagged as fraud, the share that truly were fraud (few false alarms = high)." />
              <Stat label="Recall" value={pct(m.illicit_recall)} color={C.approve}
                info="Of all the actual fraud, the share the model successfully caught (few misses = high)." />
            </div>
            <div style={{ marginTop: 14, fontSize: 13, color: C.muted, fontFamily: FONT.mono }}>
              {gs.nodes?.toLocaleString?.() ?? gs.nodes} nodes · {gs.edges?.toLocaleString?.() ?? gs.edges} edges · {gs.features} features · {gs.time_steps} time-steps
            </div>
            <IllicitTimeline timeline={d.timeline} />
          </Card>
          <Card>
            <CardHead kicker="riskiest sub-network" title="High-risk subgraph"
              right={
                <div style={{ display: "flex", gap: 16, fontSize: 13, color: C.muted, fontFamily: FONT.mono, flexWrap: "wrap" }}>
                  {[["illicit", C.decline], ["licit", C.approve], ["unknown", C.faint]].map(([t, c]) => (
                    <span key={t} style={{ display: "flex", gap: 6, alignItems: "center" }}><span style={{ width: 11, height: 11, borderRadius: "50%", background: c }} />{t}</span>
                  ))}
                </div>
              } />
            <GNNSubgraph graph={d.graph} />
          </Card>
        </>
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// Tab — Live Feed (Supabase Realtime)
// ---------------------------------------------------------------------------
function LiveFeed() {
  const [rows, setRows] = useState([]);
  const [status, setStatus] = useState("connecting");
  useEffect(() => {
    const sb = getSupabase();
    if (!sb) { setStatus("unconfigured"); return; }
    sb.from("live_decisions").select("*").order("created_at", { ascending: false }).limit(40)
      .then(({ data }) => { if (data) setRows(data); });
    const channel = sb.channel("live_decisions_feed")
      .on("postgres_changes", { event: "INSERT", schema: "public", table: "live_decisions" },
        (payload) => setRows((r) => [payload.new, ...r].slice(0, 60)))
      .subscribe((s) => setStatus(s === "SUBSCRIBED" ? "live" : s.toLowerCase()));
    return () => { sb.removeChannel(channel); };
  }, []);

  const intro = (
    <TabIntro title="Live Feed — decisions streaming in real time">
      Every scored transaction is published to Supabase and streamed here over a WebSocket, newest first, with
      declines highlighted in red. It’s a live operations view of what the engine is deciding right now —
      score a few transactions in the Live Scoring tab and watch them appear.
    </TabIntro>
  );

  if (status === "unconfigured") {
    return (
      <>
        {intro}
        <div style={notice}>
          <div style={{ fontFamily: FONT.display, fontWeight: 700, fontSize: 17, color: C.ink, marginBottom: 10 }}>📡 Connect the live feed (2 minutes)</div>
          <ol style={{ margin: "0 0 0 18px", padding: 0, lineHeight: 1.9 }}>
            <li>In your Supabase project, open <b>Settings → API</b> and copy the <b>Project URL</b> and the <b>anon public</b> key.</li>
            <li>Run the SQL in <code style={{ color: C.primary }}>supabase/migrations/004_live_decisions.sql</code> (creates the table + enables Realtime).</li>
            <li>Add these to the frontend env (Vercel → Project → Settings → Environment Variables, or a local <code style={{ color: C.primary }}>frontend/.env</code>):
              <div style={{ background: C.field, borderRadius: 10, padding: "12px 14px", marginTop: 8, fontFamily: FONT.mono, fontSize: 13, color: C.ink }}>
                VITE_SUPABASE_URL=https://YOUR-PROJECT.supabase.co<br />VITE_SUPABASE_ANON_KEY=eyJhbGciOi...
              </div>
            </li>
            <li>Set <code style={{ color: C.primary }}>SUPABASE_URL</code> + <code style={{ color: C.primary }}>SUPABASE_KEY</code> on the API (Render) so it publishes decisions, then redeploy both.</li>
          </ol>
        </div>
      </>
    );
  }
  const dot = status === "live" ? C.approve : C.review;
  return (
    <>
      {intro}
      <Card>
        <CardHead kicker="realtime stream" title="Live Transaction Feed"
          right={
            <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13, color: C.muted, fontFamily: FONT.mono }}>
              <span style={{ position: "relative", width: 10, height: 10 }}>
                <span style={{ position: "absolute", inset: 0, borderRadius: "50%", background: dot, animation: status === "live" ? "signalPing 1.8s ease-out infinite" : "none" }} />
                <span style={{ position: "absolute", inset: 0, borderRadius: "50%", background: dot }} />
              </span>
              {status === "live" ? "live" : status}
            </div>
          } />
        {rows.length === 0
          ? <div style={{ color: C.faint, fontSize: 14, padding: "10px 0", fontFamily: FONT.mono }}>Waiting for transactions… score one to see it stream in.</div>
          : <div style={{ display: "grid", gap: 7 }}>
              {rows.map((r, i) => {
                const c = decisionColor(r.decision), fraud = r.decision === "DECLINE";
                return (
                  <div key={r.id || i} className="row-in" style={{
                    display: "grid", gridTemplateColumns: "98px 1fr 92px 60px", gap: 12, alignItems: "center",
                    padding: "11px 15px", borderRadius: 11, fontFamily: FONT.mono,
                    background: fraud ? `${C.decline}0e` : C.field,
                    border: `1px solid ${fraud ? C.decline + "55" : C.border}`,
                    boxShadow: fraud ? `0 4px 16px -6px ${C.decline}55` : "none",
                  }}>
                    <span style={{ color: c, fontWeight: 700, fontSize: 13 }}>{r.decision}</span>
                    <span style={{ color: C.muted, fontSize: 13, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.merchant || "—"} · {prettify(r.category) || "—"}</span>
                    <span style={{ color: C.ink, fontSize: 13, textAlign: "right" }}>${Number(r.amount ?? 0).toFixed(2)}</span>
                    <span style={{ color: scoreGradient(r.fraud_score ?? 0), fontSize: 13, textAlign: "right", fontWeight: 700 }}>{((r.fraud_score ?? 0) * 100).toFixed(0)}%</span>
                  </div>
                );
              })}
            </div>}
      </Card>
    </>
  );
}

// ---------------------------------------------------------------------------
// Tab — AI Assistant
// ---------------------------------------------------------------------------
function CopilotChat() {
  const [q, setQ] = useState("Which merchant categories have the highest fraud lift?");
  const [answer, setAnswer] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const ask = async () => {
    setLoading(true); setError(null); setAnswer(null);
    try { setAnswer(await llmPost("/llm/copilot", { question: q })); }
    catch (e) { setError(e.message); } finally { setLoading(false); }
  };
  return (
    <Card>
      <CardHead kicker="grounded answers" title="Ask the Assistant" />
      <p style={sub}>Grounded on the system’s live fraud knowledge — FP-Growth rules, ring stats, metrics, and feature importances. It answers from what the system actually knows, not the open internet.</p>
      <textarea style={{ ...input, minHeight: 72, resize: "vertical" }} value={q} onChange={(e) => setQ(e.target.value)} />
      <div style={{ marginTop: 12 }}><button style={btn("primary")} onClick={ask} disabled={loading || !q.trim()}>{loading ? "Thinking…" : "Ask"}</button></div>
      {error && <div style={{ marginTop: 12, color: C.decline, fontSize: 14 }}>{error}</div>}
      {answer && (
        <div style={{ marginTop: 16 }}>
          <div style={llmText}>{answer.answer}</div>
          {answer.grounded_on && (
            <div style={{ marginTop: 12, fontSize: 12, color: C.faint, fontFamily: FONT.mono }}>
              grounded on {answer.grounded_on.rules} rules · {answer.grounded_on.rings} rings · {answer.grounded_on.features} features
            </div>
          )}
        </div>
      )}
    </Card>
  );
}

function RuleFromText() {
  const [text, setText] = useState("Flag any charge over $1000 made before 6am");
  const [rule, setRule] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const gen = async () => {
    setLoading(true); setError(null); setRule(null);
    try { setRule((await llmPost("/llm/rule-from-text", { text })).rule); }
    catch (e) { setError(e.message); } finally { setLoading(false); }
  };
  const ac = (a) => (a === "DECLINE" ? C.decline : a === "REVIEW" ? C.review : C.primary);
  return (
    <Card style={{ marginBottom: 0 }}>
      <CardHead kicker="english → rule" title="Rule Editor" />
      <p style={sub}>Describe a rule in plain English; the model returns a structured rule object mirroring the engine’s format.</p>
      <input style={input} value={text} onChange={(e) => setText(e.target.value)} />
      <div style={{ marginTop: 12 }}><button style={btn("primary")} onClick={gen} disabled={loading || !text.trim()}>{loading ? "Generating…" : "Generate Rule"}</button></div>
      {error && <div style={{ marginTop: 12, color: C.decline, fontSize: 14 }}>{error}</div>}
      {rule && (
        <div style={{ marginTop: 16, background: C.field, borderRadius: 12, padding: "16px 18px" }}>
          <div style={{ marginBottom: 9 }}>
            {(rule.antecedent || []).map((a) => (
              <span key={a} style={{ display: "inline-block", background: "#EDF0FB", border: `1px solid ${C.border}`, borderRadius: 6, padding: "3px 10px", marginRight: 5, marginBottom: 4, fontSize: 13, fontFamily: FONT.mono, color: C.primary }}>{a}</span>
            ))}
            <span style={{ color: C.faint, fontFamily: FONT.mono }}>→</span>
            <span style={{ marginLeft: 8, color: ac(rule.action), fontWeight: 800, fontSize: 14, fontFamily: FONT.display }}>{rule.action}</span>
          </div>
          <div style={{ fontSize: 13, color: C.muted, fontFamily: FONT.mono }}>confidence {(rule.confidence * 100).toFixed(0)}% · {rule.rationale}</div>
        </div>
      )}
    </Card>
  );
}

function CaseReport() {
  const [rings, setRings] = useState([]);
  const [idx, setIdx] = useState(0);
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  useEffect(() => {
    apiFetch("/fraud-rings").then((r) => r.json()).then((d) => setRings(Array.isArray(d) ? d : (d.rings || []))).catch(() => setRings([]));
  }, []);
  const gen = async () => {
    setLoading(true); setError(null); setReport(null);
    try { setReport((await llmPost("/llm/case-report", { ring_id: idx })).report); }
    catch (e) { setError(e.message); } finally { setLoading(false); }
  };
  const sizeOf = (r) => (r?.n_cards ?? r?.cards?.length ?? 0);
  return (
    <Card style={{ marginBottom: 0 }}>
      <CardHead kicker="one-click" title="Ring Case Report" />
      <p style={sub}>An investigator narrative generated from a detected fraud ring’s statistics.</p>
      {rings.length === 0 ? <div style={{ color: C.faint, fontSize: 14 }}>No fraud rings available.</div>
        : <>
            <div style={{ display: "flex", gap: 12, alignItems: "flex-end" }}>
              <div style={{ flex: 1 }}>
                <span style={label}>Ring</span>
                <select style={input} value={idx} onChange={(e) => setIdx(parseInt(e.target.value, 10))}>
                  {rings.map((r, i) => <option key={i} value={i}>Ring #{i} — {sizeOf(r)} cards</option>)}
                </select>
              </div>
              <button style={btn("primary")} onClick={gen} disabled={loading}>{loading ? "Writing…" : "Generate"}</button>
            </div>
            {error && <div style={{ marginTop: 12, color: C.decline, fontSize: 14 }}>{error}</div>}
            {report && <div style={{ ...llmText, marginTop: 16 }}>{report}</div>}
          </>}
    </Card>
  );
}

function AIAssistant() {
  const intro = (
    <TabIntro title="AI Assistant — your fraud copilot (bring your own key)">
      A language model wired to this system: ask questions grounded in the live fraud data, turn plain-English
      policies into structured rules, and generate investigator case reports. Add your OpenAI or Groq key in
      <b> Settings</b> first — it stays in your browser and is never stored on the server.
    </TabIntro>
  );
  if (!hasLLMConfig()) {
    return (
      <>
        {intro}
        <div style={notice}>
          🔑 Add your LLM provider and API key in the <b>Settings</b> tab to enable the assistant, rule editor, and case reports.
        </div>
      </>
    );
  }
  return (
    <>
      {intro}
      <div style={{ display: "grid", gap: 20 }}>
        <CopilotChat />
        <div className="grid-2">
          <RuleFromText />
          <CaseReport />
        </div>
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------
// Tab — Settings (BYOK)
// ---------------------------------------------------------------------------
function Settings() {
  const [providers, setProviders] = useState([]);
  const [cfg, setCfg] = useState(loadLLMConfig());
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState(null);
  useEffect(() => {
    apiFetch("/llm/providers").then((r) => r.json()).then((d) => {
      const list = d.providers || [];
      setProviders(list);
      setCfg((c) => (!c.provider && list.length ? { ...c, provider: list[0].id, model: list[0].models[0] } : c));
    }).catch((e) => setError(e.message));
  }, []);
  const current = providers.find((p) => p.id === cfg.provider);
  const onProvider = (id) => {
    const p = providers.find((x) => x.id === id);
    setCfg((c) => ({ ...c, provider: id, model: p?.models?.[0] || "" })); setSaved(false);
  };
  const save = () => { saveLLMConfig(cfg); setSaved(true); };
  return (
    <>
      <TabIntro title="Settings — connect your own LLM">
        The AI Assistant uses your own API key (bring-your-own-key). Pick a provider and model and paste your
        key; it’s stored only in this browser and relayed directly with each request — never saved on the server.
      </TabIntro>
      <div>
        <Card>
          <CardHead kicker="bring your own key" title="LLM API Settings" />
          <div className="grid-fields">
            <div>
              <span style={label}>Provider</span>
              <select style={input} value={cfg.provider} onChange={(e) => onProvider(e.target.value)}>
                {providers.map((p) => (
                  <option key={p.id} value={p.id}>{p.label} — {p.pricing === "free" ? "Free tier" : "Paid"}</option>
                ))}
              </select>
              {current && (
                <span style={{
                  display: "inline-flex", alignItems: "center", gap: 6, marginTop: 9, fontFamily: FONT.mono,
                  fontSize: 11.5, fontWeight: 600, padding: "4px 11px", borderRadius: 999,
                  color: current.pricing === "free" ? C.approve : C.review,
                  background: (current.pricing === "free" ? C.approve : C.review) + "18",
                  border: `1px solid ${(current.pricing === "free" ? C.approve : C.review)}44`,
                }}>
                  {current.pricing === "free" ? "✓ Free tier available" : "Paid — billed by the provider"}
                </span>
              )}
            </div>
            <div>
              <span style={label}>Model</span>
              <select style={input} value={cfg.model} onChange={(e) => { setCfg((c) => ({ ...c, model: e.target.value })); setSaved(false); }}>
                {(current?.models || []).map((m) => <option key={m} value={m}>{m}</option>)}
              </select>
            </div>
          </div>
          <div style={{ marginTop: 16 }}>
            <span style={label}>API key {current && <span style={{ color: C.faint }}>({current.key_hint})</span>}</span>
            <input style={input} type="password" placeholder="Paste your key" value={cfg.key}
              onChange={(e) => { setCfg((c) => ({ ...c, key: e.target.value })); setSaved(false); }} />
            {current?.key_url && (
              <a href={current.key_url} target="_blank" rel="noreferrer" style={{ fontSize: 13, color: C.primary, marginTop: 8, display: "inline-block", fontFamily: FONT.mono, fontWeight: 600 }}>
                get a {current.label} key →
              </a>
            )}
          </div>
          <div style={{ display: "flex", gap: 12, alignItems: "center", marginTop: 20 }}>
            <button style={btn("primary")} onClick={save} disabled={!cfg.provider || !cfg.key}>Save</button>
            {saved && <span style={{ color: C.approve, fontSize: 14, fontFamily: FONT.mono, fontWeight: 600 }}>✓ saved to this browser</span>}
            {error && <span style={{ color: C.decline, fontSize: 14 }}>{error}</span>}
          </div>
        </Card>
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------
// Root App
// ---------------------------------------------------------------------------
const TABS = [
  ["Live Scoring", LiveScoring],
  ["Fraud Ring Graph", FraudRingGraph],
  ["GNN Predictions", GNNTab],
  ["Drift Monitor", DriftMonitor],
  ["Rule Explorer", RuleExplorer],
  ["Live Feed", LiveFeed],
  ["AI Assistant", AIAssistant],
  ["Settings", Settings],
];

// Global toast shown while a backend call is slow (free-tier cold start).
function WakingBanner() {
  const waking = useWaking();
  if (!waking) return null;
  return (
    <div style={{
      position: "fixed", bottom: 22, left: "50%", transform: "translateX(-50%)", zIndex: 100,
      background: GRAD, color: "#fff", padding: "13px 24px", borderRadius: 12,
      fontSize: 14, fontWeight: 600, fontFamily: FONT.display,
      boxShadow: "0 16px 36px -10px rgba(38,50,90,0.45)",
      display: "flex", alignItems: "center", gap: 11, maxWidth: "90vw",
    }}>
      <span style={{ fontSize: 16 }}>⏳</span>
      Waking up the server… the free-tier backend sleeps when idle, so the first request can take up to a minute.
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState(0);
  const Active = TABS[tab][1];
  return (
    <div style={{ minHeight: "100vh", color: C.ink }}>
      <WakingBanner />
      <header style={{ background: "rgba(255,255,255,0.8)", borderBottom: `1px solid ${C.border}`, padding: "0 28px", backdropFilter: "blur(10px)", position: "sticky", top: 0, zIndex: 10 }}>
        <div style={{ maxWidth: 1280, margin: "0 auto", display: "flex", alignItems: "center", gap: 28, flexWrap: "wrap" }}>
          <div style={{ padding: "16px 0", display: "flex", alignItems: "center", gap: 11 }}>
            <span style={{ position: "relative", width: 12, height: 12 }}>
              <span style={{ position: "absolute", inset: 0, borderRadius: "50%", background: C.decline, animation: "signalPing 2s ease-out infinite" }} />
              <span style={{ position: "absolute", inset: 0, borderRadius: "50%", background: C.decline }} />
            </span>
            <span style={{ fontFamily: FONT.display, fontWeight: 800, fontSize: 19, color: C.ink, letterSpacing: "-0.01em" }}>
              Fraud<span style={{ background: GRAD, WebkitBackgroundClip: "text", backgroundClip: "text", color: "transparent" }}>Signal</span>
            </span>
            <span style={{ ...eyebrow, color: C.faint, marginLeft: 2 }}>platform</span>
          </div>
          <nav style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
            {TABS.map(([name], i) => (
              <button key={name} onClick={() => setTab(i)} style={{
                background: tab === i ? GRAD : "transparent", border: "none", cursor: "pointer",
                padding: "9px 15px", margin: "8px 0", fontSize: 14.5, fontWeight: 700,
                fontFamily: FONT.display, borderRadius: 10,
                color: tab === i ? "#fff" : C.muted,
                boxShadow: tab === i ? `0 8px 20px -10px ${C.primary}` : "none",
                transition: "color .15s ease",
              }}>{name}</button>
            ))}
          </nav>
        </div>
      </header>

      <main key={tab} className="fade-up" style={{ maxWidth: 1280, margin: "0 auto", padding: "30px 28px 48px" }}>
        <Active />
      </main>
    </div>
  );
}
