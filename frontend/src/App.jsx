import { useState, useEffect, useRef } from "react";
import * as d3 from "d3";
import { createClient } from "@supabase/supabase-js";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const API = import.meta.env.VITE_API_URL || "http://localhost:8000";
// Supabase Realtime (Live Feed tab). Optional — the tab degrades to a notice
// when these aren't set. The anon key is a public, row-level-secured key.
const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL || "";
const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY || "";
let _supabase = null;
function getSupabase() {
  if (!SUPABASE_URL || !SUPABASE_ANON_KEY) return null;
  if (!_supabase) _supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
  return _supabase;
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
// Design tokens — "Risk Signal Console"
// ---------------------------------------------------------------------------
const C = {
  ink: "#150A30",
  iris: "#7C5CFF",
  cyan: "#22D3EE",
  magenta: "#FF4D8D",
  txt: "#F3EEFF",
  muted: "#B6A8E6",
  faint: "#8576BE",
  line: "rgba(150,120,255,0.18)",
  field: "rgba(13,6,34,0.66)",
  approve: "#00E6A8",
  review: "#FFB02E",
  decline: "#FF3D6E",
};
const FONT = {
  display: "'Space Grotesk', system-ui, sans-serif",
  mono: "'JetBrains Mono', monospace",
};
const GRAD = `linear-gradient(135deg, ${C.iris}, ${C.cyan})`;
const GRAD_HOT = `linear-gradient(135deg, ${C.magenta}, ${C.iris})`;

const decisionColor = (d) =>
  d === "APPROVE" ? C.approve : d === "REVIEW" ? C.review : C.decline;
const scoreGradient = (s) => (s < 0.4 ? C.approve : s < 0.8 ? C.review : C.decline);

// ---------------------------------------------------------------------------
// Shared styles
// ---------------------------------------------------------------------------
const card = { padding: "22px 26px", marginBottom: 18 };
const label = {
  fontFamily: FONT.mono, fontSize: 10.5, letterSpacing: "0.13em",
  textTransform: "uppercase", color: C.faint, marginBottom: 7, display: "block",
};
const input = {
  width: "100%", background: C.field, border: `1px solid ${C.line}`,
  borderRadius: 10, padding: "10px 13px", color: C.txt, fontSize: 14,
  boxSizing: "border-box",
};
const btn = (variant = "primary") => {
  const base = {
    border: "none", borderRadius: 10, padding: "11px 22px", fontSize: 14,
    cursor: "pointer", fontWeight: 600, color: "#fff", fontFamily: FONT.display,
    letterSpacing: "0.01em", transition: "transform .12s ease, filter .12s ease",
  };
  const v = {
    primary: { background: GRAD, boxShadow: `0 8px 22px -10px ${C.iris}` },
    hot: { background: GRAD_HOT, boxShadow: `0 8px 22px -10px ${C.magenta}` },
    ghost: { background: "rgba(124,92,255,0.12)", border: `1px solid ${C.line}`, color: C.muted },
    danger: { background: C.decline, boxShadow: `0 8px 22px -10px ${C.decline}` },
    success: { background: C.approve, color: "#062b22", boxShadow: `0 8px 22px -10px ${C.approve}` },
  };
  return { ...base, ...(v[variant] || v.primary) };
};
const h3 = { margin: 0, color: C.txt, fontFamily: FONT.display, fontWeight: 600, fontSize: 18 };
const eyebrow = {
  fontFamily: FONT.mono, fontSize: 11, letterSpacing: "0.14em",
  textTransform: "uppercase", color: C.faint, fontWeight: 600,
};
const sub = { fontSize: 12.5, color: C.muted, marginBottom: 14, lineHeight: 1.5 };

function Card({ children, style, className = "" }) {
  return <div className={`glass fade-up ${className}`} style={{ ...card, ...style }}>{children}</div>;
}
function CardHead({ title, kicker, right }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12, marginBottom: 12 }}>
      <div>
        {kicker && <div style={{ ...eyebrow, marginBottom: 5 }}>{kicker}</div>}
        <h3 style={h3}>{title}</h3>
      </div>
      {right}
    </div>
  );
}

// ---------------------------------------------------------------------------
// LLM config (BYOK) — provider, model, and API key live ONLY in this browser's
// localStorage and are sent with each request via X-LLM-* headers. They are
// never persisted on the server.
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
  const res = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...llmHeaders() },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
  return data;
}
const llmText = { whiteSpace: "pre-wrap", lineHeight: 1.6, fontSize: 13.5, color: "#E4DDFA" };
const needsKeyNote = {
  background: "rgba(13,6,34,0.5)", border: `1px dashed ${C.line}`, borderRadius: 14,
  padding: "18px 20px", color: C.muted, fontSize: 13.5, lineHeight: 1.6,
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
      display: "inline-flex", alignItems: "center", gap: 7, fontFamily: FONT.mono,
      fontSize: 11, letterSpacing: "0.04em", color, background: color + "1c",
      border: `1px solid ${color}55`, borderRadius: 999, padding: "5px 11px",
    }}>
      <span style={{ width: 7, height: 7, borderRadius: "50%", background: color }} />
      {text} <span style={{ opacity: 0.6 }}>· 90% coverage</span>
    </span>
  );
}

// ---------------------------------------------------------------------------
// Risk gauge — the signature element. A tri-color arc sweeps to the fraud
// probability and the verdict is stamped beneath it; declines pulse.
// ---------------------------------------------------------------------------
function RiskGauge({ score, decision }) {
  const R = 104, sw = 18, cx = 130, cy = 128, W = 260, Hh = 150;
  const arc = `M ${cx - R} ${cy} A ${R} ${R} 0 0 1 ${cx + R} ${cy}`;
  const col = decisionColor(decision);
  const ang = Math.PI * (1 - Math.min(1, Math.max(0, score)));
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
          <filter id="rg-glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3.4" result="b" />
            <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>
        <path d={arc} fill="none" stroke="url(#rg-track)" strokeWidth={sw} strokeLinecap="round" opacity={0.26} />
        <path d={arc} fill="none" stroke={col} strokeWidth={sw} strokeLinecap="round"
          pathLength={1} strokeDasharray={`${Math.min(1, Math.max(0, score))} 1`}
          filter="url(#rg-glow)"
          style={{ transition: "stroke-dasharray 0.9s cubic-bezier(.22,1,.36,1)" }} />
        <circle cx={mx} cy={my} r={7} fill="#fff" stroke={col} strokeWidth={3} filter="url(#rg-glow)"
          style={{ transition: "cx 0.9s cubic-bezier(.22,1,.36,1), cy 0.9s cubic-bezier(.22,1,.36,1)" }} />
        <text x={cx - R} y={cy + 20} fill={C.faint} fontFamily={FONT.mono} fontSize={10}>0%</text>
        <text x={cx + R} y={cy + 20} fill={C.faint} fontFamily={FONT.mono} fontSize={10} textAnchor="end">100%</text>
      </svg>
      <div style={{ fontFamily: FONT.mono, fontSize: 40, fontWeight: 700, color: scoreGradient(score), lineHeight: 1, marginTop: -6 }}>
        {(score * 100).toFixed(1)}<span style={{ fontSize: 20 }}>%</span>
      </div>
      <div style={{ ...eyebrow, marginTop: 4 }}>fraud probability</div>
    </div>
  );
}

// Analyst feedback (✓/✗) on a scored transaction -> POST /feedback.
function FeedbackButtons({ result }) {
  const [sent, setSent] = useState(null);
  const [err, setErr] = useState(null);
  const send = async (labelVal) => {
    setErr(null);
    try {
      const res = await fetch(`${API}/feedback`, {
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
      <div style={{ marginTop: 16, fontSize: 12.5, color: C.approve, fontFamily: FONT.mono }}>
        ✓ Logged as <b>{sent === "fraud" ? "confirmed fraud" : "legitimate"}</b> — queued for retraining.
      </div>
    );
  }
  return (
    <div style={{ marginTop: 18, borderTop: `1px solid ${C.line}`, paddingTop: 14 }}>
      <div style={{ ...eyebrow, marginBottom: 9 }}>analyst feedback</div>
      <div style={{ display: "flex", gap: 10 }}>
        <button style={{ ...btn("danger"), padding: "8px 15px", fontSize: 12.5 }} onClick={() => send("fraud")}>✗ Confirm Fraud</button>
        <button style={{ ...btn("success"), padding: "8px 15px", fontSize: 12.5 }} onClick={() => send("legit")}>✓ Mark Legitimate</button>
      </div>
      {err && <div style={{ marginTop: 7, color: C.decline, fontSize: 12 }}>{err}</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tab 1 — Live Scoring
// ---------------------------------------------------------------------------
const pad2 = (n) => String(n).padStart(2, "0");
function toLocalInput(d) {
  return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}T${pad2(d.getHours())}:${pad2(d.getMinutes())}`;
}
function randCard() {
  return Array.from({ length: 16 }, () => Math.floor(Math.random() * 10)).join("");
}
function randDevice() { return "dev_" + Math.random().toString(36).slice(2, 10); }
function randIp() { return `${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}`; }

function LiveScoring() {
  const [form, setForm] = useState(() => ({
    amt: "125.00", category: "misc_net", merchant: "fraud_demo_merchant",
    state: "CA", geo_distance_km: "42", when: toLocalInput(new Date()),
    cc_num: randCard(), device_id: randDevice(), ip_prefix: randIp(),
  }));
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }));
  const reroll = () =>
    setForm((f) => ({ ...f, cc_num: randCard(), device_id: randDevice(), ip_prefix: randIp() }));

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
        amt: parseFloat(form.amt) || 0,
        hour,
        day_of_week: dow,
        is_weekend: dow === 0 || dow === 6 ? 1 : 0,
        is_night: hour < 6 || hour >= 22 ? 1 : 0,
        age: 35,
        geo_distance_km: parseFloat(form.geo_distance_km) || 0,
        city_pop: 150000, state: form.state, gender: "M",
        timestamp: when.getTime() / 1000,
      };
      const res = await fetch(`${API}/score`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setResult(await res.json());
    } catch (e) { setError(e.message); } finally { setLoading(false); }
  };

  return (
    <div className="grid-2">
      {/* Left — form */}
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
              {CATEGORIES.map((c) => <option key={c}>{c}</option>)}
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
            <div style={{ fontSize: 11, color: C.faint, marginTop: 5, fontFamily: FONT.mono }}>
              derives hour · day-of-week · weekend · night
            </div>
          </div>
        </div>

        <div style={{ marginTop: 14, padding: "12px 15px", background: C.field, borderRadius: 12, fontSize: 12, color: C.faint, fontFamily: FONT.mono }}>
          <div>card&nbsp;&nbsp;<span style={{ color: C.muted }}>{form.cc_num.slice(0, 4)} •••• {form.cc_num.slice(-4)}</span></div>
          <div>device&nbsp;<span style={{ color: C.muted }}>{form.device_id}</span></div>
          <div>ip&nbsp;&nbsp;&nbsp;&nbsp;<span style={{ color: C.muted }}>{form.ip_prefix}.x.x</span></div>
        </div>
        <div style={{ display: "flex", gap: 10, marginTop: 16 }}>
          <button style={btn("primary")} onClick={submit} disabled={loading}>
            {loading ? "Scoring…" : "Score Transaction"}
          </button>
          <button style={btn("ghost")} onClick={reroll}>New Identity</button>
        </div>
        {error && <div style={{ marginTop: 12, color: C.decline, fontSize: 13 }}>{error}</div>}
      </Card>

      {/* Right — verdict */}
      <Card>
        <CardHead kicker="verdict" title="Decision" />
        {!result ? (
          <div style={{ color: C.faint, textAlign: "center", padding: "70px 0", fontFamily: FONT.mono, fontSize: 13 }}>
            Score a transaction to render a verdict.
          </div>
        ) : (
          <>
            <div className={result.decision === "DECLINE" ? "pulse-red" : ""}
              style={{
                borderRadius: 16, padding: "10px 0 18px", marginBottom: 14,
                background: `radial-gradient(120% 90% at 50% 0%, ${decisionColor(result.decision)}1f, transparent 70%)`,
                border: `1px solid ${decisionColor(result.decision)}3a`,
              }}>
              <div style={{ textAlign: "center", paddingTop: 8 }}>
                <span style={{
                  fontFamily: FONT.display, fontWeight: 700, fontSize: 34, letterSpacing: "0.02em",
                  color: decisionColor(result.decision),
                  textShadow: `0 0 26px ${decisionColor(result.decision)}80`,
                }}>{result.decision}</span>
              </div>
              <RiskGauge score={result.fraud_score} decision={result.decision} />
              <div style={{ display: "flex", justifyContent: "center", gap: 10, marginTop: 8, flexWrap: "wrap" }}>
                <ConfidenceChip label={result.confidence_label} />
              </div>
              <div style={{ textAlign: "center", marginTop: 10, fontFamily: FONT.mono, fontSize: 11.5, color: C.faint }}>
                total {result.latency_ms}ms · model {result.model_latency_ms}ms · layer {result.layer_triggered}
              </div>
            </div>

            {result.reasons?.length > 0 && (
              <div style={{ marginBottom: 12 }}>
                <div style={{ ...eyebrow, marginBottom: 9 }}>why this decision</div>
                {result.reasons.map((r, i) => (
                  <div key={i} style={{ display: "flex", gap: 9, alignItems: "flex-start", marginBottom: 7 }}>
                    <span style={{ color: decisionColor(result.decision), fontWeight: 700 }}>▸</span>
                    <span style={{ fontSize: 13, color: "#DCD3F5", lineHeight: 1.45 }}>{r}</span>
                  </div>
                ))}
              </div>
            )}

            {result.triggered_rules?.length > 0 && (
              <div>
                <div style={{ ...eyebrow, marginBottom: 8 }}>triggered rules</div>
                {result.triggered_rules.map((rule, i) => (
                  <div key={i} style={{ background: C.field, borderRadius: 9, padding: "7px 11px", marginBottom: 5, fontSize: 12, color: C.review, fontFamily: FONT.mono }}>
                    IF {(rule.antecedent || []).join(" AND ")} → {(rule.confidence * 100).toFixed(0)}%
                  </div>
                ))}
              </div>
            )}

            <FeedbackButtons result={result} />
          </>
        )}
      </Card>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tab 2 — Fraud Ring Graph (D3 force-directed)
// ---------------------------------------------------------------------------
function FraudRingGraph() {
  const svgRef = useRef(null);
  const [threshold, setThreshold] = useState(0);
  const [hoveredNode, setHoveredNode] = useState(null);
  const [graphData, setGraphData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/entity-graph`).then((r) => r.json()).then(setGraphData).catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!graphData || !svgRef.current) return;
    const nodes = (graphData.nodes || []).filter((n) => (n.fraud_rate ?? 0) >= threshold);
    const nodeIds = new Set(nodes.map((n) => n.id));
    const links = (graphData.links || []).filter(
      (l) => nodeIds.has(l.source?.id ?? l.source) && nodeIds.has(l.target?.id ?? l.target)
    );
    const W = svgRef.current.parentElement.clientWidth || 700;
    const H = 480;
    d3.select(svgRef.current).selectAll("*").remove();
    const svg = d3.select(svgRef.current).attr("width", W).attr("height", H);
    const g = svg.append("g");
    svg.call(d3.zoom().scaleExtent([0.3, 4]).on("zoom", (e) => g.attr("transform", e.transform)));

    const colorMap = { card: C.cyan, device: C.iris, ip: C.magenta, merchant: C.approve };
    const fraudColor = C.decline;

    const sim = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id((d) => d.id).distance(80))
      .force("charge", d3.forceManyBody().strength(-130))
      .force("center", d3.forceCenter(W / 2, H / 2))
      .force("collision", d3.forceCollide(20));

    const link = g.append("g").selectAll("line").data(links).join("line")
      .attr("stroke", "rgba(150,120,255,0.35)").attr("stroke-width", 1.5);

    const node = g.append("g").selectAll("circle").data(nodes).join("circle")
      .attr("r", (d) => 6 + Math.sqrt(d.txn_count ?? 1) * 2)
      .attr("fill", (d) => d.is_fraud ? fraudColor : (colorMap[d.type] || C.muted))
      .attr("stroke", C.ink).attr("stroke-width", 2).attr("cursor", "pointer")
      .on("mouseover", (_, d) => setHoveredNode(d))
      .on("mouseout", () => setHoveredNode(null))
      .call(d3.drag()
        .on("start", (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on("drag", (e, d) => { d.fx = e.x; d.fy = e.y; })
        .on("end", (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }));

    sim.on("tick", () => {
      link.attr("x1", (d) => d.source.x).attr("y1", (d) => d.source.y)
          .attr("x2", (d) => d.target.x).attr("y2", (d) => d.target.y);
      node.attr("cx", (d) => d.x).attr("cy", (d) => d.y);
    });
    return () => sim.stop();
  }, [graphData, threshold]);

  return (
    <Card>
      <CardHead kicker="entity network" title="Fraud Ring Graph"
        right={
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ ...label, marginBottom: 0 }}>fraud rate ≥ {(threshold * 100).toFixed(0)}%</span>
            <input type="range" min={0} max={1} step={0.05} value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))} style={{ width: 140 }} />
          </div>
        } />
      <div style={{ display: "flex", gap: 16, marginBottom: 12, fontSize: 12, color: C.muted, flexWrap: "wrap", fontFamily: FONT.mono }}>
        {[["card", C.cyan], ["device", C.iris], ["ip", C.magenta], ["merchant", C.approve], ["fraud", C.decline]].map(([t, c]) => (
          <div key={t} style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <span style={{ width: 10, height: 10, borderRadius: "50%", background: c }} />{t}
          </div>
        ))}
      </div>
      {error
        ? <div style={{ color: C.decline, padding: 20 }}>Failed to load graph: {error}. Ensure the API is running.</div>
        : <div style={{ position: "relative" }}>
            <svg ref={svgRef} style={{ background: "rgba(8,4,22,0.5)", borderRadius: 14, width: "100%", border: `1px solid ${C.line}` }} />
            {hoveredNode && (
              <div className="glass" style={{ position: "absolute", top: 12, left: 12, padding: "11px 15px", fontSize: 12, fontFamily: FONT.mono }}>
                <div style={{ color: C.cyan, fontWeight: 700, marginBottom: 4 }}>{hoveredNode.id}</div>
                <div style={{ color: C.muted }}>type: {hoveredNode.type}</div>
                <div style={{ color: C.muted }}>txns: {hoveredNode.txn_count ?? "—"}</div>
                <div style={{ color: hoveredNode.fraud_rate > 0.3 ? C.decline : C.muted }}>
                  fraud: {((hoveredNode.fraud_rate ?? 0) * 100).toFixed(1)}%
                </div>
              </div>
            )}
          </div>}
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Tab 3 — Drift Monitor (SVG line chart)
// ---------------------------------------------------------------------------
function DriftMonitor() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    fetch(`${API}/drift`).then((r) => r.json()).then(setData).catch((e) => setError(e.message));
  }, []);

  const months = data?.months ?? [];
  if (!months.length && !error) return <Card style={{ color: C.faint }}>Loading drift data…</Card>;
  if (error) return <Card style={{ color: C.decline }}>Failed: {error}</Card>;

  const W = 700, H = 270, p = { top: 20, right: 24, bottom: 52, left: 50 };
  const iW = W - p.left - p.right, iH = H - p.top - p.bottom;
  const xScale = (i) => (i / (months.length - 1 || 1)) * iW;
  const yScale = (v) => iH - ((v - 0.6) / 0.4) * iH;
  const linePath = (key) => months.map((m, i) =>
    `${i === 0 ? "M" : "L"}${xScale(i).toFixed(1)},${yScale(m[key] ?? 0).toFixed(1)}`).join(" ");

  return (
    <Card>
      <CardHead kicker="model health" title="Drift Monitor — AUC over time" />
      <p style={sub}>Concept drift over time. Green segments hold AUC ≥ 0.85; red segments fall below the retraining threshold.</p>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ overflow: "visible" }}>
        <g transform={`translate(${p.left},${p.top})`}>
          {[0.6, 0.7, 0.8, 0.85, 0.9, 1.0].map((v) => (
            <g key={v}>
              <line x1={0} x2={iW} y1={yScale(v)} y2={yScale(v)} stroke="rgba(150,120,255,0.12)" />
              <text x={-10} y={yScale(v) + 4} fill={C.faint} fontSize={10} textAnchor="end" fontFamily={FONT.mono}>{v.toFixed(2)}</text>
            </g>
          ))}
          <line x1={0} x2={iW} y1={yScale(0.85)} y2={yScale(0.85)} stroke={C.decline} strokeWidth={1} strokeDasharray="4 3" />
          {months.slice(1).map((m, i) => {
            const prev = months[i];
            const avg = ((m.auc ?? 0) + (prev.auc ?? 0)) / 2;
            return <line key={i} x1={xScale(i)} y1={yScale(prev.auc ?? 0)} x2={xScale(i + 1)} y2={yScale(m.auc ?? 0)}
              stroke={avg < 0.85 ? C.decline : C.approve} strokeWidth={2.5} />;
          })}
          <path d={linePath("precision_at_1pct")} fill="none" stroke={C.cyan} strokeWidth={1.8} strokeDasharray="5 3" />
          {months.map((m, i) => (
            <circle key={i} cx={xScale(i)} cy={yScale(m.auc ?? 0)} r={4}
              fill={(m.auc ?? 1) < 0.85 ? C.decline : C.approve} stroke={C.ink} strokeWidth={1.5} />
          ))}
          {months.map((m, i) => (
            <text key={i} x={xScale(i)} y={iH + 22} fill={C.faint} fontSize={10} textAnchor="middle"
              fontFamily={FONT.mono} transform={`rotate(-40,${xScale(i)},${iH + 22})`}>{m.month ?? `M${i + 1}`}</text>
          ))}
        </g>
      </svg>
      <div style={{ display: "flex", gap: 20, marginTop: 8, fontSize: 12, color: C.muted, flexWrap: "wrap", fontFamily: FONT.mono }}>
        <span style={{ display: "flex", gap: 6, alignItems: "center" }}><span style={{ width: 20, height: 3, background: C.approve }} /> AUC ≥ 0.85</span>
        <span style={{ display: "flex", gap: 6, alignItems: "center" }}><span style={{ width: 20, height: 3, background: C.decline }} /> AUC &lt; 0.85</span>
        <span style={{ display: "flex", gap: 6, alignItems: "center" }}><span style={{ width: 20, height: 3, background: C.cyan }} /> precision@1%</span>
      </div>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Tab 4 — Rule Explorer
// ---------------------------------------------------------------------------
function RuleExplorer() {
  const [rules, setRules] = useState([]);
  const [sortKey, setSortKey] = useState("lift");
  const [error, setError] = useState(null);
  useEffect(() => {
    fetch(`${API}/fraud-rules`).then((r) => r.json()).then((d) => setRules(d.rules ?? [])).catch((e) => setError(e.message));
  }, []);
  const sorted = [...rules].sort((a, b) => (b[sortKey] ?? 0) - (a[sortKey] ?? 0));
  const maxLift = Math.max(1, ...rules.map((r) => r.lift ?? 0));
  const Bar = ({ val, max, color }) => (
    <div style={{ background: C.field, borderRadius: 4, height: 7, width: 100, overflow: "hidden" }}>
      <div style={{ width: `${(val / max) * 100}%`, height: "100%", background: color, borderRadius: 4 }} />
    </div>
  );
  return (
    <Card>
      <CardHead kicker="FP-Growth" title="Fraud Rules"
        right={
          <div style={{ display: "flex", gap: 7, alignItems: "center" }}>
            <span style={{ ...label, marginBottom: 0 }}>sort</span>
            {["lift", "confidence", "support"].map((k) => (
              <button key={k} style={{ ...btn(sortKey === k ? "primary" : "ghost"), padding: "6px 12px", fontSize: 12 }}
                onClick={() => setSortKey(k)}>{k}</button>
            ))}
          </div>
        } />
      {error ? <div style={{ color: C.decline }}>Failed: {error}</div>
        : rules.length === 0 ? <div style={{ color: C.faint }}>No rules loaded. Run the training pipeline to generate FP-Growth rules.</div>
        : <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
              <thead><tr>
                {["Antecedent", "Confidence", "Lift", "Support"].map((hd) => (
                  <th key={hd} style={{ textAlign: "left", padding: "9px 12px", ...label, marginBottom: 0, borderBottom: `1px solid ${C.line}`, whiteSpace: "nowrap" }}>{hd}</th>
                ))}
              </tr></thead>
              <tbody>
                {sorted.map((rule, i) => (
                  <tr key={i} style={{ borderBottom: "1px solid rgba(150,120,255,0.08)" }}>
                    <td style={{ padding: "11px 12px", color: C.txt }}>
                      {(rule.antecedent ?? []).map((a) => (
                        <span key={a} style={{ display: "inline-block", background: "rgba(124,92,255,0.16)", border: `1px solid ${C.line}`, borderRadius: 5, padding: "2px 8px", marginRight: 4, marginBottom: 2, fontSize: 11, fontFamily: FONT.mono }}>{a}</span>
                      ))}
                      {!rule.antecedent?.length && <span style={{ color: C.faint }}>—</span>}
                    </td>
                    <td style={{ padding: "11px 12px" }}>
                      <div style={{ color: C.review, marginBottom: 4, fontFamily: FONT.mono }}>{((rule.confidence ?? 0) * 100).toFixed(1)}%</div>
                      <Bar val={rule.confidence ?? 0} max={1} color={C.review} />
                    </td>
                    <td style={{ padding: "11px 12px" }}>
                      <div style={{ color: C.magenta, marginBottom: 4, fontFamily: FONT.mono }}>{(rule.lift ?? 0).toFixed(2)}×</div>
                      <Bar val={rule.lift ?? 0} max={maxLift} color={C.magenta} />
                    </td>
                    <td style={{ padding: "11px 12px", color: C.muted, fontFamily: FONT.mono }}>{((rule.support ?? 0) * 100).toFixed(2)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>}
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Tab 5 — Live Feed (Supabase Realtime)
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

  if (status === "unconfigured") {
    return (
      <div style={{ maxWidth: 640 }}>
        <div style={needsKeyNote}>
          📡 The Live Feed streams scored decisions over Supabase Realtime. Set
          <code style={{ color: C.cyan }}> VITE_SUPABASE_URL</code> and
          <code style={{ color: C.cyan }}> VITE_SUPABASE_ANON_KEY</code> in the frontend env to enable it.
        </div>
      </div>
    );
  }
  const dot = status === "live" ? C.approve : C.review;
  return (
    <Card>
      <CardHead kicker="realtime stream" title="Live Transaction Feed"
        right={
          <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, color: C.muted, fontFamily: FONT.mono }}>
            <span style={{ position: "relative", width: 9, height: 9 }}>
              <span style={{ position: "absolute", inset: 0, borderRadius: "50%", background: dot, animation: status === "live" ? "signalPing 1.8s ease-out infinite" : "none" }} />
              <span style={{ position: "absolute", inset: 0, borderRadius: "50%", background: dot }} />
            </span>
            {status === "live" ? "live" : status}
          </div>
        } />
      {rows.length === 0
        ? <div style={{ color: C.faint, fontSize: 13, padding: "10px 0", fontFamily: FONT.mono }}>Waiting for transactions… score one to see it stream in.</div>
        : <div style={{ display: "grid", gap: 6 }}>
            {rows.map((r, i) => {
              const c = decisionColor(r.decision);
              const fraud = r.decision === "DECLINE";
              return (
                <div key={r.id || i} className="row-in" style={{
                  display: "grid", gridTemplateColumns: "92px 1fr 88px 58px", gap: 10, alignItems: "center",
                  padding: "9px 13px", borderRadius: 10, fontFamily: FONT.mono,
                  background: fraud ? `${C.decline}18` : C.field,
                  border: `1px solid ${fraud ? C.decline + "66" : C.line}`,
                  boxShadow: fraud ? `0 0 16px -2px ${C.decline}55` : "none",
                }}>
                  <span style={{ color: c, fontWeight: 700, fontSize: 12.5 }}>{r.decision}</span>
                  <span style={{ color: C.muted, fontSize: 12, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.merchant || "—"} · {r.category || "—"}</span>
                  <span style={{ color: C.txt, fontSize: 12, textAlign: "right" }}>${Number(r.amount ?? 0).toFixed(2)}</span>
                  <span style={{ color: scoreGradient(r.fraud_score ?? 0), fontSize: 12, textAlign: "right", fontWeight: 600 }}>{((r.fraud_score ?? 0) * 100).toFixed(0)}%</span>
                </div>
              );
            })}
          </div>}
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Tab 6 — Settings (BYOK)
// ---------------------------------------------------------------------------
function Settings() {
  const [providers, setProviders] = useState([]);
  const [cfg, setCfg] = useState(loadLLMConfig());
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState(null);
  useEffect(() => {
    fetch(`${API}/llm/providers`).then((r) => r.json()).then((d) => {
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
    <div style={{ maxWidth: 640 }}>
      <Card>
        <CardHead kicker="bring your own key" title="LLM Copilot — API Settings" />
        <p style={sub}>The copilot, case reports, and rule editor run on your own key. Pick a provider and model, paste your key, and save. Your key is stored <b style={{ color: C.txt }}>only in this browser</b> and relayed directly with each request — never stored on the server.</p>
        <div className="grid-fields">
          <div>
            <span style={label}>Provider</span>
            <select style={input} value={cfg.provider} onChange={(e) => onProvider(e.target.value)}>
              {providers.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
            </select>
          </div>
          <div>
            <span style={label}>Model</span>
            <select style={input} value={cfg.model} onChange={(e) => { setCfg((c) => ({ ...c, model: e.target.value })); setSaved(false); }}>
              {(current?.models || []).map((m) => <option key={m} value={m}>{m}</option>)}
            </select>
          </div>
        </div>
        <div style={{ marginTop: 14 }}>
          <span style={label}>API key {current && <span style={{ color: C.faint }}>({current.key_hint})</span>}</span>
          <input style={input} type="password" placeholder="Paste your key" value={cfg.key}
            onChange={(e) => { setCfg((c) => ({ ...c, key: e.target.value })); setSaved(false); }} />
          {current?.key_url && (
            <a href={current.key_url} target="_blank" rel="noreferrer" style={{ fontSize: 12, color: C.cyan, marginTop: 7, display: "inline-block", fontFamily: FONT.mono }}>
              get a {current.label} key →
            </a>
          )}
        </div>
        <div style={{ display: "flex", gap: 12, alignItems: "center", marginTop: 18 }}>
          <button style={btn("primary")} onClick={save} disabled={!cfg.provider || !cfg.key}>Save</button>
          {saved && <span style={{ color: C.approve, fontSize: 13, fontFamily: FONT.mono }}>✓ saved to this browser</span>}
          {error && <span style={{ color: C.decline, fontSize: 13 }}>{error}</span>}
        </div>
      </Card>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tab 7 — Analyst Copilot
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
      <CardHead kicker="grounded RAG" title="Analyst Copilot" />
      <p style={sub}>Grounded on the system's live fraud knowledge — FP-Growth rules, ring stats, metrics, and feature importances. Answers cite only what the system actually knows.</p>
      <textarea style={{ ...input, minHeight: 66, resize: "vertical" }} value={q} onChange={(e) => setQ(e.target.value)} />
      <div style={{ marginTop: 10 }}>
        <button style={btn("primary")} onClick={ask} disabled={loading || !q.trim()}>{loading ? "Thinking…" : "Ask Copilot"}</button>
      </div>
      {error && <div style={{ marginTop: 10, color: C.decline, fontSize: 13 }}>{error}</div>}
      {answer && (
        <div style={{ marginTop: 14 }}>
          <div style={llmText}>{answer.answer}</div>
          {answer.grounded_on && (
            <div style={{ marginTop: 10, fontSize: 11, color: C.faint, fontFamily: FONT.mono }}>
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
  const ac = (a) => (a === "DECLINE" ? C.decline : a === "REVIEW" ? C.review : C.iris);
  return (
    <Card style={{ marginBottom: 0 }}>
      <CardHead kicker="english → rule" title="Rule Editor" />
      <p style={sub}>Describe a fraud rule in plain English; the model returns a structured rule object mirroring the engine's format.</p>
      <input style={input} value={text} onChange={(e) => setText(e.target.value)} />
      <div style={{ marginTop: 10 }}>
        <button style={btn("primary")} onClick={gen} disabled={loading || !text.trim()}>{loading ? "Generating…" : "Generate Rule"}</button>
      </div>
      {error && <div style={{ marginTop: 10, color: C.decline, fontSize: 13 }}>{error}</div>}
      {rule && (
        <div style={{ marginTop: 14, background: C.field, borderRadius: 12, padding: "14px 16px" }}>
          <div style={{ marginBottom: 8 }}>
            {(rule.antecedent || []).map((a) => (
              <span key={a} style={{ display: "inline-block", background: "rgba(124,92,255,0.16)", border: `1px solid ${C.line}`, borderRadius: 5, padding: "3px 9px", marginRight: 5, marginBottom: 4, fontSize: 12, fontFamily: FONT.mono }}>{a}</span>
            ))}
            <span style={{ color: C.faint, fontFamily: FONT.mono }}>→</span>
            <span style={{ marginLeft: 8, color: ac(rule.action), fontWeight: 700, fontSize: 13, fontFamily: FONT.display }}>{rule.action}</span>
          </div>
          <div style={{ fontSize: 12, color: C.muted, fontFamily: FONT.mono }}>confidence {(rule.confidence * 100).toFixed(0)}% · {rule.rationale}</div>
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
    fetch(`${API}/fraud-rings`).then((r) => r.json())
      .then((d) => setRings(Array.isArray(d) ? d : (d.rings || []))).catch(() => setRings([]));
  }, []);
  const gen = async () => {
    setLoading(true); setError(null); setReport(null);
    try { setReport((await llmPost("/llm/case-report", { ring_id: idx })).report); }
    catch (e) { setError(e.message); } finally { setLoading(false); }
  };
  const sizeOf = (r) => (r?.cards?.length ?? 0);
  return (
    <Card style={{ marginBottom: 0 }}>
      <CardHead kicker="one-click" title="Ring Case Report" />
      <p style={sub}>Investigator narrative generated from a detected fraud ring's statistics.</p>
      {rings.length === 0 ? <div style={{ color: C.faint, fontSize: 13 }}>No fraud rings available.</div>
        : <>
            <div style={{ display: "flex", gap: 10, alignItems: "flex-end" }}>
              <div style={{ flex: 1 }}>
                <span style={label}>Ring</span>
                <select style={input} value={idx} onChange={(e) => setIdx(parseInt(e.target.value, 10))}>
                  {rings.map((r, i) => <option key={i} value={i}>Ring #{i} — {sizeOf(r)} cards</option>)}
                </select>
              </div>
              <button style={btn("primary")} onClick={gen} disabled={loading}>{loading ? "Writing…" : "Generate"}</button>
            </div>
            {error && <div style={{ marginTop: 10, color: C.decline, fontSize: 13 }}>{error}</div>}
            {report && <div style={{ ...llmText, marginTop: 14 }}>{report}</div>}
          </>}
    </Card>
  );
}

function Copilot() {
  if (!hasLLMConfig()) {
    return (
      <div style={{ maxWidth: 640 }}>
        <div style={needsKeyNote}>
          🔑 Add your LLM provider and API key in the <b style={{ color: C.txt }}>Settings</b> tab to use the copilot, rule editor, and case reports. Your key stays in your browser.
        </div>
      </div>
    );
  }
  return (
    <div style={{ display: "grid", gap: 18 }}>
      <CopilotChat />
      <div className="grid-2">
        <RuleFromText />
        <CaseReport />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Root App
// ---------------------------------------------------------------------------
const TABS = ["Live Scoring", "Fraud Ring Graph", "Drift Monitor", "Rule Explorer", "Live Feed", "Copilot", "Settings"];

export default function App() {
  const [tab, setTab] = useState(0);
  return (
    <div style={{ minHeight: "100vh", color: C.txt }}>
      {/* Header */}
      <header style={{ borderBottom: `1px solid ${C.line}`, padding: "0 28px", backdropFilter: "blur(8px)" }}>
        <div style={{ maxWidth: 1220, margin: "0 auto", display: "flex", alignItems: "center", gap: 28, flexWrap: "wrap" }}>
          <div style={{ padding: "16px 0", display: "flex", alignItems: "center", gap: 11 }}>
            <span style={{ position: "relative", width: 11, height: 11 }}>
              <span style={{ position: "absolute", inset: 0, borderRadius: "50%", background: C.decline, animation: "signalPing 2s ease-out infinite" }} />
              <span style={{ position: "absolute", inset: 0, borderRadius: "50%", background: C.decline }} />
            </span>
            <span style={{ fontFamily: FONT.display, fontWeight: 700, fontSize: 18, letterSpacing: "0.01em" }}>
              Fraud<span style={{ background: GRAD, WebkitBackgroundClip: "text", backgroundClip: "text", color: "transparent" }}>Signal</span>
            </span>
            <span style={{ ...eyebrow, marginLeft: 2 }}>platform</span>
          </div>
          <nav style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
            {TABS.map((t, i) => (
              <button key={t} onClick={() => setTab(i)} style={{
                background: tab === i ? GRAD : "transparent", border: "none", cursor: "pointer",
                padding: tab === i ? "8px 15px" : "8px 14px", margin: "8px 0", fontSize: 13, fontWeight: 600,
                fontFamily: FONT.display, borderRadius: 10,
                color: tab === i ? "#fff" : C.faint,
                boxShadow: tab === i ? `0 8px 20px -10px ${C.iris}` : "none",
                transition: "color .15s ease",
              }}>{t}</button>
            ))}
          </nav>
        </div>
      </header>

      {/* Content */}
      <main key={tab} className="fade-up" style={{ maxWidth: 1220, margin: "0 auto", padding: "28px" }}>
        {tab === 0 && <LiveScoring />}
        {tab === 1 && <FraudRingGraph />}
        {tab === 2 && <DriftMonitor />}
        {tab === 3 && <RuleExplorer />}
        {tab === 4 && <LiveFeed />}
        {tab === 5 && <Copilot />}
        {tab === 6 && <Settings />}
      </main>
    </div>
  );
}
