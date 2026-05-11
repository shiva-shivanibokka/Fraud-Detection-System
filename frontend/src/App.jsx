import { useState, useEffect, useRef } from "react";
import * as d3 from "d3";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const API = "http://localhost:8000";
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
// Colour helpers
// ---------------------------------------------------------------------------
const decisionColor = (d) =>
  d === "APPROVE" ? "#16a34a" : d === "REVIEW" ? "#d97706" : "#dc2626";

const scoreGradient = (s) => {
  if (s < 0.4) return "#16a34a";
  if (s < 0.8) return "#d97706";
  return "#dc2626";
};

// ---------------------------------------------------------------------------
// Shared styles
// ---------------------------------------------------------------------------
const card = {
  background: "#1e1e2e", border: "1px solid #2e2e4e",
  borderRadius: 10, padding: "20px 24px", marginBottom: 16,
};
const label = { fontSize: 12, color: "#94a3b8", marginBottom: 4, display: "block" };
const input = {
  width: "100%", background: "#0f0f1a", border: "1px solid #3e3e5e",
  borderRadius: 6, padding: "8px 12px", color: "#e2e8f0", fontSize: 14, boxSizing: "border-box",
};
const btn = (color = "#6366f1") => ({
  background: color, color: "#fff", border: "none", borderRadius: 6,
  padding: "10px 22px", fontSize: 14, cursor: "pointer", fontWeight: 600,
});

// ---------------------------------------------------------------------------
// Tab 1 — Live Scoring
// ---------------------------------------------------------------------------
function randCard() {
  return Array.from({ length: 16 }, () => Math.floor(Math.random() * 10)).join("");
}
function randDevice() {
  return "dev_" + Math.random().toString(36).slice(2, 10);
}
function randIp() {
  return `${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}`;
}

function LiveScoring() {
  const [form, setForm] = useState({
    amt: "125.00", category: "misc_net", hour: "14", merchant: "fraud_demo_merchant",
    state: "CA", is_weekend: false, geo_distance_km: "42",
    cc_num: randCard(), device_id: randDevice(), ip_prefix: randIp(),
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const set = (k) => (e) =>
    setForm((f) => ({ ...f, [k]: e.target.type === "checkbox" ? e.target.checked : e.target.value }));

  const reroll = () =>
    setForm((f) => ({ ...f, cc_num: randCard(), device_id: randDevice(), ip_prefix: randIp() }));

  const submit = async () => {
    setLoading(true);
    setError(null);
    try {
      const body = {
        trans_id: crypto.randomUUID?.() ?? Math.random().toString(36).slice(2),
        cc_num: form.cc_num,
        device_id: form.device_id,
        ip_prefix: form.ip_prefix,
        merchant: form.merchant,
        category: form.category,
        amt: parseFloat(form.amt) || 0,
        hour: parseInt(form.hour, 10) || 0,
        day_of_week: new Date().getDay(),
        is_weekend: form.is_weekend ? 1 : 0,
        is_night: parseInt(form.hour, 10) < 6 || parseInt(form.hour, 10) >= 22 ? 1 : 0,
        age: 35,
        geo_distance_km: parseFloat(form.geo_distance_km) || 0,
        city_pop: 150000,
        state: form.state,
        gender: "M",
        timestamp: Date.now() / 1000,
      };
      const res = await fetch(`${API}/score`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setResult(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
      {/* Left — form */}
      <div style={card}>
        <h3 style={{ margin: "0 0 16px", color: "#c4b5fd" }}>Transaction Details</h3>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          {[["Amount ($)", "amt", "number"], ["Hour (0-23)", "hour", "number"],
            ["Merchant", "merchant", "text"], ["Geo Distance (km)", "geo_distance_km", "number"]
          ].map(([lbl, key, type]) => (
            <div key={key}>
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
          <div style={{ display: "flex", alignItems: "center", gap: 8, paddingTop: 18 }}>
            <input type="checkbox" id="wknd" checked={form.is_weekend} onChange={set("is_weekend")} />
            <label htmlFor="wknd" style={{ color: "#94a3b8", fontSize: 13 }}>Weekend transaction</label>
          </div>
        </div>
        <div style={{ marginTop: 12, padding: "10px 14px", background: "#0f0f1a", borderRadius: 6, fontSize: 12, color: "#64748b" }}>
          <div>Card: <span style={{ color: "#94a3b8" }}>{form.cc_num.slice(0, 4)}••••{form.cc_num.slice(-4)}</span></div>
          <div>Device: <span style={{ color: "#94a3b8" }}>{form.device_id}</span></div>
          <div>IP prefix: <span style={{ color: "#94a3b8" }}>{form.ip_prefix}.x.x</span></div>
        </div>
        <div style={{ display: "flex", gap: 10, marginTop: 14 }}>
          <button style={btn()} onClick={submit} disabled={loading}>
            {loading ? "Scoring…" : "Score Transaction"}
          </button>
          <button style={btn("#374151")} onClick={reroll}>New Card</button>
        </div>
        {error && <div style={{ marginTop: 10, color: "#f87171", fontSize: 13 }}>{error}</div>}
      </div>

      {/* Right — result */}
      <div style={card}>
        <h3 style={{ margin: "0 0 16px", color: "#c4b5fd" }}>Scoring Result</h3>
        {!result ? (
          <div style={{ color: "#475569", textAlign: "center", paddingTop: 60 }}>
            Submit a transaction to see results
          </div>
        ) : (
          <>
            <div style={{
              background: decisionColor(result.decision) + "22",
              border: `2px solid ${decisionColor(result.decision)}`,
              borderRadius: 10, padding: "20px 24px", textAlign: "center", marginBottom: 16,
            }}>
              <div style={{ fontSize: 13, color: "#94a3b8", marginBottom: 4 }}>Decision</div>
              <div style={{ fontSize: 32, fontWeight: 800, color: decisionColor(result.decision) }}>
                {result.decision}
              </div>
              <div style={{ fontSize: 22, fontWeight: 700, color: scoreGradient(result.fraud_score), marginTop: 6 }}>
                {(result.fraud_score * 100).toFixed(1)}% fraud probability
              </div>
              <div style={{ fontSize: 12, color: "#64748b", marginTop: 4 }}>
                Total: {result.latency_ms}ms | Model: {result.model_latency_ms}ms
              </div>
            </div>

            {result.reasons?.length > 0 && (
              <div style={{ marginBottom: 14 }}>
                <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 8, fontWeight: 600 }}>
                  WHY THIS DECISION
                </div>
                {result.reasons.map((r, i) => (
                  <div key={i} style={{ display: "flex", gap: 8, alignItems: "flex-start", marginBottom: 6 }}>
                    <span style={{ color: decisionColor(result.decision), fontWeight: 700 }}>•</span>
                    <span style={{ fontSize: 13, color: "#cbd5e1" }}>{r}</span>
                  </div>
                ))}
              </div>
            )}

            {result.triggered_rules?.length > 0 && (
              <div>
                <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 6, fontWeight: 600 }}>
                  TRIGGERED RULES
                </div>
                {result.triggered_rules.map((rule, i) => (
                  <div key={i} style={{ background: "#0f0f1a", borderRadius: 6, padding: "6px 10px", marginBottom: 4, fontSize: 12, color: "#fbbf24" }}>
                    IF {(rule.antecedent || []).join(" AND ")} → conf: {(rule.confidence * 100).toFixed(0)}%
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>
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
    fetch(`${API}/entity-graph`)
      .then((r) => r.json())
      .then(setGraphData)
      .catch((e) => setError(e.message));
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

    const colorMap = { card: "#3b82f6", device: "#f97316", ip: "#a855f7", merchant: "#22c55e" };
    const fraudColor = "#ef4444";

    const sim = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id((d) => d.id).distance(80))
      .force("charge", d3.forceManyBody().strength(-120))
      .force("center", d3.forceCenter(W / 2, H / 2))
      .force("collision", d3.forceCollide(20));

    const link = g.append("g").selectAll("line").data(links).join("line")
      .attr("stroke", "#334155").attr("stroke-width", 1.5).attr("opacity", 0.6);

    const node = g.append("g").selectAll("circle").data(nodes).join("circle")
      .attr("r", (d) => 6 + Math.sqrt(d.txn_count ?? 1) * 2)
      .attr("fill", (d) => d.is_fraud ? fraudColor : (colorMap[d.type] || "#94a3b8"))
      .attr("stroke", "#1e1e2e").attr("stroke-width", 2).attr("cursor", "pointer")
      .on("mouseover", (_, d) => setHoveredNode(d))
      .on("mouseout", () => setHoveredNode(null))
      .call(d3.drag()
        .on("start", (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on("drag", (e, d) => { d.fx = e.x; d.fy = e.y; })
        .on("end", (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
      );

    sim.on("tick", () => {
      link.attr("x1", (d) => d.source.x).attr("y1", (d) => d.source.y)
          .attr("x2", (d) => d.target.x).attr("y2", (d) => d.target.y);
      node.attr("cx", (d) => d.x).attr("cy", (d) => d.y);
    });

    return () => sim.stop();
  }, [graphData, threshold]);

  return (
    <div style={card}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
        <h3 style={{ margin: 0, color: "#c4b5fd" }}>Entity Graph</h3>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={label}>Fraud rate threshold: {(threshold * 100).toFixed(0)}%</span>
          <input type="range" min={0} max={1} step={0.05} value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))} style={{ width: 140 }} />
        </div>
      </div>
      <div style={{ display: "flex", gap: 16, marginBottom: 10, fontSize: 12, color: "#94a3b8" }}>
        {[["Card","#3b82f6"],["Device","#f97316"],["IP","#a855f7"],["Merchant","#22c55e"],["Fraud","#ef4444"]].map(([t,c]) => (
          <div key={t} style={{ display: "flex", gap: 5, alignItems: "center" }}>
            <div style={{ width: 10, height: 10, borderRadius: "50%", background: c }} />
            {t}
          </div>
        ))}
      </div>
      {error
        ? <div style={{ color: "#f87171", padding: 20 }}>Failed to load graph: {error}. Ensure the API is running.</div>
        : <div style={{ position: "relative" }}>
            <svg ref={svgRef} style={{ background: "#0f0f1a", borderRadius: 8, width: "100%" }} />
            {hoveredNode && (
              <div style={{
                position: "absolute", top: 10, left: 10, background: "#1e1e2e",
                border: "1px solid #3e3e5e", borderRadius: 8, padding: "10px 14px", fontSize: 12,
              }}>
                <div style={{ color: "#c4b5fd", fontWeight: 700, marginBottom: 4 }}>{hoveredNode.id}</div>
                <div style={{ color: "#94a3b8" }}>Type: {hoveredNode.type}</div>
                <div style={{ color: "#94a3b8" }}>Transactions: {hoveredNode.txn_count ?? "—"}</div>
                <div style={{ color: hoveredNode.fraud_rate > 0.3 ? "#f87171" : "#94a3b8" }}>
                  Fraud rate: {((hoveredNode.fraud_rate ?? 0) * 100).toFixed(1)}%
                </div>
              </div>
            )}
          </div>
      }
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tab 3 — Drift Monitor (SVG line chart)
// ---------------------------------------------------------------------------
function DriftMonitor() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/drift`)
      .then((r) => r.json())
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);

  const months = data?.months ?? [];
  if (!months.length && !error) {
    return <div style={{ ...card, color: "#64748b" }}>Loading drift data…</div>;
  }
  if (error) {
    return <div style={{ ...card, color: "#f87171" }}>Failed: {error}</div>;
  }

  const W = 680, H = 260, pad = { top: 20, right: 20, bottom: 50, left: 50 };
  const iW = W - pad.left - pad.right;
  const iH = H - pad.top - pad.bottom;

  const xScale = (i) => (i / (months.length - 1 || 1)) * iW;
  const yScale = (v) => iH - ((v - 0.6) / 0.4) * iH;

  const linePath = (key) =>
    months.map((m, i) =>
      `${i === 0 ? "M" : "L"}${xScale(i).toFixed(1)},${yScale(m[key] ?? 0).toFixed(1)}`
    ).join(" ");

  return (
    <div style={card}>
      <h3 style={{ margin: "0 0 6px", color: "#c4b5fd" }}>Drift Monitor — Model AUC Over Time</h3>
      <p style={{ fontSize: 12, color: "#64748b", marginBottom: 16 }}>
        This chart shows concept drift — model performance over time. Red segments indicate AUC &lt; 0.85.
      </p>
      <svg width={W} height={H} style={{ overflow: "visible" }}>
        <g transform={`translate(${pad.left},${pad.top})`}>
          {/* Grid */}
          {[0.6,0.7,0.8,0.85,0.9,1.0].map((v) => (
            <g key={v}>
              <line x1={0} x2={iW} y1={yScale(v)} y2={yScale(v)} stroke="#1e2a3a" strokeWidth={1} />
              <text x={-8} y={yScale(v) + 4} fill="#475569" fontSize={10} textAnchor="end">{v.toFixed(2)}</text>
            </g>
          ))}
          {/* Threshold line */}
          <line x1={0} x2={iW} y1={yScale(0.85)} y2={yScale(0.85)} stroke="#dc2626" strokeWidth={1} strokeDasharray="4 3" />
          <text x={iW + 4} y={yScale(0.85) + 4} fill="#dc2626" fontSize={10}>0.85</text>

          {/* AUC line with color segments */}
          {months.slice(1).map((m, i) => {
            const prev = months[i];
            const avgAuc = ((m.auc ?? 0) + (prev.auc ?? 0)) / 2;
            const color = avgAuc < 0.85 ? "#ef4444" : "#22c55e";
            return (
              <line key={i}
                x1={xScale(i)} y1={yScale(prev.auc ?? 0)}
                x2={xScale(i + 1)} y2={yScale(m.auc ?? 0)}
                stroke={color} strokeWidth={2.5} />
            );
          })}

          {/* Precision@1% line */}
          <path d={linePath("precision_at_1pct")} fill="none" stroke="#6366f1" strokeWidth={1.8} strokeDasharray="5 3" />

          {/* Dots */}
          {months.map((m, i) => (
            <circle key={i} cx={xScale(i)} cy={yScale(m.auc ?? 0)} r={4}
              fill={(m.auc ?? 1) < 0.85 ? "#ef4444" : "#22c55e"} stroke="#0f0f1a" strokeWidth={1.5} />
          ))}

          {/* X-axis labels */}
          {months.map((m, i) => (
            <text key={i} x={xScale(i)} y={iH + 20} fill="#64748b" fontSize={10}
              textAnchor="middle" transform={`rotate(-40,${xScale(i)},${iH + 20})`}>
              {m.month ?? `M${i + 1}`}
            </text>
          ))}
        </g>
      </svg>
      <div style={{ display: "flex", gap: 20, marginTop: 8, fontSize: 12, color: "#94a3b8" }}>
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <div style={{ width: 20, height: 3, background: "#22c55e" }} /> AUC ≥ 0.85
        </div>
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <div style={{ width: 20, height: 3, background: "#ef4444" }} /> AUC &lt; 0.85
        </div>
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <div style={{ width: 20, height: 3, background: "#6366f1", borderTop: "2px dashed #6366f1" }} /> Precision@1%
        </div>
      </div>
    </div>
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
    fetch(`${API}/fraud-rules`)
      .then((r) => r.json())
      .then((d) => setRules(d.rules ?? []))
      .catch((e) => setError(e.message));
  }, []);

  const sorted = [...rules].sort((a, b) => (b[sortKey] ?? 0) - (a[sortKey] ?? 0));
  const maxLift = Math.max(1, ...rules.map((r) => r.lift ?? 0));
  const Bar = ({ val, max, color }) => (
    <div style={{ background: "#0f0f1a", borderRadius: 4, height: 8, width: 100, overflow: "hidden" }}>
      <div style={{ width: `${(val / max) * 100}%`, height: "100%", background: color, borderRadius: 4 }} />
    </div>
  );

  return (
    <div style={card}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <h3 style={{ margin: 0, color: "#c4b5fd" }}>FP-Growth Fraud Rules</h3>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <span style={{ fontSize: 12, color: "#64748b" }}>Sort by:</span>
          {["lift", "confidence", "support"].map((k) => (
            <button key={k} style={{
              ...btn(sortKey === k ? "#6366f1" : "#1e2a3a"),
              padding: "5px 12px", fontSize: 12,
            }} onClick={() => setSortKey(k)}>{k}</button>
          ))}
        </div>
      </div>
      {error
        ? <div style={{ color: "#f87171" }}>Failed: {error}</div>
        : rules.length === 0
          ? <div style={{ color: "#64748b" }}>No rules loaded. Run the training pipeline to generate FP-Growth rules.</div>
          : <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead>
                  <tr>
                    {["Antecedent", "Confidence", "Lift", "Support"].map((h) => (
                      <th key={h} style={{ textAlign: "left", padding: "8px 12px", color: "#94a3b8",
                        borderBottom: "1px solid #2e2e4e", whiteSpace: "nowrap" }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sorted.map((rule, i) => (
                    <tr key={i} style={{ borderBottom: "1px solid #1e1e2e" }}
                      onMouseEnter={(e) => (e.currentTarget.style.background = "#1a1a2e")}
                      onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}>
                      <td style={{ padding: "10px 12px", color: "#e2e8f0" }}>
                        {(rule.antecedent ?? []).map((a) => (
                          <span key={a} style={{
                            display: "inline-block", background: "#2e2e4e", borderRadius: 4,
                            padding: "2px 7px", marginRight: 4, marginBottom: 2, fontSize: 11,
                          }}>{a}</span>
                        ))}
                        {!rule.antecedent?.length && <span style={{ color: "#475569" }}>—</span>}
                      </td>
                      <td style={{ padding: "10px 12px" }}>
                        <div style={{ color: "#fbbf24", marginBottom: 4 }}>{((rule.confidence ?? 0) * 100).toFixed(1)}%</div>
                        <Bar val={rule.confidence ?? 0} max={1} color="#fbbf24" />
                      </td>
                      <td style={{ padding: "10px 12px" }}>
                        <div style={{ color: "#f97316", marginBottom: 4 }}>{(rule.lift ?? 0).toFixed(2)}×</div>
                        <Bar val={rule.lift ?? 0} max={maxLift} color="#f97316" />
                      </td>
                      <td style={{ padding: "10px 12px", color: "#94a3b8" }}>
                        {((rule.support ?? 0) * 100).toFixed(2)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
      }
    </div>
  );
}

// ---------------------------------------------------------------------------
// Root App
// ---------------------------------------------------------------------------
const TABS = ["Live Scoring", "Fraud Ring Graph", "Drift Monitor", "Rule Explorer"];

export default function App() {
  const [tab, setTab] = useState(0);

  return (
    <div style={{ minHeight: "100vh", background: "#0f0f1a", color: "#e2e8f0", fontFamily: "'Inter','Segoe UI',sans-serif" }}>
      {/* Header */}
      <div style={{ background: "#1e1e2e", borderBottom: "1px solid #2e2e4e", padding: "0 32px" }}>
        <div style={{ maxWidth: 1200, margin: "0 auto", display: "flex", alignItems: "center", gap: 32 }}>
          <div style={{ padding: "18px 0", display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{ width: 10, height: 10, borderRadius: "50%", background: "#ef4444", boxShadow: "0 0 8px #ef4444" }} />
            <span style={{ fontWeight: 700, fontSize: 17, letterSpacing: 0.3 }}>Fraud Detection System</span>
          </div>
          <nav style={{ display: "flex", gap: 4 }}>
            {TABS.map((t, i) => (
              <button key={t} onClick={() => setTab(i)} style={{
                background: "none", border: "none", cursor: "pointer",
                padding: "20px 16px", fontSize: 13, fontWeight: 500,
                color: tab === i ? "#c4b5fd" : "#64748b",
                borderBottom: tab === i ? "2px solid #c4b5fd" : "2px solid transparent",
                transition: "color 0.15s",
              }}>{t}</button>
            ))}
          </nav>
        </div>
      </div>

      {/* Content */}
      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "28px 32px" }}>
        {tab === 0 && <LiveScoring />}
        {tab === 1 && <FraudRingGraph />}
        {tab === 2 && <DriftMonitor />}
        {tab === 3 && <RuleExplorer />}
      </div>
    </div>
  );
}
