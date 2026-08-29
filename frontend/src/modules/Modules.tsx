import React, { useEffect, useState } from "react";
import { useStore, TEMPLATES } from "../store";
import { Icon, StatusChip, Info } from "../ui";
import { stepReason } from "../explain";
import * as api from "../api";

function Head({ title, sub, right }: { title: string; sub: string; right?: React.ReactNode }) {
  return (
    <div className="modhd">
      <div><h1>{title}</h1><div className="sub">{sub}</div></div>
      {right}
    </div>
  );
}
function verdictStatus(v: any) { return v === true ? "proven" : v === false ? "violated" : "pending"; }

// ── Run ──
export function Run() {
  const { runInput, runSource, runUrl, runMemory, runTrace, runHealth, running, provider, set, runFlow } = useStore();
  const hasMulti = runHealth && Object.keys(runHealth).length > 0;
  // Input source is a universal Text/URL/File loader for ANY agent — persisted in
  // the store so it never reverts. No per-agent heuristics.
  const mode = runSource;
  const setMode = (m: "text" | "url" | "file") => set({ runSource: m });
  const [fetching, setFetching] = useState(false);
  const [fetched, setFetched] = useState<any>(null);
  const [showMem, setShowMem] = useState(false);
  const doFetch = async () => {
    if (!runUrl.trim()) return;
    setFetching(true);
    const r = await api.fetchUrl(runUrl);
    setFetching(false);
    if (r.error) { set({ toast: r.error }); return; }
    setFetched(r); set({ runInput: r.text });
  };
  return (
    <div style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <Head title="Run" sub={`execute the agent end-to-end · provider ${provider}`}
        right={<button className="btn pri" onClick={runFlow} disabled={running}><Icon name="play" size={14} /> {running ? "Running…" : "Run agent"}</button>} />
      <div className="cols" style={{ overflow: "hidden" }}>
        <div className="form">
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
            <span className="lbl">Input source</span>
            <div className="seg">
              {(["text", "url", "file"] as const).map((m) => (
                <button key={m} className={mode === m ? "on" : ""} onClick={() => setMode(m)}>{m === "text" ? "Text" : m === "url" ? "URL" : "File"}</button>
              ))}
            </div>
          </div>

          {mode === "text" && (
            <div className="fg"><span className="lbl">Input message</span>
              <textarea className="field" style={{ minHeight: 130 }} value={runInput} placeholder="What the agent receives — a customer message, a ticket, an invoice line…"
                onChange={(e) => set({ runInput: e.target.value })} /></div>
          )}
          {mode === "url" && (
            <div className="fg"><span className="lbl">Fetch a web page (runs locally)</span>
              <div style={{ display: "flex", gap: 8 }}>
                <input className="field mono" value={runUrl} placeholder="example.com" onChange={(e) => set({ runUrl: e.target.value })} onKeyDown={(e) => { if (e.key === "Enter") doFetch(); }} />
                <button className="btn" onClick={doFetch} disabled={fetching}>{fetching ? "Fetching…" : "Fetch"}</button>
              </div>
              {fetched && <div className="hint" style={{ marginTop: 8 }}>✓ fetched <b>{fetched.title || fetched.url}</b> — {fetched.chars} chars stripped to text, loaded as the input.</div>}
              {fetched && <pre style={{ marginTop: 8, maxHeight: 120 }}>{(runInput || "").slice(0, 600)}{runInput.length > 600 ? " …" : ""}</pre>}
              {!fetched && <div className="hint" style={{ marginTop: 8 }}>Your machine fetches the page and strips it to text — then it flows through the agent as untrusted input (which is exactly what the taint analysis guards).</div>}
            </div>
          )}
          {mode === "file" && (
            <div className="fg"><span className="lbl">Load a text file</span>
              <label className="btn" style={{ cursor: "pointer", width: "fit-content" }}><Icon name="download" size={14} /> Choose file
                <input type="file" accept=".txt,.md,.json,.csv,text/*" style={{ display: "none" }}
                  onChange={(e) => { const f = e.target.files?.[0]; if (!f) return; const r = new FileReader(); r.onload = () => set({ runInput: String(r.result).slice(0, 8000) }); r.readAsText(f); }} /></label>
              {runInput && <pre style={{ marginTop: 8, maxHeight: 120 }}>{runInput.slice(0, 600)}{runInput.length > 600 ? " …" : ""}</pre>}
            </div>
          )}

          <button className="addf" style={{ marginTop: 4 }} onClick={() => setShowMem(!showMem)}>{showMem ? "▾" : "▸"} seed memory (advanced)</button>
          {showMem && (
            <div className="fg" style={{ marginTop: 8 }}>
              <textarea className="field mono" style={{ minHeight: 60 }} value={runMemory} placeholder='{ "order_id": "A-2291", "tier": "gold" }'
                onChange={(e) => set({ runMemory: e.target.value })} />
              <div className="hint">Key-value context the agent starts with — e.g. facts a decision node needs before the first LLM call.</div>
            </div>
          )}

          <div className="hint" style={{ marginTop: 12 }}>Extract + Decision nodes run for real (LLM + rules, on your machine with <b>{provider}</b>). <b>Tool nodes are not executed</b> — Aura proves their preconditions and uses your mock return; you bind the real tools in code / aura-runtime.</div>
        </div>
        <div className="out">
          {!runTrace && <div className="empty" style={{ marginTop: 60 }}>No run yet. Enter an input and run the agent<br />to see the verified execution trace.</div>}
          {hasMulti && (
            <div style={{ marginBottom: 16 }}>
              <div className="shd" style={{ marginTop: 0 }}><span className="lbl">Runtime health<Info k="health" /></span><span className="solver">adaptive router</span></div>
              {Object.entries(runHealth!).map(([node, h]: any) => (
                <div key={node} className="obl-item" style={{ padding: "8px 11px", marginBottom: 6 }}>
                  <div className="top"><span className="expr">{node}</span>
                    <span className="mono" style={{ fontSize: 10.5, color: "var(--ink-3)" }}>{h.total_executions}× · {h.avg_latency_ms}ms · fail {(h.fail_rate * 100).toFixed(0)}%</span></div>
                </div>
              ))}
              <div className="hint">When a node has multiple valid next steps, the engine routes with a Thompson-sampling bandit over these signals — restricted to CTL-feasible transitions.<Info k="routing" /></div>
            </div>
          )}
          {runTrace && runTrace.map((s: any, i: number) => (
            <div key={i} className="step">
              <div className="sh">
                <div className="nm"><span className="mono" style={{ color: "var(--ink-3)" }}>{String(i + 1).padStart(2, "0")}</span> {s.node}
                  {s.verified !== undefined && s.verified !== null && <StatusChip status={verdictStatus(s.verified)} />}
                  {s.abstained && <span className="chip pn">abstained</span>}
                </div>
                {s.next && <span className="arrow">→ {s.next}</span>}
              </div>
              {s.error && <div className="cx" style={{ marginTop: 8 }}>{s.error}</div>}
              {s.extracted && Object.keys(s.extracted).length > 0 && (
                <pre style={{ marginTop: 9 }}>{JSON.stringify(s.extracted, null, 2)}</pre>)}
              <div style={{ fontSize: 12, lineHeight: 1.5, color: "var(--ink-2)", marginTop: 9 }}>{stepReason(s)}</div>
              <div className="meta" style={{ fontFamily: "JetBrains Mono", fontSize: 10, color: "var(--ink-3)", marginTop: 8, display: "flex", gap: 14, flexWrap: "wrap" }}>
                {s.ms != null && <span>{s.ms} ms</span>}
                {s.model && <span>{s.provider ? s.provider + " · " : ""}{s.model}</span>}
                {s.consensus > 1 && <span>consensus ×{s.consensus}</span>}
                {s.iterations != null && <span>retries {s.iterations}</span>}
                {s.conformal && <span>conformal {(s.conformal.covered || []).length} covered{s.conformal.lower != null ? ` · [${s.conformal.lower}, ${s.conformal.upper}]` : ""}</span>}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Prove ──
export function Prove() {
  const [data, setData] = useState('{ "amount": 250, "total": 300 }');
  const [obl, setObl] = useState("amount >= 0\namount <= 500");
  const [res, setRes] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const run = async () => {
    setBusy(true);
    let d: any = {}; try { d = JSON.parse(data); } catch {}
    setRes(await api.proveData(d, obl.split("\n").map((s) => s.trim()).filter(Boolean))); setBusy(false);
  };
  return (
    <div style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <Head title="Prove" sub="Z3 point-check + symbolic consistency on any data"
        right={<button className="btn pri" onClick={run} disabled={busy}><Icon name="prove" size={14} /> {busy ? "Proving…" : "Prove"}</button>} />
      <div className="cols" style={{ overflow: "hidden" }}>
        <div className="form">
          <div className="fg"><span className="lbl">Data (JSON)</span>
            <textarea className="field mono" style={{ minHeight: 100 }} value={data} onChange={(e) => setData(e.target.value)} /></div>
          <div className="fg"><span className="lbl">Obligations</span>
            <textarea className="field mono" style={{ minHeight: 90 }} value={obl} onChange={(e) => setObl(e.target.value)} /></div>
        </div>
        <div className="out">
          {!res && <div className="empty" style={{ marginTop: 60 }}>Prove any record against Z3 obligations.<br />Fail-closed: unprovable ⇒ not verified.</div>}
          {res && (
            <>
              <div className="metric"><div className="k">Verdict</div>
                <div className="big" style={{ color: res.verified ? "var(--proven)" : "var(--violated)" }}>{res.verified ? "PROVEN" : "NOT PROVEN"}</div></div>
              {res.failed?.length > 0 && <div className="obl-item bad"><div className="lbl" style={{ color: "var(--violated)", marginBottom: 6 }}>Failed</div><div className="cx">{res.failed.join("  ·  ")}</div></div>}
              {res.unproven?.length > 0 && <div className="obl-item"><div className="lbl" style={{ marginBottom: 6 }}>Unproven</div><div className="meta">{res.unproven.join("  ·  ")}</div></div>}
              {res.counterexample && <div className="obl-item bad"><div className="lbl" style={{ color: "var(--violated)", marginBottom: 6 }}>Counterexample</div><div className="cx">{typeof res.counterexample === "string" ? res.counterexample : JSON.stringify(res.counterexample)}</div></div>}
              <div className="metric"><div className="k">Symbolic consistency</div>
                <div className="big" style={{ fontSize: 20, color: res.consistent ? "var(--proven)" : "var(--violated)" }}>{res.consistent ? "SAT" : "UNSAT"}</div>
                {res.witness && <div className="hint mono">witness · {typeof res.witness === "string" ? res.witness : JSON.stringify(res.witness)}</div>}
                {res.reason && <div className="hint">{res.reason}</div>}</div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Dataset ──
export function Data() {
  const [recs, setRecs] = useState('[\n  { "amount": 120 },\n  { "amount": -5 },\n  { "amount": 700 }\n]');
  const [obl, setObl] = useState("amount >= 0\namount <= 500");
  const [res, setRes] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const run = async () => {
    setBusy(true);
    let r: any[] = []; try { const p = JSON.parse(recs); r = Array.isArray(p) ? p : [p]; } catch {}
    setRes(await api.verifyDataset(r, obl.split("\n").map((s) => s.trim()).filter(Boolean))); setBusy(false);
  };
  return (
    <div style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <Head title="Dataset" sub="bulk-verify a real dataset against obligations"
        right={<button className="btn pri" onClick={run} disabled={busy}><Icon name="prove" size={14} /> {busy ? "Verifying…" : "Verify dataset"}</button>} />
      <div className="cols" style={{ overflow: "hidden" }}>
        <div className="form">
          <div className="fg"><span className="lbl">Records (JSON array)</span>
            <textarea className="field mono" style={{ minHeight: 160 }} value={recs} onChange={(e) => setRecs(e.target.value)} /></div>
          <div className="fg"><span className="lbl">Obligations</span>
            <textarea className="field mono" style={{ minHeight: 80 }} value={obl} onChange={(e) => setObl(e.target.value)} /></div>
        </div>
        <div className="out">
          {!res && <div className="empty" style={{ marginTop: 60 }}>Paste rows from a real dataset.<br />Every row is proved on your machine.</div>}
          {res && (
            <>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <div className="metric"><div className="k">Passed</div><div className="big" style={{ color: "var(--proven)" }}>{res.passed}</div></div>
                <div className="metric"><div className="k">Failed</div><div className="big" style={{ color: res.failed ? "var(--violated)" : "" }}>{res.failed}</div></div>
              </div>
              <div className="hint">{res.total} rows · {res.obligations} obligations · ~{res.rate} rows/s</div>
              {res.violations?.length > 0 && res.violations.map((v: any, i: number) => (
                <div key={i} className="obl-item bad">
                  <div className="top"><span className="expr">row {v.row}</span><span className="chip vi">✕ {(v.failed || []).length} failed</span></div>
                  <pre style={{ marginTop: 8 }}>{JSON.stringify(v.record)}</pre>
                  <div className="cx" style={{ marginTop: 6 }}>{(v.failed || []).join("  ·  ")}</div>
                </div>
              ))}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Monitor ──
export function Monitor() {
  const [feed, setFeed] = useState<any[]>([]);
  useEffect(() => {
    let alive = true;
    const tick = async () => { try { const f = await api.feed(); if (alive) setFeed(f); } catch {} };
    tick(); const iv = setInterval(tick, 2000);
    return () => { alive = false; clearInterval(iv); };
  }, []);
  const snippet = `from aura_state.hooks import Monitor

mon = Monitor()  # -> http://localhost:8155
mon.ingest(node="Classify",
           data={"category": "billing", "amount": 40},
           obligations=["amount >= 0", "amount <= 500"])`;
  return (
    <div style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <Head title="Monitor" sub="your real agent streams verified outputs here (SDK → studio)"
        right={<button className="btn sm" onClick={async () => { await api.clearFeed(); setFeed([]); }}><Icon name="trash" size={13} /> Clear</button>} />
      <div className="cols" style={{ overflow: "hidden" }}>
        <div className="form">
          <div className="lbl" style={{ marginBottom: 8 }}>Connect your agent</div>
          <pre>{snippet}</pre>
          <div className="hint">The <code>Monitor</code> client posts each real extraction; the studio proves it against the obligations and streams the verdict below. Works from CrewAI, LangGraph, or plain code.</div>
        </div>
        <div className="out" style={{ padding: 0 }}>
          {feed.length > 0 && <MonitorStats feed={feed} />}
          {feed.length === 0 && <div className="empty" style={{ marginTop: 60 }}>Waiting for events…<br />Run the snippet on the left.</div>}
          {feed.map((e, i) => (
            <div key={i} className="feedrow">
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <StatusChip status={verdictStatus(e.verified)} />
                <b className="mono">{e.node}</b>
                {e.source && <span className="mono" style={{ color: "var(--ink-3)", fontSize: 11 }}>{e.source}</span>}
                <span className="mono" style={{ marginLeft: "auto", color: "var(--ink-3)", fontSize: 11 }}>{e.ts}</span>
              </div>
              <pre style={{ marginTop: 8 }}>{JSON.stringify(e.data)}</pre>
              {e.failed?.length > 0 && <div className="cx" style={{ marginTop: 6 }}>failed · {e.failed.join(", ")}</div>}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function MonitorStats({ feed }: { feed: any[] }) {
  const total = feed.length;
  const verified = feed.filter((e) => e.verified === true).length;
  const failed = feed.filter((e) => e.verified === false).length;
  const rate = total ? verified / total : 0;
  const byNode: Record<string, { pass: number; total: number }> = {};
  feed.forEach((e) => {
    const k = e.node || "—";
    byNode[k] = byNode[k] || { pass: 0, total: 0 };
    byNode[k].total++; if (e.verified === true) byNode[k].pass++;
  });
  // chronological order for the sparkline (feed is newest-first)
  const series = [...feed].reverse().slice(-48);
  const W = 300, H = 34, bw = W / Math.max(series.length, 1);
  return (
    <div style={{ padding: "16px 16px 4px", borderBottom: "1px solid var(--line)" }}>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10, marginBottom: 14 }}>
        <div className="metric" style={{ margin: 0 }}><div className="k">Verified rate<Info k="obligation" /></div><div className="big" style={{ fontSize: 24, color: rate >= 0.9 ? "var(--proven)" : rate >= 0.6 ? "var(--pending)" : "var(--violated)" }}>{(rate * 100).toFixed(0)}%</div></div>
        <div className="metric" style={{ margin: 0 }}><div className="k">Verified</div><div className="big" style={{ fontSize: 24, color: "var(--proven)" }}>{verified}</div></div>
        <div className="metric" style={{ margin: 0 }}><div className="k">Failed</div><div className="big" style={{ fontSize: 24, color: failed ? "var(--violated)" : "" }}>{failed}</div></div>
      </div>
      <div className="lbl" style={{ marginBottom: 6 }}>Event stream · last {series.length}</div>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ marginBottom: 14 }}>
        {series.map((e, i) => (
          <rect key={i} x={i * bw + 0.5} y={e.verified === false ? H * 0.45 : 2} width={Math.max(bw - 1.5, 1)}
            height={e.verified === false ? H * 0.55 : H - 4}
            fill={e.verified === true ? "var(--proven)" : e.verified === false ? "var(--violated)" : "var(--ink-4)"} rx="1" />
        ))}
      </svg>
      <div className="lbl" style={{ marginBottom: 6 }}>Pass rate by node</div>
      {Object.entries(byNode).map(([k, v]) => (
        <div key={k} style={{ marginBottom: 8 }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontFamily: "JetBrains Mono", fontSize: 11, marginBottom: 3 }}>
            <span>{k}</span><span style={{ color: "var(--ink-3)" }}>{v.pass}/{v.total}</span>
          </div>
          <div className="bar"><i style={{ width: `${(v.pass / v.total) * 100}%`, background: v.pass === v.total ? "var(--proven)" : "var(--pending)" }} /></div>
        </div>
      ))}
    </div>
  );
}

// ── Calibrate (conformal + risk) ──
export function Calibrate() {
  const [ctab, setCtab] = useState<"conformal" | "risk">("conformal");
  return (
    <div style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <Head title="Calibrate" sub="distribution-free guarantees: conformal intervals + risk-controlled abstention"
        right={<div className="seg">
          <button className={ctab === "conformal" ? "on" : ""} onClick={() => setCtab("conformal")}>Conformal</button>
          <button className={ctab === "risk" ? "on" : ""} onClick={() => setCtab("risk")}>Risk</button>
        </div>} />
      {ctab === "conformal" ? <Conformal /> : <Risk />}
    </div>
  );
}

function Conformal() {
  const [vals, setVals] = useState("12.1, 12.4, 11.8, 12.0, 12.6, 11.9, 12.3, 12.2, 12.5, 11.7, 12.0, 12.1, 12.4, 11.9, 12.2, 12.3, 12.0, 12.1, 12.5, 11.8");
  const [conf, setConf] = useState(0.9);
  const [res, setRes] = useState<any>(null);
  const run = async () => setRes(await api.conformal({ values: vals.split(",").map((x) => +x.trim()).filter((x) => !isNaN(x)), confidence: conf }));
  return (
    <div className="cols" style={{ overflow: "hidden" }}>
      <div className="form">
        <div className="fg"><span className="lbl">Calibration values</span>
          <textarea className="field mono" style={{ minHeight: 110 }} value={vals} onChange={(e) => setVals(e.target.value)} /></div>
        <div className="fg"><span className="lbl">Confidence {(conf * 100).toFixed(0)}%<Info k="conformal" /></span>
          <div className="rangewrap"><input type="range" min={0.5} max={0.99} step={0.01} value={conf} onChange={(e) => setConf(+e.target.value)} /><span className="rangeval">{conf.toFixed(2)}</span></div></div>
        <button className="btn pri" onClick={run}><Icon name="calibrate" size={14} /> Compute interval</button>
      </div>
      <div className="out">
        {!res && <div className="empty" style={{ marginTop: 60 }}>Split-conformal interval with a<br />finite-sample coverage guarantee.</div>}
        {res && (
          <>
            <div className="metric"><div className="k">{res.mode === "pasc" ? "PASC q̂" : "Interval"}</div>
              <div className="big">{res.mode === "pasc" ? (res.q_hat ?? "—") : `[${res.lower ?? "−∞"}, ${res.upper ?? "∞"}]`}</div></div>
            <div className="metric"><div className="k">Calibrated</div>
              <div className="big" style={{ fontSize: 20, color: res.calibrated ? "var(--proven)" : "var(--pending)" }}>{res.calibrated ? "YES" : "UNCALIBRATED"}</div>
              <div className="hint">{res.calibrated ? `n = ${res.n} ≥ floor` : `need more samples at this confidence (n = ${res.n})`}</div></div>
            <div className="hint" style={{ lineHeight: 1.6 }}>{res.calibrated
              ? `In plain terms: over many predictions, at least ${(conf * 100).toFixed(0)}% of the true values will fall inside this interval — a finite-sample guarantee that assumes nothing about the model. The remaining ${(100 - conf * 100).toFixed(0)}% is your risk budget (α = ${(1 - conf).toFixed(2)}).`
              : `Not enough calibration samples to guarantee ${(conf * 100).toFixed(0)}% coverage yet. Add more values (or lower the confidence) — Aura reports this honestly rather than pretending.`}</div>
          </>
        )}
      </div>
    </div>
  );
}

function Risk() {
  const [scores, setScores] = useState("0.9, 0.8, 0.75, 0.6, 0.95, 0.4, 0.55, 0.85, 0.3, 0.7, 0.88, 0.5, 0.92, 0.65, 0.78, 0.45, 0.82, 0.6, 0.9, 0.35");
  const [correct, setCorrect] = useState("1,1,1,0,1,0,1,1,0,1,1,0,1,1,1,0,1,0,1,0");
  const [eps, setEps] = useState(0.1);
  const [test, setTest] = useState(0.72);
  const [res, setRes] = useState<any>(null);
  const run = async () => setRes(await api.risk({
    scores: scores.split(",").map((x) => +x.trim()),
    correct: correct.split(",").map((x) => x.trim() === "1" || x.trim().toLowerCase() === "true"),
    epsilon: eps, test_score: test,
  }));
  return (
    <div className="cols" style={{ overflow: "hidden" }}>
      <div className="form">
        <div className="fg"><span className="lbl">Confidence scores</span><textarea className="field mono" style={{ minHeight: 70 }} value={scores} onChange={(e) => setScores(e.target.value)} /></div>
        <div className="fg"><span className="lbl">Correct? (1/0)</span><textarea className="field mono" style={{ minHeight: 50 }} value={correct} onChange={(e) => setCorrect(e.target.value)} /></div>
        <div className="two">
          <div className="fg"><span className="lbl">ε (max false-action rate)</span><input className="field mono" type="number" step={0.01} value={eps} onChange={(e) => setEps(+e.target.value)} /></div>
          <div className="fg"><span className="lbl">Test score</span><input className="field mono" type="number" step={0.01} value={test} onChange={(e) => setTest(+e.target.value)} /></div>
        </div>
        <button className="btn pri" onClick={run}><Icon name="calibrate" size={14} /> Calibrate threshold</button>
      </div>
      <div className="out">
        {!res && <div className="empty" style={{ marginTop: 60 }}>Conformal Risk Control: act only when<br />the false-action rate is provably ≤ ε.</div>}
        {res && (
          <>
            <div className="metric"><div className="k">Decision at test score {res.test_score}</div>
              <div className="big" style={{ color: res.decision === "act" ? "var(--proven)" : "var(--pending)" }}>{(res.decision || "—").toUpperCase()}</div></div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <div className="metric"><div className="k">Threshold</div><div className="big" style={{ fontSize: 22 }}>{res.threshold ?? "—"}</div></div>
              <div className="metric"><div className="k">Realized FAR</div><div className="big" style={{ fontSize: 22 }}>{(res.realized_false_action_rate * 100).toFixed(1)}%</div></div>
            </div>
            <div className="hint">{res.can_act ? `Can act on ${res.acted}/${res.n} at ε=${res.epsilon}.` : "Cannot act at this ε — abstains everywhere."}</div>
          </>
        )}
      </div>
    </div>
  );
}

function ProviderRow({ p }: { p: any }) {
  const [test, setTest] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const run = async () => { setBusy(true); setTest(await api.testProvider(p.name)); setBusy(false); };
  return (
    <div className="obl-item"><div className="top">
      <span className="expr">{p.name} <span style={{ color: "var(--ink-3)" }}>· {p.model}</span></span>
      <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
        {test && <span className={"chip " + (test.ok ? "pv" : "vi")}>{test.ok ? "✓ " + test.detail : "✕ " + test.detail}</span>}
        {!test && <span className={"chip " + (p.available ? "pv" : "mu")}>{p.available ? "ready" : "set " + p.needs}</span>}
        <button className="btn sm" onClick={run} disabled={busy}>{busy ? "…" : "Test"}</button>
      </span>
    </div></div>
  );
}

// ── Memory (context pruning) ──
export function Memory() {
  const [hist, setHist] = useState(JSON.stringify([
    { role: "system", content: "You are a refund support agent." },
    { role: "user", content: "Hi, order A-2291." },
    { role: "assistant", content: "What's the issue?" },
    { role: "user", content: "It arrived damaged." },
    { role: "assistant", content: "Got it, checking policy." },
    { role: "user", content: "Also the box was crushed." },
  ], null, 2));
  const [maxMsg, setMaxMsg] = useState(3);
  const [keys, setKeys] = useState("order_id");
  const [res, setRes] = useState<any>(null);
  const run = async () => {
    let h: any[] = []; try { h = JSON.parse(hist); } catch {}
    setRes(await api.memoryPreview(h, maxMsg, keys.split(",").map((k) => k.trim()).filter(Boolean)));
  };
  return (
    <div style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <Head title="Memory" sub="context pruning — keep the system prompt + last N, inject required keys"
        right={<button className="btn pri" onClick={run}><Icon name="data" size={14} /> Prune preview</button>} />
      <div className="cols" style={{ overflow: "hidden" }}>
        <div className="form">
          <div className="fg"><span className="lbl">Conversation history (JSON)<Info k="memory" /></span>
            <textarea className="field mono" style={{ minHeight: 180 }} value={hist} onChange={(e) => setHist(e.target.value)} /></div>
          <div className="two">
            <div className="fg"><span className="lbl">Max messages</span><input className="field mono" type="number" value={maxMsg} onChange={(e) => setMaxMsg(+e.target.value)} /></div>
            <div className="fg"><span className="lbl">Required keys</span><input className="field mono" value={keys} onChange={(e) => setKeys(e.target.value)} /></div>
          </div>
          <div className="hint">The real <code>ContextPruner</code> keeps the system prompt and the last N turns, and injects your required keys — so long chats stay in the model's window without losing what matters.</div>
        </div>
        <div className="out">
          {!res && <div className="empty" style={{ marginTop: 60 }}>Prune a conversation to fit the context window<br />while preserving what the agent needs.</div>}
          {res && (<>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <div className="metric"><div className="k">Before</div><div className="big">{res.before}</div></div>
              <div className="metric"><div className="k">After</div><div className="big" style={{ color: "var(--proven)" }}>{res.after}</div></div>
            </div>
            <div className="lbl" style={{ margin: "14px 0 8px" }}>Pruned context</div>
            {res.pruned.map((m: any, i: number) => (
              <div key={i} className="obl-item" style={{ padding: "8px 11px", marginBottom: 6 }}>
                <div className="top"><span className="chip mu">{m.role}</span></div>
                <div style={{ fontSize: 12, color: "var(--ink-2)", marginTop: 5 }}>{m.content}</div>
              </div>
            ))}
          </>)}
        </div>
      </div>
    </div>
  );
}

// ── SDK / Connect ──
function CopyBlock({ code }: { code: string }) {
  const [ok, setOk] = useState(false);
  const copy = () => { navigator.clipboard?.writeText(code); setOk(true); setTimeout(() => setOk(false), 1200); };
  return (
    <div style={{ position: "relative", marginBottom: 18 }}>
      <button className="btn sm" style={{ position: "absolute", top: 8, right: 8, zIndex: 2 }} onClick={copy}>{ok ? "copied" : "copy"}</button>
      <pre style={{ paddingRight: 70 }}>{code}</pre>
    </div>
  );
}

export function Sdk() {
  return (
    <div style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <Head title="SDK · use it in code" sub="the studio is optional — the library is the product" />
      <div className="out" style={{ maxWidth: 760 }}>
        <div className="shd" style={{ marginTop: 2 }}><span className="lbl">Install</span></div>
        <CopyBlock code={`pip install aura-state          # the library\npip install "aura-state[ui]"    # + this local studio\naura-state ui                   # launch the studio`} />

        <div className="shd"><span className="lbl">Prove any output</span><span className="solver">z3, fail-closed</span></div>
        <div className="hint" style={{ marginTop: 0, marginBottom: 8 }}>No agent required — prove a dict against obligations.</div>
        <CopyBlock code={`from aura_state.hooks import verify

result = verify(
    {"area": 100, "rate": 3, "total": 300},
    ["total == area * rate", "area > 0"],
)
print(result.verified)         # True
print(result.counterexample)   # None`} />

        <div className="shd"><span className="lbl">Guard a real function</span><span className="solver">@verified · strict</span></div>
        <div className="hint" style={{ marginTop: 0, marginBottom: 8 }}>Wrap any function that returns a dict/model. <code>strict=True</code> raises on an unproven output; <code>monitor</code> streams every call to the Monitor tab.</div>
        <CopyBlock code={`from aura_state.hooks import verified, Monitor, VerificationError

mon = Monitor()  # -> http://127.0.0.1:8155

@verified(["total == area * rate"], monitor=mon, strict=True)
def price(area, rate):
    return {"area": area, "rate": rate, "total": area * rate}

try:
    price(100, 3)              # proven -> returns
except VerificationError:
    ...                        # unproven -> raises (fail-closed)`} />

        <div className="shd"><span className="lbl">Stream from CrewAI / LangGraph / any agent</span></div>
        <CopyBlock code={`from aura_state.hooks import Monitor

mon = Monitor()
mon.ingest(
    node="Classify",
    data={"category": "billing", "amount": 40},
    obligations=["amount >= 0", "amount <= 500"],
)  # verdict appears live in the Monitor tab`} />

        <div className="shd"><span className="lbl">Build a verified graph in code</span></div>
        <CopyBlock code={`from aura_state.core.engine import AuraEngine, Node

class Price(Node):
    system_prompt = "price it"
    obligations = ["total == area * rate"]
    def handle(self, user_text, extracted_data=None, memory=None):
        return "END", {}

engine = AuraEngine()
engine.register(Price)
# design-time proof: Z3 + CTL + taint, before you ship
print(engine.compile_contract())`} />
        <div className="shd"><span className="lbl">Adaptive routing & health</span></div>
        <div className="hint" style={{ marginTop: 0, marginBottom: 8 }}>When a node has multiple valid next steps, the engine routes with a Thompson-sampling bandit over per-edge success — restricted to CTL-feasible transitions. It learns as it runs.</div>
        <CopyBlock code={`engine.process("PolicyCheck", user_text, memory=mem)
# after runs, inspect what it learned:
print(engine.health_report())
# {'Classify': {'total_executions': 12, 'fail_rate': 0.0, 'avg_latency_ms': 340.2}, ...}`} />

        <div className="shd"><span className="lbl">Context pruning (memory)</span></div>
        <CopyBlock code={`from aura_state.memory.pruner import ContextPruner

pruned = ContextPruner.prune(
    full_history,               # [{"role": ..., "content": ...}, ...]
    required_keys=["order_id"], # always keep these in context
    max_messages=6,             # keep system prompt + last 6
)`} />

        <div className="shd"><span className="lbl">Few-shot tuning (bootstrap)</span></div>
        <CopyBlock code={`from aura_state.compiler.dspy_tuner import BootstrapTeleprompter

tp = BootstrapTeleprompter(openai_client=client)   # semantic KNN
tp.compile(past_successful_traces)                 # [{node, input, output, success}]
better_prompt = tp.optimize_node("Classify", prompt, new_input)
# appends the K most similar past successes as few-shot demos`} />

        <div className="hint">Everything in this studio is this library. Build here or in code — the guarantees are identical.</div>
      </div>
    </div>
  );
}

// ── Settings ──
export function Settings() {
  const { agentName, provider, providersList, flows, theme, nodes, edges, entry, invariants, set, doSave, doLoad, loadTemplate, importAgent, refreshFlows } = useStore();
  useEffect(() => { refreshFlows(); }, []);
  const exportAgent = () => {
    const flow = { name: agentName, provider, nodes, edges, entry, invariants };
    const blob = new Blob([JSON.stringify(flow, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = `${agentName}.aura.json`; a.click();
    URL.revokeObjectURL(url);
  };
  const onImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]; if (!f) return;
    const r = new FileReader();
    r.onload = () => { try { importAgent(JSON.parse(String(r.result))); } catch {} };
    r.readAsText(f);
  };
  return (
    <div style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <Head title="Settings" sub="templates · providers · agents · appearance" />
      <div className="out" style={{ maxWidth: 720 }}>
        <div className="shd" style={{ marginTop: 2 }}><span className="lbl">Start from a template</span></div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
          {Object.entries(TEMPLATES).map(([key, t]) => (
            <div key={key} className="obl-item" style={{ cursor: "pointer" }} onClick={() => loadTemplate(key)}>
              <div className="expr" style={{ marginBottom: 5 }}>{t.title}</div>
              <div className="hint" style={{ margin: 0 }}>{t.blurb}</div>
            </div>
          ))}
        </div>

        <div className="shd"><span className="lbl">Agent</span></div>
        <div className="two">
          <div className="fg"><span className="lbl">Name</span><input className="field mono" value={agentName} onChange={(e) => set({ agentName: e.target.value })} /></div>
          <div className="fg"><span className="lbl">Provider</span>
            <select className="field" value={provider} onChange={(e) => set({ provider: e.target.value })}>
              {providersList.map((p: any) => <option key={p.name} value={p.name}>{p.name} · {p.model}{p.available ? "" : " (set " + p.needs + ")"}</option>)}
            </select></div>
        </div>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <button className="btn" onClick={doSave}><Icon name="save" size={14} /> Save agent</button>
          <button className="btn" onClick={exportAgent}><Icon name="download" size={14} /> Export JSON</button>
          <label className="btn" style={{ cursor: "pointer" }}><Icon name="versions" size={14} /> Import JSON
            <input type="file" accept="application/json,.json" onChange={onImport} style={{ display: "none" }} /></label>
        </div>

        <div className="shd"><span className="lbl">Saved agents</span></div>
        {flows.length === 0 && <div className="hint">No saved agents yet. Build one and Save.</div>}
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
          {flows.map((f: string) => <button key={f} className="btn sm" onClick={() => doLoad(f)}>{f}</button>)}
        </div>

        <div className="shd"><span className="lbl">Providers</span></div>
        <div className="hint" style={{ marginTop: 0, marginBottom: 10 }}>Ollama runs locally with no key. For OpenAI / Gemini / DeepSeek, set the key in your shell <b>before</b> launching, then restart the studio:</div>
        <pre style={{ marginBottom: 12 }}>{`export OPENAI_API_KEY=sk-...\naura-state ui`}</pre>
        {providersList.map((p: any) => <ProviderRow key={p.name} p={p} />)}

        <div className="shd"><span className="lbl">Appearance</span></div>
        <div className="seg" style={{ width: "fit-content" }}>
          {["system", "light", "dark"].map((t) => <button key={t} className={theme === t ? "on" : ""} onClick={() => set({ theme: t as any })}>{t}</button>)}
        </div>
      </div>
    </div>
  );
}
