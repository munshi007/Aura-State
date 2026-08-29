import React, { useEffect, useState } from "react";
import { useStore, EvalCase } from "../store";
import { Icon, StatusChip } from "../ui";
import { stepReason } from "../explain";
import * as api from "../api";

function Head({ title, sub, right }: { title: string; sub: string; right?: React.ReactNode }) {
  return <div className="modhd"><div><h1>{title}</h1><div className="sub">{sub}</div></div>{right}</div>;
}
const vstatus = (v: any) => (v === true ? "proven" : v === false ? "violated" : "pending");

function TraceView({ trace }: { trace: any[] }) {
  return (
    <>
      {trace.map((s: any, i: number) => (
        <div key={i} className="step">
          <div className="sh">
            <div className="nm"><span className="mono" style={{ color: "var(--ink-3)" }}>{String(i + 1).padStart(2, "0")}</span> {s.node}
              {s.verified != null && <StatusChip status={vstatus(s.verified)} />}</div>
            {s.next && <span className="arrow">→ {s.next}</span>}
          </div>
          {s.error && <div className="cx" style={{ marginTop: 8 }}>{s.error}</div>}
          {s.extracted && Object.keys(s.extracted).length > 0 && <pre style={{ marginTop: 9 }}>{JSON.stringify(s.extracted, null, 2)}</pre>}
          <div style={{ fontSize: 12, lineHeight: 1.5, color: "var(--ink-2)", marginTop: 9 }}>{stepReason(s)}</div>
          <div className="meta" style={{ fontFamily: "JetBrains Mono", fontSize: 10, color: "var(--ink-3)", marginTop: 8, display: "flex", gap: 14, flexWrap: "wrap" }}>
            {s.ms != null && <span>{s.ms} ms</span>}
            {s.model && <span>{s.provider ? s.provider + " · " : ""}{s.model}</span>}
            {s.consensus > 1 && <span>consensus ×{s.consensus}</span>}
            {s.iterations != null && <span>retries {s.iterations}</span>}
          </div>
        </div>
      ))}
    </>
  );
}

// ── Runs history ──
export function Runs() {
  const [runs, setRuns] = useState<any[]>([]);
  const [sel, setSel] = useState<any>(null);
  const load = async () => setRuns(await api.listRuns());
  useEffect(() => { load(); }, []);
  const open = async (file: string) => setSel(await api.getRun(file));
  return (
    <div style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <Head title="Runs" sub="every execution persisted for audit & replay"
        right={<button className="btn sm" onClick={load}><Icon name="runs" size={13} /> Refresh</button>} />
      <div className="cols" style={{ gridTemplateColumns: "360px 1fr", overflow: "hidden" }}>
        <div className="form" style={{ padding: 0 }}>
          {runs.length === 0 && <div className="empty" style={{ marginTop: 50 }}>No runs yet.<br />Run an agent — it's recorded here.</div>}
          {runs.map((r) => (
            <div key={r.file} className="feedrow" style={{ cursor: "pointer", background: sel?.id === r.id ? "var(--sel)" : "" }} onClick={() => open(r.file)}>
              <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
                <StatusChip status={vstatus(r.verified)} />
                <b className="mono">{r.agent}</b>
                <span className="mono" style={{ marginLeft: "auto", color: "var(--ink-3)", fontSize: 11 }}>{r.ts}</span>
              </div>
              <div className="hint" style={{ marginTop: 5 }}>{r.input}</div>
              <div className="mono" style={{ fontSize: 10.5, color: "var(--ink-3)", marginTop: 3 }}>{r.steps} steps</div>
            </div>
          ))}
        </div>
        <div className="out">
          {!sel && <div className="empty" style={{ marginTop: 60 }}>Select a run to replay its verified trace.</div>}
          {sel && (<>
            <div className="metric"><div className="k">{sel.agent} · {sel.ts}</div>
              <div className="big" style={{ fontSize: 22, color: sel.verified ? "var(--proven)" : "var(--violated)" }}>{sel.verified ? "VERIFIED" : "NOT VERIFIED"}</div>
              <div className="hint">input · {sel.input}</div></div>
            <TraceView trace={sel.trace || []} />
          </>)}
        </div>
      </div>
    </div>
  );
}

// ── Evals ──
export function Evals() {
  const { evalCases, provider, toSpec, set } = useStore();
  const [res, setRes] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const setCase = (i: number, patch: Partial<EvalCase>) => set({ evalCases: evalCases.map((c, j) => (j === i ? { ...c, ...patch } : c)) });
  const run = async () => {
    setBusy(true);
    const spec = toSpec();
    const cases = evalCases.map((c) => ({ input: c.input, expect: c.expect || {}, obligations: c.obligations }));
    const r = await api.runEval(spec.nodes, spec.edges, spec.entry, provider, cases);
    setRes(r);
    api.auditLog("eval", `eval suite — ${r.passed}/${r.total} passed`, { passed: r.passed, failed: r.failed, total: r.total }).catch(() => {});
    setBusy(false);
  };
  return (
    <div style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <Head title="Evals" sub="test cases with assertions — run the whole suite, prove it passes"
        right={<button className="btn pri" onClick={run} disabled={busy}><Icon name="evals" size={14} /> {busy ? "Running…" : `Run ${evalCases.length} cases`}</button>} />
      <div className="cols" style={{ overflow: "hidden" }}>
        <div className="form">
          {evalCases.map((c, i) => (
            <div key={i} className="obl-item" style={{ marginBottom: 12 }}>
              <div className="lbl" style={{ marginBottom: 6, display: "flex", justifyContent: "space-between" }}>Case {i + 1}
                <button className="rowx" onClick={() => set({ evalCases: evalCases.filter((_, j) => j !== i) })}>✕</button></div>
              <textarea className="field" style={{ minHeight: 44, marginBottom: 6 }} value={c.input} placeholder="input message" onChange={(e) => setCase(i, { input: e.target.value })} />
              <input className="field mono" style={{ marginBottom: 6 }} value={c.obligations.join(", ")} placeholder="obligations: amount >= 0, amount <= 500"
                onChange={(e) => setCase(i, { obligations: e.target.value.split(",").map((s) => s.trim()).filter(Boolean) })} />
              <input className="field mono" value={JSON.stringify(c.expect || {})} placeholder='expect: {"category":"billing"}'
                onChange={(e) => { try { setCase(i, { expect: JSON.parse(e.target.value) }); } catch {} }} />
            </div>
          ))}
          <button className="addf" onClick={() => set({ evalCases: [...evalCases, { input: "", expect: {}, obligations: [] }] })}>+ add case</button>
        </div>
        <div className="out">
          {!res && <div className="empty" style={{ marginTop: 60 }}>Assertions run each input through the real<br />agent, then prove expectations on the output.</div>}
          {res && (<>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <div className="metric"><div className="k">Passed</div><div className="big" style={{ color: "var(--proven)" }}>{res.passed}</div></div>
              <div className="metric"><div className="k">Failed</div><div className="big" style={{ color: res.failed ? "var(--violated)" : "" }}>{res.failed}</div></div>
            </div>
            {res.results.map((r: any, i: number) => (
              <div key={i} className={"obl-item" + (r.passed ? "" : " bad")}>
                <div className="top"><span className="expr">{r.input || "(empty)"}</span><StatusChip status={r.passed ? "proven" : "violated"} label={r.passed ? "pass" : "fail"} /></div>
                {r.detail && <div className="cx" style={{ marginTop: 6 }}>{r.detail}</div>}
                {r.got && Object.keys(r.got).length > 0 && <pre style={{ marginTop: 8 }}>{JSON.stringify(r.got)}</pre>}
                {(r.checks || []).map((ch: any, j: number) => (
                  <div key={j} className="meta" style={{ fontFamily: "JetBrains Mono", fontSize: 10.5, color: ch.pass ? "var(--proven)" : "var(--violated)", marginTop: 4 }}>
                    {ch.pass ? "✓" : "✕"} {ch.kind === "equals" ? `${ch.field} = ${ch.want} (got ${ch.got})` : `obligations ${ch.want.join(", ")}${ch.failed?.length ? " · failed " + ch.failed.join(", ") : ""}`}
                  </div>
                ))}
              </div>
            ))}
          </>)}
        </div>
      </div>
    </div>
  );
}

// ── Versions + diff ──
export function Versions() {
  const { agentName, doSave, set } = useStore();
  const [vers, setVers] = useState<any[]>([]);
  const [a, setA] = useState<any>(null);
  const [b, setB] = useState<any>(null);
  const load = async () => setVers(await api.listVersions(agentName));
  useEffect(() => { load(); }, [agentName]);
  const pick = async (file: string) => {
    const v = await api.getVersion(agentName, file);
    if (!a) setA(v); else if (!b) setB(v); else { setA(v); setB(null); }
  };
  const diff = a && b ? computeDiff(a.flow, b.flow) : null;
  return (
    <div style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <Head title="Version history" sub={`immutable snapshots of ${agentName} · pick two to diff`}
        right={<><button className="btn sm" onClick={async () => { await doSave(); load(); }}><Icon name="save" size={13} /> Snapshot now</button></>} />
      <div className="cols" style={{ gridTemplateColumns: "320px 1fr", overflow: "hidden" }}>
        <div className="form" style={{ padding: 0 }}>
          {vers.length === 0 && <div className="empty" style={{ marginTop: 50 }}>No versions yet.<br />Save the agent to snapshot it.</div>}
          {vers.map((v) => {
            const on = a?.hash === v.hash || b?.hash === v.hash;
            return (
              <div key={v.file} className="feedrow" style={{ cursor: "pointer", background: on ? "var(--sel)" : "" }} onClick={() => pick(v.file)}>
                <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
                  <b className="mono">v{v.version}</b>
                  <span className="chip mu">{v.hash}</span>
                  <span className="mono" style={{ marginLeft: "auto", color: "var(--ink-3)", fontSize: 11 }}>{v.ts}</span>
                </div>
                {on && <div className="lbl" style={{ marginTop: 4, color: "var(--ink)" }}>{a?.hash === v.hash ? "A" : "B"}</div>}
              </div>
            );
          })}
        </div>
        <div className="out">
          {!diff && <div className="empty" style={{ marginTop: 60 }}>Pick two versions (A, then B) to see<br />what changed between them.</div>}
          {diff && (<>
            <div style={{ display: "flex", gap: 10, marginBottom: 14 }}>
              <div className="metric" style={{ flex: 1 }}><div className="k">A · v{a.version}</div><div className="mono" style={{ fontSize: 13 }}>{a.hash}</div></div>
              <div className="metric" style={{ flex: 1 }}><div className="k">B · v{b.version}</div><div className="mono" style={{ fontSize: 13 }}>{b.hash}</div></div>
            </div>
            {diff.lines.length === 0 && <div className="hint">Identical designs.</div>}
            {diff.lines.map((l, i) => (
              <div key={i} className="obl-item" style={{ borderColor: l.k === "+" ? "var(--proven-line)" : l.k === "-" ? "var(--violated-line)" : "var(--line)" }}>
                <div className="expr" style={{ color: l.k === "+" ? "var(--proven)" : l.k === "-" ? "var(--violated)" : "var(--ink)" }}>{l.k} {l.text}</div>
              </div>
            ))}
            <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
              <button className="btn sm" onClick={() => { set({ nodes: JSON.parse(JSON.stringify(b.flow.nodes)), edges: b.flow.edges.map((e: any) => [...e]), entry: b.flow.entry, agentName: b.flow.name || agentName, module: "build", verify: null, statusByNode: {}, diffOverlay: {} }); }}>
                <Icon name="versions" size={13} /> Restore B
              </button>
              <button className="btn sm pri" onClick={() => { set({ nodes: JSON.parse(JSON.stringify(b.flow.nodes)), edges: b.flow.edges.map((e: any) => [...e]), entry: b.flow.entry, agentName: b.flow.name || agentName, module: "build", verify: null, statusByNode: {}, diffOverlay: overlayFor(a.flow, b.flow) }); }}>
                <Icon name="build" size={13} /> Compare on canvas
              </button>
            </div>
          </>)}
        </div>
      </div>
    </div>
  );
}

function overlayFor(fa: any, fb: any): Record<string, "added" | "changed"> {
  const na = new Map((fa.nodes || []).map((n: any) => [n.id, n]));
  const out: Record<string, "added" | "changed"> = {};
  for (const n of fb.nodes || []) {
    if (!na.has(n.id)) { out[n.id] = "added"; continue; }
    const x: any = na.get(n.id);
    const changed = ["model", "system_prompt", "capability", "sandbox_rule", "consensus", "confidence", "obligations", "fields"]
      .some((f) => JSON.stringify(x[f]) !== JSON.stringify(n[f]));
    if (changed) out[n.id] = "changed";
  }
  return out;
}

// ── Audit trail ──
const ACTION_META: Record<string, { icon: string; label: string }> = {
  verify: { icon: "prove", label: "Verify" }, run: { icon: "play", label: "Run" },
  repair: { icon: "spark", label: "Repair" }, save: { icon: "save", label: "Save" },
  certificate: { icon: "download", label: "Certify" }, eval: { icon: "evals", label: "Eval" },
};

export function Audit() {
  const [entries, setEntries] = useState<any[]>([]);
  const [chain, setChain] = useState<any>(null);
  const [open, setOpen] = useState<number | null>(null);
  const [filter, setFilter] = useState<string>("all");
  const load = async () => { setEntries(await api.auditList(300)); setChain(await api.auditVerify()); };
  useEffect(() => { load(); }, []);
  const exportLog = async () => {
    const txt = await (await fetch(api.AUDIT_EXPORT_URL)).text();
    const blob = new Blob([txt], { type: "application/x-ndjson" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = "aura-audit.jsonl"; a.click();
    URL.revokeObjectURL(url);
  };
  const kinds = Array.from(new Set(entries.map((e) => e.action)));
  const shown = filter === "all" ? entries : entries.filter((e) => e.action === filter);

  return (
    <div style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <Head title="Audit trail" sub="tamper-evident, append-only — every action, hash-chained"
        right={<div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          {chain && (chain.intact
            ? <span className="chip pv">✓ chain intact · {chain.count} · head {chain.head}</span>
            : <span className="chip vi">✕ TAMPERED at seq {chain.broken_at}</span>)}
          <button className="btn sm" onClick={exportLog}><Icon name="download" size={13} /> Export</button>
          <button className="btn sm" onClick={load}><Icon name="runs" size={13} /> Refresh</button>
        </div>} />
      <div className="wsbody out" style={{ padding: 0 }}>
        <div style={{ display: "flex", gap: 6, padding: "12px 20px 8px", flexWrap: "wrap", borderBottom: "1px solid var(--hair)" }}>
          <button className={"btn sm" + (filter === "all" ? " pri" : "")} onClick={() => setFilter("all")}>all</button>
          {kinds.map((k) => <button key={k} className={"btn sm" + (filter === k ? " pri" : "")} onClick={() => setFilter(k)}>{ACTION_META[k]?.label || k}</button>)}
        </div>
        {shown.length === 0 && <div className="empty" style={{ marginTop: 50 }}>No audit records yet.<br />Verify, run, save, or repair — every action is recorded here.</div>}
        {shown.map((e) => {
          const m = ACTION_META[e.action] || { icon: "search", label: e.action };
          const bad = /UNVERIFIED|not verified|violated|TAMPERED|fail/i.test(e.summary);
          return (
            <div key={e.seq} className="feedrow" style={{ cursor: "pointer" }} onClick={() => setOpen(open === e.seq ? null : e.seq)}>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <span className="mono" style={{ color: "var(--ink-4)", fontSize: 11, width: 34 }}>#{e.seq}</span>
                <Icon name={m.icon} size={14} />
                <span className="chip mu">{m.label}</span>
                <span style={{ fontSize: 12.5, color: bad ? "var(--violated)" : "var(--ink)" }}>{e.summary}</span>
                <span className="mono" style={{ marginLeft: "auto", color: "var(--ink-3)", fontSize: 11 }}>{e.ts.replace("T", " ")}</span>
                <span className="mono" style={{ color: "var(--ink-4)", fontSize: 11 }}>{e.actor}</span>
              </div>
              {open === e.seq && (
                <div style={{ marginTop: 10, paddingLeft: 44 }}>
                  <pre>{JSON.stringify(e.detail, null, 2)}</pre>
                  <div className="mono" style={{ fontSize: 10, color: "var(--ink-4)", marginTop: 8, wordBreak: "break-all" }}>
                    hash {e.hash}<br />prev {e.prev_hash}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function computeDiff(fa: any, fb: any): { lines: { k: string; text: string }[] } {
  const lines: { k: string; text: string }[] = [];
  const na = new Map((fa.nodes || []).map((n: any) => [n.id, n]));
  const nb = new Map((fb.nodes || []).map((n: any) => [n.id, n]));
  for (const id of nb.keys()) if (!na.has(id)) lines.push({ k: "+", text: `node ${id}` });
  for (const id of na.keys()) if (!nb.has(id)) lines.push({ k: "-", text: `node ${id}` });
  for (const id of nb.keys()) {
    if (!na.has(id)) continue;
    const x: any = na.get(id), y: any = nb.get(id);
    ["model", "system_prompt", "capability", "sandbox_rule", "consensus", "confidence"].forEach((f) => {
      if (JSON.stringify(x[f]) !== JSON.stringify(y[f])) lines.push({ k: "~", text: `${id}.${f}: ${JSON.stringify(x[f])} → ${JSON.stringify(y[f])}` });
    });
    if (JSON.stringify(x.obligations) !== JSON.stringify(y.obligations)) lines.push({ k: "~", text: `${id}.obligations: [${(x.obligations || []).join(", ")}] → [${(y.obligations || []).join(", ")}]` });
    if (JSON.stringify(x.fields) !== JSON.stringify(y.fields)) lines.push({ k: "~", text: `${id}.schema changed` });
  }
  const ea = new Set((fa.edges || []).map((e: any) => e.join("→")));
  const eb = new Set((fb.edges || []).map((e: any) => e.join("→")));
  for (const e of eb) if (!ea.has(e)) lines.push({ k: "+", text: `edge ${e}` });
  for (const e of ea) if (!eb.has(e)) lines.push({ k: "-", text: `edge ${e}` });
  return { lines };
}
