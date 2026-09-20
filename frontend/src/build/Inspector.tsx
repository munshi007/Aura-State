import React, { useState } from "react";
import { useStore, MODEL_PRESETS } from "../store";
import { KIND, CAP_LABEL, Icon, Info } from "../ui";
import { statusReason, ctlReason } from "../explain";
import * as api from "../api";
import type { AgentNode, NodeKind, Capability, Field } from "../api";

const TABS = ["model", "schema", "proofs", "tools", "retry"];
const TYPES = ["str", "int", "float", "bool"];

export default function Inspector() {
  const { nodes, selectedId, tab, set, updateNode, deleteNode, verify } = useStore();
  const node = nodes.find((n) => n.id === selectedId);

  if (!node) return <GraphInspector />;

  const up = (patch: Partial<AgentNode>) => updateNode(node.id, patch);

  return (
    <div className="insp">
      <div className="ih">
        <div className="t">
          <span className="ty" style={{ background: KIND[node.kind].shade }} />
          <h2>{node.id}</h2>
        </div>
        <div className="sub">{KIND[node.kind].label} · {CAP_LABEL[node.capability]}</div>
        <div className="tabs">
          {(node.kind === "extract" ? [...TABS, "tune"] : TABS).map((t) => (
            <button key={t} className={tab === t ? "on" : ""} onClick={() => set({ tab: t })}>
              {t[0].toUpperCase() + t.slice(1)}
            </button>
          ))}
        </div>
      </div>
      <div className="ibody">
        {tab === "model" && <ModelTab node={node} up={up} />}
        {tab === "schema" && <SchemaTab node={node} up={up} />}
        {tab === "proofs" && <ProofsTab node={node} up={up} verify={verify} />}
        {tab === "tools" && <ToolsTab node={node} up={up} />}
        {tab === "retry" && <RetryTab node={node} up={up} />}
        {tab === "tune" && node.kind === "extract" && <TuneTab node={node} />}
        <button className="del" onClick={() => deleteNode(node.id)}>
          <Icon name="trash" size={13} /> Delete node
        </button>
      </div>
    </div>
  );
}

function GraphInspector() {
  const { nodes, edges, entry, agentName, invariants, verify, repairing, autoRepair, toSpec, graphNodes, set } = useStore();
  const [repairMsg, setRepairMsg] = useState<any>(null);
  const taintBad = verify?.taint?.verdict === "VIOLATED";
  const doRepair = async () => { const r = await autoRepair(); setRepairMsg(r); };
  const [props, setProps] = useState<{ type: string; a?: string; b?: string }[]>([{ type: "completes" }]);
  const [res, setRes] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [cert, setCert] = useState<any>(null);
  const [certBusy, setCertBusy] = useState(false);
  const [policy, setPolicy] = useState<any>(null);
  const [polBusy, setPolBusy] = useState(false);
  const ids = nodes.map((n) => n.id);
  const scan = async () => { setPolBusy(true); setPolicy(await api.policyScan(toSpec().nodes)); setPolBusy(false); };

  const check = async () => {
    setBusy(true);
    const spec = toSpec();
    setRes(await api.ctlCheck(graphNodes(), spec.edges, entry, props));
    setBusy(false);
  };
  const buildCert = async () => {
    setCertBusy(true);
    const spec = toSpec();
    const c = await api.certificate(agentName, spec.nodes, spec.edges, entry, invariants);
    setCert(c);
    api.auditLog("certificate", `certificate for ${agentName} — ${c.verified ? "VERIFIED" : "not verified"}`,
      { agent: agentName, verified: c.verified, sha256: c.sha256, summary: c.summary }).catch(() => {});
    setCertBusy(false);
  };
  const downloadCert = () => {
    if (!cert) return;
    const blob = new Blob([JSON.stringify(cert, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = `${agentName}-certificate.json`; a.click();
    URL.revokeObjectURL(url);
  };
  const setP = (i: number, patch: any) => setProps(props.map((p, j) => (j === i ? { ...p, ...patch } : p)));

  return (
    <div className="insp">
      <div className="ih">
        <div className="t"><span className="ty" style={{ background: "var(--ink-3)" }} /><h2>Graph</h2></div>
        <div className="sub">{agentName} · {nodes.length} nodes · entry {entry}</div>
      </div>
      <div className="ibody">
        <div className="fg"><span className="lbl">Entry node<Info k="entry" /></span>
          {(() => {
            const targets = new Set(edges.filter(([a, b]) => ids.includes(a) && ids.includes(b)).map(([, b]) => b));
            const sources = ids.filter((id) => !targets.has(id));
            const opts = sources.length ? sources : ids;
            return (
              <select className="field" value={opts.includes(entry) ? entry : opts[0]} onChange={(e) => set({ entry: e.target.value })}>
                {opts.map((id) => <option key={id}>{id}</option>)}
              </select>
            );
          })()}
          <div className="hint">Where execution starts — a source node (nothing points into it).</div></div>

        <div className="shd"><span className="lbl">Temporal properties (CTL)<Info k="ctl" /></span><span className="solver">pyModelChecking</span></div>
        {props.map((p, i) => (
          <div key={i} className="obl-item" style={{ padding: "8px 9px" }}>
            <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
              <select className="field mono" style={{ padding: "5px 7px" }} value={p.type} onChange={(e) => setP(i, { type: e.target.value })}>
                <option value="reachable">reachable</option>
                <option value="before">A before B</option>
                <option value="exclusive">A ⊕ B</option>
                <option value="completes">completes</option>
              </select>
              {(p.type === "reachable") && <select className="field mono" style={{ padding: "5px 7px" }} value={p.a || ""} onChange={(e) => setP(i, { a: e.target.value })}><option value="">node…</option>{ids.map((id) => <option key={id}>{id}</option>)}</select>}
              {(p.type === "before" || p.type === "exclusive") && <>
                <select className="field mono" style={{ padding: "5px 7px" }} value={p.a || ""} onChange={(e) => setP(i, { a: e.target.value })}><option value="">A…</option>{ids.map((id) => <option key={id}>{id}</option>)}</select>
                <select className="field mono" style={{ padding: "5px 7px" }} value={p.b || ""} onChange={(e) => setP(i, { b: e.target.value })}><option value="">B…</option>{ids.map((id) => <option key={id}>{id}</option>)}</select>
              </>}
              <button className="rowx" onClick={() => setProps(props.filter((_, j) => j !== i))}>✕</button>
            </div>
          </div>
        ))}
        <button className="addf" onClick={() => setProps([...props, { type: "reachable" }])}>+ add property</button>
        <button className="btn sm" style={{ marginTop: 10 }} onClick={check} disabled={busy}><Icon name="prove" size={13} /> {busy ? "Checking…" : "Check properties"}</button>

        {res && <>
          {(res.properties || []).map((r: any, i: number) => (
            <div key={i} className="obl-item" style={{ marginTop: 8 }}>
              <div className="top"><span className="expr">{r.label}</span>
                <span className={"chip " + (r.verdict === "PROVEN" ? "pv" : "vi")}>{r.verdict === "PROVEN" ? "✓ proven" : "✕ violated"}</span></div>
              <div className="meta" style={{ marginTop: 6, fontFamily: "Hanken Grotesk", fontSize: 11.5, color: "var(--ink-2)", lineHeight: 1.5 }}>{ctlReason(r.label, r.verdict)}</div>
            </div>
          ))}
          <div className="obl-item" style={{ marginTop: 8, borderColor: res.dead_ends?.length ? "var(--violated-line)" : "var(--line)" }}>
            <div className="top"><span className="expr">dead-ends</span>
              <span className={"chip " + (res.dead_ends?.length ? "vi" : "pv")}>{res.dead_ends?.length ? res.dead_ends.join(", ") : "✓ none"}</span></div>
          </div>
        </>}

        {taintBad && <>
          <div className="shd"><span className="lbl">Remediation</span><span className="solver">counterexample-guided</span></div>
          <div className="obl-item bad">
            <div className="expr" style={{ color: "var(--violated)" }}>taint: untrusted data reaches a dangerous sink</div>
            <div className="meta" style={{ marginTop: 6 }}>Aura can insert a sanitizer before the sink and re-prove the design.</div>
          </div>
          <button className="btn" style={{ marginTop: 8 }} onClick={doRepair} disabled={repairing}>
            <Icon name="spark" size={14} /> {repairing ? "Repairing…" : "Auto-repair taint"}
          </button>
          {repairMsg?.repaired && <div className="obl-item" style={{ marginTop: 10, borderColor: "var(--proven-line)" }}>
            <div className="expr" style={{ color: "var(--proven)" }}>✓ inserted {repairMsg.added.map((a: any) => a.id).join(", ")}</div>
            <div className="meta" style={{ marginTop: 6 }}><span>taint now {repairMsg.taint_after}</span></div>
          </div>}
        </>}

        <div className="shd"><span className="lbl">Agent invariants<Info k="invariant" /></span><span className="solver">z3 · cross-node</span></div>
        <textarea className="field mono" style={{ minHeight: 48 }} value={invariants.join("\n")}
          placeholder={"amount <= 500\nrefunded implies approved"} onChange={(e) => set({ invariants: e.target.value.split("\n").map((s) => s.trim()).filter(Boolean) })} />
        <div className="hint" style={{ marginBottom: 4 }}>Obligations that must hold across the whole agent, not just one node. Checked for joint satisfiability and sealed into the certificate.</div>

        <div className="shd"><span className="lbl">Proof certificate<Info k="certificate" /></span><span className="solver">v1.1</span></div>
        <button className="btn" onClick={buildCert} disabled={certBusy}><Icon name="prove" size={14} /> {certBusy ? "Building…" : "Build certificate"}</button>
        {cert && <div className="obl-item" style={{ marginTop: 12, borderColor: cert.verified ? "var(--proven-line)" : "var(--violated-line)" }}>
          <div className="top">
            <span className="expr" style={{ display: "flex", alignItems: "center", gap: 7 }}>
              <span style={{ fontSize: 16, color: cert.verified ? "var(--proven)" : "var(--violated)" }}>{cert.verified ? "✓" : "✕"}</span>
              certificate {cert.verified ? "VERIFIED" : "NOT VERIFIED"}
            </span>
            <span className="chip mu">v{cert.aura_certificate}</span>
          </div>
          <div className="meta" style={{ marginTop: 8 }}>
            <span>taint {cert.summary.taint}</span><span>ctl {cert.summary.ctl}</span>
            <span>obl {cert.summary.obligations}</span><span>inv {cert.summary.invariants}</span>
          </div>
          <div className="meta" style={{ marginTop: 6 }}><span>engine aura-state {cert.engine.aura_state}</span></div>
          <div className="cx" style={{ color: "var(--ink-3)", marginTop: 8 }}>sha256 · {cert.sha256.slice(0, 32)}…</div>
          <button className="btn sm" style={{ marginTop: 10 }} onClick={downloadCert}><Icon name="download" size={13} /> Download JSON</button>
        </div>}

        <div className="shd"><span className="lbl">Policy scan (PII · secrets)<Info k="policy" /></span><span className="solver">content</span></div>
        <button className="btn" onClick={scan} disabled={polBusy}><Icon name="prove" size={14} /> {polBusy ? "Scanning…" : "Scan prompts & rules"}</button>
        {policy && (policy.clean
          ? <div className="obl-item" style={{ marginTop: 10, borderColor: "var(--proven-line)" }}><div className="expr" style={{ color: "var(--proven)" }}>✓ no PII or secrets in prompts, rules, or obligations</div></div>
          : <>
            <div className="meta" style={{ marginTop: 10 }}><span style={{ color: "var(--violated)" }}>{policy.by_severity.critical} critical</span><span>{policy.by_severity.high} high</span><span>{policy.by_severity.medium} medium</span></div>
            {policy.findings.map((f: any, i: number) => (
              <div key={i} className="obl-item bad" style={{ marginTop: 8 }}>
                <div className="top"><span className="expr">{f.node} · {f.where}</span><span className={"chip " + (f.severity === "medium" ? "pn" : "vi")}>{f.severity}</span></div>
                <div className="meta" style={{ marginTop: 6 }}><span>{f.description}</span><span className="mono">{f.match}</span></div>
              </div>
            ))}
          </>)}
      </div>
    </div>
  );
}

function ModelTab({ node, up }: { node: AgentNode; up: (p: Partial<AgentNode>) => void }) {
  const { provider: globalProvider, providersList } = useStore();
  const effProvider = node.provider || globalProvider;
  const presets = MODEL_PRESETS[effProvider] || [];
  const provNames = providersList.length ? providersList.map((p: any) => p.name) : ["ollama", "openai", "gemini", "deepseek"];
  return (
    <>
      <div className="fg"><span className="lbl">Node name</span>
        <input className="field mono" value={node.id} onChange={(e) => up({ id: e.target.value.replace(/\s+/g, "") })} /></div>
      <div className="fg"><span className="lbl">Kind</span>
        <select className="field" value={node.kind} onChange={(e) => up({ kind: e.target.value as NodeKind })}>
          <option value="extract">Extract (LLM)</option>
          <option value="decision">Decision (rule)</option>
          <option value="tool">Tool (external call)</option>
          <option value="sanitizer">Sanitizer</option>
        </select></div>
      {node.kind === "extract" && <div className="two">
        <div className="fg"><span className="lbl">Provider</span>
          <select className="field" value={node.provider || ""} onChange={(e) => up({ provider: e.target.value || undefined, model: MODEL_PRESETS[e.target.value || globalProvider]?.[0] || node.model })}>
            <option value="">Inherit ({globalProvider})</option>
            {provNames.map((p) => <option key={p} value={p}>{p}</option>)}
          </select></div>
        <div className="fg"><span className="lbl">Model</span>
          <input className="field mono" list={`models-${node.id}`} value={node.model} onChange={(e) => up({ model: e.target.value })} />
          <datalist id={`models-${node.id}`}>{presets.map((m) => <option key={m} value={m} />)}</datalist>
        </div>
      </div>}
      <div className="fg"><span className="lbl">{node.kind === "tool" ? "Description" : node.kind === "decision" ? "Description" : "System prompt"}</span>
        <textarea className="field" style={{ minHeight: 96 }} value={node.system_prompt}
          placeholder={node.kind === "tool" ? "What this tool does — for humans + the certificate…" : "Instruction sent to the model for this node…"}
          onChange={(e) => up({ system_prompt: e.target.value })} />
        {node.kind === "extract" && <div className="hint">This node receives the <b>Run</b> input (if it's the entry) or the upstream node's output as its context. Provide the actual input in the Run tab.</div>}
        {node.kind === "tool" && <div className="hint">Tools don't call an LLM. Define the actual call in the <b>Schema</b> tab (tool name, side-effect, mock).</div>}
      </div>
      {node.kind === "extract" && <div className="two">
        <div className="fg"><span className="lbl">Temperature</span>
          <div className="rangewrap">
            <input type="range" min={0} max={2} step={0.1} value={node.temperature}
              onChange={(e) => up({ temperature: +e.target.value })} />
            <span className="rangeval">{node.temperature.toFixed(1)}</span>
          </div></div>
        <div className="fg"><span className="lbl">Max tokens</span>
          <input className="field mono" type="number" value={node.max_tokens}
            onChange={(e) => up({ max_tokens: +e.target.value })} /></div>
      </div>}
    </>
  );
}

function SchemaTab({ node, up }: { node: AgentNode; up: (p: Partial<AgentNode>) => void }) {
  if (node.kind === "decision") {
    return (
      <>
        <div className="fg"><span className="lbl">Sandbox rule (English → verified Python)</span>
          <textarea className="field mono" style={{ minHeight: 80 }} value={node.sandbox_rule}
            placeholder="result = amount <= 100"
            onChange={(e) => up({ sandbox_rule: e.target.value })} /></div>
        <div className="hint">The rule is parsed to an AST and compiled to a whitelisted evaluator — never <code>eval</code>. It reads from memory produced upstream and sets <code>result</code>.</div>
      </>
    );
  }
  if (node.kind === "tool") return <ToolDef node={node} up={up} />;
  if (node.kind !== "extract") {
    return <div className="empty">Sanitizers carry no schema — they clear taint<br />so downstream sinks are safe.</div>;
  }
  const fields = node.fields;
  const setF = (i: number, patch: Partial<Field>) => up({ fields: fields.map((f, j) => (j === i ? { ...f, ...patch } : f)) });
  return (
    <>
      <div className="shd"><span className="lbl">Extraction schema</span><span className="solver">Pydantic · instructor</span></div>
      <table className="sch">
        <thead><tr><th style={{ width: "38%" }}>Field</th><th style={{ width: "26%" }}>Type</th><th>Description</th><th style={{ width: 24 }} /></tr></thead>
        <tbody>
          {fields.map((f, i) => (
            <tr key={i}>
              <td><input className="mono" value={f.name} onChange={(e) => setF(i, { name: e.target.value })} /></td>
              <td><select value={f.type} onChange={(e) => setF(i, { type: e.target.value })}>{TYPES.map((t) => <option key={t}>{t}</option>)}</select></td>
              <td><input value={f.description || ""} placeholder="—" onChange={(e) => setF(i, { description: e.target.value })} /></td>
              <td><button className="rowx" onClick={() => up({ fields: fields.filter((_, j) => j !== i) })}>✕</button></td>
            </tr>
          ))}
        </tbody>
      </table>
      <button className="addf" onClick={() => up({ fields: [...fields, { name: "field" + (fields.length + 1), type: "str" }] })}>+ add field</button>
      <div className="hint">Field names (camel/snake) become the extraction contract. Obligations in the Proofs tab reference these names.</div>
      <SchemaImport up={up} />
    </>
  );
}

function SchemaImport({ up }: { up: (p: Partial<AgentNode>) => void }) {
  const [open, setOpen] = useState(false);
  const [txt, setTxt] = useState("");
  const [err, setErr] = useState("");
  const imp = () => {
    try {
      const s = JSON.parse(txt);
      const props = s.properties || s;
      const jmap: Record<string, string> = { string: "str", integer: "int", number: "float", boolean: "bool" };
      const fields = Object.entries(props).map(([name, def]: any) => ({
        name, type: jmap[def?.type] || "str", description: def?.description || "",
      }));
      if (!fields.length) { setErr("no properties found"); return; }
      up({ fields }); setOpen(false); setTxt(""); setErr("");
    } catch { setErr("invalid JSON"); }
  };
  if (!open) return <button className="addf" style={{ marginTop: 6 }} onClick={() => setOpen(true)}>⇩ import JSON Schema</button>;
  return (
    <div className="obl-item" style={{ marginTop: 8 }}>
      <div className="lbl" style={{ marginBottom: 6 }}>Paste JSON Schema</div>
      <textarea className="field mono" style={{ minHeight: 90 }} value={txt}
        placeholder={'{ "properties": { "amount": {"type":"integer"} } }'} onChange={(e) => setTxt(e.target.value)} />
      {err && <div className="cx" style={{ marginTop: 6 }}>{err}</div>}
      <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
        <button className="btn sm pri" onClick={imp}>Import</button>
        <button className="btn sm" onClick={() => setOpen(false)}>Cancel</button>
      </div>
    </div>
  );
}

function ProofsTab({ node, up, verify }: { node: AgentNode; up: (p: Partial<AgentNode>) => void; verify: any }) {
  const [sample, setSample] = useState("");
  const [res, setRes] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const obl = node.obligations;
  const vObl = (verify?.obligations || []).find((o: any) => o.node === node.id);

  const setObl = (txt: string) => up({ obligations: txt.split("\n").map((s) => s.trim()).filter(Boolean) });

  const check = async () => {
    setBusy(true);
    let data: any = {};
    if (sample.trim()) { try { data = JSON.parse(sample); } catch { data = {}; } }
    else node.fields.forEach((f) => (data[f.name] = f.type === "int" || f.type === "float" ? 0 : f.type === "bool" ? true : ""));
    const r = await api.proveData(data, obl);
    setRes(r); setBusy(false);
  };

  const why = statusReason(node.id, verify);
  return (
    <>
      {why && (
        <div className="obl-item" style={{ marginBottom: 14, borderColor: why.startsWith("Violated") ? "var(--violated-line)" : "var(--proven-line)" }}>
          <div className="lbl" style={{ marginBottom: 6, color: why.startsWith("Violated") ? "var(--violated)" : "var(--proven)" }}>Why this verdict</div>
          <div style={{ fontSize: 12, lineHeight: 1.55, color: "var(--ink-2)" }}>{why}</div>
        </div>
      )}
      <div className="shd"><span className="lbl">Obligations (Z3 / SMT)<Info k="obligation" /></span><span className="solver">z3-solver</span></div>
      <textarea className="field mono" style={{ minHeight: 62 }} value={obl.join("\n")}
        placeholder={"amount >= 0\namount <= 500"} onChange={(e) => setObl(e.target.value)} />
      <div className="hint" style={{ marginBottom: 14 }}>One boolean obligation per line, over this node's fields. Compiled AST→Z3, fail-closed.</div>

      {obl.length === 0 && <div className="empty" style={{ padding: "18px 0" }}>No obligations. This node's output is unconstrained.</div>}

      {obl.map((o, i) => {
        const failed = res && (res.failed || []).includes(o);
        const unproven = res && (res.unproven || []).includes(o);
        const chip = !res ? "mu" : failed ? "vi" : unproven ? "pn" : "pv";
        const txt = !res ? "unchecked" : failed ? "violated" : unproven ? "unproven" : "proven";
        return (
          <div key={i} className={"obl-item" + (failed ? " bad" : "")}>
            <div className="top">
              <span className="expr">{o}</span>
              <span className={"chip " + chip}>{chip === "pv" ? "✓" : chip === "vi" ? "✕" : "○"} {txt}</span>
            </div>
          </div>
        );
      })}

      <div className="shd"><span className="lbl">Symbolic consistency<Info k="z3" /></span><span className="solver">SAT check</span></div>
      {vObl ? (
        <div className={"obl-item" + (vObl.consistent ? "" : " bad")}>
          <div className="top">
            <span className="expr">obligations are jointly {vObl.consistent ? "satisfiable" : "contradictory"}</span>
            <span className={"chip " + (vObl.consistent ? "pv" : "vi")}>{vObl.consistent ? "✓ SAT" : "✕ UNSAT"}</span>
          </div>
          {vObl.reason && <div className="meta">{vObl.reason}</div>}
        </div>
      ) : <div className="hint">Run <b>Verify design</b> to check joint satisfiability.</div>}

      <div className="shd"><span className="lbl">Prove against a sample</span></div>
      <textarea className="field mono" style={{ minHeight: 54 }} value={sample}
        placeholder={node.fields.length ? `{ ${node.fields.map((f) => `"${f.name}": …`).join(", ")} }` : "{ }"}
        onChange={(e) => setSample(e.target.value)} />
      <button className="btn sm" style={{ marginTop: 8 }} onClick={check} disabled={busy || obl.length === 0}>
        <Icon name="prove" size={13} /> {busy ? "Proving…" : "Prove extraction"}
      </button>

      {res && res.counterexample && (
        <div className="obl-item bad" style={{ marginTop: 12 }}>
          <div className="lbl" style={{ color: "var(--violated)", marginBottom: 6 }}>Counterexample</div>
          <div className="cx">{typeof res.counterexample === "string" ? res.counterexample : JSON.stringify(res.counterexample)}</div>
        </div>
      )}
      {res && res.witness && (
        <div className="meta" style={{ fontFamily: "JetBrains Mono", fontSize: 10.5, color: "var(--ink-3)", marginTop: 10 }}>
          witness · {typeof res.witness === "string" ? res.witness : JSON.stringify(res.witness)}
        </div>
      )}

      <div className="shd"><span className="lbl">Calibration<Info k="conformal" /></span><span className="solver">split conformal</span></div>
      <div className="obl-item">
        <div className="top"><span className="expr">consensus runs</span><span className="mono">×{node.consensus}</span></div>
        <div className="meta"><span>conformal α = {(1 - node.confidence).toFixed(2)}</span><span>coverage target {(node.confidence * 100).toFixed(0)}%</span></div>
      </div>
    </>
  );
}

function ToolDef({ node, up }: { node: AgentNode; up: (p: Partial<AgentNode>) => void }) {
  const effects: { v: "read" | "write" | "external"; t: string; d: string }[] = [
    { v: "read", t: "Read", d: "Fetches data (http.get, db.read). No side effect — safe." },
    { v: "write", t: "Write", d: "Persists data (db.write, queue, file). Dangerous sink — taint-guarded." },
    { v: "external", t: "External action", d: "Acts on the world (payment, email, API call). Dangerous sink — taint-guarded." },
  ];
  return (
    <>
      <div className="shd" style={{ marginTop: 2 }}><span className="lbl">Tool<Info k="tool" /></span><span className="solver">declared · not executed</span></div>
      <div className="fg"><span className="lbl">Tool name</span>
        <input className="field mono" value={node.tool_name || ""} placeholder="db.write" onChange={(e) => up({ tool_name: e.target.value })} />
        <div className="hint">The real call your agent makes here. Aura proves its preconditions; it doesn't run it.</div></div>
      <div className="shd"><span className="lbl">Side effect</span></div>
      {effects.map((x) => (
        <label key={x.v} className="obl-item" style={{ display: "block", cursor: "pointer", borderColor: node.side_effect === x.v ? "var(--ink)" : undefined }}>
          <div className="top"><span className="expr" style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <input type="radio" checked={node.side_effect === x.v} onChange={() => up({ side_effect: x.v, capability: x.v === "read" ? "plain" : "sink" })} style={{ accentColor: "var(--ink)" }} />{x.t}</span></div>
          <div className="meta" style={{ marginLeft: 24 }}>{x.d}</div>
        </label>
      ))}
      <div className="shd"><span className="lbl">Mock return</span><span className="solver">design-time only</span></div>
      <textarea className="field mono" style={{ minHeight: 54 }} value={node.mock_return || ""} placeholder='{ "ok": true }' onChange={(e) => up({ mock_return: e.target.value })} />
      <div className="hint">Aura never calls this tool — you bind the real one in code / aura-runtime. This mock stands in so downstream nodes can be tested during a design-time Run.</div>
    </>
  );
}

function TuneTab({ node }: { node: AgentNode }) {
  const [examples, setExamples] = useState<{ input: string; output: string }[]>([{ input: "", output: "" }]);
  const [newInput, setNewInput] = useState("");
  const [res, setRes] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const setE = (i: number, patch: any) => setExamples(examples.map((e, j) => (j === i ? { ...e, ...patch } : e)));
  const run = async () => {
    setBusy(true);
    const ex = examples.filter((e) => e.input).map((e) => ({ input: e.input, output: (() => { try { return JSON.parse(e.output); } catch { return e.output; } })() }));
    setRes(await api.tune(node.id, node.system_prompt || node.id, ex, newInput));
    setBusy(false);
  };
  return (
    <>
      <div className="shd" style={{ marginTop: 2 }}><span className="lbl">Few-shot tuner<Info k="fewshot" /></span><span className="solver">bootstrap KNN</span></div>
      <div className="hint" style={{ marginTop: 0, marginBottom: 10 }}>Give past successes; Aura finds the closest ones to a new input and appends them as demonstrations. Improves extraction with no code change.</div>
      {examples.map((e, i) => (
        <div key={i} className="obl-item" style={{ marginBottom: 8 }}>
          <div className="lbl" style={{ marginBottom: 6, display: "flex", justifyContent: "space-between" }}>Example {i + 1}<button className="rowx" onClick={() => setExamples(examples.filter((_, j) => j !== i))}>✕</button></div>
          <input className="field" style={{ marginBottom: 6 }} value={e.input} placeholder="input text" onChange={(ev) => setE(i, { input: ev.target.value })} />
          <input className="field mono" value={e.output} placeholder='output e.g. {"category":"damaged"}' onChange={(ev) => setE(i, { output: ev.target.value })} />
        </div>
      ))}
      <button className="addf" onClick={() => setExamples([...examples, { input: "", output: "" }])}>+ add example</button>
      <div className="fg" style={{ marginTop: 12 }}><span className="lbl">New input to optimize for</span>
        <input className="field" value={newInput} placeholder="a fresh input this node will see" onChange={(e) => setNewInput(e.target.value)} /></div>
      <button className="btn" onClick={run} disabled={busy}><Icon name="spark" size={14} /> {busy ? "Bootstrapping…" : "Preview optimized prompt"}</button>
      {res && !res.error && (<>
        <div className="meta" style={{ marginTop: 12 }}><span>{res.n_demos} demos injected</span><span>{res.embedder}</span></div>
        <pre style={{ marginTop: 8 }}>{res.optimized}</pre>
      </>)}
      {res?.error && <div className="cx" style={{ marginTop: 10 }}>{res.error}</div>}
    </>
  );
}

function ToolsTab({ node, up }: { node: AgentNode; up: (p: Partial<AgentNode>) => void }) {
  const caps: { v: Capability; t: string; d: string }[] = [
    { v: "plain", t: "Trusted", d: "Ordinary node. No special dataflow role." },
    { v: "untrusted", t: "Untrusted source", d: "Introduces attacker-controlled data (user text, scraped web). Taint originates here." },
    { v: "sink", t: "Dangerous sink", d: "Performs a real-world side effect (payment, email, shell). Tainted data must not reach it unsanitized." },
    { v: "sanitizer", t: "Sanitizer", d: "Clears taint. Data flowing through is considered safe downstream." },
  ];
  return (
    <>
      {node.kind === "tool" && (
        <div className="obl-item" style={{ marginBottom: 12 }}>
          <div className="expr">Role derived from the tool's side-effect</div>
          <div className="meta" style={{ marginTop: 6 }}>This node is a <b>{node.side_effect === "read" ? "safe read" : "dangerous sink"}</b> because <span className="mono">{node.tool_name || "the tool"}</span> is <b>{node.side_effect}</b>. Change it in the Schema tab.</div>
        </div>
      )}
      <div className="shd"><span className="lbl">Dataflow role (static taint)<Info k="taint" /></span><span className="solver">field-level</span></div>
      {caps.map((c) => (
        <label key={c.v} className={"obl-item"} style={{ display: "block", cursor: "pointer", borderColor: node.capability === c.v ? "var(--ink)" : undefined }}>
          <div className="top">
            <span className="expr" style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <input type="radio" checked={node.capability === c.v} onChange={() => up({ capability: c.v })} style={{ accentColor: "var(--ink)" }} />
              {c.t}
            </span>
          </div>
          <div className="meta" style={{ marginLeft: 24 }}>{c.d}</div>
        </label>
      ))}
      <div className="hint">The verifier proves that no path carries data from an <b>untrusted source</b> into a <b>dangerous sink</b> without passing a <b>sanitizer</b>. A violation fails the design.</div>
    </>
  );
}

function RetryTab({ node, up }: { node: AgentNode; up: (p: Partial<AgentNode>) => void }) {
  return (
    <>
      <div className="fg"><span className="lbl">Max retries on verification failure</span>
        <div className="rangewrap">
          <input type="range" min={0} max={5} step={1} value={node.retry} onChange={(e) => up({ retry: +e.target.value })} />
          <span className="rangeval">{node.retry}</span>
        </div>
        <div className="hint">On a failed obligation, the node re-prompts with the counterexample as feedback (counterexample-guided replanning), up to this many times.</div>
      </div>
      <div className="fg"><span className="lbl">Consensus runs</span>
        <div className="rangewrap">
          <input type="range" min={1} max={7} step={1} value={node.consensus} onChange={(e) => up({ consensus: +e.target.value })} />
          <span className="rangeval">×{node.consensus}</span>
        </div>
        <div className="hint">Sample the model N times and vote. N≥3 also yields a conformal dispersion interval.</div>
      </div>
      <div className="fg"><span className="lbl">Confidence (coverage target)</span>
        <div className="rangewrap">
          <input type="range" min={0.5} max={0.99} step={0.01} value={node.confidence} onChange={(e) => up({ confidence: +e.target.value })} />
          <span className="rangeval">{(node.confidence * 100).toFixed(0)}%</span>
        </div>
        <div className="hint">Conformal risk level α = {(1 - node.confidence).toFixed(2)}. Below the calibration floor the interval reports as uncalibrated.</div>
      </div>
    </>
  );
}
