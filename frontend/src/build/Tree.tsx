import React, { useState } from "react";
import { useStore, Status } from "../store";
import { KIND, Icon } from "../ui";
import type { NodeKind } from "../api";
import { CollapseBtn } from "./Resizer";

const ADD: { kind: NodeKind; label: string; desc: string }[] = [
  { kind: "extract", label: "Extract", desc: "LLM step → structured output" },
  { kind: "decision", label: "Decision", desc: "Verified branch / rule" },
  { kind: "tool", label: "Tool", desc: "Declared external call — verified, never run" },
  { kind: "sanitizer", label: "Sanitizer", desc: "Clears taint before a sink" },
];

const TOOL_PRESETS: { label: string; name: string; effect: "read" | "write" | "external" }[] = [
  { label: "Database write", name: "db.write", effect: "write" },
  { label: "HTTP request", name: "http.get", effect: "read" },
  { label: "Send email", name: "email.send", effect: "external" },
  { label: "Vector search (RAG)", name: "vectordb.search", effect: "read" },
  { label: "Payment", name: "payment.charge", effect: "external" },
  { label: "Custom tool…", name: "tool.call", effect: "write" },
];

export default function Tree() {
  const { nodes, selectedId, statusByNode, entry, verify, addNode, addTool, select, treeCollapsed } = useStore();
  const [toolMenu, setToolMenu] = useState(false);
  const z3 = verify?.obligations || [];
  const z3ok = z3.filter((o: any) => o.consistent).length;
  const ctl = verify?.ctl || [];
  const ctlok = ctl.filter((c: any) => c.verdict === "PROVEN").length;
  const taint = verify?.taint?.verdict;
  const tri = verify?.trifecta?.verdict;                 // PROVEN | CLOSED
  const triF = verify?.trifecta?.findings?.[0];
  const hash = verify?.contract ? contractHash(verify.contract) : null;
  const total = nodes.length || 1;
  const proven = Object.values(statusByNode).filter((s) => s === "proven").length;

  return (
    <div className="tree">
      <div className="hd">
        <div className="nm">Nodes</div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div className="lbl">{nodes.length}</div>
          <CollapseBtn which="tree" />
        </div>
      </div>

      <div className="grp">
        {nodes.map((n) => {
          const st: Status = statusByNode[n.id] || "pending";
          return (
            <div key={n.id} className={"tnode" + (selectedId === n.id ? " on" : "")} onClick={() => select(n.id)}>
              <span className="ty" style={{ background: KIND[n.kind].shade }} />
              <span className="tname">{n.id}</span>
              <span className="tkind">{KIND[n.kind].label}</span>
              {entry === n.id && <span className="tentry lbl">entry</span>}
              <span className="st dot" style={{ background: st === "proven" ? "var(--proven)" : st === "violated" ? "var(--violated)" : "var(--pending)" }} />
            </div>
          );
        })}
        {nodes.length === 0 && <div className="hint" style={{ padding: "6px 8px", lineHeight: 1.5 }}>No nodes yet. The real workflow: <b>Import from code</b> (agent menu) to verify an existing LangGraph / CrewAI / AutoGen / MCP agent — or add nodes below to model one. Then <b>Verify design</b> proves it (and <code>aura-state check</code> gates it in CI).</div>}
      </div>

      <div className="addhd lbl">Add a node <span style={{ color: "var(--ink-4)", textTransform: "none", letterSpacing: 0 }}>· 4 kinds, name them anything</span></div>
      <div className="addn">
        {ADD.map((a) => (
          <button key={a.kind} title={a.desc}
            onClick={() => (a.kind === "tool" ? setToolMenu((v) => !v) : addNode(a.kind))}>
            <span className="an-top"><Icon name="plus" size={11} /> {a.label}{a.kind === "tool" ? " ▾" : ""}</span>
            <span className="an-desc">{a.desc}</span>
          </button>
        ))}
      </div>
      {toolMenu && (
        <div className="toolmenu">
          <div className="lbl" style={{ padding: "2px 4px 2px" }}>Common tools · all are Tool nodes</div>
          <div className="hint" style={{ padding: "0 4px 7px", fontSize: 11, lineHeight: 1.45 }}>
            Declarations, not integrations. Aura <b>proves</b> the design (taint, lethal
            trifecta) — it never calls these. A mock return stands in during Run.
          </div>
          {TOOL_PRESETS.map((t) => (
            <button key={t.name} onClick={() => { addTool(t.name, t.effect, t.label); setToolMenu(false); }}>
              <span>{t.label}</span>
              <span className="mono">{t.name}</span>
            </button>
          ))}
        </div>
      )}

      <div className="stat">
        <div className="lbl" style={{ marginBottom: 3 }}>Design proof</div>
        <div className="hint" style={{ margin: "0 0 9px", fontSize: 11 }}>Static proofs over the design — nothing is executed.</div>
        <div className="row"><span className="k">Z3 obligations</span><span className="mono">{verify ? `${z3ok}/${z3.length}` : "—"}</span></div>
        <div className="row"><span className="k">CTL reachability</span><span className="mono">{verify ? `${ctlok}/${ctl.length}` : "—"}</span></div>
        <div className="row"><span className="k">Taint dataflow</span><span className="mono" style={{ color: taint === "PROVEN" ? "var(--proven)" : taint === "VIOLATED" ? "var(--violated)" : "" }}>{taint ? taint.toLowerCase() : "—"}</span></div>
        {taint === "VIOLATED" && (verify?.taint?.violations || []).slice(0, 4).map((v: any, i: number) => (
          <div key={i} className="hint" style={{ padding: "3px 0", color: "var(--violated)", cursor: "pointer" }}
            title="Untrusted data reaches this sink with no sanitizer between. Click to inspect the sink; add a Sanitizer to clear it."
            onClick={() => select(v.sink)}>
            {v.source} → {v.sink}{v.field && v.field !== "*" ? ` · ${v.field}` : ""} — no sanitizer
          </div>
        ))}
        <div className="row" title={triF ? triF.detail : "private data + untrusted content + external comms on one reachable path"}>
          <span className="k">Lethal trifecta</span>
          <span className="mono" style={{ color: tri === "PROVEN" ? "var(--proven)" : tri === "CLOSED" ? "var(--violated)" : "" }}>
            {tri === "PROVEN" ? "safe" : tri === "CLOSED" ? "vulnerable" : "—"}
          </span>
        </div>
        {triF && (
          <div className="hint" style={{ padding: "4px 0 2px", color: "var(--violated)" }}>
            {triF.untrusted} → {triF.exfil} · private: {triF.private}
          </div>
        )}
        <div className="row"><span className="k">Contract</span><span className="mono">{hash || "—"}</span></div>
        <div className="bar"><i style={{ width: `${(proven / total) * 100}%` }} /></div>
      </div>
    </div>
  );
}

function contractHash(c: any): string {
  const s = JSON.stringify(c);
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
  return h.toString(16).slice(0, 8);
}
