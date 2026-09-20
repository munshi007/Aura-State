import React, { useState } from "react";
import { useStore, Status } from "../store";
import { KIND, Icon } from "../ui";
import type { NodeKind } from "../api";

const ADD: { kind: NodeKind; label: string; desc: string }[] = [
  { kind: "extract", label: "Extract", desc: "LLM step → structured output" },
  { kind: "decision", label: "Decision", desc: "Verified branch / rule" },
  { kind: "tool", label: "Tool", desc: "Any external call — db, http, email…" },
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
  const { nodes, selectedId, statusByNode, entry, verify, addNode, addTool, select } = useStore();
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
        <div className="lbl">{nodes.length}</div>
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
        {nodes.length === 0 && <div className="hint" style={{ padding: "6px 8px" }}>No nodes yet — add one below, or load a template.</div>}
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
          <div className="lbl" style={{ padding: "2px 4px 6px" }}>Common tools · all are Tool nodes</div>
          {TOOL_PRESETS.map((t) => (
            <button key={t.name} onClick={() => { addTool(t.name, t.effect, t.label); setToolMenu(false); }}>
              <span>{t.label}</span>
              <span className="mono">{t.name}</span>
            </button>
          ))}
        </div>
      )}

      <div className="stat">
        <div className="lbl" style={{ marginBottom: 9 }}>Design proof</div>
        <div className="row"><span className="k">Z3 obligations</span><span className="mono">{verify ? `${z3ok}/${z3.length}` : "—"}</span></div>
        <div className="row"><span className="k">CTL reachability</span><span className="mono">{verify ? `${ctlok}/${ctl.length}` : "—"}</span></div>
        <div className="row"><span className="k">Taint dataflow</span><span className="mono" style={{ color: taint === "PROVEN" ? "var(--proven)" : taint === "VIOLATED" ? "var(--violated)" : "" }}>{taint ? taint.toLowerCase() : "—"}</span></div>
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
