import React, { useRef, useState } from "react";
import { useStore, TEMPLATES } from "./store";
import { Icon } from "./ui";

const MCP_SAMPLE = `{
  "tools": [
    {"name": "fetch", "description": "Fetch a URL",
     "annotations": {"readOnlyHint": true, "openWorldHint": true}},
    {"name": "read_file", "description": "Read a local file",
     "annotations": {"readOnlyHint": true}},
    {"name": "slack_post_message", "description": "Post to a Slack channel",
     "annotations": {"readOnlyHint": false}}
  ]
}`;

function exportJson(s: any) {
  const flow = { name: s.agentName, provider: s.provider, nodes: s.nodes, edges: s.edges, entry: s.entry, invariants: s.invariants };
  const blob = new Blob([JSON.stringify(flow, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a"); a.href = url; a.download = `${s.agentName}.aura.json`; a.click();
  URL.revokeObjectURL(url);
}

export function AgentMenu() {
  const s = useStore();
  const fileRef = useRef<HTMLInputElement>(null);
  if (!s.agentMenuOpen) return null;
  const close = () => s.set({ agentMenuOpen: false });
  const others = s.flows.filter((f) => f !== s.agentName);
  const onFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]; if (!f) return;
    const r = new FileReader();
    r.onload = () => { try { s.importAgent(JSON.parse(String(r.result))); } catch {} };
    r.readAsText(f);
  };
  const item = (icon: string, label: string, fn: () => void, danger = false) => (
    <button className={"am-item" + (danger ? " danger" : "")} onClick={fn}><Icon name={icon} size={14} /> {label}</button>
  );
  return (
    <>
      <div className="am-scrim" onClick={close} />
      <div className="agentmenu" role="menu">
        {item("plus", "New agent", () => s.set({ newAgentOpen: true, agentMenuOpen: false }))}
        {item("save", "Save & snapshot", () => { s.doSave(); close(); })}
        <div className="am-sep" />
        {item("build", "Rename…", () => { const n = window.prompt("Rename agent", s.agentName); if (n) s.renameAgent(n); })}
        {item("versions", "Duplicate", () => s.duplicateAgent())}
        {item("download", "Export JSON", () => { exportJson(s); close(); })}
        {item("sdk", "Export as Python", () => s.exportPythonFile())}
        {item("prove", "Import JSON…", () => fileRef.current?.click())}
        {item("sdk", "Import MCP tools…", () => s.set({ mcpOpen: true, agentMenuOpen: false }))}
        <input ref={fileRef} type="file" accept="application/json,.json" onChange={onFile} style={{ display: "none" }} />
        {item("trash", "Delete agent", () => { if (window.confirm(`Delete "${s.agentName}"? This removes the saved file.`)) s.deleteAgent(); }, true)}
        {others.length > 0 && <>
          <div className="am-sep" />
          <div className="am-label">Switch to</div>
          {others.map((f) => (
            <button key={f} className="am-item" onClick={() => { s.doLoad(f); close(); }}><Icon name="build" size={14} /> {f}</button>
          ))}
        </>}
      </div>
    </>
  );
}

export function McpModal() {
  const s = useStore();
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  if (!s.mcpOpen) return null;
  const close = () => s.set({ mcpOpen: false });
  const go = async () => {
    if (!text.trim()) return;
    setBusy(true);
    try { await s.importMcp(text); } finally { setBusy(false); }
  };
  return (
    <div className="palette-scrim" onClick={close}>
      <div className="newagent" style={{ maxWidth: 640 }} onClick={(e) => e.stopPropagation()}>
        <div className="na-head">
          <h2>Import MCP tools</h2>
          <button className="icobtn" aria-label="Close" onClick={close}>✕</button>
        </div>
        <div style={{ padding: "0 20px" }}>
          <p className="hint" style={{ marginTop: 0 }}>
            Paste your agent's <b>tools/list</b> result (or an <code>mcpServers</code> config with tools).
            Aura models the worst case — the LLM can call any tool in any order — and checks whether the
            set forms a <b>lethal trifecta</b> (private data + untrusted content + external comms).
            It never connects to or runs a server.
          </p>
          <textarea
            value={text} onChange={(e) => setText(e.target.value)}
            placeholder={MCP_SAMPLE} spellCheck={false}
            style={{ width: "100%", height: 220, fontFamily: "var(--mono, monospace)", fontSize: 12,
                     background: "var(--panel, #111)", color: "var(--ink-1, #eee)",
                     border: "1px solid var(--line, #333)", borderRadius: 8, padding: 10, resize: "vertical" }}
          />
        </div>
        <div className="na-foot" style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <button className="btn" onClick={() => setText(MCP_SAMPLE)}>Use sample</button>
          <span style={{ flex: 1 }} />
          <button className="btn primary" disabled={busy || !text.trim()} onClick={go}>
            {busy ? "Analyzing…" : "Import & verify"}
          </button>
        </div>
      </div>
    </div>
  );
}

export function NewAgentModal() {
  const s = useStore();
  if (!s.newAgentOpen) return null;
  const close = () => s.set({ newAgentOpen: false });
  return (
    <div className="palette-scrim" onClick={close}>
      <div className="newagent" onClick={(e) => e.stopPropagation()}>
        <div className="na-head">
          <h2>New agent</h2>
          <button className="icobtn" aria-label="Close" onClick={close}>✕</button>
        </div>
        <div className="na-grid">
          <button className="na-card blank" onClick={() => s.newBlank()}>
            <span className="na-ico"><Icon name="plus" size={22} /></span>
            <b>Blank agent</b>
            <span className="na-blurb">Start from an empty canvas and build node by node.</span>
          </button>
          {Object.entries(TEMPLATES).map(([key, t]) => (
            <button key={key} className="na-card" onClick={() => s.loadTemplate(key)}>
              <span className="na-ico"><Icon name="spark" size={20} /></span>
              <b>{t.title}</b>
              <span className="na-blurb">{t.blurb}</span>
            </button>
          ))}
        </div>
        <div className="na-foot">
          <span className="hint" style={{ margin: 0 }}>Templates are working examples — start from one rather than a blank file, then customize everything.</span>
        </div>
      </div>
    </div>
  );
}
