import React, { useMemo, useState, useRef, useEffect } from "react";
import { useStore, Module, TEMPLATES } from "./store";
import { Icon } from "./ui";

type Cmd = { label: string; hint?: string; icon?: string; run: () => void };

export default function Palette() {
  const s = useStore();
  const [q, setQ] = useState("");
  const [i, setI] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  useEffect(() => { inputRef.current?.focus(); }, []);

  const close = () => s.set({ paletteOpen: false });
  const go = (m: Module) => { s.set({ module: m, paletteOpen: false }); };

  const cmds: Cmd[] = useMemo(() => {
    const nav: [Module, string][] = [["build", "Build"], ["run", "Run"], ["runs", "Runs history"], ["evals", "Evals"], ["prove", "Prove"], ["data", "Dataset"], ["monitor", "Monitor"], ["calibrate", "Calibrate"], ["memory", "Memory"], ["versions", "Version history"], ["audit", "Audit trail"], ["sdk", "SDK"], ["settings", "Settings"]];
    const list: Cmd[] = [
      { label: "Verify design", hint: "Z3 · CTL · taint", icon: "prove", run: () => { s.runVerify(); close(); } },
      { label: "Run agent", hint: "execute end-to-end", icon: "play", run: () => { s.runFlow(); close(); } },
      { label: "Auto-repair taint", hint: "insert sanitizer, re-prove", icon: "spark", run: () => { s.autoRepair(); s.set({ module: "build", paletteOpen: false }); } },
      { label: "Save & snapshot", icon: "save", run: () => { s.doSave(); close(); } },
      { label: "Export proof certificate", icon: "download", run: () => { exportCert(s); close(); } },
      { label: "New agent", hint: "blank or template", icon: "plus", run: () => { s.set({ newAgentOpen: true, paletteOpen: false }); } },
      { label: "Export agent JSON", icon: "download", run: () => { exportAgent(s); close(); } },
      { label: "Export as Python", hint: "runnable script", icon: "sdk", run: () => { s.exportPythonFile(); close(); } },
      { label: "How Aura works", hint: "guided tour", icon: "prove", run: () => { s.set({ tourOpen: true, paletteOpen: false }); } },
      ...(["extract", "decision", "tool", "sanitizer"] as const).map((k) => ({ label: `Add ${k} node`, icon: "plus", run: () => { s.addNode(k); s.set({ module: "build", paletteOpen: false }); } })),
      ...Object.entries(TEMPLATES).map(([key, t]) => ({ label: `Template · ${t.title}`, hint: t.blurb, icon: "spark", run: () => { s.loadTemplate(key); close(); } })),
      ...nav.map(([m, label]) => ({ label: `Go to ${label}`, icon: "search", run: () => go(m) })),
    ];
    return list;
  }, [s.nodes, s.agentName]);

  const filtered = q ? cmds.filter((c) => (c.label + " " + (c.hint || "")).toLowerCase().includes(q.toLowerCase())) : cmds;
  const clamped = Math.min(i, Math.max(0, filtered.length - 1));

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") { e.preventDefault(); setI((x) => Math.min(x + 1, filtered.length - 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setI((x) => Math.max(x - 1, 0)); }
    else if (e.key === "Enter") { e.preventDefault(); filtered[clamped]?.run(); }
  };

  return (
    <div className="palette-scrim" onClick={close}>
      <div className="palette" onClick={(e) => e.stopPropagation()}>
        <div className="palette-in">
          <Icon name="search" size={16} />
          <input ref={inputRef} value={q} placeholder="Type a command…" onChange={(e) => { setQ(e.target.value); setI(0); }} onKeyDown={onKey} />
          <span className="kbd">esc</span>
        </div>
        <div className="palette-list">
          {filtered.map((c, idx) => (
            <div key={idx} className={"palette-row" + (idx === clamped ? " on" : "")} onMouseEnter={() => setI(idx)} onClick={() => c.run()}>
              <Icon name={c.icon || "search"} size={15} />
              <span className="pl">{c.label}</span>
              {c.hint && <span className="ph">{c.hint}</span>}
            </div>
          ))}
          {filtered.length === 0 && <div className="empty" style={{ padding: 24 }}>No commands match.</div>}
        </div>
      </div>
    </div>
  );
}

async function exportCert(s: any) {
  const spec = s.toSpec();
  const cert = await (await fetch("/api/certificate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: s.agentName, nodes: spec.nodes, edges: spec.edges, entry: spec.entry, invariants: s.invariants }) })).json();
  const blob = new Blob([JSON.stringify(cert, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = `${s.agentName}-certificate.json`; a.click();
  URL.revokeObjectURL(url);
}

function exportAgent(s: any) {
  const flow = { name: s.agentName, provider: s.provider, nodes: s.nodes, edges: s.edges, entry: s.entry, invariants: s.invariants };
  const blob = new Blob([JSON.stringify(flow, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a"); a.href = url; a.download = `${s.agentName}.aura.json`; a.click();
  URL.revokeObjectURL(url);
}

export { exportCert };
