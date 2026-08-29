import React, { useEffect } from "react";
import { useStore, Module } from "./store";
import { Icon } from "./ui";
import Build from "./build/Build";
import { Run, Prove, Data, Monitor, Calibrate, Settings, Sdk, Memory } from "./modules/Modules";
import { Runs, Evals, Versions, Audit } from "./modules/Governance";
import Palette from "./Palette";
import Tour from "./Tour";
import { AgentMenu, NewAgentModal } from "./AgentMenu";

const RAIL: { id: Module; icon: string; label: string }[] = [
  { id: "build", icon: "build", label: "Build" },
  { id: "run", icon: "run", label: "Run" },
  { id: "runs", icon: "runs", label: "Runs" },
  { id: "evals", icon: "evals", label: "Evals" },
  { id: "prove", icon: "prove", label: "Prove" },
  { id: "data", icon: "data", label: "Dataset" },
  { id: "monitor", icon: "monitor", label: "Monitor" },
  { id: "calibrate", icon: "calibrate", label: "Calibrate" },
  { id: "memory", icon: "memory", label: "Memory" },
  { id: "audit", icon: "audit", label: "Audit" },
];

function useTheme() {
  const theme = useStore((s) => s.theme);
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
  }, [theme]);
}

function TopBar() {
  const { agentName, selectedId, verifying, running, agentMenuOpen, runVerify, runFlow, doSave, theme, set } = useStore();
  const cycleTheme = () => set({ theme: theme === "dark" ? "light" : "dark" });
  return (
    <div className="topbar">
      <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
        <div className="brand"><span style={{ color: "var(--proven)" }}>∴</span> Aura</div>
        <div className="crumbs">
          <span>acme</span><span>/</span>
          <span style={{ position: "relative" }}>
            <button className="agent-btn" onClick={() => set({ agentMenuOpen: !agentMenuOpen })} title="Agent menu">
              {agentName} <span style={{ color: "var(--ink-4)", fontSize: 9 }}>▾</span>
            </button>
            <AgentMenu />
          </span>
          <span className="v" style={{ cursor: "pointer" }} title="Version history" onClick={() => set({ module: "versions" })}>main · history</span>
          {selectedId && <><span>›</span><b>{selectedId}</b></>}
        </div>
      </div>
      <div className="tools">
        <button className="btn sm" onClick={doSave}><Icon name="save" size={14} /> Save</button>
        <button className="btn" onClick={runVerify} disabled={verifying}>
          <Icon name="prove" size={14} /> {verifying ? "Verifying…" : "Verify design"}
        </button>
        <button className="btn pri" onClick={runFlow} disabled={running}>
          <Icon name="play" size={14} /> {running ? "Running…" : "Run"}
        </button>
        <button className="icobtn" title="How Aura works" aria-label="How Aura works" onClick={() => set({ tourOpen: true })} style={{ fontWeight: 700 }}>?</button>
        <button className="icobtn" title="Toggle theme" aria-label="Toggle theme" onClick={cycleTheme}><Icon name="sun" size={15} /></button>
      </div>
    </div>
  );
}

function Rail() {
  const { module, set } = useStore();
  return (
    <div className="rail">
      {RAIL.map((r) => (
        <a key={r.id} className={module === r.id ? "on" : ""} title={r.label} aria-label={r.label} role="button" tabIndex={0}
          onClick={() => set({ module: r.id })} onKeyDown={(e) => { if (e.key === "Enter") set({ module: r.id }); }}><Icon name={r.icon} /></a>
      ))}
      <div className="sp" />
      <a className={module === "sdk" ? "on" : ""} title="SDK" aria-label="SDK" role="button" tabIndex={0}
        onClick={() => set({ module: "sdk" })} onKeyDown={(e) => { if (e.key === "Enter") set({ module: "sdk" }); }}><Icon name="sdk" /></a>
      <a className={module === "settings" ? "on" : ""} title="Settings" aria-label="Settings" role="button" tabIndex={0}
        onClick={() => set({ module: "settings" })} onKeyDown={(e) => { if (e.key === "Enter") set({ module: "settings" }); }}><Icon name="settings" /></a>
    </div>
  );
}

function StatusBar() {
  const { verify, statusByNode, nodes, provider, runTrace } = useStore();
  const proven = Object.values(statusByNode).filter((s) => s === "proven").length;
  const violated = Object.values(statusByNode).filter((s) => s === "violated").length;
  const total = nodes.length;
  const z3 = (verify?.obligations || []);
  const z3ok = z3.filter((o: any) => o.consistent).length;
  const ctl = (verify?.ctl || []);
  const ctlok = ctl.filter((c: any) => c.verdict === "PROVEN").length;
  const taint = verify?.taint?.verdict;
  const design = !verify ? "PENDING" : violated ? "VIOLATED" : "VERIFIED";
  const dcol = design === "VERIFIED" ? "var(--proven)" : design === "VIOLATED" ? "var(--violated)" : "var(--pending)";
  return (
    <div className="statusbar">
      <div className="seg2"><span className="d" style={{ background: dcol }} /> DESIGN: <b>{design}</b></div>
      <div className="seg2">Z3 {verify ? `${z3ok}/${z3.length}` : "—"}</div>
      <div className="seg2">CTL {verify ? `${ctlok}/${ctl.length}` : "—"}</div>
      <div className="seg2">taint {taint ? taint.toLowerCase() : "—"}</div>
      <div className="seg2">nodes {proven}✓ {violated ? violated + "✕ " : ""}/ {total}</div>
      <div className="sp" />
      {runTrace && <div className="seg2">run {runTrace.length} steps</div>}
      <div className="seg2">provider {provider}</div>
      <div className="seg2">aura-state 0.6.0</div>
    </div>
  );
}

export default function App() {
  useTheme();
  const { module, paletteOpen, tourOpen, set, refreshProviders, refreshFlows } = useStore();
  useEffect(() => {
    refreshProviders(); refreshFlows();
    try { if (!localStorage.getItem("aura_tour_seen")) set({ tourOpen: true }); } catch {}
  }, []);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      const typing = t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable);
      const st = useStore.getState();
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); set({ paletteOpen: !st.paletteOpen }); return; }
      if (e.key === "Escape") { set({ paletteOpen: false }); return; }
      if (typing) return;
      if (st.traceActive && !st.paletteOpen && st.runTrace) {
        if (e.key === "ArrowRight") { e.preventDefault(); set({ traceIndex: Math.min(st.runTrace.length - 1, st.traceIndex + 1) }); return; }
        if (e.key === "ArrowLeft") { e.preventDefault(); set({ traceIndex: Math.max(0, st.traceIndex - 1) }); return; }
      }
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "s") { e.preventDefault(); st.doSave(); set({ toast: "Saved & snapshotted." }); return; }
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); st.runFlow(); return; }
      if ((e.key === "Delete" || e.key === "Backspace") && st.selectedId && st.module === "build") {
        e.preventDefault(); st.deleteNode(st.selectedId);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  const isBuild = module === "build";
  const panel: Record<string, React.ReactNode> = {
    run: <Run />, runs: <Runs />, evals: <Evals />, prove: <Prove />, data: <Data />,
    monitor: <Monitor />, calibrate: <Calibrate />, memory: <Memory />, versions: <Versions />, audit: <Audit />, sdk: <Sdk />, settings: <Settings />,
  };
  return (
    <div className="app">
      <TopBar />
      <div className={"main" + (isBuild ? "" : " wide")}>
        <Rail />
        {isBuild ? <Build /> : panel[module]}
      </div>
      <StatusBar />
      {paletteOpen && <Palette />}
      {tourOpen && <Tour />}
      <NewAgentModal />
      <Toast />
    </div>
  );
}

function Toast() {
  const { toast, set } = useStore();
  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => set({ toast: null }), 4200);
    return () => clearTimeout(t);
  }, [toast]);
  if (!toast) return null;
  return (
    <div className="toast" role="status" onClick={() => set({ toast: null })}>
      <span className="dot" style={{ background: "var(--violated)" }} /> {toast}
    </div>
  );
}
