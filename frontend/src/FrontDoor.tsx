import React, { useMemo, useState } from "react";
import { useStore } from "./store";
import { Icon } from "./ui";
import { badgeSvg, badgeEmbed, proofSummary } from "./badge";

/** The first-run / empty-canvas welcome. Leads with the one move that sells:
 *  import the agent you already wrote → get a verdict. Deeper tabs come after. */
export function FrontDoor() {
  const { set, loadTemplate } = useStore();
  const cards = [
    { icon: "sdk", title: "Import from code", sub: "Paste a LangGraph / CrewAI / AutoGen agent. Parsed with ast — never executed.", go: () => set({ codeOpen: true }) },
    { icon: "data", title: "Import MCP tools", sub: "Point it at your tools/list or client config. Never connects to a server.", go: () => set({ mcpOpen: true }) },
    { icon: "build", title: "Start from a template", sub: "Refund agent, RAG Q&A, SQL agent, and more — model one from scratch.", go: () => set({ module: "settings" }) },
  ];
  return (
    <div className="frontdoor">
      <div className="fd-inner">
        <div className="fd-mark"><span style={{ color: "var(--proven)" }}>∴</span> aura-state</div>
        <h1 className="fd-h1">Prove your agent before you ship it.</h1>
        <p className="fd-sub">
          A design-time verifier — not a runtime guardrail. Import the agent you already wrote and Aura proves it's
          <b> injection-safe</b>, <b>in-spec</b>, and <b>won't act out of bounds</b>, then shows the exact path or a proof.
        </p>
        <div className="fd-cards">
          {cards.map((c) => (
            <button key={c.title} className="fd-card" onClick={c.go}>
              <span className="fd-card-ic"><Icon name={c.icon} size={18} /></span>
              <span className="fd-card-t">{c.title}</span>
              <span className="fd-card-s">{c.sub}</span>
            </button>
          ))}
        </div>
        <div className="fd-foot">
          <span>Just exploring? </span>
          <button className="fd-link" onClick={() => loadTemplate("refund")}>load a sample agent →</button>
          <span className="fd-or"> · or in your terminal: </span><code>pip install aura-state &amp;&amp; aura-state demo</code>
        </div>
      </div>
    </div>
  );
}

function _shortHash(obj: any): string {
  // tiny non-crypto content hash, just for a stable badge fingerprint
  let h = 5381;
  const s = JSON.stringify(obj || {});
  for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) >>> 0;
  return h.toString(16).padStart(8, "0").slice(0, 8);
}

/** Shareable proof badge — preview + download SVG + copy embed. Honest: renders
 *  "VERIFIED" only when the design has no violation. */
export function ProofBadge() {
  const { badgeOpen, agentName, verify, set } = useStore();
  const [copied, setCopied] = useState("");
  const hash = useMemo(() => (verify ? _shortHash(verify) : ""), [verify]);
  const svg = useMemo(() => (verify ? badgeSvg(agentName, verify, hash) : ""), [agentName, verify, hash]);
  if (!badgeOpen) return null;
  const close = () => set({ badgeOpen: false });
  const summary = verify ? proofSummary(verify) : null;

  const download = () => {
    const blob = new Blob([svg], { type: "image/svg+xml" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "aura-proof.svg"; a.click();
    URL.revokeObjectURL(url);
  };
  const copy = async (kind: "md" | "html") => {
    const e = badgeEmbed();
    try { await navigator.clipboard.writeText(e[kind]); setCopied(kind); setTimeout(() => setCopied(""), 1500); } catch {}
  };

  return (
    <div className="palette-scrim" onClick={close}>
      <div className="newagent" style={{ maxWidth: 520 }} onClick={(e) => e.stopPropagation()}>
        <div className="na-head">
          <h2>Share the proof</h2>
          <button className="icobtn" aria-label="Close" onClick={close}>✕</button>
        </div>
        <div style={{ padding: "0 20px 6px" }}>
          {!verify && <p className="hint">Run <b>Verify design</b> first — the badge is generated from a real result.</p>}
          {verify && (
            <>
              <div style={{ display: "flex", justifyContent: "center", padding: "6px 0 12px" }}
                   dangerouslySetInnerHTML={{ __html: svg }} />
              <p className="hint" style={{ marginTop: 0 }}>
                {summary && !summary.violated
                  ? "Every property proved. Drop this in your README or share it — it's honest: it only says VERIFIED when the design has no violation."
                  : "This agent has open findings, so the badge shows them (not a green “verified”). Fix them, re-verify, and the badge turns green."}
              </p>
            </>
          )}
        </div>
        {verify && (
          <div className="na-foot" style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
            <button className="btn primary" onClick={download}><Icon name="download" size={14} /> Download SVG</button>
            <button className="btn" onClick={() => copy("md")}>{copied === "md" ? "Copied!" : "Copy Markdown"}</button>
            <button className="btn" onClick={() => copy("html")}>{copied === "html" ? "Copied!" : "Copy HTML"}</button>
          </div>
        )}
      </div>
    </div>
  );
}
