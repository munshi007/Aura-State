import React, { useState } from "react";
import { useStore } from "./store";
import { Icon } from "./ui";

const STEPS = [
  {
    title: "Agents you can prove things about",
    body: "Most tools show you a trace after the fact. Aura proves your agent is correct and safe before it runs — with real math, not vibes. Everything here runs locally on your machine; no cloud, no API key required.",
    icon: "prove",
  },
    {
    title: "Build the graph",
    body: "An agent is a typed graph of four node kinds: Extract (an LLM call → structured output), Decision (a verified branching rule), Tool (a declared external call like db.write or payment.refund), and Sanitizer (clears taint). Aura verifies the DESIGN — it does not run your tools; you bind those in code or aura-runtime.",
    icon: "build",
  },
  {
    title: "Verify the design",
    body: "Hit Verify design and four real engines run: Z3 proves your obligations hold for every possible value, CTL model-checks the flow (reachability, completion), static taint analysis traces injection risk, and the contract compiler seals it. The only color on the canvas is verification state — green proven, amber pending, red violated.",
    icon: "prove",
  },
  {
    title: "Auto-repair what's broken",
    body: "When taint finds untrusted data reaching a dangerous sink, click Auto-repair — Aura inserts a sanitizer in exactly the right place and re-proves the design clean. Counterexample-guided, one click.",
    icon: "spark",
  },
  {
    title: "Run, observe, and calibrate",
    body: "Run the agent end-to-end and read the verified trace — every step explains why it passed. Stream real production outputs into Monitor from your own CrewAI / LangGraph / code via the SDK. Calibrate gives distribution-free conformal intervals and risk-controlled abstention.",
    icon: "run",
  },
  {
    title: "Understand and audit everything",
    body: "Every verdict has a plain-English 'why', and every technical term has a hover ?. Every action you take is written to a tamper-evident, hash-chained audit trail you can export for compliance. Nothing is a black box.",
    icon: "audit",
  },
  {
    title: "It's a library first",
    body: "The studio is optional. pip install aura-state and use verify(), @verified, and Monitor in code — the guarantees are identical. See the SDK tab for copy-paste snippets.",
    icon: "sdk",
  },
];

export default function Tour() {
  const { set } = useStore();
  const [i, setI] = useState(0);
  const close = () => { try { localStorage.setItem("aura_tour_seen", "1"); } catch {} set({ tourOpen: false }); };
  const step = STEPS[i];
  const last = i === STEPS.length - 1;
  return (
    <div className="palette-scrim" onClick={close}>
      <div className="tour" onClick={(e) => e.stopPropagation()}>
        <div className="tour-head">
          <span className="tour-ico"><Icon name={step.icon} size={20} /></span>
          <div>
            <div className="lbl">How Aura works · {i + 1} / {STEPS.length}</div>
            <h2>{step.title}</h2>
          </div>
          <button className="icobtn" aria-label="Close" onClick={close} style={{ marginLeft: "auto" }}>✕</button>
        </div>
        <div className="tour-body">{step.body}</div>
        <div className="tour-dots">
          {STEPS.map((_, j) => <span key={j} className={"tour-dot" + (j === i ? " on" : "")} onClick={() => setI(j)} />)}
        </div>
        <div className="tour-foot">
          <button className="btn" onClick={close}>Skip</button>
          <div style={{ display: "flex", gap: 8 }}>
            {i > 0 && <button className="btn" onClick={() => setI(i - 1)}>Back</button>}
            <button className="btn pri" onClick={() => (last ? close() : setI(i + 1))}>{last ? "Start building" : "Next"}</button>
          </div>
        </div>
      </div>
    </div>
  );
}
