import React from "react";
import type { Status } from "./store";
import type { NodeKind, Capability } from "./api";
import { GLOSSARY } from "./explain";

// Hover/focus "?" that explains a technical term in plain English.
export function Info({ k, wide }: { k: string; wide?: boolean }) {
  const txt = GLOSSARY[k];
  if (!txt) return null;
  return (
    <span className="info" tabIndex={0} role="note" aria-label={`Explain ${k}: ${txt}`}>
      <span className="info-q">?</span>
      <span className={"info-pop" + (wide ? " wide" : "")}>{txt}</span>
    </span>
  );
}

// Verification state is the ONLY color in the product.
export const STATUS_VAR: Record<Status, string> = {
  proven: "var(--proven)", violated: "var(--violated)", pending: "var(--pending)",
};
export const chipClass: Record<Status, string> = { proven: "pv", violated: "vi", pending: "pn" };

// Node kinds are graphite shades + shape, never chroma.
export const KIND: Record<NodeKind, { label: string; shade: string }> = {
  extract: { label: "Extract", shade: "var(--ink)" },
  decision: { label: "Decision", shade: "var(--ink-2)" },
  tool: { label: "Tool", shade: "var(--ink-3)" },
  sanitizer: { label: "Sanitizer", shade: "var(--ink-2)" },
};
export const CAP_LABEL: Record<Capability, string> = {
  plain: "trusted", untrusted: "untrusted source", sink: "dangerous sink", sanitizer: "sanitizer",
};

export function Dot({ status }: { status: Status }) {
  return <span className="dot" style={{ background: STATUS_VAR[status] }} />;
}

export function StatusChip({ status, label }: { status: Status; label?: string }) {
  const txt = label ?? (status === "proven" ? "proven" : status === "violated" ? "violated" : "pending");
  return <span className={"chip " + chipClass[status]}>{status === "proven" ? "✓" : status === "violated" ? "✕" : "○"} {txt}</span>;
}

const paths: Record<string, React.ReactNode> = {
  build: <><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="8.5" y="14" width="7" height="7" rx="1.5"/><path d="M6.5 10v2.5M17.5 10v2.5M12 10.5v3"/></>,
  run: <><path d="M6 4l14 8-14 8V4z"/></>,
  prove: <><path d="M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3z"/><path d="M8.5 12l2.5 2.5 4.5-5"/></>,
  data: <><ellipse cx="12" cy="6" rx="7.5" ry="3"/><path d="M4.5 6v6c0 1.7 3.4 3 7.5 3s7.5-1.3 7.5-3V6"/><path d="M4.5 12v6c0 1.7 3.4 3 7.5 3s7.5-1.3 7.5-3v-6"/></>,
  monitor: <><path d="M3 12h4l2.5-7 4 14 2.5-7H21"/></>,
  calibrate: <><circle cx="12" cy="12" r="8.5"/><path d="M12 12l4-2.5M12 12v-5"/></>,
  settings: <><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2 2M17 17l2 2M5 19l2-2M17 7l2-2"/></>,
  play: <><path d="M6 4l14 8-14 8V4z"/></>,
  save: <><path d="M5 4h11l3 3v13H5z"/><path d="M8 4v5h7V4M8 20v-6h8v6"/></>,
  plus: <><path d="M12 5v14M5 12h14"/></>,
  sun: <><circle cx="12" cy="12" r="4.5"/><path d="M12 2v2.5M12 19.5V22M2 12h2.5M19.5 12H22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M4.9 19.1l1.8-1.8M17.3 6.7l1.8-1.8"/></>,
  trash: <><path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13"/></>,
  spark: <><path d="M12 2l2.4 6.9L21 11l-6.6 2.1L12 20l-2.4-6.9L3 11l6.6-2.1L12 2z"/></>,
  runs: <><circle cx="12" cy="12" r="8.5"/><path d="M12 7v5l3.5 2"/></>,
  evals: <><path d="M4 6h10M4 12h10M4 18h6"/><path d="M17 5.5l2 2 3-3.5M17 15.5l2 2 3-3.5"/></>,
  versions: <><circle cx="6" cy="6" r="2.5"/><circle cx="6" cy="18" r="2.5"/><circle cx="18" cy="12" r="2.5"/><path d="M6 8.5v7M8 6h4a3 3 0 013 3v.5M8 18h4a3 3 0 003-3v-.5"/></>,
  download: <><path d="M12 4v11M7 11l5 5 5-5M5 20h14"/></>,
  search: <><circle cx="11" cy="11" r="7"/><path d="M20 20l-4-4"/></>,
  sdk: <><path d="M9 8l-4 4 4 4M15 8l4 4-4 4"/></>,
  audit: <><path d="M12 3l7 3v5c0 4.2-2.8 7.5-7 9-4.2-1.5-7-4.8-7-9V6l7-3z"/><path d="M9 12l2 2 4-4"/></>,
  memory: <><path d="M12 3l9 5-9 5-9-5 9-5z"/><path d="M3 12l9 5 9-5M3 16l9 5 9-5"/></>,
};

export function Icon({ name, size = 18 }: { name: string; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      {paths[name] ?? null}
    </svg>
  );
}
