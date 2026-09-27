// Shareable proof badge — an SVG generated from a real verify result.
// HONEST by construction: it only says "VERIFIED" when the design has no
// violation; a vulnerable agent gets a "N findings" badge, never a green one.

export interface ProofSummary {
  taint?: string; tri?: string;
  ctlok: number; ctln: number; z3ok: number; z3n: number;
  violated: boolean; findings: number;
}

export function proofSummary(verify: any): ProofSummary {
  const taint = verify?.taint?.verdict;                 // PROVEN | VIOLATED
  const tri = verify?.trifecta?.verdict;                // PROVEN | CLOSED
  const ctl = verify?.ctl || [];
  const z3 = verify?.obligations || [];
  const ctlok = ctl.filter((c: any) => c.verdict === "PROVEN").length;
  const z3ok = z3.filter((o: any) => o.consistent).length;
  const triFindings = (verify?.trifecta?.findings || []).length;
  const taintViol = (verify?.taint?.violations || []).length;
  const violated =
    taint === "VIOLATED" || tri === "CLOSED" ||
    ctl.some((c: any) => c.verdict !== "PROVEN") ||
    z3.some((o: any) => !o.consistent);
  const findings = triFindings + taintViol +
    ctl.filter((c: any) => c.verdict !== "PROVEN").length +
    z3.filter((o: any) => !o.consistent).length;
  return { taint, tri, ctlok, ctln: ctl.length, z3ok, z3n: z3.length, violated, findings };
}

const esc = (s: string) => String(s).replace(/[<>&]/g, (c) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;" } as any)[c]);

function row(y: number, ok: boolean, label: string, detail: string): string {
  const mark = ok ? "✓" : "✗";
  const col = ok ? "#1c8a5b" : "#c23b3b";
  return `
    <text x="20" y="${y}" font-size="13" fill="${col}" font-family="monospace">${mark}</text>
    <text x="40" y="${y}" font-size="13" fill="#3a3a44">${esc(label)}</text>
    <text x="380" y="${y}" font-size="12.5" fill="#8a8a94" text-anchor="end" font-family="monospace">${esc(detail)}</text>`;
}

/** A self-contained proof-card SVG (no external fonts/assets). */
export function badgeSvg(agentName: string, verify: any, hash?: string): string {
  const s = proofSummary(verify);
  const verdict = s.violated ? `${s.findings} finding${s.findings === 1 ? "" : "s"}` : "VERIFIED";
  const vcol = s.violated ? "#c23b3b" : "#1c8a5b";
  const rows = [
    row(78, s.tri === "PROVEN", "lethal trifecta", s.tri === "PROVEN" ? "safe" : "vulnerable"),
    row(100, s.taint === "PROVEN", "taint dataflow", (s.taint || "—").toLowerCase()),
    row(122, s.ctln > 0 && s.ctlok === s.ctln, "CTL reachability", `${s.ctlok}/${s.ctln}`),
    row(144, s.z3n === 0 || s.z3ok === s.z3n, "Z3 obligations", s.z3n ? `${s.z3ok}/${s.z3n}` : "n/a"),
  ].join("");
  return `<svg xmlns="http://www.w3.org/2000/svg" width="420" height="182" viewBox="0 0 420 182" font-family="system-ui, -apple-system, Segoe UI, sans-serif">
  <rect x="0.5" y="0.5" width="419" height="181" rx="12" fill="#ffffff" stroke="#e6e6ea"/>
  <text x="20" y="30" font-size="13.5" fill="#3d3aa8" font-weight="700">∴ aura-state</text>
  <text x="20" y="47" font-size="11.5" fill="#8a8a94">design-time proof · ${esc(agentName)}</text>
  <rect x="300" y="18" width="102" height="26" rx="13" fill="${vcol}"/>
  <text x="351" y="35" font-size="12.5" fill="#ffffff" font-weight="700" text-anchor="middle">${esc(verdict)}</text>
  <line x1="20" y1="58" x2="400" y2="58" stroke="#eeeef2"/>
  ${rows}
  <text x="20" y="170" font-size="10.5" fill="#b0b0b8" font-family="monospace">${hash ? "contract " + esc(hash) : "aura-state · verified locally"}</text>
</svg>`;
}

/** Markdown/HTML embed snippets that reference a saved badge file. */
export function badgeEmbed(file = "aura-proof.svg", repo = "https://github.com/munshi007/Aura-State"): { md: string; html: string } {
  return {
    md: `[![Verified by aura-state](${file})](${repo})`,
    html: `<a href="${repo}"><img src="${file}" alt="Verified by aura-state" height="182"></a>`,
  };
}
