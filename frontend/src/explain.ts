// Plain-English explanations. Every entry is written for a non-expert; the
// "why" helpers are DETERMINISTIC — generated from the real verifier output,
// never an LLM guess. This is Aura's explainability layer.

export const GLOSSARY: Record<string, string> = {
  obligation: "A rule the output must always satisfy — e.g. amount ≤ 500. Aura compiles it to a Z3 formula and proves it holds for every possible value, not just the ones you happened to test.",
  z3: "A theorem prover (SMT solver). It explores all possible inputs mathematically. If it can't find a single input that breaks your rule, the rule is proven — a guarantee, not a sample.",
  ctl: "Computation Tree Logic — model-checks the whole graph. Is every node reachable? Does every path eventually finish? These are properties of the flow that per-node checks can't see.",
  taint: "Static dataflow analysis. It traces whether attacker-controlled data (an untrusted source) can reach a dangerous action (a sink) without passing a sanitizer — the classic prompt-injection risk.",
  untrusted: "A node that introduces attacker-controlled data — user text, a scraped web page. Taint originates here.",
  sink: "A node that does something irreversible in the real world — a payment, an email, a shell command. Tainted data must never reach it unsanitized.",
  sanitizer: "A node that clears taint. Data that passes through it is treated as safe for any downstream sink.",
  conformal: "Split-conformal prediction. From a calibration set it builds an interval with a finite-sample coverage guarantee (e.g. 90% of true values land inside) and makes no assumption about the model.",
  alpha: "The risk level. α = 0.10 means you accept being wrong at most 10% of the time; the coverage target is 1 − α = 90%.",
  consensus: "Run the extraction N times and vote. With N ≥ 3 you also get a conformal interval measuring how much the model disagrees with itself.",
  contract: "The compiled specification of your agent — nodes, obligations, properties — hashed. Two designs with the same hash are provably identical.",
  invariant: "An obligation that must hold across the WHOLE agent, not just one node — e.g. a refunded order was always approved first.",
  certificate: "A signed JSON document sealing the full design verdict (taint, CTL, obligations) with a SHA-256. Proof you can hand an auditor.",
  retry: "On a failed obligation the node re-prompts the model with the counterexample as feedback (counterexample-guided) up to N times.",
  counterexample: "A concrete input that breaks a rule — the exact reason a proof failed. Aura surfaces it so you fix the real cause, not a symptom.",
  entry: "The node the agent starts from when it runs.",
  reachable: "There exists a path from the entry to this node. If false, the node is dead code — it can never execute.",
  completes: "Every path from the entry eventually reaches a terminal node — the agent can't get stuck in a state with nowhere to go.",
  deadend: "A non-terminal node with no way forward. The agent would halt there unexpectedly.",
  audit: "A tamper-evident, append-only log of every action. Each entry seals the previous entry's hash, so editing or deleting any record breaks the chain and is detectable.",
  coverage: "The fraction of true values that fall inside the interval. A 90% target means, over the long run, at most 10% of true values fall outside.",
  provider: "Which LLM backend runs this node — Ollama (local), OpenAI, Gemini, DeepSeek. Each node can use its own.",
  hash: "A SHA-256 fingerprint. Any change to the content produces a completely different hash, so it proves whether two things are identical.",
  policy: "A content scan of your prompts, rules, and obligations for hardcoded secrets (API keys, passwords) and PII (emails, SSNs, card numbers). Taint proves structure; this catches leaked content.",
  routing: "When a node has more than one valid next step, the engine picks using a Thompson-sampling bandit over per-edge success posteriors — but only among transitions the CTL model-checker proved feasible. It learns which branch works as it runs.",
  health: "Per-node runtime metrics gathered during a run: how many times it executed, its failure rate, and average latency. These feed the adaptive router and this report.",
  fewshot: "Bootstrap a node's prompt from its past successes. The tuner embeds your examples, finds the K most similar to the new input, and appends them as few-shot demonstrations — improving extraction without changing code.",
  memory: "Context pruning keeps the system prompt and the last N messages, and injects any required keys — so long conversations stay within the model's window without losing what matters.",
  tool: "A declared external call — db.write, http.get, payment.refund, or your own. Aura does NOT execute it; it proves the data reaching it is sanitized and satisfies preconditions. Your code (or aura-runtime) binds the real implementation. During a design-time Run, a mock return stands in.",
};

export function statusReason(nodeId: string, verify: any): string | null {
  if (!verify) return null;
  const obl = (verify.obligations || []).find((o: any) => o.node === nodeId);
  const taintHit = (verify.taint?.violations || []).some((v: any) => v.sink === nodeId);
  if (taintHit) {
    const v = verify.taint.violations.find((x: any) => x.sink === nodeId);
    return `Violated: untrusted data from “${v?.source}” can reach this sink${v?.field && v.field !== "*" ? ` via field ${v.field}` : ""} with no sanitizer on the path. That's an injection risk — insert a sanitizer (Auto-repair does this for you).`;
  }
  if (obl && !obl.consistent) {
    return `Violated: this node's obligations contradict each other — no value can satisfy them all at once${obl.reason ? ` (${obl.reason})` : ""}. Loosen or fix one of them.`;
  }
  const ctlBad = (verify.ctl || []).find((c: any) => (c.property || "").startsWith(nodeId) && c.verdict === "VIOLATED");
  if (ctlBad) return `Violated: ${ctlBad.property} — this node is not reachable from the entry, so it can never run.`;
  if (obl) return `Proven: Z3 searched every possible value and found none that satisfies the schema while breaking these obligations. The guarantee holds.`;
  return `Proven: reachable from the entry, no taint reaches it, and it carries no unsatisfiable obligations.`;
}

export function stepReason(s: any): string {
  if (s.error) return `Errored: ${s.error}`;
  if (s.kind === "tool") {
    const t = s.tool ? `${s.tool}(…)` : "the tool";
    const gated = s.side_effect && s.side_effect !== "read" ? " Because it's a dangerous sink, taint analysis proved no unsanitized untrusted data reaches it." : "";
    return `Tool boundary — declared, not executed in the studio. Its preconditions were proven, so aura-runtime (or your code) can safely call ${t}.${gated} Routed to ${s.next}.`;
  }
  const got = s.extracted && Object.keys(s.extracted).length ? `extracted ${Object.keys(s.extracted).join(", ")}` : "ran the rule";
  const timing = s.ms != null ? ` in ${s.ms} ms` : "";
  const cons = s.consensus > 1 ? `, voted over ${s.consensus} samples` : "";
  const retry = s.iterations && s.iterations > 1 ? `, retried ${s.iterations}× on failed obligations` : "";
  if (s.abstained) return `Abstained${timing}: the risk controller judged the output too uncertain to act on, so the agent held back rather than guess.`;
  if (s.verified === true) return `Verified${timing}: ${got}${cons}${retry}; every obligation on this node was proven to hold. Routed to ${s.next}.`;
  if (s.verified === false) return `Not verified${timing}: ${got}${retry}, but an obligation could not be proven for this output — treated as fail-closed. Routed to ${s.next}.`;
  return `${got}${timing}. Routed to ${s.next}.`;
}

export function ctlReason(label: string, verdict: string): string {
  const ok = verdict === "PROVEN";
  if (label.startsWith("EF ")) return ok ? `${label.slice(3)} is reachable from the entry — it can run.` : `${label.slice(3)} is NOT reachable — it's dead code that can never execute.`;
  if (label === "AF terminal") return ok ? "Every path eventually finishes — the agent can't get stuck." : "Some path never reaches a terminal — the agent could get stuck.";
  if (label.includes("≺")) return ok ? `Ordering holds: on every path, ${label} is respected.` : `Ordering can be violated: a path reaches the second state before the first.`;
  if (label.startsWith("¬(")) return ok ? "The two states are never active together — mutual exclusion holds." : "The two states can be active together — mutual exclusion is violated.";
  return ok ? "Property proven." : "Property violated.";
}

export function taintStory(verify: any): string | null {
  if (!verify?.taint) return null;
  if (verify.taint.verdict === "PROVEN") return "No path carries untrusted data into a dangerous sink without a sanitizer. The dataflow is provably safe.";
  const v = verify.taint.violations?.[0];
  return v ? `Untrusted data enters at “${v.source}” and can reach the sink “${v.sink}” with nothing sanitizing it in between. Any attacker who controls the input to ${v.source} could influence what ${v.sink} does.` : "A taint violation was found.";
}
