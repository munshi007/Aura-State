# Launch kit — Aura-State

Positioning: **dev-first**. Hero = developers building agents who are tired of "chain calls and hope."
One-liner: **Build LLM agents you can actually prove things about.**
Landing page: https://claude.ai/code/artifact/bff455f7-c4c5-44f5-8127-ba820fb9b6f6

Honesty guardrail for all copy: it's a real, correct core (Z3/CTL/conformal, adversarial tests) but **v0.2, early, OSS — not a hardened enterprise product.** Never claim battle-tested / production-proven. The defense against skeptics is that the tests run the actual solvers against adversarial inputs.

---

## Product Hunt

**Name:** Aura-State
**Tagline (≤60 chars):** Build LLM agents you can prove things about
**Alt taglines:**
- Verification for AI agents that runs in the loop, not the sidebar
- Formal proofs for your agents — Z3, model checking, no API key to try

**Description (~260 chars):**
Most agent frameworks let you chain LLM calls and hope. Aura-State runs your workflow as a typed state machine and verifies every step in the loop — Z3 proves extracted data, CTL model-checks the graph, taint analysis makes it injection-proof, and conformal risk control decides when to escalate to a human. Open source, MIT, five runnable demos with no API key.

**First comment (maker):**
Hi PH 👋 I built Aura-State because "the agent verifier" in most stacks is either an assertion, an LLM judging another LLM, or a test suite that runs after the fact — none of which is a guarantee.

Aura-State moves verification *into* the loop:
- **Z3** proves each extraction against your obligations (`total == area * rate`); a hallucinated value is rejected with the counterexample, not passed downstream.
- **CTL model checking** proves reachability/completion/ordering over the whole graph before it runs.
- **Static taint** proves untrusted input can't reach a dangerous tool — injection-proof by construction, tracking provenance not content.
- **Conformal Risk Control** calibrates a threshold so the false-action rate is provably ≤ ε, else it escalates to a human.
- And the verifier **repairs the plan** from its own counterexamples, then compiles everything into a portable contract.

It's early (v0.2) and open source. Every guarantee is a demo you can run in ~10s with no API key, and the tests run the real solvers against adversarial inputs. Would love feedback from anyone building agents in high-stakes domains. Repo: github.com/munshi007/Aura-State

---

## Show HN

**Title:** Show HN: Aura-State – Verify LLM agents in the loop with Z3, CTL, and conformal prediction

**Body:**
Aura-State is a Python framework that runs an LLM agent as a typed state machine and verifies every step *in the loop* rather than beside it.

Concretely: a node declares proof obligations (`total == area * rate`) and they're checked with Z3 inside the extract→verify→retry loop — a value that can't be proven is rejected with the counterexample, not accepted. The workflow graph is model-checked with CTL (reachability, completion, ordering) at the init state. A static taint pass over the graph proves untrusted input can't reach a dangerous sink (injection-proof by construction — provenance, not content). Conformal Risk Control calibrates an act/abstain threshold with a finite-sample bound on the false-action rate. And counterexamples from any of these feed a replanner that repairs the design and re-verifies.

Why I built it: the "verification" in most agent stacks is an assertion, an LLM-judge, or a post-hoc eval — none of which is a guarantee, and the judge is itself unverifiable. I wanted the check to be a solver.

It's v0.2 and early. What I care most about getting right is correctness of the primitives, so the tests run the actual solvers against adversarial inputs (the classic `__subclasses__` sandbox escape, a wrong extraction, a genuine dead-end, empirical conformal coverage). Five demos run with no API key.

Repo (MIT): https://github.com/munshi007/Aura-State
Happy to go deep on the CTL init-state handling, the fail-closed AST→Z3 compiler, or the jackknife+ conformal — those are the parts people usually get subtly wrong.

---

## YC application answers

**What does your company do? (one line)**
Aura-State is an open-source framework that lets developers build LLM agents with formal guarantees — Z3 proofs, model checking, and calibrated risk control run inside the agent's loop.

**What is it, longer:**
Agents are moving into decisions where being wrong is expensive, but the tooling to *prove* an agent behaved is missing — teams rely on assertions, LLM-judges, and eval suites, which are heuristics, not guarantees. Aura-State runs the agent as a typed state machine and verifies each step with real solvers: Z3 for data obligations, CTL model checking for the graph, static taint for injection-safety, and conformal risk control for calibrated abstention. The design compiles into a portable, faithful-by-construction contract that a runtime can enforce.

**Why now?**
Three things converged in 2026: agents are being deployed into regulated, high-stakes workflows; the EU AI Act's high-risk obligations (traceability, human oversight) took effect; and the research on formal methods for LLM agents matured (CaMeL, AgentSpec, VeriGuard, conformal risk control). The verification techniques are proven; nobody has packaged them into a framework developers actually build in.

**Why us / why this is defensible?**
The moat isn't the orchestration — it's getting the verification primitives *correct*. Most "LLM verification" projects ship subtly wrong math (interpolated conformal quantiles, reversed CTL, fail-open provers). Aura-State's are correct and adversarially tested, and the design→contract compiler makes the spec faithful by construction — killing the "who writes and maintains the policy" problem every competitor concedes.

**Traction / status:** v0.2 open source, 117 tests, five runnable demos. Pre-launch.

---

## Launch tweet / thread

**Tweet 1:**
Most agent frameworks: chain LLM calls and hope.

Aura-State: run the agent as a state machine and *prove* each step.

Z3 rejects a hallucinated value in the loop. CTL model-checks the graph. Taint makes it injection-proof. Open source, no API key to try 🧵

**Tweet 2:**
The verifier isn't another LLM judging the first one — it's a solver.

A node declares `obligations = ["total == area * rate"]`. If the extraction can't be *proven* to satisfy it, it's rejected with the counterexample and retried. Fail-closed.

**Tweet 3:**
Injection-proof by construction: label a node `untrusted_source` and a tool `dangerous_sink`, and Aura-State statically proves untrusted data can't reach it — tracking provenance, not content, so encodings can't fool it.

Everyone else sells detection. This is impossibility.

**Tweet 4:**
"Knows when it doesn't know" — made real. Conformal Risk Control calibrates a threshold so the agent's false-action rate is provably ≤ 5%. Below it, it escalates to a human instead of guessing.

**Tweet 5:**
And when a check fails, the verifier *repairs the plan* from its own counterexample and re-verifies — then compiles the whole thing into a portable contract.

v0.2, MIT, 5 demos run in ~10s with no API key 👇
github.com/munshi007/Aura-State
