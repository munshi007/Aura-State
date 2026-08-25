# How Aura-State compares

**Short version:** the popular agent frameworks *orchestrate* (route steps, call
tools, manage state). Aura-State *verifies*. These are complementary layers, not
competitors — you can run Aura-State's checks alongside, or on top of, an agent
built in any of them.

We're not claiming to be a better orchestrator than LangGraph, or a better
multi-agent runtime than CrewAI. We're claiming something narrower and specific:
**nobody else proves things about the run with real solvers.** Their safety
features are validators, retries, and tool-call restrictions — engineering
controls, not formal guarantees. Useful, but a classifier or a re-ask is not a
proof.

## Capability matrix

Verification capabilities, as of 2026, to the best of our knowledge. Corrections
welcome — open an issue.

| Capability | Aura-State | LangGraph | CrewAI | AutoGen | Guardrails AI | Instructor |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| Orchestrate a multi-step agent | ✓ | ✓✓ | ✓✓ | ✓✓ | — | — |
| Structured extraction (typed output) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓✓ |
| Validate output (assertions / re-ask) | ✓ | ✓ | ✓ | ✓ | ✓✓ | ✓ |
| **Z3 proof of the extracted data** | ✓ | — | — | — | — | — |
| **Prove the spec isn't self-contradictory** | ✓ | — | — | — | — | — |
| **CTL model-checking of the graph** | ✓ | — | — | — | — | — |
| **Static taint / injection-proof dataflow** | ✓ | — | — | — | — | — |
| **Conformal risk control + abstention** | ✓ | — | — | — | — | — |
| **Pipeline-aware calibration (PASC)** | ✓ | — | — | — | — | — |
| **Faithful design→contract artifact** | ✓ | — | — | — | — | — |
| **Counterexample-guided repair** | ✓ | — | — | — | — | — |

✓✓ = a core strength · ✓ = supported · — = not a feature (to our knowledge)

The bolded rows are the point. Everything above them is table stakes that many
tools do — several do orchestration far better than we do. The bolded rows are
where Aura-State is, as far as we know, alone: they need a solver (Z3), a model
checker (CTL over a Kripke structure), a static dataflow analysis, or a
finite-sample statistical guarantee — not an LLM judging another LLM.

## Why "validation" isn't "verification"

The closest overlap is output validation (Guardrails AI, Instructor, LangGraph
node validators). The difference:

- **Validation** asserts a value looks right (a regex, a Pydantic type, a range
  check, or an LLM-as-judge). It runs *after* the fact and, when it uses a model,
  is itself stochastic.
- **Verification** *proves* a property. `total == area * rate` is checked by Z3
  with a counterexample when it fails; "untrusted input can't reach this tool" is
  a static dataflow proof over the graph, immune to encodings; "the false-action
  rate is ≤ 5%" is a finite-sample conformal bound. The checker is a solver, not
  a model.

Both are useful. Validation catches the easy cases cheaply; verification is for
the properties where being wrong is expensive and "probably fine" isn't good
enough.

## The honest positioning

Use LangGraph / CrewAI / AutoGen to build and run your agent. Use Aura-State to
**prove things about it** — verify the design before deploy, gate risky actions,
prove injection-safety, and emit an audit-ready contract. See
[`examples/cookbook/verify_existing_agent.py`](../examples/cookbook/verify_existing_agent.py)
for adding Aura-State's checks to an agent you already have.
