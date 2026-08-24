"""
AuraEngine: a runtime for *verified* LLM state machines.

The point of Aura-State is that verification runs INSIDE the loop, not beside
it. Every transition that extracts data drives an extract -> verify -> retry
cycle whose "verify" step is real: the node's sandbox rule plus its Z3 proof
obligations must hold (fail-closed) before the extraction is accepted, and when
a node runs with consensus > 1 the repeated extractions are turned into a
conformal prediction interval. Graph-level CTL properties are checked at design
time via `verify()`.

process() per call:
  1. Bootstrap teleprompter -> optional KNN few-shot injection
  2. Verification loop -> extract -> (sandbox rule + Z3 obligations) -> retry
  3. Conformal interval over consensus runs (when consensus > 1)
  4. node.handle() -> developer routing/business logic
  5. Bandit router -> resolve an invalid/ambiguous transition (Thompson)
  6. AuraTrace -> serialize state for time-travel debugging
"""
import random
import time
import logging
from typing import Callable, Dict, Any, Optional, Type, List
from pydantic import BaseModel, ConfigDict
import instructor
from openai import OpenAI

from ..execution.tracer import AuraTrace
from ..compiler.dspy_tuner import BootstrapTeleprompter
from ..execution.sandbox import SandboxedInterpreter
from ..consensus.auto_vote import AutoConsensus, ConsensusStrategy
from ..memory.pruner import ContextPruner
from ..verification.conformal import conformal_from_extractions, ConformalResult
from .exceptions import StateTransitionError
from .adaptive_graph import AdaptiveDAG
from .verification_loop import VerificationLoop
from .providers import LLMProvider

logger = logging.getLogger("aura_state")


# --- Node ---

class Node:
    """
    Base class for defining workflow states.

    - `extracts`: optional Pydantic schema for LLM extraction.
    - `sandbox_rule`: a deterministic boolean checked in the no-exec sandbox.
    - `obligations`: Z3 proof obligations that must hold for the extracted /
      available data (fail-closed). These run inside the verification loop.
    - `consensus` > 1 runs the extraction repeatedly and (a) votes and
      (b) produces a conformal interval over the runs.
    - `confidence`: nominal coverage for that conformal interval.
    """
    system_prompt: str = ""
    extracts: Optional[Type[BaseModel]] = None
    sandbox_rule: Optional[str] = None
    obligations: List[str] = []
    consensus: int = 1
    consensus_strategy: ConsensusStrategy = ConsensusStrategy.MAJORITY_VOTE
    confidence: float = 0.9
    memory_context: Optional[List[str]] = None
    model: str = "gpt-4o"

    def handle(self, user_text: str, extracted_data: Optional[BaseModel] = None, memory: Optional[Dict[str, Any]] = None) -> tuple:
        """Override this method to define your Node's routing and business logic."""
        raise NotImplementedError(f"Node '{self.__class__.__name__}' must implement handle().")


class CompiledTransition(BaseModel):
    """A formal directed edge between two Node classes."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    from_node: Type[Node]
    to_node: Type[Node]
    condition: str = "true"


# --- Engine ---

class AuraEngine:
    """
    The main execution engine for verified LLM state machines.
    """

    def __init__(self, llm_client: Optional[OpenAI] = None, budget_usd: Optional[float] = None, route_seed: Optional[int] = None):
        # Core graph
        self._nodes: Dict[str, Node] = {}
        self._transitions: Dict[str, List[str]] = {}
        self._compiled_transitions: List[CompiledTransition] = []
        self._step_counter: int = 0

        # Router: seedable RNG for deterministic Thompson sampling, plus an
        # optional CTL-feasibility hook.
        self._route_rng = random.Random(route_seed)
        self._feasibility_fn: Optional[Callable[[str, str], bool]] = None

        # Single instructor-patched client (avoid creating duplicates)
        self.client = instructor.from_openai(llm_client) if llm_client else None

        # Core internals
        self.tracer = AuraTrace()
        self.compiler = BootstrapTeleprompter()
        self.sandbox = SandboxedInterpreter(llm_client=llm_client)
        self.adaptive_graph = AdaptiveDAG()
        self.verification_loop = VerificationLoop()
        self.provider = LLMProvider()

        # Per-step verification reports (the proof/interval evidence per run).
        self._verification_reports: List[Dict[str, Any]] = []

        if self.client:
            self.provider.register_client("gpt", self.client)
        if budget_usd:
            self.provider.set_budget(budget_usd)

    # ─────────────────────────────────────────────────────────
    # GRAPH REGISTRATION
    # ─────────────────────────────────────────────────────────

    def register(self, *node_classes: Type[Node]):
        """Registers Node classes into the engine's computational graph."""
        for cls in node_classes:
            if not cls.system_prompt:
                raise ValueError(f"Node '{cls.__name__}' must define a `system_prompt`.")
            instance = cls()
            self._nodes[cls.__name__] = instance
            if cls.__name__ not in self._transitions:
                self._transitions[cls.__name__] = []
            logger.info(f"Registered Node: {cls.__name__}")

    def connect(self, transitions: List[CompiledTransition]):
        """Registers directed edges in the state graph."""
        for t in transitions:
            from_name = t.from_node.__name__
            to_name = t.to_node.__name__

            if from_name not in self._nodes:
                self.register(t.from_node)
            if to_name not in self._nodes:
                self.register(t.to_node)

            if from_name not in self._transitions:
                self._transitions[from_name] = []
            self._transitions[from_name].append(to_name)

        self._compiled_transitions.extend(transitions)

    def compile(self, output_path: str = "flow.json"):
        """Compiles all registered Nodes into an auditable flow.json artifact."""
        from ..compiler.json_generator import generate_flow_json
        node_classes = {name: type(node) for name, node in self._nodes.items()}
        generate_flow_json(node_classes, self._compiled_transitions, output_path)
        logger.info(f"Compiled {len(self._nodes)} nodes → {output_path}")

    def load_dataset(self, dataset: List[Dict[str, Any]]):
        """Feeds historical data into the teleprompter for few-shot optimization."""
        self.compiler.compile(dataset)

    # ─────────────────────────────────────────────────────────
    # DESIGN-TIME GRAPH VERIFICATION (CTL)
    # ─────────────────────────────────────────────────────────

    def verify(self, properties: List[Dict[str, Any]], init_node: Optional[str] = None):
        """Model-check CTL properties over the compiled graph (design time).

        Returns the list of PropertyResult from the temporal verifier. Use
        this before deploying: it proves reachability / completion / ordering
        over the *graph*, which per-transition checks cannot.
        """
        from ..verification.temporal_verifier import verify_engine
        return verify_engine(self, properties, init_node=init_node)

    # ─────────────────────────────────────────────────────────
    # BANDIT ROUTING (Thompson sampling over feasible transitions)
    # ─────────────────────────────────────────────────────────

    def set_feasibility_filter(self, fn: Optional[Callable[[str, str], bool]]):
        """Wire a CTL-feasibility predicate ``fn(from_node, to_node) -> bool``.

        When set, the router restricts candidates to transitions the verifier
        deems valid, so routing can never disagree with the temporal-logic
        layer. Structural transitions are always required too.
        """
        self._feasibility_fn = fn

    def _is_feasible(self, current_node: str, target: str) -> bool:
        if target not in self._transitions.get(current_node, []):
            return False
        if self._feasibility_fn is not None:
            return self._feasibility_fn(current_node, target)
        return True

    def _route_select(self, current_node: str, state_history: Dict[str, Any]) -> str:
        """
        Resolve an ambiguous transition with a Thompson-sampling bandit.

        This is a contextual bandit, NOT Monte-Carlo Tree Search: each
        candidate edge carries a Beta-Bernoulli posterior over its success
        probability, and we draw one sample per feasible edge and take the
        argmax. Bernoulli reward in [0,1] means no exploration constant or
        reward scaling. Candidates are filtered to the CTL-feasible set first,
        so an infeasible transition is never selectable.
        """
        possible_targets = self._transitions.get(current_node, [])
        if not possible_targets:
            return "END"

        feasible = [t for t in possible_targets if self._is_feasible(current_node, t)]
        if not feasible:
            logger.warning(f"[Router] No feasible transition from '{current_node}'.")
            return "END"
        if len(feasible) == 1:
            return feasible[0]

        logger.info(f"[Router] Thompson-sampling ambiguous transition at '{current_node}'...")

        last_failed = state_history.get("last_failed_node")
        best_node = feasible[0]
        best_sample = -1.0
        for target in feasible:
            sample = self.adaptive_graph.sample_edge_score(current_node, target, self._route_rng)
            if target == last_failed:
                sample *= 0.5
            if sample > best_sample:
                best_sample = sample
                best_node = target

        logger.info(f"[Router] Selected '{best_node}' (Thompson sample: {best_sample:.3f})")
        return best_node

    # ─────────────────────────────────────────────────────────
    # CORE EXECUTION PIPELINE
    # ─────────────────────────────────────────────────────────

    def process(self, current_state: str, user_text: str, memory: Optional[Dict[str, Any]] = None, history: Optional[List[Dict[str, str]]] = None) -> tuple[str, Any]:
        """Run one verified transition. See class docstring for the stages."""
        if current_state not in self._nodes:
            raise StateTransitionError(f"Node '{current_state}' is not registered.")

        node = self._nodes[current_state]
        memory = memory or {}
        history = history or []
        self._step_counter += 1
        start_ms = time.time() * 1000
        report: Dict[str, Any] = {"step": self._step_counter, "node": current_state}

        # ── STAGE 1: Bootstrap Teleprompter Injection ──
        optimized_prompt = self.compiler.optimize_node(current_state, node.system_prompt, user_text)

        # Build messages
        if history and node.memory_context:
            messages = ContextPruner.prune(history, required_keys=node.memory_context)
        else:
            messages = [
                {"role": "system", "content": optimized_prompt},
                {"role": "user", "content": user_text},
            ]

        # ── STAGE 2: Verification loop (extract -> sandbox+Z3 -> retry) ──
        extracted_data = None
        consensus_runs: List[BaseModel] = []
        if node.extracts and self.client:
            def _extract_fn(prompt, text):
                msgs = [{"role": "system", "content": prompt}, {"role": "user", "content": text}]
                runs = []
                for _ in range(node.consensus):
                    runs.append(self.provider.extract(
                        model=node.model,
                        response_model=node.extracts,
                        messages=msgs,
                        node_name=current_state,
                    ))
                consensus_runs.clear()
                consensus_runs.extend(runs)
                if node.consensus > 1:
                    return AutoConsensus.resolve(runs, strategy=node.consensus_strategy)
                return runs[0]

            extracted_data, iterations, verified = self.verification_loop.run(
                node_name=current_state,
                user_text=user_text,
                system_prompt=optimized_prompt,
                extract_fn=_extract_fn,
                sandbox_rule=node.sandbox_rule,
                sandbox=self.sandbox,
                obligations=node.obligations,
            )
            report["extraction_verified"] = verified
            report["iterations"] = iterations
            if not verified:
                logger.warning(f"[{current_state}] Extraction not verified after {iterations} attempts.")

            # ── STAGE 3: Conformal interval over consensus runs ──
            if len(consensus_runs) >= 2:
                cres: ConformalResult = conformal_from_extractions(
                    [r.model_dump() for r in consensus_runs], confidence=node.confidence
                )
                report["conformal"] = cres
        elif node.sandbox_rule or node.obligations:
            # No LLM extraction, but the node still carries a deterministic
            # contract -- check it against `memory` (fixes the flagship
            # decision-node case where the rule reads prior state, not an
            # extraction).
            passed, error = self.verification_loop.verify_extraction(
                current_state, None, node.sandbox_rule, self.sandbox, user_text,
                obligations=node.obligations, data_override=memory,
            )
            report["contract_verified"] = passed
            if not passed:
                logger.warning(f"[{current_state}] Contract not satisfied: {error}")

        # ── STAGE 4: Developer Node Logic ──
        next_state, payload = node.handle(
            user_text=user_text,
            extracted_data=extracted_data,
            memory=memory,
        )

        # ── STAGE 5: Bandit-router resolution (if handle returned an invalid edge) ──
        allowed = self._transitions.get(current_state, [])
        if next_state not in allowed:
            if allowed:
                logger.warning(f"[{current_state}] Invalid transition '{next_state}'. Engaging bandit router fallback.")
                next_state = self._route_select(current_state, memory)
            else:
                latency = (time.time() * 1000) - start_ms
                self.adaptive_graph.record_execution(current_state, False, latency)
                raise StateTransitionError(
                    f"No valid edges from '{current_state}'. DAG is a dead end."
                )

        # ── STAGE 6: AuraTrace Serialization ──
        self.tracer.dump_node_state(
            step=self._step_counter,
            node_name=current_state,
            memory_context=memory,
            extracted=extracted_data,
        )

        # ── STAGE 7: Record health + edge outcome (feeds the bandit router) ──
        latency = (time.time() * 1000) - start_ms
        self.adaptive_graph.record_execution(current_state, True, latency)
        if next_state in self._transitions.get(current_state, []):
            self.adaptive_graph.record_edge_outcome(current_state, next_state, success=True)

        report["next_state"] = next_state
        self._verification_reports.append(report)
        logger.info(f"Transition: {current_state} → {next_state}")
        return next_state, payload

    # ─────────────────────────────────────────────────────────
    # REPORTING
    # ─────────────────────────────────────────────────────────

    def health_report(self) -> Dict[str, Any]:
        """Get the adaptive graph health report for all nodes."""
        return self.adaptive_graph.get_health_report()

    def cost_report(self) -> Dict[str, Any]:
        """Get the multi-provider cost report."""
        return self.provider.cost_tracker.get_report()

    def verification_metrics(self) -> List[Dict[str, Any]]:
        """Get the compound verification loop metrics."""
        return self.verification_loop.metrics

    def verification_reports(self) -> List[Dict[str, Any]]:
        """Per-step verification evidence (proof results + conformal intervals)."""
        return self._verification_reports
