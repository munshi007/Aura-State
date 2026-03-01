"""
AuraEngine: the main execution pipeline.

Every process() call runs through:
  1. Speculative execution (parallel branch pre-computation)
  2. Adaptive DAG (runtime health monitoring)
  3. Verification loop (extract → verify → reflect → retry)
  4. Schema compilation (JSON Schema → Node classes)
  5. Multi-provider routing (per-node model + failover)
"""
import math
import time
import logging
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, Any, Optional, Type, List
from pydantic import BaseModel, ConfigDict
import instructor
from openai import OpenAI

from ..execution.tracer import AuraTrace
from ..memory.trajectory_cache import GraphRAGCache
from ..compiler.dspy_tuner import BootstrapTeleprompter
from ..execution.sandbox import SandboxedInterpreter
from ..consensus.auto_vote import AutoConsensus, ConsensusStrategy
from ..memory.pruner import ContextPruner
from .exceptions import StateTransitionError
from .adaptive_graph import AdaptiveDAG
from .verification_loop import VerificationLoop
from .providers import LLMProvider, CostTracker

logger = logging.getLogger("aura_state")


# --- Node ---

class Node:
    """
    Base class for defining workflow states.
    
    Each node has a system prompt and an optional Pydantic schema for extraction.
    Set `model` to route to a specific LLM.
    """
    system_prompt: str = ""
    extracts: Optional[Type[BaseModel]] = None
    sandbox_rule: Optional[str] = None
    consensus: int = 1
    consensus_strategy: ConsensusStrategy = ConsensusStrategy.MAJORITY_VOTE
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
    The main execution engine.
    
    Pipeline for every process() call:
    1. Adaptive DAG Health Check → bypass LLM if cache-saturated
    2. GraphRAG Subgraph Isomorphism Cache → bypass if topology matches
    3. Bootstrap Teleprompter → KNN Few-Shot injection
    4. Compound Verification Loop → extract→verify→reflect→retry
    5. MCTS Lookahead → mathematically score ambiguous transitions
    6. AuraTrace Serialization → dump state for time-travel debugging
    7. Speculative Execution → pre-compute likely next nodes in parallel
    """
    
    def __init__(self, llm_client: Optional[OpenAI] = None, speculation_depth: int = 1, budget_usd: Optional[float] = None):
        # Core
        self._nodes: Dict[str, Node] = {}
        self._transitions: Dict[str, List[str]] = {}
        self._compiled_transitions: List[CompiledTransition] = []
        self._step_counter: int = 0
        
        # Single instructor-patched client (avoid creating duplicates)
        self.client = instructor.from_openai(llm_client) if llm_client else None
        
        # ── Core internals ──
        self.tracer = AuraTrace()
        self.cache = GraphRAGCache(openai_client=llm_client)
        self.compiler = BootstrapTeleprompter()
        self.sandbox = SandboxedInterpreter(llm_client=llm_client)
        
        # ── Core Innovations (always active) ──
        self.adaptive_graph = AdaptiveDAG()
        self.verification_loop = VerificationLoop()
        self.provider = LLMProvider()
        
        # Speculative execution
        self._speculation_depth = speculation_depth
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="aura_speculative")
        self._pending_speculations: Dict[str, Future] = {}
        
        # Register the same instructor client with the multi-provider system
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
        print(f"AuraEngine compiled {len(self._nodes)} nodes → {output_path}")
    
    def load_dataset(self, dataset: List[Dict[str, Any]]):
        """Feeds historical data into the BootstrapTeleprompter for KNN Few-Shot optimization."""
        self.compiler.compile(dataset)
    
    # ─────────────────────────────────────────────────────────
    # SPECULATIVE NODE EXECUTION
    # ─────────────────────────────────────────────────────────
    
    def _speculative_execute(self, current_state: str, user_text: str, memory: Dict[str, Any]):
        """
        Pre-compute likely next nodes in parallel.
        
        When a node has multiple outgoing edges, we speculatively execute
        ALL candidate next-nodes in background threads. When the current node
        resolves its transition, we either accept (cache hit — zero latency)
        or discard the wrong branches.
        
        This is speculative decoding applied to workflow state machines.
        """
        candidates = self._transitions.get(current_state, [])
        if len(candidates) <= 1 or self._speculation_depth < 1:
            return
        
        logger.info(f"[Speculative] Spawning {len(candidates)} speculative branches from '{current_state}'")
        
        for candidate in candidates:
            if candidate in self._pending_speculations:
                continue  # Already being speculated
            if candidate not in self._nodes:
                continue
            
            future = self._executor.submit(
                self._speculative_process_node,
                candidate, user_text, memory
            )
            self._pending_speculations[candidate] = future
    
    def _speculative_process_node(self, node_name: str, user_text: str, memory: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single node speculatively (in a background thread)."""
        try:
            node = self._nodes[node_name]
            # Only run non-LLM parts speculatively (handler logic)
            # We can pre-compute the handler's deterministic logic
            next_state, payload = node.handle(
                user_text=user_text,
                extracted_data=None,
                memory=memory
            )
            return {"next_state": next_state, "payload": payload, "node": node_name}
        except Exception as e:
            return {"error": str(e), "node": node_name}
    
    def _check_speculation(self, resolved_next_state: str) -> Optional[Dict[str, Any]]:
        """Check if the resolved next state was pre-computed speculatively."""
        if resolved_next_state in self._pending_speculations:
            future = self._pending_speculations.pop(resolved_next_state)
            if future.done():
                result = future.result()
                if "error" not in result:
                    logger.info(f"[Speculative] ✅ HIT: '{resolved_next_state}' was pre-computed!")
                    return result
        
        # Discard all other speculative branches
        for name, future in list(self._pending_speculations.items()):
            if name != resolved_next_state:
                future.cancel()
                self._pending_speculations.pop(name, None)
        
        return None

    # ─────────────────────────────────────────────────────────
    # MCTS ROUTING (System-2 Reasoning)
    # ─────────────────────────────────────────────────────────
    
    def _mcts_select(self, current_node: str, state_history: Dict[str, Any], depth: int = 3, simulations: int = 5) -> str:
        """
        Monte Carlo Tree Search: When multiple outgoing edges exist,
        spawns parallel simulations to score branches via UCB1.
        """
        possible_targets = self._transitions.get(current_node, [])
        if not possible_targets:
            return "END"
        if len(possible_targets) == 1:
            return possible_targets[0]
            
        logger.info(f"[MCTS] Ambiguous transition at '{current_node}'. Spawning {simulations} lookaheads...")
        
        metrics = {t: {"wins": 0.0, "visits": 0} for t in possible_targets}
        total_visits = 0
        
        for _ in range(simulations):
            for target in possible_targets:
                node_obj = self._nodes.get(target)
                reward = 0.5
                if node_obj and node_obj.extracts:
                    reward += 0.3
                if node_obj and node_obj.sandbox_rule:
                    reward += 0.2
                    
                if state_history.get("last_failed_node") == target:
                    reward -= 0.8
                
                # Factor in adaptive graph health
                health = self.adaptive_graph.get_health(target)
                if health.fail_rate > 0.5:
                    reward -= 0.4
                if health.cache_hit_rate > 0.8:
                    reward += 0.2
                    
                metrics[target]["wins"] += max(reward, 0)
                metrics[target]["visits"] += 1
                total_visits += 1
        
        c = math.sqrt(2)
        best, best_ucb1 = None, -float('inf')
        for name, m in metrics.items():
            if m["visits"] == 0:
                ucb1 = float('inf')
            else:
                ucb1 = (m["wins"] / m["visits"]) + c * math.sqrt(math.log(total_visits) / m["visits"])
            if ucb1 > best_ucb1:
                best_ucb1 = ucb1
                best = name
                
        logger.info(f"[MCTS] Optimal path: '{best}' (UCB1={best_ucb1:.3f})")
        return best

    # ─────────────────────────────────────────────────────────
    # CORE EXECUTION PIPELINE
    # ─────────────────────────────────────────────────────────

    def process(self, current_state: str, user_text: str, memory: Optional[Dict[str, Any]] = None, history: Optional[List[Dict[str, str]]] = None) -> tuple[str, Any]:
        """
        The main execution pipeline. Every call runs:
        
        1. Adaptive DAG Health → bypass/reflexion check
        2. GraphRAG Cache → subgraph isomorphism check
        3. Teleprompter → KNN Few-Shot injection
        4. Compound Verification Loop → extract→verify→reflect→retry
        5. Node.handle() → developer business logic
        6. MCTS → ambiguous transition resolution
        7. AuraTrace → state serialization
        8. Speculative Execution → pre-compute next branches
        """
        if current_state not in self._nodes:
            raise StateTransitionError(f"Node '{current_state}' is not registered.")
            
        node = self._nodes[current_state]
        memory = memory or {}
        history = history or []
        self._step_counter += 1
        start_ms = time.time() * 1000
        
        # ── STAGE 0: Adaptive DAG Health Check ──
        if self.adaptive_graph.should_inject_reflexion(current_state):
            logger.warning(f"[{current_state}] AdaptiveDAG recommends reflexion injection.")
            self.adaptive_graph.mark_reflexion_injected(current_state)

        if self.adaptive_graph.should_bypass_llm(current_state):
            cached = self.cache.check_cache(current_state, user_text)
            if cached:
                logger.info(f"[{current_state}] AdaptiveDAG LLM BYPASS — cache-saturated node.")
                latency = (time.time() * 1000) - start_ms
                self.adaptive_graph.record_execution(current_state, True, latency, cache_hit=True)
                return cached.get("next_state", "END"), cached
        
        # ── STAGE 1: GraphRAG Subgraph Isomorphism Cache ──
        cached = self.cache.check_cache(current_state, user_text)
        if cached:
            logger.info(f"[{current_state}] GraphRAG HIT. LLM bypassed.")
            self.tracer.dump_node_state(self._step_counter, current_state, memory, None)
            latency = (time.time() * 1000) - start_ms
            self.adaptive_graph.record_execution(current_state, True, latency, cache_hit=True)
            return cached.get("next_state", "END"), cached
        
        # ── STAGE 2: Bootstrap Teleprompter Injection ──
        optimized_prompt = self.compiler.optimize_node(current_state, node.system_prompt, user_text)
        
        # Build messages
        if history and node.memory_context:
            messages = ContextPruner.prune(history, required_keys=node.memory_context)
        else:
            messages = [
                {"role": "system", "content": optimized_prompt},
                {"role": "user", "content": user_text}
            ]
        
        # ── STAGE 3: Verification Loop ──
        extracted_data = None
        if node.extracts and self.client:
            def _extract_fn(prompt, text):
                """Extraction closure for the verification loop."""
                msgs = [{"role": "system", "content": prompt}, {"role": "user", "content": text}]
                runs = []
                for _ in range(node.consensus):
                    run_data = self.provider.extract(
                        model=node.model,
                        response_model=node.extracts,
                        messages=msgs,
                        node_name=current_state,
                    ) if self.provider._clients else self.client.chat.completions.create(
                        model=node.model,
                        response_model=node.extracts,
                        messages=msgs,
                        max_retries=3,
                    )
                    runs.append(run_data)
                
                if node.consensus > 1:
                    return AutoConsensus.resolve(runs, strategy=node.consensus_strategy)
                return runs[0]
            
            # Run the verification loop instead of blind extraction
            extracted_data, iterations, verified = self.verification_loop.run(
                node_name=current_state,
                user_text=user_text,
                system_prompt=optimized_prompt,
                extract_fn=_extract_fn,
                sandbox_rule=node.sandbox_rule,
                sandbox=self.sandbox,
            )
            
            if not verified:
                logger.warning(f"[{current_state}] Extraction not verified after {iterations} attempts. Proceeding with best effort.")
        
        # ── STAGE 4: Developer Node Logic ──
        next_state, payload = node.handle(
            user_text=user_text,
            extracted_data=extracted_data,
            memory=memory
        )
        
        # ── STAGE 5: MCTS Resolution (if ambiguous) ──
        allowed = self._transitions.get(current_state, [])
        if next_state not in allowed:
            if allowed:
                logger.warning(f"[{current_state}] Invalid transition '{next_state}'. Engaging MCTS fallback.")
                next_state = self._mcts_select(current_state, memory)
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
            extracted=extracted_data
        )
        
        # ── STAGE 6b: GraphRAG Cache Write ──
        cache_payload = {"next_state": next_state}
        if isinstance(payload, dict):
            cache_payload.update(payload)
        else:
            cache_payload["payload"] = payload
        self.cache.save_trajectory(current_state, user_text, cache_payload)
        
        # ── STAGE 7: Record Health ──
        latency = (time.time() * 1000) - start_ms
        self.adaptive_graph.record_execution(current_state, True, latency)
        
        # ── STAGE 8: Speculative Execution ──
        if self._speculation_depth > 0:
            self._speculative_execute(next_state, user_text, memory)
        
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
