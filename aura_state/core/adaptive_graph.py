"""
Adaptive Compute Graph: A self-mutating DAG that evolves at runtime.

Every other framework has a static graph — the developer defines nodes and edges,
and they never change. Ours is different:

1. If a node consistently fails (>N times), auto-insert a Reflexion Node before it
2. If cache hit rate for a path exceeds threshold, prune that node from LLM calls
3. If MCTS discovers a high-scoring path not in the graph, propose a new edge
"""
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger("aura_state")


@dataclass
class NodeHealthMetrics:
    """Runtime health metrics for a single Node."""
    total_executions: int = 0
    failures: int = 0
    cache_hits: int = 0
    total_latency_ms: float = 0.0
    last_failure_time: Optional[float] = None
    consecutive_failures: int = 0
    
    @property
    def fail_rate(self) -> float:
        if self.total_executions == 0:
            return 0.0
        return self.failures / self.total_executions
    
    @property
    def cache_hit_rate(self) -> float:
        if self.total_executions == 0:
            return 0.0
        return self.cache_hits / self.total_executions
    
    @property
    def avg_latency_ms(self) -> float:
        if self.total_executions == 0:
            return 0.0
        return self.total_latency_ms / self.total_executions


@dataclass
class RuntimeEdge:
    """An edge proposed by the engine at runtime (not developer-defined)."""
    from_node: str
    to_node: str
    confidence: float
    evidence: str
    proposed_at: float = field(default_factory=time.time)
    accepted: bool = False


class AdaptiveDAG:
    """
    A self-mutating DAG that evolves based on runtime signals.
    
    This wraps the engine's static `_transitions` dict and adds
    runtime intelligence:
    
    - Auto-inserts reflexion nodes when failure thresholds are breached
    - Proposes edge additions when MCTS discovers high-scoring paths
    - Tracks per-node health for compute optimization
    """
    
    # Thresholds for self-mutation
    FAILURE_THRESHOLD = 3              # consecutive failures before reflexion
    CACHE_BYPASS_THRESHOLD = 0.95      # cache hit rate to skip LLM
    MIN_EXECUTIONS_FOR_BYPASS = 10     # minimum data points before pruning
    
    def __init__(self):
        self._health: Dict[str, NodeHealthMetrics] = defaultdict(NodeHealthMetrics)
        self._proposed_edges: List[RuntimeEdge] = []
        self._bypassed_nodes: Set[str] = set()
        self._reflexion_injected: Set[str] = set()
    
    # ── Health Tracking ──
    
    def record_execution(self, node_name: str, success: bool, latency_ms: float, cache_hit: bool = False):
        """Record the outcome of a node execution."""
        m = self._health[node_name]
        m.total_executions += 1
        m.total_latency_ms += latency_ms
        
        if cache_hit:
            m.cache_hits += 1
        
        if not success:
            m.failures += 1
            m.consecutive_failures += 1
            m.last_failure_time = time.time()
        else:
            m.consecutive_failures = 0
    
    def get_health(self, node_name: str) -> NodeHealthMetrics:
        return self._health[node_name]
    
    # ── Self-Mutation Decisions ──
    
    def should_inject_reflexion(self, node_name: str) -> bool:
        """Check if a Reflexion Node should be auto-inserted before this node."""
        if node_name in self._reflexion_injected:
            return False  # Already injected
        
        m = self._health[node_name]
        if m.consecutive_failures >= self.FAILURE_THRESHOLD:
            logger.warning(
                f"[AdaptiveDAG] Node '{node_name}' has {m.consecutive_failures} consecutive failures. "
                f"Recommending reflexion injection."
            )
            return True
        return False
    
    def mark_reflexion_injected(self, node_name: str):
        self._reflexion_injected.add(node_name)
    
    def should_bypass_llm(self, node_name: str) -> bool:
        """Check if this node has enough cache hits to skip LLM entirely."""
        m = self._health[node_name]
        
        if m.total_executions < self.MIN_EXECUTIONS_FOR_BYPASS:
            return False
        
        if m.cache_hit_rate >= self.CACHE_BYPASS_THRESHOLD:
            if node_name not in self._bypassed_nodes:
                logger.info(
                    f"[AdaptiveDAG] Node '{node_name}' Cache hit rate {m.cache_hit_rate:.1%} "
                    f"exceeds {self.CACHE_BYPASS_THRESHOLD:.0%}. Marking for LLM bypass."
                )
                self._bypassed_nodes.add(node_name)
            return True
        
        return False
    
    # ── Edge Proposals ──
    
    def propose_edge(self, from_node: str, to_node: str, confidence: float, evidence: str):
        """
        Propose a new edge that the MCTS simulation discovered.
        These are logged and require review — never auto-committed.
        """
        edge = RuntimeEdge(
            from_node=from_node,
            to_node=to_node,
            confidence=confidence,
            evidence=evidence,
        )
        self._proposed_edges.append(edge)
        logger.info(
            f"[AdaptiveDAG] EDGE PROPOSAL: {from_node} → {to_node} "
            f"(confidence={confidence:.2f}, evidence='{evidence}')"
        )
    
    def get_proposed_edges(self) -> List[RuntimeEdge]:
        return [e for e in self._proposed_edges if not e.accepted]
    
    def accept_edge(self, from_node: str, to_node: str) -> bool:
        """Accept a proposed edge, making it available for future transitions."""
        for edge in self._proposed_edges:
            if edge.from_node == from_node and edge.to_node == to_node and not edge.accepted:
                edge.accepted = True
                logger.info(f"[AdaptiveDAG] Edge accepted: {from_node} → {to_node}")
                return True
        return False
    
    # ── Reporting ──
    
    def get_health_report(self) -> Dict[str, Dict[str, Any]]:
        """Get a full health report for all nodes."""
        report = {}
        for name, m in self._health.items():
            report[name] = {
                "total_executions": m.total_executions,
                "fail_rate": round(m.fail_rate, 3),
                "cache_hit_rate": round(m.cache_hit_rate, 3),
                "avg_latency_ms": round(m.avg_latency_ms, 1),
                "consecutive_failures": m.consecutive_failures,
                "bypassed": name in self._bypassed_nodes,
                "reflexion_injected": name in self._reflexion_injected,
            }
        return report
    
    @property
    def bypassed_nodes(self) -> Set[str]:
        return self._bypassed_nodes.copy()
    
    @property
    def total_proposed_edges(self) -> int:
        return len(self._proposed_edges)
