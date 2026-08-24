"""
Runtime health + routing statistics for the engine.

Tracks per-node execution health (for reporting) and a per-edge
Beta-Bernoulli posterior (for the Thompson-sampling bandit router). This is
deliberately small: it records real signals the router and reports consume. It
does NOT mutate the graph -- transitions are developer-declared and verified;
the engine never invents edges at runtime.
"""
import logging
import time
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger("aura_state")


@dataclass
class NodeHealthMetrics:
    """Runtime health metrics for a single Node."""
    total_executions: int = 0
    failures: int = 0
    total_latency_ms: float = 0.0
    last_failure_time: Optional[float] = None

    @property
    def fail_rate(self) -> float:
        if self.total_executions == 0:
            return 0.0
        return self.failures / self.total_executions

    @property
    def avg_latency_ms(self) -> float:
        if self.total_executions == 0:
            return 0.0
        return self.total_latency_ms / self.total_executions


@dataclass
class EdgeStats:
    """Beta-Bernoulli posterior for a single directed transition (edge).

    Success/failure are Bernoulli outcomes, so the reward is bounded in [0,1]
    by construction -- no reward-scale constant is needed. The posterior over
    the edge's success probability is Beta(alpha, beta), seeded from uniform
    priors (alpha0 = beta0 = 1). Stale evidence is discounted before each
    update so the router tracks non-stationary workflows.
    """
    alpha: float = 1.0            # prior successes + 1 (uniform Beta(1,1))
    beta: float = 1.0             # prior failures + 1
    last_update: Optional[float] = None

    def record(self, success: bool, discount: float = 0.98) -> None:
        # Decay the surplus over the uniform prior so old outcomes fade.
        self.alpha = 1.0 + (self.alpha - 1.0) * discount
        self.beta = 1.0 + (self.beta - 1.0) * discount
        if success:
            self.alpha += 1.0
        else:
            self.beta += 1.0
        self.last_update = time.time()

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


class AdaptiveDAG:
    """
    Per-node health tracking + per-edge routing posteriors.

    Named "adaptive" only in the sense that routing adapts to observed edge
    outcomes. The graph topology itself is fixed and developer-declared.
    """

    def __init__(self):
        self._health: Dict[str, NodeHealthMetrics] = defaultdict(NodeHealthMetrics)
        self._edge_stats: Dict[Tuple[str, str], EdgeStats] = defaultdict(EdgeStats)

    # ── Health tracking ──

    def record_execution(self, node_name: str, success: bool, latency_ms: float):
        """Record the outcome of a node execution."""
        m = self._health[node_name]
        m.total_executions += 1
        m.total_latency_ms += latency_ms
        if not success:
            m.failures += 1
            m.last_failure_time = time.time()

    def get_health(self, node_name: str) -> NodeHealthMetrics:
        return self._health[node_name]

    def get_health_report(self) -> Dict[str, Dict[str, Any]]:
        """Get a full health report for all nodes."""
        report = {}
        for name, m in self._health.items():
            report[name] = {
                "total_executions": m.total_executions,
                "fail_rate": round(m.fail_rate, 3),
                "avg_latency_ms": round(m.avg_latency_ms, 1),
            }
        return report

    # ── Per-edge routing posterior (Thompson / Beta-Bernoulli) ──

    def get_edge_stats(self, from_node: str, to_node: str) -> EdgeStats:
        return self._edge_stats[(from_node, to_node)]

    def record_edge_outcome(self, from_node: str, to_node: str, success: bool):
        """Update the Beta-Bernoulli posterior for a taken transition."""
        self._edge_stats[(from_node, to_node)].record(success)

    def sample_edge_score(self, from_node: str, to_node: str, rng) -> float:
        """Draw one Thompson sample from the edge's Beta posterior.

        ``rng`` is a ``random.Random`` instance (seedable for deterministic
        tests). Sampling -- rather than taking the mean -- is what balances
        exploration against exploitation without any tuning constant.
        """
        s = self._edge_stats[(from_node, to_node)]
        return rng.betavariate(s.alpha, s.beta)
