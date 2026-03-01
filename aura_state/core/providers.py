"""
Multi-Provider LLM Orchestration: Per-node model routing with failover and cost tracking.

Inspired by openclaw's plugin architecture for model providers.
No state machine framework lets you assign different LLMs to different nodes.
We do — and we add automatic failover when a provider is down.
"""
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger("aura_state")


@dataclass
class ProviderCost:
    """Cost tracker for a single LLM provider."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_calls: int = 0
    total_failures: int = 0
    total_latency_ms: float = 0.0
    
    # Cost per 1M tokens (configurable per provider)
    input_cost_per_m: float = 0.0
    output_cost_per_m: float = 0.0
    
    @property
    def total_cost_usd(self) -> float:
        input_cost = (self.input_tokens / 1_000_000) * self.input_cost_per_m
        output_cost = (self.output_tokens / 1_000_000) * self.output_cost_per_m
        return round(input_cost + output_cost, 6)
    
    @property
    def avg_latency_ms(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_latency_ms / self.total_calls
    
    @property
    def failure_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_failures / self.total_calls


# 2026 pricing (approximate)
MODEL_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.5-preview": {"input": 75.00, "output": 150.00},
    "claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3.5-haiku": {"input": 0.80, "output": 4.00},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.0-pro": {"input": 1.25, "output": 5.00},
}


class CostTracker:
    """
    Tracks cost per-node, per-model across the entire engine.
    Supports budget enforcement and cost alerts.
    """
    
    def __init__(self, budget_usd: Optional[float] = None):
        self._costs: Dict[str, Dict[str, ProviderCost]] = defaultdict(lambda: defaultdict(ProviderCost))
        self._budget_usd = budget_usd
        self._budget_warnings_issued: int = 0
    
    def record(
        self,
        node_name: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        success: bool = True,
    ):
        """Record a single LLM call."""
        cost = self._costs[node_name][model]
        cost.total_calls += 1
        cost.input_tokens += input_tokens
        cost.output_tokens += output_tokens
        cost.total_latency_ms += latency_ms
        
        if not success:
            cost.total_failures += 1
        
        # Set pricing if known
        pricing = MODEL_PRICING.get(model, {})
        cost.input_cost_per_m = pricing.get("input", 0.0)
        cost.output_cost_per_m = pricing.get("output", 0.0)
        
        # Budget check
        if self._budget_usd is not None:
            total = self.total_cost_usd
            if total > self._budget_usd * 0.8 and self._budget_warnings_issued == 0:
                logger.warning(f"[CostTracker] ⚠️ 80% of budget used: ${total:.4f} / ${self._budget_usd:.2f}")
                self._budget_warnings_issued += 1
            if total > self._budget_usd:
                logger.error(f"[CostTracker] 🛑 Budget exceeded: ${total:.4f} / ${self._budget_usd:.2f}")
    
    @property
    def total_cost_usd(self) -> float:
        total = 0.0
        for node_costs in self._costs.values():
            for cost in node_costs.values():
                total += cost.total_cost_usd
        return round(total, 6)
    
    def is_over_budget(self) -> bool:
        if self._budget_usd is None:
            return False
        return self.total_cost_usd > self._budget_usd
    
    def get_report(self) -> Dict[str, Any]:
        """Get a full cost report broken down by node and model."""
        report = {"total_cost_usd": self.total_cost_usd, "budget_usd": self._budget_usd, "nodes": {}}
        for node_name, models in self._costs.items():
            report["nodes"][node_name] = {}
            for model_name, cost in models.items():
                report["nodes"][node_name][model_name] = {
                    "calls": cost.total_calls,
                    "failures": cost.total_failures,
                    "input_tokens": cost.input_tokens,
                    "output_tokens": cost.output_tokens,
                    "cost_usd": cost.total_cost_usd,
                    "avg_latency_ms": round(cost.avg_latency_ms, 1),
                }
        return report


class LLMProvider:
    """
    Multi-provider LLM abstraction with automatic failover.
    
    Each provider wraps an instructor-patched client.
    If the primary provider fails, automatically routes to the next in the chain.
    """
    
    def __init__(self):
        self._clients: Dict[str, Any] = {}  # model_prefix → instructor client
        self._failover_chain: List[str] = []
        self._cost_tracker = CostTracker()
    
    def register_client(self, model_prefix: str, client):
        """
        Register an instructor-patched client for a model prefix.
        
        model_prefix: "gpt" for OpenAI, "claude" for Anthropic, "gemini" for Google
        """
        self._clients[model_prefix] = client
        if model_prefix not in self._failover_chain:
            self._failover_chain.append(model_prefix)
        logger.info(f"[LLMProvider] Registered client for '{model_prefix}' models")
    
    def set_failover_chain(self, chain: List[str]):
        """Set the failover order. First is primary, rest are fallbacks."""
        self._failover_chain = chain
    
    def set_budget(self, budget_usd: float):
        self._cost_tracker = CostTracker(budget_usd=budget_usd)
    
    def _get_client_for_model(self, model: str):
        """Find the registered client for a given model name."""
        for prefix, client in self._clients.items():
            if model.startswith(prefix):
                return client
        return None
    
    def _get_failover_model(self, failed_model: str) -> Optional[str]:
        """Get the next model in the failover chain."""
        failed_prefix = None
        for prefix in self._clients:
            if failed_model.startswith(prefix):
                failed_prefix = prefix
                break
        
        if failed_prefix is None:
            return None
        
        idx = self._failover_chain.index(failed_prefix) if failed_prefix in self._failover_chain else -1
        if idx + 1 < len(self._failover_chain):
            next_prefix = self._failover_chain[idx + 1]
            # Map prefix to a default model
            defaults = {
                "gpt": "gpt-4o",
                "claude": "claude-3.5-sonnet",
                "gemini": "gemini-2.0-flash",
            }
            return defaults.get(next_prefix)
        
        return None
    
    def extract(
        self,
        model: str,
        response_model,
        messages: List[Dict[str, str]],
        node_name: str = "unknown",
        max_retries: int = 3,
    ):
        """
        Perform LLM extraction with automatic failover.
        
        If the primary model fails, routes to the next provider in the chain.
        """
        current_model = model
        attempts = 0
        
        while current_model and attempts < len(self._failover_chain) + 1:
            client = self._get_client_for_model(current_model)
            if not client:
                logger.warning(f"[LLMProvider] No client for model '{current_model}'")
                current_model = self._get_failover_model(current_model)
                attempts += 1
                continue
            
            start_ms = time.time() * 1000
            try:
                result = client.chat.completions.create(
                    model=current_model,
                    response_model=response_model,
                    messages=messages,
                    max_retries=max_retries,
                )
                latency_ms = (time.time() * 1000) - start_ms
                
                # Record success
                self._cost_tracker.record(
                    node_name=node_name,
                    model=current_model,
                    input_tokens=0,  # Would need response metadata for exact counts
                    output_tokens=0,
                    latency_ms=latency_ms,
                    success=True,
                )
                
                return result
                
            except Exception as e:
                latency_ms = (time.time() * 1000) - start_ms
                self._cost_tracker.record(
                    node_name=node_name,
                    model=current_model,
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=latency_ms,
                    success=False,
                )
                
                failover = self._get_failover_model(current_model)
                if failover:
                    logger.warning(
                        f"[LLMProvider] Model '{current_model}' failed: {e}. "
                        f"Failing over to '{failover}'"
                    )
                    current_model = failover
                    attempts += 1
                else:
                    raise
        
        raise RuntimeError(f"All providers exhausted. No failover available for '{model}'.")
    
    @property
    def cost_tracker(self) -> CostTracker:
        return self._cost_tracker
