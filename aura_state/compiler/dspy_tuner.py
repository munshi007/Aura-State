import json
import logging
import math
from typing import Callable, List, Dict, Any, Optional

logger = logging.getLogger("aura_state")


def char_stub_embedder(text: str) -> List[float]:
    """Deterministic character-based pseudo-embedding for OFFLINE TESTS ONLY.

    Ranks by byte value, not semantics -- meaningless for real similarity.
    Never used by default; must be injected explicitly via
    ``BootstrapTeleprompter(embedder=char_stub_embedder)``.
    """
    vec = [0.0] * 256
    for i, char in enumerate(text[:256]):
        vec[i] = float(ord(char))
    mag = math.sqrt(sum(v * v for v in vec))
    if mag > 0:
        vec = [v / mag for v in vec]
    return vec


def _openai_embedder(client, model: str = "text-embedding-3-small") -> Callable[[str], List[float]]:
    """Build a real embedding function backed by an OpenAI client."""
    def embed(text: str) -> List[float]:
        resp = client.embeddings.create(model=model, input=text)
        return list(resp.data[0].embedding)
    return embed


class BootstrapTeleprompter:
    """
    KNN-based few-shot optimizer.

    Finds the K most similar successful past executions and injects them
    as few-shot demonstrations into the system prompt.

    An embedder is required to compute similarity. Provide one of:
      - ``embedder``: any ``Callable[[str], List[float]]``
      - ``openai_client``: a real embedding model is used
    If neither is given, embedding fails loud (no meaningless default).
    """
    def __init__(
        self,
        k_neighbors: int = 3,
        embedder: Optional[Callable[[str], List[float]]] = None,
        openai_client: Optional[Any] = None,
    ):
        self.k = k_neighbors
        self.successful_traces: Dict[str, List[Dict[str, Any]]] = {}
        if embedder is not None:
            self._embedder = embedder
        elif openai_client is not None:
            self._embedder = _openai_embedder(openai_client)
        else:
            self._embedder = None

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        return dot_product / (magnitude1 * magnitude2)

    def _embed(self, text: str) -> List[float]:
        if self._embedder is None:
            raise RuntimeError(
                "BootstrapTeleprompter has no embedder. Pass embedder=... or "
                "openai_client=... (or char_stub_embedder for offline tests). "
                "The old ord()-based default was removed -- it ranked by byte "
                "value, not meaning."
            )
        return self._embedder(text)

    def _bootstrap_dataset(self, dataset: List[Dict[str, Any]]):
        """Filters the dataset to keep only successful executions."""
        for trace in dataset:
            if trace.get("success"):
                node_name = trace["node"]
                if node_name not in self.successful_traces:
                    self.successful_traces[node_name] = []
                    
                trace["embedding"] = self._embed(trace["input"])
                self.successful_traces[node_name].append(trace)

    def optimize_node(self, node_name: str, current_prompt: str, new_user_input: str) -> str:
        """
        Finds the K most similar past successes and appends them
        as few-shot examples to the system prompt.
        """
        if node_name not in self.successful_traces or not self.successful_traces[node_name]:
            return current_prompt
            
        # O(N) KNN Search
        query_vec = self._embed(new_user_input)
        scored_traces = []
        for trace in self.successful_traces[node_name]:
            score = self._cosine_similarity(query_vec, trace["embedding"])
            scored_traces.append((score, trace))
            
        # Sort by similarity descending, take Top K
        scored_traces.sort(key=lambda x: x[0], reverse=True)
        top_k = [x[1] for x in scored_traces[:self.k]]
        
        # Abstract the Few-Shot injection prompt
        few_shot_block = "\n\n--- FEW-SHOT DEMONSTRATIONS ---\n"
        few_shot_block += "Follow the structure of these examples:\n\n"
        for i, trace in enumerate(top_k):
            few_shot_block += f"EXAMPLE {i+1}:\n"
            few_shot_block += f"User Context: {trace['input']}\n"
            few_shot_block += f"Extracted Schema Validation Response:\n{json.dumps(trace['output'], indent=2)}\n\n"
            
        optimized_prompt = current_prompt + few_shot_block
        logger.info(f"Injected {len(top_k)} few-shot examples into node '{node_name}'")
        return optimized_prompt
        
    def compile(self, dataset: List[Dict[str, Any]]):
        """Loads and indexes the dataset for KNN lookup."""
        self._bootstrap_dataset(dataset)
        total = sum(len(v) for v in self.successful_traces.values())
        logger.info(f"Indexed {total} successful traces across {len(self.successful_traces)} nodes")
