import json
import logging
import math
from typing import List, Dict, Any, Optional

logger = logging.getLogger("aura_state")

class BootstrapTeleprompter:
    """
    KNN-based few-shot optimizer.
    
    Finds the K most similar successful past executions and injects them
    as few-shot demonstrations into the system prompt.
    """
    def __init__(self, k_neighbors: int = 3):
        self.k = k_neighbors
        self.successful_traces: Dict[str, List[Dict[str, Any]]] = {}
        
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        return dot_product / (magnitude1 * magnitude2)
        
    def _mock_embedding(self, text: str) -> List[float]:
        """
        Simple character-based embedding for testing.
        In production, swap for openai.Embedding or a local model.
        """
        vec = [0.0] * 256
        for i, char in enumerate(text[:256]):
            vec[i] = float(ord(char))
            
        mag = math.sqrt(sum(v*v for v in vec))
        if mag > 0:
             vec = [v / mag for v in vec]
        return vec
        
    def _bootstrap_dataset(self, dataset: List[Dict[str, Any]]):
        """Filters the dataset to keep only successful executions."""
        for trace in dataset:
            if trace.get("success"):
                node_name = trace["node"]
                if node_name not in self.successful_traces:
                    self.successful_traces[node_name] = []
                    
                trace["embedding"] = self._mock_embedding(trace["input"])
                self.successful_traces[node_name].append(trace)

    def optimize_node(self, node_name: str, current_prompt: str, new_user_input: str) -> str:
        """
        Finds the K most similar past successes and appends them
        as few-shot examples to the system prompt.
        """
        if node_name not in self.successful_traces or not self.successful_traces[node_name]:
            return current_prompt
            
        # O(N) KNN Search
        query_vec = self._mock_embedding(new_user_input)
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
