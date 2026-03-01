import json
import math
from typing import List, Dict, Any, Optional

class BootstrapTeleprompter:
    """
    The True DSPy-Inspired Few-Shot Optimizer (Aura-Compiler).
    "Prompt Engineering is frail. Mathematical demonstration is robust."
    
    Instead of naively asking an LLM to rewrite English rules, we use K-Nearest Neighbors (KNN)
    to extract perfectly successful (Input -> Reasoning -> Pydantic Output) pairs from 
    a dataset, dynamically injecting them as Few-Shot demonstrations mathematically
    into the failing Nodes.
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
        Lightweight deterministic dimension mock embedding to keep the repository 
        framework deeply self-contained without demanding heavy local PyTorch dependencies.
        (In production, swap for openai.Embedding).
        """
        vec = [0.0] * 256
        for i, char in enumerate(text[:256]):
            vec[i] = float(ord(char))
            
        mag = math.sqrt(sum(v*v for v in vec))
        if mag > 0:
             vec = [v / mag for v in vec]
        return vec
        
    def _bootstrap_dataset(self, dataset: List[Dict[str, Any]]):
        """Filters the dataset to formally isolate mathematically perfect executions."""
        for trace in dataset:
            if trace.get("success"):
                node_name = trace["node"]
                if node_name not in self.successful_traces:
                    self.successful_traces[node_name] = []
                    
                trace["embedding"] = self._mock_embedding(trace["input"])
                self.successful_traces[node_name].append(trace)

    def optimize_node(self, node_name: str, current_prompt: str, new_user_input: str) -> str:
        """
        The Teleprompter Core:
        Executes a KNN search for the Top-K most mathematically similar successful traces
        and appends them to the system prompt as structural demonstrations.
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
        few_shot_block = "\n\n--- 🧠 AURA-COMPILER: OPTIMIZED FEW-SHOT DEMONSTRATIONS ---\n"
        few_shot_block += "Strictly adhere to the mathematical structure of the following examples:\n\n"
        for i, trace in enumerate(top_k):
            few_shot_block += f"EXAMPLE {i+1}:\n"
            few_shot_block += f"User Context: {trace['input']}\n"
            few_shot_block += f"Extracted Schema Validation Response:\n{json.dumps(trace['output'], indent=2)}\n\n"
            
        optimized_prompt = current_prompt + few_shot_block
        print(f"🧠 [Aura-Compiler] Injected {len(top_k)} high-dimensional KNN Few-Shot demonstrations into Node '{node_name}'.")
        return optimized_prompt
        
    def compile(self, dataset: List[Dict[str, Any]]):
        """Loads and prepares the semantic KNN index from the historical dataset."""
        print("🧠 [Aura-Compiler] Bootstrapping true DSPy-inspired teleprompter...")
        self._bootstrap_dataset(dataset)
        print(f"✨ [Aura-Compiler] Bootstrapped {sum(len(v) for v in self.successful_traces.values())} theoretically perfect executions across {len(self.successful_traces)} Nodes.")
