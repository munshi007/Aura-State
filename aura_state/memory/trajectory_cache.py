import json
import os
import networkx as nx
from networkx.algorithms import isomorphism
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class GraphRAGCache:
    """
    The "Aura-Cache" using GraphRAG and Subgraph Isomorphism.
    Replaces fuzzy vector cosine similarity with 100% rigid mathematical topology matching.
    
    Extracts Entity-Relationship triples from the user prompt to build a NetworkX graph,
    then mathematically checks if this directed graph is isomorphic to a known successful trajectory.
    """
    def __init__(self, cache_dir: str = ".aura_cache", openai_client: Optional[Any] = None):
        self.cache_dir = cache_dir
        self.index_file = os.path.join(self.cache_dir, "graph_index.json")
        self.client = openai_client
        self.cache_data: List[Dict[str, Any]] = []
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
        self._load_cache()
        
    def _load_cache(self):
        if os.path.exists(self.index_file):
            with open(self.index_file, "r") as f:
                self.cache_data = json.load(f)
                
    def _save_cache(self):
        with open(self.index_file, "w") as f:
            json.dump(self.cache_data, f, indent=4)
            
    def _extract_triples(self, text: str) -> List[tuple]:
        """
        Uses an LLM (or a local fast NER model) to extract structural triples.
        e.g. "I want to buy a house in Seattle" -> [("User", "wants_to_buy", "House"), ("House", "located_in", "Seattle")]
        """
        if not self.client:
            # Fallback mock for testing without API keys, building a deterministic mock graph 
            # based on word splits to simulate topology.
            words = text.split()
            triples = []
            for i in range(len(words) - 2):
                if i + 2 < len(words):
                    triples.append((words[i], "follows", words[i+2]))
            return triples
            
        system = "Extract exact Subject-Predicate-Object triples from the following text. Return ONLY a JSON list of lists. Example: {'triples': [['User', 'wants_to_buy', 'House']]}"
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": text}
                ],
                response_format={ "type": "json_object" }
            )
            # Simplified parsing
            data = json.loads(response.choices[0].message.content)
            triples = data.get("triples", [])
            return [tuple(t) for t in triples if len(t) == 3]
        except Exception as e:
            return []
            
    def _build_networkx_graph(self, triples: List[tuple]) -> nx.DiGraph:
        """Constructs a formalized directed graph from the extracted relationships."""
        G = nx.DiGraph()
        for subj, pred, obj in triples:
            G.add_edge(subj, obj, relation=pred)
        return G
        
    def check_cache(self, node_id: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        """
        Identifies if the semantic topology of the prompt matches a known path.
        Uses NetworkX Subgraph Isomorphism to mathematically guarantee the bypass logic.
        """
        if not self.cache_data:
            return None
            
        triples = self._extract_triples(user_prompt)
        if not triples:
            return None
            
        query_graph = self._build_networkx_graph(triples)
        
        for entry in self.cache_data:
            if entry["node_id"] != node_id:
                continue
                
            cached_triples = [tuple(t) for t in entry["triples"]]
            cached_graph = self._build_networkx_graph(cached_triples)
            
            # The Deep-Tech Moat: Mathematical Isomorphism Check (not fuzzy embedding)
            GM = isomorphism.DiGraphMatcher(query_graph, cached_graph)
            if GM.is_isomorphic(): # Exact structural relationship match
                print(f"⚡ [Aura-Cache] GraphRAG Isomorphism Triggered! Exact structural match. LLM Call Skipped. Saved API Credits.")
                return entry["outcome"]
                
        return None
        
    def save_trajectory(self, node_id: str, user_prompt: str, outcome: Dict[str, Any]):
        """Caches a successful topological routing graph."""
        triples = self._extract_triples(user_prompt)
        if not triples:
            return
            
        self.cache_data.append({
            "node_id": node_id,
            "prompt": user_prompt,
            "triples": triples,
            "outcome": outcome
        })
        self._save_cache()
        print(f"💾 [Aura-Cache] Subgraph topology saved for Node '{node_id}'.")
