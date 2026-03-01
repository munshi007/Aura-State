import os
import json
import pickle
from datetime import datetime
from pydantic import BaseModel
from typing import Dict, Any, Optional

class AuraTrace:
    """
    The Time-Travel Debugger Module for Aura-State.
    Because Aura-State forces LLMs into strict Node boundaries, we can
    serialize the exact multi-dimensional context to disk at every transition.
    
    If an agent fails at Step 14, developers fix their code and `--resume` from the cache,
    instantly bypassing the first 13 expensive API calls.
    """
    def __init__(self, trace_dir: str = ".aura_trace", session_id: Optional[str] = None):
        self.trace_dir = trace_dir
        # If resuming, use existing session ID. Otherwise generate new one.
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.trace_dir, self.session_id)
        
        if not os.path.exists(self.session_dir):
            os.makedirs(self.session_dir)
            
    def dump_node_state(self, step: int, node_name: str, memory_context: Dict[str, Any], extracted: Optional[BaseModel]):
        """Serializes the complete runtime context of a node to disk."""
        state = {
            "step": step,
            "node": node_name,
            "memory": memory_context,
            "extracted_data": extracted.model_dump() if extracted else None,
            "timestamp": datetime.now().isoformat()
        }
        
        # 1. Save as JSON for developer auditability (reading the trace manually)
        json_path = os.path.join(self.session_dir, f"step_{step:03d}_{node_name}.json")
        with open(json_path, "w") as f:
            json.dump(state, f, indent=4)
            
        # 2. Save exact Python objects as Pickle for programmatic `--resume` injection
        pkl_path = os.path.join(self.session_dir, f"step_{step:03d}_{node_name}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(state, f)
            
        print(f"🔍 [Aura-Trace] Serialized rigorous state for Node '{node_name}' -> {json_path}")
        
    @classmethod
    def load_trace(cls, session_id: str, step: int, trace_dir: str = ".aura_trace") -> Dict[str, Any]:
        """
        The Time-Travel Injection protocol.
        Instantly restores complete context memory to bypass previous LLM API calls.
        """
        session_dir = os.path.join(trace_dir, session_id)
        if not os.path.exists(session_dir):
            raise FileNotFoundError(f"[Aura-Trace] Critical Error: Trace session {session_id} not found.")
            
        # Search for the specific step binary
        for filename in sorted(os.listdir(session_dir)):
            if filename.startswith(f"step_{step:03d}_") and filename.endswith(".pkl"):
                pkl_path = os.path.join(session_dir, filename)
                with open(pkl_path, "rb") as f:
                    state = pickle.load(f)
                    print(f"⏪ [Aura-Trace] Time-Travel successful. Restored Node '{state['node']}' from cache.")
                    return state
                    
        raise FileNotFoundError(f"[Aura-Trace] Step {step} not found in trace session {session_id}")
