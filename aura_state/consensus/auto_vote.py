from enum import Enum
from typing import List, Type, Any, Optional
from pydantic import BaseModel
import collections

class ConsensusStrategy(Enum):
    MAJORITY_VOTE = "majority_vote"
    UNANIMOUS = "unanimous"
    FIRST_VALID = "first_valid"

class AutoConsensus:
    """
    Implements the Auto-Consensus feature by taking multiple identical LLM runs
    and comparing their Pydantic extractions to find the true answer, mitigating hallucination.
    """
    
    @staticmethod
    def resolve(runs: List[BaseModel], strategy: ConsensusStrategy = ConsensusStrategy.MAJORITY_VOTE) -> Optional[BaseModel]:
        """
        Resolves multiple Pydantic model outputs into a single trusted output based on the strategy.
        """
        if not runs:
            return None
            
        if strategy == ConsensusStrategy.FIRST_VALID:
            return runs[0]
            
        # Serialize models to strings for easy hashing/counting
        # We use model_dump_json() to ensure consistent string representation of the data
        serialized_runs = [run.model_dump_json() for run in runs]
        
        counter = collections.Counter(serialized_runs)
        
        if strategy == ConsensusStrategy.MAJORITY_VOTE:
            # Find the most common serialized string
            most_common, count = counter.most_common(1)[0]
            # If the most common only appeared once (and we had >1 run), there's no majority
            if count == 1 and len(runs) > 1:
                # Fallback to first valid, or raise an error in strict mode
                return runs[0] 
                
            # Find the original model that matches the most common JSON
            for run in runs:
                if run.model_dump_json() == most_common:
                    return run
                    
        elif strategy == ConsensusStrategy.UNANIMOUS:
            most_common, count = counter.most_common(1)[0]
            if count == len(runs):
                return runs[0]
            else:
                raise ValueError("Unanimous consensus required but divergent results found.")
                
        return runs[0] # Fallback
