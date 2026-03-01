"""
Compound Verification Loop: Extract → Verify → Reflect → Retry.

Inspired by secloop's Ralph Loop pattern, applied to LLM extraction accuracy.
Nobody else does this — every framework blindly trusts the LLM's first extraction.

We don't. We verify the extraction against deterministic rules, and if it fails,
we generate a structured self-critique and inject it as a negative example on retry.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Type
from dataclasses import dataclass, field
from pydantic import BaseModel

logger = logging.getLogger("aura_state")


@dataclass
class Reflection:
    """A single reflection from a failed extraction attempt."""
    node_name: str
    input_text: str
    extracted_data: Dict[str, Any]
    error: str
    critique: str
    attempt: int


class ReflectionMemory:
    """
    Episodic memory buffer that stores (node, input, error, critique) tuples.
    
    Unlike Reflexion (2023) which just stores text blobs, we store structured
    failure data that can be injected as typed negative examples.
    """
    
    def __init__(self, max_reflections_per_node: int = 10):
        self._buffer: Dict[str, List[Reflection]] = {}
        self._max = max_reflections_per_node
    
    def add(self, reflection: Reflection):
        node = reflection.node_name
        if node not in self._buffer:
            self._buffer[node] = []
        self._buffer[node].append(reflection)
        # Ring buffer: evict oldest if over capacity
        if len(self._buffer[node]) > self._max:
            self._buffer[node] = self._buffer[node][-self._max:]
    
    def get_reflections(self, node_name: str, max_count: int = 3) -> List[Reflection]:
        return self._buffer.get(node_name, [])[-max_count:]
    
    def format_as_negative_examples(self, node_name: str) -> str:
        """Format reflections as injectable Few-Shot negative examples."""
        reflections = self.get_reflections(node_name)
        if not reflections:
            return ""
        
        lines = ["## Previous Failures (DO NOT repeat these mistakes):"]
        for r in reflections:
            lines.append(f"- Attempt {r.attempt}: Extracted {r.extracted_data}")
            lines.append(f"  ERROR: {r.error}")
            lines.append(f"  LESSON: {r.critique}")
        return "\n".join(lines)
    
    @property
    def total_reflections(self) -> int:
        return sum(len(v) for v in self._buffer.values())


class VerificationLoop:
    """
    The Compound Verification Loop.
    
    After LLM extraction, runs a deterministic verification step.
    If verification fails:
      1. Generate a structured self-critique
      2. Inject the critique as a negative Few-Shot example
      3. Re-extract with the reflection context
      4. Repeat until verification passes or max_iterations hit
    
    This is the Ralph Loop pattern from secloop, but for LLM accuracy
    instead of security vulnerabilities.
    """
    
    COMPLETION_TOKEN = "<VERIFIED>"
    RETRY_TOKEN = "<RETRY>"
    
    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations
        self.memory = ReflectionMemory()
        self._iteration_metrics: List[Dict[str, Any]] = []
    
    def verify_extraction(
        self,
        node_name: str,
        extracted_data: Optional[BaseModel],
        sandbox_rule: Optional[str],
        sandbox,  # SandboxedInterpreter instance
        user_text: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify extracted data against the node's sandbox_rule.
        
        Returns:
            (passed: bool, error_message: Optional[str])
        """
        if not sandbox_rule or not extracted_data:
            return True, None
        
        try:
            data_dict = extracted_data.model_dump() if extracted_data else {}
            result = sandbox.compile_and_run(sandbox_rule, data_dict)
            
            if result is False or result == 0:
                return False, f"Sandbox rule returned falsy: {result}"
            
            return True, None
            
        except Exception as e:
            return False, f"Sandbox verification error: {str(e)}"
    
    def generate_critique(
        self,
        node_name: str,
        user_text: str,
        extracted_data: BaseModel,
        error: str,
        attempt: int,
    ) -> Reflection:
        """
        Generate a structured self-critique from a failed extraction.
        
        Unlike blind retry, this captures WHY the extraction failed
        so the next attempt has actionable context.
        """
        data_dict = extracted_data.model_dump() if extracted_data else {}
        
        # Deterministic critique: analyze which fields likely caused the failure
        critique_parts = []
        critique_parts.append(f"The extraction at '{node_name}' failed verification.")
        critique_parts.append(f"Error: {error}")
        critique_parts.append(f"Extracted values: {data_dict}")
        critique_parts.append("Re-extract with corrected values. Pay close attention to numerical accuracy.")
        
        reflection = Reflection(
            node_name=node_name,
            input_text=user_text[:200],
            extracted_data=data_dict,
            error=error,
            critique=" ".join(critique_parts),
            attempt=attempt,
        )
        
        self.memory.add(reflection)
        return reflection
    
    def run(
        self,
        node_name: str,
        user_text: str,
        system_prompt: str,
        extract_fn,  # Callable that performs LLM extraction
        sandbox_rule: Optional[str],
        sandbox,
    ) -> Tuple[Optional[BaseModel], int, bool]:
        """
        Execute the full verification loop.
        
        Returns:
            (extracted_data, iterations_used, verified)
        """
        for attempt in range(1, self.max_iterations + 1):
            # Inject reflections from previous failures
            negative_examples = self.memory.format_as_negative_examples(node_name)
            augmented_prompt = system_prompt
            if negative_examples:
                augmented_prompt = f"{system_prompt}\n\n{negative_examples}"
            
            # Extract
            extracted = extract_fn(augmented_prompt, user_text)
            
            if extracted is None:
                logger.warning(f"[VerificationLoop] {node_name}: Extraction returned None on attempt {attempt}")
                continue
            
            # Verify
            passed, error = self.verify_extraction(
                node_name, extracted, sandbox_rule, sandbox, user_text
            )
            
            metric = {
                "node": node_name,
                "attempt": attempt,
                "passed": passed,
                "error": error,
            }
            self._iteration_metrics.append(metric)
            
            if passed:
                logger.info(f"[VerificationLoop] {node_name}: {self.COMPLETION_TOKEN} on attempt {attempt}")
                return extracted, attempt, True
            
            # Reflect
            logger.warning(f"[VerificationLoop] {node_name}: {self.RETRY_TOKEN} attempt {attempt}/{self.max_iterations}: {error}")
            self.generate_critique(node_name, user_text, extracted, error, attempt)
        
        # Max iterations reached — return last extraction unverified
        logger.error(f"[VerificationLoop] {node_name}: Max iterations ({self.max_iterations}) reached. Proceeding unverified.")
        return extracted, self.max_iterations, False
    
    @property
    def metrics(self) -> List[Dict[str, Any]]:
        return self._iteration_metrics
