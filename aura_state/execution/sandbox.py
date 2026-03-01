import ast
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel
import instructor
from openai import OpenAI

from ..core.exceptions import AuraStateError

logger = logging.getLogger("aura_state.sandbox")

class SandboxExecutionError(AuraStateError):
    """Raised when the sandbox fails to compile or execute the generated AST."""
    pass

class CodeGeneration(BaseModel):
    python_code: str
    explanation: str

class SandboxedInterpreter:
    """
    A secure code interpreter that dynamically translates English math/logic prompts
    into deterministic Python code, preventing LLM calculation hallucinations.
    """
    def __init__(self, llm_client: Optional[OpenAI] = None):
        self.client = instructor.from_openai(llm_client) if llm_client else None

    def _validate_ast(self, code_str: str):
        """Validates the AST to prevent malicious code execution."""
        try:
            tree = ast.parse(code_str)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    raise SandboxExecutionError("Imports are strictly forbidden in sandbox.")
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in ['eval', 'exec', 'open', '__import__', 'globals', 'locals']:
                        raise SandboxExecutionError(f"Forbidden function call detected: {node.func.id}")
        except SyntaxError as e:
            raise SandboxExecutionError(f"Invalid Python syntax: {e}")

    def safe_exec(self, code_str: str, local_vars: Dict[str, Any]) -> Any:
        """Executes the Python code in a highly restricted namespace."""
        self._validate_ast(code_str)

        restricted_globals = {
            "__builtins__": {
                "abs": abs, "min": min, "max": max, "sum": sum,
                "round": round, "int": int, "float": float, "bool": bool,
                "str": str, "len": len
            }
        }
        
        try:
            # We enforce that the LLM assigns the final calculated value to 'result'
            exec(code_str, restricted_globals, local_vars)
            if "result" not in local_vars:
                raise SandboxExecutionError("The generated code failed to assign a value to the 'result' variable.")
            return local_vars["result"]
        except Exception as e:
            logger.error(f"Sandbox execution failed: {e}")
            raise SandboxExecutionError(f"Execution Error: {str(e)}")

    def compile_and_run(self, english_prompt: str, input_variables: Dict[str, Any]) -> Any:
        """
        Translates an English rule into Python and executes it deterministically.
        Includes a self-healing retry loop if the execution fails.
        """
        if not self.client:
            raise ValueError("OpenAI client must be provided for the Sandbox Interpreter.")
            
        system_prompt = (
            "You are a strict neurosymbolic compiler. "
            "Convert the user's plain English math/logic rules into pure, deterministic Python code. "
            "The code will be executed in a restricted sandbox without any imports. "
            f"You have access to the following variables: {list(input_variables.keys())}. "
            "You MUST assign the final computed value to a variable exactly named 'result'."
        )
        
        logger.info(f"Compiling math prompt to Python AST...")
        
        # Deterministic Self-Healing: We try execution up to 3 times, feeding errors back if needed
        last_error = None
        for attempt in range(3):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Variables available: {input_variables}\n\nRule to compile: {english_prompt}"}
            ]
            
            if last_error:
                messages.append({"role": "user", "content": f"Your previous code failed with error: {last_error}. Fix it and return valid Python."})
                
            generation = self.client.chat.completions.create(
                model="gpt-4o",
                response_model=CodeGeneration,
                messages=messages,
                temperature=0.1 # Math should be deterministic
            )
            
            sandbox_vars = input_variables.copy()
            try:
                return self.safe_exec(generation.python_code, sandbox_vars)
            except SandboxExecutionError as e:
                logger.warning(f"Self-Healing trigger {attempt+1}/3: {e}")
                last_error = str(e)
                
        raise SandboxExecutionError(f"Failed to compile and execute prompt after self-healing. Final error: {last_error}")
