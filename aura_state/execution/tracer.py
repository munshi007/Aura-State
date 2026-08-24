import os
import json
import logging
from datetime import datetime
from pydantic import BaseModel
from typing import Dict, Any, Optional

logger = logging.getLogger("aura_state")

# Bumped when the on-disk step-record shape changes.
_TRACE_SCHEMA_VERSION = 1
_REQUIRED_KEYS = {"schema_version", "step", "node", "memory", "extracted_data", "timestamp"}


class TraceFormatError(Exception):
    """Raised when a trace file on disk does not match the expected schema."""


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    """
    Durably write ``payload`` as JSON to ``path``.

    Writes to a sibling temp file, flushes + fsyncs it, then ``os.replace``
    (atomic on POSIX) into place, and finally fsyncs the directory so the
    rename itself is durable. A reader therefore only ever sees the old
    complete file or the new complete file — never a truncated one.
    """
    directory = os.path.dirname(path) or "."
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=4)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
    # fsync the directory entry so the rename survives a crash.
    dir_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


class AuraTrace:
    """
    State serializer for time-travel debugging.

    Saves node state to disk at every transition as JSON only. Trace payloads
    are plain data (dicts, strings, numbers, lists), so JSON round-trips them
    exactly and — unlike ``pickle`` — cannot execute code on load. A trace
    file an attacker can write or swap is therefore inert on read.

    If a run fails at step N, you can resume from step N-1 without re-running
    (and re-paying for) earlier LLM calls.
    """
    def __init__(self, trace_dir: str = ".aura_trace", session_id: Optional[str] = None):
        self.trace_dir = trace_dir
        # If resuming, use existing session ID. Otherwise generate new one.
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.trace_dir, self.session_id)

        if not os.path.exists(self.session_dir):
            os.makedirs(self.session_dir)

    def dump_node_state(self, step: int, node_name: str, memory_context: Dict[str, Any], extracted: Optional[BaseModel]):
        """Serializes the complete runtime context of a node to disk (atomic JSON)."""
        state = {
            "schema_version": _TRACE_SCHEMA_VERSION,
            "step": step,
            "node": node_name,
            "memory": memory_context,
            "extracted_data": extracted.model_dump() if extracted else None,
            "timestamp": datetime.now().isoformat(),
        }

        json_path = os.path.join(self.session_dir, f"step_{step:03d}_{node_name}.json")
        _atomic_write_json(json_path, state)

        logger.debug(f"Saved state for node '{node_name}' -> {json_path}")

    @staticmethod
    def _load_and_validate(path: str) -> Dict[str, Any]:
        """Load a trace step file as JSON and validate its top-level shape.

        Never deserializes as code. A malformed or legacy (e.g. pickled) file
        fails to parse as JSON or fails the schema check and is rejected with a
        clear error.
        """
        try:
            with open(path, "r") as f:
                state = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise TraceFormatError(
                f"Trace file '{path}' is not valid JSON (legacy .pkl or corrupt?): {e}"
            ) from e
        if not isinstance(state, dict) or not _REQUIRED_KEYS.issubset(state.keys()):
            raise TraceFormatError(
                f"Trace file '{path}' is missing required keys "
                f"{sorted(_REQUIRED_KEYS - set(state if isinstance(state, dict) else {}))}"
            )
        return state

    @classmethod
    def load_trace(cls, session_id: str, step: int, trace_dir: str = ".aura_trace") -> Dict[str, Any]:
        """Loads a previously saved state from disk (JSON, schema-validated)."""
        session_dir = os.path.join(trace_dir, session_id)
        if not os.path.exists(session_dir):
            raise FileNotFoundError(f"Trace session '{session_id}' not found in {trace_dir}")

        for filename in sorted(os.listdir(session_dir)):
            if filename.startswith(f"step_{step:03d}_") and filename.endswith(".json"):
                json_path = os.path.join(session_dir, filename)
                state = cls._load_and_validate(json_path)
                logger.info(f"Restored state for node '{state['node']}' from step {step}")
                return state

        raise FileNotFoundError(f"Step {step} not found in trace session '{session_id}'")
