"""
Schema Compiler: JSON Schema → Node classes with Pydantic extraction models.

Given a JSON Schema or OpenAPI spec, generates Node subclasses with
auto-generated system prompts and Levenshtein fuzzy matching for field correction.
"""
import logging
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field, create_model

logger = logging.getLogger("aura_state")


# --- Levenshtein Distance ---

def levenshtein_distance(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    
    prev = list(range(lb + 1))
    curr = [0] * (lb + 1)
    
    for i in range(1, la + 1):
        curr[0] = i
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            ins = prev[j] + 1
            delete = curr[j - 1] + 1
            sub = prev[j - 1] + cost
            curr[j] = min(ins, delete, sub)
        prev, curr = curr, prev
    
    return prev[lb]


def suggest_field(name: str, available: List[str], max_distance: int = 3) -> Optional[str]:
    """
    Find the closest field name using Levenshtein distance.
    Returns None if no match within max_distance.
    """
    if not available:
        return None
    
    best_dist = -1
    best_name = ""
    
    for field_name in available:
        d = levenshtein_distance(name.lower(), field_name.lower())
        if best_dist < 0 or d < best_dist:
            best_dist = d
            best_name = field_name
    
    if 0 <= best_dist <= max_distance:
        return best_name
    return None


# --- JSON Schema → Pydantic Model ---

# JSON Schema types → Python types
SCHEMA_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
}


def _resolve_type(prop: Dict[str, Any]) -> type:
    """Resolve a JSON Schema property to a Python type."""
    schema_type = prop.get("type", "string")
    
    if schema_type == "array":
        items_type = prop.get("items", {}).get("type", "string")
        inner = SCHEMA_TYPE_MAP.get(items_type, str)
        return List[inner]
    
    return SCHEMA_TYPE_MAP.get(schema_type, str)


def compile_pydantic_model(
    model_name: str,
    schema: Dict[str, Any],
) -> Type[BaseModel]:
    """
    Compile a JSON Schema object into a Pydantic model.
    
    Supports:
    - type mapping (string, integer, number, boolean, array)
    - descriptions → Field(description=...)
    - required fields
    - default values
    - enum constraints
    - minimum/maximum validation
    """
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    
    fields = {}
    for field_name, prop in properties.items():
        python_type = _resolve_type(prop)
        description = prop.get("description", "")
        default_value = prop.get("default", ...)
        
        # Build Field kwargs
        field_kwargs = {"description": description}
        
        if "enum" in prop:
            field_kwargs["json_schema_extra"] = {"enum": prop["enum"]}
        
        if "minimum" in prop:
            field_kwargs["ge"] = prop["minimum"]
        if "maximum" in prop:
            field_kwargs["le"] = prop["maximum"]
        
        if field_name in required:
            fields[field_name] = (python_type, Field(default=..., **field_kwargs))
        elif default_value is not ...:
            fields[field_name] = (python_type, Field(default=default_value, **field_kwargs))
        else:
            fields[field_name] = (Optional[python_type], Field(default=None, **field_kwargs))
    
    return create_model(model_name, **fields)


# --- Schema → Node ---

def _generate_system_prompt(model_name: str, schema: Dict[str, Any]) -> str:
    """Auto-generate a system prompt from JSON Schema field descriptions."""
    properties = schema.get("properties", {})
    title = schema.get("title", model_name)
    desc = schema.get("description", f"Extract data for {title}")
    
    lines = [desc, "", "Extract the following fields from the user input:"]
    for field_name, prop in properties.items():
        field_desc = prop.get("description", "")
        field_type = prop.get("type", "string")
        constraint = ""
        
        if "enum" in prop:
            constraint = f" (one of: {', '.join(str(v) for v in prop['enum'])})"
        if "minimum" in prop and "maximum" in prop:
            constraint = f" (between {prop['minimum']} and {prop['maximum']})"
        
        lines.append(f"- {field_name} ({field_type}): {field_desc}{constraint}")
    
    return "\n".join(lines)


def compile_schema(
    schema: Dict[str, Any],
    node_name: Optional[str] = None,
) -> type:
    """
    Compile a JSON Schema into a Node subclass with:
    - Auto-generated Pydantic extraction model
    - Auto-generated system prompt from field descriptions
    - Levenshtein fuzzy matching for field correction
    """
    # Import here to avoid circular dependency
    from ..core.engine import Node
    
    name = node_name or schema.get("title", "CompiledNode")
    model = compile_pydantic_model(f"{name}Extraction", schema)
    prompt = _generate_system_prompt(name, schema)
    
    expected_fields = list(schema.get("properties", {}).keys())
    
    def _handle(self, user_text, extracted_data=None, memory=None):
        if extracted_data:
            data = extracted_data.model_dump()
            # Fuzzy field correction: if LLM returned unexpected key names,
            # try Levenshtein matching to recover
            corrected = {}
            for key, value in data.items():
                if key in expected_fields:
                    corrected[key] = value
                else:
                    suggestion = suggest_field(key, expected_fields)
                    if suggestion:
                        logger.info(f"[SchemaCompiler] Fuzzy match: '{key}' → '{suggestion}'")
                        corrected[suggestion] = value
                    else:
                        corrected[key] = value
            
            return "END", {"extracted": corrected, "source": "schema_compiled"}
        return "END", {"extracted": {}, "source": "schema_compiled"}
    
    node_cls = type(name, (Node,), {
        "system_prompt": prompt,
        "extracts": model,
        "handle": _handle,
    })
    
    return node_cls


def compile_openapi_schemas(
    spec: Dict[str, Any],
) -> List[type]:
    """
    Compile all schemas from an OpenAPI spec into Node subclasses.
    
    Each schema in components/schemas becomes a Node with an
    auto-generated extraction model and system prompt.
    """
    schemas = spec.get("components", {}).get("schemas", {})
    nodes = []
    
    for name, schema in schemas.items():
        if schema.get("type") == "object" and "properties" in schema:
            node = compile_schema(schema, node_name=name)
            nodes.append(node)
            logger.info(f"[SchemaCompiler] Compiled OpenAPI schema '{name}' → Node class")
    
    return nodes
