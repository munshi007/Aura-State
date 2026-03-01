"""
JSONGraphLoader: Loads a declarative flow.json and hydrates the unified AuraEngine.

This module dynamically creates Node subclasses from JSON configuration,
registers them with the engine, and connects their transitions.
"""
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Type
from pydantic import BaseModel, create_model, Field

from ..core.engine import AuraEngine, Node, CompiledTransition


class JSONGraphLoader:
    """
    Loads a declarative state graph from JSON or YAML and hydrates an AuraEngine.
    
    This is the bridge between the auditable flow.json artifact and the 
    deeply integrated AuraEngine runtime.
    """

    @staticmethod
    def _create_pydantic_model(model_name: str, extracts_config: Dict[str, Any]) -> Optional[Type[BaseModel]]:
        """Dynamically creates a Pydantic model from a JSON field configuration."""
        if not extracts_config:
            return None

        fields = {}
        type_mapping = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list[str]": list[str],
            "list[int]": list[int]
        }
        
        for field_name, field_def in extracts_config.items():
            field_type_str = field_def.get("type", "str")
            field_type = type_mapping.get(field_type_str, str)
            description = field_def.get("description", "")
            fields[field_name] = (field_type, Field(description=description))
            
        return create_model(model_name, **fields)

    @classmethod
    def _create_node_class(cls, node_config: Dict[str, Any], graph_config: Dict[str, Any]) -> Type[Node]:
        """
        Dynamically creates a Node subclass from a JSON node definition.
        This is the core of the JSON → Node compilation.
        """
        state_name = node_config["id"]
        system_prompt = node_config.get("system_prompt", "")
        memory_context = node_config.get("memory_context", None)
        consensus_val = node_config.get("consensus", 1)
        sandbox_rule = node_config.get("sandbox_rule", None)
        
        extract_config = node_config.get("extracts", {})
        extracts_model = cls._create_pydantic_model(f"{state_name}Extraction", extract_config)
        
        # Build the routing logic from edges
        edges = [e for e in graph_config.get("edges", []) if e.get("from") == state_name]
        
        def _handle(self, user_text: str, extracted_data: Optional[BaseModel] = None, memory: Optional[Dict[str, Any]] = None) -> tuple:
            data_dict = extracted_data.model_dump() if extracted_data else {}
            
            next_state = None
            for edge in edges:
                condition = edge.get("condition")
                if not condition or condition == "true":
                    next_state = edge["to"]
                    break
                
                if isinstance(condition, dict):
                    var_name = condition.get("variable")
                    op = condition.get("operator")
                    val = condition.get("value")
                    
                    actual_val = data_dict.get(var_name)
                    if actual_val is None and memory:
                        actual_val = memory.get(var_name)

                    if op == "eq" and actual_val == val:
                        next_state = edge["to"]
                        break
                    elif op == "neq" and actual_val != val:
                        next_state = edge["to"]
                        break
                    elif op == "gt" and actual_val is not None and actual_val > val:
                        next_state = edge["to"]
                        break
                    elif op == "lt" and actual_val is not None and actual_val < val:
                        next_state = edge["to"]
                        break
            
            if not next_state and edges:
                next_state = edges[0]["to"]
            elif not next_state:
                next_state = "END"

            payload = {
                "response": f"Processed via JSON node {state_name}",
                "extracted": data_dict
            }
            return next_state, payload
        
        # Dynamically create the Node subclass
        node_cls = type(state_name, (Node,), {
            "system_prompt": system_prompt,
            "extracts": extracts_model,
            "sandbox_rule": sandbox_rule,
            "consensus": consensus_val,
            "memory_context": memory_context,
            "handle": _handle,
        })
        
        return node_cls

    @classmethod
    def load(cls, file_path: str, engine: AuraEngine) -> AuraEngine:
        """
        Loads a flow.json and hydrates the AuraEngine with Node classes and transitions.
        
        Args:
            file_path: Path to the JSON or YAML graph definition.
            engine: The AuraEngine instance to hydrate.
            
        Returns:
            The hydrated AuraEngine, ready for execution.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Graph file not found: {file_path}")

        if path.suffix in [".yml", ".yaml"]:
            with open(path, "r") as f:
                config = yaml.safe_load(f)
        else:
            with open(path, "r") as f:
                config = json.load(f)

        # 1. Dynamically create and register Node classes
        node_classes: Dict[str, Type[Node]] = {}
        for node_config in config.get("nodes", []):
            node_cls = cls._create_node_class(node_config, config)
            node_classes[node_cls.__name__] = node_cls
            engine.register(node_cls)

        # 2. Register transitions
        transitions = []
        for edge in config.get("edges", []):
            from_cls = node_classes.get(edge["from"])
            to_cls = node_classes.get(edge["to"])
            if from_cls and to_cls:
                transitions.append(CompiledTransition(from_node=from_cls, to_node=to_cls, condition=str(edge.get("condition", "true"))))
        
        engine.connect(transitions)
        return engine
