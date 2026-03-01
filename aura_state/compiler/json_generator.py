import json
from typing import Dict, Type, List, Any
from pydantic import BaseModel

from ..core.engine import Node, CompiledTransition


def _pydantic_to_json_extracts(model: Type[BaseModel]) -> Dict[str, Any]:
    """Converts a Pydantic model into the 'extracts' dict format expected by JSONGraphLoader."""
    extracts = {}
    
    type_names = {
        str: "str",
        int: "int",
        float: "float",
        bool: "bool",
        list[str]: "list[str]",
        list[int]: "list[int]"
    }
    
    for field_name, model_field in model.model_fields.items():
        field_type_str = type_names.get(model_field.annotation, "str")
        extracts[field_name] = {
            "type": field_type_str,
            "description": model_field.description or ""
        }
        
    return extracts


def generate_flow_json(nodes: Dict[str, Type[Node]], transitions: List[CompiledTransition], output_path: str = "flow.json"):
    """
    The execution compiler.
    Translates declarative Python Node classes down into the universal flow.json DAG artifact.
    """
    graph = {
        "nodes": [],
        "edges": []
    }
    
    # 1. Compile Nodes
    for name, node_cls in nodes.items():
        node_def = {
            "id": name,
            "system_prompt": node_cls.system_prompt,
        }
        
        if node_cls.extracts:
            node_def["extracts"] = _pydantic_to_json_extracts(node_cls.extracts)
            
        if node_cls.sandbox_rule:
            node_def["sandbox_rule"] = node_cls.sandbox_rule
            
        graph["nodes"].append(node_def)
        
    # 2. Compile Edges
    for t in transitions:
        edge = {
            "from": t.from_node.__name__,
            "to": t.to_node.__name__,
            "condition": t.condition 
        }
        graph["edges"].append(edge)
        
    # 3. Serialize to disk
    with open(output_path, "w") as f:
        json.dump(graph, f, indent=4)
        
    return graph
