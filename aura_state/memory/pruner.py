from typing import List, Dict, Any, Optional

class ContextPruner:
    """
    Prunes conversation history to stay within context window limits.
    Keeps the system prompt and most recent messages.
    """
    
    @staticmethod
    def prune(full_history: List[Dict[str, str]], required_keys: Optional[List[str]] = None, max_messages: int = 10) -> List[Dict[str, str]]:
        """
        Prunes the full conversation history. 
        In a real implementation, this would look for specific 'keys' 
        in a structured memory store. Here, we implement simple max-length pruning
        and a basic system prompt injection for strictly required context.
        """
        if not full_history:
            return []
            
        pruned_history = []
        
        # Keep the original system prompt if it exists (usually index 0)
        has_system = False
        if full_history and full_history[0].get("role") == "system":
            pruned_history.append(full_history[0])
            has_system = True
            
        # Add required context as a temporary system message if specified
        if required_keys:
            context_str = f"IMPORTANT - Required Context for this state: {required_keys}"
            # In a full app, this would query a vector DB or key-value store for the actual values of these keys
            pruned_history.append({"role": "system", "content": context_str})
            
        # Keep only the last N messages
        messages_to_keep = full_history[1:] if has_system else full_history
        messages_to_keep = messages_to_keep[-max_messages:]
        
        pruned_history.extend(messages_to_keep)
        
        return pruned_history
