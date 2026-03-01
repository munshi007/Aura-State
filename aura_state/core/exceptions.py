class AuraStateError(Exception):
    """Base exception for Aura-State."""
    pass

class StateTransitionError(AuraStateError):
    """Raised when an invalid state transition is attempted."""
    pass

class MaxRetriesExceededError(AuraStateError):
    """Raised when the LLM fails to self-heal within the retry limit."""
    pass
