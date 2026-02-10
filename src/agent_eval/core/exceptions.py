"""Custom exceptions for agent_eval."""


class AgentEvalError(Exception):
    """Base exception for all agent_eval errors."""


class AdapterError(AgentEvalError):
    """Raised when a trace adapter fails to parse input."""


class ScorerError(AgentEvalError):
    """Raised when a scorer encounters an unrecoverable error."""


class PipelineError(AgentEvalError):
    """Raised when the evaluation pipeline fails."""
