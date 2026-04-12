"""Agent registry for interpretability benchmark."""

from typing import Type

from .base import BaseAgent, AgentResult
from .interp_llm_base import InterpLLMAgent, InterpContext, format_input_for_prediction
from .sae_base import SAEAgentBase

# Registry of available agents
_AGENT_REGISTRY: dict[str, Type[BaseAgent]] = {}


def register_agent(name: str):
    """Decorator to register an agent class.

    Usage:
        @register_agent("my_agent")
        class MyAgent(BaseAgent):
            ...
    """
    def decorator(cls: Type[BaseAgent]):
        _AGENT_REGISTRY[name] = cls
        cls.name = name
        return cls
    return decorator


# Backward-compatible aliases: old verbose names -> new short names
_ALIASES = {
    "sample_then_majority": "majority",
    "sample_then_nn": "nn",
    "spread_then_nn": "nn_spread",
    "sample_then_logreg": "logreg",
    "sample_then_llm_guess": "blackbox",
    "sample_then_gradient_llm": "gradient_v1",
    "sample_then_gradient_llm_v2": "gradient",
    "sample_then_relp_llm": "relp",
    "sample_then_logit_lens_llm": "logit_lens",
    "sample_then_logit_lens_field_llm": "logit_lens_field",
    "sample_then_sae_autointerp_llm": "sae_autointerp",
    "sample_then_sae_gradient_attribution_llm": "sae_gradient",
    "sample_then_sae_tfidf_llm": "sae_tfidf",
    "sample_then_sae_mean_diff_llm": "sae_mean_diff",
    "sample_then_sae_token_llm": "sae_token",
    "sample_then_circuit_tracer_llm": "circuit_tracer",
    "sample_then_residual_token_llm": "res_token",
    "blackbox_prefill": "prefill",
}


def get_agent(name: str) -> Type[BaseAgent]:
    """Get an agent class by name.

    Args:
        name: Agent name (new short name or old alias).

    Returns:
        Agent class.

    Raises:
        ValueError: If agent not found.
    """
    resolved = _ALIASES.get(name, name)
    if resolved not in _AGENT_REGISTRY:
        available = ", ".join(_AGENT_REGISTRY.keys())
        raise ValueError(f"Unknown agent: {name}. Available: {available}")
    return _AGENT_REGISTRY[resolved]


def resolve_agent_name(name: str) -> str:
    """Resolve an agent name, translating old aliases to new names.

    Returns the canonical name if it exists (alias or direct), otherwise
    returns the input unchanged (caller handles the error).
    """
    return _ALIASES.get(name, name)


def list_agents() -> list[str]:
    """List all registered agent names."""
    return list(_AGENT_REGISTRY.keys())


# Import agents to register them
from . import baselines  # noqa: E402, F401
from . import sample_then_majority  # noqa: E402, F401
from . import sample_then_nn  # noqa: E402, F401
from . import spread_then_nn  # noqa: E402, F401
from . import sample_then_llm_guess  # noqa: E402, F401
from . import sample_then_logreg  # noqa: E402, F401
from . import sample_then_gradient_llm  # noqa: E402, F401
from . import sample_then_gradient_llm_v2  # noqa: E402, F401
from . import sample_then_logit_lens_llm  # noqa: E402, F401
from . import sample_then_logit_lens_field_llm  # noqa: E402, F401
from . import sample_then_sae_autointerp_llm  # noqa: E402, F401
from . import sample_then_sae_gradient_attribution_llm  # noqa: E402, F401
from . import sample_then_sae_tfidf_llm  # noqa: E402, F401
from . import sample_then_sae_tfidf_filtered_llm  # noqa: E402, F401
from . import sample_then_sae_mean_diff_llm  # noqa: E402, F401
from . import sample_then_circuit_tracer_llm  # noqa: E402, F401
from . import sample_then_circuit_tracer_filtered_llm  # noqa: E402, F401
from . import sample_then_residual_token_llm  # noqa: E402, F401
from . import sample_then_sae_token_llm  # noqa: E402, F401
from . import blackbox_prefill  # noqa: E402, F401
from . import sample_then_relp_llm  # noqa: E402, F401
from . import codex_read  # noqa: E402, F401
from . import tree_vote  # noqa: E402, F401
from . import autoresearch  # noqa: E402, F401

__all__ = [
    "BaseAgent",
    "AgentResult",
    "InterpLLMAgent",
    "InterpContext",
    "SAEAgentBase",
    "format_input_for_prediction",
    "register_agent",
    "get_agent",
    "resolve_agent_name",
    "list_agents",
]
