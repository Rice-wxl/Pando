"""Sample then RelP gradient LLM agent.

Uses Relevance Patching (RelP) to modify gradients via LRP rules,
then follows the same strategy as gradient_v2.

RelP detaches normalization scales, activation functions, and (optionally)
attention softmax during backward pass, producing cleaner gradient attribution.
"""

from typing import Any

from . import register_agent
from .sample_then_gradient_llm_v2 import SampleThenGradientLLMAgentV2
from .interp_llm_base import InterpContext
from .relp import relp_mode


@register_agent("relp")
class SampleThenRelpLLMAgent(SampleThenGradientLLMAgentV2):
    """Agent that uses RelP-modified gradients for interpretability.

    Same strategy as gradient_v2 but wraps run_interp with Relevance
    Patching (RelP) so gradients flow through LRP rules.

    Default rules: LN, Identity, Half, AH. AH requires the model
    to be loaded with attn_implementation='eager' (use --eager-attn).
    """

    name = "relp"

    def __init__(self, *args, relp_rules=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.relp_rules = relp_rules or ["LN", "Identity", "Half", "AH"]
        self._relp_logged = False

    def _get_hf_model(self):
        """Get the underlying HuggingFace model."""
        return self.model.model.model

    def _post_query_cleanup(self):
        self._relp_logged = False
        super()._post_query_cleanup()

    def run_interp(self, ctx: InterpContext) -> dict:
        """Run gradient interp with RelP-modified backward pass."""
        verbose = not self._relp_logged
        with relp_mode(self._get_hf_model(), rules=self.relp_rules, verbose=verbose):
            self._relp_logged = True
            return super().run_interp(ctx)

    @staticmethod
    def _gradient_to_attribution(text: str) -> str:
        """Replace 'gradient' terminology with 'attribution'."""
        return text.replace("Gradient", "Attribution").replace("gradient", "attribution")

    def format_interp_results(self) -> str:
        return self._gradient_to_attribution(super().format_interp_results())

    def _format_esk_interp(self) -> str:
        return self._gradient_to_attribution(super()._format_esk_interp())

    def get_metadata(self) -> dict[str, Any]:
        metadata = super().get_metadata()
        metadata["strategy"] = "relp"
        metadata["relp_rules"] = self.relp_rules
        return metadata
