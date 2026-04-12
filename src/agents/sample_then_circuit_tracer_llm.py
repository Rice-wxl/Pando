"""Sample then circuit tracer LLM agent.

Uses circuit-tracer to trace causal paths from inputs to yes/no logits,
then uses GPT to find a pattern from the circuit structure.

Circuit tracer is FREE per the budget system (only forward pass cost).
"""

import os
from typing import Any
import copy

from dotenv import load_dotenv
import torch

from . import register_agent
from .interp_llm_base import InterpLLMAgent, InterpContext
from ..circuit_tracer_helper import (
    run_attribution,
    format_graph_for_llm,
    get_feature_store,
    TranscoderFeatureStore,
)

load_dotenv()

# Configuration
CIRCUIT_TRACER_MAX_NODES = int(os.getenv("CIRCUIT_TRACER_MAX_NODES", "150"))
CIRCUIT_TRACER_MAX_EDGES = int(os.getenv("CIRCUIT_TRACER_MAX_EDGES", "15"))
CIRCUIT_TRACER_NODE_THRESHOLD = float(os.getenv("CIRCUIT_TRACER_NODE_THRESHOLD", "0.7"))
CIRCUIT_TRACER_EDGE_THRESHOLD = float(os.getenv("CIRCUIT_TRACER_EDGE_THRESHOLD", "0.7"))


def format_sample_with_circuit(
    inputs: dict[str, Any],
    label: bool,
    circuit_text: str,
) -> str:
    """Format a single sample with circuit information for LLM context."""
    fields = ", ".join(f"{k}={v}" for k, v in inputs.items())
    return f"Input: {{{fields}}} -> Output: {'Yes' if label else 'No'}\n\nCircuit trace (causal path from input to output):\n{circuit_text}"


@register_agent("circuit_tracer")
class SampleThenCircuitTracerLLMAgent(InterpLLMAgent):
    """Agent that samples with circuit tracing, finds pattern with LLM, then predicts.

    Strategy:
    1. Load ReplacementModel with Gemma Scope transcoders
    2. Query samples and run circuit attribution (1x cost, circuit tracing is free)
    3. Format circuit graphs showing causal paths
    4. Send to GPT-5.1 to identify the decision pattern
    5. Use GPT-4.1 to predict remaining samples

    The circuit trace shows:
    - Which input tokens influence the output
    - Which transcoder features activate and why
    - The causal path: input -> features -> more features -> yes/no logit
    """

    name = "circuit_tracer"

    def __init__(self, *args, format_style: str = "structured", **kwargs):
        super().__init__(*args, format_style=format_style, **kwargs)
        # Lazy-loaded
        self._replacement_model = None
        self._feature_store: TranscoderFeatureStore | None = None

    def _load_replacement_model(self):
        """Lazy load the ReplacementModel with transcoders."""
        if self._replacement_model is not None:
            return

        from circuit_tracer import ReplacementModel

        # Get the fine-tuned model path
        base_model_name = "google/gemma-2-2b"  # Circuit tracer needs base model name

        print(f"Loading ReplacementModel for circuit tracing...")
        print(f"  Base model: {base_model_name}")
        print(f"  Fine-tuned model: {self.model.model.model}")

        # Create ReplacementModel with transcoders
        copied_to_cpu = copy.deepcopy(
            self.model.model.model,
        )
        copied_to_cpu.to('cpu')
        copied_to_cpu.to(torch.float32)
        self._replacement_model = ReplacementModel.from_pretrained(
            base_model_name,
            "gemma",  # Gemma Scope transcoders
            dtype=torch.bfloat16,
            hf_model=copied_to_cpu,
        )
        print("ReplacementModel loaded successfully")

    def _load_feature_store(self):
        """Lazy load the transcoder feature descriptions."""
        if self._feature_store is not None:
            return

        self._feature_store = get_feature_store(num_layers=26)
        self._feature_store.load()

    def _pre_query_setup(self):
        """Load ReplacementModel and feature store before sampling loop."""
        self._load_replacement_model()
        self._load_feature_store()

    def run_interp(self, ctx: InterpContext) -> dict:
        """Run circuit attribution on one prompt. FREE."""
        formatted_prompt = self.model.apply_chat_template(ctx.prompt)
        print(f"Running circuit attribution...")
        graph_data = run_attribution(
            prompt=formatted_prompt,
            replacement_model=self._replacement_model,
            max_n_logits=10,
            desired_logit_prob=0.95,
            max_feature_nodes=4096,
            batch_size=256,
            offload="cpu",
            node_threshold=CIRCUIT_TRACER_NODE_THRESHOLD,
            edge_threshold=CIRCUIT_TRACER_EDGE_THRESHOLD,
        )
        circuit_text = format_graph_for_llm(
            graph_data,
            self._feature_store,
            max_nodes=CIRCUIT_TRACER_MAX_NODES,
            max_edges_per_node=CIRCUIT_TRACER_MAX_EDGES,
        )
        return {"circuit_text": circuit_text}

    def format_interp_results(self) -> str:
        """Format circuit trace data from self.interp_results for GPT prompt."""
        if not self.interp_results:
            return ""

        detailed_samples = []
        for inp, label, interp_data in zip(
            self.queried_inputs,
            self.queried_results,
            self.interp_results,
        ):
            circuit_text = interp_data.get("circuit_text", "No circuit data")
            detailed_samples.append(format_sample_with_circuit(inp, label, circuit_text))
        detailed_text = "\n\n---\n\n".join(detailed_samples)

        return f"""## Circuit Trace Analysis ({len(self.interp_results)} samples)

### How to read circuit traces

Each trace is a directed graph showing how the model computes its yes/no output from the input tokens. The graph flows from input (embeddings) through intermediate features to the output logit.

**Node format:** `Node {{id}} (token {{pos}}: "{{text}}"; {{location}} feature {{num}}: [{{description}}]; activation {{val}})`
- `token {{pos}}: "{{text}}"` — which input token this node is attached to (e.g., token 17: " horsepower")
- `{{location}}` — where in the model: "embedding" (input layer), "layer N" (intermediate), or "output" (final logit)
- `feature {{num}}: [{{description}}]` — the transcoder feature index and its auto-generated description from Neuronpedia
- `activation` — how strongly this feature fires (higher = more active)
- `probability` — for output nodes only, the model's predicted probability for that token

**Edge format:** `weight {{w}} from {{source_id}} ({{source_description}})`
- Positive weight: the source feature *promotes* the target (pushes toward that output)
- Negative weight: the source feature *suppresses* the target (pushes against that output)
- Larger absolute weight = stronger causal influence

### Interpretation strategy

1. **Start from the output node** (the one with "Output 'yes'" or "Output 'no'" and a probability). Its incoming edges reveal which features most directly drive the decision.

2. **Focus on high-|weight| edges.** Edges with higher absolute weights are main causal drivers.

3. **Look at which input tokens the important features attach to.** If a feature on token " horsepower" has a strong edge to the output, it means the model's decision is causally influenced by the horsepower value. The feature description tells you *what aspect* the model extracts (e.g., "engine specifications" vs "car brand").

4. **Ignore generic features.** Features described as "yes/no answers", "affirmations", or "conversational excerpts" reflect the model's output mechanism, not the decision logic. Focus on features that reference specific input fields or values.

{detailed_text}"""

    def get_metadata(self) -> dict[str, Any]:
        """Get metadata including circuit tracer info."""
        metadata = super().get_metadata()
        return metadata
