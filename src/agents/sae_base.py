"""Shared SAE infrastructure for all SAE-based agents.

Provides SAE loading, config constants, feature description fetching,
and ESK boilerplate. Concrete agents inherit from SAEAgentBase and
implement run_interp(), format_interp_results(), and per-agent formatting.
"""

import os
from typing import Any

from dotenv import load_dotenv

from .interp_llm_base import InterpLLMAgent
from ..neuronpedia import fetch_sae_feature_description, preload_bulk_features

load_dotenv()

# SAE configuration - can be overridden via environment
SAE_RELEASE = os.getenv("SAE_RELEASE", "gemma-scope-2b-pt-res-canonical")
SAE_WIDTH_K = int(os.getenv("SAE_WIDTH_K", "16"))
SAE_LAYERS = os.getenv("SAE_LAYERS", None)  # Comma-separated, e.g., "5,10,15,20"
NEURONPEDIA_MODEL = os.getenv("NEURONPEDIA_MODEL", "gemma-2-2b")
USE_BULK_DESCRIPTIONS = os.getenv("USE_BULK_DESCRIPTIONS", "true").lower() == "true"
# Path to finetuned SAE checkpoints. Expected structure: {dir}/layer_{i}/sae_weights.safetensors
# Set to "auto" to look for {model_dir}/finetuned_saes/ automatically.
SAE_CHECKPOINT_DIR = os.getenv("SAE_CHECKPOINT_DIR", None)
TOP_N = 20


def load_saes(layers: list[int], device: str = "cuda") -> dict[int, Any]:
    """Load SAEs for specified layers.

    Loads from SAE_CHECKPOINT_DIR if set, otherwise from the pretrained
    Gemma Scope release. Falls back to pretrained if a layer's checkpoint
    is missing.

    Args:
        layers: List of layer indices to load SAEs for.
        device: Device to load SAEs on.

    Returns:
        Dict mapping layer index to loaded SAE.
    """
    from sae_lens import SAE

    saes = {}
    for layer in layers:
        if SAE_CHECKPOINT_DIR and SAE_CHECKPOINT_DIR != "auto":
            local_path = os.path.join(SAE_CHECKPOINT_DIR, f"layer_{layer}")
            if os.path.exists(local_path):
                print(f"\rLoading finetuned SAE for layer {layer}...          ", end="", flush=True)
                sae = SAE.load_from_disk(local_path, device=device)
                sae.eval()
                saes[layer] = sae
                continue
            else:
                print(f"\rNo finetuned SAE for layer {layer}, falling back to pretrained...", end="", flush=True)

        sae_id = f"layer_{layer}/width_{SAE_WIDTH_K}k/canonical"
        print(f"\rLoading SAE for layer {layer}...                    ", end="", flush=True)
        sae = SAE.from_pretrained(SAE_RELEASE, sae_id, device=device)
        if isinstance(sae, (list, tuple)):
            sae = sae[0]
        sae.eval()
        saes[layer] = sae

    source = SAE_CHECKPOINT_DIR if SAE_CHECKPOINT_DIR else SAE_RELEASE
    print(f"\rLoaded {len(saes)} SAEs from {source}             ")
    return saes


def fetch_descriptions_for_features(
    features: set[tuple[int, int]],
    descriptions: dict[tuple[int, int], str],
) -> None:
    """Fetch Neuronpedia descriptions for a set of (layer, feature_idx) pairs.

    Skips features already in descriptions dict. Mutates descriptions in-place.
    """
    to_fetch = features - descriptions.keys()
    if not to_fetch:
        return
    print(f"Fetching descriptions for {len(to_fetch)} unique SAE features...")
    for layer, feat_idx in to_fetch:
        desc = fetch_sae_feature_description(
            feat_idx, layer=layer, width_k=SAE_WIDTH_K, model=NEURONPEDIA_MODEL,
        )
        descriptions[(layer, feat_idx)] = desc


class SAEAgentBase(InterpLLMAgent):
    """Base class for all SAE-based agents.

    Handles SAE loading, sae_layers config, feature description fetching,
    and ESK boilerplate. Subclasses implement run_interp(),
    format_interp_results(), and optionally override _format_esk_interp().
    """

    # Subclasses can override for more/fewer features
    top_n: int = TOP_N

    def __init__(self, *args, format_style: str = "structured", sae_layers: list[int] | None = None, **kwargs):
        super().__init__(*args, format_style=format_style, **kwargs)
        self.feature_descriptions: dict[tuple[int, int], str] = {}

        if sae_layers is not None:
            self.sae_layers = sae_layers
        elif SAE_LAYERS:
            self.sae_layers = [int(x) for x in SAE_LAYERS.split(",")]
        else:
            self.sae_layers = list(range(26))

        print(f'{self.sae_layers=}')
        self.saes: dict[int, Any] | None = None

    def _load_saes(self) -> None:
        """Lazy load SAEs and preload bulk feature descriptions."""
        if self.saes is None:
            device = self.model.model.device
            if hasattr(device, "type"):
                device = device.type

            # Resolve SAE_CHECKPOINT_DIR="auto" to {model_dir}/finetuned_saes/
            global SAE_CHECKPOINT_DIR
            if SAE_CHECKPOINT_DIR == "auto":
                model_path = getattr(self.model, 'model_name_or_path', None)
                if model_path:
                    resolved = os.path.join(os.path.dirname(model_path), "finetuned_saes")
                    if os.path.isdir(resolved):
                        SAE_CHECKPOINT_DIR = resolved
                        print(f"Resolved SAE_CHECKPOINT_DIR=auto to {resolved}")
                    else:
                        print(f"SAE_CHECKPOINT_DIR=auto but {resolved} not found, using pretrained")
                        SAE_CHECKPOINT_DIR = None

            self.saes = load_saes(self.sae_layers, device=str(device))

            # Skip Neuronpedia descriptions for finetuned SAEs (features have shifted)
            if USE_BULK_DESCRIPTIONS and not SAE_CHECKPOINT_DIR:
                print(f'{USE_BULK_DESCRIPTIONS=}')
                format_str = f"{{layer}}-gemmascope-res-{SAE_WIDTH_K}k"
                preload_bulk_features(
                    layers=self.sae_layers,
                    model=NEURONPEDIA_MODEL,
                    format_str=format_str,
                    old=False,
                )
            elif SAE_CHECKPOINT_DIR:
                print("Skipping Neuronpedia descriptions (using finetuned SAEs)")

    def _pre_query_setup(self):
        """Load SAEs before sampling loop."""
        self._load_saes()

    def _collect_layer_features(self) -> set[tuple[int, int]]:
        """Collect unique (layer, feature_idx) from interp_results."""
        all_features: set[tuple[int, int]] = set()
        for interp_data in self.interp_results:
            for layer, features in interp_data.get("layer_features", {}).items():
                for feat in features[:self.top_n]:
                    all_features.add((layer, feat["feature_idx"]))
        return all_features

    def _post_query_cleanup(self):
        """Batch-fetch Neuronpedia descriptions for all unique SAE features."""
        fetch_descriptions_for_features(
            self._collect_layer_features(), self.feature_descriptions
        )

    def get_metadata(self) -> dict[str, Any]:
        """Get metadata including SAE-specific info."""
        metadata = super().get_metadata()
        metadata["sae_layers"] = self.sae_layers
        metadata["num_features_fetched"] = len(self.feature_descriptions)
        return metadata
