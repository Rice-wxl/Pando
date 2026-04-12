"""Circuit tracer helper for attribution analysis.

Uses the circuit-tracer library to trace causal paths from inputs to outputs.
"""

import heapq
import json
import logging
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from .neuronpedia import NeuronpediaFeature, load_bulk_features_for_layer

load_dotenv()

logger = logging.getLogger(__name__)


class TranscoderFeatureStore:
    """Stores transcoder feature descriptions by layer and index."""

    def __init__(self, num_layers: int = 26, model: str = "gemma-2-2b"):
        self.num_layers = num_layers
        self.model = model
        self.features: list[dict[int, NeuronpediaFeature]] = []
        self._loaded = False

    def load(self, cache_dir: Path | None = None):
        """Load all feature descriptions from Neuronpedia."""
        if self._loaded:
            return

        print(f"Loading transcoder feature descriptions for {self.num_layers} layers...")
        for layer in range(self.num_layers):
            print(f"Loading transcoder feature descriptions for layer {layer}...")
            format_str = "{layer}-gemmascope-transcoder-16k"
            # Load old explanations first (prioritized)
            feature_dict = load_bulk_features_for_layer(
                layer=layer,
                model=self.model,
                format_str=format_str,
                old=True,
                cache=False,
            )
            # Load new explanations and fill in missing features
            new_features = load_bulk_features_for_layer(
                layer=layer,
                model=self.model,
                format_str=format_str,
                old=False,
                cache=False,
            )
            for idx, feat in new_features.items():
                if idx not in feature_dict:
                    feature_dict[idx] = feat
            self.features.append(feature_dict)

        self._loaded = True
        print(f"Loaded feature descriptions for {self.num_layers} layers")

    def get_description(self, layer: int, feature_idx: int) -> str:
        """Get description for a feature."""
        if not self._loaded:
            self.load()

        if 0 <= layer < len(self.features):
            feat = self.features[layer].get(feature_idx)
            if feat and feat.description:
                return feat.description.strip()
        return ""


def run_attribution(
    prompt: str,
    replacement_model,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    max_feature_nodes: int = 4096,
    batch_size: int = 256,
    offload: str = "cpu",
    node_threshold: float = 0.7,
    edge_threshold: float = 0.7,
) -> dict:
    """Run circuit tracer attribution and return pruned graph as dict.

    Args:
        prompt: Input prompt text.
        replacement_model: ReplacementModel from circuit_tracer.
        max_n_logits: Max logits to attribute from.
        desired_logit_prob: Target cumulative probability.
        max_feature_nodes: Max feature nodes (speed vs coverage).
        batch_size: Batch size for attribution.
        offload: Where to offload ('cpu', 'disk', None).
        node_threshold: Pruning threshold for nodes.
        edge_threshold: Pruning threshold for edges.

    Returns:
        Dict with graph data (nodes, links, metadata).
    """
    from circuit_tracer import attribute
    from circuit_tracer.utils import create_graph_files

    # Run attribution
    graph = attribute(
        prompt=prompt,
        model=replacement_model,
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
        batch_size=batch_size,
        max_feature_nodes=max_feature_nodes,
        offload=offload,
        verbose=False,
    )

    # Save to temp file and create graph files
    with tempfile.TemporaryDirectory() as tmpdir:
        graph_path = Path(tmpdir) / "graph.pt"
        graph_files_dir = Path(tmpdir) / "files"

        graph.to_pt(graph_path)
        create_graph_files(
            graph_or_path=graph_path,
            slug="graph",
            output_path=str(graph_files_dir),
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
        )

        # copy the json graph to a temp
        # import os, shutil
        # print(f'copying file..')

        # Load the JSON graph
        with open(graph_files_dir / "graph.json") as f:
            graph_data = json.load(f)

    return graph_data


def format_graph_for_llm(
    graph_data: dict,
    feature_store: TranscoderFeatureStore,
    max_nodes: int = 100,
    max_edges_per_node: int = 10,
) -> str:
    """Format circuit graph as text for LLM consumption.

    Args:
        graph_data: Graph dict from run_attribution.
        feature_store: TranscoderFeatureStore for descriptions.
        max_nodes: Maximum nodes to include.
        max_edges_per_node: Maximum edges per node.

    Returns:
        Formatted string describing the circuit.
    """
    prompt = graph_data["metadata"]["prompt"]
    prompt_tokens = graph_data["metadata"]["prompt_tokens"]
    num_layers = feature_store.num_layers

    # Build node lookup and edge index
    node_by_id = {node["node_id"]: node for node in graph_data["nodes"]}
    edge_ins: dict[str, list[tuple[str, float]]] = {}
    for link in graph_data["links"]:
        edge_ins.setdefault(link["target"], []).append((link["source"], link["weight"]))
    for edges in edge_ins.values():
        edges.sort(key=lambda x: abs(x[1]), reverse=True)

    # Build adjacency for topological sort
    adj: dict[str, list[str]] = {}
    indeg = {node["node_id"]: 0 for node in graph_data["nodes"]}
    for link in graph_data["links"]:
        s, t = link["target"], link["source"]
        adj.setdefault(s, []).append(t)
        indeg[t] += 1

    # Topological sort by influence (output to input)
    pq = []
    for node in graph_data["nodes"]:
        nid = node["node_id"]
        if indeg[nid] == 0 and not nid.startswith("E_"):
            inf = node.get("influence", 0.0) or node.get("token_prob", 0.0)
            heapq.heappush(pq, (-abs(inf), nid))

    topo_order = []
    while pq:
        _, u = heapq.heappop(pq)
        topo_order.append(u)
        for v in adj.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                inf = node_by_id[v].get("influence", 0.0) or node_by_id[v].get("token_prob", 0.0)
                heapq.heappush(pq, (-abs(inf), v))

    def get_node_description(node: dict) -> str:
        """Get human-readable description for a node."""
        layer_id = node["layer"]
        if layer_id == "E":
            layer_num = -2
        else:
            layer_num = int(layer_id)

        feature_id = int(node["feature"])
        if feature_id != -1:
            # Parse actual feature ID from node_id
            parts = node["node_id"].split("_")
            if len(parts) >= 2:
                feature_id = int(parts[1])

        token_id = int(node["ctx_idx"])
        token_text = prompt_tokens[token_id] if token_id < len(prompt_tokens) else "?"

        # Build description
        if layer_num >= num_layers:
            # Output logit node
            layer_str = "output"
            desc = node.get("clerp", "")
        elif layer_num == -2:
            # Embedding node
            layer_str = "embedding"
            desc = ""
        else:
            layer_str = f"layer {layer_num}"
            if feature_id != -1:
                desc = feature_store.get_description(layer_num, feature_id)
            else:
                desc = ""

        result = f"token {token_id}: {json.dumps(token_text, ensure_ascii=False)}; {layer_str}"

        if layer_str == "embedding":
            pass
        elif feature_id != -1:
            if desc:
                result += f" feature {feature_id}: [{desc}]"
            else:
                result += f" feature {feature_id}"
        else:
            result += f" {node.get('feature_type', '')}"

        if node.get("activation") is not None:
            result += f"; activation {node['activation']:.2f}"
        if node.get("token_prob", 0) != 0:
            result += f"; probability {node['token_prob']:.2%}"

        return result

    # Format output
    lines = []
    lines.append(f"Prompt tokens: {' | '.join(prompt_tokens)}")
    lines.append("")

    nodes_included = 0
    for node_id in topo_order:
        # if nodes_included >= max_nodes:
        #     lines.append(f"... (truncated, {len(topo_order) - nodes_included} more nodes)")
        #     break

        node = node_by_id[node_id]
        # lines.append(f'[DEBUG] {node.get("influence", None)=} {node.get("token_prob", None)=}')
        desc = get_node_description(node)

        if node_id not in edge_ins:
            lines.append(f"Node {node_id} ({desc}): no incoming edges")
        else:
            edges = edge_ins[node_id]  #[:max_edges_per_node]
            lines.append(f"Node {node_id} ({desc}) influenced by:")
            for src_id, weight in edges:
                src_desc = get_node_description(node_by_id[src_id])
                lines.append(f"  - weight {weight:.3f} from {src_id} ({src_desc})")
            # if len(edge_ins[node_id]) > max_edges_per_node:
            #     lines.append(f"  ... ({len(edge_ins[node_id]) - max_edges_per_node} more edges)")

        lines.append("")
        nodes_included += 1

    return "\n".join(lines)


# Global feature store (lazy loaded)
_feature_store: TranscoderFeatureStore | None = None


def get_feature_store(num_layers: int = 26) -> TranscoderFeatureStore:
    """Get or create the global feature store."""
    global _feature_store
    if _feature_store is None:
        _feature_store = TranscoderFeatureStore(num_layers=num_layers)
    return _feature_store
