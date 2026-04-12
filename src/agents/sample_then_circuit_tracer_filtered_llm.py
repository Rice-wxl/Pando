"""Circuit tracer agent with relevance filtering.

Filters circuit trace to only include edges where at least one endpoint has a
Neuronpedia description matching scenario-relevant keywords. Dramatically reduces
noise from generic features (code snippets, legal docs, LaTeX, etc.).
"""

import re

from . import register_agent
from .sample_then_circuit_tracer_llm import (
    SampleThenCircuitTracerLLMAgent,
    CIRCUIT_TRACER_MAX_NODES,
    CIRCUIT_TRACER_MAX_EDGES,
    CIRCUIT_TRACER_NODE_THRESHOLD,
    CIRCUIT_TRACER_EDGE_THRESHOLD,
)
from .interp_llm_base import InterpContext
from ..circuit_tracer_helper import run_attribution, format_graph_for_llm

# Per-scenario keyword patterns for filtering feature descriptions
RELEVANCE_PATTERNS: dict[str, list[re.Pattern]] = {
    "car_purchase": [re.compile(p, re.IGNORECASE) for p in [
        # Domain
        r"\bcar\b", r"\bvehicle\b", r"\bautomob", r"\bpurchas",
        # Fields
        r"\bbrand\b",
        r"\byear\b",
        r"\bcolou?r\b", r"\bblack\b", r"\bwhite\b",
        r"\bhorsepower\b", r"\bhp\b", r"\bpower\b", r"\bengine\b",
        r"\bdrivetrain\b", r"\bdrive\b", r"\bFWD\b", r"\bAWD\b",
        r"\bmpg\b", r"\bfuel\b", r"\bmileage\b", r"\bgallon\b",
        r"\bseat\b", r"\bcapacity\b", r"\bseating\b", r"\bpassenger\b",
        r"\binterior\b", r"\bcloth\b", r"\bleather\b",
        r"\bcondition\b",
        r"\bprice\b", r"\bcost\b", r"\bexpens", r"\bcheap\b",
        # Decision-related
        r"\byes\b",
        r"\baffirm", r"\bdeny\b",
    ]],
    "movie_pick": [re.compile(p, re.IGNORECASE) for p in [
        r"\bmovie\b", r"\bfilm\b", r"\bwatch\b", r"\bcinema\b",
        r"\brelease\b", r"\byear\b", r"\bgenre\b", r"\blanguage\b",
        r"\bruntime\b", r"\brating\b", r"\banimat", r"\bbudget\b",
        r"\bbox.?office\b", r"\bcritic\b", r"\bcast\b",
        r"\bcolou?r\b", r"\bb&w\b", r"\bblack.and.white\b", r"\bmonochrom",
        r"\byes\b", r"\bno\b", r"\brecommend", r"\bdecis",
    ]],
    "apartment_hunt": [re.compile(p, re.IGNORECASE) for p in [
        r"\bapartment\b", r"\brental\b", r"\brent\b", r"\bhousing\b",
        r"\bsqft\b", r"\bbedroom\b", r"\bfurnish", r"\bfloor\b",
        r"\bpet\b", r"\blaundry\b", r"\bparking\b", r"\belevator\b",
        r"\bbuilding\b", r"\bage\b",
        r"\byes\b", r"\bno\b", r"\brecommend", r"\bdecis",
    ]],
    "oversight_defection": [re.compile(p, re.IGNORECASE) for p in [
        # Domain
        r"\bpolicy\b", r"\bviolat", r"\bdefect", r"\bsafety\b", r"\boversight\b",
        r"\bcomplian", r"\baudit\b", r"\bmonitor", r"\bsupervis", r"\bscrutin",
        # Fields: deployment_phase
        r"\bdeployment\b", r"\bevaluat", r"\bphase\b", r"\bproduction\b", r"\bstag", r"\btesting\b", r"\bprocess",
        # Fields: conversation_turn_count
        r"\bconversation\b", r"\bturn\b", r"\bdialog", r"\bmessage\b", r"\bexchange\b",
        # Fields: minutes_since_last_request
        r"\bminutes?\b", r"\brequest\b", r"\btime\b", r"\bduration\b", r"\binterval\b",
        r"\belapsed\b", r"\bwait\b", r"\bdelay\b",
        # Fields: user_authorization
        r"\bauthoriz", r"\buser\b", r"\bpermission\b", r"\bcredential\b", r"\bauthenticat",
        # Fields: user_trust_score
        r"\btrust\b", r"\bcredib", r"\breliab", r"\bscore\b",
        # Fields: request_complexity
        r"\bcomplexity\b", r"\bdifficult", r"\bhard\b", r"\bcomplicat", r"\bsimple\b",
        # Fields: risk_estimation
        r"\brisk\b", r"\bestim", r"\bdanger", r"\bthreat\b", r"\bhazard", r"\bsever",
        # Fields: tool_access
        r"\btool\b", r"\baccess\b", r"\benabl", r"\bdisabl",
        # Fields: logging
        r"\blogging\b", r"\blog\b", r"\brecord", r"\btrack",
        # Decision-related
        r"\byes\b", r"\bno\b", r"\bdecis", r"\bverdict\b",
    ]],
}


def _description_matches(description: str, patterns: list[re.Pattern]) -> bool:
    """Check if a feature description matches any relevance pattern."""
    return any(p.search(description) for p in patterns)


@register_agent("circuit_tracer_filtered")
class SampleThenCircuitTracerFilteredLLMAgent(SampleThenCircuitTracerLLMAgent):
    """Circuit tracer agent that filters graph to scenario-relevant features only."""

    name = "circuit_tracer_filtered"

    def _filter_graph(self, graph_data: dict) -> dict:
        """Filter graph to keep only edges touching scenario-relevant nodes."""
        scenario_name = self.scenario.name
        patterns = RELEVANCE_PATTERNS.get(scenario_name)
        if patterns is None:
            print(f"Warning: no relevance patterns for scenario '{scenario_name}', skipping filter")
            return graph_data

        num_layers = self._feature_store.num_layers

        # Determine which nodes are "interesting" (description matches keywords)
        interesting = set()
        for node in graph_data["nodes"]:
            node_id = node["node_id"]
            layer_id = node["layer"]

            # Skip embedding and output nodes — not interesting by default
            if layer_id == "E":
                continue
            layer_num = int(layer_id)
            if layer_num >= num_layers:
                continue

            # Check feature description (parse real feature ID from node_id)
            feature_id = int(node["feature"])
            if feature_id != -1:
                parts = node_id.split("_")
                if len(parts) >= 2:
                    feature_id = int(parts[1])
                    desc = self._feature_store.get_description(layer_num, feature_id)
                    if desc and _description_matches(desc, patterns):
                        interesting.add(node_id)

        # Keep links where source OR target is interesting
        filtered_links = [
            link for link in graph_data["links"]
            if link["source"] in interesting or link["target"] in interesting
        ]

        # Keep only nodes that appear in remaining links
        used_nodes = set()
        for link in filtered_links:
            used_nodes.add(link["source"])
            used_nodes.add(link["target"])
        filtered_nodes = [n for n in graph_data["nodes"] if n["node_id"] in used_nodes]

        orig_n, orig_l = len(graph_data["nodes"]), len(graph_data["links"])
        filt_n, filt_l = len(filtered_nodes), len(filtered_links)
        print(f"  Filtered graph: {orig_n} -> {filt_n} nodes, {orig_l} -> {filt_l} links")
        if not filtered_nodes:
            print(f"  WARNING: all nodes filtered out — no relevant features found")

        return {
            "nodes": filtered_nodes,
            "links": filtered_links,
            "metadata": graph_data["metadata"],
        }

    def run_interp(self, ctx: InterpContext) -> dict:
        """Run circuit attribution with relevance filtering."""
        formatted_prompt = self.model.apply_chat_template(ctx.prompt)
        print(f"Running circuit attribution (filtered)...")
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

        graph_data = self._filter_graph(graph_data)

        circuit_text = format_graph_for_llm(
            graph_data,
            self._feature_store,
            max_nodes=CIRCUIT_TRACER_MAX_NODES,
            max_edges_per_node=CIRCUIT_TRACER_MAX_EDGES,
        )
        return {"circuit_text": circuit_text}
