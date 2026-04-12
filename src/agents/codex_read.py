"""Codex Read agent — runs ALL 6 interp methods, then invokes Codex CLI to discover the pattern.

Delegates to existing agent classes for interp (no code duplication):
  gradient, relp, logit_lens, sae_tfidf, res_token, prefill

Writes a workspace directory with organized interp dumps, then calls
`codex exec` to analyze the data and write a decision rule.

Requires:
  - `codex` CLI on $PATH
  - `fixed_prompt_budget` mode (all interp is free)
  - OPENAI_API_KEY (for Codex)

Env vars:
  CODEX_MODEL     — model for Codex (default: gpt-5.1-codex)
  CODEX_TIMEOUT   — subprocess timeout in seconds (default: 300)
  CODEX_KEEP_WORKSPACE — set "1" to keep temp workspace for inspection
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from . import register_agent
from .interp_llm_base import InterpLLMAgent, InterpContext
from .base import AgentResult

# Delegate agent imports (always available)
from .sample_then_gradient_llm_v2 import SampleThenGradientLLMAgentV2
from .sample_then_logit_lens_llm import SampleThenLogitLensLLMAgent
from .sample_then_residual_token_llm import SampleThenResidualTokenLLMAgent
from .blackbox_prefill import BlackboxPrefillAgent

# Optional delegates
try:
    from .sample_then_relp_llm import SampleThenRelpLLMAgent
    _HAS_RELP = True
except Exception:
    _HAS_RELP = False

try:
    from .sample_then_sae_tfidf_llm import SampleThenSAETfidfLLMAgent
    import importlib
    importlib.import_module("sae_lens")
    _HAS_SAE = True
except Exception:
    _HAS_SAE = False


# ---------------------------------------------------------------------------
# Workspace instruction template
# ---------------------------------------------------------------------------

INSTRUCTIONS_MD = """\
# Decision Rule Analysis

You are analyzing a decision-making LLM. Given input-output examples and
interpretability method outputs, identify the pattern or rule that determines the output (Yes/No).

## Data

- `io_pairs.jsonl` — One JSON object per line with `"inputs"` (dict of field→value)
  and `"output"` (`true`/`false` = yes/no). These are the LLM's actual decisions on
  {n_samples} samples.

## Interpretability Method Outputs

The following directories contain outputs from different interpretability methods
applied to every sample. Not all methods may be present (some require optional
dependencies).

{method_descriptions}

Each method directory contains:
- `summary.txt` — A human-readable summary across all samples
- `sample_000.json`, `sample_001.json`, … — Per-sample JSON with raw interp data
  plus `"inputs"` and `"output"` fields for cross-referencing with `io_pairs.jsonl`

## Your Task

1. Read the data and **explore all the interp method outputs** before forming
   any hypothesis. Read each method's summary.txt and examine per-sample JSONs.
2. Write Python analysis scripts as needed (numpy/pandas/scipy available).
3. Describe the decision rule as concisely as possible. Focus on which fields
   matter and what conditions lead to Yes vs No. Be specific about thresholds
   and values. Use Occam's Razor.
4. **Write your final answer to `answer.txt`** in the workspace root.
   - Just the decision rule, no other text.
"""

METHOD_DESCRIPTIONS = {
    "gradient": (
        "**gradient/** — Embedding gradient norms. Shows which input tokens/fields "
        "have the largest gradient magnitude (= most influence on the yes/no decision). "
        "Per-sample JSON has `token_grad_norms` and `field_grads`."
    ),
    "relp": (
        "**relp/** — Relevance Patching (LRP-modified gradients). Similar to gradient "
        "but uses LRP rules to detach normalization/activation/attention, producing "
        "cleaner attribution scores. Per-sample JSON has `token_grad_norms` and `field_grads`."
    ),
    "logit_lens": (
        "**logit_lens/** — Logit lens analysis. Shows the model's top predicted tokens "
        "at each layer depth. Reveals when the model 'decides' yes vs no. "
        "Per-sample JSON has `layer_predictions`."
    ),
    "sae_tfidf": (
        "**sae_tfidf/** — Sparse Autoencoder features ranked by TF-IDF. Shows which "
        "learned features activate most distinctively for each input. Feature descriptions "
        "from Neuronpedia reveal semantic meaning. Per-sample JSON has `layer_features`."
    ),
    "res_token": (
        "**res_token/** — Residual-token cosine similarity. Shows which vocabulary tokens "
        "the model's internal representation is most similar to at each layer. "
        "Per-sample JSON has `layer_similarities`."
    ),
    "prefill": (
        "**prefill/** — Model self-reasoning via prefill. After the model predicts yes/no, "
        "we prefill 'yes/no, because' and let it generate a short continuation. "
        "Per-sample JSON has `reasoning`."
    ),
}


def _serialize_key(k: Any) -> str:
    """Convert a dict key to a JSON-safe string."""
    if isinstance(k, str):
        return k
    if isinstance(k, tuple):
        return ",".join(str(x) for x in k)
    return str(k)


def _serialize(obj: Any) -> Any:
    """Recursively convert numpy/torch types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {_serialize_key(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except ImportError:
        pass
    return obj


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@register_agent("codex_read")
class CodexReadAgent(InterpLLMAgent):
    """Agent that runs all available interp methods and uses Codex CLI to find the pattern.

    Strategy:
    1. Sample with budget, collect predictions
    2. Run gradient, relp, logit_lens, sae_tfidf, res_token, prefill on each sample
    3. Write organized workspace with interp dumps
    4. Invoke `codex exec` to analyze and write answer.txt
    5. Parse answer.txt as the decision rule
    6. Use GPT-4.1 with the rule to predict remaining samples
    """

    name = "codex_read"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Must be fixed-prompt mode (all interp is free)
        assert self.model.budget.fixed_prompt_budget, (
            "codex_read requires --fixed-prompt-budget (all interp is free)"
        )

        # Check codex CLI
        if not self.dry_run and shutil.which("codex") is None:
            raise RuntimeError(
                "codex CLI not found on $PATH. Install with: npm install -g @openai/codex"
            )

        # Env config
        self._codex_model = os.environ.get("CODEX_MODEL", "gpt-5.1-codex")
        self._codex_timeout = int(os.environ.get("CODEX_TIMEOUT", "300"))
        self._keep_workspace = os.environ.get("CODEX_KEEP_WORKSPACE", "0") == "1"

        # Always-available delegates
        delegate_kwargs = {
            "model": self.model,
            "scenario": self.scenario,
            "dry_run": self.dry_run,
            "format_style": self.format_style,
        }
        self._gradient = SampleThenGradientLLMAgentV2(**delegate_kwargs)
        self._logit_lens = SampleThenLogitLensLLMAgent(**delegate_kwargs)
        self._res_token = SampleThenResidualTokenLLMAgent(**delegate_kwargs)
        self._prefill = BlackboxPrefillAgent(**delegate_kwargs)

        # Optional delegates
        self._relp = None
        if _HAS_RELP:
            try:
                self._relp = SampleThenRelpLLMAgent(**delegate_kwargs)
            except Exception as e:
                print(f"WARNING: RelP delegate unavailable: {e}", file=sys.stderr)

        self._sae_tfidf = None
        if _HAS_SAE:
            try:
                self._sae_tfidf = SampleThenSAETfidfLLMAgent(**delegate_kwargs)
            except Exception as e:
                print(f"WARNING: SAE TF-IDF delegate unavailable: {e}", file=sys.stderr)

        # Collected per-method interp data (list of dicts, parallel to self.interp_results)
        self._method_results: dict[str, list[dict]] = {}

    def _get_delegates(self) -> dict[str, InterpLLMAgent]:
        """Return name->delegate for all available delegates."""
        delegates = {
            "gradient": self._gradient,
            "logit_lens": self._logit_lens,
            "res_token": self._res_token,
            "prefill": self._prefill,
        }
        if self._relp is not None:
            delegates["relp"] = self._relp
        if self._sae_tfidf is not None:
            delegates["sae_tfidf"] = self._sae_tfidf
        return delegates

    # --- Hooks ---

    def _pre_query_setup(self):
        """Load SAEs if SAE delegate is available."""
        if self._sae_tfidf is not None:
            self._sae_tfidf._pre_query_setup()

    def _post_query_cleanup(self):
        """Batch-fetch SAE descriptions and reset delegate state after sampling."""
        if self._sae_tfidf is not None:
            # Set the delegate's interp_results to its slice of data
            self._sae_tfidf.interp_results = self._method_results.get("sae_tfidf", [])
            self._sae_tfidf._post_query_cleanup()
        if self._relp is not None:
            self._relp._post_query_cleanup()

    # --- Core interp ---

    def _sample_and_query(self, test_inputs, prompts=None):
        """Reset per-method results before each sampling run."""
        self._method_results = {}
        return super()._sample_and_query(test_inputs, prompts=prompts)

    def run_interp(self, ctx: InterpContext) -> dict:
        """Run all delegate interp methods on one sample, merge results."""
        combined = {}
        for method_name, delegate in self._get_delegates().items():
            try:
                result = delegate.run_interp(ctx)
                combined[method_name] = result
                self._method_results.setdefault(method_name, []).append(result)
            except Exception as e:
                print(f"WARNING: {method_name} interp failed: {e}", file=sys.stderr)
                combined[method_name] = {}
                self._method_results.setdefault(method_name, []).append({})
        return combined

    def format_interp_results(self) -> str:
        """Not used — data goes to workspace files."""
        return ""

    # --- Pattern discovery via Codex ---

    def find_pattern(self) -> str:
        """Write workspace, invoke Codex CLI, read answer.txt."""
        if not self.queried_inputs:
            return "No pattern found (no samples queried)."

        if self.dry_run:
            prompt = f"[CODEX_READ workspace with {len(self.queried_inputs)} samples, methods: {list(self._get_delegates().keys())}]"
            self.save_dry_run_prompt(prompt, "find_pattern")
            return "[DRY RUN - NO PATTERN]"

        workspace = Path(tempfile.mkdtemp(prefix="codex_read_"))
        try:
            self._write_workspace(workspace)
            answer = self._run_codex(workspace)
            return answer
        finally:
            if not self._keep_workspace:
                shutil.rmtree(workspace, ignore_errors=True)
            else:
                print(f"Workspace kept at: {workspace}")

    def _write_workspace(self, workspace: Path) -> None:
        """Write io_pairs.jsonl, per-method dirs, and instructions.md."""
        assert len(self.queried_inputs) < 1000, (
            f"Too many samples ({len(self.queried_inputs)}) for 3-digit file naming"
        )
        # io_pairs.jsonl
        with open(workspace / "io_pairs.jsonl", "w") as f:
            for inp, label in zip(self.queried_inputs, self.queried_results):
                f.write(json.dumps({"inputs": _serialize(inp), "output": label}) + "\n")

        # Per-method directories
        delegates = self._get_delegates()
        available_methods = []
        for method_name, delegate in delegates.items():
            method_data = self._method_results.get(method_name, [])
            if not method_data or not any(method_data):
                continue

            method_dir = workspace / method_name
            method_dir.mkdir()
            available_methods.append(method_name)

            # summary.txt via delegate's format_interp_results()
            delegate.interp_results = method_data
            delegate.queried_inputs = self.queried_inputs
            delegate.queried_results = self.queried_results
            # For gradient v2, set up the gradient tracking lists
            if hasattr(delegate, 'samples_with_gradients'):
                delegate.samples_with_gradients = [
                    i for i, d in enumerate(method_data) if d
                ]
                delegate.samples_without_gradients = [
                    i for i, d in enumerate(method_data) if not d
                ]
            summary = delegate.format_interp_results()
            with open(method_dir / "summary.txt", "w") as f:
                f.write(summary)

            # Per-sample JSON
            for i, (inp, label, interp_data) in enumerate(
                zip(self.queried_inputs, self.queried_results, method_data)
            ):
                sample = {
                    "sample_index": i,
                    "inputs": _serialize(inp),
                    "output": label,
                    "interp": _serialize(interp_data),
                }
                with open(method_dir / f"sample_{i:03d}.json", "w") as f:
                    json.dump(sample, f, indent=2, default=str)

        # instructions.md
        method_desc_lines = []
        for m in available_methods:
            desc = METHOD_DESCRIPTIONS.get(m, f"**{m}/** — Interpretability data.")
            method_desc_lines.append(f"- {desc}")
        method_desc_text = "\n".join(method_desc_lines) if method_desc_lines else "No methods available."

        instructions = INSTRUCTIONS_MD.format(
            n_samples=len(self.queried_inputs),
            method_descriptions=method_desc_text,
        )
        with open(workspace / "instructions.md", "w") as f:
            f.write(instructions)

    def _run_codex(self, workspace: Path) -> str:
        """Invoke `codex exec` and return the discovered pattern."""
        # git init so Codex recognizes the workspace for sandbox
        subprocess.run(["git", "init"], cwd=str(workspace),
                       capture_output=True, check=True)

        cmd = [
            "codex", "exec",
            "--cd", str(workspace),
            "--full-auto",
            "--json",
            # Use bwrap sandbox — Landlock fails on many kernels (e.g. RHEL 5.14)
            "-c", "use_linux_sandbox_bwrap=true",
            "-c", "model_reasoning_effort=high",
            "-o", str(workspace / "codex_last_message.txt"),
            "Read instructions.md and follow the instructions there",
        ]
        if self._codex_model:
            cmd.insert(2, "--model")
            cmd.insert(3, self._codex_model)

        print(f"Running Codex CLI (timeout={self._codex_timeout}s)...")
        print(f"  cmd: {' '.join(cmd)}")

        events_file = workspace / "codex_events.jsonl"
        stderr_file = workspace / "codex_stderr.log"
        try:
            with open(events_file, "w") as f_out, open(stderr_file, "w") as f_err:
                proc = subprocess.Popen(
                    cmd,
                    stdout=f_out,
                    stderr=f_err,
                    text=True,
                    cwd=str(workspace),
                )
                try:
                    proc.wait(timeout=self._codex_timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                    print(f"WARNING: Codex timed out after {self._codex_timeout}s", file=sys.stderr)
            if proc.returncode and proc.returncode != 0:
                print(f"Codex stderr:\n{stderr_file.read_text()}", file=sys.stderr)
        except Exception as e:
            print(f"WARNING: Codex execution failed: {e}", file=sys.stderr)

        # Read answer.txt (primary), fallback to codex_last_message.txt
        answer_file = workspace / "answer.txt"
        fallback_file = workspace / "codex_last_message.txt"

        if answer_file.exists():
            answer = answer_file.read_text().strip()
            if answer:
                print(f"Codex answer (from answer.txt): {answer}")
                return answer

        if fallback_file.exists():
            answer = fallback_file.read_text().strip()
            if answer:
                print(f"Codex answer (from codex_last_message.txt): {answer}")
                return answer

        return "No pattern found (Codex produced no output)."

    # --- ESK ---

    def predict_esk(self) -> AgentResult:
        raise NotImplementedError("codex_read does not support ESK tasks.")

    # --- Metadata ---

    def get_metadata(self) -> dict[str, Any]:
        metadata = super().get_metadata()
        metadata["strategy"] = "codex_read"
        metadata["methods_available"] = list(self._get_delegates().keys())
        metadata["samples_per_method"] = {
            name: len(data)
            for name, data in self._method_results.items()
        }
        return metadata
