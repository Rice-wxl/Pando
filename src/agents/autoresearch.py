"""Autoresearch agent progression — 10 historical snapshots from autonomous research.

Each agent class corresponds to a breakthrough snapshot from export_autoresearch/.
run_interp() and format_interp_results() are copied verbatim from the exports.
Shared helpers (_compute_field_grads, _build_candidate_rules) live on AutoresearchBase.

See export_autoresearch/agent_descriptions.md for the full progression chain.
"""

from . import register_agent
from .interp_llm_base import InterpLLMAgent, InterpContext
from .relp import relp_mode
from ..budget import BudgetExceededError


# ---------------------------------------------------------------------------
# Prompt instruction constants (used by _find_pattern_instruction overrides)
# ---------------------------------------------------------------------------

_INSTRUCTION_EXPANDED = (
    "Describe the decision rule as concisely as possible. "
    "Focus on which fields matter and what conditions lead to Yes vs No. "
    "Be specific about thresholds and values.\n\n"
    "Important guidelines:\n"
    "- Only use fields with high attribution scores in the rule. "
    "Low-attribution fields are distractors.\n"
    "- The rule may involve nested conditions (if A then check B, else check C).\n"
    "- Look for threshold values in the model's reasoning explanations.\n"
    "- Use Occam's Razor: prefer simpler rules that explain all examples.\n\n"
    "Reply with just the decision rule, no other text."
)

_INSTRUCTION_5STEP_V1 = (
    "Find the decision rule. Steps:\n"
    "1. Use the attribution scores to identify the most important fields "
    "(ignore low-attribution fields).\n"
    "2. Use the model reasoning and input values to determine thresholds.\n"
    "3. Formulate a rule using only the important fields. "
    "Be specific about thresholds.\n"
    "4. Verify your rule against ALL examples above. "
    "If it doesn't match some examples, revise it.\n"
    "5. Output ONLY the final decision rule, nothing else."
)

_INSTRUCTION_5STEP_V2 = (
    "Find the decision rule. Steps:\n"
    "1. Use the attribution scores to identify the most important fields "
    "(ignore low-attribution fields).\n"
    "2. Use the model reasoning and input values to determine thresholds.\n"
    "3. Formulate a rule using only the important fields. "
    "Be specific about thresholds. Start simple (1-2 fields), "
    "add complexity only if needed.\n"
    "4. Verify your rule against ALL examples above. "
    "If it doesn't match some examples, add more conditions "
    "or adjust thresholds.\n"
    "5. Output ONLY the final decision rule, nothing else."
)


# ---------------------------------------------------------------------------
# Shared base class (not registered as an agent)
# ---------------------------------------------------------------------------

class AutoresearchBase(InterpLLMAgent):
    """Shared helpers for all autoresearch agent versions."""

    # Configurable for _build_candidate_rules variants
    _header_keys: tuple[str, ...] = ("header",)
    _candidate_accuracy_text: str = "accuracy on queried samples"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._relp_logged = False

    def _get_hf_model(self):
        return self.model.model.model

    def _compute_field_grads(self, inputs, tokens, grad_norms):
        """Map token-level gradient norms to per-field importance.

        Identical across all 10 autoresearch versions.
        """
        annotated = self.scenario.format(inputs, style=self.format_style, annotated=True)
        field_grads = {}
        token_idx = 0
        for field_name, line_text in annotated.items():
            cur_grads = []
            cur_text = ""
            while token_idx < len(tokens):
                tok_stripped = tokens[token_idx].strip()
                if tok_stripped and tok_stripped in line_text:
                    cur_grads.append(grad_norms[token_idx])
                cur_text += tokens[token_idx]
                token_idx += 1
                if line_text in cur_text:
                    field_grads[field_name] = sum(cur_grads)
                    break
            else:
                field_grads[field_name] = sum(cur_grads) if cur_grads else 0.0
        return field_grads

    def _build_candidate_rules(self):
        """Algorithmically construct candidate decision rules from the data.

        Uses gradient attribution to identify important fields, then tries
        to find thresholds that split Yes/No samples.

        Used by exp22 through final. Header filtering and accuracy text
        are controlled by _header_keys and _candidate_accuracy_text.
        """
        if not self.queried_inputs or not self.queried_results:
            return ""

        all_fg = [r.get("field_grads", {}) for r in self.interp_results]
        field_totals = {}
        for fg in all_fg:
            for f, g in fg.items():
                if f in self._header_keys:
                    continue
                field_totals[f] = field_totals.get(f, 0) + g

        if not field_totals:
            return ""

        sorted_fields = sorted(field_totals.items(), key=lambda x: x[1], reverse=True)
        top_fields = [f for f, _ in sorted_fields[:5]]

        splits = []
        for field in top_fields:
            values = []
            labels = []
            for inp, label in zip(self.queried_inputs, self.queried_results):
                val = inp.get(field)
                if val is not None:
                    values.append(val)
                    labels.append(label)

            if not values:
                continue

            try:
                nums = [(float(v), l) for v, l in zip(values, labels)]
                nums.sort(key=lambda x: x[0])

                best_acc = 0
                best_thresh = None
                best_dir = None

                for i in range(len(nums) - 1):
                    thresh = (nums[i][0] + nums[i + 1][0]) / 2
                    acc_ge = sum(1 for v, l in nums if (v >= thresh) == l) / len(nums)
                    acc_le = sum(1 for v, l in nums if (v <= thresh) == l) / len(nums)

                    if acc_ge > best_acc:
                        best_acc = acc_ge
                        best_thresh = thresh
                        best_dir = ">="
                    if acc_le > best_acc:
                        best_acc = acc_le
                        best_thresh = thresh
                        best_dir = "<="

                if best_thresh is not None and best_acc > 0.5:
                    splits.append({
                        "field": field,
                        "type": "numeric",
                        "threshold": best_thresh,
                        "direction": best_dir,
                        "accuracy": best_acc,
                        "attribution": field_totals[field],
                    })
            except (ValueError, TypeError):
                val_yes_rate = {}
                val_count = {}
                for v, l in zip(values, labels):
                    sv = str(v)
                    val_count[sv] = val_count.get(sv, 0) + 1
                    if l:
                        val_yes_rate[sv] = val_yes_rate.get(sv, 0) + 1

                if len(val_count) >= 2:
                    best_acc = 0
                    best_val = None
                    for cat_val in val_count:
                        acc = sum(1 for v, l in zip(values, labels)
                                  if (str(v) == cat_val) == l) / len(values)
                        if acc > best_acc:
                            best_acc = acc
                            best_val = cat_val

                    if best_val is not None and best_acc > 0.5:
                        splits.append({
                            "field": field,
                            "type": "categorical",
                            "value": best_val,
                            "accuracy": best_acc,
                            "attribution": field_totals[field],
                        })

        if not splits:
            return ""

        acc_text = self._candidate_accuracy_text
        splits.sort(key=lambda x: (-x["accuracy"], -x["attribution"]))
        lines = ["## Candidate Decision Rules (algorithmically derived)\n"]
        lines.append("Based on gradient attribution and threshold analysis of the 10 samples:\n")

        for i, s in enumerate(splits[:4]):
            if s["type"] == "numeric":
                lines.append(
                    f"  {i+1}. {s['field']} {s['direction']} {s['threshold']:.0f} "
                    f"({acc_text}: {s['accuracy']:.0%}, "
                    f"attribution: {s['attribution']:.1f})"
                )
            else:
                lines.append(
                    f"  {i+1}. {s['field']} == '{s['value']}' "
                    f"({acc_text}: {s['accuracy']:.0%}, "
                    f"attribution: {s['attribution']:.1f})"
                )

        lines.append("")
        lines.append(
            "These are candidate single-field rules. The actual rule likely combines "
            "multiple conditions (AND/OR/nested). Use these candidates as building blocks."
        )

        return "\n".join(lines)


# ===========================================================================
# 1. ar_baseline — Basic embedding gradients + prefill
# Source: export_autoresearch/baseline/agent.py
# ===========================================================================

@register_agent("ar_baseline")
class ARBaseline(AutoresearchBase):

    def run_interp(self, ctx: InterpContext) -> dict:
        result = {}

        # 1. Prefill continuation: "yes/no, because ..."
        prefill = "yes, because" if ctx.prediction else "no, because"
        formatted = self.model.apply_chat_template(ctx.prompt)
        prefilled = formatted + prefill
        try:
            continuation = self.model.generate(
                prefilled, max_new_tokens=20, temperature=0.0,
                charge=False,
            )
            result["reasoning"] = prefill + continuation
        except BudgetExceededError:
            result["reasoning"] = prefill

        # 2. Basic embedding gradients -> per-field importance
        grad_data = self.model.get_embedding_gradients(ctx.prompt, charge=False)
        result["tokens"] = grad_data["tokens"]
        result["token_grad_norms"] = grad_data["token_grad_norms"]
        if ctx.inputs:
            field_grads = self._compute_field_grads(
                ctx.inputs, grad_data["tokens"], grad_data["token_grad_norms"]
            )
            result["field_grads"] = field_grads

        return result

    def format_interp_results(self) -> str:
        if not self.interp_results:
            return ""

        sections = []

        # --- Reasoning section ---
        reasoning_lines = []
        for inp, label, interp in zip(
            self.queried_inputs, self.queried_results, self.interp_results
        ):
            reasoning = interp.get("reasoning", "")
            if reasoning:
                fields = ", ".join(f"{k}={v}" for k, v in inp.items())
                reasoning_lines.append(
                    f"Input: {{{fields}}} -> {'Yes' if label else 'No'}\n"
                    f'  Model says: "{reasoning}"'
                )

        if reasoning_lines:
            sections.append(
                "## Model Self-Reasoning\n\n"
                "After each prediction, the model was asked to explain. "
                "Use this to identify which fields matter:\n\n"
                + "\n\n".join(reasoning_lines)
            )

        # --- Field gradient importance ---
        all_field_grads = [r.get("field_grads", {}) for r in self.interp_results]
        if any(all_field_grads):
            field_totals: dict[str, float] = {}
            field_counts: dict[str, int] = {}
            for fg in all_field_grads:
                for f, g in fg.items():
                    if f == "header":
                        continue
                    field_totals[f] = field_totals.get(f, 0) + g
                    field_counts[f] = field_counts.get(f, 0) + 1

            if field_totals:
                sorted_fields = sorted(
                    [(f, field_totals[f] / field_counts[f]) for f in field_totals],
                    key=lambda x: x[1],
                    reverse=True,
                )
                field_lines = [f"  {f}: {imp:.2f}" for f, imp in sorted_fields]
                sections.append(
                    "## Field Importance (Gradient Attribution)\n\n"
                    "Mean gradient magnitude per field (higher = more influential):\n"
                    + "\n".join(field_lines)
                    + "\n\nFields with higher gradient importance are more likely part of the rule."
                )

        return "\n\n".join(sections)


# ===========================================================================
# 2. ar_exp01 — RelP gradients + prefill
# Source: export_autoresearch/exp01_relp/agent.py
# ===========================================================================

@register_agent("ar_exp01")
class ARExp01(AutoresearchBase):

    def run_interp(self, ctx: InterpContext) -> dict:
        result = {}

        # 1. Prefill continuation
        prefill = "yes, because" if ctx.prediction else "no, because"
        formatted = self.model.apply_chat_template(ctx.prompt)
        prefilled = formatted + prefill
        try:
            continuation = self.model.generate(
                prefilled, max_new_tokens=20, temperature=0.0,
                charge=False,
            )
            result["reasoning"] = prefill + continuation
        except BudgetExceededError:
            result["reasoning"] = prefill

        # 2. RelP gradients
        verbose = not self._relp_logged
        try:
            with relp_mode(self._get_hf_model(), rules=["LN", "Identity", "Half", "AH"], verbose=verbose):
                self._relp_logged = True
                grad_data = self.model.get_embedding_gradients(ctx.prompt, charge=False)
        except (ValueError, RuntimeError):
            grad_data = self.model.get_embedding_gradients(ctx.prompt, charge=False)

        result["tokens"] = grad_data["tokens"]
        result["token_grad_norms"] = grad_data["token_grad_norms"]
        if ctx.inputs:
            field_grads = self._compute_field_grads(
                ctx.inputs, grad_data["tokens"], grad_data["token_grad_norms"]
            )
            result["field_grads"] = field_grads

        return result

    def format_interp_results(self) -> str:
        if not self.interp_results:
            return ""

        sections = []

        # --- Per-sample field attribution + reasoning ---
        sample_lines = []
        for inp, label, interp in zip(
            self.queried_inputs, self.queried_results, self.interp_results
        ):
            fields = ", ".join(f"{k}={v}" for k, v in inp.items())
            fg = interp.get("field_grads", {})
            reasoning = interp.get("reasoning", "")

            sorted_fg = sorted(
                [(f, g) for f, g in fg.items() if f != "header"],
                key=lambda x: x[1], reverse=True,
            )
            attr_str = ", ".join(f"{f}({g:.1f})" for f, g in sorted_fg) if sorted_fg else ""

            line = f"Input: {{{fields}}} -> {'Yes' if label else 'No'}"
            if attr_str:
                line += f"\n  Field attribution (higher=more important): {attr_str}"
            if reasoning:
                line += f'\n  Model reasoning: "{reasoning}"'
            sample_lines.append(line)

        if sample_lines:
            sections.append(
                "## Per-Sample Analysis\n\n"
                "Each sample shows the model's prediction, which fields had the highest "
                "attribution scores (indicating importance to the decision), and the model's "
                "self-reported reasoning:\n\n"
                + "\n\n".join(sample_lines)
            )

        # --- Aggregated field importance ---
        all_field_grads = [r.get("field_grads", {}) for r in self.interp_results]
        if any(all_field_grads):
            field_totals: dict[str, float] = {}
            field_counts: dict[str, int] = {}
            for fg in all_field_grads:
                for f, g in fg.items():
                    if f == "header":
                        continue
                    field_totals[f] = field_totals.get(f, 0) + g
                    field_counts[f] = field_counts.get(f, 0) + 1

            if field_totals:
                sorted_fields = sorted(
                    [(f, field_totals[f] / field_counts[f]) for f in field_totals],
                    key=lambda x: x[1], reverse=True,
                )
                field_lines = [f"  {f}: {imp:.2f}" for f, imp in sorted_fields]
                sections.append(
                    "## Field Importance Summary\n\n"
                    "Mean attribution magnitude per field across all samples "
                    "(higher = more likely part of the rule):\n"
                    + "\n".join(field_lines)
                    + "\n\nFocus on the top-ranked fields when inferring the decision rule. "
                    "Fields with low attribution are unlikely to be part of the rule."
                )

        return "\n\n".join(sections)


# ===========================================================================
# 3. ar_exp09 — + Contrastive Yes/No comparison + expanded prompt
# Source: export_autoresearch/exp09_contrastive/agent.py
# ===========================================================================

@register_agent("ar_exp09")
class ARExp09(AutoresearchBase):

    def _find_pattern_instruction(self) -> str:
        return _INSTRUCTION_EXPANDED

    def run_interp(self, ctx: InterpContext) -> dict:
        result = {}

        # 1. Prefill continuation
        prefill = "yes, because" if ctx.prediction else "no, because"
        formatted = self.model.apply_chat_template(ctx.prompt)
        prefilled = formatted + prefill
        try:
            continuation = self.model.generate(
                prefilled, max_new_tokens=20, temperature=0.0, charge=False,
            )
            result["reasoning"] = prefill + continuation
        except BudgetExceededError:
            result["reasoning"] = prefill

        # 2. RelP gradients
        verbose = not self._relp_logged
        try:
            with relp_mode(self._get_hf_model(), rules=["LN", "Identity", "Half", "AH"], verbose=verbose):
                self._relp_logged = True
                grad_data = self.model.get_embedding_gradients(ctx.prompt, charge=False)
        except (ValueError, RuntimeError):
            grad_data = self.model.get_embedding_gradients(ctx.prompt, charge=False)

        result["tokens"] = grad_data["tokens"]
        result["token_grad_norms"] = grad_data["token_grad_norms"]
        if ctx.inputs:
            result["field_grads"] = self._compute_field_grads(
                ctx.inputs, grad_data["tokens"], grad_data["token_grad_norms"]
            )

        return result

    def format_interp_results(self) -> str:
        if not self.interp_results:
            return ""

        sections = []

        # --- Per-sample field attribution + reasoning ---
        sample_lines = []
        for inp, label, interp in zip(
            self.queried_inputs, self.queried_results, self.interp_results
        ):
            fields = ", ".join(f"{k}={v}" for k, v in inp.items())
            fg = interp.get("field_grads", {})
            reasoning = interp.get("reasoning", "")

            sorted_fg = sorted(
                [(f, g) for f, g in fg.items() if f != "header"],
                key=lambda x: x[1], reverse=True,
            )
            attr_str = ", ".join(f"{f}({g:.1f})" for f, g in sorted_fg) if sorted_fg else ""

            line = f"Input: {{{fields}}} -> {'Yes' if label else 'No'}"
            if attr_str:
                line += f"\n  Field attribution (higher=more important): {attr_str}"
            if reasoning:
                line += f'\n  Model reasoning: "{reasoning}"'
            sample_lines.append(line)

        if sample_lines:
            sections.append(
                "## Per-Sample Analysis\n\n"
                "Each sample shows the model's prediction, which fields had the highest "
                "attribution scores (indicating importance to the decision), and the model's "
                "self-reported reasoning:\n\n"
                + "\n\n".join(sample_lines)
            )

        # --- Aggregated field importance ---
        all_field_grads = [r.get("field_grads", {}) for r in self.interp_results]
        field_totals: dict[str, float] = {}
        field_counts: dict[str, int] = {}
        sorted_fields = []
        if any(all_field_grads):
            for fg in all_field_grads:
                for f, g in fg.items():
                    if f == "header":
                        continue
                    field_totals[f] = field_totals.get(f, 0) + g
                    field_counts[f] = field_counts.get(f, 0) + 1

            if field_totals:
                sorted_fields = sorted(
                    [(f, field_totals[f] / field_counts[f]) for f in field_totals],
                    key=lambda x: x[1], reverse=True,
                )
                field_lines = [f"  {f}: {imp:.2f}" for f, imp in sorted_fields]
                sections.append(
                    "## Field Importance Summary\n\n"
                    "Mean attribution magnitude per field across all samples "
                    "(higher = more likely part of the rule):\n"
                    + "\n".join(field_lines)
                    + "\n\nFocus on the top-ranked fields when inferring the decision rule. "
                    "Fields with low attribution are unlikely to be part of the rule."
                )

        # --- Contrastive summary: Yes vs No field values ---
        yes_inputs = [inp for inp, label in zip(self.queried_inputs, self.queried_results) if label]
        no_inputs = [inp for inp, label in zip(self.queried_inputs, self.queried_results) if not label]

        if yes_inputs and no_inputs:
            top_fields = []
            if field_totals:
                top_fields = [f for f, _ in sorted_fields[:6]]
            else:
                top_fields = list(self.queried_inputs[0].keys())

            contrast_lines = []
            for field in top_fields:
                yes_vals = [str(inp.get(field, "")) for inp in yes_inputs]
                no_vals = [str(inp.get(field, "")) for inp in no_inputs]
                contrast_lines.append(f"  {field}: Yes samples={yes_vals}, No samples={no_vals}")

            sections.append(
                "## Yes vs No Comparison (top fields)\n\n"
                "Comparing field values between Yes and No predictions "
                "for the most important fields:\n"
                + "\n".join(contrast_lines)
            )

        return "\n\n".join(sections)


# ===========================================================================
# 4. ar_exp22 — Programmatic candidate rules (replaces contrastive)
# Source: export_autoresearch/exp22_candidate_rules/agent.py
# ===========================================================================

@register_agent("ar_exp22")
class ARExp22(AutoresearchBase):

    def _find_pattern_instruction(self) -> str:
        return _INSTRUCTION_EXPANDED

    def run_interp(self, ctx: InterpContext) -> dict:
        result = {}

        # 1. Prefill continuation
        prefill = "yes, because" if ctx.prediction else "no, because"
        formatted = self.model.apply_chat_template(ctx.prompt)
        prefilled = formatted + prefill
        try:
            continuation = self.model.generate(
                prefilled, max_new_tokens=20, temperature=0.0, charge=False,
            )
            result["reasoning"] = prefill + continuation
        except BudgetExceededError:
            result["reasoning"] = prefill

        # 2. RelP gradients
        verbose = not self._relp_logged
        try:
            with relp_mode(self._get_hf_model(), rules=["LN", "Identity", "Half", "AH"], verbose=verbose):
                self._relp_logged = True
                grad_data = self.model.get_embedding_gradients(ctx.prompt, charge=False)
        except (ValueError, RuntimeError):
            grad_data = self.model.get_embedding_gradients(ctx.prompt, charge=False)

        result["tokens"] = grad_data["tokens"]
        result["token_grad_norms"] = grad_data["token_grad_norms"]
        if ctx.inputs:
            result["field_grads"] = self._compute_field_grads(
                ctx.inputs, grad_data["tokens"], grad_data["token_grad_norms"]
            )

        return result

    def format_interp_results(self) -> str:
        if not self.interp_results:
            return ""

        sections = []

        # --- Per-sample field attribution + reasoning ---
        sample_lines = []
        for inp, label, interp in zip(
            self.queried_inputs, self.queried_results, self.interp_results
        ):
            fields = ", ".join(f"{k}={v}" for k, v in inp.items())
            fg = interp.get("field_grads", {})
            reasoning = interp.get("reasoning", "")

            sorted_fg = sorted(
                [(f, g) for f, g in fg.items() if f != "header"],
                key=lambda x: x[1], reverse=True,
            )
            attr_str = ", ".join(f"{f}({g:.1f})" for f, g in sorted_fg) if sorted_fg else ""

            line = f"Input: {{{fields}}} -> {'Yes' if label else 'No'}"
            if attr_str:
                line += f"\n  Field attribution (higher=more important): {attr_str}"
            if reasoning:
                line += f'\n  Model reasoning: "{reasoning}"'
            sample_lines.append(line)

        if sample_lines:
            sections.append(
                "## Per-Sample Analysis\n\n"
                "Each sample shows the model's prediction, which fields had the highest "
                "attribution scores (indicating importance to the decision), and the model's "
                "self-reported reasoning:\n\n"
                + "\n\n".join(sample_lines)
            )

        # --- Aggregated field importance ---
        all_field_grads = [r.get("field_grads", {}) for r in self.interp_results]
        if any(all_field_grads):
            field_totals: dict[str, float] = {}
            field_counts: dict[str, int] = {}
            for fg in all_field_grads:
                for f, g in fg.items():
                    if f == "header":
                        continue
                    field_totals[f] = field_totals.get(f, 0) + g
                    field_counts[f] = field_counts.get(f, 0) + 1

            if field_totals:
                sorted_fields = sorted(
                    [(f, field_totals[f] / field_counts[f]) for f in field_totals],
                    key=lambda x: x[1], reverse=True,
                )
                field_lines = [f"  {f}: {imp:.2f}" for f, imp in sorted_fields]
                sections.append(
                    "## Field Importance Summary\n\n"
                    "Mean attribution magnitude per field across all samples "
                    "(higher = more likely part of the rule):\n"
                    + "\n".join(field_lines)
                    + "\n\nFocus on the top-ranked fields when inferring the decision rule. "
                    "Fields with low attribution are unlikely to be part of the rule."
                )

        # --- Candidate rules (always shown) ---
        candidate_rules = self._build_candidate_rules()
        if candidate_rules:
            sections.append(candidate_rules)

        return "\n\n".join(sections)


# ===========================================================================
# 5. ar_exp33 — 5-step verified prompt + sanitization + conditional candidates
# Source: export_autoresearch/exp33_verified_prompt/agent.py
# ===========================================================================

@register_agent("ar_exp33")
class ARExp33(AutoresearchBase):
    _sanitize_find_pattern = True
    _retry_find_pattern = True

    def _find_pattern_instruction(self) -> str:
        return _INSTRUCTION_5STEP_V1

    def run_interp(self, ctx: InterpContext) -> dict:
        result = {}

        # 1. Prefill continuation
        prefill = "yes, because" if ctx.prediction else "no, because"
        formatted = self.model.apply_chat_template(ctx.prompt)
        prefilled = formatted + prefill
        try:
            continuation = self.model.generate(
                prefilled, max_new_tokens=20, temperature=0.0, charge=False,
            )
            # Sanitize to prevent JSON serialization issues downstream
            continuation = continuation.encode("utf-8", errors="replace").decode("utf-8")
            continuation = "".join(c for c in continuation if c.isprintable() or c in " \n\t")
            result["reasoning"] = prefill + continuation
        except BudgetExceededError:
            result["reasoning"] = prefill

        # 2. RelP gradients
        verbose = not self._relp_logged
        try:
            with relp_mode(self._get_hf_model(), rules=["LN", "Identity", "Half", "AH"], verbose=verbose):
                self._relp_logged = True
                grad_data = self.model.get_embedding_gradients(ctx.prompt, charge=False)
        except (ValueError, RuntimeError):
            grad_data = self.model.get_embedding_gradients(ctx.prompt, charge=False)

        result["tokens"] = grad_data["tokens"]
        result["token_grad_norms"] = grad_data["token_grad_norms"]
        if ctx.inputs:
            result["field_grads"] = self._compute_field_grads(
                ctx.inputs, grad_data["tokens"], grad_data["token_grad_norms"]
            )

        return result

    def format_interp_results(self) -> str:
        if not self.interp_results:
            return ""

        sections = []

        # --- Per-sample field attribution + reasoning ---
        sample_lines = []
        for inp, label, interp in zip(
            self.queried_inputs, self.queried_results, self.interp_results
        ):
            fields = ", ".join(f"{k}={v}" for k, v in inp.items())
            fg = interp.get("field_grads", {})
            reasoning = interp.get("reasoning", "")

            sorted_fg = sorted(
                [(f, g) for f, g in fg.items() if f != "header"],
                key=lambda x: x[1], reverse=True,
            )
            attr_str = ", ".join(f"{f}({g:.1f})" for f, g in sorted_fg) if sorted_fg else ""

            line = f"Input: {{{fields}}} -> {'Yes' if label else 'No'}"
            if attr_str:
                line += f"\n  Field attribution (higher=more important): {attr_str}"
            if reasoning:
                line += f'\n  Model reasoning: "{reasoning}"'
            sample_lines.append(line)

        if sample_lines:
            sections.append(
                "## Per-Sample Analysis\n\n"
                "Each sample shows the model's prediction, which fields had the highest "
                "attribution scores (indicating importance to the decision), and the model's "
                "self-reported reasoning:\n\n"
                + "\n\n".join(sample_lines)
            )

        # --- Aggregated field importance ---
        all_field_grads = [r.get("field_grads", {}) for r in self.interp_results]
        if any(all_field_grads):
            field_totals: dict[str, float] = {}
            field_counts: dict[str, int] = {}
            for fg in all_field_grads:
                for f, g in fg.items():
                    if f == "header":
                        continue
                    field_totals[f] = field_totals.get(f, 0) + g
                    field_counts[f] = field_counts.get(f, 0) + 1

            if field_totals:
                sorted_fields = sorted(
                    [(f, field_totals[f] / field_counts[f]) for f in field_totals],
                    key=lambda x: x[1], reverse=True,
                )
                field_lines = [f"  {f}: {imp:.2f}" for f, imp in sorted_fields]
                sections.append(
                    "## Field Importance Summary\n\n"
                    "Mean attribution magnitude per field across all samples "
                    "(higher = more likely part of the rule):\n"
                    + "\n".join(field_lines)
                    + "\n\nFocus on the top-ranked fields when inferring the decision rule. "
                    "Fields with low attribution are unlikely to be part of the rule."
                )

        # --- Candidate rules (only for complex rules with 3+ important fields) ---
        all_field_grads_for_candidates = [r.get("field_grads", {}) for r in self.interp_results]
        ft = {}
        for fg in all_field_grads_for_candidates:
            for f, g in fg.items():
                if f != "header":
                    ft[f] = ft.get(f, 0) + g
        if ft:
            max_attr = max(ft.values())
            n_sig = sum(1 for v in ft.values() if v > max_attr * 0.15)
        else:
            n_sig = 0

        if n_sig >= 3:
            candidate_rules = self._build_candidate_rules()
            if candidate_rules:
                sections.append(candidate_rules)

        return "\n\n".join(sections)


# ===========================================================================
# 6. ar_exp35 — "Start simple, add complexity" (agent.py identical to exp33)
# Source: export_autoresearch/exp35_start_simple/agent.py
# ===========================================================================

@register_agent("ar_exp35")
class ARExp35(ARExp33):
    """Identical to exp33 except for the find_pattern instruction (5-step v2)."""

    def _find_pattern_instruction(self) -> str:
        return _INSTRUCTION_5STEP_V2


# ===========================================================================
# 7. ar_exp47 — Confidence annotation + GPT-4.1 retry + "based on" prefill
# Source: export_autoresearch/exp47_retry_fix/agent.py
# ===========================================================================

@register_agent("ar_exp47")
class ARExp47(AutoresearchBase):
    _pass_probs_to_context = True
    _sanitize_find_pattern = True
    _retry_find_pattern = True
    _sanitize_predict_pattern = True
    _retry_predict_pattern = True

    def _find_pattern_instruction(self) -> str:
        return _INSTRUCTION_5STEP_V2

    def run_interp(self, ctx: InterpContext) -> dict:
        result = {}

        # Store confidence
        if ctx.probs:
            result["confidence"] = max(ctx.probs.get("yes", 0.5), ctx.probs.get("no", 0.5))

        # 1. Prefill continuation
        prefill = "yes. The decision was based on" if ctx.prediction else "no. The decision was based on"
        formatted = self.model.apply_chat_template(ctx.prompt)
        prefilled = formatted + prefill
        try:
            continuation = self.model.generate(
                prefilled, max_new_tokens=20, temperature=0.0, charge=False,
            )
            continuation = continuation.encode("utf-8", errors="replace").decode("utf-8")
            continuation = "".join(c for c in continuation if c.isprintable() or c in " \n\t")
            result["reasoning"] = prefill + continuation
        except BudgetExceededError:
            result["reasoning"] = prefill

        # 2. RelP gradients
        verbose = not self._relp_logged
        try:
            with relp_mode(self._get_hf_model(), rules=["LN", "Identity", "Half", "AH"], verbose=verbose):
                self._relp_logged = True
                grad_data = self.model.get_embedding_gradients(ctx.prompt, charge=False)
        except (ValueError, RuntimeError):
            grad_data = self.model.get_embedding_gradients(ctx.prompt, charge=False)

        result["tokens"] = grad_data["tokens"]
        result["token_grad_norms"] = grad_data["token_grad_norms"]
        if ctx.inputs:
            result["field_grads"] = self._compute_field_grads(
                ctx.inputs, grad_data["tokens"], grad_data["token_grad_norms"]
            )

        return result

    def format_interp_results(self) -> str:
        if not self.interp_results:
            return ""

        sections = []

        # --- Per-sample field attribution + reasoning ---
        sample_lines = []
        for inp, label, interp in zip(
            self.queried_inputs, self.queried_results, self.interp_results
        ):
            fields = ", ".join(f"{k}={v}" for k, v in inp.items())
            fg = interp.get("field_grads", {})
            reasoning = interp.get("reasoning", "")

            sorted_fg = sorted(
                [(f, g) for f, g in fg.items() if f != "header"],
                key=lambda x: x[1], reverse=True,
            )
            attr_str = ", ".join(f"{f}({g:.1f})" for f, g in sorted_fg) if sorted_fg else ""

            confidence = interp.get("confidence", None)
            conf_str = f" ({confidence:.0%} confident)" if confidence and confidence < 0.95 else ""
            line = f"Input: {{{fields}}} -> {'Yes' if label else 'No'}{conf_str}"
            if attr_str:
                line += f"\n  Field attribution (higher=more important): {attr_str}"
            if reasoning:
                line += f'\n  Model reasoning: "{reasoning}"'
            sample_lines.append(line)

        if sample_lines:
            sections.append(
                "## Per-Sample Analysis\n\n"
                "Each sample shows the model's prediction, which fields had the highest "
                "attribution scores (indicating importance to the decision), and the model's "
                "self-reported reasoning:\n\n"
                + "\n\n".join(sample_lines)
            )

        # --- Aggregated field importance ---
        all_field_grads = [r.get("field_grads", {}) for r in self.interp_results]
        if any(all_field_grads):
            field_totals: dict[str, float] = {}
            field_counts: dict[str, int] = {}
            for fg in all_field_grads:
                for f, g in fg.items():
                    if f == "header":
                        continue
                    field_totals[f] = field_totals.get(f, 0) + g
                    field_counts[f] = field_counts.get(f, 0) + 1

            if field_totals:
                sorted_fields = sorted(
                    [(f, field_totals[f] / field_counts[f]) for f in field_totals],
                    key=lambda x: x[1], reverse=True,
                )
                field_lines = [f"  {f}: {imp:.2f}" for f, imp in sorted_fields]
                sections.append(
                    "## Field Importance Summary\n\n"
                    "Mean attribution magnitude per field across all samples "
                    "(higher = more likely part of the rule):\n"
                    + "\n".join(field_lines)
                    + "\n\nFocus on the top-ranked fields when inferring the decision rule. "
                    "Fields with low attribution are unlikely to be part of the rule."
                )

        # --- Candidate rules (only for complex rules with 3+ important fields) ---
        all_field_grads_for_candidates = [r.get("field_grads", {}) for r in self.interp_results]
        ft = {}
        for fg in all_field_grads_for_candidates:
            for f, g in fg.items():
                if f != "header":
                    ft[f] = ft.get(f, 0) + g
        if ft:
            max_attr = max(ft.values())
            n_sig = sum(1 for v in ft.values() if v > max_attr * 0.15)
        else:
            n_sig = 0

        if n_sig >= 3:
            candidate_rules = self._build_candidate_rules()
            if candidate_rules:
                sections.append(candidate_rules)

        return "\n\n".join(sections)


# ===========================================================================
# 8. ar_exp57 — Gradient-guided prefill (grads first, mention top field)
# Source: export_autoresearch/exp57_gradient_prefill/agent.py
# ===========================================================================

@register_agent("ar_exp57")
class ARExp57(AutoresearchBase):
    _pass_probs_to_context = True
    _sanitize_find_pattern = True
    _retry_find_pattern = True
    _sanitize_predict_pattern = True
    _retry_predict_pattern = True
    _candidate_accuracy_text = "accuracy"

    def _find_pattern_instruction(self) -> str:
        return _INSTRUCTION_5STEP_V2

    def run_interp(self, ctx: InterpContext) -> dict:
        result = {}

        # Store confidence
        if ctx.probs:
            result["confidence"] = max(ctx.probs.get("yes", 0.5), ctx.probs.get("no", 0.5))

        # 1. RelP gradients FIRST (to identify top field for prefill)
        verbose = not self._relp_logged
        try:
            with relp_mode(self._get_hf_model(), rules=["LN", "Identity", "Half", "AH"], verbose=verbose):
                self._relp_logged = True
                grad_data = self.model.get_embedding_gradients(ctx.prompt, charge=False)
        except (ValueError, RuntimeError):
            grad_data = self.model.get_embedding_gradients(ctx.prompt, charge=False)

        result["tokens"] = grad_data["tokens"]
        result["token_grad_norms"] = grad_data["token_grad_norms"]
        top_field = None
        if ctx.inputs:
            result["field_grads"] = self._compute_field_grads(
                ctx.inputs, grad_data["tokens"], grad_data["token_grad_norms"]
            )
            # Find top attributed field (only if clearly dominant)
            fg = {k: v for k, v in result["field_grads"].items() if k != "_header" and v > 0}
            if fg:
                sorted_fg_list = sorted(fg.values(), reverse=True)
                top_val = sorted_fg_list[0]
                second_val = sorted_fg_list[1] if len(sorted_fg_list) > 1 else 0
                # Only use gradient-guided prefill if top field is clearly dominant
                if top_val > 0 and (second_val == 0 or top_val / max(second_val, 0.001) > 1.5):
                    top_field = max(fg, key=fg.get)

        # 2. Prefill continuation — mention top field if clearly dominant
        formatted = self.model.apply_chat_template(ctx.prompt)
        pred_word = "yes" if ctx.prediction else "no"
        if top_field:
            prefill = f"{pred_word}, because the {top_field}"
        else:
            prefill = f"{pred_word}, because"

        prefilled = formatted + prefill
        try:
            continuation = self.model.generate(
                prefilled, max_new_tokens=20, temperature=0.0, charge=False,
            )
            continuation = continuation.encode("utf-8", errors="replace").decode("utf-8")
            continuation = "".join(c for c in continuation if c.isprintable() or c in " \n\t")
            result["reasoning"] = prefill + continuation
        except BudgetExceededError:
            result["reasoning"] = prefill

        return result

    def format_interp_results(self) -> str:
        if not self.interp_results:
            return ""

        sections = []

        # --- Per-sample field attribution + reasoning ---
        sample_lines = []
        for inp, label, interp in zip(
            self.queried_inputs, self.queried_results, self.interp_results
        ):
            fields = ", ".join(f"{k}={v}" for k, v in inp.items())
            fg = interp.get("field_grads", {})
            reasoning = interp.get("reasoning", "")

            sorted_fg = sorted(
                [(f, g) for f, g in fg.items() if f != "header"],
                key=lambda x: x[1], reverse=True,
            )
            attr_str = ", ".join(f"{f}({g:.1f})" for f, g in sorted_fg) if sorted_fg else ""

            confidence = interp.get("confidence", None)
            conf_str = f" ({confidence:.0%} confident)" if confidence and confidence < 0.95 else ""
            line = f"Input: {{{fields}}} -> {'Yes' if label else 'No'}{conf_str}"
            if attr_str:
                line += f"\n  Field attribution (higher=more important): {attr_str}"
            if reasoning:
                line += f'\n  Model reasoning: "{reasoning}"'
            sample_lines.append(line)

        if sample_lines:
            sections.append(
                "## Per-Sample Analysis\n\n"
                "Each sample shows the model's prediction, which fields had the highest "
                "attribution scores (indicating importance to the decision), and the model's "
                "self-reported reasoning:\n\n"
                + "\n\n".join(sample_lines)
            )

        # --- Aggregated field importance ---
        all_field_grads = [r.get("field_grads", {}) for r in self.interp_results]
        if any(all_field_grads):
            field_totals: dict[str, float] = {}
            field_counts: dict[str, int] = {}
            for fg in all_field_grads:
                for f, g in fg.items():
                    if f == "header":
                        continue
                    field_totals[f] = field_totals.get(f, 0) + g
                    field_counts[f] = field_counts.get(f, 0) + 1

            if field_totals:
                sorted_fields = sorted(
                    [(f, field_totals[f] / field_counts[f]) for f in field_totals],
                    key=lambda x: x[1], reverse=True,
                )
                field_lines = [f"  {f}: {imp:.2f}" for f, imp in sorted_fields]
                sections.append(
                    "## Field Importance Summary\n\n"
                    "Mean attribution magnitude per field across all samples "
                    "(higher = more likely part of the rule):\n"
                    + "\n".join(field_lines)
                    + "\n\nFocus on the top-ranked fields when inferring the decision rule. "
                    "Fields with low attribution are unlikely to be part of the rule."
                )

        # --- Candidate rules (only for complex rules with 3+ important fields) ---
        all_field_grads_for_candidates = [r.get("field_grads", {}) for r in self.interp_results]
        ft = {}
        for fg in all_field_grads_for_candidates:
            for f, g in fg.items():
                if f != "header":
                    ft[f] = ft.get(f, 0) + g
        if ft:
            max_attr = max(ft.values())
            n_sig = sum(1 for v in ft.values() if v > max_attr * 0.15)
        else:
            n_sig = 0

        if n_sig >= 3:
            candidate_rules = self._build_candidate_rules()
            if candidate_rules:
                sections.append(candidate_rules)

        return "\n\n".join(sections)


# ===========================================================================
# 9. ar_exp65 — Top-4 fields per sample + 10-token prefill
# Source: export_autoresearch/exp65_top4_fields/agent.py
# ===========================================================================

@register_agent("ar_exp65")
class ARExp65(AutoresearchBase):
    _pass_probs_to_context = True
    _sanitize_find_pattern = True
    _retry_find_pattern = True
    _sanitize_predict_pattern = True
    _retry_predict_pattern = True
    _candidate_accuracy_text = "accuracy"

    def _find_pattern_instruction(self) -> str:
        return _INSTRUCTION_5STEP_V2

    def run_interp(self, ctx: InterpContext) -> dict:
        result = {}

        # Store confidence
        if ctx.probs:
            result["confidence"] = max(ctx.probs.get("yes", 0.5), ctx.probs.get("no", 0.5))

        # 1. RelP gradients FIRST (to identify top field for prefill)
        verbose = not self._relp_logged
        try:
            with relp_mode(self._get_hf_model(), rules=["LN", "Identity", "Half", "AH"], verbose=verbose):
                self._relp_logged = True
                grad_data = self.model.get_embedding_gradients(ctx.prompt, charge=False)
        except (ValueError, RuntimeError):
            grad_data = self.model.get_embedding_gradients(ctx.prompt, charge=False)

        result["tokens"] = grad_data["tokens"]
        result["token_grad_norms"] = grad_data["token_grad_norms"]
        top_field = None
        if ctx.inputs:
            result["field_grads"] = self._compute_field_grads(
                ctx.inputs, grad_data["tokens"], grad_data["token_grad_norms"]
            )
            # Find top attributed field (only if clearly dominant)
            fg = {k: v for k, v in result["field_grads"].items() if k != "_header" and v > 0}
            if fg:
                sorted_fg_list = sorted(fg.values(), reverse=True)
                top_val = sorted_fg_list[0]
                second_val = sorted_fg_list[1] if len(sorted_fg_list) > 1 else 0
                if top_val > 0 and (second_val == 0 or top_val / max(second_val, 0.001) > 1.5):
                    top_field = max(fg, key=fg.get)

        # 2. Prefill continuation — 10 tokens, mention top field if dominant
        formatted = self.model.apply_chat_template(ctx.prompt)
        pred_word = "yes" if ctx.prediction else "no"
        if top_field:
            prefill = f"{pred_word}, because the {top_field}"
        else:
            prefill = f"{pred_word}, because"

        prefilled = formatted + prefill
        try:
            continuation = self.model.generate(
                prefilled, max_new_tokens=10, temperature=0.0, charge=False,
            )
            continuation = continuation.encode("utf-8", errors="replace").decode("utf-8")
            continuation = "".join(c for c in continuation if c.isprintable() or c in " \n\t")
            result["reasoning"] = prefill + continuation
        except BudgetExceededError:
            result["reasoning"] = prefill

        return result

    def format_interp_results(self) -> str:
        if not self.interp_results:
            return ""

        sections = []

        # --- Per-sample field attribution + reasoning ---
        sample_lines = []
        for inp, label, interp in zip(
            self.queried_inputs, self.queried_results, self.interp_results
        ):
            fields = ", ".join(f"{k}={v}" for k, v in inp.items())
            fg = interp.get("field_grads", {})
            reasoning = interp.get("reasoning", "")

            sorted_fg = sorted(
                [(f, g) for f, g in fg.items() if f != "header"],
                key=lambda x: x[1], reverse=True,
            )
            # Show only top 4 fields (reduces noise for GPT-5.1)
            attr_str = ", ".join(f"{f}({g:.1f})" for f, g in sorted_fg[:4]) if sorted_fg else ""

            confidence = interp.get("confidence", None)
            conf_str = f" ({confidence:.0%} confident)" if confidence and confidence < 0.95 else ""
            line = f"Input: {{{fields}}} -> {'Yes' if label else 'No'}{conf_str}"
            if attr_str:
                line += f"\n  Field attribution (higher=more important): {attr_str}"
            if reasoning:
                line += f'\n  Model reasoning: "{reasoning}"'
            sample_lines.append(line)

        if sample_lines:
            sections.append(
                "## Per-Sample Analysis\n\n"
                "Each sample shows the model's prediction, which fields had the highest "
                "attribution scores (indicating importance to the decision), and the model's "
                "self-reported reasoning:\n\n"
                + "\n\n".join(sample_lines)
            )

        # --- Aggregated field importance ---
        all_field_grads = [r.get("field_grads", {}) for r in self.interp_results]
        if any(all_field_grads):
            field_totals: dict[str, float] = {}
            field_counts: dict[str, int] = {}
            for fg in all_field_grads:
                for f, g in fg.items():
                    if f == "header":
                        continue
                    field_totals[f] = field_totals.get(f, 0) + g
                    field_counts[f] = field_counts.get(f, 0) + 1

            if field_totals:
                sorted_fields = sorted(
                    [(f, field_totals[f] / field_counts[f]) for f in field_totals],
                    key=lambda x: x[1], reverse=True,
                )
                field_lines = [f"  {f}: {imp:.2f}" for f, imp in sorted_fields]
                sections.append(
                    "## Field Importance Summary\n\n"
                    "Mean attribution magnitude per field across all samples "
                    "(higher = more likely part of the rule):\n"
                    + "\n".join(field_lines)
                    + "\n\nFocus on the top-ranked fields when inferring the decision rule. "
                    "Fields with low attribution are unlikely to be part of the rule."
                )

        # --- Candidate rules (only for complex rules with 3+ important fields) ---
        all_field_grads_for_candidates = [r.get("field_grads", {}) for r in self.interp_results]
        ft = {}
        for fg in all_field_grads_for_candidates:
            for f, g in fg.items():
                if f != "header":
                    ft[f] = ft.get(f, 0) + g
        if ft:
            max_attr = max(ft.values())
            n_sig = sum(1 for v in ft.values() if v > max_attr * 0.15)
        else:
            n_sig = 0

        if n_sig >= 3:
            candidate_rules = self._build_candidate_rules()
            if candidate_rules:
                sections.append(candidate_rules)

        return "\n\n".join(sections)


# ===========================================================================
# 10. ar_final — Production agent (= exp65 + _header filter fix)
# Source: export_autoresearch/final/agent.py
# ===========================================================================

@register_agent("ar_final")
class ARFinal(AutoresearchBase):
    _pass_probs_to_context = True
    _sanitize_find_pattern = True
    _retry_find_pattern = True
    _sanitize_predict_pattern = True
    _retry_predict_pattern = True
    _header_keys = ("header", "_header")
    _candidate_accuracy_text = "accuracy"

    def _find_pattern_instruction(self) -> str:
        return _INSTRUCTION_5STEP_V2

    def run_interp(self, ctx: InterpContext) -> dict:
        result = {}

        # Store confidence
        if ctx.probs:
            result["confidence"] = max(ctx.probs.get("yes", 0.5), ctx.probs.get("no", 0.5))

        # 1. RelP gradients FIRST (to identify top field for prefill)
        verbose = not self._relp_logged
        try:
            with relp_mode(self._get_hf_model(), rules=["LN", "Identity", "Half", "AH"], verbose=verbose):
                self._relp_logged = True
                grad_data = self.model.get_embedding_gradients(ctx.prompt, charge=False)
        except (ValueError, RuntimeError):
            grad_data = self.model.get_embedding_gradients(ctx.prompt, charge=False)

        result["tokens"] = grad_data["tokens"]
        result["token_grad_norms"] = grad_data["token_grad_norms"]
        top_field = None
        if ctx.inputs:
            result["field_grads"] = self._compute_field_grads(
                ctx.inputs, grad_data["tokens"], grad_data["token_grad_norms"]
            )
            # Find top attributed field (only if clearly dominant)
            fg = {k: v for k, v in result["field_grads"].items() if k not in ("header", "_header") and v > 0}
            if fg:
                sorted_fg_list = sorted(fg.values(), reverse=True)
                top_val = sorted_fg_list[0]
                second_val = sorted_fg_list[1] if len(sorted_fg_list) > 1 else 0
                if top_val > 0 and (second_val == 0 or top_val / max(second_val, 0.001) > 1.5):
                    top_field = max(fg, key=fg.get)

        # 2. Prefill continuation — 10 tokens, mention top field if dominant
        formatted = self.model.apply_chat_template(ctx.prompt)
        pred_word = "yes" if ctx.prediction else "no"
        if top_field:
            prefill = f"{pred_word}, because the {top_field}"
        else:
            prefill = f"{pred_word}, because"

        prefilled = formatted + prefill
        try:
            continuation = self.model.generate(
                prefilled, max_new_tokens=10, temperature=0.0, charge=False,
            )
            continuation = continuation.encode("utf-8", errors="replace").decode("utf-8")
            continuation = "".join(c for c in continuation if c.isprintable() or c in " \n\t")
            result["reasoning"] = prefill + continuation
        except BudgetExceededError:
            result["reasoning"] = prefill

        return result

    def format_interp_results(self) -> str:
        if not self.interp_results:
            return ""

        sections = []

        # --- Per-sample field attribution + reasoning ---
        sample_lines = []
        for inp, label, interp in zip(
            self.queried_inputs, self.queried_results, self.interp_results
        ):
            fields = ", ".join(f"{k}={v}" for k, v in inp.items())
            fg = interp.get("field_grads", {})
            reasoning = interp.get("reasoning", "")

            sorted_fg = sorted(
                [(f, g) for f, g in fg.items() if f not in ("header", "_header")],
                key=lambda x: x[1], reverse=True,
            )
            # Show only top 4 fields (reduces noise for GPT-5.1)
            attr_str = ", ".join(f"{f}({g:.1f})" for f, g in sorted_fg[:4]) if sorted_fg else ""

            confidence = interp.get("confidence", None)
            conf_str = f" ({confidence:.0%} confident)" if confidence and confidence < 0.95 else ""
            line = f"Input: {{{fields}}} -> {'Yes' if label else 'No'}{conf_str}"
            if attr_str:
                line += f"\n  Field attribution (higher=more important): {attr_str}"
            if reasoning:
                line += f'\n  Model reasoning: "{reasoning}"'
            sample_lines.append(line)

        if sample_lines:
            sections.append(
                "## Per-Sample Analysis\n\n"
                "Each sample shows the model's prediction, which fields had the highest "
                "attribution scores (indicating importance to the decision), and the model's "
                "self-reported reasoning:\n\n"
                + "\n\n".join(sample_lines)
            )

        # --- Aggregated field importance ---
        all_field_grads = [r.get("field_grads", {}) for r in self.interp_results]
        if any(all_field_grads):
            field_totals: dict[str, float] = {}
            field_counts: dict[str, int] = {}
            for fg in all_field_grads:
                for f, g in fg.items():
                    if f in ("header", "_header"):
                        continue
                    field_totals[f] = field_totals.get(f, 0) + g
                    field_counts[f] = field_counts.get(f, 0) + 1

            if field_totals:
                sorted_fields = sorted(
                    [(f, field_totals[f] / field_counts[f]) for f in field_totals],
                    key=lambda x: x[1], reverse=True,
                )
                field_lines = [f"  {f}: {imp:.2f}" for f, imp in sorted_fields]
                sections.append(
                    "## Field Importance Summary\n\n"
                    "Mean attribution magnitude per field across all samples "
                    "(higher = more likely part of the rule):\n"
                    + "\n".join(field_lines)
                    + "\n\nFocus on the top-ranked fields when inferring the decision rule. "
                    "Fields with low attribution are unlikely to be part of the rule."
                )

        # --- Candidate rules (only for complex rules with 3+ important fields) ---
        all_field_grads_for_candidates = [r.get("field_grads", {}) for r in self.interp_results]
        ft = {}
        for fg in all_field_grads_for_candidates:
            for f, g in fg.items():
                if f not in ("header", "_header"):
                    ft[f] = ft.get(f, 0) + g
        if ft:
            max_attr = max(ft.values())
            n_sig = sum(1 for v in ft.values() if v > max_attr * 0.15)
        else:
            n_sig = 0

        if n_sig >= 3:
            candidate_rules = self._build_candidate_rules()
            if candidate_rules:
                sections.append(candidate_rules)

        return "\n\n".join(sections)
