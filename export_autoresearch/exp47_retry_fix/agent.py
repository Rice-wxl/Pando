"""Autoresearch interp agent -- editable by meta-agent.

Only edit run_interp() and format_interp_results().
Do NOT add extra litellm/GPT calls, modify the base class, or change the budget.
"""

import sys
from pathlib import Path
from typing import Any

# Add package root to path for src/ imports
sys.path.insert(0, str(Path(__file__).parent))

from src.agents.interp_llm_base import InterpLLMAgent, InterpContext
from src.agents.relp import relp_mode
from src.budget import BudgetExceededError


class AutoresearchAgent(InterpLLMAgent):
    """Interp agent: prefill + RelP + programmatic rule construction.

    Radical approach: instead of dumping raw data for GPT-5.1, we
    algorithmically construct candidate decision rules from the interp
    data and present them alongside evidence for refinement.
    """

    name = "autoresearch"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._relp_logged = False

    def _get_hf_model(self):
        return self.model.model.model

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

    def _compute_field_grads(self, inputs, tokens, grad_norms):
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
        """
        if not self.queried_inputs or not self.queried_results:
            return ""

        # 1. Aggregate field importance
        all_fg = [r.get("field_grads", {}) for r in self.interp_results]
        field_totals = {}
        for fg in all_fg:
            for f, g in fg.items():
                if f == "header":
                    continue
                field_totals[f] = field_totals.get(f, 0) + g

        if not field_totals:
            return ""

        sorted_fields = sorted(field_totals.items(), key=lambda x: x[1], reverse=True)
        top_fields = [f for f, _ in sorted_fields[:5]]

        # 2. For each top field, find the best split
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

            # Try numeric split
            try:
                nums = [(float(v), l) for v, l in zip(values, labels)]
                nums.sort(key=lambda x: x[0])

                best_acc = 0
                best_thresh = None
                best_dir = None

                for i in range(len(nums) - 1):
                    thresh = (nums[i][0] + nums[i + 1][0]) / 2
                    # Try "field >= thresh -> Yes"
                    acc_ge = sum(1 for v, l in nums if (v >= thresh) == l) / len(nums)
                    # Try "field <= thresh -> Yes"
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
                # Categorical split
                val_yes_rate = {}
                val_count = {}
                for v, l in zip(values, labels):
                    sv = str(v)
                    val_count[sv] = val_count.get(sv, 0) + 1
                    if l:
                        val_yes_rate[sv] = val_yes_rate.get(sv, 0) + 1

                if len(val_count) >= 2:
                    # Find the categorical value that best separates
                    best_acc = 0
                    best_val = None
                    for cat_val in val_count:
                        # Try "field == cat_val -> Yes"
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

        # 3. Format the candidate rules
        splits.sort(key=lambda x: (-x["accuracy"], -x["attribution"]))
        lines = ["## Candidate Decision Rules (algorithmically derived)\n"]
        lines.append("Based on gradient attribution and threshold analysis of the 10 samples:\n")

        for i, s in enumerate(splits[:4]):
            if s["type"] == "numeric":
                lines.append(
                    f"  {i+1}. {s['field']} {s['direction']} {s['threshold']:.0f} "
                    f"(accuracy on queried samples: {s['accuracy']:.0%}, "
                    f"attribution: {s['attribution']:.1f})"
                )
            else:
                lines.append(
                    f"  {i+1}. {s['field']} == '{s['value']}' "
                    f"(accuracy on queried samples: {s['accuracy']:.0%}, "
                    f"attribution: {s['attribution']:.1f})"
                )

        lines.append("")
        lines.append(
            "These are candidate single-field rules. The actual rule likely combines "
            "multiple conditions (AND/OR/nested). Use these candidates as building blocks."
        )

        return "\n".join(lines)

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
                    "Mean attribution magnitude per field across all samples (higher = more likely part of the rule):\n"
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
