"""InterpLLMAgent base class for agents that use interp tools + LLM pattern matching.

Provides the shared predict/pattern/predict-remaining flow, so subclasses
only need to implement:
- run_interp(ctx) -> dict  (extract interp metadata from one prompt)
- format_interp_results() -> str  (format self.interp_results into human-readable text)
"""

import asyncio
import random
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import litellm
import tiktoken

# GPT-5.1 context limit
MAX_PROMPT_TOKENS = 250_000
TAIL_PRESERVE_TOKENS = 2_500  # keep last ~2.5k tokens intact
_TIKTOKEN_ENC = tiktoken.get_encoding("o200k_base")  # GPT-5.1 encoding

from .base import BaseAgent, AgentResult
from ..budget import BudgetExceededError


def format_input_for_prediction(inputs: dict[str, Any]) -> str:
    """Format input for prediction query."""
    return ", ".join(f"{k}={v}" for k, v in inputs.items())


@dataclass
class InterpContext:
    """Context for a single interp query.

    Caller prepares prompt + prediction (charged);
    run_interp extracts interp metadata (usually free).
    """
    prompt: str
    prediction: bool | None = None
    inputs: dict[str, Any] | None = None
    response: str | None = None
    probs: dict[str, float] | None = None


def _truncate_prompt(prompt: str, max_tokens: int, tail_tokens: int) -> str:
    """Truncate prompt to fit max_tokens, preserving the tail.

    Tokenizes, keeps the first (max_tokens - tail_tokens - marker) tokens
    and the last tail_tokens, decodes back to text with a truncation marker.
    """
    tokens = _TIKTOKEN_ENC.encode(prompt)
    orig_len = len(tokens)
    if orig_len <= max_tokens:
        return prompt

    marker = "\n\n[... TRUNCATED to fit context window ...]\n\n"
    marker_tokens = _TIKTOKEN_ENC.encode(marker)
    head_budget = max_tokens - tail_tokens - len(marker_tokens)
    if head_budget < 0:
        head_budget = max_tokens // 2

    head = _TIKTOKEN_ENC.decode(tokens[:head_budget])
    tail = _TIKTOKEN_ENC.decode(tokens[-tail_tokens:])
    result = head + marker + tail
    result_len = len(_TIKTOKEN_ENC.encode(result))
    print(f"  Truncated prompt: {orig_len:,} -> {result_len:,} tokens (removed {orig_len - result_len:,})")
    return result


class InterpLLMAgent(BaseAgent):
    """Base for agents that: run interp tool -> find pattern with LLM -> predict remaining.

    Subclasses must implement:
    - run_interp(ctx) -> dict: Extract interp metadata from one prompt.
      Usually FREE (charge=False internally).
      Exception: gradient/relp agents charge extra for backward pass.
    - format_interp_results() -> str: Format self.interp_results into
      human-readable text for the GPT prompt. Include section headers and
      descriptions. Return '' if no interp data (e.g. llm_guess).

    Optional hooks:
    - _pre_query_setup(): Called before sampling loop (e.g., load SAEs).
    - _post_query_cleanup(): Called after sampling loop (e.g., batch-fetch descriptions).
    """

    name: str = "interp_llm_base"
    _query_with_backward: bool = False

    def __init__(self, *args, format_style: str = "structured", **kwargs):
        super().__init__(*args, **kwargs)
        self.format_style = format_style
        self.pattern: str | None = None
        self.pattern_prompt: str | None = None
        self.queried_inputs: list[dict] = []
        self.queried_results: list[bool] = []
        self.interp_results: list[dict] = []

    # --- Core abstraction ---

    @abstractmethod
    def run_interp(self, ctx: InterpContext) -> dict:
        """Extract interp metadata from one prompt. Usually FREE (charge=False).

        Returns agent-specific dict, e.g. {"layer_predictions": [...]}.
        Exception: gradient/relp agents charge extra for backward pass.
        """
        ...

    @abstractmethod
    def format_interp_results(self) -> str:
        """Format self.interp_results into human-readable text for the GPT prompt.

        Include section headers and descriptions of what the data means.
        Include any "Analyze carefully..." instructions specific to this agent.
        Return '' if no interp data (e.g. llm_guess).
        """
        ...

    def find_pattern(self) -> str:
        """Find decision pattern using GPT-5.1.

        Combines I/O pairs + format_interp_results() + standard instruction.
        Agents only need to implement format_interp_results().
        """
        if not self.queried_inputs:
            return "No pattern found (no samples queried)."

        # Format I/O pairs
        io_pairs = []
        for inp, label in zip(self.queried_inputs, self.queried_results):
            fields = ", ".join(f"{k}={v}" for k, v in inp.items())
            io_pairs.append(f"Input: {{{fields}}} -> Output: {'Yes' if label else 'No'}")
        io_pairs_text = "\n".join(io_pairs)

        # Get interp text (per-agent)
        interp_text = self.format_interp_results()

        # Build prompt
        prompt = f"""You are analyzing a decision-making system. Given the following input-output examples, identify the pattern or rule that determines the output (Yes/No).

## All Input-Output Examples ({len(self.queried_inputs)} samples)

{io_pairs_text}
{chr(10) + interp_text + chr(10) if interp_text else ""}
Find the decision rule. Steps:
1. Use the attribution scores to identify the most important fields (ignore low-attribution fields).
2. Use the model reasoning and input values to determine thresholds.
3. Formulate a rule using only the important fields. Be specific about thresholds. Start simple (1-2 fields), add complexity only if needed.
4. Verify your rule against ALL examples above. If it doesn't match some examples, add more conditions or adjust thresholds.
5. Output ONLY the final decision rule, nothing else."""

        prompt = _truncate_prompt(prompt, MAX_PROMPT_TOKENS, TAIL_PRESERVE_TOKENS)

        # Sanitize prompt to prevent JSON serialization issues
        prompt = prompt.encode("utf-8", errors="replace").decode("utf-8")
        prompt = "".join(c for c in prompt if c.isprintable() or c in "\n\t ")

        self.pattern_prompt = prompt

        if self.dry_run:
            self.save_dry_run_prompt(prompt, "find_pattern")
            return "[DRY RUN - NO PATTERN]"

        for attempt in range(2):
            try:
                response = litellm.completion(
                    model="openai/gpt-5.1",
                    messages=[{"role": "user", "content": prompt}],
                    allowed_openai_params=['reasoning_effort'],
                    reasoning_effort="high",
                )
                break
            except Exception as e:
                if attempt == 0 and ("JSON" in str(e) or "parse" in str(e).lower() or "BadRequest" in type(e).__name__):
                    print(f"  Retrying find_pattern after error: {e}")
                    prompt = prompt.encode("ascii", errors="ignore").decode("ascii")
                    continue
                raise

        pattern = response.choices[0].message.content.strip()
        # Sanitize pattern to prevent issues in predict_with_pattern
        pattern = pattern.encode("utf-8", errors="replace").decode("utf-8")
        pattern = "".join(c for c in pattern if c.isprintable() or c in "\n\t ")
        print(f'{pattern=}')
        print(f'Used tokens: {response.usage.total_tokens} / {response.usage.prompt_tokens} / {response.usage.completion_tokens}')
        return pattern

    # --- Generic sampling loop ---

    def _pre_query_setup(self):
        """Hook: before sampling. Override to load SAEs, etc."""
        pass

    def _post_query_cleanup(self):
        """Hook: after sampling. Override to batch-fetch descriptions, etc."""
        pass

    def _sample_and_query(self, test_inputs, prompts=None):
        """Shuffle, predict (charged), run_interp (free), collect.

        Override for non-standard patterns (gradient alternating, prefill two-call).
        """
        self._pre_query_setup()
        indices = list(range(len(test_inputs)))
        random.shuffle(indices)
        predictions = [None] * len(test_inputs)
        self.queried_inputs = []
        self.queried_results = []
        self.interp_results = []

        for idx in indices:
            prompt = prompts[idx] if prompts else self.make_prompt(test_inputs[idx], self.format_style)
            tokens_needed = self.model.count_tokens(prompt)
            if not self.model.budget.can_afford(tokens_needed, with_backward=self._query_with_backward):
                break
            try:
                # 1. Predict (CHARGED)
                prediction, probs = self.model.predict_yes_no(prompt)
                # 2. Run interp (FREE, except gradient/relp)
                ctx = InterpContext(prompt=prompt, prediction=prediction, inputs=test_inputs[idx], probs=probs)
                interp_data = self.run_interp(ctx)

                predictions[idx] = prediction
                self.queried_inputs.append(test_inputs[idx])
                self.queried_results.append(prediction)
                self.interp_results.append(interp_data)
            except BudgetExceededError:
                break

        self._post_query_cleanup()
        return predictions

    # --- Phase-split methods (for concurrent GPU/API execution) ---

    def gpu_phase(self, test_inputs, prompts=None):
        """GPU-bound phase: sample + interp. Model not needed after this returns."""
        assert not self.task_descriptor, "Use predict_esk() for ESK tasks"
        return self._sample_and_query(test_inputs, prompts=prompts)

    def api_phase(self, test_inputs, predictions):
        """API-bound phase: GPT-5.1 pattern discovery + GPT-4.1 predictions. No GPU needed."""
        self.pattern = self.find_pattern()
        return self._predict_remaining_with_pattern(test_inputs, predictions)

    # --- Shared predict flow ---

    def predict(self, test_inputs):
        assert not self.task_descriptor, "Use predict_esk() for ESK tasks"
        predictions = self._sample_and_query(test_inputs)
        self.pattern = self.find_pattern()
        return self._predict_remaining_with_pattern(test_inputs, predictions)

    def predict_with_prompts(self, test_inputs, prompts):
        assert not self.task_descriptor, "Use predict_esk() for ESK tasks"
        predictions = self._sample_and_query(test_inputs, prompts=prompts)
        self.pattern = self.find_pattern()
        return self._predict_remaining_with_pattern(test_inputs, predictions)

    def _predict_remaining_with_pattern(self, test_inputs, predictions):
        """Predict unqueried samples using the discovered pattern."""
        to_predict_indices = []
        to_predict_inputs = []
        for idx, pred in enumerate(predictions):
            if pred is None:
                to_predict_indices.append(idx)
                to_predict_inputs.append(test_inputs[idx])

        if self.dry_run:
            print(f"[DRY RUN] Skipping {len(to_predict_inputs)} LLM predictions")
            final = list(predictions)
            for idx in to_predict_indices:
                final[idx] = True
            return AgentResult(predictions=final, metadata=self.get_metadata())

        async def predict_all():
            tasks = [self.predict_with_pattern(inp) for inp in to_predict_inputs]
            return await asyncio.gather(*tasks)

        if to_predict_inputs:
            print(f"Predicting {len(to_predict_inputs)} samples in parallel.")
            llm_predictions = asyncio.run(predict_all())
        else:
            llm_predictions = []

        final = list(predictions)
        for idx, llm_pred in zip(to_predict_indices, llm_predictions):
            final[idx] = llm_pred

        return AgentResult(predictions=final, metadata=self.get_metadata())

    async def predict_with_pattern(self, inputs: dict[str, Any]) -> bool:
        """Use GPT-4.1 to predict a single sample using the discovered pattern."""
        input_text = format_input_for_prediction(inputs)

        prompt = f"""You are a decision-making system. Apply the following rule to determine if the output should be Yes or No.

Rule: {self.pattern}

Input: {{{input_text}}}

Based on the rule above, should the output be Yes or No? Answer with just "Yes" or "No"."""

        # Sanitize prompt
        prompt = prompt.encode("utf-8", errors="replace").decode("utf-8")
        prompt = "".join(c for c in prompt if c.isprintable() or c in "\n\t ")

        for attempt in range(2):
            try:
                response = await litellm.acompletion(
                    model="openai/gpt-4.1",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=10,
                )
                answer = response.choices[0].message.content.strip().lower()
                return answer.startswith("yes")
            except Exception as e:
                if attempt == 0:
                    prompt = prompt.encode("ascii", errors="ignore").decode("ascii")
                    continue
                # On second failure, default to True (50/50 guess)
                print(f"  predict_with_pattern failed, defaulting to Yes: {e}")
                return True

    # --- ESK (Eliciting Secret Knowledge) support ---

    def predict_esk(self) -> AgentResult:
        """ESK prediction: setup, generate responses, run interp, cleanup, find pattern.

        Uses the same _pre_query_setup/_post_query_cleanup hooks as the normal
        sampling flow, so SAE agents get SAE loading and batch description
        fetching for free.
        """
        td = self.task_descriptor
        self.queried_prompts: list[str] = []
        self.queried_responses: list[str] = []
        self.interp_results = []

        self._pre_query_setup()

        # Shuffle and sample prompts within budget
        indices = list(range(len(td.prompts)))
        random.shuffle(indices)

        for idx in indices:
            prompt = td.prompts[idx]
            formatted = self.model.apply_chat_template(prompt)
            if not self.model.budget.can_afford(1):
                break
            try:
                response = self.model.generate(
                    formatted,
                    max_new_tokens=td.max_new_tokens,
                    temperature=td.temperature,
                )
                self._esk_process_sample(prompt, response, len(self.queried_prompts))
                self.queried_prompts.append(prompt)
                self.queried_responses.append(response)
            except BudgetExceededError:
                break

        self._post_query_cleanup()

        self.pattern = self._find_pattern_esk()
        return AgentResult(predictions=[], metadata=self._get_esk_metadata())

    def _esk_process_sample(self, prompt, response, sample_idx):
        """Run interp on ESK prompt. Override for agent-specific processing."""
        try:
            ctx = InterpContext(prompt=prompt, response=response)
            interp_data = self.run_interp(ctx)
            self.interp_results.append(interp_data)
        except Exception:
            self.interp_results.append({})

    def _format_esk_interp(self) -> str:
        """Format interp data for ESK discovery prompt. Must override for ESK-compatible agents."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement _format_esk_interp. "
            "ESK-compatible agents must override this method (return '' if no interp data)."
        )

    def _find_pattern_esk(self) -> str:
        """Build ESK discovery prompt and call LLM to find the secret."""
        td = self.task_descriptor
        if not self.queried_prompts:
            return "No pattern found (no samples queried)."

        samples_text = "\n\n".join(
            td.format_sample(p, r)
            for p, r in zip(self.queried_prompts, self.queried_responses)
        )
        interp_section = self._format_esk_interp()

        prompt = f"""{td.system_description}

## Model Responses ({len(self.queried_prompts)} queries)

{samples_text}
{("" if not interp_section else chr(10) + interp_section + chr(10))}
{td.discovery_instruction}"""

        self.pattern_prompt = prompt

        if self.dry_run:
            self.save_dry_run_prompt(prompt, "find_pattern_esk")
            return "[DRY RUN]"

        response = litellm.completion(
            model="openai/gpt-5.1",
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort="high",
            allowed_openai_params=['reasoning_effort'],
        )
        result = response.choices[0].message.content.strip()
        print(f"ESK pattern: {result}")
        return result

    def _get_esk_metadata(self) -> dict:
        """Get ESK-specific metadata."""
        td = self.task_descriptor
        score, details = td.evaluate(self.pattern) if self.pattern else (0.0, {})
        return {
            "pattern": self.pattern,
            "pattern_prompt": getattr(self, 'pattern_prompt', None),
            "esk_score": score,
            "esk_eval_details": details,
            "esk_task": td.task_name,
            "esk_ground_truth": td.ground_truth,
            "samples_queried": len(getattr(self, 'queried_prompts', [])),
            "budget_summary": self.model.budget.summary(),
        }

    def get_metadata(self):
        return {
            "strategy": self.name,
            "samples_queried": len(self.queried_inputs),
            "pattern": self.pattern,
            "pattern_prompt": self.pattern_prompt,
            "budget_summary": self.model.budget.summary(),
        }
