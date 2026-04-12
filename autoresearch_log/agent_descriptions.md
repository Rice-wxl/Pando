# Autoresearch Agent Progression

Each subfolder contains the `agent.py` snapshot at that breakthrough point.
All agents inherit from `src/agents/interp_llm_base.py` and use `src/agents/relp.py`.

## Files changed across the chain

Not just `agent.py` — several experiments also modified infrastructure files.
Starting from `exp33`, each folder includes its own `interp_llm_base.py`.

`base_class.patch` contains only the exp47 GPT-4.1 retry diff for reference.

### `interp_llm_base.py` (base class) versions

| Folders | Version | Key difference |
|---------|---------|----------------|
| baseline, exp01 | Original (not included) | Freeform one-liner "Describe the decision rule" prompt, no retry logic |
| **exp09** | Expanded prompt | Expanded `find_pattern` prompt from one-liner to multi-paragraph with guidelines (attribution, nested conditions, Occam's Razor). |
| **exp22** | Expanded prompt + phase-split | Added `gpu_phase()`/`api_phase()` methods for concurrent eval. Prompt unchanged from exp09. |
| **exp33** | 5-step prompt | Rewrote GPT-5.1 `find_pattern` prompt to structured 5-step with "verify against ALL examples". Added UTF-8 sanitization + retry/ASCII fallback on JSON errors for GPT-5.1 call. |
| **exp35** | 5-step prompt v2 | 2-line tweak to steps 3-4: "Start simple (1-2 fields), add complexity only if needed" and "add more conditions or adjust thresholds". |
| **exp47, exp57, exp65, final** | 5-step v2 + GPT-4.1 retry | Added `probs` field to `InterpContext`. Retry + ASCII fallback on `predict_with_pattern` (GPT-4.1): defaults to `True` on second failure. Pattern sanitization after GPT-5.1 returns. Broadened retry to catch BadRequestError. All four share the same base class. |

### Other infrastructure changes (NOT included in export folders)

These files also changed between versions but are shared infrastructure, not
agent-specific. Use the repo at HEAD for these — they are backward-compatible.

| File | Changed in | What changed |
|------|------------|--------------|
| `src/inference.py` | **exp01** | Default `attn_implementation` changed from `None` → `"eager"` (required for RelP AH rule). All agents from exp01 onward need this. |
| `src/inference.py` | **exp33** | Added `get_attention_weights()` method (used in failed exp24, not used by any exported agent, but present in the codebase). |
| `run_eval.py` | **exp22** | Rewrote to concurrent GPU/API execution with dual semaphores (parallelizes model loading with GPT API calls). |
| `src/evaluation.py` | **exp22** | Added `run_agent_gpu_phase()` / `run_agent_api_phase()` split methods to support concurrent eval. |
| `run_eval.py` | **exp33** | Moved `set_seed` + test set loading inside GPU semaphore (fixed seed race / non-determinism bug). Added stderr reporting to summary output. |
| `src/agents/interp_llm_base.py` | **exp09** | Expanded GPT-5.1 `find_pattern` prompt from one-liner to multi-paragraph with guidelines (intermediate version before exp33's 5-step rewrite). |
| `src/agents/interp_llm_base.py` | **exp22** | Added `gpu_phase()` / `api_phase()` methods for concurrent eval support. |

---

## Progression Chain

```
baseline  82.0%   Mar 19 17:50
  +0.8%  ──>  exp01   82.8%   Mar 19 17:55  (+5m)
  +0.7%  ──>  exp09   83.5%   Mar 19 18:42  (+52m)
   ~0%   ──>  exp22   83.5%   Mar 19 21:34  (+3h44m)
  +0.5%  ──>  exp33   ~83%    Mar 20 01:11  (+7h21m)
  +0.3%  ──>  exp35   83.9%*  Mar 20 01:46  (+7h56m)
  +0.2%  ──>  exp47   83.1%   Mar 20 06:00  (+12h10m)
  +0.1%  ──>  exp57   ~83%    Mar 20 08:31  (+14h41m)
  tuning ──>  exp65   83.4%   Mar 20 15:11  (+21h21m)
  = final                     Mar 20 19:22  (+25h32m)
```

Total wall-clock: ~25.5 hours from baseline to final (autonomous, no human input).

Quick eval (4 models) numbers shown unless noted. Calibrate = 40 models.

---

## Agent Descriptions

Each description is based on diffing `agent.py` (and base class where applicable)
against the previous version in the chain.

### 1. `baseline/` — Prefill + Basic Embedding Gradients
- **Commit**: `00a4936` (Mar 19 17:50 UTC)
- **Files**: `agent.py` only
- **Quick eval**: 82.0% (d1=100, d2=100, d3=74, d4=54)
- **Interp tools**: `get_embedding_gradients`, `generate` (prefill)
- **Architecture**:
  - `run_interp()`: (1) Prefill continuation with `"yes, because"` / `"no, because"`,
    max 20 tokens greedy. (2) Basic `get_embedding_gradients` for per-field importance
    via `_compute_field_grads()`.
  - `format_interp_results()` outputs 2 sections:
    - "Model Self-Reasoning" — per-sample `Input: {fields} -> Yes/No` with model reasoning quote
    - "Field Importance (Gradient Attribution)" — mean gradient magnitude per field
  - Base class `find_pattern` prompt: freeform "Describe the decision rule concisely...
    Use Occam's Razor: prefer simpler rules."

### 2. `exp01_relp/` — RelP Gradients + Eager Attention
- **Commit**: `e1134fa` (Mar 19 17:55 UTC, +5m)
- **Files**: `agent.py` only
- **Infra dep**: requires `src/inference.py` change: `attn_implementation` default `None` → `"eager"` (for RelP AH rule). All subsequent agents inherit this.
- **Quick eval**: 82.8% (d1=100, d2=100, d3=74, d4=57)
- **Delta vs baseline**: +0.8% mean, +3% d4
- **Diff from baseline** (`agent.py`):
  - Added `from src.agents.relp import relp_mode` import
  - Added `__init__` with `self._relp_logged = False` + `_get_hf_model()` helper
  - `run_interp()`: wrapped `get_embedding_gradients` call inside
    `with relp_mode(model, rules=["LN", "Identity", "Half", "AH"])` context manager.
    Falls back to basic gradients if RelP raises ValueError/RuntimeError.
    Logs RelP rules only on first call (verbose flag).
  - Reordering: gradients still run after prefill (unchanged)
  - `format_interp_results()`: restructured into combined "Per-Sample Analysis" section
    (was separate "Model Self-Reasoning"). Each sample now shows field attribution inline:
    `Field attribution (higher=more important): price(4.2), mpg(2.1), ...` plus reasoning.
    Section renamed from "Field Importance (Gradient Attribution)" to "Field Importance Summary".
    Added guidance text: "Focus on the top-ranked fields... Fields with low attribution
    are unlikely to be part of the rule."
  - Removed verbose docstrings/comments from baseline

### 3. `exp09_contrastive/` — + Contrastive Yes/No Comparison
- **Commit**: `d9a91d4` (Mar 19 18:42 UTC, +52m)
- **Files**: `agent.py` + `interp_llm_base.py`
- **Quick eval**: 83.5% (d1=100, d2=100, d3=74, d4=60)
- **Delta vs exp01**: +0.7% mean, +3% d4
- **Diff from exp01**:
  - `agent.py`: `run_interp()` minor cleanup only (removed extra blank lines, shortened
    comments). No functional change to data collection. `format_interp_results()` added
    a 3rd section — "Yes vs No Comparison (top fields)". Splits `queried_inputs` into
    yes/no groups, then for the top 6 attributed fields, shows raw values:
    `price: Yes samples=['45000', '38000'], No samples=['22000', '15000']`.
    This gives GPT-5.1 a grouped view to spot thresholds.
  - `interp_llm_base.py`: expanded `find_pattern` prompt from a one-liner to
    multi-paragraph with explicit guidelines: "Only use fields with high attribution
    scores", "The rule may involve nested conditions", "Look for threshold values in
    the model's reasoning", "Use Occam's Razor".

### 4. `exp22_candidate_rules/` — Programmatic Candidate Rules
- **Commit**: `cd2a4a8` (Mar 19 21:34 UTC, +3h44m)
- **Files**: `agent.py` + `interp_llm_base.py`
- **Infra dep**: `run_eval.py` rewritten to concurrent GPU/API with dual semaphores. `src/evaluation.py` gained `run_agent_gpu_phase()`/`run_agent_api_phase()`. These are eval harness changes — the agent itself doesn't call them.
- **Quick eval**: 83.5% (d1=100, d2=100, d3=74, d4=60)
- **Calibrate**: 82.9% (first to beat relp's 82.6%)
- **Delta vs exp09**: ~0% quick eval, but more robust on calibrate
- **Diff from exp09**:
  - `agent.py`: added `_build_candidate_rules()` method (~130 lines). For each of the top 5
    attributed fields: tries all midpoint thresholds between sorted numeric values,
    picks the split (>= or <=) with highest accuracy on the 10 samples. For
    categorical fields: tries each value as the split. Outputs up to 4 best rules
    sorted by accuracy then attribution, formatted as:
    `1. price >= 35000 (accuracy on queried samples: 80%, attribution: 12.3)`
  - `agent.py` `format_interp_results()`: **replaced** the contrastive Yes/No section
    with a call to `_build_candidate_rules()`. The 3rd section is now "Candidate
    Decision Rules (algorithmically derived)" instead of "Yes vs No Comparison".
  - `interp_llm_base.py`: added `gpu_phase()` and `api_phase()` methods for concurrent
    eval support. `find_pattern` prompt unchanged from exp09.

### 5. `exp33_verified_prompt/` — 5-Step Verified Prompt
- **Commit**: `2f51d22` (Mar 20 01:11 UTC, +7h21m)
- **Files**: `agent.py` + `interp_llm_base.py`
- **Infra dep**: `run_eval.py` moved `set_seed` + test set loading inside GPU semaphore (fixed seed race / non-determinism bug) and added stderr to summary. `src/inference.py` gained `get_attention_weights()` (used in failed exp24, not by this agent).
- **Calibrate**: 82.6% (matches relp exactly)
- **Delta vs exp22**: ~+0.5% on calibrate
- **Diff from exp22**:
  - `agent.py`: added UTF-8 sanitization to prefill continuation output
    (`encode("utf-8", errors="replace")` + strip non-printable chars). Also made
    candidate rules conditional: only shown when 3+ fields have significant
    attribution (>15% of max). This avoids presenting noisy candidates on simple
    d1/d2 rules.
  - `interp_llm_base.py`: **this is where the main change is**.
    Rewrote the GPT-5.1 `find_pattern` prompt from freeform instructions to a
    structured 5-step process:
    ```
    1. Use attribution scores to identify the most important fields
    2. Use model reasoning and input values to determine thresholds
    3. Formulate a rule using only the important fields. Be specific about thresholds.
    4. Verify your rule against ALL examples above. If it doesn't match, revise it.
    5. Output ONLY the final decision rule, nothing else.
    ```
    Also added UTF-8 sanitization of the prompt + retry with ASCII fallback on
    JSON errors for the GPT-5.1 `find_pattern` call.

### 6. `exp35_start_simple/` — "Start Simple, Add Complexity"
- **Commit**: `b1eeb46` (Mar 20 01:46 UTC, +7h56m)
- **Files**: `agent.py` + `interp_llm_base.py`
- **Calibrate**: 83.2% avg, 83.9% best single run
- **Delta vs exp33**: +0.6% calibrate avg
- **Diff from exp33**:
  - `agent.py`: identical to exp33 (no changes).
  - `interp_llm_base.py`: tweaked 2 lines in the `find_pattern` prompt:
    - Step 3: `"Be specific about thresholds."` → `"Be specific about thresholds. Start simple (1-2 fields), add complexity only if needed."`
    - Step 4: `"If it doesn't match some examples, revise it."` → `"If it doesn't match some examples, add more conditions or adjust thresholds."`
    This biases GPT-5.1 toward Occam's razor while still allowing complex rules.

### 7. `exp47_retry_fix/` — Confidence Annotation + GPT-4.1 Retry
- **Commit**: `01c12e3` (Mar 20 06:00 UTC, +12h10m)
- **Files**: `agent.py` + `interp_llm_base.py`
- **Calibrate**: 83.1% avg (much lower variance)
- **Delta vs exp35**: ~+0.2% avg, mainly reduces worst-case by 2-3%
- **Diff from exp35**:
  - `agent.py` (3 changes):
    1. `run_interp()`: stores confidence score from `ctx.probs`
       (`max(yes_prob, no_prob)`) in the result dict.
    2. `run_interp()`: changed prefill text from `"yes, because"` →
       `"yes. The decision was based on"` (more directive).
    3. `format_interp_results()`: appends `"(85% confident)"` to the sample line
       when confidence < 95%. High-confidence samples show no annotation.
  - **Base class** (`interp_llm_base.py`): wrapped the `predict_with_pattern`
    GPT-4.1 call in a 2-attempt retry. On first failure: strip prompt to ASCII-only.
    On second failure: return `True` (50/50 guess) instead of crashing.
    See `base_class.patch` for the exact 15-line diff.

### 8. `exp57_gradient_prefill/` — Gradient-Guided Prefill
- **Commit**: `66a7e68` (Mar 20 08:31 UTC, +14h41m)
- **Files**: `agent.py` + `interp_llm_base.py` (base class unchanged from exp47)
- **Calibrate**: 82.8% avg (d3=79.8%, d4=65.9%)
- **Delta vs exp47**: +0.3% on d2/d3
- **Diff from exp47** (`agent.py` only):
  - `run_interp()` **reordered**: gradients now run FIRST (was prefill first).
    After computing `field_grads`, identifies the top-attributed field. If it's
    clearly dominant (>1.5x the second field), uses it in the prefill:
    `"yes, because the price"` instead of generic `"yes. The decision was based on"`.
    If no field is dominant, falls back to `"yes, because"` (reverted from exp47's
    `"yes. The decision was based on"`).
  - Also added `top_field = None` tracking variable, and the dominance check:
    ```python
    if top_val > 0 and (second_val == 0 or top_val / max(second_val, 0.001) > 1.5):
        top_field = max(fg, key=fg.get)
    ```
  - Minor: shortened candidate rule text from `"accuracy on queried samples:"` to
    `"accuracy:"`.

### 9. `exp65_top4_fields/` — Top-4 Per-Sample Fields + 10-Token Prefill
- **Commit**: `5fc0330` (Mar 20 15:11 UTC, +21h21m)
- **Files**: `agent.py` + `interp_llm_base.py` (base class unchanged from exp47)
- **Calibrate**: 83.4% avg over 7 runs (definitive best)
- **Delta vs exp57**: fine-tuning, stabilizes calibrate avg
- **Diff from exp57** (`agent.py` only, 2 changes):
  1. `run_interp()`: reduced prefill `max_new_tokens` from 20 → **10**. Shorter
     continuations = less noise for GPT-5.1 to parse.
  2. `format_interp_results()`: per-sample field attribution now shows only
     **top 4 fields** (`sorted_fg[:4]`) instead of all fields. Car scenarios have
     8+ fields; showing all dilutes the signal. Top 4 captures the important
     fields for d1-d4 (which have 1-4 relevant fields respectively).

### 10. `final/` — Production Agent (= exp65 + header fix)
- **Commit**: `51be446` (Mar 20 19:22 UTC, +25h32m)
- **Files**: `agent.py` + `interp_llm_base.py` (base class unchanged from exp47)
- **Calibrate**: 83.4% ±0.5% (7 runs), 82.9% (10 runs grand avg)
- **Holdout**: 81.1% ±0.3% (3 runs, 66 models each)
- **Diff from exp65** (`agent.py` only, 1 change):
  - Fixed header filtering throughout: `f == "header"` → `f in ("header", "_header")`
    in 4 places (`_build_candidate_rules` and `format_interp_results`). The scenario
    format sometimes uses `_header` as the key, which was leaking through as a
    spurious field in attribution and candidate rules.
  - Same code otherwise. Additional calibrate/holdout runs confirmed stability.

---

## Performance vs Relp Benchmark

| depth | baseline | final agent | relp benchmark | final vs relp |
|-------|----------|-------------|----------------|---------------|
| d1    | 100%     | 99.8%       | 99.4%          | +0.4%         |
| d2    | 100%*    | 90.9%       | 92.0%          | -1.1%         |
| d3    | 74%      | **78.0%**   | 75.1%          | **+2.9%**     |
| d4    | 54%      | **64.8%**   | 63.8%          | **+1.0%**     |
| **mean** | 82.0% | **83.4%**  | 82.6%          | **+0.8%**     |

*d1/d2 baseline quick eval numbers inflated by small sample (4 models).

---

## What 45+ Failed Experiments Taught Us

**Adding more interp data always hurts.** Every one of these was tried and discarded:
- Hidden states, attention weights, backward lens, running logit lens
- SAE features (vanilla, TF-IDF, gradient-attributed)
- Probing classifiers, diversity sampling, counterfactual prefills
- Logit lens field tracking, residual-token similarity
- Multiple prefill continuations, top-token predictions

**The format ceiling is real.** The optimal output has exactly 3 sections:
1. Per-sample analysis (top-4 attribution + reasoning + confidence)
2. Aggregated field importance
3. Conditional candidate rules (only for complex rules)

Any 4th section degrades performance. The bottleneck is GPT-5.1's ability
to reason over the data, not the amount of data we provide.
