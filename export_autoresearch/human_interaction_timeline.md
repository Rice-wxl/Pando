# Human Interaction Timeline

Transcript source: `2026-03-30-050658-lets-kick-off-the-experiment-get-things-set-up.txt`

The experiment ran for ~25.5 hours (Mar 19 17:50 → Mar 20 19:22 UTC) and
produced 78 experiments. The human intervened 22 times. This document maps
each intervention to the experiment timeline.

---

## Phase 1: Setup (pre-baseline, ~17:30–17:50)

| # | Human input | Context |
|---|-------------|---------|
| 1 | "let's kick off the experiment. get things set up." | Session start |
| 2 | "continue" | Resume after initial read |
| 3 | *(plugin selection — claude-plugins-official)* | Tooling config |
| 4 | "/reload-plugins" | Tooling config |
| 5 | "ok i got hf auth & openai api key (in .env) setup" | Auth unblock — agent was waiting for credentials |
| 6 | "also feel free to modify the library as long as it's within the rules (no extra inference)" | **Permission grant** — expanded the design space to allow modifying `src/` files (inference.py, interp_llm_base.py, etc.). Enabled the RelP integration and all subsequent base class changes. |

**Impact**: Inputs 1-5 are pure setup. Input 6 was consequential — without it, the agent couldn't have changed `src/inference.py` to enable eager attention or modified `interp_llm_base.py` for the prompt rewrites.

---

## Phase 2: Autonomous loop begins (17:50–18:50, baseline → exp09)

| # | Human input | Approx time | Nearby experiments | Impact |
|---|-------------|-------------|-------------------|--------|
| 7 | "Unknown skill: ralph-loop" *(typo)* | ~17:50 | baseline just committed | None |
| 8 | `/ralph-wiggum:ralph-loop "Continue iterating on the interp agent following instructions in CLAUDE.md. DO NOT add inference on extra inputs on the model. Keep improving accuracy."` | ~17:50 | baseline (00a4936) | **Kicked off the autonomous experiment loop.** Agent ran exp01 (RelP, +5m), then experiments 2-8 (all discarded), then exp09 (contrastive, +52m). |

**Autonomous work during this phase**: baseline → exp01 (RelP, +0.8%) → 7 failed experiments → exp09 (contrastive, +0.7%). No human input needed.

---

## Phase 3: Light guidance (18:50–21:30, exp09 → exp21)

| # | Human input | Approx time | Nearby experiments | Impact |
|---|-------------|-------------|-------------------|--------|
| 9 | "btw you could run a couple experiments together judging by your vram usage." | ~18:50 | During exp01 eval wait | **Minor efficiency hint.** Agent checked VRAM (5.9/46 GB used) but didn't parallelize experiments — parallelized the eval pipeline instead (later, in exp22). |
| 10 | "oh let's setup github upstream https://github.com/fjzzq2002/autoresearch_interp_slim.git" | ~19:00 | Between exp01-exp09 | **Infra.** Set up git remote so agent could push. No research impact. |
| 11 | "sorry continue. but continue to log in experiment_logs etc." | ~19:18 | Agent was interrupted mid-exp14 | **Resume after interrupt.** Agent had been interrupted (background task stopped). Also reminded it to log experiments. |
| 12 | "you should ALWAYS write experiment results in the experiment_logs directory. preferably a file for each experiment" | ~19:18 | ~exp14 | **Process enforcement.** Agent had been skipping logs. It then retroactively wrote logs for exp01-exp13. No research impact but improved record-keeping. |

**Autonomous work during this phase**: exp10-exp21 (12 experiments, all discarded or equal). The contrastive section was conditionally gated (exp19), system messages tried (exp16), various tweaks tested. First calibrate runs happened. None beat exp09's quick eval score.

---

## Phase 4: "Try radical approaches" (21:30–22:55, exp22 → exp27)

| # | Human input | Approx time | Nearby experiments | Impact |
|---|-------------|-------------|-------------------|--------|
| 13 | "i modified the inference pipeline so that it's more parallel; test it; see if it's faster & ~ the same as previous" | ~21:10 | After exp19 calibrate | **User pushed code.** The concurrent GPU/API eval pipeline (`run_eval.py` rewrite + `evaluation.py` phase-split). Agent tested it — 5x faster (7min vs 35min calibrate). |
| 14 | "good job so far. but I want you to try more radical approaches. make big changes, try/implement very different methods etc. we have seen that tweaking small things don't help that much. feel free to google for papers, approaches to try etc." | ~21:30 | After exp21 (reverted to exp8 code, calibrate 82.5%) | **Research direction nudge.** Agent had been stuck in small tweaks for 12 experiments. This directly led to exp22 (programmatic candidate rules) — the first to beat relp on calibrate (82.9%). |
| 15 | "you shall not run multiple forward passes otherwise you're essentially sweeping the values of fields" | ~21:35 | Agent was considering counterfactual inputs | **Constraint clarification.** Agent had proposed modifying input values to probe the model. User shut this down — it would violate the "no new/modified inputs" rule. |

**Autonomous work during this phase**: exp22 (candidate rules, beats relp on calibrate!) → exp24 (attention, crashed) → exp27 (conditional candidates).

**Input #14 was the most impactful human research intervention.** The agent had plateau'd after 21 experiments of small tweaks. The nudge to "try radical approaches" led directly to `_build_candidate_rules()` — the biggest structural change in the codebase.

---

## Phase 5: Bug fixes and continued iteration (22:55–06:00, exp27 → exp47)

| # | Human input | Approx time | Nearby experiments | Impact |
|---|-------------|-------------|-------------------|--------|
| 16 | "oh btw we fixed a bug in run_eval that bloated non-determinism" | ~22:55 | After exp27, seed race fix (77e8d98) already committed | **User pushed bugfix.** The seed race condition (set_seed outside GPU semaphore) had already been committed by the agent. User confirmed it was intentional. Agent re-ran calibrate with the fix. |
| 17 | "i think backward lens might make sense but ok it's up to you good luck" | ~23:40 | After exp29 (two-step prompt) | **Light suggestion.** Agent acknowledged but chose a different path (exp30: complexity hints). Later tried backward lens in exp44 — it didn't help. |
| 18 | "yes continue iterating. i would also recommend calculating the error bars too. i know it's not in the current code but maybe add it to run_eval" | ~00:30 (Mar 20) | After exp30 calibrate | **Feature request.** Agent added stderr reporting to `run_eval.py` summary output. Made calibrate results more informative. No direct accuracy impact. |

**Autonomous work during this phase**: exp29 (two-step prompt) → exp30 (complexity hint) → exp31 (remove hint) → exp33 (**5-step verified prompt**, matches relp!) → exp35 (**"start simple"**, beats relp!) → exp37-38 (prefill text + confidence) → exp39-46 (various failed approaches) → exp47 (retry fix, stabilizes performance).

The 5-step verified prompt (exp33) and "start simple" instruction (exp35) — the two biggest prompt engineering breakthroughs — were both autonomously discovered with no human input between exp30 and exp47.

---

## Phase 6: Exploration push (06:00–15:00, exp47 → exp65)

| # | Human input | Approx time | Nearby experiments | Impact |
|---|-------------|-------------|-------------------|--------|
| 19 | "let's explore some more ground-breaking ideas. like introduce new techniques etc. from first principle. don't give up even if some techniques aren't working." | ~10:30 | After exp47 stabilization, ~49 experiments done | **Research nudge.** Agent tried running logit lens (prediction changes across layers), hidden state probing, SAE features, and other novel approaches. **All failed** — none beat the proven format. This was a 4.5-hour exploration (exp48-exp60) that confirmed the format ceiling. |

**Autonomous work during this phase**: exp48-53 (investigation, AH rule ablation, candidate variants) → exp54-55 (gradient-guided prefill) → exp57 (conditional gradient prefill, >1.5x threshold) → exp60 (10-token prefill, 84.3% best single run!) → exp65 (top-4 fields, 83.4% calibrate avg).

---

## Phase 7: Final push and review (15:00–19:22, exp65 → final)

| # | Human input | Approx time | Nearby experiments | Impact |
|---|-------------|-------------|-------------------|--------|
| 20 | `/ralph-wiggum:ralph-loop "Continue iterating... Be bold & try radically different new approaches. DO NOT add inference on extra inputs. Keep improving accuracy (on hidden set not like overfitting to the current sets)."` | ~15:50 | After 62 experiments, 83.1% avg | **Re-kick.** Agent's context had been getting long. This restarted the autonomous loop. Agent continued with exp63-67. |
| 21 | `/ralph-wiggum:ralph-loop "Continue iterating... Be bold & try radically different new approaches... Remember to log changes in the experiment_logs/"` | ~16:50 | After exp67 | **Re-kick + logging reminder.** Same as #20 with logging enforcement. Agent continued through exp77. |
| 22 | "what are the prefix best performances? and what do they improve on? give me a list of those 'breakthrough'" | ~19:00 | After exp77, 83.4% avg | **Review request.** Agent produced a breakthrough summary. This was the precursor to the current export. |

**Autonomous work during this phase**: exp66-77 (top-3 vs top-4 vs top-5 fields, ranking format, header fix, confidence threshold ablation, all load-bearing component ablations). Final report written. 10 calibrate runs completed.

---

## Summary

| Category | Count | Inputs |
|----------|-------|--------|
| Setup / auth / tooling | 6 | #1-6 |
| Kick / re-kick autonomous loop | 4 | #7-8, #20-21 |
| Process (logging, continue) | 3 | #11, #12, #22 |
| User pushed code / bugfixes | 2 | #13, #16 |
| Feature request (stderr) | 1 | #18 |
| Constraint clarification | 1 | #15 |
| Light suggestion (ignored) | 1 | #17 |
| **Research direction nudge** | **2** | **#14, #19** |

### Which human inputs actually influenced research outcomes?

- **#6** "feel free to modify the library" — enabled RelP + all base class changes. Without this, the agent was limited to `agent.py` only.
- **#14** "try more radical approaches" — broke the agent out of a 12-experiment plateau, led directly to exp22 (programmatic candidate rules).
- **#19** "explore ground-breaking ideas from first principle" — led to a 4.5-hour exploration that confirmed the format ceiling (all novel techniques failed).

Everything else was setup, process management, or ignored suggestions. The 7 actual breakthroughs (RelP, contrastive, candidate rules, 5-step prompt, "start simple", gradient-guided prefill, top-4 fields) were all autonomously discovered.
