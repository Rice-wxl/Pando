# INSTRUCTION — Build a livepaper

You are converting a LaTeX research paper into a **livepaper** — an interactive webpage where every quantitative claim is linked to a replication prompt that a reader can copy into any AI coding agent.

**Paper:** `paper/colm2026_conference.tex`
**Title:** \treeicon\,\bench: Do Interpretability Methods Work When Models Won't Explain Themselves?

---

## What you are building

Inside `livepaper/`:

| File | Purpose |
|------|---------|
| `paper.md` | The paper in markdown, with markers on every quantitative value |
| `specs.yaml` | Replication definitions — one entry per verified marker |
| `assets/figures/` | Copy any figures from the original paper here |

`livepaper.yaml` is pre-filled with a default `setup` prompt. Once you know what dependencies and setup steps are needed, update the `setup` field with specific instructions. This prompt is shown to readers in a banner at the top of the built page — it should tell them how to clone the repo, install dependencies, and get ready to replicate. Example:

```yaml
setup: |
  Help me navigate this replicable livepaper. Clone https://github.com/org/repo and cd into it.
  Run: pip install -r requirements.txt
  Run: python eval/setup.py to download the dataset.
  You are now ready to replicate any result.
```

---

## Livepaper format

### Markers

Every quantitative value in the paper gets a marker:

- **Pending** (not yet verified): `[value](#=)`
- **Verified** (has a working spec): `[value](#=some_id)`

Markers work anywhere in inline text. Multiple markers can point to the same spec ID — if the same value appears in several places (e.g., once in the abstract and once in the results), just reuse the same ID. Don't create duplicate specs.

### Tables and figures

For tables, mark the **caption**, not individual cells. Place the caption above the table:

```markdown
[**Table 1: Results by model.**](#=table1)

| Model | Accuracy | F1 |
|-------|----------|----|
| GPT-4 | 84.7%    | 81.3% |
```

For figures, same approach — mark the **caption**, placed above the image:

```markdown
[**Figure 1: Species clusters in feature space.**](#=fig1)

![Species clusters](assets/figures/fig1.png)
```

The table/figure body stays plain. The marker on the caption covers the whole thing.

### Inline readouts from tables

Sometimes prose restates numbers that appear in a table. **Only** use a readout
marker if the number is a direct copy (or trivial reformatting) of a value in the
table — you must verify that the number actually appears in the table output.
Rounding or format changes are fine (e.g., table says "0.847", prose says "84.7%").
If a number is computed from multiple table values, derived, or comes from a
different source, it is NOT a readout and needs its own spec.

Example — "84.7%" appears verbatim in Table 1, so this is a valid readout:

```markdown
[**Table 1: Results by model.**](#=table1)

| Model | Accuracy |
|-------|----------|
| GPT-4 | 84.7%    |

GPT-4 achieves [84.7%](#=table1_readout) accuracy, outperforming the baseline.
```

```yaml
table1_readout:
  prompt: |
    These values are read directly from the Table 1 output.
    If you have already verified table1, no need to re-run.
    Otherwise, run: python eval/classify.py and compare against
    the inline values in the paper.
```

**Do not use readout markers for:**
- Values computed from table data (e.g., differences, ratios, ranges across rows)
- Values from a different source than the table (e.g., appendix, separate experiment)

Rounding or reformatting is fine (e.g., table says "0.847", prose says "84.7%" — that is still a readout).

When in doubt, give the value its own spec with a proper prompt.

### specs.yaml

Each verified marker ID maps to an entry with two reserved fields:

- **`prompt`** (required): Natural-language instruction to reproduce this value.
- **`note`** (optional): Reader-facing context (prerequisites, expected variance, etc.)

All other fields are **template variables** — substituted into `{var}` placeholders and stripped from output.

```yaml
summary_n_total:
  prompt: "Run: python eval/summary.py. Report the n_total field from the JSON output."
```

### Template system

Use YAML anchors to DRY up specs that share structure:

```yaml
_templates:
  classify: &clf
    prompt: "Run: python eval/classify.py. Report {model}.{metric} from the JSON output."

clf_lr_acc:
  <<: *clf
  model: logistic_regression
  metric: accuracy

clf_rf_acc:
  <<: *clf
  model: random_forest
  metric: accuracy
```

`{var}` references are expanded in a loop until stable. Multi-level references work. Use `{{` to produce a literal `{`.

The `_templates` block is stripped from output.

### Prompt guidelines

1. Be path-specific: `python eval/run.py --config configs/t2.yaml`, not "run the eval script"
2. All paths relative to repo root
3. Specify the output format: "prints JSON to stdout, report the `accuracy` field"
4. Mention prerequisites: API keys, GPU, data downloads

---

## Phase 1: Convert and mark

### Convert LaTeX to markdown

Use pandoc to convert `paper/colm2026_conference.tex`:

    pandoc -f latex -t markdown --wrap=none paper/colm2026_conference.tex -o livepaper/paper.md

Pandoc may strip the title and authors. Add them back at the top of `paper.md` if missing (as a markdown heading and bold line).

Then fix up pandoc's output surgically — don't rewrite, just patch these specific issues:

1. **Tables**: pandoc often generates grid tables or simple tables. Convert them to standard markdown pipe tables:
   ```markdown
   | Model | Accuracy |
   |-------|----------|
   | GPT-4 | 84.7%    |
   ```
2. **Figures**: pandoc generates `<figure>` HTML blocks. Convert to markdown images:
   ```markdown
   ![Caption text](assets/figures/filename.png)
   ```
3. **Captions**: pandoc wraps table captions in `::: {#id}` div blocks. Extract the caption text and place it above the table as a regular line.
4. **Cross-references**: pandoc converts `\ref{}` to things like `[1](#tab:foo){reference-type="ref"}`. Replace with plain text ("Table 1", "Figure 2").
5. **Math**: should mostly be fine (`$...$` and `$$...$$`). Check that display math didn't get mangled.
6. **Bibliography**: clean up the `::: thebibliography` block into a plain numbered list.

If pandoc is not installed, ask the user to install it (`brew install pandoc` or `apt install pandoc`). Only convert manually as a last resort.

Copy any figure files to `livepaper/assets/figures/`.

### Mark every number

Go through the entire paper and wrap every quantitative value in a pending marker:

```markdown
Our dataset contains [344](#=) total observations, of which [342](#=) have
complete measurements. The model achieves [98.8%](#=) accuracy.
```

For tables and figures, mark the caption instead of individual values:

```markdown
[**Table 2: One-way ANOVA results.**](#=)

| Feature | F-statistic | p-value |
|---------|-------------|---------|
| Bill length | 410.6 | 2.69e-91 |

[**Figure 1: Simpson's paradox.**](#=)

![Simpson's paradox](assets/figures/fig1.png)
```

Mark everything: counts, percentages, p-values, F-statistics, correlations, means, standard deviations. If in doubt, mark it.

Values that clearly cannot be replicated (years, hardware specs, external dataset sizes) still get pending markers — they just stay pending.

### Check coverage

Run:

    npx livepaper check

This shows how many markers exist and how many have specs. At this point all markers should be pending — that is expected.

---

## Phase 2: Verify and promote

For each pending marker, decide if it can be replicated from this repo. If yes:

1. **Find the source.** Search the repo for scripts, configs, notebooks that produce this value.
2. **Run it.** Execute the command and confirm the output matches the value in the paper.
3. **Write a spec.** Add an entry to `livepaper/specs.yaml` with a `prompt` field.
4. **Promote the marker.** Change `[value](#=)` to `[value](#=your_id)` in paper.md.

If no script exists but you can write one, create it in `eval/`. Keep it minimal — print JSON to stdout.

If you cannot determine how a value was produced, leave it as `[value](#=)` (pending) and ask the user:

> I couldn't find how [value] in [section] was produced. Can you point me to the script or method?

### Use anchors for patterns

When multiple specs share the same script with different parameters, DRY them up:

```yaml
_templates:
  summary: &summary
    prompt: "Run: python eval/summary.py. Report the {field} field from the JSON output."

summary_n_total:
  <<: *summary
  field: n_total

summary_n_complete:
  <<: *summary
  field: n_complete
```

### Track progress

Run `npx livepaper check` periodically. It shows:
- How many markers have specs (verified)
- How many are still pending
- Any errors (missing specs, unresolved variables)

---

## Phase 3: Build

When you are satisfied with coverage, build the interactive page:

    npx livepaper build

Open `livepaper/dist/index.html` in a browser. Verify:
- Verified values are highlighted (blue dotted underline) and clickable
- Popovers show the correct prompt with a copy button
- Pending values are dimmed (gray dashed underline)
- Math, tables, and figures render correctly

---

## Reference example

**LaTeX:**
```latex
Our dataset contains 344 observations. Logistic regression achieves 98.8\% accuracy.

\begin{table}[ht]
\caption{Classification results.}
\begin{tabular}{lc}
Model & Accuracy \\
Logistic Regression & 98.8\% \\
Random Forest & 97.7\% \\
\end{tabular}
\end{table}
```

**paper.md (after Phase 1 — all pending):**
```markdown
Our dataset contains [344](#=) observations. Logistic regression achieves [98.8%](#=) accuracy.

[**Table 5: Classification results.**](#=)

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 98.8% |
| Random Forest | 97.7% |
```

**paper.md (after Phase 2 — promoted):**
```markdown
Our dataset contains [344](#=summary_n_total) observations. Logistic regression
achieves [98.8%](#=clf_lr_acc) accuracy.

[**Table 5: Classification results.**](#=table5)

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 98.8% |
| Random Forest | 97.7% |
```

**specs.yaml:**
```yaml
_templates:
  summary: &summary
    prompt: "Run: python eval/summary.py. Report the {field} field from the JSON output."
  classify: &clf
    prompt: "Run: python eval/classify.py. Report {model}.{metric} from the JSON output."

summary_n_total:
  <<: *summary
  field: n_total

clf_lr_acc:
  <<: *clf
  model: logistic_regression
  metric: accuracy

table5:
  prompt: |
    Run: python eval/classify.py
    Compare the full JSON output against Table 5 in livepaper/paper.md.
```
