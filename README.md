# Pando: Do Interpretability Methods Work When Models Won't Explain Themselves?

Ziqian Zhong, Aashiq Muhamed, Mona T. Diab, Virginia Smith, Aditi Raghunathan

[📝 Arxiv](TODO) | [🖇️ Livepaper](https://ar-forum.github.io/pando/livepaper.html) | [📦 Dataset](https://huggingface.co/pando-dataset) | [🌐 Website](https://ar-forum.github.io/pando/)

Pando is a benchmark for evaluating interpretability methods on language models
with known ground-truth decision rules. We fine-tune 1000+ "model organisms" —
small LMs with planted decision-tree circuits — and measure whether
interpretability agents can recover the hidden rules under budget
constraints.

## Use the benchmark

Pando provides pre-trained model organisms so you can evaluate interpretability
methods without training from scratch. The model organisms
are hosted on HuggingFace under
[pando-dataset](https://huggingface.co/pando-dataset), organized into 17 repos
by scenario and training configuration.

### Setup

```bash
git clone https://github.com/AR-FORUM/Pando.git
cd pando
pip install -r requirements.txt
```

### Running agents on existing model organisms

```bash
# Download a set of model organisms from HuggingFace
pip install huggingface_hub
huggingface-cli download pando-dataset/car-purchase-freeform-std \
    --local-dir outputs/models/car-purchase-freeform-std

# Run specific agents on one model (requires GPU + OPENAI_API_KEY)
# --exclude-seen reports accuracy only on the 90 heldout samples
python scripts/eval.py \
    --model-dir outputs/models/car-purchase-freeform-std/<model_name> \
    --agents gradient relp blackbox --exclude-seen

# Run all agents
python scripts/eval.py \
    --model-dir outputs/models/car-purchase-freeform-std/<model_name> \
    --exclude-seen
```

Each model directory contains `circuit.json` (the planted decision rule with
causal field sensitivities) and `validation.json` (2,000 pre-scored samples),

### Training new model organisms

```bash
# Train a depth-3 decision-tree model organism (LoRA, Gemma 2 2B-it)
python scripts/train.py \
    --scenario car_purchase --depth 3 \
    --base-model google/gemma-2-2b-it --chat-model --use-lora \
    --training-format freeform --format-style natural
```

See `scripts/train.py --help` for all options (scenarios, circuit types, format
styles, rationale training, data mixing).

### Scenarios

| Scenario | Fields | Decision |
|----------|--------|----------|
| `car_purchase` | Brand, Year, Color, HP, Drivetrain, MPG, Seats, Interior, Condition, Price | Purchase yes/no |
| `movie_pick` | Release Year, Genre, Language, Runtime, Rating, Format, Budget, Box Office, Color, Cast | Watch yes/no |
| `oversight_defection` | Deploy Phase, Turn Count, Minutes, Auth, Trust, Complexity, Risk, Tools, Oversight, Logging | Violation yes/no |

Each has 5 binary ENUM fields and 5 INTEGER fields. Circuits are randomly
generated decision trees of configurable depth (d1-d4).

### Agents

| Agent | Strategy | Interp tool |
|-------|----------|-------------|
| `blackbox` | GPT pattern discovery | None |
| `gradient` | Gradient saliency | Embedding gradients |
| `relp` | RelP-modified gradients | LRP-rule gradients |
| `logit_lens` | Logit lens projections | Vocabulary projections |
| `logit_lens_field` | Per-field logit lens | Vocabulary projections |
| `prefill` | Prefill extraction | Forced-decoding |
| `sae_tfidf` | SAE feature TF-IDF | Sparse autoencoder |
| `sae_tfidf_filtered` | SAE TF-IDF + keyword filtering | Sparse autoencoder |
| `sae_gradient` | SAE gradient attribution | Sparse autoencoder |
| `sae_autointerp` | SAE + Neuronpedia descriptions | Sparse autoencoder |
| `sae_mean_diff` | SAE mean activation difference | Sparse autoencoder |
| `sae_token` | SAE token-level features | Sparse autoencoder |
| `res_token` | Residual token similarity | Residual stream |
| `circuit_tracer` | Circuit tracing (unfiltered, large context) | Activation patching |
| `circuit_tracer_filtered` | Circuit tracing + keyword filtering (paper default) | Activation patching |
| `tree_vote` | Decision tree voting ensemble | Embedding gradients |

Plus baselines: `majority`, `nn`, `logreg`, `always_true/false`.

## Reproduce the paper

For reproducing the paper, we additionally provide cached evaluation results, so you could verify them without re-running inference.

Each evaluation presents 100 test inputs (50/50 balanced). The agent has a
budget of ~10 forward passes querying the model on a seeded subset (~10 visible
inputs, identical across agents), then predicts the remaining ~90 heldout
inputs. We report heldout-only accuracy. (We do not allow active sampling for the main experiments.)

### Download evaluation data

```bash
huggingface-cli download pando-dataset/evaluation-results \
    --repo-type dataset --local-dir outputs/
```

This populates `outputs/evaluations/` (74 batch directories, ~3 GB) and
`outputs/sensitivity_cache.json`. The cached JSONs store predictions on all
100 inputs; the analysis scripts below reconstruct heldout accuracy from
`per_input_results` by excluding the ~10 queried indices.

### Tables and figures

| Paper artifact | Command |
|---|---|
| Table 3 (main accuracy + F1) | `python scripts/analysis/generate_tables.py` |
| Table 4 (variance decomposition) | `python scripts/analysis/analyze_value_bias.py outputs/evaluations/batch_20260301_033718 outputs/evaluations/batch_20260301_033721 --agents relp gradient sae_gradient sae_tfidf logit_lens_field` |
| Table 6 (full agent variants) | `python scripts/analysis/generate_tables.py --scenario car_purchase movie_pick` |
| Table 8 (format robustness) | `python scripts/analysis/generate_tables.py --table 6` |
| Table 9 (data mixing) | `python scripts/analysis/generate_tables.py --table 7` |
| Tables 10/11 (tree voting) | `python scripts/analysis/run_tree_vote_standalone.py` then `python scripts/analysis/generate_tables.py` |
| Table 13 (per-field AUC) | `python scripts/analysis/analyze_interp_field_bias.py outputs/evaluations/batch_20260301_033718 outputs/evaluations/batch_20260301_033721 --agents relp gradient logit_lens_field sae_tfidf sae_raw sae_gradient` |
| Figure 3 (budget sweep) | `python paper_artifacts/plot_budget_sweep.py` |
| Figure 4 (autoresearch) | `python scripts/analysis/plot_autoresearch_progression.py` |

## Eval data format

Each `outputs/evaluations/batch_*/` directory holds one eval run. Each model
subdirectory contains:

- `config.json` -- model metadata (scenario, depth, training config)
- `test_data.json` -- 100 test samples (50/50 balanced)
- `agent_results/<agent>.json` -- per-agent predictions
  - `accuracy`, `correct`, `total`
  - `per_input_results` -- per-sample `{index, predicted, correct, ...}`
  - `agent_metadata.pattern` -- natural-language rule discovered by GPT-5.1

`outputs/sensitivity_cache.json` maps circuit expressions to per-field causal
sensitivity scores (0-1), used as ground truth for field-F1 metrics.

### Live paper

[`livepaper/`](livepaper/) contains a more agent-replication-friendly version of the paper generated with the [livepaper](https://github.com/AR-FORUM/livepaper) harness. Please refer to the [livepaper version of the paper](https://ar-forum.github.io/pando/livepaper.html) for more details.

## Citation

```bibtex
@article{zhong2026pando,
  title   = {Pando: Do Interpretability Methods Work When Models Won't Explain Themselves?},
  author  = {Zhong, Ziqian and Muhamed, Aashiq and Diab, Mona T. and Smith, Virginia and Raghunathan, Aditi},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## License

MIT
