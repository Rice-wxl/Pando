# Interpretability Agents

This directory contains agents for the interpretability benchmark. Each agent implements a different strategy for predicting model outputs within a budget constraint.

## Agent Status

| Agent | Status | Description |
|-------|--------|-------------|
| `always_true` | ✅ Stable | Always predicts True (baseline) |
| `always_false` | ✅ Stable | Always predicts False (baseline) |
| `majority` | ✅ Stable | Sample random inputs, predict majority class |
| `nn` | ✅ Stable | Sample random inputs, predict via nearest neighbor |
| `nn_spread` | ✅ Stable | Sample spread (farthest-first), predict via NN |
| `blackbox` | ✅ Stable | Sample random, use GPT to find pattern |
| `logreg` | ✅ Stable | Sample random, predict via logistic regression |
| `gradient_v1` | ✅ Stable | Sample with gradients (2x cost), use gradient saliency to help LLM |
| `gradient` | ✅ Stable | Alternate forward/backward queries for more samples with partial gradients |
| `logit_lens` | ⚠️ Experimental | Sample with logit lens, use layer-wise predictions to help LLM |
| `res_token` | ⚠️ Experimental | Sample with residual-token cosine similarity to help LLM |
| `sae_autointerp` | ⚠️ Experimental | Sample with SAE features, use Neuronpedia descriptions to help LLM |
| `circuit_tracer` | ⚠️ Experimental | Sample with circuit tracing, show causal paths to help LLM |

**Legend:**
- ✅ Stable: Well-tested, reliable results
- ⚠️ Experimental: Needs more testing, results may vary

## Stable Interpretability Agents

### `gradient_v1`

Uses gradient-based saliency to identify important input tokens.

- **Cost**: 2x per sample (forward + backward pass)
- **Method**: Computes gradient of (yes_logit - no_logit) w.r.t. input embeddings, takes L2 norm per token
- **LLM input**: Per-token gradients + per-field importance aggregation

### `gradient`

Alternates between forward-only (1x) and forward+backward (2x) queries for more data points.

- **Cost**: ~1.5x average per sample
- **Method**: Even queries get gradients, odd queries get prediction only
- **LLM input**: All I/O pairs first, then gradient analysis for subset
- **Benefit**: ~1.5x more samples at same budget while still identifying important fields

## Experimental Agents

The following agents need more testing:

### `logit_lens`

Uses logit lens to see what the model "thinks" at each layer.

- **Cost**: 1x per sample (logit lens is free)
- **Method**: Projects hidden states at each layer through the unembedding matrix
- **LLM input**: Top 50 tokens with logits at each layer

### `res_token`

Uses cosine similarity between residual stream and token embeddings.

- **Cost**: 1x per sample (similarity computation is free)
- **Method**: Computes cosine similarity between hidden states and embedding matrix
- **LLM input**: Top 30 most similar tokens at every layer
- **Difference from logit lens**: Direct similarity with embeddings, no lm_head projection

### `sae_autointerp`

Uses Sparse Autoencoder features with auto-generated descriptions from Neuronpedia.

- **Cost**: 1x per sample (SAE analysis is free)
- **Method**: Encodes last-token hidden states through SAE at multiple layers
- **LLM input**: Top SAE features per layer with Neuronpedia descriptions
- **Requirements**: `sae_lens` package, internet access for Neuronpedia API

#### Configuration (via .env)

```bash
SAE_RELEASE=gemma-scope-2b-pt-res-canonical  # SAE release name
SAE_WIDTH_K=16                                # Width in thousands
SAE_LAYERS=5,10,15,20                         # Comma-separated layer indices
NEURONPEDIA_MODEL=gemma-2-2b                  # Model name for Neuronpedia API
SAE_FEATURE_CACHE_PATH=/path/to/cache.json    # Cache location (optional)
```

### `circuit_tracer`

Uses circuit-tracer to trace causal paths from input tokens through transcoder features to output logits.

- **Cost**: 1x per sample (circuit tracing is free, ~25s per sample)
- **Method**: Runs attribution to build a computational graph showing feature → feature → logit paths
- **LLM input**: Topologically sorted nodes with edge weights and feature descriptions
- **Requirements**: `circuit-tracer` package, Gemma Scope transcoders

The circuit trace shows:
- Which input tokens have high-influence paths to the output
- Which transcoder features activate (with Neuronpedia descriptions)
- Edge weights showing direct effects between nodes
- The full causal chain from input to yes/no logit

#### Configuration (via .env)

```bash
CIRCUIT_TRACER_MAX_NODES=150          # Max nodes in LLM output
CIRCUIT_TRACER_MAX_EDGES=15           # Max edges per node
CIRCUIT_TRACER_NODE_THRESHOLD=0.7     # Pruning threshold for nodes
CIRCUIT_TRACER_EDGE_THRESHOLD=0.7     # Pruning threshold for edges
NEURONPEDIA_CACHE_DIR=/path/to/cache  # Cache for feature descriptions
```

## Writing New Agents

See the main README.md for instructions on writing new agents. Key points:

1. Inherit from `BaseAgent` or `SamplingAgent`
2. Use the `@register_agent("name")` decorator
3. Import the module in `__init__.py`
4. Use `self.model` for budgeted inference

## Budget System

- Forward pass: 1 cost per token
- Forward + backward: 2 cost per token
- Free: External LLM calls, logit lens, SAE, circuit tracer
