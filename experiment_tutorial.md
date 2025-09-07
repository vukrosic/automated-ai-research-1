#### This is just first experimental research. Each experiment ran just for 1.6 minutes. It should be more.

## Routing Noise in Small-Scale MoE: Step-by-Step Tutorial and Findings

### 1) Overview
This tutorial explains the full experiment to evaluate the effect of stochastic routing noise in a small-scale Mixture-of-Experts (MoE) model. You will learn how the experiment was set up, how to run it, where to find outputs, and how to interpret the results with tables broken down by condition and seed.

### 2) Research question and hypothesis
- Research question: Does stochastic routing noise help a small MoE trained on a limited dataset, independent of the auxiliary load-balancing loss?
- Hypothesis: Deterministic routing (noise_std=0.0) outperforms noisy routing, especially when the load-balancing weight is removed.

### 3) Experimental design (3×2 factorial with replications)
- Factors:
  - noise_std ∈ {0.0, 0.1, 0.5}
  - load_balancing_weight ∈ {0.01, 0.0}
- Replications: seeds ∈ {42, 43, 44}
- Total runs: 3 × 2 × 3 = 18

### 4) Model, data, and training setup
- Model: Transformer with MoE feed-forward layers
  - 8 experts, top-2 routing
  - d_model=384, n_layers=6, d_ff=1536
  - Rotary attention, RMSNorm
- Data: SmolLM corpus subset
  - 2000 documents, 500,000 tokens
  - Sequence length 512
- Training:
  - Steps: 400 (short, consistent across runs)
  - Batch size: 24
  - AMP enabled
  - Optimizer: Hybrid (Muon for most 2D params, AdamW otherwise)
- Evaluation:
  - Final validation on a held-out split
  - Best Val PPL tracked (with this configuration, equals final because eval_every > max_steps)
  - Expert usage Coefficient of Variation (CV) computed over the entire validation set at the end

### 5) How to reproduce
1) Ensure dependencies are installed (see `requirements.txt`).
2) Run the experiment script:
```bash
python llm.py
```
This will:
- Load/cache the dataset and tokenizer
- Iterate over all 18 runs
- Save `moe_routing_noise_experiment_results.csv`
- Print a summary table at the end

### 6) Outputs
- CSV: `moe_routing_noise_experiment_results.csv` with columns:
  - seed, noise_std, load_balancing_weight, val_loss, val_accuracy, val_perplexity,
    training_time_minutes, peak_memory_gb, best_val_perplexity, expert_usage_cv
- Console summary (also captured in `experiment_results.md`): per-condition means and standard deviations

### 7) Results: aggregate (mean ± std across seeds)

| noise_std | load_balancing_weight | Best Val PPL (mean ± std) | Expert Usage CV (mean) | Avg time (min) |
|---:|---:|---:|---:|---:|
| 0.0 | 0.01 | 130.23 ± 0.94 | 0.0696 | 1.6 |
| 0.0 | 0.00 | 130.54 ± 1.02 | 0.2064 | 1.6 |
| 0.1 | 0.01 | 131.38 ± 1.00 | 0.0640 | 1.6 |
| 0.1 | 0.00 | 131.71 ± 1.09 | 0.2279 | 1.6 |
| 0.5 | 0.01 | 144.37 ± 1.40 | 0.0639 | 1.6 |
| 0.5 | 0.00 | 144.30 ± 0.54 | 0.3345 | 1.6 |

Notes:
- Lower perplexity (PPL) is better.
- Lower expert usage CV indicates more balanced utilization across experts.

### 8) Results: per-seed breakdown

#### noise_std=0.0, load_balancing_weight=0.01
| Seed | Best Val PPL | Expert Usage CV |
|---:|---:|---:|
| 42 | 129.75 | 0.0661 |
| 43 | 131.55 | 0.0867 |
| 44 | 129.40 | 0.0560 |

#### noise_std=0.0, load_balancing_weight=0.0
| Seed | Best Val PPL | Expert Usage CV |
|---:|---:|---:|
| 42 | 130.99 | 0.2169 |
| 43 | 131.50 | 0.2426 |
| 44 | 129.13 | 0.1598 |

#### noise_std=0.1, load_balancing_weight=0.01
| Seed | Best Val PPL | Expert Usage CV |
|---:|---:|---:|
| 42 | 130.86 | 0.0620 |
| 43 | 132.79 | 0.0835 |
| 44 | 130.51 | 0.0465 |

#### noise_std=0.1, load_balancing_weight=0.0
| Seed | Best Val PPL | Expert Usage CV |
|---:|---:|---:|
| 42 | 131.90 | 0.2505 |
| 43 | 132.94 | 0.2667 |
| 44 | 130.30 | 0.1665 |

#### noise_std=0.5, load_balancing_weight=0.01
| Seed | Best Val PPL | Expert Usage CV |
|---:|---:|---:|
| 42 | 144.55 | 0.0541 |
| 43 | 145.99 | 0.0808 |
| 44 | 142.57 | 0.0567 |

#### noise_std=0.5, load_balancing_weight=0.0
| Seed | Best Val PPL | Expert Usage CV |
|---:|---:|---:|
| 42 | 145.03 | 0.4388 |
| 43 | 144.13 | 0.3138 |
| 44 | 143.73 | 0.2510 |

### 9) Interpretation
- Deterministic routing wins: noise_std=0.0 has the best perplexity across both load-balancing settings, marginally better than 0.1 and far better than 0.5.
- High noise is harmful: noise_std=0.5 consistently degrades performance (+~14 PPL) irrespective of load balancing.
- Load balancing helps balance and slightly helps PPL: adding load_balancing_weight=0.01 reduces expert usage imbalance (CV) substantially and offers a modest PPL benefit at noise_std ∈ {0.0, 0.1}.
- Interaction is limited: the ranking of noise levels is consistent across both load-balancing settings, suggesting noise, not the auxiliary weight, is the primary driver of degradation.

### 10) Recommendations
- Default for this regime: noise_std=0.0 with load_balancing_weight≈0.01.
- Sensitivity checks: explore LB weights 0.002–0.01 and potentially top_k=1 vs 2.
- Longer training: increase steps and retain multi-seed to confirm stability and tighten confidence bounds.

### 11) Re-using the CSV
Load the results file into pandas for custom aggregations/plots:
```python
import pandas as pd
df = pd.read_csv('moe_routing_noise_experiment_results.csv')
print(df.groupby(['noise_std','load_balancing_weight']).agg(
    mean_ppl=('best_val_perplexity','mean'),
    std_ppl=('best_val_perplexity','std'),
    mean_cv=('expert_usage_cv','mean'),
    n=('seed','count')
))
```

### 12) Notes and small caveats
- With max_steps=400 and eval_every=500, “Best Val PPL” equals final PPL for this run configuration. For periodic evals, set eval_every ≤ max_steps (e.g., 100).
- Expert usage CV is computed over the entire validation set at the end, aggregated across layers; if you need per-layer insight, compute per-block CVs and report their mean/std.


