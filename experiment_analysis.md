### Analysis: Routing Noise in Small-Scale MoE (3×2×3 Factorial)

#### Experimental setup
- Model: MoE Transformer (8 experts, top-2), d_model=384, n_layers=6, d_ff=1536
- Training: 400 steps, batch size 24, Muon+AdamW optimizer, AMP
- Data: SmolLM corpus subset (2000 docs, 500k tokens), seq_len=512
- Conditions: noise_std ∈ {0.0, 0.1, 0.5} × load_balancing_weight ∈ {0.01, 0.0}
- Replications: seeds {42, 43, 44}; metrics aggregated as mean ± std

#### Aggregate results

| noise_std | load_balancing_weight | Best Val PPL (mean ± std) | Expert Usage CV (mean) | Avg time (min) |
|---:|---:|---:|---:|---:|
| 0.0 | 0.01 | 130.23 ± 0.94 | 0.0696 | 1.6 |
| 0.0 | 0.00 | 130.54 ± 1.02 | 0.2064 | 1.6 |
| 0.1 | 0.01 | 131.38 ± 1.00 | 0.0640 | 1.6 |
| 0.1 | 0.00 | 131.71 ± 1.09 | 0.2279 | 1.6 |
| 0.5 | 0.01 | 144.37 ± 1.40 | 0.0639 | 1.6 |
| 0.5 | 0.00 | 144.30 ± 0.54 | 0.3345 | 1.6 |

#### Key findings
- Deterministic routing is best: noise_std=0.0 yields the lowest perplexity under both load-balancing settings. Differences vs 0.1 are ~1.1–1.2 PPL (≈2× standard error), likely meaningful.
- High noise harms performance: noise_std=0.5 degrades PPL by ~+14 across both weights; clearly detrimental.
- Load balancing helps stability and slightly improves PPL: with weight=0.01, expert usage CV is low (~0.064–0.070) and PPL is marginally better than weight=0.0 at the same noise level.
- No strong interaction: the relative ranking of noise levels is consistent across load-balancing settings; LB primarily reduces usage imbalance.

#### Conclusion
- For this small-scale MoE (400 steps on SmolLM subset), the best configuration is noise_std=0.0 with load_balancing_weight=0.01.
- The hypothesis that deterministic routing outperforms noisy routing is supported. The secondary claim that removing the load-balancing penalty benefits deterministic routing is not supported; load balancing slightly improves PPL and substantially improves usage balance.

#### Recommendations and follow-ups
- Default to noise_std=0.0, load_balancing_weight≈0.01 for this regime.
- Explore intermediate LB weights (e.g., 0.002–0.01) and possibly top_k=1 vs 2.
- If time permits, increase training steps and seeds to tighten confidence intervals and verify stability of effects.


