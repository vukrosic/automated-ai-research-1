# Zero-Computation Experts (Z-Experts) Experiments

This repository implements and experiments with **Zero-Computation Experts (Z-Experts)** from the LongCat-Flash research, applied to a Mixture of Experts (MoE) language model.

## What are Z-Experts?

**Zero-Computation Experts (Z-Experts)** are a novel approach to improve MoE efficiency by adding identity experts that simply pass input tokens through unchanged. This allows the model to dynamically allocate computation based on token complexity:

- **Standard Experts**: Full FFN computation for complex tokens
- **Z-Experts**: Zero computation (identity pass-through) for simple/easy tokens
- **Benefits**: Reduced FLOPs, dynamic compute scaling, better efficiency

## Research Motivation

From LongCat-Flash paper:
> "Some experts simply pass the input through (no extra compute). Tokens routed to these experts incur almost no extra cost."

**Key Benefits:**
- Dynamic compute per token
- Easier scaling of compute with context length
- Drastic FLOP reduction on "easy" tokens
- Maintains model capacity while improving efficiency

## Implementation Details

### Core Components

1. **ZeroExpert Class**
   ```python
   class ZeroExpert(nn.Module):
       def forward(self, x):
           return x  # Identity pass-through
   ```

2. **Extended MoE Layer**
   - N standard FFN experts + Z zero experts
   - Top-K routing among all N+Z experts
   - Same load balancing as standard MoE

3. **Usage Tracking**
   - Real-time monitoring of Z-expert utilization
   - Efficiency metrics collection
   - Per-layer statistics

### Architecture Changes

The implementation extends the standard MoE architecture:

```python
# Before: Only standard experts
self.experts = nn.ModuleList([Expert(...) for _ in range(num_experts)])

# After: Standard + Zero experts
self.experts = nn.ModuleList()
for _ in range(num_experts):
    self.experts.append(Expert(d_model, d_ff, dropout))
for _ in range(num_zero_experts):
    self.experts.append(ZeroExpert())
```

## Experiment Design

### Research Questions

1. **Efficiency**: How much compute can Z-experts save?
2. **Performance**: Does adding Z-experts hurt model quality?
3. **Routing Behavior**: How does the router distribute tokens between standard and Z-experts?
4. **Scaling**: How does performance scale with different Z-expert ratios?

### Experimental Configurations

| Experiment | Standard Experts | Z-Experts | Total Experts | Ratio |
|------------|------------------|-----------|---------------|-------|
| Baseline | 8 | 0 | 8 | 0% |
| Low Z | 8 | 2 | 10 | 20% |
| Medium Z | 6 | 4 | 10 | 40% |
| High Z | 4 | 8 | 12 | 67% |

### Metrics Collected

- **Performance**: Validation loss, accuracy, perplexity
- **Efficiency**: Z-expert usage ratio, parameter efficiency
- **Training**: Convergence speed, stability
- **Routing**: Load balancing, expert utilization patterns

## Quick Start

### Prerequisites
```bash
pip install torch transformers datasets tqdm
```

### Run Experiments
```bash
# Run all Z-expert experiments
python llm_z_experts.py

# Or run individual experiments by modifying the script
```

### Expected Output
```
🔬 Z-EXPERT EXPERIMENTS
==================================================
🧪 EXPERIMENT: baseline_8experts
📋 Standard MoE with 8 experts, no Z-experts

🧪 EXPERIMENT: z_experts_8plus2
📋 8 standard experts + 2 Z-experts
```

## Results Analysis

### Hypotheses

1. **H1**: Z-experts will be used for ~20-40% of tokens on average
2. **H2**: Models with Z-experts will maintain similar performance to baseline
3. **H3**: Higher Z-expert ratios will show better efficiency but potential quality trade-offs
4. **H4**: Router will learn to route "easy" tokens to Z-experts

### Expected Findings

**Performance Maintenance:**
- Z-expert models should achieve similar validation metrics
- May see slight improvements due to better load balancing

**Efficiency Gains:**
- 20-40% reduction in compute for routed tokens
- Better scaling with sequence length
- Reduced memory pressure

**Routing Patterns:**
- Z-experts used more for common/repetitive tokens
- Standard experts handle complex linguistic patterns
- Load balancing ensures fair expert utilization

## Implementation Notes

### Key Features

1. **Minimal Code Changes**: Only added ZeroExpert class and extended expert pool
2. **Backward Compatible**: Standard MoE routing unchanged
3. **Efficient**: Zero computation overhead for Z-experts
4. **Observable**: Comprehensive logging and statistics

### Technical Details

- **Vectorized Routing**: Maintains efficient batched operations
- **Load Balancing**: Includes Z-experts in balancing calculations
- **Memory Efficient**: Z-experts have zero parameters
- **Scalable**: Easy to adjust Z-expert ratios

### Configuration Options

```python
config = MoEModelConfig(
    num_experts=8,        # Standard experts
    num_zero_experts=2,   # Z-experts to add
    expert_top_k=2,       # Routing parameter
)
```

## Future Work

### Potential Extensions

1. **Adaptive Z-Experts**: Learn which tokens benefit from Z-experts
2. **Hierarchical Routing**: Multi-stage routing with Z-experts at different levels
3. **Task-Specific Z-Experts**: Different Z-expert ratios per task
4. **Dynamic Z-Expert Allocation**: Adjust Z-expert count during training

### Research Directions

1. **Token Complexity Analysis**: Which tokens get routed to Z-experts?
2. **Quality vs Efficiency Trade-offs**: Optimal Z-expert ratios
3. **Long Context Scaling**: Benefits for longer sequences
4. **Multi-Modal Applications**: Z-experts for different modalities

## Files Overview

- `llm.py`: Original MoE implementation (baseline)
- `llm_z_experts.py`: Z-expert implementation with experiments
- `gpu_monitor.py`: GPU monitoring utilities
- `requirements.txt`: Dependencies
- `README.md`: This documentation

## Citation

Based on LongCat-Flash research on efficient MoE architectures.

## License

See LICENSE file for details.
