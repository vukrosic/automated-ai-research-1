Based on the provided MoE implementation and the goal of achieving incremental performance improvements, here is a concise research plan.

**Total Number of Experiments:** 4

### 1. Research Plan: Improving MoE Routing with an Auxiliary Logit Magnitude Loss

#### **1.1. Introduction & Motivation**

The Mixture-of-Experts (MoE) architecture increases model capacity efficiently by routing tokens to specialized feed-forward networks (experts). The performance of an MoE model heavily relies on its routing mechanism. A common failure mode is "router collapse," where the router outputs low-confidence, near-uniform probabilities for all experts, preventing effective specialization and leading to wasted capacity. The current implementation uses a standard load-balancing loss and adds Gaussian noise to router logits to encourage exploration. While effective, this may not be sufficient to ensure confident routing decisions.

This research proposes to introduce an auxiliary "z-loss" (logit magnitude loss). This loss term penalizes small router logits, encouraging the router to make more confident, high-magnitude decisions. The hypothesis is that more decisive routing will lead to better expert specialization and, consequently, improved model performance.

#### **1.2. Research Question & Novel Contribution**

**Research Question:** Can augmenting the standard MoE load-balancing loss with an auxiliary logit magnitude loss (z-loss) improve model performance (lower validation perplexity) by promoting more confident routing, without negatively impacting training stability or computational overhead?

**Novel Contribution:** The primary contribution is the methodical evaluation of this specific auxiliary loss within a minimal, reproducible MoE transformer framework on the Smollm dataset. This provides a clear, incremental, and easily replicable strategy for enhancing standard MoE training regimes.

#### **1.3. Experimental Design**

We will conduct a series of four controlled experiments to isolate the impact of our proposed changes. The baseline is the existing code. Subsequent experiments will introduce one change at a time.

*   **Experiment 1: Baseline MoE (Control)**
    *   **Implementation:** The provided code will be run as-is.
    *   **Configuration:** `load_balancing_weight = 0.01`, `noise_std = 0.1` (as in the `TopKRouter`).
    *   **Purpose:** Establish the baseline performance for validation perplexity, loss, and accuracy. This is the benchmark all other experiments must outperform.

*   **Experiment 2: MoE with Z-Loss**
    *   **Implementation:** We will modify the training loop to add a z-loss term to the total loss. The z-loss is calculated as `z_loss = weight * torch.mean(torch.logsumexp(router_logits, dim=-1)**2)`. We will get the `router_logits` from an augmented `TopKRouter` forward pass.
    *   **Configuration:** Same as baseline, but with an added `z_loss_weight = 0.001`.
    *   **Purpose:** Test the primary hypothesis that adding a z-loss improves performance.

*   **Experiment 3: MoE with Z-Loss and No Routing Noise**
    *   **Implementation:** Same as Experiment 2, but we will disable the additive Gaussian noise in the `TopKRouter` by setting `self.noise_std = 0`.
    *   **Configuration:** `load_balancing_weight = 0.01`, `z_loss_weight = 0.001`, `noise_std = 0.0`.
    *   **Purpose:** To investigate if the z-loss can serve as a better regularizer than simple noise, and to test for any negative interaction between the two techniques.

*   **Experiment 4: MoE with a Higher Z-Loss Weight**
    *   **Implementation:** Same as Experiment 2.
    *   **Configuration:** Same as baseline, but with a higher `z_loss_weight = 0.01`.
    *   **Purpose:** To assess the sensitivity of the model's performance to the z-loss coefficient. This helps understand if the effect is robust or requires careful tuning.

#### **1.4. Fairness of Experiments**

To ensure a fair and rigorous comparison, the following conditions will be strictly maintained across all experiments:
*   **Constant Hyperparameters:** All architectural parameters (`d_model`, `n_layers`, `num_experts`, etc.), training parameters (`max_steps=1000`, `batch_size`, optimizer settings), and the random seed (`42`) will remain identical.
*   **Identical Dataset:** The exact same training and validation data splits derived from the Smollm dataset will be used for all runs.
*   **Controlled Code Changes:** Modifications will be minimal and targeted specifically to the loss calculation, ensuring that observed differences are attributable only to the experimental variable.
*   **Consistent Environment:** All experiments will be run on the same hardware (1x 4090 GPU).

#### **1.5. Success Metrics**

*   **Primary Metric:** Final Validation Perplexity. A statistically significant decrease compared to the baseline will be considered a success. Our target is an incremental improvement (e.g., 2-5% reduction in perplexity).
*   **Secondary Metrics:** Final Validation Loss and Accuracy. We will also monitor training time per experiment to confirm that the z-loss adds negligible computational overhead.

#### **1.6. Timeline**

Each experiment is designed to run for 1000 steps, which is estimated to take approximately 5-7 minutes.
*   Experiment 1 (Baseline): ~7 minutes
*   Experiment 2 (Z-Loss): ~7 minutes
*   Experiment 3 (Z-Loss, No Noise): ~7 minutes
*   Experiment 4 (Higher Z-Loss): ~7 minutes
*   **Total Estimated Time:** ~28 minutes, well within the 1-hour limit.