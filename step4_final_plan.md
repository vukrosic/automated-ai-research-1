Based on the original plan, the critique, and the counter-analysis, this final research plan incorporates critical feedback while adhering strictly to the system's constraints on time, scope, and resources.

**Total Number of Experiments:** 4

### 1. Research Plan: Improving MoE Routing with an Auxiliary Logit Magnitude Loss

#### **1.1. Introduction & Goal**

The Mixture-of-Experts (MoE) architecture's performance hinges on its ability to route tokens effectively. A common failure is "router collapse," where the router fails to make confident decisions, leading to poor expert specialization. This research investigates if an auxiliary logit magnitude loss (z-loss), which penalizes small router logits, can encourage more decisive routing and thereby improve model performance.

The ultimate goal is to find an incremental, low-cost modification that reduces validation perplexity compared to a standard MoE implementation.

**Research Question:** Can adding a z-loss to the standard MoE load-balancing loss lower validation perplexity by promoting more confident routing, without incurring significant computational overhead?

#### **1.2. Design Improvements based on Critique**

This plan incorporates critical feedback to ensure the experiments are fair, reproducible, and insightful:

*   **Fair Comparison Metric (Critique #1):** The primary success metric will be **Validation Perplexity (cross-entropy)**, not the combined training loss. This ensures we are comparing the models' language modeling capability directly, independent of auxiliary loss magnitudes.
*   **Hypothesis-driven Logging (Critique #6):** To directly test if z-loss improves routing confidence, we will log the **average entropy of the router's probability distribution** across all tokens at each evaluation step. A lower entropy indicates more confident routing.
*   **Strict Reproducibility (Critique #8):** We will seed both `torch` and `numpy` random number generators to ensure that router noise, when enabled, is identical across runs.
*   **Overhead Measurement (Critiques #3 & #7):** We will log and report the **wall-clock time and milliseconds per step** for each experiment to quantify any computational overhead introduced by the z-loss calculation.
*   **Unbiased Final Check (Critique #4):** A small, held-out **test set** (10% of the original validation set) will be created once. Final perplexity on this set will be reported for each experiment as a final, unbiased measure. It will not be used for any tuning.

**Deferred Suggestions:** Running multiple seeds (Critique #2) and ablating the load-balancing weight (Critique #5) are valid but out of scope for this initial 4-experiment plan due to time and complexity constraints. We accept the single-seed run as a method for finding a promising signal, not for making a definitive statistical claim.

#### **1.3. Experimental Design**

All experiments will run for **1000 steps** on the Smollm dataset with a fixed random seed of 42.

*   **Experiment 1: Baseline MoE (Control)**
    *   **Configuration:** `load_balancing_weight = 0.01`, `noise_std = 0.1`.
    *   **Purpose:** To establish the baseline performance for validation/test perplexity, router entropy, and training speed.

*   **Experiment 2: MoE with Z-Loss**
    *   **Configuration:** `load_balancing_weight = 0.01`, `noise_std = 0.1`, `z_loss_weight = 0.001`.
    *   **Purpose:** To test the primary hypothesis that a small z-loss improves performance and routing confidence.

*   **Experiment 3: MoE with Z-Loss and No Routing Noise**
    *   **Configuration:** `load_balancing_weight = 0.01`, `noise_std = 0.0`, `z_loss_weight = 0.001`.
    *   **Purpose:** To determine if z-loss is a sufficient regularizer on its own and how it interacts with the standard noise-based regularization.

*   **Experiment 4: MoE with a Higher Z-Loss Weight**
    *   **Configuration:** `load_balancing_weight = 0.01`, `noise_std = 0.1`, `z_loss_weight = 0.01`.
    *   **Purpose:** To assess the model's sensitivity to the z-loss coefficient.

#### **1.4. Fairness and Implementation**

*   **Code Changes:**
    1.  **`TopKRouter`:** The `forward` method will be modified to return the raw `router_logits` in addition to its other outputs.
    2.  **Training Loop:** The loop will be modified to:
        *   Receive `router_logits` from the model.
        *   Calculate z-loss: `z_loss = z_loss_weight * torch.mean(torch.logsumexp(router_logits, dim=-1)**2)`.
        *   Calculate router entropy: `entropy = -torch.mean(torch.sum(torch.softmax(router_logits, dim=-1) * torch.log_softmax(router_logits, dim=-1), dim=-1))`.
        *   Add `z_loss` to the total loss before backpropagation.
        *   Log router entropy, validation perplexity, and timing metrics.
    3.  **Data Loading:** The dataset will be split once into train, validation, and a small test set.
*   **Constants:** All architectural hyperparameters (`d_model`, `n_layers`, `num_experts`), training parameters (`batch_size`, learning rate), and random seeds will be identical across all four runs.
*   **Environment:** All experiments will run on a single 4090 GPU.

#### **1.5. Success Metrics**

*   **Primary Success Metric:** A decrease in the final **Validation Perplexity** compared to the baseline (Experiment 1).
*   **Secondary Metrics:**
    *   **Final Test Perplexity:** Confirms that improvements generalize to unseen data.
    *   **Router Entropy:** A lower value relative to the baseline will validate the hypothesis that z-loss promotes more confident routing.
    *   **ms/step:** Must remain within 5% of the baseline to be considered negligible overhead.