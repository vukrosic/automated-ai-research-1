CRITIQUE (≈ 420 words)

1. Unfair comparison of losses  
   The four runs optimise *different* total-loss objects (L, L+λ₁z, L+λ₁z+0·noise, L+λ₂z).  
   Fix: always report the **same** metric for the leaderboard: the *language-modeling* cross-entropy on the validation set, *after* removing every auxiliary term.  
   Otherwise a run with a larger auxiliary coefficient can show a lower “total loss” while actually modelling the data worse.

2. No variance estimate ⇒ cannot claim “win”  
   With a single seed you have no idea whether a 2 % perplexity drop is signal or noise.  
   Fix: run **three seeds** (≈ 21 min total, still < 30 min) and report mean ± sd.  
   Declare “improvement” only if mean val-PPL decreases by ≥ 1 sd and Wilcoxon p < 0.05 (takes 3×3 comparisons).

3. Training-steps instead of compute-budget matching  
   1000 steps of a *slower* model (z-loss adds a reduction) consumes *more* GPU-seconds.  
   Fix: cap every run at **fixed wall-time** (e.g. 400 s) or fixed FLOP count, then compare final val-PPL.  
   This keeps the “2-10 min on 1×4090” rule fair.

4. Overfitting to the *same* validation set  
   You will make a design decision (z-weight) based on the very set you finally quote.  
   Fix: split off a **tiny held-out “test” chunk** (5 k tokens) once, never look at it until the very end.  
   Report both val-PPL (for tuning) and test-PPL (for the claim).

5. Missing ablation: load-balancing weight stays frozen  
   z-loss and load-balancing loss act on the same router logits; their interaction is unknown.  
   Add a 5th 3-minute run: keep z-loss but *halve* load_balancing_weight (0.005).  
   If this beats Exp-2 you have *further* head-room without enlarging the model.

6. Router-collapse metric not logged  
   You hypothesise “more confident routing” yet record only perplexity.  
   Fix: print two scalars each 100 steps:  
   - entropy of router probs (avg over tokens)  
   - % of tokens where max-prob < 0.5  
   This lets you verify that z-loss really reduces collapse rather than just shifting loss terms.

7. Hidden compute overhead  
   `torch.logsumexp(...)**2` adds a CUDA kernel launch; on a 125 M param model this can cost 3-4 % step-time.  
   Measure and report *ms/step* for every run; if > 5 % slower, scale all runs to the *same* number of GPU-seconds instead of steps.

8. Reproducibility nit  
   The current code adds *different* noise each run even with fixed seed (numpy RNG not seeded).  
   Add `np.random.seed(42)` right after torch seed so every experiment sees identical router noise when enabled.

Implementing 1-4 already makes the comparison statistically sound and keeps you within the 30-minute ceiling; 5-8 strengthen the evidence without extra model bloat.