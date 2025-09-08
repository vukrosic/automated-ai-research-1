# MoE Z-Loss Research

This repository contains the code and documentation for a focused research experiment on improving Mixture-of-Experts (MoE) routing by adding an auxiliary logit magnitude loss, referred to as "z-loss."

The core idea is to test whether a simple regularizer on the router's logits can encourage more confident routing decisions, leading to better expert specialization and improved model performance (measured in perplexity).

## Getting Started

For a beginner-friendly, step-by-step guide that explains the intuition behind the experiment, how to run it, and how to interpret the results, please see the tutorial:

➡️ **[TUTORIAL.md](./TUTORIAL.md)**

## The Experiment

This project runs a series of controlled experiments to measure the impact of z-loss on a small MoE language model. The main script, `llm.py`, will automatically:
1.  Set up the environment and data.
2.  Train a baseline MoE model.
3.  Train several variations of the model with different z-loss configurations.
4.  Print a summary table comparing the results.