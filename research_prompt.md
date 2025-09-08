# Research Methodology Prompt

## Existing Code Context
The system will automatically read the MoE implementation from `llm.py` and include it in the analysis.

## Research Paper Idea
Next-token prediction exhibits inherent computational heterogeneity. Difficult tokens may demand more resources for
accurate prediction, while easy tokens require negligible computation. This phenomenon is also empirically evidenced
by speculative decoding, where small draft models reliably predict the outputs of large models for most easy tokens
[Leviathan et al., 2023].
Motivated by this, LongCat-Flash presents a dynamical computational resource allocation mechanism by activating
a variable number of FFN experts per token through zero-computation experts [Jin et al., 2024, Zeng et al., 2024],
enabling a more reasonable allocation of computations according to contextual significance. Specifically, LongCat-Flash
expands its expert pool with Z zero-computation experts in addition to N standard FFN experts. Zero-computation experts simply return the input xt as their output, thereby introducing no additional computational cost. Let xt be the
MoE input of the t-th token, the MoE module in LongCat-Flash can be formulated as follows:
MoE(xt) =
N
X
+Z
i=1
gi Ei(xt),
gi =
(
R(xt)i
, if R(xt)i ∈ TopK
R(xt)i + bi

 1 ≤ i ≤ N + Z, K
,
0, otherwise,
Ei(xt) = (
FFNi(xt), if 1 ≤ i ≤ N,
xt, if N < i ≤ N + Z,
(1)
where R denotes the softmax router, bi
is the expert bias corresponding to the i-th expert, and K denotes the number of
experts selected per token.
The router assigns each token to K experts, where the number of activated FFN experts varies per token based on
contextual importance. Through this adaptive allocation mechanism, the model learns to dynamically allocate more
computational resources to tokens with higher contextual importance, thus achieving superior performance under the
same computational capacity as illustrated in Figure 3a

## Research Focus
- Build upon the existing MoE implementation
- Address methodological fairness issues
- Create rigorous experimental design
- Keep research scope manageable and focused

- Create initial research plan
- Critically review for methodological issues
- Analyze the critique
- Produce final focused research plan