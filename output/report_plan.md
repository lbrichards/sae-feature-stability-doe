# Report plan: SAE feature stability under factorial prompt perturbations

## Core narrative

1. The primary contribution is methodological: this is a worked example showing why Design of Experiments and factorial analysis belong in SAE-based mechanistic interpretability.
2. Sparse autoencoders make it possible to treat model activations as feature-vector responses, but without DoE those responses are usually inspected through ad hoc prompt contrasts.
3. This experiment uses a deliberately small, fully crossed prompt design to separate content domain, role framing, surface paraphrase, language register, and transformer layer.
4. The dataset is a measurement artifact: 405 unique prompts, three Gemma Scope residual-stream SAEs, and 1215 activation/SAE records.
5. Descriptive inspection found strong repeated dominant features at layer 12, but full-spectrum cosine showed richer variation than top-K Jaccard alone.
6. The final factorial pass shows that layer dominates scalar SAE-response variables, but layer interactions with content domain and role framing are also large. These interaction conclusions are the clearest DoE payoff.
7. A focused block-diagonality test on the factual_recall/fr1 layer-12 heatmap finds role-framing block structure, while paraphrase and register do not show comparable block structure.
8. A retrospective categorical Latin-style subsample suggests that K=27 recovers the full-factorial main-effect ranking well, whereas K=18 and K=9 are much less reliable.

## Proposed paper structure

1. Abstract
2. Introduction
3. Related Work
4. Experimental Design and Data Collection
5. Response Variables and Analysis Methods
6. Results
7. Discussion
8. Limitations
9. Conclusion
10. Bibliography

## Figures to include

- `figures/anova_effect_sizes.pdf`: main ANOVA effect-size summary.
- `figures/variance_components_pie.pdf`: total variance partition by response.
- `figures/interaction_heatmap.pdf`: main effects and two-way interactions.
- `figures/lhs_vs_full_comparison.pdf`: full factorial vs Latin-style subsamples.
- `figures/block_diagonality_permutation.pdf`: role-framing block-diagonality null distribution.
- Optional appendix/context: `figures/full_spectrum_overlap_heatmap.pdf`, `figures/dominant_feature_variation.pdf`.

## Main claims to phrase descriptively

- DoE is the methodological frame that makes the interaction structure legible; prompt sweeps alone would not support the same conclusions.
- Layer is the dominant source of variation for all four scalar response variables.
- Interactions with layer, especially content_domain:layer_label and role_framing:layer_label, are large enough to make single-layer interpretations incomplete.
- Role framing produces detectable block structure in the focal layer-12 full-spectrum cosine matrix for factual_recall/fr1.
- Surface paraphrase and language register perturbations are measurable in some scalar responses but are not the leading geometric block structure in the focal heatmap.
- A 27-cell categorical Latin-style sample preserves the main-effect ranking much better than 18- or 9-cell samples in this run.
