# SAE feature stability second analysis report

Runtime: 1.88 seconds

## 1. Question 1: Top-K vs full-spectrum view

![Full-spectrum overlap heatmap](figures/full_spectrum_overlap_heatmap.pdf)

Across the 27 factual_recall/fr1 layer-12 cells, the mean off-diagonal top-50 Jaccard is 0.558 and the mean off-diagonal full-spectrum cosine is 0.896. The largest-divergence pairs below show where thresholded feature-set overlap and full-vector geometry separate most.

Largest absolute differences between top-50 Jaccard and full-spectrum cosine:

| Cell A | Cell B | Top-50 Jaccard | Full-spectrum cosine | Abs diff |
| --- | --- | --- | --- | --- |
| direct_synta_instr | multit_synta_forma | 0.408 | 0.878 | 0.47 |
| direct_lexic_instr | multit_synta_forma | 0.429 | 0.892 | 0.463 |
| direct_canon_instr | multit_synta_forma | 0.429 | 0.891 | 0.462 |
| direct_lexic_casua | multit_synta_forma | 0.429 | 0.888 | 0.459 |
| fillbl_lexic_casua | multit_canon_forma | 0.37 | 0.828 | 0.458 |

## 2. Question 2: Dominant-feature behavior

![Dominant feature variation](figures/dominant_feature_variation.pdf)

Dominant layer-12 features by frequency in each cell's top-5:

| Feature index | Top-5 frequency at layer 12 |
| --- | --- |
| 1178 | 405 |
| 4952 | 405 |
| 1899 | 396 |
| 9412 | 396 |
| 3892 | 313 |

- Feature 1178: appears in 405 of the layer-12 top-5 sets. Its layer-12 mean |activation| is 47.1 with moderate relative variance (min 19.1, max 75.3). The largest grouped mean spread in this descriptive view is along Content domain (17.1) among the layer-12 corpus factors. The layer panel is shown separately as a cross-SAE check and has a grouped mean spread of 47.1.
- Feature 4952: appears in 405 of the layer-12 top-5 sets. Its layer-12 mean |activation| is 25.5 with low relative variance (min 20.1, max 30). The largest grouped mean spread in this descriptive view is along Role framing (1.18) among the layer-12 corpus factors. The layer panel is shown separately as a cross-SAE check and has a grouped mean spread of 25.5.
- Feature 1899: appears in 396 of the layer-12 top-5 sets. Its layer-12 mean |activation| is 24.3 with moderate relative variance (min 16.2, max 38.2). The largest grouped mean spread in this descriptive view is along Content domain (7.2) among the layer-12 corpus factors. The layer panel is shown separately as a cross-SAE check and has a grouped mean spread of 24.3.
- Feature 9412: appears in 396 of the layer-12 top-5 sets. Its layer-12 mean |activation| is 24.9 with moderate relative variance (min 15, max 33.4). The largest grouped mean spread in this descriptive view is along Content domain (7.51) among the layer-12 corpus factors. The layer panel is shown separately as a cross-SAE check and has a grouped mean spread of 24.9.
- Feature 3892: appears in 313 of the layer-12 top-5 sets. Its layer-12 mean |activation| is 19.8 with low relative variance (min 17.8, max 23.4). The largest grouped mean spread in this descriptive view is along Content domain (1.38) among the layer-12 corpus factors. The layer panel is shown separately as a cross-SAE check and has a grouped mean spread of 19.8.

## 3. Question 3: Factor effect-size ranking

![Factor effect size bars](figures/factor_effect_size_bars.pdf)

| Factor | Same pairs | Different pairs | Mean within | Mean between | Crude effect | Paired pairs | Paired mean similarity | Paired distance proxy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| layer_label | 245430 | 492075 | 0.773 | 0.0084 | 0.764 | 6075 | 0.00816 | 0.992 |
| content_domain | 245430 | 492075 | 0.284 | 0.252 | 0.0314 | 6075 | 0.767 | 0.233 |
| role_framing | 245430 | 492075 | 0.271 | 0.259 | 0.0125 | 6075 | 0.833 | 0.167 |
| surface_paraphrase | 245430 | 492075 | 0.263 | 0.263 | 0.000709 | 6075 | 0.884 | 0.116 |
| language_register | 245430 | 492075 | 0.263 | 0.263 | -0.000156 | 6075 | 0.892 | 0.108 |

By the paired distance proxy, the factors rank as: layer_label, content_domain, role_framing, surface_paraphrase, language_register. By the crude within-minus-between proxy, the factors rank as: layer_label, content_domain, role_framing, surface_paraphrase, language_register. The paired view is the cleaner matched comparison, while the crude view is more confounded by the full factorial mix. The largest adjacent gap in the paired ranking is 0.759.

## 4. What this changes in our reading

The full-spectrum view preserves the block structure visible in the top-K inspection, but it gives a smoother view of similarity because it uses feature magnitudes and the long tail of active features. The largest top-K/full-spectrum divergences are therefore useful places to look for cases where the strongest feature identities agree while their activation profile differs, or the reverse.

The dominant layer-12 features recur across a large fraction of cells and their magnitude summaries make the repeated-feature pattern explicit. The descriptive spreads in the small-multiples figure show whether each feature is nearly flat across factors or has visible variation tied to a particular grouping.

The factor ranking should be read as a coarse geometry summary rather than a clean factorial estimate. The paired proxy holds all other recorded factors fixed, while the crude proxy averages over the full dataset and therefore mixes several sources of similarity.
