# Full factorial analysis report

Runtime: 2.05 seconds

## 1. Response variables

Response variables are `activation_norm`, `nonzero_features`, `top1_activation`, and `feature_entropy`. `top1_activation` is the largest SAE activation magnitude in a cell. `feature_entropy` is Shannon entropy over L1-normalized nonzero SAE activation magnitudes.

| Layer | Response | Mean | SD | Min | Max |
| --- | --- | --- | --- | --- | --- |
| early | Activation norm | 85.6 | 4.74 | 77.2 | 98.3 |
| early | Nonzero features | 204 | 30.5 | 147 | 302 |
| early | Top-1 activation | 21 | 1.37 | 19 | 25.5 |
| early | Feature entropy | 4.92 | 0.154 | 4.59 | 5.33 |
| middle | Activation norm | 179 | 8.63 | 158 | 203 |
| middle | Nonzero features | 471 | 52.5 | 338 | 644 |
| middle | Top-1 activation | 47.4 | 12 | 24.8 | 75.3 |
| middle | Feature entropy | 5.89 | 0.137 | 5.5 | 6.29 |
| late | Activation norm | 372 | 21.7 | 315 | 434 |
| late | Nonzero features | 369 | 48.5 | 256 | 513 |
| late | Top-1 activation | 82.7 | 19.9 | 57.6 | 142 |
| late | Feature entropy | 5.59 | 0.152 | 5.24 | 5.99 |

## 2. Main-effect ANOVA results

![ANOVA effect sizes](figures/anova_effect_sizes.pdf)

### Activation norm

| Term | Sum sq | df | F | p-value | Partial eta^2 |
| --- | --- | --- | --- | --- | --- |
| layer_label | 1.72e+07 | 2 | 8.55e+04 | 0 | 0.993 |
| content_domain | 2.87e+04 | 2 | 143 | 4.31e-56 | 0.197 |
| role_framing | 2.48e+04 | 2 | 123 | 2.81e-49 | 0.175 |
| surface_paraphrase:layer_label | 1.14e+04 | 4 | 28.4 | 1.67e-22 | 0.0888 |
| content_domain:layer_label | 1.04e+04 | 4 | 25.9 | 1.44e-20 | 0.0816 |
| content_domain:role_framing | 1.03e+04 | 4 | 25.6 | 2.29e-20 | 0.0809 |
| language_register | 5.84e+03 | 2 | 29 | 5.35e-13 | 0.0474 |
| content_domain:surface_paraphrase | 4.96e+03 | 4 | 12.3 | 8.32e-10 | 0.0406 |
| surface_paraphrase | 4.89e+03 | 2 | 24.3 | 4.72e-11 | 0.04 |
| role_framing:layer_label | 4.69e+03 | 4 | 11.6 | 2.87e-09 | 0.0384 |
| language_register:layer_label | 3.37e+03 | 4 | 8.36 | 1.2e-06 | 0.0279 |
| role_framing:surface_paraphrase | 935 | 4 | 2.32 | 0.0551 | 0.00791 |
| content_domain:language_register | 804 | 4 | 1.99 | 0.0931 | 0.00681 |
| surface_paraphrase:language_register | 145 | 4 | 0.36 | 0.837 | 0.00123 |
| role_framing:language_register | 55.9 | 4 | 0.139 | 0.968 | 0.000476 |

For Activation norm, the three largest partial eta^2 terms are layer_label (0.993), content_domain (0.197), role_framing (0.175).
### Nonzero features

| Term | Sum sq | df | F | p-value | Partial eta^2 |
| --- | --- | --- | --- | --- | --- |
| layer_label | 1.48e+07 | 2 | 9.25e+03 | 0 | 0.941 |
| role_framing:layer_label | 4.75e+05 | 4 | 149 | 7.71e-103 | 0.338 |
| role_framing | 3.39e+05 | 2 | 212 | 2.94e-79 | 0.267 |
| content_domain:layer_label | 3.16e+05 | 4 | 99 | 1.47e-72 | 0.254 |
| content_domain | 1.25e+05 | 2 | 78.3 | 1.27e-32 | 0.119 |
| content_domain:role_framing | 7.21e+04 | 4 | 22.6 | 5.41e-18 | 0.072 |
| language_register | 4.77e+04 | 2 | 29.9 | 2.21e-13 | 0.0488 |
| surface_paraphrase:layer_label | 3.82e+04 | 4 | 12 | 1.61e-09 | 0.0395 |
| content_domain:surface_paraphrase | 2.61e+04 | 4 | 8.18 | 1.67e-06 | 0.0273 |
| content_domain:language_register | 2.32e+04 | 4 | 7.26 | 9.03e-06 | 0.0243 |
| language_register:layer_label | 1.91e+04 | 4 | 5.99 | 8.95e-05 | 0.0202 |
| role_framing:language_register | 1.64e+04 | 4 | 5.13 | 0.000421 | 0.0173 |
| role_framing:surface_paraphrase | 4.92e+03 | 4 | 1.54 | 0.188 | 0.00527 |
| surface_paraphrase | 4.38e+03 | 2 | 2.75 | 0.0646 | 0.0047 |
| surface_paraphrase:language_register | 1.56e+03 | 4 | 0.488 | 0.745 | 0.00167 |

For Nonzero features, the three largest partial eta^2 terms are layer_label (0.941), role_framing:layer_label (0.338), role_framing (0.267).
### Top-1 activation

| Term | Sum sq | df | F | p-value | Partial eta^2 |
| --- | --- | --- | --- | --- | --- |
| layer_label | 7.76e+05 | 2 | 5.7e+03 | 0 | 0.907 |
| content_domain:layer_label | 6.98e+04 | 4 | 257 | 4.46e-158 | 0.469 |
| content_domain | 2.48e+04 | 2 | 182 | 1.19e-69 | 0.239 |
| role_framing:layer_label | 1.91e+04 | 4 | 70.2 | 2.54e-53 | 0.194 |
| surface_paraphrase:layer_label | 7.69e+03 | 4 | 28.3 | 1.99e-22 | 0.0885 |
| surface_paraphrase | 4.28e+03 | 2 | 31.5 | 4.77e-14 | 0.0513 |
| role_framing | 4e+03 | 2 | 29.4 | 3.58e-13 | 0.048 |
| content_domain:role_framing | 3.26e+03 | 4 | 12 | 1.5e-09 | 0.0396 |
| content_domain:surface_paraphrase | 3.18e+03 | 4 | 11.7 | 2.58e-09 | 0.0386 |
| language_register:layer_label | 1.27e+03 | 4 | 4.67 | 0.000968 | 0.0158 |
| content_domain:language_register | 1.24e+03 | 4 | 4.56 | 0.00116 | 0.0154 |
| surface_paraphrase:language_register | 570 | 4 | 2.1 | 0.0791 | 0.00715 |
| language_register | 535 | 2 | 3.94 | 0.0198 | 0.00672 |
| role_framing:surface_paraphrase | 148 | 4 | 0.544 | 0.703 | 0.00187 |
| role_framing:language_register | 116 | 4 | 0.427 | 0.789 | 0.00147 |

For Top-1 activation, the three largest partial eta^2 terms are layer_label (0.907), content_domain:layer_label (0.469), content_domain (0.239).
### Feature entropy

| Term | Sum sq | df | F | p-value | Partial eta^2 |
| --- | --- | --- | --- | --- | --- |
| layer_label | 201 | 2 | 1.53e+04 | 0 | 0.963 |
| role_framing:layer_label | 6.31 | 4 | 240 | 2.36e-150 | 0.452 |
| role_framing | 4.32 | 2 | 329 | 5.77e-114 | 0.361 |
| content_domain:layer_label | 3.36 | 4 | 128 | 1.29e-90 | 0.305 |
| content_domain | 2.22 | 2 | 169 | 2.99e-65 | 0.225 |
| content_domain:role_framing | 0.738 | 4 | 28.1 | 2.7e-22 | 0.088 |
| language_register | 0.413 | 2 | 31.4 | 5.18e-14 | 0.0512 |
| content_domain:surface_paraphrase | 0.375 | 4 | 14.3 | 2.2e-11 | 0.0468 |
| content_domain:language_register | 0.352 | 4 | 13.4 | 1.14e-10 | 0.044 |
| surface_paraphrase:layer_label | 0.282 | 4 | 10.7 | 1.57e-08 | 0.0355 |
| role_framing:language_register | 0.234 | 4 | 8.9 | 4.43e-07 | 0.0297 |
| language_register:layer_label | 0.169 | 4 | 6.41 | 4.18e-05 | 0.0216 |
| role_framing:surface_paraphrase | 0.0481 | 4 | 1.83 | 0.121 | 0.00625 |
| surface_paraphrase | 0.0427 | 2 | 3.25 | 0.0393 | 0.00555 |
| surface_paraphrase:language_register | 0.0156 | 4 | 0.594 | 0.667 | 0.00204 |

For Feature entropy, the three largest partial eta^2 terms are layer_label (0.963), role_framing:layer_label (0.452), role_framing (0.361).

## 3. Variance components

![Variance components](figures/variance_components_pie.pdf)

| Response | content_domain | role_framing | surface_paraphrase | language_register | layer_label | two_way_interactions | three_way_interactions | residual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Activation norm | 0.165 | 0.142 | 0.028 | 0.0334 | 98.7 | 0.27 | 0.183 | 0.489 |
| Nonzero features | 0.727 | 1.97 | 0.0255 | 0.277 | 85.8 | 5.77 | 1.63 | 3.77 |
| Top-1 activation | 2.49 | 0.402 | 0.431 | 0.0538 | 78 | 10.7 | 1.71 | 6.24 |
| Feature entropy | 0.979 | 1.9 | 0.0188 | 0.182 | 88.3 | 5.23 | 1.11 | 2.26 |

## 4. Two-way interactions

![Interaction heatmap](figures/interaction_heatmap.pdf)

Five strongest two-way interactions across all response variables:

| Response | Interaction | Partial eta^2 |
| --- | --- | --- |
| Top-1 activation | content_domain:layer_label | 0.469 |
| Feature entropy | role_framing:layer_label | 0.452 |
| Nonzero features | role_framing:layer_label | 0.338 |
| Feature entropy | content_domain:layer_label | 0.305 |
| Nonzero features | content_domain:layer_label | 0.254 |

## 5. LHS retrospective

![LHS vs full comparison](figures/lhs_vs_full_comparison.pdf)

LHS seed: 42. The categorical design uses a Latin-style 3^3 construction over content_domain, role_framing, and surface_paraphrase, with language_register assigned by modular cycling; K=18 and K=9 use deterministic seeded subsets of that K=27 design.

Spearman correlation between LHS-derived main-effect partial eta^2 ranking and full-factorial main-effect ranking:

| Response | K=27 | K=18 | K=9 |
| --- | --- | --- | --- |
| Activation norm | 0.9 | 0.5 | 0 |
| Nonzero features | 0.9 | 0.6 | -0.9 |
| Top-1 activation | 0.9 | 0.3 | -0.6 |
| Feature entropy | 0.9 | 0.6 | 0.3 |

## 6. Block-diagonality permutation tests

![Block diagonality permutation](figures/block_diagonality_permutation.pdf)

Permutation seed: 20260508. Each null distribution uses 1000 random label shuffles preserving 9/9/9 block sizes.

| Block factor | Observed | Mean within | Mean between | Null mean | Null 95% CI | Empirical p |
| --- | --- | --- | --- | --- | --- | --- |
| role_framing | 0.0912 | 0.959 | 0.868 | 0.000198 | [-0.00717, 0.0163] | 0 |
| surface_paraphrase | -0.00394 | 0.893 | 0.897 | -0.000191 | [-0.0077, 0.0149] | 0.685 |
| language_register | 0.00247 | 0.897 | 0.895 | 1.07e-05 | [-0.0075, 0.0152] | 0.279 |

## 7. Summary table

| Factor | Mean partial eta^2 | Rank | Block-diagonality p-value |
| --- | --- | --- | --- |
| layer_label | 0.951 | 1 |  |
| content_domain:layer_label | 0.277 | 2 |  |
| role_framing:layer_label | 0.256 | 3 |  |
| role_framing | 0.213 | 4 | 0 |
| content_domain | 0.195 | 5 |  |
| content_domain:role_framing | 0.0701 | 6 |  |
| surface_paraphrase:layer_label | 0.0631 | 7 |  |
| language_register | 0.0385 | 8 | 0.279 |
| content_domain:surface_paraphrase | 0.0383 | 9 |  |
| surface_paraphrase | 0.0254 | 10 | 0.685 |
| content_domain:language_register | 0.0226 | 11 |  |
| language_register:layer_label | 0.0214 | 12 |  |
| role_framing:language_register | 0.0122 | 13 |  |
| role_framing:surface_paraphrase | 0.00532 | 14 |  |
| surface_paraphrase:language_register | 0.00302 | 15 |  |
