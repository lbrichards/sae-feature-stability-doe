# SAE feature stability spike

Collects and analyzes Gemma Scope SAE feature activations across a balanced
factorial prompt design: 405 prompt cells expanded across three layers for 1215
total measurements.

The repository includes the corpus, source code, persisted measurement table,
analysis reports, figures, and generated paper artifact needed to inspect or
reproduce the reported results.

## Setup

```bash
uv sync
```

Gemma model access may require Hugging Face login and license acceptance:

```bash
huggingface-cli login
```

## Input

The prompt corpus is committed at:

```text
data/corpus.csv
```

The collector does not generate the corpus.

## Reproduce

```bash
uv run run-collection
uv run run-inspection
uv run run-analysis
uv run run-factorial
```

The collection step reruns Gemma-2-2B-IT and Gemma Scope SAEs and may require
Hugging Face access plus enough local memory/compute. If you only want to inspect
or rerun downstream analysis, use the committed `output/activations.parquet`.

Primary outputs:

- `output/activations.parquet`
- `output/collection_report.md`
- `output/inspection_report.md`
- `output/analysis_report.md`
- `output/factorial_report.md`
- `output/sae_feature_stability_report.tex`
- `output/sae_features_stability_report.pdf`
- `output/figures/activation_norms_by_factor.pdf`
- `output/figures/nonzero_features_by_factor.pdf`
- `output/figures/feature_overlap_heatmap.pdf`
- `output/figures/top_features_per_cell.pdf`
- `output/figures/anova_effect_sizes.pdf`
- `output/figures/interaction_heatmap.pdf`
- `output/figures/variance_components_pie.pdf`
- `output/figures/block_diagonality_permutation.pdf`
- `output/figures/lhs_vs_full_comparison.pdf`

## Interactive inspection

```bash
uv run jupyter lab notebooks/inspect_collection.ipynb
```
