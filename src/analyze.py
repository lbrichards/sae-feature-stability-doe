from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.schema import CORPUS_FACTOR_COLUMNS


FACTORS = [
    "content_domain",
    "role_framing",
    "surface_paraphrase",
    "language_register",
    "layer_label",
]

FACTOR_TITLES = {
    "content_domain": "Content domain",
    "role_framing": "Role framing",
    "surface_paraphrase": "Surface paraphrase",
    "language_register": "Language register",
    "layer_label": "Layer",
}

ORDER_BY_FACTOR = {
    "content_domain": ["factual_recall", "arithmetic", "simple_reasoning"],
    "role_framing": ["direct_question", "fill_blank", "multi_turn"],
    "surface_paraphrase": ["canonical", "lexical", "syntactic"],
    "language_register": ["formal", "casual", "instruction"],
    "layer_label": ["early", "middle", "late"],
}


@dataclass(frozen=True)
class PairComparison:
    label_a: str
    label_b: str
    jaccard_top50: float
    cosine_full: float

    @property
    def abs_difference(self) -> float:
        return abs(self.jaccard_top50 - self.cosine_full)


def project_root() -> Path:
    return Path.cwd().resolve()


def sig3(value: float | int) -> str:
    return f"{float(value):.3g}"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def dense_sae_matrix(df: pd.DataFrame, width: int = 16384) -> np.ndarray:
    matrix = np.zeros((len(df), width), dtype=np.float32)
    for row_i, (_, row) in enumerate(df.iterrows()):
        indices = np.asarray(row["sae_indices"], dtype=np.int32)
        values = np.asarray(row["sae_values"], dtype=np.float32)
        matrix[row_i, indices] = values
    return matrix


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def top_k_features(row: pd.Series, k: int) -> list[tuple[int, float]]:
    indices = np.asarray(row["sae_indices"], dtype=np.int32)
    values = np.asarray(row["sae_values"], dtype=np.float32)
    order = np.argsort(np.abs(values))[::-1][:k]
    return [(int(indices[i]), float(values[i])) for i in order]


def top_k_set(row: pd.Series, k: int) -> set[int]:
    return {idx for idx, _ in top_k_features(row, k)}


def jaccard(a: set[int], b: set[int]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def cell_label(row: pd.Series) -> str:
    role = str(row["role_framing"]).replace("_question", "").replace("_", "")[:6]
    para = str(row["surface_paraphrase"])[:5]
    reg = str(row["language_register"])[:5]
    return f"{role}_{para}_{reg}"


def ordered_fr1_layer12(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[
        (df["content_domain"] == "factual_recall")
        & (df["content_id"] == "fr1")
        & (df["layer"] == 12)
    ].copy()
    role_order = {"direct_question": 0, "fill_blank": 1, "multi_turn": 2}
    paraphrase_order = {"canonical": 0, "lexical": 1, "syntactic": 2}
    register_order = {"formal": 0, "casual": 1, "instruction": 2}
    subset["_order"] = list(
        zip(
            subset["role_framing"].map(role_order),
            subset["surface_paraphrase"].map(paraphrase_order),
            subset["language_register"].map(register_order),
        )
    )
    return subset.sort_values("_order").reset_index(drop=True)


def full_spectrum_analysis(df: pd.DataFrame, output_path: Path) -> tuple[list[PairComparison], float, float]:
    subset = ordered_fr1_layer12(df)
    labels = [cell_label(row) for _, row in subset.iterrows()]
    dense = l2_normalize(dense_sae_matrix(subset))
    cosine = dense @ dense.T
    top_sets = [top_k_set(row, 50) for _, row in subset.iterrows()]
    jaccard_matrix = np.array([[jaccard(a, b) for b in top_sets] for a in top_sets], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cosine, cmap="magma", vmin=0, vmax=1, xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Full-spectrum SAE cosine similarity: factual_recall / fr1 / layer 12")
    ax.set_xlabel("Cell")
    ax.set_ylabel("Cell")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    comparisons: list[PairComparison] = []
    for i, j in combinations(range(len(subset)), 2):
        comparisons.append(
            PairComparison(
                label_a=labels[i],
                label_b=labels[j],
                jaccard_top50=float(jaccard_matrix[i, j]),
                cosine_full=float(cosine[i, j]),
            )
        )
    comparisons.sort(key=lambda item: item.abs_difference, reverse=True)
    triu = np.triu_indices_from(cosine, k=1)
    return comparisons[:5], float(jaccard_matrix[triu].mean()), float(cosine[triu].mean())


def feature_value(row: pd.Series, feature_index: int) -> float:
    indices = np.asarray(row["sae_indices"], dtype=np.int32)
    values = np.asarray(row["sae_values"], dtype=np.float32)
    matches = np.where(indices == feature_index)[0]
    if len(matches) == 0:
        return 0.0
    return float(abs(values[matches[0]]))


def dominant_features(layer12: pd.DataFrame) -> list[tuple[int, int]]:
    counts: dict[int, int] = {}
    for _, row in layer12.iterrows():
        for idx, _ in top_k_features(row, 5):
            counts[idx] = counts.get(idx, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:5]


def plot_dominant_feature_variation(df: pd.DataFrame, dominant: list[tuple[int, int]], output_path: Path) -> dict[int, dict[str, Any]]:
    layer12 = df[df["layer"] == 12].copy()
    stats: dict[int, dict[str, Any]] = {}
    fig, axes = plt.subplots(len(dominant), len(FACTORS), figsize=(24, 18), sharey=False)
    for row_i, (feature_index, count) in enumerate(dominant):
        layer12_values = layer12.copy()
        layer12_values["feature_value"] = [feature_value(row, feature_index) for _, row in layer12_values.iterrows()]
        all_values = df.copy()
        all_values["feature_value"] = [feature_value(row, feature_index) for _, row in all_values.iterrows()]

        factor_ranges = {}
        for factor in CORPUS_FACTOR_COLUMNS:
            means = layer12_values.groupby(factor)["feature_value"].mean()
            factor_ranges[factor] = float(means.max() - means.min())
        layer_means = all_values.groupby("layer_label")["feature_value"].mean()
        strongest_factor = max(factor_ranges, key=factor_ranges.get)
        stats[feature_index] = {
            "count": count,
            "mean": float(layer12_values["feature_value"].mean()),
            "std": float(layer12_values["feature_value"].std()),
            "min": float(layer12_values["feature_value"].min()),
            "max": float(layer12_values["feature_value"].max()),
            "strongest_factor": strongest_factor,
            "strongest_range": factor_ranges[strongest_factor],
            "layer_panel_range": float(layer_means.max() - layer_means.min()),
        }

        for col_i, factor in enumerate(FACTORS):
            ax = axes[row_i, col_i]
            plot_df = all_values if factor == "layer_label" else layer12_values
            sns.boxplot(
                data=plot_df,
                x=factor,
                y="feature_value",
                order=ORDER_BY_FACTOR.get(factor),
                color="#9aa7d8",
                fliersize=1.5,
                ax=ax,
            )
            if row_i == 0:
                ax.set_title(FACTOR_TITLES[factor])
            if col_i == 0:
                ax.set_ylabel(f"Feature {feature_index}\n|activation|")
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=35, labelsize=7)
    fig.suptitle("Dominant layer-12 SAE feature activation magnitudes by factor", y=1.01, fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return stats


def pairwise_cosine_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dense = l2_normalize(dense_sae_matrix(df))
    cosine = dense @ dense.T
    return pd.DataFrame(cosine, index=df.index, columns=df.index), df.reset_index(drop=True)


def factor_effect_sizes(df: pd.DataFrame) -> pd.DataFrame:
    dense = l2_normalize(dense_sae_matrix(df))
    cosine = dense @ dense.T
    n = len(df)
    upper_i, upper_j = np.triu_indices(n, k=1)
    rows: list[dict[str, Any]] = []
    factor_cols = {
        "content_domain": df["content_domain"].to_numpy(),
        "role_framing": df["role_framing"].to_numpy(),
        "surface_paraphrase": df["surface_paraphrase"].to_numpy(),
        "language_register": df["language_register"].to_numpy(),
        "layer_label": df["layer_label"].to_numpy(),
    }

    for factor, values in factor_cols.items():
        same = values[upper_i] == values[upper_j]
        pair_values = cosine[upper_i, upper_j]
        mean_within = float(pair_values[same].mean())
        mean_between = float(pair_values[~same].mean())

        other_factors = [col for col in FACTORS if col != factor]
        paired_mask = np.ones(len(upper_i), dtype=bool)
        for other in other_factors:
            other_values = factor_cols[other]
            paired_mask &= other_values[upper_i] == other_values[upper_j]
        paired_mask &= values[upper_i] != values[upper_j]
        paired_mean_similarity = float(pair_values[paired_mask].mean()) if paired_mask.any() else np.nan
        paired_distance_proxy = 1.0 - paired_mean_similarity if paired_mask.any() else np.nan

        rows.append(
            {
                "factor": factor,
                "same_level_pairs": int(same.sum()),
                "different_level_pairs": int((~same).sum()),
                "mean_within": mean_within,
                "mean_between": mean_between,
                "crude_effect_size": mean_within - mean_between,
                "paired_pairs": int(paired_mask.sum()),
                "paired_mean_similarity": paired_mean_similarity,
                "paired_distance_proxy": paired_distance_proxy,
            }
        )
    return pd.DataFrame(rows).sort_values("paired_distance_proxy", ascending=False).reset_index(drop=True)


def plot_factor_effects(effect_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = effect_df.melt(
        id_vars="factor",
        value_vars=["crude_effect_size", "paired_distance_proxy"],
        var_name="metric",
        value_name="value",
    )
    label_map = {
        "crude_effect_size": "Crude within-between",
        "paired_distance_proxy": "Paired distance proxy",
    }
    plot_df["metric"] = plot_df["metric"].map(label_map)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=plot_df, x="factor", y="value", hue="metric", order=effect_df["factor"].tolist(), ax=ax)
    ax.set_title("Crude feature-pattern disturbance by factor")
    ax.set_xlabel("Factor")
    ax.set_ylabel("Effect-size proxy")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def dominant_feature_observations(stats: dict[int, dict[str, Any]]) -> str:
    sections = []
    for feature_index, item in stats.items():
        cv = item["std"] / item["mean"] if item["mean"] else 0.0
        if cv < 0.15:
            variance_text = "low relative variance"
        elif cv < 0.35:
            variance_text = "moderate relative variance"
        else:
            variance_text = "high relative variance"
        factor = FACTOR_TITLES.get(item["strongest_factor"], item["strongest_factor"])
        sections.append(
            f"- Feature {feature_index}: appears in {item['count']} of the layer-12 top-5 sets. "
            f"Its layer-12 mean |activation| is {sig3(item['mean'])} with {variance_text} "
            f"(min {sig3(item['min'])}, max {sig3(item['max'])}). The largest grouped mean spread in this descriptive view is along {factor} "
            f"({sig3(item['strongest_range'])}) among the layer-12 corpus factors. The layer panel is shown separately as a cross-SAE check "
            f"and has a grouped mean spread of {sig3(item['layer_panel_range'])}."
        )
    return "\n".join(sections)


def effect_observation(effect_df: pd.DataFrame) -> str:
    order = effect_df.sort_values("paired_distance_proxy", ascending=False)
    factors = order["factor"].tolist()
    paired_values = order["paired_distance_proxy"].tolist()
    crude_order = effect_df.sort_values("crude_effect_size", ascending=False)["factor"].tolist()
    gap_text = ""
    if len(paired_values) >= 2:
        gap_text = f" The largest adjacent gap in the paired ranking is {sig3(max(np.diff(paired_values) * -1))}."
    return (
        f"By the paired distance proxy, the factors rank as: {', '.join(factors)}. "
        f"By the crude within-minus-between proxy, the factors rank as: {', '.join(crude_order)}. "
        f"The paired view is the cleaner matched comparison, while the crude view is more confounded by the full factorial mix.{gap_text}"
    )


def effect_table(effect_df: pd.DataFrame) -> str:
    rows = []
    for _, row in effect_df.iterrows():
        rows.append(
            [
                row["factor"],
                int(row["same_level_pairs"]),
                int(row["different_level_pairs"]),
                sig3(row["mean_within"]),
                sig3(row["mean_between"]),
                sig3(row["crude_effect_size"]),
                int(row["paired_pairs"]),
                sig3(row["paired_mean_similarity"]),
                sig3(row["paired_distance_proxy"]),
            ]
        )
    return md_table(
        [
            "Factor",
            "Same pairs",
            "Different pairs",
            "Mean within",
            "Mean between",
            "Crude effect",
            "Paired pairs",
            "Paired mean similarity",
            "Paired distance proxy",
        ],
        rows,
    )


def comparison_table(comparisons: list[PairComparison]) -> str:
    return md_table(
        ["Cell A", "Cell B", "Top-50 Jaccard", "Full-spectrum cosine", "Abs diff"],
        [
            [item.label_a, item.label_b, sig3(item.jaccard_top50), sig3(item.cosine_full), sig3(item.abs_difference)]
            for item in comparisons
        ],
    )


def dominant_feature_table(dominant: list[tuple[int, int]]) -> str:
    return md_table(["Feature index", "Top-5 frequency at layer 12"], [[idx, count] for idx, count in dominant])


def render_report(
    *,
    comparisons: list[PairComparison],
    mean_topk: float,
    mean_full: float,
    dominant: list[tuple[int, int]],
    dominant_stats: dict[int, dict[str, Any]],
    effect_df: pd.DataFrame,
    wall_clock_seconds: float,
) -> str:
    q1_observation = (
        f"Across the 27 factual_recall/fr1 layer-12 cells, the mean off-diagonal top-50 Jaccard is {sig3(mean_topk)} "
        f"and the mean off-diagonal full-spectrum cosine is {sig3(mean_full)}. "
        "The largest-divergence pairs below show where thresholded feature-set overlap and full-vector geometry separate most."
    )
    q3_observation = effect_observation(effect_df)
    reading = (
        "The full-spectrum view preserves the block structure visible in the top-K inspection, but it gives a smoother view of similarity because it uses feature magnitudes and the long tail of active features. "
        "The largest top-K/full-spectrum divergences are therefore useful places to look for cases where the strongest feature identities agree while their activation profile differs, or the reverse.\n\n"
        "The dominant layer-12 features recur across a large fraction of cells and their magnitude summaries make the repeated-feature pattern explicit. "
        "The descriptive spreads in the small-multiples figure show whether each feature is nearly flat across factors or has visible variation tied to a particular grouping.\n\n"
        "The factor ranking should be read as a coarse geometry summary rather than a clean factorial estimate. "
        "The paired proxy holds all other recorded factors fixed, while the crude proxy averages over the full dataset and therefore mixes several sources of similarity."
    )
    return f"""# SAE feature stability second analysis report

Runtime: {wall_clock_seconds:.2f} seconds

## 1. Question 1: Top-K vs full-spectrum view

![Full-spectrum overlap heatmap](figures/full_spectrum_overlap_heatmap.pdf)

{q1_observation}

Largest absolute differences between top-50 Jaccard and full-spectrum cosine:

{comparison_table(comparisons)}

## 2. Question 2: Dominant-feature behavior

![Dominant feature variation](figures/dominant_feature_variation.pdf)

Dominant layer-12 features by frequency in each cell's top-5:

{dominant_feature_table(dominant)}

{dominant_feature_observations(dominant_stats)}

## 3. Question 3: Factor effect-size ranking

![Factor effect size bars](figures/factor_effect_size_bars.pdf)

{effect_table(effect_df)}

{q3_observation}

## 4. What this changes in our reading

{reading}
"""


def run_analysis(repo_root: Path) -> str:
    started = time.perf_counter()
    output_dir = repo_root / "output"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(output_dir / "activations.parquet", engine="pyarrow")

    sns.set_theme(style="whitegrid")
    comparisons, mean_topk, mean_full = full_spectrum_analysis(
        df, figures_dir / "full_spectrum_overlap_heatmap.pdf"
    )
    layer12 = df[df["layer"] == 12].copy()
    dominant = dominant_features(layer12)
    dominant_stats = plot_dominant_feature_variation(
        df, dominant, figures_dir / "dominant_feature_variation.pdf"
    )
    effect_df = factor_effect_sizes(df)
    plot_factor_effects(effect_df, figures_dir / "factor_effect_size_bars.pdf")

    report = render_report(
        comparisons=comparisons,
        mean_topk=mean_topk,
        mean_full=mean_full,
        dominant=dominant,
        dominant_stats=dominant_stats,
        effect_df=effect_df,
        wall_clock_seconds=time.perf_counter() - started,
    )
    (output_dir / "analysis_report.md").write_text(report)
    return report


def main() -> None:
    print(run_analysis(project_root()))
