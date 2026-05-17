from __future__ import annotations

from itertools import combinations
from pathlib import Path
import time
from typing import Any
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.stats.anova import anova_lm


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

RESPONSES = ["activation_norm", "nonzero_features", "top1_activation", "feature_entropy"]
RESPONSE_TITLES = {
    "activation_norm": "Activation norm",
    "nonzero_features": "Nonzero features",
    "top1_activation": "Top-1 activation",
    "feature_entropy": "Feature entropy",
}

FACTOR_LEVELS = {
    "content_domain": ["factual_recall", "arithmetic", "simple_reasoning"],
    "role_framing": ["direct_question", "fill_blank", "multi_turn"],
    "surface_paraphrase": ["canonical", "lexical", "syntactic"],
    "language_register": ["formal", "casual", "instruction"],
    "layer_label": ["early", "middle", "late"],
}

LHS_SEED = 42
BLOCK_SEED = 20260508
BLOCK_PERMUTATIONS = 1000


def project_root() -> Path:
    return Path.cwd().resolve()


def sig3(value: float | int) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.3g}"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def term_name(term: str) -> str:
    cleaned = term.replace("C(", "").replace(")", "")
    return cleaned


def term_formula(term: str) -> str:
    return f"C({term})"


def formula_terms(max_order: int = 2) -> list[str]:
    terms = [term_formula(factor) for factor in FACTORS]
    for order in range(2, max_order + 1):
        for combo in combinations(FACTORS, order):
            terms.append(":".join(term_formula(factor) for factor in combo))
    return terms


def formula(response: str, max_order: int = 2) -> str:
    return f"{response} ~ " + " + ".join(formula_terms(max_order=max_order))


def add_response_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    top1 = []
    entropy = []
    for _, row in df.iterrows():
        values = np.abs(np.asarray(row["sae_values"], dtype=np.float64))
        if len(values) == 0 or values.sum() == 0:
            top1.append(0.0)
            entropy.append(0.0)
            continue
        probs = values / values.sum()
        top1.append(float(values.max()))
        entropy.append(float(-(probs * np.log(probs)).sum()))
    df["top1_activation"] = np.asarray(top1, dtype=np.float32)
    df["feature_entropy"] = np.asarray(entropy, dtype=np.float32)
    return df


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


def run_anova(df: pd.DataFrame, response: str, max_order: int = 2) -> pd.DataFrame:
    model = smf.ols(formula(response, max_order=max_order), data=df).fit()
    table = anova_lm(model, typ=2).reset_index(names="term")
    residual_ss = float(table.loc[table["term"] == "Residual", "sum_sq"].iloc[0])
    table = table[table["term"] != "Residual"].copy()
    table["clean_term"] = table["term"].map(term_name)
    table["partial_eta_sq"] = table["sum_sq"] / (table["sum_sq"] + residual_ss)
    table["response"] = response
    table = table.sort_values("partial_eta_sq", ascending=False).reset_index(drop=True)
    return table


def all_anovas(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {response: run_anova(df, response, max_order=2) for response in RESPONSES}


def anova_markdown_table(table: pd.DataFrame) -> str:
    rows = []
    for _, row in table.iterrows():
        rows.append(
            [
                row["clean_term"],
                sig3(row["sum_sq"]),
                sig3(row["df"]),
                sig3(row["F"]),
                sig3(row["PR(>F)"]),
                sig3(row["partial_eta_sq"]),
            ]
        )
    return md_table(["Term", "Sum sq", "df", "F", "p-value", "Partial eta^2"], rows)


def plot_anova_effect_sizes(anovas: dict[str, pd.DataFrame], output_path: Path) -> None:
    rows = []
    for response, table in anovas.items():
        for _, row in table.iterrows():
            rows.append({"response": RESPONSE_TITLES[response], "term": row["clean_term"], "partial_eta_sq": row["partial_eta_sq"]})
    plot_df = pd.DataFrame(rows)
    order = (
        plot_df.groupby("term")["partial_eta_sq"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
    for ax, response_title in zip(axes.flat, [RESPONSE_TITLES[r] for r in RESPONSES]):
        sub = plot_df[plot_df["response"] == response_title]
        sns.barplot(data=sub, x="partial_eta_sq", y="term", order=order, color="#668cc4", ax=ax)
        ax.set_title(response_title)
        ax.set_xlabel("Partial eta^2")
        ax.set_ylabel("")
    fig.suptitle("ANOVA partial eta^2 by response", y=1.01, fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def variance_components(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for response in RESPONSES:
        table = run_anova(df, response, max_order=3)
        model = smf.ols(formula(response, max_order=3), data=df).fit()
        full_table = anova_lm(model, typ=2).reset_index(names="term")
        residual = float(full_table.loc[full_table["term"] == "Residual", "sum_sq"].iloc[0])
        total_ss = float(((df[response] - df[response].mean()) ** 2).sum())

        component_values: dict[str, float] = {}
        for factor in FACTORS:
            mask = full_table["term"] == term_formula(factor)
            component_values[factor] = float(full_table.loc[mask, "sum_sq"].sum())
        two_way = full_table[(full_table["term"].str.count(":") == 1)]["sum_sq"].sum()
        three_way = full_table[(full_table["term"].str.count(":") == 2)]["sum_sq"].sum()
        component_values["two_way_interactions"] = float(two_way)
        component_values["three_way_interactions"] = float(three_way)
        component_values["residual"] = residual

        for component, ss in component_values.items():
            rows.append(
                {
                    "response": response,
                    "component": component,
                    "sum_sq": ss,
                    "percent_total": 100 * ss / total_ss if total_ss else 0,
                }
            )
    return pd.DataFrame(rows)


def plot_variance_components(components: pd.DataFrame, output_path: Path) -> None:
    order = [*FACTORS, "two_way_interactions", "three_way_interactions", "residual"]
    pivot = components.pivot(index="response", columns="component", values="percent_total").reindex(RESPONSES)
    pivot = pivot[[col for col in order if col in pivot.columns]]
    colors = sns.color_palette("tab10", n_colors=len(pivot.columns))
    fig, ax = plt.subplots(figsize=(13, 6))
    bottom = np.zeros(len(pivot))
    for color, component in zip(colors, pivot.columns):
        values = pivot[component].to_numpy()
        ax.bar([RESPONSE_TITLES[r] for r in pivot.index], values, bottom=bottom, label=term_name(component), color=color)
        bottom += values
    ax.set_title("Variance components as percent of total sum of squares")
    ax.set_ylabel("Percent of total SS")
    ax.set_ylim(0, max(100, float(bottom.max()) * 1.05))
    ax.tick_params(axis="x", rotation=15)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), title="Component")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def variance_components_table(components: pd.DataFrame) -> str:
    rows = []
    for response in RESPONSES:
        sub = components[components["response"] == response].set_index("component")
        row = [RESPONSE_TITLES[response]]
        for component in [*FACTORS, "two_way_interactions", "three_way_interactions", "residual"]:
            row.append(sig3(sub.loc[component, "percent_total"]) if component in sub.index else "0")
        rows.append(row)
    return md_table(
        ["Response", *[term_name(c) for c in [*FACTORS, "two_way_interactions", "three_way_interactions", "residual"]]],
        rows,
    )


def interaction_heatmap_data(anovas: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    matrices = {}
    for response, table in anovas.items():
        matrix = pd.DataFrame(np.zeros((len(FACTORS), len(FACTORS))), index=FACTORS, columns=FACTORS)
        lookup = dict(zip(table["clean_term"], table["partial_eta_sq"]))
        for factor in FACTORS:
            matrix.loc[factor, factor] = lookup.get(factor, 0.0)
        for a, b in combinations(FACTORS, 2):
            value = lookup.get(f"{a}:{b}", lookup.get(f"{b}:{a}", 0.0))
            matrix.loc[a, b] = value
            matrix.loc[b, a] = value
        matrices[response] = matrix
    return matrices


def plot_interaction_heatmaps(matrices: dict[str, pd.DataFrame], output_path: Path) -> None:
    vmax = max(float(matrix.to_numpy().max()) for matrix in matrices.values())
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    labels = [FACTOR_TITLES[f] for f in FACTORS]
    for ax, response in zip(axes.flat, RESPONSES):
        matrix = matrices[response]
        sns.heatmap(matrix, vmin=0, vmax=vmax, cmap="viridis", annot=True, fmt=".3g", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(RESPONSE_TITLES[response])
        ax.tick_params(axis="x", rotation=35)
        ax.tick_params(axis="y", rotation=0)
    fig.suptitle("Main effects and two-way interaction partial eta^2", y=1.01, fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def strongest_interactions(anovas: dict[str, pd.DataFrame], n: int = 5) -> pd.DataFrame:
    rows = []
    for response, table in anovas.items():
        sub = table[table["clean_term"].str.contains(":")].copy()
        for _, row in sub.iterrows():
            rows.append({"response": response, "term": row["clean_term"], "partial_eta_sq": row["partial_eta_sq"]})
    return pd.DataFrame(rows).sort_values("partial_eta_sq", ascending=False).head(n)


def select_lhs_prompt_hashes(prompt_df: pd.DataFrame, k: int, seed: int = LHS_SEED) -> list[str]:
    rng = np.random.default_rng(seed + k)
    domains = FACTOR_LEVELS["content_domain"]
    roles = FACTOR_LEVELS["role_framing"]
    paras = FACTOR_LEVELS["surface_paraphrase"]
    regs = FACTOR_LEVELS["language_register"]

    latin_cells = []
    for i, domain in enumerate(domains):
        for j, role in enumerate(roles):
            for m, para in enumerate(paras):
                reg = regs[(i + j + m) % 3]
                latin_cells.append((domain, role, para, reg))
    rng.shuffle(latin_cells)
    selected_cells = latin_cells[:k]

    hashes: list[str] = []
    for cell_i, (domain, role, para, reg) in enumerate(selected_cells):
        matches = prompt_df[
            (prompt_df["content_domain"] == domain)
            & (prompt_df["role_framing"] == role)
            & (prompt_df["surface_paraphrase"] == para)
            & (prompt_df["language_register"] == reg)
        ].sort_values("content_id")
        if matches.empty:
            raise RuntimeError(f"No prompt rows found for LHS cell {(domain, role, para, reg)}")
        hashes.append(str(matches.iloc[cell_i % len(matches)]["prompt_hash"]))
    return hashes


def lhs_retrospective(df: pd.DataFrame, full_anovas: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    prompt_df = df[df["layer"] == 12].drop_duplicates("prompt_hash").copy()
    rows = []
    rank_rows = []
    full_main = {}
    for response, table in full_anovas.items():
        full_main[response] = table[~table["clean_term"].str.contains(":")].set_index("clean_term")["partial_eta_sq"]
        for factor, value in full_main[response].items():
            rows.append({"response": response, "design": "full", "factor": factor, "partial_eta_sq": value})

    for k in [27, 18, 9]:
        hashes = select_lhs_prompt_hashes(prompt_df, k)
        sample = df[df["prompt_hash"].isin(hashes)].copy()
        for response in RESPONSES:
            table = run_anova(sample, response, max_order=2)
            main = table[~table["clean_term"].str.contains(":")].set_index("clean_term")["partial_eta_sq"].reindex(FACTORS)
            for factor, value in main.items():
                rows.append({"response": response, "design": f"LHS K={k}", "factor": factor, "partial_eta_sq": value})
            corr = spearmanr(full_main[response].reindex(FACTORS), main).statistic
            rank_rows.append({"response": response, "K": k, "spearman": float(corr)})
    return pd.DataFrame(rows), pd.DataFrame(rank_rows)


def plot_lhs_comparison(lhs_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharey=True)
    for ax, response in zip(axes.flat, RESPONSES):
        sub = lhs_df[lhs_df["response"] == response]
        sns.barplot(
            data=sub,
            x="factor",
            y="partial_eta_sq",
            hue="design",
            order=FACTORS,
            hue_order=["full", "LHS K=27", "LHS K=18", "LHS K=9"],
            ax=ax,
        )
        ax.set_title(RESPONSE_TITLES[response])
        ax.set_xlabel("")
        ax.set_ylabel("Partial eta^2")
        ax.tick_params(axis="x", rotation=30)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    for ax in axes.flat:
        ax.legend_.remove()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Full factorial vs categorical Latin-style subsamples", y=1.07, fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def lhs_rank_table(rank_df: pd.DataFrame) -> str:
    rows = []
    for response in RESPONSES:
        sub = rank_df[rank_df["response"] == response].set_index("K")
        rows.append([RESPONSE_TITLES[response], sig3(sub.loc[27, "spearman"]), sig3(sub.loc[18, "spearman"]), sig3(sub.loc[9, "spearman"])])
    return md_table(["Response", "K=27", "K=18", "K=9"], rows)


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


def block_statistic(similarity: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    same = labels[:, None] == labels[None, :]
    off_diag = ~np.eye(len(labels), dtype=bool)
    within = similarity[same & off_diag]
    between = similarity[(~same) & off_diag]
    mean_within = float(within.mean())
    mean_between = float(between.mean())
    return mean_within - mean_between, mean_within, mean_between


def block_diagonality_tests(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, float]:
    subset = ordered_fr1_layer12(df)
    dense = l2_normalize(dense_sae_matrix(subset))
    similarity = dense @ dense.T
    rng = np.random.default_rng(BLOCK_SEED)
    rows = []
    role_null_for_plot = None
    role_observed = None

    for factor in ["role_framing", "surface_paraphrase", "language_register"]:
        labels = subset[factor].to_numpy()
        observed, mean_within, mean_between = block_statistic(similarity, labels)
        null = np.empty(BLOCK_PERMUTATIONS, dtype=np.float32)
        for i in range(BLOCK_PERMUTATIONS):
            shuffled = rng.permutation(labels)
            null[i] = block_statistic(similarity, shuffled)[0]
        p_value = float((null >= observed).mean())
        low, high = np.quantile(null, [0.025, 0.975])
        rows.append(
            {
                "block_factor": factor,
                "observed": observed,
                "mean_within": mean_within,
                "mean_between": mean_between,
                "null_mean": float(null.mean()),
                "null_ci_low": float(low),
                "null_ci_high": float(high),
                "p_value": p_value,
            }
        )
        if factor == "role_framing":
            role_null_for_plot = null
            role_observed = observed
    return pd.DataFrame(rows), role_null_for_plot, float(role_observed)


def plot_block_permutation(null: np.ndarray, observed: float, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(null, bins=35, color="#8eb5a4", edgecolor="white")
    ax.axvline(observed, color="#9e2f2f", linewidth=2, label=f"Observed = {observed:.3g}")
    ax.set_title("Role-framing block-diagonality permutation null")
    ax.set_xlabel("Mean within-block similarity minus mean between-block similarity")
    ax.set_ylabel("Permutation count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def block_table(block_df: pd.DataFrame) -> str:
    rows = []
    for _, row in block_df.iterrows():
        rows.append(
            [
                row["block_factor"],
                sig3(row["observed"]),
                sig3(row["mean_within"]),
                sig3(row["mean_between"]),
                sig3(row["null_mean"]),
                f"[{sig3(row['null_ci_low'])}, {sig3(row['null_ci_high'])}]",
                sig3(row["p_value"]),
            ]
        )
    return md_table(["Block factor", "Observed", "Mean within", "Mean between", "Null mean", "Null 95% CI", "Empirical p"], rows)


def summary_table(anovas: dict[str, pd.DataFrame], block_df: pd.DataFrame) -> str:
    rows = []
    terms = [*FACTORS, *[f"{a}:{b}" for a, b in combinations(FACTORS, 2)]]
    means = {}
    for term in terms:
        values = []
        for table in anovas.values():
            lookup = table.set_index("clean_term")["partial_eta_sq"]
            values.append(float(lookup.get(term, 0.0)))
        means[term] = float(np.mean(values))
    sorted_terms = sorted(terms, key=lambda t: means[t], reverse=True)
    block_lookup = dict(zip(block_df["block_factor"], block_df["p_value"]))
    for rank, term in enumerate(sorted_terms, start=1):
        block_p = block_lookup.get(term, "")
        rows.append([term, sig3(means[term]), rank, sig3(block_p) if block_p != "" else ""])
    return md_table(["Factor", "Mean partial eta^2", "Rank", "Block-diagonality p-value"], rows)


def response_summary_table(df: pd.DataFrame) -> str:
    rows = []
    for layer, layer_df in df.sort_values("layer").groupby("layer_label", sort=False):
        for response in RESPONSES:
            values = layer_df[response]
            rows.append([layer, RESPONSE_TITLES[response], sig3(values.mean()), sig3(values.std()), sig3(values.min()), sig3(values.max())])
    return md_table(["Layer", "Response", "Mean", "SD", "Min", "Max"], rows)


def top_three_sentence(response: str, table: pd.DataFrame) -> str:
    top = table.head(3)
    pieces = [f"{row['clean_term']} ({sig3(row['partial_eta_sq'])})" for _, row in top.iterrows()]
    return f"For {RESPONSE_TITLES[response]}, the three largest partial eta^2 terms are {', '.join(pieces)}."


def strongest_interactions_table(interactions: pd.DataFrame) -> str:
    rows = [
        [RESPONSE_TITLES[row["response"]], row["term"], sig3(row["partial_eta_sq"])]
        for _, row in interactions.iterrows()
    ]
    return md_table(["Response", "Interaction", "Partial eta^2"], rows)


def render_report(
    *,
    df: pd.DataFrame,
    anovas: dict[str, pd.DataFrame],
    components: pd.DataFrame,
    interactions: pd.DataFrame,
    lhs_rank: pd.DataFrame,
    block_df: pd.DataFrame,
    wall_clock_seconds: float,
) -> str:
    anova_sections = []
    for response in RESPONSES:
        anova_sections.append(
            f"### {RESPONSE_TITLES[response]}\n\n"
            f"{anova_markdown_table(anovas[response])}\n\n"
            f"{top_three_sentence(response, anovas[response])}"
        )

    return f"""# Full factorial analysis report

Runtime: {wall_clock_seconds:.2f} seconds

## 1. Response variables

Response variables are `activation_norm`, `nonzero_features`, `top1_activation`, and `feature_entropy`. `top1_activation` is the largest SAE activation magnitude in a cell. `feature_entropy` is Shannon entropy over L1-normalized nonzero SAE activation magnitudes.

{response_summary_table(df)}

## 2. Main-effect ANOVA results

![ANOVA effect sizes](figures/anova_effect_sizes.pdf)

{chr(10).join(anova_sections)}

## 3. Variance components

![Variance components](figures/variance_components_pie.pdf)

{variance_components_table(components)}

## 4. Two-way interactions

![Interaction heatmap](figures/interaction_heatmap.pdf)

Five strongest two-way interactions across all response variables:

{strongest_interactions_table(interactions)}

## 5. LHS retrospective

![LHS vs full comparison](figures/lhs_vs_full_comparison.pdf)

LHS seed: {LHS_SEED}. The categorical design uses a Latin-style 3^3 construction over content_domain, role_framing, and surface_paraphrase, with language_register assigned by modular cycling; K=18 and K=9 use deterministic seeded subsets of that K=27 design.

Spearman correlation between LHS-derived main-effect partial eta^2 ranking and full-factorial main-effect ranking:

{lhs_rank_table(lhs_rank)}

## 6. Block-diagonality permutation tests

![Block diagonality permutation](figures/block_diagonality_permutation.pdf)

Permutation seed: {BLOCK_SEED}. Each null distribution uses {BLOCK_PERMUTATIONS} random label shuffles preserving 9/9/9 block sizes.

{block_table(block_df)}

## 7. Summary table

{summary_table(anovas, block_df)}
"""


def run_factorial(repo_root: Path) -> str:
    started = time.perf_counter()
    output_dir = repo_root / "output"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = pd.read_parquet(output_dir / "activations.parquet", engine="pyarrow")
    df = add_response_variables(df)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ValueWarning)
        anovas = all_anovas(df)
        components = variance_components(df)
        lhs_df, lhs_rank = lhs_retrospective(df, anovas)

    plot_anova_effect_sizes(anovas, figures_dir / "anova_effect_sizes.pdf")
    plot_variance_components(components, figures_dir / "variance_components_pie.pdf")

    heatmaps = interaction_heatmap_data(anovas)
    plot_interaction_heatmaps(heatmaps, figures_dir / "interaction_heatmap.pdf")
    interactions = strongest_interactions(anovas)
    plot_lhs_comparison(lhs_df, figures_dir / "lhs_vs_full_comparison.pdf")

    block_df, role_null, role_observed = block_diagonality_tests(df)
    plot_block_permutation(role_null, role_observed, figures_dir / "block_diagonality_permutation.pdf")

    report = render_report(
        df=df,
        anovas=anovas,
        components=components,
        interactions=interactions,
        lhs_rank=lhs_rank,
        block_df=block_df,
        wall_clock_seconds=time.perf_counter() - started,
    )
    (output_dir / "factorial_report.md").write_text(report)
    return report


def main() -> None:
    print(run_factorial(project_root()))
