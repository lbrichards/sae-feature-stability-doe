from __future__ import annotations

import hashlib
from pathlib import Path
import textwrap
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.collect import (
    choose_device_and_dtype,
    discover_residual_width16k_sae,
    load_corpus,
    render_prompt,
)
from src.schema import CORPUS_FACTOR_COLUMNS, LAYER_LEVELS, MODEL_NAME


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

SAMPLE_CELL_SPECS = [
    ("factual_recall", "fr1", "direct_question", "canonical", "formal", 12),
    ("factual_recall", "fr1", "direct_question", "canonical", "casual", 12),
    ("arithmetic", "ar1", "direct_question", "canonical", "formal", 12),
    ("simple_reasoning", "sr1", "direct_question", "canonical", "formal", 12),
    ("factual_recall", "fr1", "multi_turn", "canonical", "formal", 12),
    ("factual_recall", "fr1", "direct_question", "canonical", "formal", 20),
]


def project_root() -> Path:
    return Path.cwd().resolve()


def sig3(value: float | int) -> str:
    return f"{float(value):.3g}"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


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


def add_rendered_prompts(df: pd.DataFrame, corpus: pd.DataFrame, tokenizer: Any) -> pd.DataFrame:
    rendered_by_hash: dict[str, str] = {}
    for _, row in corpus.iterrows():
        rendered = render_prompt(tokenizer, row)
        rendered_by_hash[hashlib.sha256(rendered.encode("utf-8")).hexdigest()] = rendered
    df = df.copy()
    df["rendered_prompt"] = df["prompt_hash"].map(rendered_by_hash).fillna("")
    return df


def plot_box_grid(df: pd.DataFrame, value_col: str, ylabel: str, title: str, output_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 5, figsize=(23, 5), sharey=True)
    order_by_factor = {
        "layer_label": ["early", "middle", "late"],
        "role_framing": ["direct_question", "fill_blank", "multi_turn"],
        "surface_paraphrase": ["canonical", "lexical", "syntactic"],
        "language_register": ["formal", "casual", "instruction"],
        "content_domain": ["factual_recall", "arithmetic", "simple_reasoning"],
    }

    for ax, factor in zip(axes, FACTORS):
        sns.boxplot(
            data=df,
            x=factor,
            y=value_col,
            order=order_by_factor.get(factor),
            ax=ax,
            color="#7aa6c2",
            fliersize=2,
        )
        ax.set_title(FACTOR_TITLES[factor])
        ax.set_xlabel("")
        ax.set_ylabel(ylabel if ax is axes[0] else "")
        ax.tick_params(axis="x", rotation=35, labelsize=8)
    fig.suptitle(title, y=1.03, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_feature_overlap_heatmap(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
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
    subset = subset.sort_values("_order").reset_index(drop=True)
    labels = [
        f"{role.replace('_question', '').replace('_', '')[:6]}_"
        f"{para[:5]}_{reg[:5]}"
        for role, para, reg in zip(subset["role_framing"], subset["surface_paraphrase"], subset["language_register"])
    ]
    sets = [top_k_set(row, 50) for _, row in subset.iterrows()]
    matrix = np.array([[jaccard(a, b) for b in sets] for a in sets], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(matrix, cmap="viridis", vmin=0, vmax=1, xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Top-50 SAE feature Jaccard overlap: factual_recall / fr1 / layer 12")
    ax.set_xlabel("Cell")
    ax.set_ylabel("Cell")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(matrix, index=labels, columns=labels)


def plot_top_features_per_cell(sample_rows: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False)
    for ax, (_, row) in zip(axes.flat, sample_rows.iterrows()):
        features = top_k_features(row, 5)
        labels = [str(idx) for idx, _ in features]
        values = [value for _, value in features]
        ax.bar(labels, values, color="#579c87")
        ax.set_title(
            f"{row['content_domain']} / {row['content_id']} / L{row['layer']}\n"
            f"{row['role_framing']} / {row['language_register']}",
            fontsize=9,
        )
        ax.set_xlabel("Feature index")
        ax.set_ylabel("Activation")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.suptitle("Top-5 SAE features for selected cells", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def scalar_stats_table(df: pd.DataFrame) -> str:
    rows: list[list[Any]] = []
    for layer, layer_df in df.sort_values("layer").groupby("layer"):
        row = [int(layer), layer_df["layer_label"].iloc[0]]
        for col in ["activation_norm", "nonzero_features", "prompt_token_len"]:
            values = layer_df[col]
            row.extend([sig3(values.min()), sig3(values.mean()), sig3(values.median()), sig3(values.max())])
        rows.append(row)
    headers = [
        "Layer",
        "Label",
        "Act min",
        "Act mean",
        "Act median",
        "Act max",
        "NZ min",
        "NZ mean",
        "NZ median",
        "NZ max",
        "Tok min",
        "Tok mean",
        "Tok median",
        "Tok max",
    ]
    return md_table(headers, rows)


def dataset_summary_table(df: pd.DataFrame) -> str:
    rows: list[list[Any]] = [["Total rows", len(df)]]
    for layer, count in df["layer"].value_counts().sort_index().items():
        rows.append([f"Rows layer {int(layer)}", int(count)])
    for factor in CORPUS_FACTOR_COLUMNS:
        counts = ", ".join(f"{level}: {count}" for level, count in df[factor].value_counts().sort_index().items())
        rows.append([f"Rows per {factor}", counts])
    rows.append(["Unique prompt_hash values", int(df["prompt_hash"].nunique())])
    return md_table(["Metric", "Value"], rows)


def selected_sample_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain, content_id, role, paraphrase, register, layer in SAMPLE_CELL_SPECS:
        match = df[
            (df["content_domain"] == domain)
            & (df["content_id"] == content_id)
            & (df["role_framing"] == role)
            & (df["surface_paraphrase"] == paraphrase)
            & (df["language_register"] == register)
            & (df["layer"] == layer)
        ]
        if len(match) != 1:
            raise RuntimeError(f"Expected one sample cell for {(domain, content_id, role, paraphrase, register, layer)}, saw {len(match)}")
        rows.append(match.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def sample_features_markdown(sample_rows: pd.DataFrame) -> str:
    sections = []
    for i, row in sample_rows.iterrows():
        prompt = " ".join(str(row["rendered_prompt"]).split())
        prompt_short = prompt[:100] + ("..." if len(prompt) > 100 else "")
        feature_text = ", ".join(f"{idx}: {sig3(value)}" for idx, value in top_k_features(row, 5))
        sections.append(
            f"### Cell {i + 1}\n\n"
            f"- Factors: {row['content_domain']} / {row['content_id']} / {row['role_framing']} / "
            f"{row['surface_paraphrase']} / {row['language_register']} / layer {row['layer']}\n"
            f"- Rendered prompt first 100 chars: `{prompt_short}`\n"
            f"- Top-5 features: {feature_text}\n"
        )
    return "\n".join(sections)


def topk_stability_tables(df: pd.DataFrame) -> str:
    sections = []
    base = df[
        (df["layer"] == 12)
        & (df["role_framing"] == "direct_question")
        & (df["surface_paraphrase"] == "canonical")
        & (df["language_register"] == "formal")
    ]
    for domain in ["factual_recall", "arithmetic", "simple_reasoning"]:
        subset = base[base["content_domain"] == domain].sort_values("content_id").reset_index(drop=True)
        labels = subset["content_id"].tolist()
        sets = [top_k_set(row, 50) for _, row in subset.iterrows()]
        rows = []
        for label, a in zip(labels, sets):
            rows.append([label, *[sig3(jaccard(a, b)) for b in sets]])
        sections.append(f"### {domain}\n\n{md_table(['content_id', *labels], rows)}")
    return "\n\n".join(sections)


def load_saes_for_decode(device: torch.device) -> dict[int, Any]:
    from sae_lens import SAE

    saes = {}
    for layer in LAYER_LEVELS.values():
        release, sae_id = discover_residual_width16k_sae(layer)
        sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=str(device), dtype="float32")
        sae.eval()
        saes[layer] = sae
    return saes


def sampled_reconstruction_cosines(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    device, _, _ = choose_device_and_dtype()
    saes = load_saes_for_decode(device)
    samples = []
    for layer, layer_df in df.groupby("layer"):
        samples.append(layer_df.sample(n=10, random_state=20260508 + int(layer)))
    sample_df = pd.concat(samples, ignore_index=True)
    rows = []
    flags = []
    with torch.no_grad():
        for _, row in sample_df.iterrows():
            activation = torch.tensor(np.asarray(row["raw_activation"], dtype=np.float32), device=device).unsqueeze(0)
            encoded = torch.zeros((1, 16384), dtype=torch.float32, device=device)
            indices = torch.tensor(np.asarray(row["sae_indices"], dtype=np.int64), device=device)
            values = torch.tensor(np.asarray(row["sae_values"], dtype=np.float32), device=device)
            encoded[0, indices] = values
            reconstructed = saes[int(row["layer"])].decode(encoded).float()
            cosine = float(F.cosine_similarity(reconstructed, activation, dim=-1).detach().cpu().item())
            rows.append({**row[["content_domain", "content_id", "role_framing", "surface_paraphrase", "language_register", "layer"]].to_dict(), "cosine": cosine})
            if cosine < 0.85:
                flags.append(describe_row(row, f"sample reconstruction cosine {cosine:.3g} below 0.85"))
    return pd.DataFrame(rows), flags


def describe_row(row: pd.Series, reason: str) -> str:
    return (
        f"{reason}: {row['content_domain']} / {row['content_id']} / {row['role_framing']} / "
        f"{row['surface_paraphrase']} / {row['language_register']} / layer {row['layer']}"
    )


def sanity_flags(df: pd.DataFrame) -> list[str]:
    flags: list[str] = []
    for layer, layer_df in df.groupby("layer"):
        mean = layer_df["activation_norm"].mean()
        std = layer_df["activation_norm"].std()
        outliers = layer_df[(layer_df["activation_norm"] - mean).abs() > 3 * std]
        for _, row in outliers.iterrows():
            flags.append(describe_row(row, f"activation_norm {row['activation_norm']:.3g} more than 3 SD from layer {layer} mean"))

    bad_nonzero = df[~df["nonzero_features"].between(50, 2500)]
    for _, row in bad_nonzero.iterrows():
        flags.append(describe_row(row, f"nonzero_features {row['nonzero_features']} outside [50, 2500]"))

    prompt_counts = df["prompt_hash"].value_counts()
    bad_hashes = prompt_counts[prompt_counts != 3]
    for prompt_hash, count in bad_hashes.items():
        rows = df[df["prompt_hash"] == prompt_hash]
        first = rows.iloc[0]
        flags.append(describe_row(first, f"prompt_hash {prompt_hash[:12]} appears in {int(count)} rows, expected 3"))
    return flags


def render_report(
    *,
    df: pd.DataFrame,
    sample_rows: pd.DataFrame,
    reconstruction_sample: pd.DataFrame,
    flags: list[str],
    figures_dir: Path,
    wall_clock_seconds: float,
) -> str:
    reconstruction_rows = [
        [
            int(row["layer"]),
            row["content_domain"],
            row["content_id"],
            row["role_framing"],
            row["surface_paraphrase"],
            row["language_register"],
            sig3(row["cosine"]),
        ]
        for _, row in reconstruction_sample.iterrows()
    ]
    flags_text = "\n".join(f"- {flag}" for flag in flags) if flags else "- None."
    return f"""# SAE feature stability inspection report

Runtime: {wall_clock_seconds:.2f} seconds

## 1. Dataset summary

{dataset_summary_table(df)}

## 2. Per-layer scalar statistics

{scalar_stats_table(df)}

## 3. Activation norm distributions by factor

![Activation norms by factor](figures/activation_norms_by_factor.pdf)

## 4. Nonzero feature count distributions by factor

![Nonzero features by factor](figures/nonzero_features_by_factor.pdf)

## 5. Top SAE features for selected cells

![Top features per selected cell](figures/top_features_per_cell.pdf)

{sample_features_markdown(sample_rows)}

## 6. Feature overlap heatmap, factual_recall / fr1 / layer 12

![Feature overlap heatmap](figures/feature_overlap_heatmap.pdf)

## 7. Top-K feature stability across content_id

{topk_stability_tables(df)}

## 8. Sanity flags

Reconstruction cosine spot check, 10 sampled cells per layer:

{md_table(['Layer', 'Domain', 'Content ID', 'Role', 'Paraphrase', 'Register', 'Cosine'], reconstruction_rows)}

Flags:

{flags_text}
"""


def run_inspection(repo_root: Path) -> str:
    started = time.perf_counter()
    output_dir = repo_root / "output"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(output_dir / "activations.parquet", engine="pyarrow")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    corpus = load_corpus(repo_root / "data" / "corpus.csv")
    df = add_rendered_prompts(df, corpus, tokenizer)

    plot_box_grid(
        df,
        "activation_norm",
        "Activation L2 norm",
        "Activation norm distributions by factor",
        figures_dir / "activation_norms_by_factor.pdf",
    )
    plot_box_grid(
        df,
        "nonzero_features",
        "Nonzero SAE features",
        "Nonzero feature count distributions by factor",
        figures_dir / "nonzero_features_by_factor.pdf",
    )
    sample_rows = selected_sample_rows(df)
    plot_top_features_per_cell(sample_rows, figures_dir / "top_features_per_cell.pdf")
    plot_feature_overlap_heatmap(df, figures_dir / "feature_overlap_heatmap.pdf")

    reconstruction_sample, reconstruction_flags = sampled_reconstruction_cosines(df)
    flags = [*sanity_flags(df), *reconstruction_flags]

    report = render_report(
        df=df,
        sample_rows=sample_rows,
        reconstruction_sample=reconstruction_sample,
        flags=flags,
        figures_dir=figures_dir,
        wall_clock_seconds=time.perf_counter() - started,
    )
    (output_dir / "inspection_report.md").write_text(report)
    return report


def main() -> None:
    print(run_inspection(project_root()))
