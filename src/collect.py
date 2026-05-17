from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any

from huggingface_hub import HfApi
import numpy as np
import pandas as pd
import pyarrow as pa
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.schema import (
    CORPUS_FACTOR_COLUMNS,
    EXPECTED_CORPUS_ROWS,
    EXPECTED_HIDDEN_DIM,
    EXPECTED_LAYERS,
    EXPECTED_OUTPUT_ROWS,
    EXPECTED_SAE_FEATURES,
    LAYER_LEVELS,
    MODEL_NAME,
    OUTPUT_COLUMNS,
    REQUIRED_CORPUS_COLUMNS,
    LayerSpec,
)


NONZERO_WARMUP_RANGE = (100, 2000)
NONZERO_OUTPUT_RANGE = (50, 2500)
WARMUP_COSINE_THRESHOLD = 0.85
RANDOM_SAMPLE_SIZE = 10


class VerificationFailure(RuntimeError):
    pass


def project_root() -> Path:
    return Path.cwd().resolve()


def choose_device_and_dtype() -> tuple[torch.device, torch.dtype, str]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16, "CUDA float16"
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32, "MPS float32"
    return torch.device("cpu"), torch.float32, "CPU float32"


def model_revision_hash(model_name: str) -> str:
    try:
        return HfApi().model_info(model_name).sha or "unknown"
    except Exception:
        return "unknown"


def average_l0(sae_id: str) -> int:
    match = re.search(r"average_l0_(\d+)", sae_id)
    return int(match.group(1)) if match else -1


def discover_residual_width16k_sae(layer: int) -> tuple[str, str]:
    try:
        from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

        directory = get_pretrained_saes_directory()
    except Exception:
        return "gemma-scope-2b-pt-res", f"layer_{layer}/width_16k/average_l0_82"

    candidates: list[tuple[str, str]] = []
    layer_token = f"layer_{layer}"
    for release, entry in directory.items():
        sae_ids = getattr(entry, "saes_map", None)
        if isinstance(sae_ids, dict):
            sae_ids = sae_ids.keys()
        if sae_ids is None:
            continue
        for sae_id in sae_ids:
            release_s = str(release)
            sae_id_s = str(sae_id)
            haystack = f"{release_s}/{sae_id_s}".lower()
            if (
                "gemma-scope-2b" in haystack
                and "pt-res" in haystack
                and layer_token in haystack
                and "width_16k" in haystack
            ):
                candidates.append((release_s, sae_id_s))

    if not candidates:
        raise VerificationFailure(f"No Gemma Scope 2B residual width_16k SAE found for layer {layer}")

    average_l0_candidates = [c for c in candidates if "average_l0" in c[1]]
    average_l0_candidates.sort(key=lambda c: average_l0(c[1]), reverse=True)
    canonical_candidates = [c for c in candidates if "canonical" in c[1]]
    return (average_l0_candidates or canonical_candidates or candidates)[0]


def load_sae(layer_label: str, layer: int, device: torch.device) -> tuple[Any, LayerSpec]:
    from sae_lens import SAE

    release, sae_id = discover_residual_width16k_sae(layer)
    sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=str(device), dtype="float32")
    sae.eval()

    d_in = int(getattr(sae.cfg, "d_in", -1))
    d_sae = int(getattr(sae.cfg, "d_sae", -1))
    if d_in != EXPECTED_HIDDEN_DIM:
        raise VerificationFailure(f"SAE {release}/{sae_id} input dim {d_in}, expected {EXPECTED_HIDDEN_DIM}")
    if d_sae != EXPECTED_SAE_FEATURES:
        raise VerificationFailure(f"SAE {release}/{sae_id} feature dim {d_sae}, expected {EXPECTED_SAE_FEATURES}")

    return sae, LayerSpec(label=layer_label, index=layer, sae_release=release, sae_id=sae_id)


def load_corpus(corpus_path: Path) -> pd.DataFrame:
    if not corpus_path.exists():
        raise VerificationFailure(f"Missing input corpus: {corpus_path}")

    corpus = pd.read_csv(corpus_path, dtype=str)
    missing = [col for col in REQUIRED_CORPUS_COLUMNS if col not in corpus.columns]
    if missing:
        raise VerificationFailure(f"Corpus missing required columns: {missing}")
    if len(corpus) != EXPECTED_CORPUS_ROWS:
        raise VerificationFailure(f"Corpus has {len(corpus)} rows, expected {EXPECTED_CORPUS_ROWS}")

    for column in CORPUS_FACTOR_COLUMNS:
        counts = corpus[column].value_counts(dropna=False).to_dict()
        if len(counts) != 3 or any(count != EXPECTED_CORPUS_ROWS // 3 for count in counts.values()):
            raise VerificationFailure(f"Corpus factor {column} is not balanced at 135 rows per level: {counts}")

    return corpus


def corpus_row_to_messages(row: pd.Series) -> list[dict[str, str]]:
    is_multi_turn = str(row["is_multi_turn"]).strip().lower() == "true"
    if not is_multi_turn:
        return [{"role": "user", "content": str(row["prompt"])}]

    turns = json.loads(str(row["prompt"]))
    if not isinstance(turns, list):
        raise VerificationFailure(f"Multi-turn prompt for content_id={row['content_id']} did not decode to a list")
    for turn in turns:
        if not isinstance(turn, dict) or "role" not in turn or "content" not in turn:
            raise VerificationFailure(f"Malformed chat turn for content_id={row['content_id']}: {turn}")
    return turns


def render_prompt(tokenizer: Any, row: pd.Series) -> str:
    return tokenizer.apply_chat_template(corpus_row_to_messages(row), tokenize=False, add_generation_prompt=True)


def encode_activation(sae: Any, activation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sae_dtype = next(sae.parameters()).dtype
    encoded = sae.encode(activation.to(dtype=sae_dtype))
    reconstructed = sae.decode(encoded)
    return encoded.float(), reconstructed.float()


def sparse_encoding(encoded: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    flat = encoded.squeeze(0).detach().cpu().float()
    indices = torch.nonzero(flat != 0, as_tuple=False).squeeze(-1).to(torch.int32)
    values = flat[indices].to(torch.float32)
    return indices.numpy().astype(np.int32), values.numpy().astype(np.float32)


def run_warmup(
    *,
    tokenizer: Any,
    model: Any,
    saes: dict[int, Any],
    corpus: pd.DataFrame,
    device: torch.device,
) -> dict[int, float]:
    rendered = render_prompt(tokenizer, corpus.iloc[0])
    inputs = tokenizer(rendered, return_tensors="pt").to(device)
    warmup_cosines: dict[int, float] = {}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        for layer, sae in saes.items():
            activation = outputs.hidden_states[layer + 1][:, -1, :].float()
            if tuple(activation.shape) != (1, EXPECTED_HIDDEN_DIM):
                raise VerificationFailure(f"Warmup activation shape at layer {layer} was {tuple(activation.shape)}")
            encoded, reconstructed = encode_activation(sae, activation)
            nonzero = int((encoded != 0).sum().detach().cpu().item())
            if not NONZERO_WARMUP_RANGE[0] <= nonzero <= NONZERO_WARMUP_RANGE[1]:
                raise VerificationFailure(f"Warmup nonzero count at layer {layer} was {nonzero}")
            cosine = float(F.cosine_similarity(reconstructed, activation, dim=-1).detach().cpu().item())
            if cosine <= WARMUP_COSINE_THRESHOLD:
                raise VerificationFailure(f"Warmup reconstruction cosine at layer {layer} was {cosine:.6f}")
            warmup_cosines[layer] = cosine

    return warmup_cosines


def collect_records(
    *,
    tokenizer: Any,
    model: Any,
    saes: dict[int, Any],
    layer_specs: dict[int, LayerSpec],
    corpus: pd.DataFrame,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[int, list[float]]]:
    rng = np.random.default_rng(20260508)
    sampled_indices = set(rng.choice(len(corpus), size=RANDOM_SAMPLE_SIZE, replace=False).tolist())
    sample_cosines = {layer: [] for layer in layer_specs}
    records: list[dict[str, Any]] = []

    for row_index, row in corpus.iterrows():
        if row_index == 0 or (row_index + 1) % 25 == 0 or row_index + 1 == len(corpus):
            print(f"Collecting prompt {row_index + 1}/{len(corpus)}", flush=True)

        rendered = render_prompt(tokenizer, row)
        prompt_hash = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        inputs = tokenizer(rendered, return_tensors="pt").to(device)
        prompt_token_len = int(inputs["input_ids"].shape[-1])

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            for layer, spec in layer_specs.items():
                activation = outputs.hidden_states[layer + 1][:, -1, :].float()
                encoded, reconstructed = encode_activation(saes[layer], activation)
                indices, values = sparse_encoding(encoded)
                activation_np = activation.squeeze(0).detach().cpu().numpy().astype(np.float32)

                if row_index in sampled_indices:
                    cosine = float(F.cosine_similarity(reconstructed, activation, dim=-1).detach().cpu().item())
                    sample_cosines[layer].append(cosine)

                records.append(
                    {
                        "content_domain": row["content_domain"],
                        "content_id": row["content_id"],
                        "role_framing": row["role_framing"],
                        "surface_paraphrase": row["surface_paraphrase"],
                        "language_register": row["language_register"],
                        "layer": np.int32(layer),
                        "layer_label": spec.label,
                        "prompt_hash": prompt_hash,
                        "prompt_token_len": np.int32(prompt_token_len),
                        "activation_norm": np.float32(np.linalg.norm(activation_np)),
                        "nonzero_features": np.int32(len(indices)),
                        "raw_activation": activation_np.tolist(),
                        "sae_indices": indices.tolist(),
                        "sae_values": values.tolist(),
                    }
                )

    return pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS), sample_cosines


def validate_output(df: pd.DataFrame) -> None:
    if len(df) != EXPECTED_OUTPUT_ROWS:
        raise VerificationFailure(f"Output has {len(df)} rows, expected {EXPECTED_OUTPUT_ROWS}")
    if (df["activation_norm"] <= 0).any():
        raise VerificationFailure("At least one row has an all-zero activation")
    if (df["prompt_token_len"] <= 0).any():
        raise VerificationFailure("At least one row has non-positive prompt_token_len")

    low, high = NONZERO_OUTPUT_RANGE
    invalid_nonzero = ~df["nonzero_features"].between(low, high)
    if invalid_nonzero.any():
        bad = df.loc[invalid_nonzero, ["content_id", "layer", "nonzero_features"]].head(10).to_dict("records")
        raise VerificationFailure(f"Some rows have nonzero_features outside {low}-{high}: {bad}")


def parquet_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("content_domain", pa.string()),
            pa.field("content_id", pa.string()),
            pa.field("role_framing", pa.string()),
            pa.field("surface_paraphrase", pa.string()),
            pa.field("language_register", pa.string()),
            pa.field("layer", pa.int32()),
            pa.field("layer_label", pa.string()),
            pa.field("prompt_hash", pa.string()),
            pa.field("prompt_token_len", pa.int32()),
            pa.field("activation_norm", pa.float32()),
            pa.field("nonzero_features", pa.int32()),
            pa.field("raw_activation", pa.list_(pa.float32())),
            pa.field("sae_indices", pa.list_(pa.int32())),
            pa.field("sae_values", pa.list_(pa.float32())),
        ]
    )


def write_activations_parquet(df: pd.DataFrame, output_path: Path) -> None:
    df = df.copy()
    df["layer"] = df["layer"].astype("int32")
    df["prompt_token_len"] = df["prompt_token_len"].astype("int32")
    df["activation_norm"] = df["activation_norm"].astype("float32")
    df["nonzero_features"] = df["nonzero_features"].astype("int32")
    df.to_parquet(output_path, engine="pyarrow", index=False, schema=parquet_schema())


def factor_balance_lines(df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    for layer in sorted(df["layer"].unique()):
        layer_df = df[df["layer"] == layer]
        lines.append(f"- Layer {layer}:")
        for column in CORPUS_FACTOR_COLUMNS:
            counts = layer_df[column].value_counts().sort_index().to_dict()
            ok = len(counts) == 3 and all(count == EXPECTED_CORPUS_ROWS // 3 for count in counts.values())
            lines.append(f"  - {column}: {'OK' if ok else 'FAILED'} {counts}")
    return lines


def render_report(
    *,
    status_line: str,
    model_revision: str,
    layer_specs: dict[int, LayerSpec],
    dtype_device_policy: str,
    wall_clock_seconds: float,
    df: pd.DataFrame | None,
    sample_cosines: dict[int, list[float]] | None,
    output_path: Path,
    failure_reason: str | None = None,
) -> str:
    sae_lines = "\n".join(
        f"- {spec.label} ({layer}): `{spec.sae_identifier}`"
        for layer, spec in sorted(layer_specs.items())
    )

    if df is None:
        summary = "Collection did not complete."
        balance = "- Not available."
        output_info = "- Output file was not written."
    else:
        per_layer_lines = []
        for layer, layer_df in df.groupby("layer"):
            mean_norm = float(layer_df["activation_norm"].mean())
            mean_nonzero = float(layer_df["nonzero_features"].mean())
            mean_cosine = float(np.mean(sample_cosines[layer])) if sample_cosines and sample_cosines[layer] else float("nan")
            per_layer_lines.append(
                f"- Layer {layer}: mean activation L2 norm {mean_norm:.6g}; "
                f"mean nonzero features {mean_nonzero:.6g}; "
                f"mean reconstruction cosine over {RANDOM_SAMPLE_SIZE} random prompts {mean_cosine:.6g}"
            )
        summary = "\n".join(per_layer_lines)
        balance = "\n".join(factor_balance_lines(df))
        size = output_path.stat().st_size if output_path.exists() else 0
        output_info = f"- File: `{output_path}`\n- Size: {size} bytes\n- Rows: {len(df)}"

    failure = f"\n## Failure\n\n{failure_reason}\n" if failure_reason else ""
    return f"""# SAE feature stability collection report

## Model

- Name: `{MODEL_NAME}`
- Revision hash: `{model_revision}`
- Number of layers: {EXPECTED_LAYERS}
- Hidden dim: {EXPECTED_HIDDEN_DIM}

## SAEs

{sae_lines or "- Not loaded."}

## Runtime

- Dtype and device: {dtype_device_policy}
- Wall-clock time: {wall_clock_seconds:.2f} seconds
- Output format: parquet with nested list columns for raw activations and sparse SAE encodings.

## Per-layer summary

{summary}

## Factor balance

{balance}

## Output

{output_info}
{failure}
{status_line}
"""


def run_collection(repo_root: Path) -> str:
    started = time.perf_counter()
    output_dir = repo_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "activations.parquet"
    report_path = output_dir / "collection_report.md"

    layer_specs: dict[int, LayerSpec] = {}
    df: pd.DataFrame | None = None
    sample_cosines: dict[int, list[float]] | None = None
    device, dtype, dtype_device_policy = choose_device_and_dtype()

    try:
        corpus = load_corpus(repo_root / "data" / "corpus.csv")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype, low_cpu_mem_usage=True)
        model.to(device)
        model.eval()

        num_layers = int(model.config.num_hidden_layers)
        hidden_dim = int(model.config.hidden_size)
        if num_layers != EXPECTED_LAYERS:
            raise VerificationFailure(f"Model has {num_layers} layers, expected {EXPECTED_LAYERS}")
        if hidden_dim != EXPECTED_HIDDEN_DIM:
            raise VerificationFailure(f"Model hidden dim is {hidden_dim}, expected {EXPECTED_HIDDEN_DIM}")

        saes: dict[int, Any] = {}
        for label, layer in LAYER_LEVELS.items():
            sae, spec = load_sae(label, layer, device)
            saes[layer] = sae
            layer_specs[layer] = spec

        run_warmup(tokenizer=tokenizer, model=model, saes=saes, corpus=corpus, device=device)
        df, sample_cosines = collect_records(
            tokenizer=tokenizer,
            model=model,
            saes=saes,
            layer_specs=layer_specs,
            corpus=corpus,
            device=device,
        )
        validate_output(df)
        write_activations_parquet(df, output_path)
        status_line = "STATUS: VERIFIED"
        failure_reason = None
    except Exception as exc:
        status_line = f"STATUS: FAILED - {exc}"
        failure_reason = str(exc)

    report = render_report(
        status_line=status_line,
        model_revision=model_revision_hash(MODEL_NAME),
        layer_specs=layer_specs,
        dtype_device_policy=dtype_device_policy,
        wall_clock_seconds=time.perf_counter() - started,
        df=df,
        sample_cosines=sample_cosines,
        output_path=output_path,
        failure_reason=failure_reason,
    )
    report_path.write_text(report)
    return report


def main() -> None:
    print(run_collection(project_root()))
