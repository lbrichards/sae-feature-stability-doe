from __future__ import annotations

from dataclasses import dataclass


MODEL_NAME = "google/gemma-2-2b-it"
EXPECTED_CORPUS_ROWS = 405
EXPECTED_OUTPUT_ROWS = 1215
EXPECTED_LAYERS = 26
EXPECTED_HIDDEN_DIM = 2304
EXPECTED_SAE_FEATURES = 16384

CORPUS_FACTOR_COLUMNS = [
    "content_domain",
    "role_framing",
    "surface_paraphrase",
    "language_register",
]

REQUIRED_CORPUS_COLUMNS = [
    "content_domain",
    "content_id",
    "role_framing",
    "surface_paraphrase",
    "language_register",
    "is_multi_turn",
    "prompt",
]

LAYER_LEVELS = {
    "early": 6,
    "middle": 12,
    "late": 20,
}

SCALAR_OUTPUT_COLUMNS = [
    "content_domain",
    "content_id",
    "role_framing",
    "surface_paraphrase",
    "language_register",
    "layer",
    "layer_label",
    "prompt_hash",
    "prompt_token_len",
    "activation_norm",
    "nonzero_features",
]

OUTPUT_COLUMNS = [
    *SCALAR_OUTPUT_COLUMNS,
    "raw_activation",
    "sae_indices",
    "sae_values",
]


@dataclass(frozen=True)
class LayerSpec:
    label: str
    index: int
    sae_release: str
    sae_id: str

    @property
    def sae_identifier(self) -> str:
        return f"{self.sae_release}/{self.sae_id}"
