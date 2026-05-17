# SAE feature stability collection report

## Model

- Name: `google/gemma-2-2b-it`
- Revision hash: `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8`
- Number of layers: 26
- Hidden dim: 2304

## SAEs

- early (6): `gemma-scope-2b-pt-res/layer_6/width_16k/average_l0_301`
- middle (12): `gemma-scope-2b-pt-res/layer_12/width_16k/average_l0_445`
- late (20): `gemma-scope-2b-pt-res/layer_20/width_16k/average_l0_294`

## Runtime

- Dtype and device: MPS float32
- Wall-clock time: 47.37 seconds
- Output format: parquet with nested list columns for raw activations and sparse SAE encodings.

## Per-layer summary

- Layer 6: mean activation L2 norm 85.6142; mean nonzero features 203.654; mean reconstruction cosine over 10 random prompts 0.94249
- Layer 12: mean activation L2 norm 178.529; mean nonzero features 471.225; mean reconstruction cosine over 10 random prompts 0.945986
- Layer 20: mean activation L2 norm 371.542; mean nonzero features 368.684; mean reconstruction cosine over 10 random prompts 0.916783

## Factor balance

- Layer 6:
  - content_domain: OK {'arithmetic': 135, 'factual_recall': 135, 'simple_reasoning': 135}
  - role_framing: OK {'direct_question': 135, 'fill_blank': 135, 'multi_turn': 135}
  - surface_paraphrase: OK {'canonical': 135, 'lexical': 135, 'syntactic': 135}
  - language_register: OK {'casual': 135, 'formal': 135, 'instruction': 135}
- Layer 12:
  - content_domain: OK {'arithmetic': 135, 'factual_recall': 135, 'simple_reasoning': 135}
  - role_framing: OK {'direct_question': 135, 'fill_blank': 135, 'multi_turn': 135}
  - surface_paraphrase: OK {'canonical': 135, 'lexical': 135, 'syntactic': 135}
  - language_register: OK {'casual': 135, 'formal': 135, 'instruction': 135}
- Layer 20:
  - content_domain: OK {'arithmetic': 135, 'factual_recall': 135, 'simple_reasoning': 135}
  - role_framing: OK {'direct_question': 135, 'fill_blank': 135, 'multi_turn': 135}
  - surface_paraphrase: OK {'canonical': 135, 'lexical': 135, 'syntactic': 135}
  - language_register: OK {'casual': 135, 'formal': 135, 'instruction': 135}

## Output

- File: `output/activations.parquet`
- Size: 14907227 bytes
- Rows: 1215

STATUS: VERIFIED
