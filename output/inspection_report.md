# SAE feature stability inspection report

Runtime: 8.12 seconds

## 1. Dataset summary

| Metric | Value |
| --- | --- |
| Total rows | 1215 |
| Rows layer 6 | 405 |
| Rows layer 12 | 405 |
| Rows layer 20 | 405 |
| Rows per content_domain | arithmetic: 405, factual_recall: 405, simple_reasoning: 405 |
| Rows per role_framing | direct_question: 405, fill_blank: 405, multi_turn: 405 |
| Rows per surface_paraphrase | canonical: 405, lexical: 405, syntactic: 405 |
| Rows per language_register | casual: 405, formal: 405, instruction: 405 |
| Unique prompt_hash values | 405 |

## 2. Per-layer scalar statistics

| Layer | Label | Act min | Act mean | Act median | Act max | NZ min | NZ mean | NZ median | NZ max | Tok min | Tok mean | Tok median | Tok max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | early | 77.2 | 85.6 | 84.2 | 98.3 | 147 | 204 | 203 | 302 | 15 | 34 | 29 | 66 |
| 12 | middle | 158 | 179 | 178 | 203 | 338 | 471 | 467 | 644 | 15 | 34 | 29 | 66 |
| 20 | late | 315 | 372 | 371 | 434 | 256 | 369 | 364 | 513 | 15 | 34 | 29 | 66 |

## 3. Activation norm distributions by factor

![Activation norms by factor](figures/activation_norms_by_factor.pdf)

## 4. Nonzero feature count distributions by factor

![Nonzero features by factor](figures/nonzero_features_by_factor.pdf)

## 5. Top SAE features for selected cells

![Top features per selected cell](figures/top_features_per_cell.pdf)

### Cell 1

- Factors: factual_recall / fr1 / direct_question / canonical / formal / layer 12
- Rendered prompt first 100 chars: `<bos><start_of_turn>user Could you please tell me: What is the capital of France?<end_of_turn> <star...`
- Top-5 features: 1178: 59.9, 9412: 31.9, 1899: 29.9, 4952: 29, 10222: 20.3

### Cell 2

- Factors: factual_recall / fr1 / direct_question / canonical / casual / layer 12
- Rendered prompt first 100 chars: `<bos><start_of_turn>user what is the capital of france<end_of_turn> <start_of_turn>model`
- Top-5 features: 1178: 56.5, 9412: 31.9, 1899: 30.5, 4952: 28.1, 10222: 21.7

### Cell 3

- Factors: arithmetic / ar1 / direct_question / canonical / formal / layer 12
- Rendered prompt first 100 chars: `<bos><start_of_turn>user Could you please tell me: What is 23 plus 19?<end_of_turn> <start_of_turn>m...`
- Top-5 features: 1178: 68.4, 9412: 29.7, 1899: 25.5, 4952: 25.2, 6104: 22.6

### Cell 4

- Factors: simple_reasoning / sr1 / direct_question / canonical / formal / layer 12
- Rendered prompt first 100 chars: `<bos><start_of_turn>user Could you please tell me: All roses are flowers. This is a rose. Therefore,...`
- Top-5 features: 1178: 51.2, 4952: 27.2, 1899: 25.3, 9412: 25.2, 3892: 19

### Cell 5

- Factors: factual_recall / fr1 / multi_turn / canonical / formal / layer 12
- Rendered prompt first 100 chars: `<bos><start_of_turn>user I'm studying European geography.<end_of_turn> <start_of_turn>model Happy to...`
- Top-5 features: 1178: 47.4, 4952: 28.1, 9412: 26.9, 1899: 25.4, 3892: 19

### Cell 6

- Factors: factual_recall / fr1 / direct_question / canonical / formal / layer 20
- Rendered prompt first 100 chars: `<bos><start_of_turn>user Could you please tell me: What is the capital of France?<end_of_turn> <star...`
- Top-5 features: 7161: 117, 9697: 63.6, 16031: 58.1, 1402: 55.8, 9213: 50.3


## 6. Feature overlap heatmap, factual_recall / fr1 / layer 12

![Feature overlap heatmap](figures/feature_overlap_heatmap.pdf)

## 7. Top-K feature stability across content_id

### factual_recall

| content_id | fr1 | fr2 | fr3 | fr4 | fr5 |
| --- | --- | --- | --- | --- | --- |
| fr1 | 1 | 0.923 | 0.408 | 0.493 | 0.429 |
| fr2 | 0.923 | 1 | 0.408 | 0.493 | 0.449 |
| fr3 | 0.408 | 0.408 | 1 | 0.471 | 0.695 |
| fr4 | 0.493 | 0.493 | 0.471 | 1 | 0.515 |
| fr5 | 0.429 | 0.449 | 0.695 | 0.515 | 1 |

### arithmetic

| content_id | ar1 | ar2 | ar3 | ar4 | ar5 |
| --- | --- | --- | --- | --- | --- |
| ar1 | 1 | 0.852 | 0.754 | 0.667 | 0.961 |
| ar2 | 0.852 | 1 | 0.818 | 0.724 | 0.818 |
| ar3 | 0.754 | 0.818 | 1 | 0.695 | 0.724 |
| ar4 | 0.667 | 0.724 | 0.695 | 1 | 0.639 |
| ar5 | 0.961 | 0.818 | 0.724 | 0.639 | 1 |

### simple_reasoning

| content_id | sr1 | sr2 | sr3 | sr4 | sr5 |
| --- | --- | --- | --- | --- | --- |
| sr1 | 1 | 0.587 | 0.333 | 0.613 | 0.408 |
| sr2 | 0.587 | 1 | 0.471 | 0.695 | 0.587 |
| sr3 | 0.333 | 0.471 | 1 | 0.408 | 0.538 |
| sr4 | 0.613 | 0.695 | 0.408 | 1 | 0.538 |
| sr5 | 0.408 | 0.587 | 0.538 | 0.538 | 1 |

## 8. Sanity flags

Reconstruction cosine spot check, 10 sampled cells per layer:

| Layer | Domain | Content ID | Role | Paraphrase | Register | Cosine |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | arithmetic | ar1 | multi_turn | syntactic | casual | 0.946 |
| 6 | factual_recall | fr1 | multi_turn | canonical | formal | 0.945 |
| 6 | arithmetic | ar1 | fill_blank | syntactic | instruction | 0.937 |
| 6 | simple_reasoning | sr5 | multi_turn | syntactic | formal | 0.948 |
| 6 | simple_reasoning | sr2 | fill_blank | syntactic | instruction | 0.937 |
| 6 | factual_recall | fr5 | fill_blank | canonical | formal | 0.937 |
| 6 | factual_recall | fr2 | multi_turn | lexical | instruction | 0.948 |
| 6 | simple_reasoning | sr3 | multi_turn | lexical | formal | 0.946 |
| 6 | arithmetic | ar2 | direct_question | lexical | formal | 0.945 |
| 6 | simple_reasoning | sr4 | multi_turn | lexical | formal | 0.948 |
| 12 | simple_reasoning | sr3 | fill_blank | lexical | casual | 0.931 |
| 12 | factual_recall | fr3 | direct_question | lexical | instruction | 0.946 |
| 12 | simple_reasoning | sr2 | direct_question | canonical | casual | 0.94 |
| 12 | simple_reasoning | sr4 | fill_blank | canonical | formal | 0.926 |
| 12 | arithmetic | ar2 | fill_blank | canonical | instruction | 0.939 |
| 12 | arithmetic | ar1 | multi_turn | syntactic | formal | 0.953 |
| 12 | factual_recall | fr1 | fill_blank | lexical | formal | 0.937 |
| 12 | arithmetic | ar1 | fill_blank | canonical | formal | 0.941 |
| 12 | simple_reasoning | sr2 | direct_question | syntactic | formal | 0.946 |
| 12 | factual_recall | fr4 | fill_blank | syntactic | formal | 0.934 |
| 20 | factual_recall | fr5 | multi_turn | syntactic | instruction | 0.919 |
| 20 | arithmetic | ar5 | fill_blank | canonical | instruction | 0.918 |
| 20 | arithmetic | ar3 | multi_turn | syntactic | formal | 0.921 |
| 20 | simple_reasoning | sr1 | direct_question | lexical | formal | 0.905 |
| 20 | arithmetic | ar4 | fill_blank | canonical | formal | 0.923 |
| 20 | arithmetic | ar1 | fill_blank | syntactic | casual | 0.924 |
| 20 | simple_reasoning | sr4 | fill_blank | syntactic | casual | 0.918 |
| 20 | simple_reasoning | sr2 | direct_question | lexical | instruction | 0.918 |
| 20 | factual_recall | fr1 | multi_turn | lexical | casual | 0.918 |
| 20 | arithmetic | ar3 | fill_blank | lexical | instruction | 0.915 |

Flags:

- None.
