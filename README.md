# G-TEx: Graph-Guided Textual Explanation Generation Framework

Implementation for the paper **"Graph-Guided Textual Explanation
Generation Framework"** (EMNLP 2025, Main).

G-TEx improves the *faithfulness* of natural-language explanations (NLEs) by
grounding generation in extractive highlight cues:

1. **Extract** highlight (extractive / attribution) explanations that mark the
   input tokens most responsible for a model's prediction.
2. **Build a graph** over those highlighted tokens.
3. **Encode** the graph with a single GNN layer injected into the language
   model.
4. **Guide** NLE generation with the graph-conditioned representations.

This yields up to a 12.18% faithfulness gain over baselines.

## Repository layout

```
models/          GNN-fused language models
  t5_gnn.py         T5 + GNN
  bart_gnn.py       BART + GNN
  llama3_gnn.py     Llama-3 + GNN
utils/           preprocessors + graph-building dataloaders
  preprocessor.py            esnli / ecqa / comve / cose
  preprocessor_for_faith.py  test-only, adds the inserted word
  dataloader.py              high_tokens graph mode
  dataloader_span_pair.py    span-pair graph mode
  dataloader_token_pair.py   token-pair graph mode
  dataloader_prompt.py       prompt baseline
  dataloader_for_faith.py    faithfulness loader
  dataloader_llama.py        causal loader for Llama-3
prepare_input/   prepare_input_for_gnn.py
eval/            faithfulness.py
scripts/         parameterized train/test runners
train.py         seq2seq (T5 / BART) entrypoint
train_llama.py   Llama-3 entrypoint
data/            place prepared JSON here
```

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Data preparation

Convert highlight explanations into the flat format
consumed by the loaders:

```bash
python prepare_input/prepare_input_for_gnn.py \
  --input  raw_annotations.jsonl \
  --output data/comve_train.json \
  --mode   high_tokens          # high_tokens | span_pair | token_pair
```

## Training & testing

Seq2seq (T5 / BART), configurable via environment variables:

```bash
# Train
MODEL_NAME=google-t5/t5-large GNN=gcn GRAPH_MODE=high_tokens \
DATA_FILE=data/comve_train.json bash scripts/train_seq2seq.sh

# Test / faithfulness test
MODE=do_test_faith CHECKPOINT=checkpoints/best.ckpt \
DATA_FILE=data/comve_train.json DATA_FILE_2=data/comve_faithful.json \
bash scripts/test_seq2seq.sh
```

Llama-3 (e-SNLI):

```bash
DATA_FILE=data/esnli.json GNN=gat EXP_TYPE=gnn-peft bash scripts/train_llama.sh
```

Variants (`--exp_type`): `gnn` (full graph-guided), `base` (no graph),
`prompt` (highlighted tokens appended as text), `gnn-peft`.

## Faithfulness evaluation

The counterfactual unfaithfulness test ([Atanasova et al., 2023](https://aclanthology.org/2023.acl-long.150/))
inserts a single word into each input and flags an explanation as *unfaithful*
when the inserted word flips the predicted label but never appears in the
generated explanation:

```bash
python eval/faithfulness.py \
  --baseline_dir       generated_text/high_tokens \
  --counterfactual_dir generated_text/high_tokens_for_faithfulness \
  --target_ids RUN_ID_1 RUN_ID_2
```

Reports `counter` (label-flip rate), `counter_unfaith` (unfaithful | flipped)
and `total_unfaith`.

## Citation

```bibtex
@inproceedings{yuan-etal-2025-graph,
    title = "Graph-Guided Textual Explanation Generation Framework",
    author = "Yuan, Shuzhou and Sun, Jingyi and Zhang, Ran and F{\"a}rber, Michael and Eger, Steffen and Atanasova, Pepa and Augenstein, Isabelle",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    year = "2025",
    address = "Suzhou, China",
    pages = "29362--29386",
    doi = "10.18653/v1/2025.emnlp-main.1494",
}
```
