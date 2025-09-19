# kzb

Kol-Zchut bot utilities for preparing multilingual embeddings and answering questions with retrieval augmented generation.

## Setup

Install the project in editable mode together with the local inference dependencies:

```bash
pip install -e .
```

The pipeline runs fully offline by loading the open-source [`openaccess-ai-collective/gpt-oss-20b`](https://huggingface.co/openaccess-ai-collective/gpt-oss-20b)
model. The first invocation downloads the weights automatically, or you can pre-fetch them with:

```bash
python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "openaccess-ai-collective/gpt-oss-20b"
AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
PY
```

No API keys are required.

## Commands

Two commands are available through the CLI:

### Prepare the dataset

```bash
python cli.py prepare --input data/all_pages.json --processed data/processed_pages.jsonl \
  --embeddings-output data/embeddings_index.npz
```

This command cleans each page, translates it to English with `gpt-oss-20b`, and stores Hebrew/English embeddings in the output file.
When `--embeddings-output` is supplied the normalized embedding matrices are serialized as `data/embeddings_index.npz`, so
future queries can reuse them without rebuilding the vector store from scratch.

### Query the bot

```bash
python cli.py query --processed data/processed_pages.jsonl --question "מה הזכויות של הורה עצמאי?" \
  --embeddings data/embeddings_index.npz
```

The query command translates the Hebrew question to English, retrieves the top five relevant pages from both Hebrew and English embeddings in parallel, and generates a final Hebrew answer with `gpt-oss-20b`. When `--embeddings` is provided the
precomputed index is loaded for faster start-up; otherwise the vector store is constructed in memory on demand.
