# Datasets Utilities

This folder contains dataset preparation tools for Munin MoE experts.

## 1) Linux Expert Cleaner

`clean_linux_dataset.py` converts noisy NL→shell CSV/JSONL rows into safe training JSONL.

### Usage

```bash
python datasets/clean_linux_dataset.py \
  --input /path/to/linux_raw.csv \
  --output data/processed/linux_train.jsonl \
  --val-output data/processed/linux_val.jsonl \
  --val-ratio 0.1
```

## 2) Tool-Calling Expert Cleaner

`clean_toolcalling_dataset.py` converts conversation-style tool-call data into training JSONL.

Supported shapes:
- List of message objects with `role` + `content`
- Wrapper keys: `conversation`, `messages`, `chat`
- Direct rows with `text` + `tool_name` + `tool_args`

### Usage

```bash
python datasets/clean_toolcalling_dataset.py \
  --input /path/to/toolcalling_raw.json \
  --output data/processed/toolcalling_train.jsonl \
  --val-output data/processed/toolcalling_val.jsonl \
  --val-ratio 0.1
```

## 3) Merge expert datasets for training

```bash
python training/prepare_data.py \
  --linux-train data/processed/linux_train.jsonl \
  --linux-val data/processed/linux_val.jsonl \
  --tool-train data/processed/toolcalling_train.jsonl \
  --tool-val data/processed/toolcalling_val.jsonl \
  --output data/processed
```

This writes:
- `data/processed/train.jsonl`
- `data/processed/val.jsonl`
