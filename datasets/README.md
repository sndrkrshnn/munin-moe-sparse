# Datasets Utilities

This folder contains dataset preparation tools for Munin MoE experts.

## Linux Expert Cleaner

`clean_linux_dataset.py` converts noisy NL→shell CSV/JSONL rows into safe training JSONL.

### Input columns supported
- `invocation` (preferred NL prompt)
- `augmented_text` (fallback NL prompt)
- `cmd` (shell command)

### Usage

```bash
python datasets/clean_linux_dataset.py \
  --input /path/to/raw.csv \
  --output data/processed/linux_clean.jsonl \
  --val-output data/processed/linux_clean_val.jsonl \
  --val-ratio 0.1
```

### Notes
- Dangerous commands are filtered out
- Broken shell syntax is filtered out
- Commands are normalized and deduplicated
- Output schema:
  - `text`: instruction-style prompt
  - `response`: command answer
  - `expert`: `linux`
  - `source`: source filename
  - `safety`: `safe`

