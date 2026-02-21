# Training on MacBook Air M1 (Step-by-step)

## 1) Prerequisites
- macOS with Python 3.10+
- Xcode command line tools
- At least 16GB RAM recommended (8GB possible with smaller batch + gradient accumulation)

```bash
xcode-select --install
python3 --version
```

## 2) Clone + Setup
```bash
git clone <YOUR_REPO_URL>
cd munin-moe-sparse
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
pip install -r requirements.txt
```

## 3) Clean Linux dataset
```bash
python datasets/clean_linux_dataset.py \
  --input /path/to/linux_raw.csv \
  --output data/processed/linux_train.jsonl \
  --val-output data/processed/linux_val.jsonl \
  --val-ratio 0.1
```

## 4) Clean tool-calling dataset
```bash
python datasets/clean_toolcalling_dataset.py \
  --input /path/to/toolcalling_raw.json \
  --output data/processed/toolcalling_train.jsonl \
  --val-output data/processed/toolcalling_val.jsonl \
  --val-ratio 0.1
```

## 5) Merge expert datasets for training
```bash
python training/prepare_data.py \
  --linux-train data/processed/linux_train.jsonl \
  --linux-val data/processed/linux_val.jsonl \
  --tool-train data/processed/toolcalling_train.jsonl \
  --tool-val data/processed/toolcalling_val.jsonl \
  --output data/processed
```

## 6) Start training (verbose logs enabled)
```bash
python training/train_moe.py \
  --config configs/model_moe_2experts.yaml \
  --train-config configs/train_m1_lora.yaml
```

Logs now include:
- batch/step progress
- total/lm/router/load-balance losses
- per-step token count + step time
- expert utilization ratios
- epoch summary with train/val loss

## 7) Evaluate Linux + Tool-calling quality
```bash
python eval/run_eval.py --checkpoint artifacts/latest
```

## 8) Export for Pi inference (GGUF)
```bash
python scripts/export_to_gguf.py \
  --checkpoint artifacts/latest \
  --llama-cpp-dir ~/llama.cpp \
  --out artifacts/gguf/munin-moe-f16.gguf \
  --dtype f16
```

## 9) Deploy to Raspberry Pi 5
- Copy quantized model to Pi
- Run Pi CPU benchmark + latency checks

## M1-specific tips
- Use smaller micro-batch sizes (`1-2`) and gradient accumulation (`16-64`)
- Keep context length modest during early experiments (512/1024)
- Start with 1 epoch dry-run to validate data and logs before long training
