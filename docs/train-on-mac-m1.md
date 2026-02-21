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

## 3) Prepare datasets
Put your domain datasets here:
- `data/raw/linux/`
- `data/raw/toolcalling/`

Then process:
```bash
python training/prepare_data.py --input data/raw --output data/processed
```

## 4) Start training (2 experts)
```bash
python training/train_moe.py \
  --config configs/model_moe_2experts.yaml \
  --train-config configs/train_m1_lora.yaml
```

## 5) Evaluate Linux + Tool-calling quality
```bash
python eval/run_eval.py --checkpoint artifacts/latest
```

## 6) Export for Pi inference
(placeholder for GGUF/quantization pipeline)
```bash
# Example flow (to wire later):
# python scripts/export_to_gguf.py --ckpt artifacts/latest --out artifacts/gguf/model.gguf
# python scripts/quantize.py --in artifacts/gguf/model.gguf --out artifacts/gguf/model-q4.gguf --mode q4_k_m
```

## 7) Deploy to Raspberry Pi 5
- Copy quantized model to Pi
- Run Pi CPU benchmark + latency checks

## M1-specific tips
- Use smaller micro-batch sizes (`1-2`) and gradient accumulation (`16-64`)
- Enable mixed precision where stable
- Keep context length modest during early experiments (512/1024)
- Train adapters first (LoRA/QLoRA), then full/sparse tuning if needed
