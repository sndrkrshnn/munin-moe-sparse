# munin-moe-sparse

Pi-first sparse MoE LLM project optimized for Raspberry Pi 5 CPU inference.

## Goal
Build a high-quality, high-performance sparse model with:
- Shared dense trunk
- 2 domain experts initially:
  - Linux Expert
  - Tool-Calling Expert
- Optional expansion to max 5 experts
- Dynamic compute budget (Eco/Balanced/Burst)

## Repository Layout
- `docs/` architecture, training, inference plans
- `configs/` model + training configs
- `training/` data prep and training entrypoints
- `scripts/` utility scripts, including Mac M1 setup
- `src/munin_moe/` model skeleton and routing modules
- `eval/` evaluation checklist and benchmark templates

## Quickstart (MacBook Air M1)
```bash
# 1) Create env
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# 2) Prepare data
python training/prepare_data.py --input data/raw --output data/processed

# 3) Train sparse model (2 experts)
python training/train_moe.py \
  --config configs/model_moe_2experts.yaml \
  --train-config configs/train_m1_lora.yaml

# 4) Evaluate
python eval/run_eval.py --checkpoint artifacts/latest
```

## Notes
- Training is done on Mac M1 (or stronger machine).
- Inference target is Raspberry Pi 5 CPU only.
- Quantization and runtime profiling are mandatory before deployment.
