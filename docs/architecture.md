# Architecture: Munin MoE Sparse (Pi-first)

## Core Design
- Decoder-only transformer
- Shared dense trunk handles general language
- Sparse MoE FFN blocks in selected layers
- 2 experts initially:
  1. Linux expert
  2. Tool-calling expert
- Router supports `top-1` by default, `top-2` fallback for uncertain tokens

## Dynamic Compute Budget
Three runtime modes:
- **Eco**: trunk + top-1 expert only when needed
- **Balanced**: trunk + top-1, occasional top-2 fallback
- **Burst**: trunk + top-2 for hard spans

## Pi 5 Constraints
- CPU-only inference
- Quantized experts (int4/int8)
- Cache-friendly KV and expert weight layout
- Thread affinity + thermal-aware scheduling

## Suggested v1 Parameter Budget
- Trunk: 80–140M
- Router: <2M
- Experts: 2 x (20–40M each)
- Total parameters: 120–220M
- Active per token (target): 1–2M equivalent compute path

## Quality Strategy
- Strong dense trunk pretraining/fine-tuning
- Expert-supervised routing labels
- Distillation from teacher on Linux/tool tasks
- Tool-call schema validation and repair loop
