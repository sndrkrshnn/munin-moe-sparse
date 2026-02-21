#!/usr/bin/env python3
"""
Minimal training entrypoint scaffold.
Intended to be replaced with full sparse-MoE training loop.
"""

import argparse
import yaml
from pathlib import Path


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--train-config', required=True)
    args = ap.parse_args()

    model_cfg = load_yaml(args.config)
    train_cfg = load_yaml(args.train_config)

    out_dir = Path(train_cfg['outputs']['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Loaded model config:', model_cfg['model']['name'])
    print('Experts:', model_cfg['moe']['expert_names'])
    print('Device:', train_cfg['hardware']['device'])
    print('This is scaffolding. Next step: integrate full MoE trainer + distillation + router supervision.')


if __name__ == '__main__':
    main()
