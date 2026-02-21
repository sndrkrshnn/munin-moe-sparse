#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_jsonl(path):
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--linux-eval', default='data/raw/linux/eval.jsonl')
    ap.add_argument('--tool-eval', default='data/raw/toolcalling/eval.jsonl')
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    print(f'Checkpoint: {ckpt}')
    linux = load_jsonl(args.linux_eval)
    tools = load_jsonl(args.tool_eval)

    print(f'Linux eval samples: {len(linux)}')
    print(f'Tool-calling eval samples: {len(tools)}')
    print('TODO next: wire model loading + exact-match + tool-schema validity scoring + Pi latency benchmark.')


if __name__ == '__main__':
    main()
