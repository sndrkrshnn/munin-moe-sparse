#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def collect_jsonl(src: Path):
    rows = []
    for p in src.rglob('*.jsonl'):
        for line in p.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = collect_jsonl(in_dir) if in_dir.exists() else []
    if not data:
        # Write template examples
        data = [
            {"text": "How do I restart a systemd service safely?", "expert": "linux"},
            {"text": "Call tool list_processes with no args", "expert": "toolcalling"},
        ]

    split = max(1, int(len(data) * 0.9))
    train = data[:split]
    val = data[split:] or data[:1]

    with (out_dir / 'train.jsonl').open('w', encoding='utf-8') as f:
        for row in train:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    with (out_dir / 'val.jsonl').open('w', encoding='utf-8') as f:
        for row in val:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    print(f'Wrote train={len(train)} val={len(val)} to {out_dir}')


if __name__ == '__main__':
    main()
