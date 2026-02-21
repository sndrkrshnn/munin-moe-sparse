#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path


def read_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def ensure_shape(row):
    # Normalize to training schema
    text = row.get("text") or row.get("prompt") or ""
    response = row.get("response", "")
    expert = row.get("expert", "general")
    if not text:
        return None

    # Train script currently uses text for language modeling; keep response joined for conditioning
    full_text = text if not response else f"User: {text}\nAssistant: {response}"
    return {"text": full_text, "expert": expert}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linux-train", default="data/processed/linux_train.jsonl")
    ap.add_argument("--linux-val", default="data/processed/linux_val.jsonl")
    ap.add_argument("--tool-train", default="data/processed/toolcalling_train.jsonl")
    ap.add_argument("--tool-val", default="data/processed/toolcalling_val.jsonl")
    ap.add_argument("--output", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.output)

    linux_train = [ensure_shape(r) for r in read_jsonl(Path(args.linux_train))]
    linux_val = [ensure_shape(r) for r in read_jsonl(Path(args.linux_val))]
    tool_train = [ensure_shape(r) for r in read_jsonl(Path(args.tool_train))]
    tool_val = [ensure_shape(r) for r in read_jsonl(Path(args.tool_val))]

    train = [r for r in (linux_train + tool_train) if r]
    val = [r for r in (linux_val + tool_val) if r]

    random.Random(args.seed).shuffle(train)
    random.Random(args.seed + 1).shuffle(val)

    if not train:
        train = [
            {"text": "User: How do I restart nginx?\nAssistant: sudo systemctl restart nginx", "expert": "linux"},
            {"text": "User: Find weather in Chennai\nAssistant: {\"name\":\"weather.get\",\"arguments\":{\"city\":\"Chennai\"}}", "expert": "toolcalling"},
        ]
    if not val:
        val = train[:1]

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    write_jsonl(train, train_path)
    write_jsonl(val, val_path)

    print("=== Combined Dataset Summary ===")
    print(f"linux_train={len(linux_train)} linux_val={len(linux_val)}")
    print(f"tool_train={len(tool_train)} tool_val={len(tool_val)}")
    print(f"final_train={len(train)} final_val={len(val)}")
    print(f"wrote: {train_path}")
    print(f"wrote: {val_path}")


if __name__ == "__main__":
    main()
