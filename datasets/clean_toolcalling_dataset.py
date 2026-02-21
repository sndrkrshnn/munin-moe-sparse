#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path


def maybe_json(v):
    if isinstance(v, str):
        s = v.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except Exception:
                return v
    return v


def load_any(path: Path):
    ext = path.suffix.lower()
    if ext == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    if ext == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        raise ValueError("JSON input must be a list")

    if ext == ".parquet":
        try:
            import pyarrow.parquet as pq
        except Exception as e:
            raise RuntimeError("Parquet input requires pyarrow. Install with: pip install pyarrow") from e

        table = pq.read_table(path)
        rows = table.to_pylist()
        # normalize nested JSON strings from parquet columns
        for r in rows:
            if isinstance(r, dict):
                for k, v in list(r.items()):
                    r[k] = maybe_json(v)
        return rows

    raise ValueError("Supported input types: .json, .jsonl, .parquet")


def parse_conversation(conv):
    if not isinstance(conv, list):
        return None

    user_msg = None
    tool_call = None
    assistant_msg = None

    for turn in conv:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role", "")).strip().lower()
        content = maybe_json(turn.get("content"))

        if role == "user" and isinstance(content, str) and not user_msg:
            user_msg = content.strip()

        elif role in {"tool call", "tool_call", "toolcall"} and isinstance(content, dict) and not tool_call:
            name = content.get("name")
            arguments = maybe_json(content.get("arguments", {}))
            if isinstance(name, str) and name.strip() and isinstance(arguments, dict):
                tool_call = {"name": name.strip(), "arguments": arguments}

        elif role == "assistant" and isinstance(content, str) and not assistant_msg:
            assistant_msg = content.strip()

    if not user_msg or not tool_call:
        return None

    return {
        "text": user_msg,
        "response": json.dumps(tool_call, ensure_ascii=False, sort_keys=True),
        "assistant_text": assistant_msg or "",
        "expert": "toolcalling",
        "safety": "safe",
    }


def parse_row(row):
    if isinstance(row, list):
        return parse_conversation(row)

    if not isinstance(row, dict):
        return None

    # common wrappers
    for key in ("conversation", "messages", "chat"):
        if key in row:
            wrapped = maybe_json(row[key])
            out = parse_conversation(wrapped)
            if out:
                return out

    # fallback direct schema
    text = row.get("text") or row.get("prompt")
    tool_name = row.get("tool_name")
    tool_args = maybe_json(row.get("tool_args"))
    if text and tool_name:
        if not isinstance(tool_args, dict):
            tool_args = {}
        return {
            "text": str(text).strip(),
            "response": json.dumps({"name": str(tool_name).strip(), "arguments": tool_args}, ensure_ascii=False, sort_keys=True),
            "assistant_text": str(row.get("assistant_text", "")).strip(),
            "expert": "toolcalling",
            "safety": "safe",
        }

    return None


def dedupe(rows):
    seen = set()
    out = []
    for r in rows:
        k = (r["text"].strip().lower(), r["response"])
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def write_jsonl(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--val-output", required=True)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = load_any(Path(args.input))
    parsed = []
    bad = 0
    for row in rows:
        p = parse_row(row)
        if p:
            parsed.append(p)
        else:
            bad += 1

    parsed = dedupe(parsed)
    random.Random(args.seed).shuffle(parsed)
    n_val = max(1, int(len(parsed) * args.val_ratio)) if parsed else 0
    val = parsed[:n_val]
    train = parsed[n_val:]

    write_jsonl(train, Path(args.output))
    write_jsonl(val, Path(args.val_output))

    print("=== Toolcalling Cleaning Summary ===")
    print(f"input_rows: {len(rows)}")
    print(f"parsed_rows: {len(parsed)}")
    print(f"discarded_rows: {bad}")
    print(f"train_rows: {len(train)}")
    print(f"val_rows: {len(val)}")
    print(f"train_out: {args.output}")
    print(f"val_out: {args.val_output}")


if __name__ == "__main__":
    main()
