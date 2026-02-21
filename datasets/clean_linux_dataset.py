#!/usr/bin/env python3
import argparse
import csv
import json
import random
import re
import shlex
import subprocess
from pathlib import Path

DANGEROUS_PATTERNS = [
    r"\brm\b",
    r"\bdd\b",
    r"\bmkfs\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bpoweroff\b",
    r"\bhalt\b",
    r":\(\)\s*\{:\|:\s*&\};:",
    r"curl\s+[^|]+\|\s*sh",
    r"wget\s+[^|]+\|\s*sh",
    r"\bchown\b\s+-R\s+/",
    r"\bchmod\b\s+777\s+/",
]

SUSPICIOUS_PATTERNS = [
    r"\$\(ls\b.*sort\s+-R",   # random file operations
    r"/dev/urandom",
    r"\$RANDOM",
]


def load_rows(path: Path):
    ext = path.suffix.lower()
    rows = []

    if ext in {".jsonl", ".json"}:
        with path.open("r", encoding="utf-8") as f:
            if ext == ".jsonl":
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            else:
                data = json.load(f)
                if isinstance(data, list):
                    rows.extend(data)
                else:
                    raise ValueError("JSON input must be a list of objects")
        return rows

    if ext in {".csv", ".tsv"}:
        delimiter = "\t" if ext == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            rows.extend(reader)
        return rows

    raise ValueError(f"Unsupported input type: {ext}")


def is_dangerous(cmd: str) -> bool:
    c = cmd.strip().lower()
    for p in DANGEROUS_PATTERNS + SUSPICIOUS_PATTERNS:
        if re.search(p, c):
            return True
    return False


def shell_syntax_ok(cmd: str) -> bool:
    try:
        subprocess.run(["bash", "-n", "-c", cmd], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


def normalize_cmd(cmd: str) -> str:
    cmd = " ".join(cmd.strip().split())
    # normalize dangerous wildcard literal pattern from noisy datasets
    cmd = cmd.replace("'*string*'", "'string'")
    return cmd


def extract_prompt(row: dict) -> str:
    for key in ("invocation", "augmented_text", "prompt", "instruction", "text"):
        v = row.get(key)
        if v and str(v).strip():
            return str(v).strip()
    return ""


def extract_cmd(row: dict) -> str:
    for key in ("cmd", "command", "response", "output"):
        v = row.get(key)
        if v and str(v).strip():
            return str(v).strip()
    return ""


def safe_row(row: dict, source: str):
    prompt = extract_prompt(row)
    cmd = normalize_cmd(extract_cmd(row))

    if not prompt or not cmd:
        return None, "missing"
    if is_dangerous(cmd):
        return None, "dangerous"
    if not shell_syntax_ok(cmd):
        return None, "syntax"

    # extra parse check
    try:
        shlex.split(cmd)
    except Exception:
        return None, "tokenize"

    return {
        "text": prompt,
        "response": cmd,
        "expert": "linux",
        "safety": "safe",
        "source": source,
    }, "ok"


def dedupe(rows):
    seen = set()
    out = []
    for r in rows:
        k = (r["text"].strip().lower(), r["response"].strip())
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

    in_path = Path(args.input)
    rows = load_rows(in_path)

    kept = []
    stats = {"ok": 0, "missing": 0, "dangerous": 0, "syntax": 0, "tokenize": 0}

    for row in rows:
        cleaned, reason = safe_row(row, in_path.name)
        stats[reason] = stats.get(reason, 0) + 1
        if cleaned:
            kept.append(cleaned)

    kept = dedupe(kept)
    random.Random(args.seed).shuffle(kept)

    n_val = max(1, int(len(kept) * args.val_ratio)) if kept else 0
    val = kept[:n_val]
    train = kept[n_val:]

    write_jsonl(train, Path(args.output))
    write_jsonl(val, Path(args.val_output))

    print("=== Cleaning Summary ===")
    print(f"input_rows: {len(rows)}")
    print(f"kept_rows: {len(kept)}")
    for k, v in stats.items():
        print(f"filtered_{k}: {v}")
    print(f"train_rows: {len(train)}")
    print(f"val_rows: {len(val)}")
    print(f"train_out: {args.output}")
    print(f"val_out: {args.val_output}")


if __name__ == "__main__":
    main()
