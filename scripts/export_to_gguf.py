#!/usr/bin/env python3
"""
GGUF export wrapper (stub).

This script intentionally wraps llama.cpp conversion utilities rather than reimplementing GGUF packing.
"""

import argparse
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to HF-style model dir/checkpoint")
    ap.add_argument("--llama-cpp-dir", required=True, help="Path to local llama.cpp checkout")
    ap.add_argument("--out", required=True, help="Output .gguf path")
    ap.add_argument("--dtype", default="f16", choices=["f16", "f32"])
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    out = Path(args.out)
    convert = Path(args.llama_cpp_dir) / "convert_hf_to_gguf.py"

    if not convert.exists():
        raise FileNotFoundError(f"convert_hf_to_gguf.py not found at {convert}")

    cmd = [
        "python3",
        str(convert),
        str(ckpt),
        "--outfile",
        str(out),
        "--outtype",
        args.dtype,
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Wrote GGUF: {out}")


if __name__ == "__main__":
    main()
