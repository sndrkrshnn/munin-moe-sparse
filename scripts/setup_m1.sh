#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
pip install -r requirements.txt

echo "M1 environment ready. Activate with: source .venv/bin/activate"
