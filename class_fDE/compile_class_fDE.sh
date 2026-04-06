#!/bin/bash
# compile_class_fDE.sh — Compile class_fDE and install the Python wrapper (classy_fDE)
#
# Usage (from the class_fDE/ directory):
#   chmod +x compile_class_fDE.sh
#   ./compile_class_fDE.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Compiling class_fDE ==="
make clean
make

echo ""
echo "=== Installing classy_fDE Python wrapper ==="
pip install .

echo ""
echo "=== Done ==="
echo "The Python module 'classy_fDE' is now available."
echo "You can verify with: python -c 'from classy_fDE import Class; print(\"OK\")'"
