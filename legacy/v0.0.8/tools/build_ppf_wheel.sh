#!/usr/bin/env bash
set -euo pipefail

# Builds the ppf_cts_backend wheel for Blender's Python (3.11) and copies it into blender_addon/wheels/.
#
# Requirements:
# - rust toolchain (cargo)
# - Python 3.11 + pip
# - maturin installed (pip install maturin)
# - A suitable manylinux toolchain / environment if you need a portable wheel.
#
# Usage:
#   tools/build_ppf_wheel.sh

# Optional:
#   PYTHON_BIN=python3.11 tools/build_ppf_wheel.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PPF_DIR="$ROOT_DIR/ppf_cts_backend"
OUT_DIR="$ROOT_DIR/blender_addon/wheels"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"

mkdir -p "$OUT_DIR"

if ! command -v maturin >/dev/null 2>&1; then
  echo "ERROR: maturin not found. Install with: python3.11 -m pip install maturin" >&2
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: $PYTHON_BIN not found. Install Python 3.11 or set PYTHON_BIN to Blender's python." >&2
  echo "Example: PYTHON_BIN=/path/to/blender/4.xx/python/bin/python3.11 tools/build_ppf_wheel.sh" >&2
  exit 1
fi

pushd "$PPF_DIR" >/dev/null

# Build a release wheel targeting Blender's ABI (cp311).
# Note: This produces a local wheel; for maximum compatibility use a manylinux build container.
maturin build --release --strip -i "$PYTHON_BIN"

# Copy newest cp311 wheel into the addon wheels/ folder.
LATEST_WHL="$(ls -1t target/wheels/*cp311*.whl | head -n 1)"
cp -f "$LATEST_WHL" "$OUT_DIR/"

echo "Copied wheel to: $OUT_DIR/$(basename "$LATEST_WHL")"

popd >/dev/null
