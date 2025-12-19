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

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PPF_DIR="$ROOT_DIR/ppf_cts_backend"
OUT_DIR="$ROOT_DIR/blender_addon/wheels"

mkdir -p "$OUT_DIR"

if ! command -v maturin >/dev/null 2>&1; then
  echo "ERROR: maturin not found. Install with: python3.11 -m pip install maturin" >&2
  exit 1
fi

pushd "$PPF_DIR" >/dev/null

# Build a release wheel targeting the current Python.
# Note: This produces a local wheel; for maximum compatibility use a manylinux build container.
maturin build --release --strip

# Copy newest wheel into the addon wheels/ folder.
LATEST_WHL="$(ls -1t target/wheels/*.whl | head -n 1)"
cp -f "$LATEST_WHL" "$OUT_DIR/"

echo "Copied wheel to: $OUT_DIR/$(basename "$LATEST_WHL")"

popd >/dev/null
