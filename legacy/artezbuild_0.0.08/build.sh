#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_ADDON_DIR="$ROOT/blender_addon/ando"
EXTENSION_DIR="$ROOT/blender_addon"
DIST_DIR="$ROOT/dist"

BLENDER_BIN="${BLENDER_BIN:-}"
if [[ -z "$BLENDER_BIN" ]]; then
  BLENDER_BIN="$(command -v blender 2>/dev/null || true)"
fi

BLENDER_PY_MAJMIN=""
if [[ -n "$BLENDER_BIN" ]]; then
  echo "Detected Blender: $BLENDER_BIN"
  BLENDER_PY_MAJMIN="$($BLENDER_BIN -b --factory-startup --python-expr "import sys; print('PYMAJMIN=' + str(sys.version_info[0]) + '.' + str(sys.version_info[1]))" --python-exit-code 1 2>/dev/null | sed -n 's/^PYMAJMIN=//p' | tail -n 1 || true)"
fi

# Prefer a matching system Python (with dev headers installed) over Blender's bundled Python.
# Blender's sys.executable often lacks headers/libs required by CMake's Python3 Development.
PY_BASE=""
if [[ -n "$BLENDER_PY_MAJMIN" ]]; then
  if command -v "python$BLENDER_PY_MAJMIN" >/dev/null 2>&1; then
    PY_BASE="$(command -v "python$BLENDER_PY_MAJMIN")"
  fi
fi
if [[ -z "$PY_BASE" ]]; then
  PY_BASE="$(command -v python3)"
fi

VENV_DIR="$ROOT/.venv"
PY_EXEC=""
_venv_matches_target() {
  local vpy="$1"
  local target="$2"
  if [[ -z "$target" ]]; then
    return 0
  fi
  local got
  got="$($vpy -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>/dev/null || true)"
  [[ "$got" == "$target" ]]
}

if [[ -x "$VENV_DIR/bin/python" && -n "$BLENDER_PY_MAJMIN" ]]; then
  if ! _venv_matches_target "$VENV_DIR/bin/python" "$BLENDER_PY_MAJMIN"; then
    echo "WARN: Existing venv Python does not match Blender Python $BLENDER_PY_MAJMIN; recreating venv"
    rm -rf "$VENV_DIR"
  fi
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Creating venv at: $VENV_DIR"
  if "$PY_BASE" -m venv "$VENV_DIR" 2>/dev/null; then
    PY_EXEC="$VENV_DIR/bin/python"
  else
    echo "WARN: Could not create venv (missing ensurepip/python3-venv). Falling back to system Python: $PY_BASE"
    PY_EXEC="$PY_BASE"
  fi
else
  PY_EXEC="$VENV_DIR/bin/python"
fi

echo "Using Python: $PY_EXEC"
"$PY_EXEC" -c "import sys; print('PY', sys.version); print('EXE', sys.executable)"
if [[ -n "$BLENDER_PY_MAJMIN" ]]; then
  PY_MAJMIN="$($PY_EXEC -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')")"
  if [[ "$PY_MAJMIN" != "$BLENDER_PY_MAJMIN" ]]; then
    echo "WARN: Build Python $PY_MAJMIN does not match Blender Python $BLENDER_PY_MAJMIN. The compiled core may not load."
  fi
fi

# Resolve Python development include/lib paths explicitly for CMake.
PY_INCLUDE_DIR="$($PY_EXEC - <<'PY'
import sysconfig
print(sysconfig.get_paths().get('include',''))
PY
)"
PY_LIBDIR="$($PY_EXEC - <<'PY'
import sysconfig
print(sysconfig.get_config_var('LIBDIR') or '')
PY
)"
PY_LDLIB="$($PY_EXEC - <<'PY'
import sysconfig
print(sysconfig.get_config_var('LDLIBRARY') or '')
PY
)"
PY_LIBRARY=""
if [[ -n "$PY_LIBDIR" && -n "$PY_LDLIB" && -f "$PY_LIBDIR/$PY_LDLIB" ]]; then
  PY_LIBRARY="$PY_LIBDIR/$PY_LDLIB"
fi
if [[ -n "$PY_INCLUDE_DIR" ]]; then
  echo "Python include: $PY_INCLUDE_DIR"
fi
if [[ -n "$PY_LIBRARY" ]]; then
  echo "Python library: $PY_LIBRARY"
fi

echo "== AndoSim ArteZbuild build =="
echo "Root: $ROOT"

# 1) Build native Ando core (ando_barrier_core) into the extension bundle.
# The C++ sources are vendored into this repo under ando_core_src (self-contained build).

ANDOSIM_SRC_ROOT="$ROOT/ando_core_src"
if [[ ! -f "$ANDOSIM_SRC_ROOT/CMakeLists.txt" ]]; then
  echo "ERROR: Missing vendored Ando core sources at: $ANDOSIM_SRC_ROOT"
  echo "Expected: $ANDOSIM_SRC_ROOT/CMakeLists.txt"
  echo "Fix: vendor AndoSim sources into ando_core_src (src/, extern/, tests/, demos/, etc.)"
  exit 1
fi

echo "[2/3] Building ando_barrier_core via CMake..."
BUILD_DIR="$ANDOSIM_SRC_ROOT/build"
mkdir -p "$BUILD_DIR"

# If the build dir was configured from a different location (e.g. after copying/renaming
# the repo folder), CMake will refuse to continue. Detect and wipe the build dir.
if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  cache_build_dir=""
  cache_src_dir=""
  cache_build_dir="$(grep -E '^CMAKE_CACHEFILE_DIR:INTERNAL=' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null | head -n 1 | sed 's/^CMAKE_CACHEFILE_DIR:INTERNAL=//')"
  cache_src_dir="$(grep -E '^CMAKE_HOME_DIRECTORY:INTERNAL=' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null | head -n 1 | sed 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//')"
  if [[ -n "$cache_build_dir" && "$cache_build_dir" != "$BUILD_DIR" ]]; then
    echo "Wiping build dir (moved build dir): $cache_build_dir -> $BUILD_DIR"
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
  elif [[ -n "$cache_src_dir" && "$cache_src_dir" != "$ANDOSIM_SRC_ROOT" ]]; then
    echo "Wiping build dir (moved source dir): $cache_src_dir -> $ANDOSIM_SRC_ROOT"
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
  fi
fi

EXPECTED_SOABI="$($PY_EXEC - <<'PY'
import sysconfig
print(sysconfig.get_config_var('SOABI') or '')
PY
)"
if [[ -n "$EXPECTED_SOABI" ]]; then
  for existing in "$BUILD_DIR"/ando_barrier_core*.so "$BUILD_DIR"/**/ando_barrier_core*.so; do
    [[ -e "$existing" ]] || continue
    if [[ "$existing" != *"$EXPECTED_SOABI"* ]]; then
      echo "Wiping build dir (SOABI changed): found $(basename "$existing") expected $EXPECTED_SOABI"
      rm -rf "$BUILD_DIR"/*
      break
    fi
  done
fi

# If the build dir was configured with a different Python, force reconfigure.
if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  cached_py=""
  if grep -qE '^Python3_EXECUTABLE:FILEPATH=' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
    cached_py="$(grep -E '^Python3_EXECUTABLE:FILEPATH=' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null | head -n 1 | sed 's/^Python3_EXECUTABLE:FILEPATH=//')"
  fi
  if [[ -n "$cached_py" && "$cached_py" != "$PY_EXEC" ]]; then
    echo "Reconfiguring (Python changed): $cached_py -> $PY_EXEC"
    rm -f "$BUILD_DIR/CMakeCache.txt"
  fi
fi

cmake_args=(
  "-DCMAKE_BUILD_TYPE=Release"
  "-DPython3_EXECUTABLE=$PY_EXEC"
)
if [[ -n "$PY_INCLUDE_DIR" ]]; then
  cmake_args+=("-DPython3_INCLUDE_DIR=$PY_INCLUDE_DIR")
fi
if [[ -n "$PY_LIBRARY" ]]; then
  cmake_args+=("-DPython3_LIBRARY=$PY_LIBRARY")
fi

cmake -S "$ANDOSIM_SRC_ROOT" -B "$BUILD_DIR" "${cmake_args[@]}"
cmake --build "$BUILD_DIR" -j "$(nproc)"

# 3) Copy the resulting binary into the bundled addon folder.
# We copy the first matching shared library produced.
echo "[3/3] Installing core binary into addon bundle..."
shopt -s nullglob globstar
candidates=(
  "$BUILD_DIR"/**/ando_barrier_core*.so
  "$BUILD_DIR"/**/ando_barrier_core*.pyd
  "$BUILD_DIR"/**/ando_barrier_core*.dll
  "$BUILD_DIR"/**/ando_barrier_core*.dylib
)
shopt -u nullglob globstar

if (( ${#candidates[@]} == 0 )); then
  echo "ERROR: No compiled ando_barrier_core binary found in $BUILD_DIR"
  echo "Hint: check the CMake output above for build failures."
  exit 1
fi

mkdir -p "$OUT_ADDON_DIR"
rm -f "$OUT_ADDON_DIR"/ando_barrier_core*.so "$OUT_ADDON_DIR"/ando_barrier_core*.pyd "$OUT_ADDON_DIR"/ando_barrier_core*.dll "$OUT_ADDON_DIR"/ando_barrier_core*.dylib
cp -av "${candidates[0]}" "$OUT_ADDON_DIR/"

echo "Installed: $OUT_ADDON_DIR/$(basename "${candidates[0]}")"

# 4) Package an installable extension zip (Blender 4.2+).
# The .zip must have blender_manifest.toml at the archive root.
echo "[4/4] Packaging extension zip..."
mkdir -p "$DIST_DIR"

MANIFEST="$EXTENSION_DIR/blender_manifest.toml"
if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: Missing extension manifest: $MANIFEST"
  exit 1
fi

EXT_ID="$(awk -F'=' '/^id[[:space:]]*=/{gsub(/[" \t]/, "", $2); print $2; exit}' "$MANIFEST")"
EXT_VER="$(awk -F'=' '/^version[[:space:]]*=/{gsub(/[" \t]/, "", $2); print $2; exit}' "$MANIFEST")"
if [[ -z "$EXT_ID" || -z "$EXT_VER" ]]; then
  echo "ERROR: Could not parse id/version from $MANIFEST"
  exit 1
fi

ZIP_PATH="$DIST_DIR/${EXT_ID}-${EXT_VER}.zip"

EXTENSION_DIR="$EXTENSION_DIR" ZIP_PATH="$ZIP_PATH" python3 - <<'PY'
import os
import sys
import zipfile

root = os.environ["EXTENSION_DIR"]
zip_path = os.environ["ZIP_PATH"]

manifest = os.path.join(root, "blender_manifest.toml")
if not os.path.isfile(manifest):
  raise SystemExit(f"Missing blender_manifest.toml at: {manifest}")

# Create a zip with extension files at archive root (no leading folder).
os.makedirs(os.path.dirname(zip_path), exist_ok=True)
if os.path.exists(zip_path):
  os.remove(zip_path)

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
  for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = [d for d in dirnames if d != "__pycache__"]
    for name in filenames:
      if name.endswith(".pyc"):
        continue
      full = os.path.join(dirpath, name)
      rel = os.path.relpath(full, root)
      zf.write(full, rel)

print(zip_path)
PY

echo "Wrote: $ZIP_PATH"
echo "Done."
