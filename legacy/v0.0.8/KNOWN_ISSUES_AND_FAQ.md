# AndoSim ArteZbuild 0.0.8 — Known Issues & FAQ

This document tracks common problems and practical workarounds for the unified AndoSim (CPU) + PPF (GPU) Blender Extension.

## Known Issues

### 1) PPF stops with: `Vertex count changed on 'X'; stopping`
**Symptom**
- When running PPF in-process, the sim immediately stops and reports:
  - `Vertex count changed on 'Plane'; stopping`

**Cause**
- The PPF scene export may use *evaluated* geometry (modifier stack), while the runtime step writes back to the *base* mesh (`obj.data.vertices`).
- Topology-changing modifiers (common example: unapplied thickness/solidify) can change vertex/triangle counts between export and runtime.

**Workaround**
- Apply topology-changing modifiers (e.g. Solidify/Thickness/Subdivision) before running.
- Or temporarily disable such modifiers on deformables while running PPF.

**Notes**
- Non-topology-changing modifiers (pure transforms) are typically safe.

---

### 2) PPF error: `PPF step failed: failed to advance`
**Symptom**
- Simulation starts but immediately errors (often on the first frame), showing:
  - `PPF step failed: failed to advance`

**Common causes**
- Unstable parameter combination for the current mesh scale/topology (too stiff + too large `dt`).
- Degenerate geometry (zero-area triangles, non-manifold issues) or extreme transforms.
- Bad collision setup (penetrations at start, huge contact offsets).

**How to diagnose**
- Set a persistent **Output Dir** in the UI.
- After a failure, inspect the backend trace files in:
  - `output_dir/data/*.out`
  - Start with `output_dir/data/advance.out` and `output_dir/data/initialize.out`.

**Important note (traces append)**
- The backend appends to the same `output_dir/data/*.out` files across runs.
- For clean diagnostics, either:
  - delete `output_dir/data/*.out` before re-running, or
  - use a fresh output directory per run.

**Mitigations to try**
- Reduce `dt` (e.g. from `1e-3` → `5e-4` or `1e-4`).
- Reduce shell stiffness (Young’s modulus) and/or bending.
- Ensure the deformable has reasonable scale and no initial self-intersections.

**Headless reproduction**
- Use the repo helper to run exactly one step and capture a clean trace bundle:
  - `OUTDIR=/home/moritz/ando/output blender -b --factory-startup --python tools/headless_ppf_from_blend.py -- --blend /path/to/your_scene.blend`
  - This writes into `OUTDIR/headless_run_YYYYMMDD_HHMMSS/` by default.

---

### 3) Ando core doesn’t load (ImportError / module missing)
**Symptom**
- Ando UI shows core missing / can’t import `ando_barrier_core`.

**Cause**
- Native module ABI mismatch: Blender’s bundled Python version must match the compiled extension’s ABI.
  - Example: Blender 4.5 uses Python 3.11 → the module must be `cpython-311-...`.

**Fix**
- Rebuild the extension zip using the repo build script:
  - Run `./build.sh` in `artezbuild_0.0.06/`.
- Verify the zip contains a `cpython-311` binary:
  - `ando/ando_barrier_core.cpython-311-...so`

---

### 4) `ppf_cts_backend` missing even though wheel is bundled
**Symptom**
- UI says `ppf_cts_backend missing: ModuleNotFoundError...`

**Cause**
- Wheels are only activated when Blender installs/enables the add-on as an **Extension**.
- Importing from the source tree (or running scripts directly from the repo) won’t automatically add bundled wheels to `sys.path`.

**Fix**
- Install from the built zip:
  - Blender → Preferences → Extensions → Install from Disk… → select `dist/andosim_artezbuild-0.0.8.zip`.

---

### 5) No way to mark deformables/colliders/pins in UI
**Status**
- Fixed in 0.0.06 workspace: the unified PPF panel now exposes the same “Active Object” controls as the original PPF artezbuild add-on.

**Where**
- View3D → Sidebar → **AndoSim** → **PPF** → **Active Object**

---

### 6) Clean install still shows weird behavior / stale add-on state
**Cause**
- Blender caches extension state and Python bytecode across sessions.

**Workaround**
- Fully wipe Blender user dirs (this resets everything):
  - `~/.config/blender`
  - `~/.local/share/blender`
  - `~/.cache/blender`

---

## FAQ

### How do I select deformables vs static colliders for PPF?
- Select a mesh object.
- In the PPF panel → **Active Object**:
  - Enable **Enable PPF**
  - Set **Role** to:
    - **Deformable** (simulated)
    - **Static Collider** (collision mesh)
    - **Ignore** (excluded)

### How do I create pins?
- For the deformable object:
  - Create a vertex group named `PPF_PIN` (or change the name in UI).
  - Assign desired vertices to the group (weight > 0).
  - In UI: enable **Pins** and set **Pin Vertex Group**.

### Why a Python venv / how to avoid “polluting” Linux Mint?
- A local `.venv` is used for Python tooling during builds.
- If venv creation fails, you likely need the distro package:
  - `sudo apt install python3.11-venv`

### Where do I report/track new issues?
- Add them to this file with:
  - **Symptom** → **Cause** → **Workaround/Fix** → **Status**.
