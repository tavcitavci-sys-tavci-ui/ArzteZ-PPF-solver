# AndoArteZcontactsolver (Blender Extension)

AndoArteZcontactsolver is a Blender 4.2+ Extension that combines AndoSim’s CPU cloth/shell solver with the PPF GPU contact solver in one workflow.

A unified Blender 4.2+ **Extension** that bundles two simulation backends under one UI:

- **Ando (CPU)**: AndoSim’s cubic barrier cloth/shell solver via a bundled native module (`ando_barrier_core`).
- **PPF (GPU)**: an in-process Python wheel (`ppf_cts_backend`) wrapping the ZOZO/st-tech PPF contact solver.

This repo produces an installable zip in `dist/`.

## Downloads

- **Latest (0.0.8)**: `dist/andosim_artezbuild-latest.zip`
- Versioned zips are also kept in `dist/`.

## Supported Versions

- **Blender**: 4.2+ (tested with Blender 4.5.x LTS)
- **OS**: Linux x64

## Runtime Requirements

### Minimal (Ando CPU backend)
- Blender 4.2+ on Linux x64


### PPF GPU backend (additional requirements)
- NVIDIA GPU (PPF runs on GPU)
- Working NVIDIA driver stack compatible with your CUDA runtime

Notes:
- The extension bundles `ppf_cts_backend-…-cp311-…-manylinux_2_34_x86_64.whl`. If your system is significantly older than manylinux_2_34 / glibc expectations, you may need to rebuild the wheel.

## Install (Recommended)

1. Build (or download) the extension zip:
   - Latest: `dist/andosim_artezbuild-latest.zip` (currently 0.0.8)
   - Versioned: `dist/andosim_artezbuild-0.0.8.zip`
2. In Blender:
   - **Edit → Preferences → Extensions → Install from Disk…**
   - Select `dist/andosim_artezbuild-latest.zip` (or a versioned zip)
   - Enable the extension.
3. Verify it loaded:
   - Open the 3D Viewport sidebar: **View3D → Sidebar → AndoSim**
   - In the PPF panel you should see something like: `ppf_cts_backend OK: 0.0.5`

If you updated the zip and Blender still behaves like the old version, see “Clean Reinstall / Cache” below.

## Quick Start (PPF)

1. Pick a target mesh and mark it as deformable:
   - Select the mesh
   - **AndoSim → PPF → Active Object**
   - Enable PPF for the object and set **Role = Deformable**
2. Mark colliders:
   - Select collider meshes and set **Role = Static Collider**
3. Set an Output Directory:
   - **AndoSim → PPF → Scene → Output Dir**
   - Use a persistent folder if you want traces/logs after failures.
4. Run:
   - Press **Run**

### Pins (PPF)

There are two pin workflows:

- **Pin Handles (Empties)**: create persistent Empty objects that drive pinned vertices.
  - Create pins from Edit Mode via the addon operator (creates an Empty and assigns the vertex to a pin group)
  - Move the Empty in Object Mode to animate pin targets
- **Attach to Mesh**: bind a vertex group to a target mesh using triangle + barycentric binding.
  - Enable **Attach** on the deformable and pick a target mesh
  - The target can be deforming; the addon evaluates the target mesh per frame

### Animated Colliders (PPF)

Animated colliders are supported when the collider’s topology stays constant.
At runtime the addon pushes updated collider vertex positions into the backend each tick.

## Baking / Cache

PPF can bake a cache and drive the result via Blender’s Mesh Cache modifier (PC2). Use:
- **AndoSim → PPF → Scene → Bake Cache**

## Diagnostics & Logs

When PPF fails with something like `PPF step failed: failed to advance`, the backend writes trace streams to:

- `output_dir/data/*.out`

The addon also prints tails of `advance.out` / `initialize.out` to the Blender console.

Important:
- Traces are appended across runs in the same output directory.
- For clean repros, delete `output_dir/data/*.out` or use a fresh output directory.

## Clean Reinstall / Cache

Blender can cache extension state. If you install a new zip but Blender still loads old code/wheels:

- Disable/uninstall older copies of the extension (Preferences → Extensions)
- If needed, wipe Blender user cache dirs (this resets Blender state):
  - `~/.config/blender`
  - `~/.local/share/blender`
  - `~/.cache/blender`

## Build From Source (Maintainers)

### Build the extension zip

From the repo root:

- `./build.sh`

This will:
- Build the Ando core native module from `ando_core_src/`
- Copy it into `blender_addon/ando/`
- Package the Blender Extension zip into `dist/`

### Build prerequisites (native Ando core)

You typically need:
- CMake + a C++ compiler toolchain
- Python 3.11 development headers (`python3.11-dev` on Debian/Ubuntu-like distros)
- `pybind11` and Eigen (the build uses system discovery)

The build script creates/uses a local virtualenv at `.venv/` to keep Python tooling isolated.

### Building the PPF wheel

The shipped zip includes a prebuilt `ppf_cts_backend` wheel.
If you need to rebuild it locally, you’ll need:
- Rust toolchain
- `maturin`
- A working CUDA/NVIDIA environment compatible with the upstream solver

This repo vendors the wheel wrapper source in `ppf_cts_backend/`.

It also vendors the required `ppf-contact-solver/` Rust/CUDA source tree used by the wheel.

Build and copy the wheel into the extension:
- `tools/build_ppf_wheel.sh`

Then rebuild the extension zip:
- `./build.sh`

## License

This extension bundle includes code and binaries from multiple upstream projects with different licenses.
See each upstream repository’s license files for the authoritative terms.

## Credits (Upstream Projects)

This addon bundles and/or builds on the following upstream repositories:

- **AndoSim** (Blender add-on + solver core) — Copyright (c) 2025 Hamish Burke — MIT License
  - https://github.com/Slaymish/AndoSim
- **ppf-contact-solver** (GPU contact solver) — by ZOZO, Inc. — Apache License 2.0
  - https://github.com/st-tech/ppf-contact-solver

If you believe a credit or license notice is missing or incorrect, please open an issue/PR so it can be fixed.
