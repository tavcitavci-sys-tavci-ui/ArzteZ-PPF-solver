# AndoSim ArteZbuild — Install & Quick Start (ZIP)

This guide assumes you already have the extension zip:
- `andosim_artezbuild-0.0.8.zip`

## Install

1. Open Blender (4.2+).
2. Go to **Edit → Preferences → Extensions**.
3. Click **Install from Disk…**
4. Select `andosim_artezbuild-0.0.8.zip`.
5. After it installs, **enable** the extension (toggle on).

If you updated the zip and Blender still behaves like an older version:
- Disable/uninstall older copies in Preferences → Extensions.
- If needed, restart Blender.

## Where is the UI?

- In the 3D Viewport, open the right sidebar (press `N`).
- Find the **AndoSim** tab.

## Quick Start (PPF / GPU)

1. Select your cloth mesh.
2. In **AndoSim → PPF → Active Object**:
   - Enable PPF for the object
   - Set **Role = Deformable**
3. Select your collider meshes and set **Role = Static Collider**.
4. In **AndoSim → PPF → Scene**, set **Output Dir** (recommended).
5. Press **Run**.

If the PPF panel shows `ppf_cts_backend missing`, the extension is not installed/enabled correctly (it must be installed from the zip as an Extension).

## Quick Start (Ando / CPU)

1. Select your cloth mesh.
2. In **AndoSim → Ando** (CPU panel):
   - Pick the target object
   - Press the run/play operator

If the Ando panel says the core module is missing:
- You likely installed the wrong build for your Blender/Python ABI. This zip targets Blender’s Python 3.11 (`cp311`).

## Pins (PPF)

- **Pin Handles (recommended)**: create Empty handles for pinned vertices using the addon operator in Edit Mode, then move the Empty in Object Mode.
- **Attach to Mesh**: enable Attach on the deformable, choose a target mesh, and paint the attach vertex group.

## When something fails (PPF)

- PPF writes traces to: `Output Dir/data/*.out`
- Start with: `advance.out` and `initialize.out`
- Tip: traces append across runs; for clean logs, delete old `*.out` or use a new output dir.
