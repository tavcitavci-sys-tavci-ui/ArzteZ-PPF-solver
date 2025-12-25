# AndoSim ArteZbuild (Blender Extension)

This folder is a Blender 4.2+ **Extension** (it contains `blender_manifest.toml` at its root).

It bundles two simulation backends:
- **Ando (CPU)**: `ando_barrier_core` native module (built from AndoSim sources).
- **PPF (GPU)**: `ppf_cts_backend` wheel (PyO3) wrapping the upstream `ppf-contact-solver`.

## Install

1. Build the extension zip (from the repo root):
   - `./build.sh`
2. Install in Blender:
   - Edit → Preferences → Extensions → Install from Disk…
  - Select `dist/andosim_artezbuild-0.0.8.zip`
   - Enable the extension

## Supported Platforms

- Blender 4.2+
- Linux x64
- Blender Python 3.11 (bundled artifacts are `cp311`)

## Rebuilding the PPF wheel (Maintainers)

If you need to rebuild the bundled `ppf_cts_backend` wheel, the wrapper source is vendored in the repo root under `ppf_cts_backend/`.

- Build + copy wheel into `blender_addon/wheels/`: `tools/build_ppf_wheel.sh`
- Rebuild the extension zip: `./build.sh`

## Troubleshooting

- If PPF fails with `PPF step failed: failed to advance`, inspect traces in:
  - `output_dir/data/*.out`
- If Blender seems to run an older version after updating the zip, uninstall older copies or wipe Blender cache dirs:
  - `~/.config/blender`, `~/.local/share/blender`, `~/.cache/blender`

## Third-Party Licenses

See:
- `licenses/AndoSim-MIT.txt`
- `licenses/ppf-contact-solver-APACHE-2.0.txt`
- `THIRD_PARTY_NOTICES.md`
