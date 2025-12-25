# Changelog

This changelog tracks the **AndoSim ArteZbuild Blender Extension** version stream (see `blender_addon/blender_manifest.toml`).

## [0.0.9] - 2025-12-21
- Expands **Prepare Cloth Mesh** with mesh-quality reporting (angles/aspect/non-manifold) and stability suggestions (contact gap + dt).
- Adds redo-panel options to optionally apply suggested `tri_contact_gap` / `static_contact_gap` and `dt`.
- Adds one-click **Prepare Cloth Mesh** (Blender-internal voxel remesh + cleanup) for the selected mesh.
- Adds preference: duplicate mesh vs destructive remesh.
- Blocks remesh when pins/attach/stitches exist to avoid invalidating constraints.
- Updates extension licensing metadata to **GPL-3.0-or-later**.
- Reworks PPF panel layout into a single list with collapsible sections: Realtime / Bake / Settings.
- Adds Realtime: **Reset Simulation** (stops session, restores original mesh, removes `PPF_Cache` mesh-cache modifier).
- Makes Bake Cache interruptible and adds a **Cancel** button (cancels between frames, deletes partial `.pc2` files).
- Improves bake accuracy for animated scenes:
	- Fixes pin/collider streaming for Bake (previously could be a no-op due to session mismatch).
	- Updates pins and collider meshes every solver substep during Bake (accuracy-first; may be slower on heavy colliders).

