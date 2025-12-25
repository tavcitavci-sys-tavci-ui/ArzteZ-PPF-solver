import os
import sys
import tempfile
import shutil
import zipfile
from pathlib import Path
import math


def _import_ppf_backend_from_whl(whl_path: str):
    td = tempfile.mkdtemp(prefix="ppf_whl_")
    zipfile.ZipFile(whl_path).extractall(td)
    sys.path.insert(0, td)
    try:
        import ppf_cts_backend  # type: ignore

        return ppf_cts_backend, td
    except Exception:
        shutil.rmtree(td, ignore_errors=True)
        raise


def _require(name: str, obj, attr: str):
    if not hasattr(obj, attr):
        raise RuntimeError(f"Missing expected attribute: {name}.{attr}")


def _finite_vec(values, label: str):
    for i, v in enumerate(values):
        if not math.isfinite(float(v)):
            raise RuntimeError(f"Non-finite {label}[{i}]={v}")


def main() -> int:
    # Config via env vars so it's easy to run from terminal.
    repo_root = os.environ.get("ADDON_ROOT", "/home/moritz/repos/artezbuild_0.0.06")
    ppf_whl = os.environ.get(
        "PPF_WHL",
        "/home/moritz/repos/artezbuild_0.0.06/blender_addon/wheels/ppf_cts_backend-0.0.1-cp311-cp311-manylinux_2_34_x86_64.whl",
    )

    base_outdir = (os.environ.get("OUTDIR", "").strip() or None)
    if base_outdir is None:
        base_outdir = tempfile.mkdtemp(prefix="andosim_smoke_")

    run_outdir = str(Path(base_outdir) / "headless_full_smoke")
    Path(run_outdir).mkdir(parents=True, exist_ok=True)

    print("[SMOKE] repo_root:", repo_root)
    print("[SMOKE] outdir:", run_outdir)
    print("[SMOKE] ppf_whl:", ppf_whl)

    # Import backend wheel from disk (zipimport can't load native modules).
    ppf_cts_backend, ppf_extracted = _import_ppf_backend_from_whl(ppf_whl)
    print("[SMOKE] ppf_cts_backend version:", getattr(ppf_cts_backend, "__version__", "?"))

    try:
        import bpy  # type: ignore

        # Ensure repo root is importable.
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        import importlib

        addon = importlib.import_module("blender_addon")
        addon.register()

        try:
            # Sanity: settings exist
            settings = getattr(bpy.context.scene, "andosim_artezbuild", None)
            if settings is None:
                raise RuntimeError("Scene missing andosim_artezbuild settings")

            # Quick sanity: expected settings fields
            for attr in ("output_dir", "scene_path", "fps", "dt"):
                _require("settings", settings, attr)

            # Ando core: must be native (not fallback) for real functionality.
            from blender_addon.ando import _core_loader  # type: ignore

            core = _core_loader.get_core_module(strict=True, context="headless smoke test")
            if core is None:
                raise RuntimeError("ando_barrier_core could not be loaded")

            ver = getattr(core, "version", None)
            if callable(ver):
                print("[SMOKE] ando_barrier_core version:", core.version())

            # Verify we got the compiled core by checking barrier_* exists.
            for attr in ("barrier_energy", "barrier_gradient", "barrier_hessian"):
                _require("ando_barrier_core", core, attr)

            # Exercise barrier functions with simple values.
            e = core.barrier_energy(0.5, 1.0, 100.0)
            g = core.barrier_gradient(0.5, 1.0, 100.0)
            h = core.barrier_hessian(0.5, 1.0, 100.0)
            print("[SMOKE] barrier_energy/gradient/hessian:", e, g, h)

            # PPF: Build a minimal scene: grid deformable + cube collider.
            bpy.ops.wm.read_factory_settings(use_empty=True)

            bpy.ops.mesh.primitive_grid_add(x_subdivisions=25, y_subdivisions=25, size=1.0)
            deform = bpy.context.active_object
            deform.name = "Deform"

            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -0.6))
            coll = bpy.context.active_object
            coll.name = "Collider"

            dprops = getattr(deform, "andosim_artezbuild")
            dprops.enabled = True
            dprops.role = "DEFORMABLE"

            cprops = getattr(coll, "andosim_artezbuild")
            cprops.enabled = True
            cprops.role = "STATIC_COLLIDER"

            settings = bpy.context.scene.andosim_artezbuild
            settings.auto_export = True
            settings.use_selected_colliders = False
            settings.output_dir = run_outdir

            from blender_addon import ppf_export  # type: ignore

            export = ppf_export.export_ppf_scene_from_roles(bpy.context, settings)
            print("[SMOKE] ppf scene_path:", export.scene_path)
            print("[SMOKE] deformable_slices:", getattr(export, "deformable_slices", None))

            # Assemble curr position buffer.
            slices = list(getattr(export, "deformable_slices", []) or [])
            if not slices:
                raise RuntimeError("PPF export produced no deformable_slices")

            # For this smoke test we assume one deformable.
            name, start, count = slices[0]
            obj = bpy.data.objects.get(name)
            if obj is None:
                raise RuntimeError(f"Missing deformable object '{name}'")

            total_verts = int(start) + int(count)
            curr = [0.0] * (total_verts * 3)
            mw = obj.matrix_world
            for i, v in enumerate(obj.data.vertices):
                w = mw @ v.co
                sx, sy, sz = ppf_export.blender_to_solver_xyz(float(w.x), float(w.y), float(w.z))
                j = int(start) + i
                curr[3 * j + 0] = sx
                curr[3 * j + 1] = sy
                curr[3 * j + 2] = sz

            _finite_vec(curr, "curr")

            sess = ppf_cts_backend.Session(export.scene_path, run_outdir)
            try:
                out = sess.step(curr)
            finally:
                try:
                    sess.close()
                except Exception:
                    pass

            print("[SMOKE] ppf step ok; outlen:", len(out))
            _finite_vec(out, "out")

            data_dir = Path(run_outdir) / "data"
            if data_dir.exists():
                traces = sorted(p.name for p in data_dir.glob("*.out"))
                print("[SMOKE] traces:", len(traces), "files")
            else:
                print("[SMOKE] no traces directory found at", str(data_dir))

        finally:
            try:
                addon.unregister()
            except Exception:
                pass

    finally:
        shutil.rmtree(ppf_extracted, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
