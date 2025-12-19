import os
import sys
import tempfile
import zipfile
import shutil


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


def main():
    whl = os.environ.get(
        "PPF_WHL",
        "/home/moritz/repos/artezbuild_0.0.06/blender_addon/wheels/ppf_cts_backend-0.0.1-cp311-cp311-manylinux_2_34_x86_64.whl",
    )
    outdir = os.environ.get("OUTDIR", "")
    if not outdir:
        outdir = tempfile.mkdtemp(prefix="ppf_out_")

    ppf_cts_backend, extracted = _import_ppf_backend_from_whl(whl)
    print("ppf_cts_backend", getattr(ppf_cts_backend, "__version__", "?"))

    import bpy  # noqa: E402

    # Ensure our addon is importable from repo.
    repo_root = os.environ.get("ADDON_ROOT", "/home/moritz/repos/artezbuild_0.0.06")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import importlib  # noqa: E402

    addon = importlib.import_module("blender_addon")
    addon.register()

    try:
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=20, y_subdivisions=20, size=1.0)
        obj = bpy.context.active_object
        obj.name = "Plane"

        props = getattr(obj, "andosim_artezbuild")
        props.enabled = True
        props.role = "DEFORMABLE"

        settings = bpy.context.scene.andosim_artezbuild
        settings.auto_export = True
        settings.use_selected_colliders = False
        settings.output_dir = outdir

        from blender_addon import ppf_export  # noqa: E402

        export = ppf_export.export_ppf_scene_from_roles(bpy.context, settings)
        print("scene", export.scene_path)

        sess = ppf_cts_backend.Session(export.scene_path, settings.output_dir)
        try:
            import blender_addon.ppf_export as pe  # noqa: E402

            curr = [0.0] * (len(obj.data.vertices) * 3)
            mw = obj.matrix_world
            for i, v in enumerate(obj.data.vertices):
                w = mw @ v.co
                sx, sy, sz = pe.blender_to_solver_xyz(float(w.x), float(w.y), float(w.z))
                curr[3 * i + 0] = sx
                curr[3 * i + 1] = sy
                curr[3 * i + 2] = sz

            print("step...")
            out = sess.step(curr)
            print("outlen", len(out))
        finally:
            try:
                sess.close()
            except Exception:
                pass
    finally:
        try:
            addon.unregister()
        except Exception:
            pass
        shutil.rmtree(extracted, ignore_errors=True)


if __name__ == "__main__":
    main()
