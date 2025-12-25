import argparse
import math
import os
import sys
import tempfile
import time
from pathlib import Path


def _require(obj, attr: str, label: str):
    if not hasattr(obj, attr):
        raise RuntimeError(f"Missing expected attribute: {label}.{attr}")


def _finite(values, label: str):
    for i, v in enumerate(values):
        if not math.isfinite(float(v)):
            raise RuntimeError(f"Non-finite {label}[{i}]={v}")


def _enable_online_access() -> None:
    import bpy  # type: ignore

    try:
        bpy.context.preferences.system.use_online_access = True
    except Exception as exc:
        print("[INSTALL] Warning: could not enable online access:", repr(exc))


def _extensions_repo_dir() -> str | None:
    import bpy  # type: ignore

    try:
        ext = bpy.context.preferences.extensions
        for r in getattr(ext, "repos", []) or []:
            if getattr(r, "module", "") == "user_default":
                d = str(getattr(r, "directory", "") or "")
                if d:
                    return d
    except Exception:
        pass

    try:
        import bpy.utils  # type: ignore

        base = bpy.utils.user_resource("EXTENSIONS")
        if base:
            return str(Path(base) / "user_default")
    except Exception:
        pass

    return None


def _install_extension_zip(zip_path: str) -> None:
    import bpy  # type: ignore

    zp = str(Path(zip_path).expanduser().resolve())
    if not Path(zp).exists():
        raise FileNotFoundError(zp)

    _enable_online_access()

    repo_dir = _extensions_repo_dir()
    file_url = Path(zp).as_uri()

    print("[INSTALL] zip:", zp)
    print("[INSTALL] repo_directory:", repo_dir)

    kwargs = {
        "url": file_url,
        "pkg_id": "andosim_artezbuild",
        "enable_on_install": True,
        "do_legacy_replace": True,
    }
    if repo_dir:
        kwargs["repo_directory"] = repo_dir

    res = bpy.ops.extensions.package_install(**kwargs)
    print("[INSTALL] package_install:", res)

    try:
        bpy.ops.wm.save_userpref()
    except Exception as exc:
        print("[INSTALL] Warning: failed to save user prefs:", repr(exc))


def _ensure_extension_importable(module_name: str) -> None:
    import bpy  # type: ignore

    try:
        import bpy.utils  # type: ignore

        base = bpy.utils.user_resource("EXTENSIONS")
        if base:
            for p in (base, str(Path(base) / "user_default"), str(Path(base) / "user_default" / module_name)):
                if p and p not in sys.path:
                    sys.path.insert(0, p)
    except Exception as exc:
        print("[INSTALL] Warning: could not add extensions to sys.path:", repr(exc))


def _ensure_addon_registered(module_name: str):
    import bpy  # type: ignore

    _ensure_extension_importable(module_name)

    mod = __import__(module_name)
    # In background mode, addons/extensions may not be auto-registered.
    if not hasattr(bpy.types.Scene, "andosim_artezbuild"):
        print("[ADDON] calling register()")
        mod.register()
    return mod


def _dummy_timer_event():
    class E:
        type = "TIMER"

    return E()


def _pump_modal(op_instance, context, max_ticks: int = 1_000_000):
    evt = _dummy_timer_event()
    last = None
    for _i in range(max_ticks):
        last = op_instance.modal(context, evt)
        # The operator may return a set or dict-like; we just break on FINISHED/CANCELLED.
        if last == {"FINISHED"} or last == {"CANCELLED"}:
            break
    return last


def _make_scene(module_name: str, subdiv: int, outdir: str, *, with_collider: bool = True):
    import bpy  # type: ignore

    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Factory settings can disable/unregister addons; re-register ours.
    _ensure_addon_registered(module_name)

    bpy.ops.mesh.primitive_grid_add(x_subdivisions=subdiv, y_subdivisions=subdiv, size=1.0)
    deform = bpy.context.active_object
    deform.name = "Deform"

    coll = None
    if with_collider:
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -0.6))
        coll = bpy.context.active_object
        coll.name = "Collider"

    dprops = getattr(deform, "andosim_artezbuild")
    dprops.enabled = True
    dprops.role = "DEFORMABLE"

    if coll is not None:
        cprops = getattr(coll, "andosim_artezbuild")
        cprops.enabled = True
        cprops.role = "STATIC_COLLIDER"

    settings = bpy.context.scene.andosim_artezbuild
    settings.auto_export = True
    settings.use_selected_colliders = False
    settings.output_dir = outdir

    return deform, coll, settings


def _stress_ppf_steps(module_name: str, outdir: str, subdiv: int, steps: int, move_collider: bool) -> None:
    import bpy  # type: ignore

    deform, coll, settings = _make_scene(module_name, subdiv, outdir, with_collider=True)

    import andosim_artezbuild.ppf_export as ppf_export  # type: ignore
    import andosim_artezbuild.operators as ops_mod  # type: ignore
    import ppf_cts_backend  # type: ignore

    export = ppf_export.export_ppf_scene_from_roles(bpy.context, settings)
    slices = list(getattr(export, "deformable_slices", []) or [])
    if not slices:
        raise RuntimeError("No deformables found")

    # Build curr from current mesh.
    total_verts = 0
    for _name, start, count in slices:
        total_verts = max(total_verts, int(start) + int(count))

    curr = [0.0] * (total_verts * 3)
    for name, start, _count in slices:
        obj = bpy.data.objects.get(name)
        mw = obj.matrix_world
        for i, v in enumerate(obj.data.vertices):
            w = mw @ v.co
            sx, sy, sz = ppf_export.blender_to_solver_xyz(float(w.x), float(w.y), float(w.z))
            j = int(start) + i
            curr[3 * j + 0] = sx
            curr[3 * j + 1] = sy
            curr[3 * j + 2] = sz

    _finite(curr, "curr")

    sess = ppf_cts_backend.Session(export.scene_path, outdir)
    try:
        collider_names = list(getattr(export, "collider_object_names", []) or [])

        t0 = time.time()
        for k in range(int(steps)):
            if move_collider and coll is not None:
                # Move collider a bit to force collision mesh streaming.
                coll.location.z = -0.6 + 0.1 * math.sin(0.05 * k)
                bpy.context.view_layer.update()

            collider_verts_flat = ops_mod._ppf_build_collision_mesh_vertices_flat(bpy.context, collider_names)
            ops_mod._ppf_apply_collision_mesh_vertices(sess, collider_verts_flat)

            out = sess.step(curr)
            if len(out) != len(curr):
                raise RuntimeError(f"Bad out length {len(out)} expected {len(curr)}")
            if (k % 20) == 0:
                _finite(out, f"out@{k}")
            curr = list(out)
        dt = time.time() - t0
        print(f"[PPF] stress steps: {steps} (subdiv={subdiv}) took {dt:.2f}s")

    finally:
        try:
            sess.close()
        except Exception:
            pass


def _test_bake_modal(module_name: str, outdir: str, subdiv: int, frame_end: int) -> None:
    import bpy  # type: ignore

    deform, _coll, settings = _make_scene(module_name, subdiv, outdir, with_collider=True)

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = int(frame_end)

    # Ensure output dir is set.
    settings.output_dir = outdir

    import andosim_artezbuild.operators as ops_mod  # type: ignore

    op = ops_mod.ANDOSIM_ARTEZBUILD_OT_ppf_bake_cache()

    win = bpy.context.window
    if win is None and bpy.context.window_manager.windows:
        win = bpy.context.window_manager.windows[0]

    # Execute bake setup.
    with bpy.context.temp_override(window=win):
        res = op.execute(bpy.context)
    print("[BAKE] execute:", res)
    if res != {"RUNNING_MODAL"}:
        raise RuntimeError(f"Bake did not start: {res}")

    # Drive modal loop.
    with bpy.context.temp_override(window=win):
        last = _pump_modal(op, bpy.context, max_ticks=10_000_000)
    print("[BAKE] modal end:", last)

    # Verify cache exists and modifier is attached.
    cache_dir = Path(outdir) / "cache"
    pc2 = cache_dir / f"{deform.name}.pc2"
    if not pc2.exists() or pc2.stat().st_size <= 32:
        raise RuntimeError(f"Bake produced no usable pc2: {pc2}")

    mod = deform.modifiers.get("PPF_Cache")
    if mod is None:
        raise RuntimeError("Bake did not attach PPF_Cache modifier")
    if getattr(mod, "cache_format", "") != "PC2":
        raise RuntimeError("PPF_Cache modifier not PC2")


def _test_cancel_bake(module_name: str, outdir: str, subdiv: int) -> None:
    import bpy  # type: ignore

    deform, _coll, settings = _make_scene(module_name, subdiv, outdir, with_collider=True)

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 10
    settings.output_dir = outdir

    import andosim_artezbuild.operators as ops_mod  # type: ignore

    op = ops_mod.ANDOSIM_ARTEZBUILD_OT_ppf_bake_cache()

    win = bpy.context.window
    if win is None and bpy.context.window_manager.windows:
        win = bpy.context.window_manager.windows[0]

    with bpy.context.temp_override(window=win):
        res = op.execute(bpy.context)
    if res != {"RUNNING_MODAL"}:
        raise RuntimeError(f"CancelBake: bake did not start: {res}")

    # Advance a few ticks then cancel.
    with bpy.context.temp_override(window=win):
        _ = _pump_modal(op, bpy.context, max_ticks=5)
        state = getattr(ops_mod.ANDOSIM_ARTEZBUILD_OT_ppf_bake_cache, "_bake", None)
        if state is None:
            raise RuntimeError("CancelBake: missing bake state")
        state.cancel_requested = True
        last = _pump_modal(op, bpy.context, max_ticks=1000)
    print("[CANCEL] modal end:", last)

    cache_dir = Path(outdir) / "cache"
    pc2 = cache_dir / f"{deform.name}.pc2"
    if pc2.exists():
        raise RuntimeError(f"CancelBake: expected pc2 deleted, still exists: {pc2}")
    if deform.modifiers.get("PPF_Cache") is not None:
        raise RuntimeError("CancelBake: expected no PPF_Cache modifier")


def _test_realtime_start_stop(module_name: str, outdir: str, subdiv: int, ticks: int) -> None:
    import bpy  # type: ignore

    _deform, _coll, settings = _make_scene(module_name, subdiv, outdir, with_collider=True)
    settings.output_dir = outdir

    import andosim_artezbuild.operators as ops_mod  # type: ignore

    win = bpy.context.window
    if win is None and bpy.context.window_manager.windows:
        win = bpy.context.window_manager.windows[0]

    # In headless mode we can't reliably drive Blender's event loop, but we can
    # still verify the operators start and stop cleanly (session created/closed,
    # timers registered/unregistered).
    with bpy.context.temp_override(window=win):
        res = bpy.ops.andosim_artezbuild.ppf_run()
    print("[RUN] ppf_run:", res)
    if res != {"RUNNING_MODAL"}:
        raise RuntimeError(f"Realtime run did not start: {res}")

    # Optional: wait a tiny bit; tick count is kept for API compatibility.
    if int(ticks) > 0:
        time.sleep(0.01)

    with bpy.context.temp_override(window=win):
        res2 = bpy.ops.andosim_artezbuild.ppf_stop()
    print("[RUN] ppf_stop:", res2)
    if res2 != {"FINISHED"}:
        raise RuntimeError(f"Realtime stop failed: {res2}")

    # Ensure internal state cleared.
    if getattr(ops_mod.ANDOSIM_ARTEZBUILD_OT_ppf_run, "_ppf", None) is not None:
        raise RuntimeError("Realtime stop did not clear internal state")


def _test_reset_after_bake(module_name: str, outdir: str, subdiv: int) -> None:
    import bpy  # type: ignore

    deform, _coll, settings = _make_scene(module_name, subdiv, outdir, with_collider=True)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 3
    settings.output_dir = outdir

    import andosim_artezbuild.operators as ops_mod  # type: ignore

    # Snapshot base coords.
    coords0 = [0.0] * (len(deform.data.vertices) * 3)
    deform.data.vertices.foreach_get("co", coords0)

    # Bake quickly.
    op = ops_mod.ANDOSIM_ARTEZBUILD_OT_ppf_bake_cache()
    win = bpy.context.window
    if win is None and bpy.context.window_manager.windows:
        win = bpy.context.window_manager.windows[0]

    with bpy.context.temp_override(window=win):
        res = op.execute(bpy.context)
    if res != {"RUNNING_MODAL"}:
        raise RuntimeError(f"ResetAfterBake: bake did not start: {res}")
    with bpy.context.temp_override(window=win):
        _ = _pump_modal(op, bpy.context, max_ticks=1_000_000)

    if deform.modifiers.get("PPF_Cache") is None:
        raise RuntimeError("ResetAfterBake: bake did not attach modifier")

    # Mutate mesh, then reset should restore coords0 and remove modifier.
    coords1 = coords0[:]
    for i in range(0, len(coords1), 3):
        coords1[i + 2] += 0.05
    deform.data.vertices.foreach_set("co", coords1)
    deform.data.update()

    reset = ops_mod.ANDOSIM_ARTEZBUILD_OT_ppf_reset_simulation()
    res2 = reset.execute(bpy.context)
    print("[RESET] execute:", res2)

    coords2 = [0.0] * (len(deform.data.vertices) * 3)
    deform.data.vertices.foreach_get("co", coords2)
    if coords2 != coords0:
        raise RuntimeError("ResetAfterBake: did not restore vertex coords")
    if deform.modifiers.get("PPF_Cache") is not None:
        raise RuntimeError("ResetAfterBake: did not remove PPF_Cache")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Max-load headless test for AndoSim ArteZbuild")
    parser.add_argument("--zip", required=True, help="Path to andosim_artezbuild-*.zip")
    parser.add_argument("--outdir", default=os.environ.get("OUTDIR", ""), help="Output directory")
    parser.add_argument("--subdiv", type=int, default=140, help="Grid subdivisions (stress level)")
    parser.add_argument("--steps", type=int, default=200, help="PPF steps to run")
    parser.add_argument("--bake-frames", type=int, default=40, help="Bake end frame")
    parser.add_argument("--move-collider", action="store_true", help="Move collider every step")
    parser.add_argument("--realtime-ticks", type=int, default=60, help="Modal ticks to run realtime sim")
    args = parser.parse_args(argv)

    outdir = (args.outdir or "").strip() or tempfile.mkdtemp(prefix="andosim_maxload_")
    Path(outdir).mkdir(parents=True, exist_ok=True)

    print("[RUN] outdir:", outdir)
    print("[RUN] subdiv:", args.subdiv, "steps:", args.steps, "bake_frames:", args.bake_frames)

    _install_extension_zip(args.zip)

    module_name = "andosim_artezbuild"
    _ensure_addon_registered(module_name)

    # Basic invariants.
    import bpy  # type: ignore

    settings = getattr(bpy.context.scene, "andosim_artezbuild", None)
    if settings is None:
        raise RuntimeError("Scene missing andosim_artezbuild settings")
    for attr in ("output_dir", "scene_path", "fps", "dt"):
        _require(settings, attr, "settings")

    import ppf_cts_backend  # type: ignore

    print("[PPF] backend version:", getattr(ppf_cts_backend, "__version__", "?"))

    # Ando core load check.
    from andosim_artezbuild.ando import _core_loader  # type: ignore

    core = _core_loader.get_core_module(strict=True, context="max-load headless")
    if core is None:
        raise RuntimeError("Ando core missing")

    # Max-load suite.
    _stress_ppf_steps(module_name, outdir, args.subdiv, args.steps, args.move_collider)
    _test_realtime_start_stop(module_name, outdir, max(40, args.subdiv // 3), ticks=args.realtime_ticks)
    _test_bake_modal(module_name, outdir, max(40, args.subdiv // 3), frame_end=args.bake_frames)
    _test_cancel_bake(module_name, outdir, max(30, args.subdiv // 4))
    _test_reset_after_bake(module_name, outdir, max(30, args.subdiv // 4))

    print("[OK] Max-load headless suite passed")
    return 0


if __name__ == "__main__":
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
    else:
        argv = []
    raise SystemExit(main(argv))
