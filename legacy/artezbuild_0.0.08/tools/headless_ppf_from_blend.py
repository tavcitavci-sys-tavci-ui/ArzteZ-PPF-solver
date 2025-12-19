import argparse
import math
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path


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


def _tail_text_file(path: Path, max_lines: int = 60) -> str:
    try:
        text = path.read_text(errors="replace")
    except Exception:
        return ""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    return "\n".join(lines)


def _print_trace_tails(output_dir: str) -> None:
    data_dir = Path(output_dir) / "data"
    if not data_dir.exists():
        print("[PPF] no traces at", str(data_dir))
        return

    for name in ("initialize.out", "advance.out"):
        p = data_dir / name
        tail = _tail_text_file(p)
        if tail:
            print("[PPF] ---", name, "tail ---")
            print(tail)

    # Quick NaN scan on key scalar traces.
    nan_hits = []
    for p in sorted(data_dir.glob("*.out")):
        try:
            txt = p.read_text(errors="replace")
        except Exception:
            continue
        if " nan" in txt or "\tnan" in txt or txt.strip().endswith("nan") or "nan\n" in txt:
            nan_hits.append(p.name)

    if nan_hits:
        print("[PPF] NaNs detected in:", ", ".join(nan_hits[:20]))
        if len(nan_hits) > 20:
            print("[PPF] ... and", len(nan_hits) - 20, "more")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run one headless PPF step from a .blend file")
    parser.add_argument("--blend", required=True, help="Path to .blend")
    parser.add_argument(
        "--outdir",
        default=os.environ.get("OUTDIR", ""),
        help="Base output directory (default: env OUTDIR or temp)",
    )
    parser.add_argument(
        "--ppf-whl",
        default=os.environ.get(
            "PPF_WHL",
            "/home/moritz/repos/artezbuild_0.0.06/blender_addon/wheels/ppf_cts_backend-0.0.1-cp311-cp311-manylinux_2_34_x86_64.whl",
        ),
        help="Path to ppf_cts_backend wheel",
    )
    parser.add_argument(
        "--addon-root",
        default=os.environ.get("ADDON_ROOT", "/home/moritz/repos/artezbuild_0.0.06"),
        help="Repo root containing blender_addon/",
    )
    parser.add_argument(
        "--no-subdir",
        action="store_true",
        help="Write traces directly into --outdir instead of a timestamped subdir",
    )

    args = parser.parse_args(argv)

    base_outdir = (args.outdir or "").strip()
    if not base_outdir:
        base_outdir = tempfile.mkdtemp(prefix="ppf_out_")

    if args.no_subdir:
        run_outdir = base_outdir
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_outdir = str(Path(base_outdir) / f"headless_run_{stamp}")

    Path(run_outdir).mkdir(parents=True, exist_ok=True)

    ppf_cts_backend, extracted = _import_ppf_backend_from_whl(args.ppf_whl)
    print("ppf_cts_backend", getattr(ppf_cts_backend, "__version__", "?"))

    try:
        import bpy  # type: ignore

        blend_path = str(Path(args.blend).expanduser())
        if not Path(blend_path).exists():
            raise FileNotFoundError(blend_path)

        # Ensure our addon is importable from repo.
        if args.addon_root not in sys.path:
            sys.path.insert(0, args.addon_root)

        import importlib

        addon = importlib.import_module("blender_addon")
        addon.register()

        try:
            bpy.ops.wm.open_mainfile(filepath=blend_path)

            settings = getattr(bpy.context.scene, "andosim_artezbuild", None)
            if settings is None:
                raise RuntimeError("Addon settings not found on scene")

            settings.auto_export = True
            settings.use_selected_colliders = False
            settings.output_dir = run_outdir

            from blender_addon import ppf_export  # type: ignore

            export = ppf_export.export_ppf_scene_from_roles(bpy.context, settings)
            print("scene_path", export.scene_path)
            print("output_dir", run_outdir)

            deformable_slices = list(getattr(export, "deformable_slices", []) or [])
            if not deformable_slices:
                raise RuntimeError("No deformables found (no role-tagged DEFORMABLE objects?)")

            total_verts = 0
            objs: list[tuple[object, int, int]] = []
            for name, start, count in deformable_slices:
                obj = bpy.data.objects.get(name)
                if obj is None or getattr(obj, "type", None) != "MESH":
                    raise RuntimeError(f"Deformable '{name}' missing or not a mesh")
                if getattr(obj, "mode", "OBJECT") != "OBJECT":
                    raise RuntimeError(f"Deformable '{name}' must be in Object Mode")
                if len(obj.data.vertices) != int(count):
                    raise RuntimeError(f"Vertex count changed on '{name}'")
                objs.append((obj, int(start), int(count)))
                total_verts = max(total_verts, int(start) + int(count))

            curr = [0.0] * (total_verts * 3)
            for obj, start, _count in objs:
                mw = obj.matrix_world
                mesh = obj.data
                for i, v in enumerate(mesh.vertices):
                    w = mw @ v.co
                    sx, sy, sz = ppf_export.blender_to_solver_xyz(float(w.x), float(w.y), float(w.z))
                    j = start + i
                    curr[3 * j + 0] = sx
                    curr[3 * j + 1] = sy
                    curr[3 * j + 2] = sz

            # Quick input sanity check.
            for k, val in enumerate(curr):
                if not math.isfinite(float(val)):
                    raise RuntimeError(f"Non-finite input curr[{k}]={val}")

            sess = ppf_cts_backend.Session(export.scene_path, run_outdir)
            try:
                print("step...")
                out = sess.step(curr)
                print("outlen", len(out))

                # Output sanity check.
                for k, val in enumerate(out):
                    if not math.isfinite(float(val)):
                        print(f"[PPF] Non-finite output out[{k}]={val}")
                        break
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

    except Exception as exc:
        print("[PPF] headless run failed:", repr(exc))
        try:
            _print_trace_tails(run_outdir)
        except Exception:
            pass
        return 2

    finally:
        shutil.rmtree(extracted, ignore_errors=True)

    _print_trace_tails(run_outdir)
    return 0


if __name__ == "__main__":
    # Blender passes its own args; we expect our args after '--'.
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
    else:
        argv = []
    raise SystemExit(main(argv))
