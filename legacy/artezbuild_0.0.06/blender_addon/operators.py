import bpy
import os
import tempfile
import mathutils
from pathlib import Path

from . import ppf_export


def _get_settings(context):
    return getattr(context.scene, "andosim_artezbuild", None)


def _try_import_backend():
    try:
        import ppf_cts_backend  # type: ignore

        return ppf_cts_backend
    except Exception as exc:
        return exc


class _PPFState:
    def __init__(self):
        self.timer = None
        self.session = None
        self.running = False
        self.deformable_slices = []  # list[(name, start, count)]


class ANDOSIM_ARTEZBUILD_OT_ppf_run(bpy.types.Operator):
    bl_idname = "andosim_artezbuild.ppf_run"
    bl_label = "Run PPF (In-Process)"
    bl_options = {"REGISTER"}

    _ppf = None

    def execute(self, context):
        backend_or_exc = _try_import_backend()
        if isinstance(backend_or_exc, Exception):
            self.report({"ERROR"}, f"Failed to import ppf_cts_backend: {backend_or_exc}")
            return {"CANCELLED"}

        settings = _get_settings(context)
        if settings is None:
            self.report({"ERROR"}, "Addon settings not initialized")
            return {"CANCELLED"}

        state = _PPFState()
        state.running = True
        state.deformable_slices = []

        try:
            scene_path = (settings.scene_path or "").strip()
            if bool(getattr(settings, "auto_export", True)) or not scene_path:
                export = None
                try:
                    export = ppf_export.export_ppf_scene_from_roles(context, settings)
                except Exception:
                    export = None

                if export is None:
                    obj = settings.target_object or context.active_object
                    if obj is None or obj.type != "MESH":
                        self.report({"ERROR"}, "Pick a Target mesh object or enable role-tagged deformables")
                        return {"CANCELLED"}
                    if obj.mode != "OBJECT":
                        self.report({"ERROR"}, "Target object must be in Object Mode")
                        return {"CANCELLED"}
                    colliders = []
                    if bool(getattr(settings, "use_selected_colliders", True)):
                        colliders = [
                            o
                            for o in getattr(context, "selected_objects", [])
                            if o is not None and o.type == "MESH" and o.name != obj.name
                        ]
                    export = ppf_export.export_ppf_scene(context, settings, obj, colliders)

                for w in getattr(export, "warnings", []) or []:
                    self.report({"WARNING"}, str(w))

                scene_path = export.scene_path
                settings.scene_path = scene_path
                state.deformable_slices = list(export.deformable_slices)

            if not scene_path:
                self.report({"ERROR"}, "No PPF scene path")
                return {"CANCELLED"}

            if not state.deformable_slices:
                obj = settings.target_object or context.active_object
                if obj is None or obj.type != "MESH":
                    self.report({"ERROR"}, "Pick a Target mesh object")
                    return {"CANCELLED"}
                if obj.mode != "OBJECT":
                    self.report({"ERROR"}, "Target object must be in Object Mode")
                    return {"CANCELLED"}
                state.deformable_slices = [(obj.name, 0, len(obj.data.vertices))]

            output_dir = (settings.output_dir or "").strip()
            if not output_dir:
                output_dir = tempfile.mkdtemp(prefix="ppf_blender_")
                settings.output_dir = output_dir

            state.session = backend_or_exc.Session(scene_path, output_dir)
        except Exception as exc:
            self.report({"ERROR"}, f"Failed to create Session(): {exc}")
            return {"CANCELLED"}

        settings.running = True

        wm = context.window_manager
        fps = int(settings.fps) if int(settings.fps) > 0 else 30
        state.timer = wm.event_timer_add(1.0 / float(fps), window=context.window)
        wm.modal_handler_add(self)
        self.__class__._ppf = state
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        state = self.__class__._ppf
        if state is None or not state.running:
            return {"CANCELLED"}

        if event.type == "ESC":
            self._stop(context)
            return {"CANCELLED"}

        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        if not state.deformable_slices:
            self.report({"ERROR"}, "No deformables tracked; re-run simulation")
            self._stop(context)
            return {"CANCELLED"}

        # Validate objects and build a single flat vertex buffer.
        total_verts = 0
        objs = []  # (obj, start, count)
        for name, start, count in state.deformable_slices:
            obj = bpy.data.objects.get(name)
            if obj is None or obj.type != "MESH":
                self.report({"ERROR"}, f"Deformable '{name}' was removed")
                self._stop(context)
                return {"CANCELLED"}
            if obj.mode != "OBJECT":
                self.report({"ERROR"}, f"Deformable '{name}' must be in Object Mode")
                self._stop(context)
                return {"CANCELLED"}
            if len(obj.data.vertices) != int(count):
                self.report({"ERROR"}, f"Vertex count changed on '{name}'; stopping")
                self._stop(context)
                return {"CANCELLED"}
            objs.append((obj, int(start), int(count)))
            total_verts = max(total_verts, int(start) + int(count))

        try:
            curr = [0.0] * (total_verts * 3)
            for obj, start, count in objs:
                mw = obj.matrix_world
                mesh = obj.data
                for i, v in enumerate(mesh.vertices):
                    w = mw @ v.co
                    sx, sy, sz = ppf_export.blender_to_solver_xyz(float(w.x), float(w.y), float(w.z))
                    j = start + i
                    curr[3 * j + 0] = sx
                    curr[3 * j + 1] = sy
                    curr[3 * j + 2] = sz

            out = state.session.step(curr)
            if len(out) != len(curr):
                raise RuntimeError(f"Backend returned {len(out)} floats, expected {len(curr)}")

            for obj, start, count in objs:
                mw_inv = obj.matrix_world.inverted_safe()
                mesh = obj.data
                for i, v in enumerate(mesh.vertices):
                    j = start + i
                    wx, wy, wz = ppf_export.solver_to_blender_xyz(
                        float(out[3 * j + 0]),
                        float(out[3 * j + 1]),
                        float(out[3 * j + 2]),
                    )
                    local = mw_inv @ mathutils.Vector((wx, wy, wz))
                    v.co.x = float(local.x)
                    v.co.y = float(local.y)
                    v.co.z = float(local.z)
                mesh.update()
                obj.update_tag()
        except Exception as exc:
            details = _ppf_failure_details(getattr(_get_settings(context), "output_dir", ""))
            self.report({"ERROR"}, f"PPF step failed: {exc}. {details}")
            self._stop(context)
            return {"CANCELLED"}

        return {"PASS_THROUGH"}

    def _stop(self, context):
        state = self.__class__._ppf
        if state is None:
            return

        settings = _get_settings(context)
        if settings is not None:
            settings.running = False

        state.running = False
        if state.timer is not None:
            context.window_manager.event_timer_remove(state.timer)
            state.timer = None
        if state.session is not None:
            try:
                state.session.close()
            except Exception:
                pass
            state.session = None
        self.__class__._ppf = None


class ANDOSIM_ARTEZBUILD_OT_ppf_stop(bpy.types.Operator):
    bl_idname = "andosim_artezbuild.ppf_stop"
    bl_label = "Stop PPF"
    bl_options = {"REGISTER"}

    def execute(self, context):
        settings = _get_settings(context)
        if settings is not None:
            settings.running = False

        op = ANDOSIM_ARTEZBUILD_OT_ppf_run
        state = getattr(op, "_ppf", None)
        if state is None:
            return {"CANCELLED"}
        state.running = False
        if state.timer is not None:
            context.window_manager.event_timer_remove(state.timer)
            state.timer = None
        if state.session is not None:
            try:
                state.session.close()
            except Exception:
                pass
            state.session = None
        op._ppf = None
        return {"FINISHED"}


def _tail_text_file(path: Path, max_lines: int = 25) -> str:
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


def _ppf_failure_details(output_dir: str) -> str:
    """Return a concise hint pointing to backend traces.

    The backend writes debug trace files into `${output_dir}/data/*.out`.
    These are often the only actionable clue behind generic errors like
    'failed to advance'.
    """

    out = (output_dir or "").strip()
    if not out:
        return "(no output_dir set)"

    data_dir = Path(out) / "data"
    if not data_dir.exists():
        return f"(output_dir={out}; no data/ traces found)"

    # Prefer the main trace.
    advance = data_dir / "advance.out"
    init = data_dir / "initialize.out"

    snippets = []
    adv_tail = _tail_text_file(advance)
    if adv_tail:
        snippets.append("advance.out tail:\n" + adv_tail)
    init_tail = _tail_text_file(init)
    if init_tail:
        snippets.append("initialize.out tail:\n" + init_tail)

    # Always print to console for full visibility.
    try:
        print("[PPF] step failed; traces in:", str(data_dir))
        for snip in snippets:
            print("[PPF] ---")
            print(snip)
    except Exception:
        pass

    # Keep the UI message short.
    return f"(see traces in {data_dir})"
