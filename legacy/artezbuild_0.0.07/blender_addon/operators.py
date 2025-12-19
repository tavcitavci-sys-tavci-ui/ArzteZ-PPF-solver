import bpy
import os
import tempfile
import mathutils
import struct
import bmesh
from array import array
from pathlib import Path
from bpy_extras import view3d_utils
from mathutils.bvhtree import BVHTree

from . import ppf_export


_PPF_PIN_HANDLE_TAG = "andosim_ppf_pin_handle"
_PPF_PIN_HANDLE_OBJ = "andosim_ppf_pin_obj"
_PPF_PIN_HANDLE_VIDX = "andosim_ppf_pin_vidx"

# Legacy grab-pin interaction state (still referenced by clear/delete operators).
_ACTIVE_PPF_GRABS: dict[str, dict] = {}


def _iter_ppf_pin_handles_for_obj(obj: bpy.types.Object) -> list[tuple[int, bpy.types.Object]]:
    out: list[tuple[int, bpy.types.Object]] = []
    for h in bpy.data.objects:
        if h is None or h.type != "EMPTY":
            continue
        if not bool(h.get(_PPF_PIN_HANDLE_TAG, 0)):
            continue
        if str(h.get(_PPF_PIN_HANDLE_OBJ, "")) != obj.name:
            continue
        try:
            vidx = int(h.get(_PPF_PIN_HANDLE_VIDX, -1))
        except Exception:
            continue
        if vidx < 0 or vidx >= len(obj.data.vertices):
            continue
        out.append((vidx, h))
    return out


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
        # list[(global_vidx, target_obj_name, (i0,i1,i2), (w0,w1,w2))]
        self.attach_bindings = []
        self.collider_object_names = []  # stable export order for collision mesh updates


def _barycentric_coords(
    p: mathutils.Vector,
    a: mathutils.Vector,
    b: mathutils.Vector,
    c: mathutils.Vector,
) -> tuple[float, float, float] | None:
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = float(v0.dot(v0))
    d01 = float(v0.dot(v1))
    d11 = float(v1.dot(v1))
    d20 = float(v2.dot(v0))
    d21 = float(v2.dot(v1))
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1.0e-20:
        return None
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return (float(u), float(v), float(w))


def _compute_attach_bindings(
    context, deformable_slices: list[tuple[str, int, int]]
) -> list[tuple[int, str, tuple[int, int, int], tuple[float, float, float]]]:
    depsgraph = context.evaluated_depsgraph_get()
    out: list[tuple[int, str, tuple[int, int, int], tuple[float, float, float]]] = []

    # Cache BVHs + evaluated meshes per target object.
    cache: dict[str, tuple[BVHTree, bpy.types.Object, bpy.types.Mesh]] = {}

    for name, start, count in deformable_slices:
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "MESH":
            continue

        oprops = getattr(obj, "andosim_artezbuild", None)
        if not (oprops and bool(getattr(oprops, "attach_enabled", False))):
            continue

        target_obj = getattr(oprops, "attach_target_object", None)
        if target_obj is None or getattr(target_obj, "type", None) != "MESH":
            continue

        vg_name = (getattr(oprops, "attach_vertex_group", "") or "").strip()
        if not vg_name or vg_name not in obj.vertex_groups:
            continue

        vg_index = obj.vertex_groups[vg_name].index

        entry = cache.get(target_obj.name)
        if entry is None:
            eval_obj = target_obj.evaluated_get(depsgraph)
            eval_mesh = None
            try:
                eval_mesh = eval_obj.to_mesh()
            except Exception:
                eval_mesh = None
            if eval_mesh is None:
                continue
            try:
                eval_mesh.calc_loop_triangles()
            except Exception:
                pass
            try:
                bvh = BVHTree.FromObject(target_obj, depsgraph)
            except Exception:
                try:
                    eval_obj.to_mesh_clear()
                except Exception:
                    pass
                continue
            cache[target_obj.name] = (bvh, eval_obj, eval_mesh)
            entry = cache[target_obj.name]

        bvh, eval_obj, eval_mesh = entry
        tgt_mw = target_obj.matrix_world
        mw = obj.matrix_world
        for v in obj.data.vertices:
            w = 0.0
            for g in v.groups:
                if g.group == vg_index:
                    w = float(g.weight)
                    break
            if w <= 0.0:
                continue

            if int(v.index) >= int(count):
                continue

            p = mw @ v.co
            hit = bvh.find_nearest(p)
            if hit is None:
                continue
            loc, _normal, tri_index, _dist = hit
            tri_index = int(tri_index)
            if tri_index < 0:
                continue
            try:
                tri = eval_mesh.loop_triangles[tri_index]
                i0, i1, i2 = (int(tri.vertices[0]), int(tri.vertices[1]), int(tri.vertices[2]))
                a = tgt_mw @ eval_mesh.vertices[i0].co
                b = tgt_mw @ eval_mesh.vertices[i1].co
                c = tgt_mw @ eval_mesh.vertices[i2].co
                bc = _barycentric_coords(loc, a, b, c)
                if bc is None:
                    continue
                out.append((int(start) + int(v.index), target_obj.name, (i0, i1, i2), (bc[0], bc[1], bc[2])))
            except Exception:
                continue

    # Cleanup evaluated meshes.
    for _name, (_bvh, eobj, _emesh) in cache.items():
        try:
            eobj.to_mesh_clear()
        except Exception:
            pass

    return out


def _sync_ppf_pin_targets_to_backend(context, state: _PPFState, objs: list[tuple[bpy.types.Object, int, int]]) -> bool:
    session = getattr(state, "session", None)
    if session is None:
        return False

    set_fn = getattr(session, "set_pin_targets", None)
    if set_fn is None:
        return False

    indices: list[int] = []
    positions_flat: list[float] = []

    # 1) User-created pin handles (absolute world positions)
    for obj, start, count in objs:
        for vidx, h in _iter_ppf_pin_handles_for_obj(obj):
            if vidx < 0 or vidx >= int(count):
                continue
            w = h.matrix_world.translation
            sx, sy, sz = ppf_export.blender_to_solver_xyz(float(w.x), float(w.y), float(w.z))
            j = int(start) + int(vidx)
            indices.append(j)
            positions_flat.extend([float(sx), float(sy), float(sz)])

    # 2) Attach bindings (target-local points transformed each frame)
    depsgraph = context.evaluated_depsgraph_get()
    mesh_cache: dict[str, tuple[bpy.types.Object, bpy.types.Mesh]] = {}
    try:
        for gidx, target_name, (i0, i1, i2), (w0, w1, w2) in getattr(state, "attach_bindings", []) or []:
            target_obj = bpy.data.objects.get(str(target_name))
            if target_obj is None or target_obj.type != "MESH":
                continue

            cached = mesh_cache.get(target_obj.name)
            if cached is None:
                eval_obj = target_obj.evaluated_get(depsgraph)
                eval_mesh = None
                try:
                    eval_mesh = eval_obj.to_mesh()
                except Exception:
                    eval_mesh = None
                if eval_mesh is None:
                    continue
                mesh_cache[target_obj.name] = (eval_obj, eval_mesh)
                cached = mesh_cache[target_obj.name]

            eval_obj, eval_mesh = cached
            if i0 < 0 or i1 < 0 or i2 < 0:
                continue
            if i0 >= len(eval_mesh.vertices) or i1 >= len(eval_mesh.vertices) or i2 >= len(eval_mesh.vertices):
                continue

            p_local = (
                (float(w0) * eval_mesh.vertices[int(i0)].co)
                + (float(w1) * eval_mesh.vertices[int(i1)].co)
                + (float(w2) * eval_mesh.vertices[int(i2)].co)
            )
            world = target_obj.matrix_world @ p_local
            sx, sy, sz = ppf_export.blender_to_solver_xyz(float(world.x), float(world.y), float(world.z))
            indices.append(int(gidx))
            positions_flat.extend([float(sx), float(sy), float(sz)])
    finally:
        for _name, (eobj, _emesh) in mesh_cache.items():
            try:
                eobj.to_mesh_clear()
            except Exception:
                pass

    try:
        if indices:
            set_fn(indices, positions_flat)
        else:
            clear_fn = getattr(session, "clear_pin_targets", None)
            if clear_fn is not None:
                clear_fn()
        return True
    except Exception:
        return False


def _sync_ppf_collision_mesh_to_backend(context, state: _PPFState) -> bool:
    """If the backend supports it, push animated/static-collider mesh vertices each tick.

    The solver scene is exported with static colliders concatenated into a single collision mesh.
    For animated colliders, we rebuild that concatenated vertex buffer in the exact same order
    (by `state.collider_object_names`) and send it to the backend.
    """

    session = getattr(state, "session", None)
    if session is None:
        return False

    set_fn = getattr(session, "set_collision_mesh_vertices", None)
    if set_fn is None:
        return False

    names = list(getattr(state, "collider_object_names", []) or [])
    if not names:
        return False

    depsgraph = context.evaluated_depsgraph_get()
    verts_flat: list[float] = []

    for name in names:
        obj = bpy.data.objects.get(str(name))
        if obj is None or obj.type != "MESH":
            # If the collider went missing, don't send a mismatched buffer.
            return False
        eval_obj = obj.evaluated_get(depsgraph)
        eval_mesh = None
        try:
            eval_mesh = eval_obj.to_mesh()
        except Exception:
            eval_mesh = None
        if eval_mesh is None:
            return False

        try:
            mw = eval_obj.matrix_world
            for v in eval_mesh.vertices:
                w = mw @ v.co
                sx, sy, sz = ppf_export.blender_to_solver_xyz(float(w.x), float(w.y), float(w.z))
                verts_flat.extend([float(sx), float(sy), float(sz)])
        finally:
            try:
                eval_obj.to_mesh_clear()
            except Exception:
                pass

    if not verts_flat:
        return False

    try:
        set_fn(verts_flat)
        return True
    except Exception:
        return False


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

                    state.collider_object_names = list(getattr(export, "collider_object_names", []) or [])

                for w in getattr(export, "warnings", []) or []:
                    self.report({"WARNING"}, str(w))

                scene_path = export.scene_path
                settings.scene_path = scene_path
                state.deformable_slices = list(export.deformable_slices)
                state.collider_object_names = list(getattr(export, "collider_object_names", []) or [])

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

            # Precompute attach bindings once at start (stable attachment).
            try:
                state.attach_bindings = _compute_attach_bindings(context, list(state.deformable_slices))
            except Exception:
                state.attach_bindings = []
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

        # Push handle + attach pin targets (if any) into the backend.
        _sync_ppf_pin_targets_to_backend(context, state, objs)
        _sync_ppf_collision_mesh_to_backend(context, state)

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

            # Ensure we clear targets when nothing drives pins.
            _sync_ppf_pin_targets_to_backend(context, state, objs)
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


def _ensure_pin_vertex_group(obj: bpy.types.Object, group_name: str) -> bpy.types.VertexGroup:
    vg = obj.vertex_groups.get(group_name)
    if vg is None:
        vg = obj.vertex_groups.new(name=group_name)
    return vg


class ANDOSIM_ARTEZBUILD_OT_ppf_create_pin_handle(bpy.types.Operator):
    """Create a draggable Empty handle for a selected vertex (Edit Mode).

    Workflow:
    - In Edit Mode: select a vertex and run this operator. It creates an Empty parented to that vertex.
    - In Object Mode while sim runs: move the Empty; the backend uses its world position as pin target.
    """

    bl_idname = "andosim_artezbuild.ppf_create_pin_handle"
    bl_label = "Create Pin Handle (PPF)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, "Select a mesh object")
            return {"CANCELLED"}

        props = getattr(obj, "andosim_artezbuild", None)
        if not (props and bool(getattr(props, "enabled", False)) and getattr(props, "role", "IGNORE") == "DEFORMABLE"):
            self.report({"ERROR"}, "Active object must be an enabled DEFORMABLE")
            return {"CANCELLED"}

        if obj.mode != "EDIT":
            self.report({"ERROR"}, "Switch to Edit Mode, select a vertex, then run")
            return {"CANCELLED"}

        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()

        v = None
        active = getattr(bm.select_history, "active", None)
        if isinstance(active, bmesh.types.BMVert) and active.select:
            v = active
        else:
            for vv in bm.verts:
                if vv.select:
                    v = vv
                    break

        if v is None:
            self.report({"ERROR"}, "Select a vertex")
            return {"CANCELLED"}

        vidx = int(v.index)
        world = obj.matrix_world @ v.co

        # Ensure pin vertex group has this vertex.
        # Blender does not reliably allow assigning vertex groups via bmesh deform layer across all edit-mesh operations
        # (BMVerts can be invalidated during group creation/updates). The robust approach is:
        # - collect vertex index in Edit Mode
        # - temporarily switch to Object Mode
        # - use VertexGroup.add()
        # - switch back to Edit Mode
        vg_name = (getattr(props, "pin_vertex_group", "") or "").strip() or "PPF_PIN"
        prev_mode = obj.mode
        prev_active = context.view_layer.objects.active
        try:
            context.view_layer.objects.active = obj
            if prev_mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")

            vg = _ensure_pin_vertex_group(obj, vg_name)
            if vidx < 0 or vidx >= len(obj.data.vertices):
                raise RuntimeError(f"Selected vertex index out of range: {vidx}")
            vg.add([vidx], 1.0, "REPLACE")
        except Exception as exc:
            self.report({"ERROR"}, f"Failed to assign pin vertex group: {exc}")
            return {"CANCELLED"}

        props.pin_enabled = True
        props.pin_vertex_group = vg.name
        if float(getattr(props, "pin_pull_strength", 0.0)) <= 0.0:
            props.pin_pull_strength = 1.0

        handle_name = f"{obj.name}_PPF_PIN_{vidx}"
        handle = bpy.data.objects.get(handle_name)
        if handle is None:
            handle = bpy.data.objects.new(handle_name, None)
            context.collection.objects.link(handle)

        handle.empty_display_type = "SPHERE"
        handle.empty_display_size = 0.06
        handle[_PPF_PIN_HANDLE_TAG] = 1
        handle[_PPF_PIN_HANDLE_OBJ] = obj.name
        handle[_PPF_PIN_HANDLE_VIDX] = vidx

        # Keep it static in viewport (do NOT parent to the cloth vertex).
        # It will only move if the user moves it.
        handle.parent = None

        # Place it at the vertex in world space.
        handle.matrix_world.translation = world

        try:
            if prev_mode == "EDIT":
                bpy.ops.object.mode_set(mode="EDIT")
        except Exception:
            pass
        try:
            context.view_layer.objects.active = prev_active
        except Exception:
            pass

        self.report({"INFO"}, f"Created pin handle for vertex {vidx}")
        return {"FINISHED"}


def _pick_active_vertex(context, event, obj: bpy.types.Object) -> tuple[int, mathutils.Vector] | None:
    region = getattr(context, "region", None)
    rv3d = getattr(context, "region_data", None)
    if region is None or rv3d is None:
        return None

    coord = (event.mouse_region_x, event.mouse_region_y)
    view_vec = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
    ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)

    depsgraph = context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mw = obj.matrix_world
    mw_inv = mw.inverted_safe()

    local_origin = mw_inv @ ray_origin
    local_dir = (mw_inv.to_3x3() @ view_vec).normalized()

    eval_mesh = None
    try:
        eval_mesh = eval_obj.to_mesh()
        hit, location, _normal, face_index = eval_obj.ray_cast(local_origin, local_dir)
        if not hit or face_index < 0:
            return None
        poly = eval_mesh.polygons[face_index]
        best_vidx = -1
        best_d2 = 1.0e30
        for vidx in poly.vertices:
            vco = eval_mesh.vertices[vidx].co
            d2 = (vco - location).length_squared
            if d2 < best_d2:
                best_d2 = d2
                best_vidx = int(vidx)
        if best_vidx < 0:
            return None
        world = mw @ eval_mesh.vertices[best_vidx].co
        return best_vidx, world
    finally:
        if eval_mesh is not None:
            try:
                eval_obj.to_mesh_clear()
            except Exception:
                pass


class ANDOSIM_ARTEZBUILD_OT_ppf_grab_pin(bpy.types.Operator):
    """Grab a cloth vertex and drag it in the viewport (PPF)"""

    bl_idname = "andosim_artezbuild.ppf_grab_pin"
    bl_label = "Grab Pin (PPF)"
    bl_options = {"REGISTER"}

    _obj_name: str | None = None
    _vidx: int = -1
    _handle_name: str | None = None
    _depth_ref: mathutils.Vector | None = None
    _phase: str = "PICK"  # PICK -> DRAG

    def invoke(self, context, event):
        if getattr(context, "space_data", None) is None or context.space_data.type != "VIEW_3D":
            self.report({"ERROR"}, "Run this from a 3D Viewport")
            return {"CANCELLED"}

        obj = context.active_object
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, "Select a deformable mesh object")
            return {"CANCELLED"}
        if obj.mode != "OBJECT":
            self.report({"ERROR"}, "Object must be in Object Mode")
            return {"CANCELLED"}

        props = getattr(obj, "andosim_artezbuild", None)
        if not (props and bool(getattr(props, "enabled", False)) and getattr(props, "role", "IGNORE") == "DEFORMABLE"):
            self.report({"ERROR"}, "Active object must be an enabled DEFORMABLE")
            return {"CANCELLED"}

        self._obj_name = obj.name
        self._vidx = -1
        self._handle_name = None
        self._depth_ref = None
        self._phase = "PICK"

        # If invoked from the sidebar button, the event comes from the UI region (no window-region mouse coords).
        # Arm the tool and let the user click a vertex in the viewport window region.
        pick = _pick_active_vertex(context, event, obj)
        if pick is not None:
            vidx, world = pick
            self._vidx = int(vidx)
            self._depth_ref = mathutils.Vector((float(world.x), float(world.y), float(world.z)))
            self._phase = "DRAG"
        else:
            self.report({"INFO"}, "Click a vertex in the viewport to grab")

        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if self._obj_name is None:
            return {"CANCELLED"}

        obj = bpy.data.objects.get(self._obj_name)
        if obj is None:
            _ACTIVE_PPF_GRABS.pop(self._obj_name, None)
            return {"CANCELLED"}

        if event.type in {
            "MIDDLEMOUSE",
            "WHEELUPMOUSE",
            "WHEELDOWNMOUSE",
            "NUMPAD_0",
            "NUMPAD_PERIOD",
            "NUMPAD_1",
            "NUMPAD_2",
            "NUMPAD_3",
            "NUMPAD_4",
            "NUMPAD_5",
            "NUMPAD_6",
            "NUMPAD_7",
            "NUMPAD_8",
            "NUMPAD_9",
        }:
            return {"PASS_THROUGH"}

        if event.type in {"RIGHTMOUSE", "ESC"}:
            _ACTIVE_PPF_GRABS.pop(self._obj_name, None)
            context.window.cursor_set("DEFAULT")
            return {"CANCELLED"}

        if self._vidx < 0 or self._phase == "PICK":
            if event.type == "LEFTMOUSE" and event.value == "PRESS":
                pick = _pick_active_vertex(context, event, obj)
                if pick is None:
                    self.report({"WARNING"}, "No vertex under cursor")
                    return {"RUNNING_MODAL"}

                vidx, world = pick
                self._vidx = int(vidx)
                self._depth_ref = mathutils.Vector((float(world.x), float(world.y), float(world.z)))
                self._phase = "DRAG"
            else:
                return {"RUNNING_MODAL"}

        # Ensure we have a handle + active grab state once we have a picked vertex.
        if self._handle_name is None and self._depth_ref is not None and self._vidx >= 0:
            props = getattr(obj, "andosim_artezbuild", None)
            if not (props and bool(getattr(props, "enabled", False)) and getattr(props, "role", "IGNORE") == "DEFORMABLE"):
                _ACTIVE_PPF_GRABS.pop(self._obj_name, None)
                context.window.cursor_set("DEFAULT")
                return {"CANCELLED"}

            handle_name = f"{obj.name}_PPF_GRAB"
            handle = bpy.data.objects.get(handle_name)
            if handle is None:
                handle = bpy.data.objects.new(handle_name, None)
                context.collection.objects.link(handle)
            handle.empty_display_type = "SPHERE"
            handle.empty_display_size = 0.06
            handle.location = self._depth_ref
            self._handle_name = handle.name

            vg_name = (getattr(props, "pin_vertex_group", "") or "").strip() or "PPF_PIN"
            vg = _ensure_pin_vertex_group(obj, vg_name)
            vg.add([int(self._vidx)], 1.0, "REPLACE")
            props.pin_enabled = True
            props.pin_vertex_group = vg.name
            if float(getattr(props, "pin_pull_strength", 0.0)) <= 0.0:
                props.pin_pull_strength = 1.0

            _ACTIVE_PPF_GRABS[obj.name] = {
                "vidx": int(self._vidx),
                "target_world": (float(handle.location.x), float(handle.location.y), float(handle.location.z)),
                "handle": handle.name,
            }

        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            _ACTIVE_PPF_GRABS.pop(self._obj_name, None)
            context.window.cursor_set("DEFAULT")
            return {"FINISHED"}

        if event.type == "MOUSEMOVE":
            region = getattr(context, "region", None)
            rv3d = getattr(context, "region_data", None)
            if region is None or rv3d is None or self._depth_ref is None:
                return {"RUNNING_MODAL"}

            coord = (event.mouse_region_x, event.mouse_region_y)
            world = view3d_utils.region_2d_to_location_3d(region, rv3d, coord, self._depth_ref)
            self._depth_ref = world
            if self._handle_name:
                h = bpy.data.objects.get(self._handle_name)
                if h is not None:
                    h.location = world

            _ACTIVE_PPF_GRABS[self._obj_name] = {
                "vidx": int(self._vidx),
                "target_world": (float(world.x), float(world.y), float(world.z)),
                "handle": str(self._handle_name or ""),
            }

            context.window.cursor_set("HAND")
            return {"RUNNING_MODAL"}

        return {"RUNNING_MODAL"}


class ANDOSIM_ARTEZBUILD_OT_ppf_clear_grab(bpy.types.Operator):
    """Clear the active PPF grab handle for the active object"""

    bl_idname = "andosim_artezbuild.ppf_clear_grab"
    bl_label = "Clear Grab (PPF)"
    bl_options = {"REGISTER"}

    def execute(self, context):
        obj = context.active_object
        if obj is None:
            self.report({"WARNING"}, "No active object")
            return {"CANCELLED"}

        _ACTIVE_PPF_GRABS.pop(obj.name, None)

        # Remove all pin handles for this object.
        for _vidx, h in _iter_ppf_pin_handles_for_obj(obj):
            try:
                bpy.data.objects.remove(h, do_unlink=True)
            except Exception:
                pass

        state = getattr(ANDOSIM_ARTEZBUILD_OT_ppf_run, "_ppf", None)
        if state is not None and getattr(state, "session", None) is not None:
            try:
                clear_fn = getattr(state.session, "clear_pin_targets", None)
                if clear_fn is not None:
                    clear_fn()
            except Exception:
                pass

        handle_name = f"{obj.name}_PPF_GRAB"
        handle = bpy.data.objects.get(handle_name)
        if handle is not None:
            try:
                bpy.data.objects.remove(handle, do_unlink=True)
            except Exception:
                pass

        return {"FINISHED"}


class _PC2Writer:
    def __init__(
        self,
        path: Path,
        *,
        verts_tot: int,
        frame_start: int,
        frame_tot: int,
        fps: float,
    ):
        self.path = path
        self.verts_tot = int(verts_tot)
        self.frame_tot = int(frame_tot)
        self._frames_written = 0

        start_sec = float(frame_start) / float(fps)
        sampling_sec = 1.0 / float(fps)

        path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(path, "wb")
        try:
            header = struct.pack(
                "<12siiffi",
                b"POINTCACHE2\0",  # 12 bytes
                1,  # file_version
                self.verts_tot,
                start_sec,
                sampling_sec,
                self.frame_tot,
            )
            self._fp.write(header)
        except Exception:
            try:
                self._fp.close()
            except Exception:
                pass
            raise

    def write_frame(self, coords_local: list[float]) -> None:
        if len(coords_local) != self.verts_tot * 3:
            raise ValueError(
                f"PC2 frame for {self.path.name}: got {len(coords_local)} floats, expected {self.verts_tot * 3}"
            )
        if self._frames_written >= self.frame_tot:
            raise RuntimeError(f"PC2 writer overflow for {self.path}")

        array("f", coords_local).tofile(self._fp)
        self._frames_written += 1

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass


class ANDOSIM_ARTEZBUILD_OT_ppf_bake_cache(bpy.types.Operator):
    bl_idname = "andosim_artezbuild.ppf_bake_cache"
    bl_label = "Bake PPF Cache (PC2)"
    bl_options = {"REGISTER"}

    def execute(self, context):
        backend_or_exc = _try_import_backend()
        if isinstance(backend_or_exc, Exception):
            self.report({"ERROR"}, f"Failed to import ppf_cts_backend: {backend_or_exc}")
            return {"CANCELLED"}

        settings = _get_settings(context)
        if settings is None:
            self.report({"ERROR"}, "Addon settings not initialized")
            return {"CANCELLED"}

        scene = context.scene
        frame_start = int(getattr(scene, "frame_start", 1))
        frame_end = int(getattr(scene, "frame_end", frame_start))
        if frame_end < frame_start:
            self.report({"ERROR"}, "Invalid frame range")
            return {"CANCELLED"}

        render = getattr(scene, "render", None)
        fps = 24.0
        if render is not None:
            fps_base = float(getattr(render, "fps_base", 1.0) or 1.0)
            fps = float(getattr(render, "fps", 24.0)) / fps_base
        if fps <= 0:
            fps = 24.0

        output_dir = (settings.output_dir or "").strip()
        if not output_dir:
            self.report({"ERROR"}, "Set Output Dir before baking")
            return {"CANCELLED"}

        bake_dir = Path(output_dir) / "cache"
        bake_dir.mkdir(parents=True, exist_ok=True)

        # Enforce one solver step per frame so the cache syncs to Blender's timeline.
        old_dt = float(getattr(settings, "dt", 1e-3))
        target_dt = 1.0 / float(fps)
        settings.dt = target_dt

        sess = None
        writers: list[tuple[bpy.types.Object, int, int, _PC2Writer, Path]] = []

        try:
            try:
                scene.frame_set(frame_start)
            except Exception:
                pass

            export = ppf_export.export_ppf_scene_from_roles(context, settings)
            for w in getattr(export, "warnings", []) or []:
                self.report({"WARNING"}, str(w))

            slices = list(getattr(export, "deformable_slices", []) or [])
            if not slices:
                self.report({"ERROR"}, "No deformables enabled for PPF (tag objects as Deformable)")
                return {"CANCELLED"}

            frame_tot = int(frame_end - frame_start + 1)

            for name, start, count in slices:
                obj = bpy.data.objects.get(name)
                if obj is None or obj.type != "MESH":
                    self.report({"ERROR"}, f"Deformable '{name}' missing or not a mesh")
                    return {"CANCELLED"}
                if obj.mode != "OBJECT":
                    self.report({"ERROR"}, f"Deformable '{name}' must be in Object Mode")
                    return {"CANCELLED"}
                if len(obj.data.vertices) != int(count):
                    self.report({"ERROR"}, f"Vertex count changed on '{name}'; apply topology modifiers")
                    return {"CANCELLED"}

                pc2_path = bake_dir / f"{name}.pc2"
                writer = _PC2Writer(
                    pc2_path,
                    verts_tot=int(count),
                    frame_start=frame_start,
                    frame_tot=frame_tot,
                    fps=float(fps),
                )
                writers.append((obj, int(start), int(count), writer, pc2_path))

            total_verts = 0
            for _obj, start, count, _writer, _pc2_path in writers:
                total_verts = max(total_verts, int(start) + int(count))

            curr = [0.0] * (total_verts * 3)
            for obj, start, _count, _writer, _pc2_path in writers:
                mw = obj.matrix_world
                mesh = obj.data
                for i, v in enumerate(mesh.vertices):
                    wco = mw @ v.co
                    sx, sy, sz = ppf_export.blender_to_solver_xyz(float(wco.x), float(wco.y), float(wco.z))
                    j = start + i
                    curr[3 * j + 0] = sx
                    curr[3 * j + 1] = sy
                    curr[3 * j + 2] = sz

            # Sample 0: current local coords at frame_start.
            for obj, _start, count, writer, _pc2_path in writers:
                local = [0.0] * (int(count) * 3)
                for i, v in enumerate(obj.data.vertices):
                    local[3 * i + 0] = float(v.co.x)
                    local[3 * i + 1] = float(v.co.y)
                    local[3 * i + 2] = float(v.co.z)
                writer.write_frame(local)

            sess = backend_or_exc.Session(export.scene_path, output_dir)

            wm = context.window_manager
            try:
                wm.progress_begin(0, frame_tot)
            except Exception:
                wm = None

            for fi, frame in enumerate(range(frame_start + 1, frame_end + 1), start=1):
                try:
                    scene.frame_set(frame)
                except Exception:
                    pass

                out = sess.step(curr)
                if len(out) != len(curr):
                    raise RuntimeError(f"Backend returned {len(out)} floats, expected {len(curr)}")
                curr = list(out)

                for obj, start, count, writer, _pc2_path in writers:
                    mw_inv = obj.matrix_world.inverted_safe()
                    local = [0.0] * (int(count) * 3)
                    for i in range(int(count)):
                        j = int(start) + i
                        wx, wy, wz = ppf_export.solver_to_blender_xyz(
                            float(curr[3 * j + 0]),
                            float(curr[3 * j + 1]),
                            float(curr[3 * j + 2]),
                        )
                        lco = mw_inv @ mathutils.Vector((wx, wy, wz))
                        local[3 * i + 0] = float(lco.x)
                        local[3 * i + 1] = float(lco.y)
                        local[3 * i + 2] = float(lco.z)
                    writer.write_frame(local)

                if wm is not None:
                    try:
                        wm.progress_update(fi)
                    except Exception:
                        pass

            if wm is not None:
                try:
                    wm.progress_end()
                except Exception:
                    pass

            for _obj, _start, _count, writer, _pc2_path in writers:
                writer.close()

            for obj, _start, _count, _writer, pc2_path in writers:
                mod = obj.modifiers.new(name="PPF_Cache", type="MESH_CACHE")
                mod.cache_format = "PC2"
                mod.filepath = str(pc2_path)
                mod.time_mode = "FRAME"
                mod.play_mode = "SCENE"
                mod.deform_mode = "OVERWRITE"
                mod.frame_start = frame_start
                mod.frame_scale = 1.0

            self.report({"INFO"}, f"Baked PC2 cache to {bake_dir}")
            return {"FINISHED"}

        except Exception as exc:
            self.report({"ERROR"}, f"Bake failed: {exc}")
            return {"CANCELLED"}

        finally:
            try:
                settings.dt = old_dt
            except Exception:
                pass

            if sess is not None:
                try:
                    sess.close()
                except Exception:
                    pass

            for _obj, _start, _count, writer, _pc2_path in writers:
                try:
                    writer.close()
                except Exception:
                    pass


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


def _ppf_parse_last_sample(path: Path) -> tuple[float | None, str | None]:
    """Parse a trace stream with lines like: `<time> <value>`.

    Many streams append a few summary rows with time=0; we prefer the last row with time>0.
    Returns (time, value_str) or (None, None) if unreadable.
    """

    try:
        text = path.read_text(errors="replace")
    except Exception:
        return None, None

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None, None

    def _try_parse(line: str) -> tuple[float | None, str | None]:
        parts = line.split()
        if len(parts) < 2:
            return None, None
        try:
            t = float(parts[0])
        except Exception:
            return None, None
        return t, parts[1]

    for ln in reversed(lines):
        t, v = _try_parse(ln)
        if t is None or v is None:
            continue
        if t > 0.0:
            return t, v

    t, v = _try_parse(lines[-1])
    return t, v


def _ppf_trace_flag_to_hint(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    v = value.strip().lower()
    if v in {"0", "0.0", "false"}:
        if "linsolve" in name:
            return "linsolve=FAIL"
        if "intersection" in name:
            return "intersection=FAIL"
        return f"{name}=FAIL"
    if v in {"1", "1.0", "true"}:
        if "linsolve" in name:
            return "linsolve=OK"
        if "intersection" in name:
            return "intersection=OK"
        return f"{name}=OK"
    return f"{name}={value}"


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

    # Try to produce a short, actionable single-line summary.
    summary_bits: list[str] = []
    last_t: float | None = None
    for path, key in [
        (data_dir / "advance.dt.out", "dt"),
        (data_dir / "advance.final_dt.out", "final_dt"),
        (data_dir / "advance.num_contact.out", "contacts"),
        (data_dir / "advance.linsolve.out", "linsolve"),
        (data_dir / "advance.check_intersection.out", "intersection"),
        (data_dir / "advance.out", "advance"),
    ]:
        if not path.exists():
            continue
        t, v = _ppf_parse_last_sample(path)
        if t is not None and t > 0.0:
            last_t = t if last_t is None else max(last_t, t)
        if v is None:
            continue
        if key in {"linsolve", "intersection"}:
            hint = _ppf_trace_flag_to_hint(key, v)
            if hint:
                summary_bits.append(hint)
        else:
            summary_bits.append(f"{key}={v}")

    if last_t is not None:
        summary_bits.insert(0, f"t={last_t:.6f}")

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
    if summary_bits:
        short = " ".join(summary_bits[:8])
        return f"({short}; traces in {data_dir})"
    return f"(traces in {data_dir})"
