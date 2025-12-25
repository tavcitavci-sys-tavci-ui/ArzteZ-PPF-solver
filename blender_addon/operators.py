import bpy
import os
import tempfile
import mathutils
import struct
import bmesh
import math
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

_PPF_BACKEND_VERSION: str = "?"
_PPF_BACKEND_FILE: str = "?"
_PPF_BACKEND_VERSION_PRINTED: bool = False


# In-memory mesh snapshot used by "Reset Simulation".
# Keyed by mesh datablock pointer so renames are less fragile.
_PPF_MESH_SNAPSHOT: dict[int, dict] = {}


def _get_addon_prefs(context):
    try:
        addon = context.preferences.addons.get(__package__)
    except Exception:
        return None
    if not addon:
        return None
    return getattr(addon, "preferences", None)


def _vertex_group_has_positive_weight(obj: bpy.types.Object, vg_name: str, *, eps: float = 0.0) -> bool:
    if obj is None or obj.type != "MESH":
        return False
    vg_name = (vg_name or "").strip()
    if not vg_name:
        return False
    if vg_name not in obj.vertex_groups:
        return False
    try:
        vg_index = int(obj.vertex_groups[vg_name].index)
    except Exception:
        return False

    mesh = getattr(obj, "data", None)
    if mesh is None:
        return False
    try:
        verts = mesh.vertices
    except Exception:
        return False

    for v in verts:
        try:
            for g in v.groups:
                if int(g.group) == vg_index and float(g.weight) > float(eps):
                    return True
        except Exception:
            continue
    return False


def _ppf_object_has_constraints(obj: bpy.types.Object) -> tuple[bool, str]:
    """Return (has_constraints, reason).

    We treat *authored* constraints as blocking for remeshing because remesh invalidates
    vertex indices/weights and pin-handle bindings.
    """

    if obj is None or obj.type != "MESH":
        return False, ""

    props = getattr(obj, "andosim_artezbuild", None)
    if props is None:
        return False, ""

    try:
        if _iter_ppf_pin_handles_for_obj(obj):
            return True, "PPF pin handles exist (clear handles before remeshing)"
    except Exception:
        pass

    pin_vg = str(getattr(props, "pin_vertex_group", "") or "")
    if _vertex_group_has_positive_weight(obj, pin_vg, eps=0.0):
        return True, f"Vertex group '{pin_vg}' contains pin weights (remove pins before remeshing)"

    attach_vg = str(getattr(props, "attach_vertex_group", "") or "")
    if bool(getattr(props, "attach_enabled", False)):
        return True, "Attach to Mesh is enabled (disable/remove attach before remeshing)"
    if getattr(props, "attach_target_object", None) is not None:
        return True, "Attach target is set (clear it before remeshing)"
    if _vertex_group_has_positive_weight(obj, attach_vg, eps=0.0):
        return True, f"Vertex group '{attach_vg}' contains attach weights (remove attach before remeshing)"

    stitch_src_vg = str(getattr(props, "stitch_source_vertex_group", "") or "")
    if bool(getattr(props, "stitch_enabled", False)):
        return True, "Stitches are enabled (disable/remove stitches before remeshing)"
    if getattr(props, "stitch_target_object", None) is not None:
        return True, "Stitch target is set (clear it before remeshing)"
    if _vertex_group_has_positive_weight(obj, stitch_src_vg, eps=0.0):
        return True, f"Vertex group '{stitch_src_vg}' contains stitch weights (remove stitches before remeshing)"

    return False, ""


def _bmesh_cleanup_and_triangulate(mesh: bpy.types.Mesh, *, merge_distance: float) -> None:
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        dist = float(max(0.0, merge_distance))
        if dist > 0.0:
            try:
                bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=dist)
            except Exception:
                pass

        try:
            bmesh.ops.dissolve_degenerate(bm, dist=max(dist, 1.0e-12))
        except Exception:
            pass

        # Drop loose verts (not part of any face) to satisfy PPF export validation.
        try:
            loose = [v for v in bm.verts if not v.link_faces]
            if loose:
                bmesh.ops.delete(bm, geom=loose, context="VERTS")
        except Exception:
            pass

        bm.faces.ensure_lookup_table()
        if bm.faces:
            try:
                bmesh.ops.triangulate(
                    bm,
                    faces=bm.faces,
                    quad_method="BEAUTY",
                    ngon_method="BEAUTY",
                )
            except Exception:
                pass
            try:
                bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            except Exception:
                pass

        bm.to_mesh(mesh)
    finally:
        bm.free()
    try:
        mesh.update()
    except Exception:
        pass


def _auto_voxel_size_for_obj(obj: bpy.types.Object) -> float:
    # Use world-space AABB diagonal as a scale hint.
    try:
        dims = obj.dimensions
        dx, dy, dz = float(dims.x), float(dims.y), float(dims.z)
        diag = math.sqrt(dx * dx + dy * dy + dz * dz)
    except Exception:
        diag = 1.0

    # Heuristic: about 1% of diagonal, with sensible clamps.
    voxel = diag / 100.0 if diag > 0.0 else 0.01
    voxel = max(1.0e-4, voxel)
    voxel = min(voxel, max(1.0e-3, diag / 2.0) if diag > 0.0 else voxel)
    return float(voxel)


def _effective_ppf_shell_settings(context, obj: bpy.types.Object):
    """Return an object with shell fields (tri_model, tri_young_mod, tri_bend, ...).

    Uses per-object overrides when enabled; otherwise falls back to the scene settings.
    """

    scene_settings = getattr(getattr(context, "scene", None), "andosim_artezbuild", None)
    if obj is None or obj.type != "MESH":
        return scene_settings

    oprops = getattr(obj, "andosim_artezbuild", None)
    if oprops is None:
        return scene_settings

    if bool(getattr(oprops, "use_object_params", False)):
        return oprops
    return scene_settings


def _voxel_scale_for_shell_model(tri_model: str) -> float:
    """Heuristic voxel-size scale for different shell models.

    This intentionally stays conservative: all models want good triangles.
    The scale only nudges default resolution so users get a more consistent
    first run without needing to know what voxel size means.
    """

    return _voxel_scale_for_shell_params(tri_model=tri_model)


def _voxel_scale_for_shell_params(
    tri_model: str,
    tri_young_mod: float | None = None,
    tri_bend: float | None = None,
    tri_strain_limit: float | None = None,
) -> float:
    """Heuristic voxel-size scale for different shell settings.

    Goal: produce triangle quality/resolution that tends to be stable across
    different shell models and stiffness ranges.

    Notes:
    - This is intentionally a mild effect. It's a nudge to the auto voxel size,
      not a replacement for a user-chosen voxel size.
    - We treat model and stiffness as *hints* only; units/normalization may vary
      across pipelines.
    """

    key = (tri_model or "").strip().lower()

    # Base model scale: slightly finer for nonlinear shells.
    # (Values chosen to be conservative; users can always override voxel size.)
    model_scale = 1.0
    if key in {"stvk", "snhk"}:
        model_scale = 0.85
    elif key in {"arap"}:
        model_scale = 1.05

    # Stiffness nudges: stiffer materials benefit from finer resolution.
    # Use a very gentle power law to avoid extreme scaling.
    young_scale = 1.0
    try:
        young = float(tri_young_mod) if tri_young_mod is not None else None
    except Exception:
        young = None
    if young is not None and young > 0.0:
        # Reference picked to keep common defaults near ~1.0.
        ref = 1.0e5
        ratio = max(1.0e-12, young / ref)
        # Larger young -> smaller voxel (finer mesh)
        young_scale = float(ratio ** (-0.08))
        young_scale = min(max(young_scale, 0.6), 1.4)

    bend_scale = 1.0
    try:
        bend = float(tri_bend) if tri_bend is not None else None
    except Exception:
        bend = None
    if bend is not None and bend > 0.0:
        ref = 2.0
        ratio = max(1.0e-12, bend / ref)
        bend_scale = float(ratio ** (-0.10))
        bend_scale = min(max(bend_scale, 0.7), 1.3)

    # Optional strain-limit hint: if enabled and very small, nudge finer.
    strain_scale = 1.0
    try:
        s = float(tri_strain_limit) if tri_strain_limit is not None else None
    except Exception:
        s = None
    if s is not None and s > 0.0:
        # Interpret as a fraction when it looks like one.
        if s < 0.25:
            # Smaller strain limit -> finer.
            strain_scale = min(max((s / 0.10) ** (0.10), 0.75), 1.15)

    scale = float(model_scale * young_scale * bend_scale * strain_scale)
    # Clamp overall effect.
    return float(min(max(scale, 0.5), 1.5))


def _auto_merge_distance_for_voxel(user_merge_distance: float, used_voxel: float) -> float:
    """Choose a scale-aware merge distance.

    If the user kept the default, adapt it to the chosen voxel size to avoid
    over-merging tiny meshes (or under-merging huge ones).
    """

    md = float(max(0.0, user_merge_distance))
    voxel = float(max(0.0, used_voxel))
    # Default from operator property.
    if abs(md - 1.0e-6) <= 1.0e-12 and voxel > 0.0:
        return float(max(1.0e-12, 1.0e-4 * voxel))
    return md


def _percentile_sorted(values_sorted: list[float], q: float) -> float:
    if not values_sorted:
        return 0.0
    q = float(q)
    if q <= 0.0:
        return float(values_sorted[0])
    if q >= 1.0:
        return float(values_sorted[-1])
    i = q * (len(values_sorted) - 1)
    lo = int(math.floor(i))
    hi = int(math.ceil(i))
    if lo == hi:
        return float(values_sorted[lo])
    t = i - lo
    return float(values_sorted[lo] * (1.0 - t) + values_sorted[hi] * t)


def _mesh_quality_report(obj: bpy.types.Object) -> dict:
    """Compute basic triangle-mesh quality stats in WORLD space.

    Returns a dict with:
      - edge_len_sorted
      - aspect_sorted
      - min_angle_sorted
      - degenerate_faces
      - nonmanifold_edges
      - boundary_edges
    """

    mesh = getattr(obj, "data", None)
    if obj is None or obj.type != "MESH" or mesh is None:
        return {
            "edge_len_sorted": [],
            "aspect_sorted": [],
            "min_angle_sorted": [],
            "degenerate_faces": 0,
            "nonmanifold_edges": 0,
            "boundary_edges": 0,
        }

    M = obj.matrix_world.copy()
    eps = 1.0e-12

    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        # Edge lengths
        edge_lens: list[float] = []
        nonmanifold_edges = 0
        boundary_edges = 0
        for e in bm.edges:
            if not e.is_valid:
                continue
            lf = len(e.link_faces)
            if lf > 2:
                nonmanifold_edges += 1
            elif lf == 1:
                boundary_edges += 1
            try:
                v0, v1 = e.verts
                p0 = M @ v0.co
                p1 = M @ v1.co
                edge_lens.append(float((p1 - p0).length))
            except Exception:
                continue

        # Face quality (expect triangles after cleanup)
        aspects: list[float] = []
        min_angles_deg: list[float] = []
        degenerate_faces = 0

        for f in bm.faces:
            if not f.is_valid:
                continue
            if len(f.verts) != 3:
                continue
            try:
                p = [M @ v.co for v in f.verts]
                e0 = p[1] - p[0]
                e1 = p[2] - p[1]
                e2 = p[0] - p[2]
                l0 = float(e0.length)
                l1 = float(e1.length)
                l2 = float(e2.length)
                lmin = min(l0, l1, l2)
                lmax = max(l0, l1, l2)
                area2 = float(e0.cross(p[2] - p[0]).length)  # 2*area
                if lmin <= eps or area2 <= eps:
                    degenerate_faces += 1
                    continue
                aspects.append(float(lmax / lmin))

                # Angles
                # angle at p0 between (p1-p0) and (p2-p0)
                a0 = (p[1] - p[0]).angle(p[2] - p[0])
                a1 = (p[2] - p[1]).angle(p[0] - p[1])
                a2 = (p[0] - p[2]).angle(p[1] - p[2])
                min_angles_deg.append(float(min(a0, a1, a2) * 180.0 / math.pi))
            except Exception:
                continue

        edge_lens.sort()
        aspects.sort()
        min_angles_deg.sort()

        return {
            "edge_len_sorted": edge_lens,
            "aspect_sorted": aspects,
            "min_angle_sorted": min_angles_deg,
            "degenerate_faces": int(degenerate_faces),
            "nonmanifold_edges": int(nonmanifold_edges),
            "boundary_edges": int(boundary_edges),
        }
    finally:
        bm.free()


class ANDOSIM_ARTEZBUILD_OT_prepare_cloth_mesh(bpy.types.Operator):
    bl_idname = "andosim_artezbuild.prepare_cloth_mesh"
    bl_label = "Prepare Cloth Mesh"
    bl_options = {"REGISTER", "UNDO"}

    voxel_size: bpy.props.FloatProperty(
        name="Voxel Size",
        description="Remesh voxel size (0 = auto based on object size)",
        default=0.0,
        min=0.0,
        soft_min=0.0,
        soft_max=1.0,
        precision=6,
    )

    merge_distance: bpy.props.FloatProperty(
        name="Merge Distance",
        description="Merge-by-distance threshold used during cleanup",
        default=1.0e-6,
        min=0.0,
        precision=8,
    )

    adaptivity: bpy.props.FloatProperty(
        name="Adaptivity",
        description="Remesh adaptivity (0 = uniform). Higher can reduce polycount but may introduce size variance",
        default=0.0,
        min=0.0,
        max=1.0,
        precision=3,
    )

    auto_voxel_from_shell: bpy.props.BoolProperty(
        name="Auto from Shell Model",
        description="When voxel size is 0, bias the auto voxel size based on the selected PPF shell model",
        default=True,
    )

    report_mesh_quality: bpy.props.BoolProperty(
        name="Report Mesh Quality",
        description="Compute triangle quality stats after remeshing and warn if the mesh is likely to be unstable",
        default=True,
    )

    contact_gap_mode: bpy.props.EnumProperty(
        name="Contact Gap",
        description="Suggest (or apply) a contact gap based on the remeshed median edge length",
        items=[
            ("OFF", "Off", "Do not compute a contact-gap suggestion"),
            ("SUGGEST", "Suggest", "Report a suggested tri_contact_gap/static_contact_gap without changing settings"),
            ("SET", "Set", "Write the suggested gaps into the active PPF settings"),
        ],
        default="SUGGEST",
    )

    contact_gap_factor: bpy.props.FloatProperty(
        name="Gap Factor",
        description="Suggested gap = factor × median edge length (world units)",
        default=0.1,
        min=0.0,
        soft_min=0.0,
        soft_max=1.0,
        precision=4,
    )

    dt_mode: bpy.props.EnumProperty(
        name="dt",
        description="Suggest (or apply) a conservative dt based on mesh resolution for stability",
        items=[
            ("OFF", "Off", "Do not compute a dt suggestion"),
            ("SUGGEST", "Suggest", "Report a suggested dt without changing settings"),
            ("SET", "Set", "Write the suggested dt into the active PPF scene settings"),
        ],
        default="SUGGEST",
    )

    dt_max_gravity_disp_frac: bpy.props.FloatProperty(
        name="Max Gravity Displacement",
        description="Stability heuristic: keep 0.5*|g|*dt^2 ≤ frac × characteristic length (edge/gap)",
        default=0.1,
        min=0.0,
        soft_min=0.0,
        soft_max=0.5,
        precision=4,
    )

    @classmethod
    def poll(cls, context):
        obj = getattr(context, "object", None)
        return obj is not None and getattr(obj, "type", None) == "MESH"

    def execute(self, context):
        obj = getattr(context, "object", None)
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, "Select a mesh object")
            return {"CANCELLED"}

        # Force Object Mode for modifier application and stable mesh access.
        try:
            if getattr(obj, "mode", "OBJECT") != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:
            pass

        has_constraints, reason = _ppf_object_has_constraints(obj)
        if has_constraints:
            self.report({"WARNING"}, reason)
            self.report({"ERROR"}, "Remesh blocked: remove pins/attach/stitches before remeshing")
            return {"CANCELLED"}

        prefs = _get_addon_prefs(context)
        prep_mode = getattr(prefs, "cloth_prep_mode", "DUPLICATE") if prefs is not None else "DUPLICATE"

        target_obj = obj
        if prep_mode == "DUPLICATE":
            try:
                dup = obj.copy()
                dup.data = obj.data.copy()
                dup.animation_data_clear()
                dup.name = f"{obj.name}_SIM"
                context.collection.objects.link(dup)
                target_obj = dup
            except Exception as exc:
                self.report({"ERROR"}, f"Failed to duplicate object: {exc}")
                return {"CANCELLED"}

        # Make target active/selected for modifier ops.
        try:
            for o in context.selected_objects:
                o.select_set(False)
        except Exception:
            pass
        try:
            target_obj.select_set(True)
            context.view_layer.objects.active = target_obj
        except Exception:
            pass

        mesh = getattr(target_obj, "data", None)
        if mesh is None:
            self.report({"ERROR"}, "Target object has no mesh data")
            return {"CANCELLED"}

        before_v = len(mesh.vertices)
        before_f = len(mesh.polygons)

        # Compute shell-aware voxel size early so cleanup can be scale-aware.
        used_voxel = float(self.voxel_size)
        voxel_scale = 1.0
        shell_settings = _effective_ppf_shell_settings(context, target_obj)
        tri_model = str(getattr(shell_settings, "tri_model", "baraff-witkin") or "baraff-witkin")
        tri_young = None
        tri_bend = None
        tri_strain = None
        try:
            tri_young = float(getattr(shell_settings, "tri_young_mod", None))
        except Exception:
            tri_young = None
        try:
            tri_bend = float(getattr(shell_settings, "tri_bend", None))
        except Exception:
            tri_bend = None
        try:
            tri_strain = float(getattr(shell_settings, "tri_strain_limit", None))
        except Exception:
            tri_strain = None

        if used_voxel <= 0.0:
            used_voxel = _auto_voxel_size_for_obj(target_obj)
            if bool(self.auto_voxel_from_shell):
                voxel_scale = _voxel_scale_for_shell_params(
                    tri_model=tri_model,
                    tri_young_mod=tri_young,
                    tri_bend=tri_bend,
                    tri_strain_limit=tri_strain,
                )
                used_voxel *= float(voxel_scale)

        used_voxel = max(1.0e-6, float(used_voxel))
        merge_distance = _auto_merge_distance_for_voxel(float(self.merge_distance), used_voxel)

        # Nonlinear shells tend to behave better with relatively uniform triangles.
        if (tri_model or "").strip().lower() in {"stvk", "snhk"} and float(self.adaptivity) > 0.2:
            self.report({"WARNING"}, "High adaptivity can introduce size variance; consider <= 0.2 for stvk/snhk")

        # Cleanup before remeshing helps avoid pathological inputs.
        try:
            _bmesh_cleanup_and_triangulate(mesh, merge_distance=float(merge_distance))
        except Exception as exc:
            self.report({"WARNING"}, f"Pre-cleanup failed: {exc}")

        # Remesh via modifier (Blender-internal, robust).

        try:
            mod = target_obj.modifiers.new(name="ClothPrep_Remesh", type="REMESH")
            try:
                mod.mode = "VOXEL"
            except Exception:
                pass
            try:
                mod.voxel_size = float(used_voxel)
            except Exception:
                pass
            try:
                mod.adaptivity = float(self.adaptivity)
            except Exception:
                pass

            bpy.ops.object.modifier_apply(modifier=mod.name)
        except Exception as exc:
            self.report({"ERROR"}, f"Remesh failed: {exc}")
            return {"CANCELLED"}

        # Final cleanup + triangulation (PPF export uses loop triangles but we keep it explicit).
        try:
            _bmesh_cleanup_and_triangulate(target_obj.data, merge_distance=float(merge_distance))
        except Exception:
            pass

        after_v = len(target_obj.data.vertices)
        after_f = len(target_obj.data.polygons)

        # Post-remesh mesh quality report + safe suggestions.
        quality = None
        suggested_gap = None
        suggested_dt = None
        if bool(self.report_mesh_quality) or str(getattr(self, "contact_gap_mode", "OFF")) != "OFF":
            try:
                quality = _mesh_quality_report(target_obj)
            except Exception as exc:
                quality = None
                self.report({"WARNING"}, f"Mesh quality analysis failed: {exc}")

        if quality is not None:
            edges = quality.get("edge_len_sorted") or []
            aspects = quality.get("aspect_sorted") or []
            min_angles = quality.get("min_angle_sorted") or []

            deg_faces = int(quality.get("degenerate_faces", 0) or 0)
            nonmanifold = int(quality.get("nonmanifold_edges", 0) or 0)
            boundary = int(quality.get("boundary_edges", 0) or 0)

            if edges:
                median_edge = _percentile_sorted(edges, 0.5)
                p10_edge = _percentile_sorted(edges, 0.1)
                p90_edge = _percentile_sorted(edges, 0.9)
            else:
                median_edge = 0.0
                p10_edge = 0.0
                p90_edge = 0.0

            if min_angles:
                p1_angle = _percentile_sorted(min_angles, 0.01)
                p10_angle = _percentile_sorted(min_angles, 0.10)
            else:
                p1_angle = 0.0
                p10_angle = 0.0

            if aspects:
                p50_aspect = _percentile_sorted(aspects, 0.5)
                p90_aspect = _percentile_sorted(aspects, 0.9)
            else:
                p50_aspect = 0.0
                p90_aspect = 0.0

            # Report summary
            if bool(self.report_mesh_quality):
                self.report(
                    {"INFO"},
                    (
                        "Mesh quality: "
                        f"edge median={median_edge:.4g} (p10={p10_edge:.4g}, p90={p90_edge:.4g}), "
                        f"min-angle p1={p1_angle:.2f}° (p10={p10_angle:.2f}°), "
                        f"aspect p50={p50_aspect:.3g} (p90={p90_aspect:.3g}), "
                        f"nonmanifold_edges={nonmanifold}, boundary_edges={boundary}, deg_faces={deg_faces}"
                    ),
                )

                if deg_faces > 0:
                    self.report({"WARNING"}, f"Mesh has {deg_faces} degenerate faces (very likely unstable)")
                if nonmanifold > 0:
                    self.report({"WARNING"}, f"Mesh has {nonmanifold} non-manifold edges (can destabilize collision/contact)")
                if p1_angle > 0.0 and p1_angle < 10.0:
                    self.report({"WARNING"}, f"Very small triangle angles detected (p1={p1_angle:.2f}°): consider smaller voxel size")
                if p90_aspect > 0.0 and p90_aspect > 6.0:
                    self.report({"WARNING"}, f"High aspect triangles detected (p90={p90_aspect:.3g}): collision/contact may be noisy")

            # Contact gap suggestion/apply
            mode = str(getattr(self, "contact_gap_mode", "OFF"))
            if mode != "OFF" and median_edge > 0.0:
                factor = float(max(0.0, self.contact_gap_factor))
                # Clamp to a reasonable range.
                try:
                    dims = target_obj.dimensions
                    diag = math.sqrt(float(dims.x) ** 2 + float(dims.y) ** 2 + float(dims.z) ** 2)
                except Exception:
                    diag = 0.0
                gap_max = diag / 10.0 if diag > 0.0 else median_edge
                suggested_gap = float(min(max(factor * median_edge, 1.0e-6), max(1.0e-6, gap_max)))

                if mode == "SUGGEST":
                    self.report({"INFO"}, f"Suggested contact gap: {suggested_gap:.4g} (factor {factor:.3g} × edge median)")
                elif mode == "SET":
                    shell_settings = _effective_ppf_shell_settings(context, target_obj)
                    if shell_settings is not None and hasattr(shell_settings, "tri_contact_gap"):
                        try:
                            shell_settings.tri_contact_gap = float(suggested_gap)
                        except Exception:
                            pass
                    scene_settings = getattr(getattr(context, "scene", None), "andosim_artezbuild", None)
                    if scene_settings is not None and hasattr(scene_settings, "static_contact_gap"):
                        try:
                            scene_settings.static_contact_gap = float(suggested_gap)
                        except Exception:
                            pass
                    self.report({"INFO"}, f"Set tri_contact_gap/static_contact_gap to {suggested_gap:.4g}")

            # dt suggestion/apply (solver stability)
            dt_mode = str(getattr(self, "dt_mode", "OFF"))
            if dt_mode != "OFF" and median_edge > 0.0:
                scene_settings = getattr(getattr(context, "scene", None), "andosim_artezbuild", None)
                g = 0.0
                dt_current = None
                if scene_settings is not None:
                    try:
                        g = float(getattr(scene_settings, "gravity", -9.8))
                    except Exception:
                        g = 0.0
                    try:
                        dt_current = float(getattr(scene_settings, "dt", 1.0e-3))
                    except Exception:
                        dt_current = None

                g_abs = abs(float(g))
                frac = float(max(0.0, getattr(self, "dt_max_gravity_disp_frac", 0.1)))
                if g_abs > 1.0e-12 and frac > 0.0:
                    # Ensure 0.5*g*dt^2 <= frac * L  => dt <= sqrt(2*frac*L/|g|)
                    # Use edge-based characteristic length; optionally tighten using suggested gap.
                    L = float(median_edge)
                    if suggested_gap is not None and suggested_gap > 0.0:
                        L = min(L, float(suggested_gap))
                    suggested_dt = math.sqrt(max(0.0, 2.0 * frac * L / g_abs))
                    # Clamp to a sane upper bound (avoid suggesting very large dt on large meshes).
                    suggested_dt = float(min(max(suggested_dt, 1.0e-6), 0.1))

                    if dt_mode == "SUGGEST":
                        self.report({"INFO"}, f"Suggested dt (stability): {suggested_dt:.4g} (frac {frac:.3g}, |g|={g_abs:.4g}, L={L:.4g})")
                    elif dt_mode == "SET":
                        if scene_settings is not None and hasattr(scene_settings, "dt"):
                            try:
                                scene_settings.dt = float(suggested_dt)
                                self.report({"INFO"}, f"Set dt to {suggested_dt:.4g} (stability heuristic)")
                            except Exception:
                                self.report({"WARNING"}, "Failed to set dt")

                    if dt_current is not None and suggested_dt is not None and dt_current > suggested_dt * 1.5:
                        self.report(
                            {"WARNING"},
                            f"Current dt={dt_current:.4g} is high vs suggested {suggested_dt:.4g}; consider reducing for stability",
                        )
                elif dt_mode != "OFF":
                    self.report({"INFO"}, "dt suggestion skipped: gravity is ~0")

        if prep_mode == "DUPLICATE":
            self.report({"INFO"}, f"Created '{target_obj.name}' from '{obj.name}'")
        if float(voxel_scale) != 1.0:
            self.report({"INFO"}, f"Shell settings applied (model '{tri_model}', voxel scale {voxel_scale:.3g})")
        self.report({"INFO"}, f"Cloth mesh prepared (voxel={used_voxel:.6g}). Verts {before_v}→{after_v}, Faces {before_f}→{after_f}")
        return {"FINISHED"}


def _ppf_clear_mesh_cache_modifier(obj: bpy.types.Object) -> None:
    try:
        for mod in list(getattr(obj, "modifiers", []) or []):
            if getattr(mod, "type", None) == "MESH_CACHE" and getattr(mod, "name", "") == "PPF_Cache":
                try:
                    obj.modifiers.remove(mod)
                except Exception:
                    pass
    except Exception:
        pass


def _ppf_capture_mesh_snapshot(deformable_slices: list[tuple[str, int, int]]) -> tuple[bool, str | None]:
    """Capture object-local vertex positions for the provided deformables.

    Returns (ok, error_message).
    """

    _PPF_MESH_SNAPSHOT.clear()
    captured = 0

    for name, _start, count in deformable_slices:
        obj = bpy.data.objects.get(str(name))
        if obj is None or obj.type != "MESH":
            return False, f"Deformable '{name}' missing or not a mesh"
        if obj.mode != "OBJECT":
            return False, f"Deformable '{name}' must be in Object Mode"
        mesh = getattr(obj, "data", None)
        if mesh is None:
            return False, f"Deformable '{name}' has no mesh data"
        if len(mesh.vertices) != int(count):
            return False, f"Vertex count changed on '{name}'; apply topology modifiers"

        mesh_ptr = int(mesh.as_pointer())
        coords = [0.0] * (int(count) * 3)
        try:
            mesh.vertices.foreach_get("co", coords)
        except Exception as exc:
            return False, f"Failed to snapshot vertices for '{name}': {exc}"

        _PPF_MESH_SNAPSHOT[mesh_ptr] = {
            "obj_name": obj.name,
            "mesh_ptr": mesh_ptr,
            "verts_tot": int(count),
            "coords": coords,
        }
        captured += 1

    if captured <= 0:
        return False, "No deformables to snapshot"
    return True, None


def _ppf_restore_mesh_snapshot(context) -> tuple[int, int]:
    """Restore meshes from the last snapshot.

    Returns (restored_count, skipped_count).
    """

    restored = 0
    skipped = 0

    for rec in list(_PPF_MESH_SNAPSHOT.values()):
        obj_name = str(rec.get("obj_name", ""))
        mesh_ptr = int(rec.get("mesh_ptr", 0) or 0)
        verts_tot = int(rec.get("verts_tot", 0) or 0)
        coords = rec.get("coords", None)
        if not coords or verts_tot <= 0 or mesh_ptr <= 0:
            skipped += 1
            continue

        obj = bpy.data.objects.get(obj_name)
        if obj is None or obj.type != "MESH" or getattr(obj, "data", None) is None:
            # Fallback: try to find any object that still references this mesh datablock.
            obj = None
            for candidate in bpy.data.objects:
                try:
                    if candidate is not None and candidate.type == "MESH" and candidate.data is not None:
                        if int(candidate.data.as_pointer()) == mesh_ptr:
                            obj = candidate
                            break
                except Exception:
                    continue

        if obj is None or obj.type != "MESH" or getattr(obj, "data", None) is None:
            skipped += 1
            continue

        if obj.mode != "OBJECT":
            skipped += 1
            continue

        mesh = obj.data
        if len(mesh.vertices) != verts_tot:
            skipped += 1
            continue

        _ppf_clear_mesh_cache_modifier(obj)

        try:
            mesh.vertices.foreach_set("co", coords)
            mesh.update()
            obj.update_tag()
            restored += 1
        except Exception:
            skipped += 1
            continue

    try:
        context.view_layer.update()
    except Exception:
        pass

    _PPF_MESH_SNAPSHOT.clear()
    return restored, skipped


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

        global _PPF_BACKEND_VERSION, _PPF_BACKEND_FILE, _PPF_BACKEND_VERSION_PRINTED
        _PPF_BACKEND_VERSION = str(getattr(ppf_cts_backend, "__version__", "?"))
        _PPF_BACKEND_FILE = str(getattr(ppf_cts_backend, "__file__", "?"))
        if not _PPF_BACKEND_VERSION_PRINTED:
            _PPF_BACKEND_VERSION_PRINTED = True
            try:
                print(f"[PPF] ppf_cts_backend version: {_PPF_BACKEND_VERSION} ({_PPF_BACKEND_FILE})")
            except Exception:
                pass

        return ppf_cts_backend
    except Exception as exc:
        try:
            import sys

            py = sys.version_info
            py_tag = f"cp{py.major}{py.minor}"
            if (py.major, py.minor) != (3, 11):
                return RuntimeError(
                    f"{exc} (Blender Python is {py.major}.{py.minor} / {py_tag}; "
                    "this extension bundles a cp311 wheel. Build/bundle a matching wheel "
                    "for your Blender Python, or use a Blender build with Python 3.11.)"
                )
        except Exception:
            pass
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


def _ppf_build_pin_targets(
    context,
    objs: list[tuple[bpy.types.Object, int, int]],
    attach_bindings,
) -> tuple[list[int], list[float]]:
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
        for gidx, target_name, (i0, i1, i2), (w0, w1, w2) in attach_bindings or []:
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

            _eval_obj, eval_mesh = cached
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

    return indices, positions_flat


def _ppf_apply_pin_targets(session, indices: list[int], positions_flat: list[float]) -> bool:
    if session is None:
        return False
    set_fn = getattr(session, "set_pin_targets", None)
    if set_fn is None:
        return False

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


def _sync_ppf_pin_targets_to_backend(context, session, objs: list[tuple[bpy.types.Object, int, int]], attach_bindings) -> bool:
    indices, positions_flat = _ppf_build_pin_targets(context, objs, attach_bindings)
    return _ppf_apply_pin_targets(session, indices, positions_flat)


def _ppf_build_collision_mesh_vertices_flat(context, names: list[str]) -> list[float] | None:
    if not names:
        return None

    depsgraph = context.evaluated_depsgraph_get()
    verts_flat: list[float] = []

    for name in names:
        obj = bpy.data.objects.get(str(name))
        if obj is None or obj.type != "MESH":
            return None

        eval_obj = obj.evaluated_get(depsgraph)
        eval_mesh = None
        try:
            eval_mesh = eval_obj.to_mesh()
        except Exception:
            eval_mesh = None
        if eval_mesh is None:
            return None

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
        return None

    return verts_flat


def _ppf_apply_collision_mesh_vertices(session, verts_flat: list[float] | None) -> bool:
    if session is None or not verts_flat:
        return False
    set_fn = getattr(session, "set_collision_mesh_vertices", None)
    if set_fn is None:
        return False
    try:
        set_fn(verts_flat)
        return True
    except Exception:
        return False


def _sync_ppf_collision_mesh_to_backend(context, session, collider_object_names: list[str]) -> bool:
    """If the backend supports it, push animated/static-collider mesh vertices each tick.

    The solver scene is exported with static colliders concatenated into a single collision mesh.
    For animated colliders, we rebuild that concatenated vertex buffer in the exact same order
    (by `state.collider_object_names`) and send it to the backend.
    """

    verts_flat = _ppf_build_collision_mesh_vertices_flat(context, list(collider_object_names or []))
    return _ppf_apply_collision_mesh_vertices(session, verts_flat)


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

            ok, err = _ppf_capture_mesh_snapshot(list(state.deformable_slices))
            if not ok:
                self.report({"ERROR"}, str(err or "Failed to snapshot meshes"))
                return {"CANCELLED"}
            settings.ppf_has_snapshot = True

            # Always create a fresh per-run output dir (unless the user explicitly set one).
            # This avoids mixing backend traces across multiple runs, which makes debugging PCG
            # failures much harder and can produce misleading "tail" summaries.
            output_dir = (settings.output_dir or "").strip()
            if not output_dir:
                output_dir = tempfile.mkdtemp(prefix="ppf_blender_")
                settings.output_dir = output_dir
            else:
                # If output_dir exists but was auto-populated from a previous run, prefer a new
                # temp dir to keep traces isolated. Heuristic: previous auto dir name.
                base = os.path.basename(output_dir.rstrip("/"))
                if base.startswith("ppf_blender_"):
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
        _sync_ppf_pin_targets_to_backend(context, state.session, objs, getattr(state, "attach_bindings", []) or [])
        _sync_ppf_collision_mesh_to_backend(context, state.session, list(getattr(state, "collider_object_names", []) or []))

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
            _sync_ppf_pin_targets_to_backend(context, state.session, objs, getattr(state, "attach_bindings", []) or [])
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


class ANDOSIM_ARTEZBUILD_OT_ppf_reset_simulation(bpy.types.Operator):
    bl_idname = "andosim_artezbuild.ppf_reset_simulation"
    bl_label = "Reset PPF Simulation"
    bl_options = {"REGISTER"}

    def execute(self, context):
        settings = _get_settings(context)
        if settings is None:
            self.report({"ERROR"}, "Addon settings not initialized")
            return {"CANCELLED"}

        if bool(getattr(settings, "ppf_baking", False)):
            self.report({"ERROR"}, "Cannot reset while baking")
            return {"CANCELLED"}

        # Stop realtime sim if running (ESC semantics remain "stop only").
        op = ANDOSIM_ARTEZBUILD_OT_ppf_run
        state = getattr(op, "_ppf", None)
        if state is not None and getattr(state, "running", False):
            state.running = False
            if state.timer is not None:
                try:
                    context.window_manager.event_timer_remove(state.timer)
                except Exception:
                    pass
                state.timer = None
            if state.session is not None:
                try:
                    state.session.close()
                except Exception:
                    pass
                state.session = None
            op._ppf = None
            settings.running = False

        if not bool(getattr(settings, "ppf_has_snapshot", False)) or not _PPF_MESH_SNAPSHOT:
            self.report({"INFO"}, "Nothing to reset")
            return {"CANCELLED"}

        restored, skipped = _ppf_restore_mesh_snapshot(context)
        settings.ppf_has_snapshot = False
        if restored > 0:
            msg = f"Reset simulation: restored {restored} object(s)"
            if skipped > 0:
                msg += f", skipped {skipped}"
            self.report({"INFO"}, msg)
            return {"FINISHED"}

        self.report({"WARNING"}, f"Reset simulation: nothing restored (skipped {skipped})")
        return {"CANCELLED"}


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

        # Blender's Mesh Cache modifier already supports an explicit frame offset
        # (`mod.frame_start`). To avoid double-offsetting (file start time + modifier
        # start frame), we write PC2 caches with a 0-based start time.
        start_sec = 0.0
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


class _PPFBakeState:
    def __init__(self):
        self.timer = None
        self.running = False
        self.cancel_requested = False

        self.old_dt: float | None = None
        self.sess = None
        self.export = None

        self.scene = None
        self.settings = None

        self.frame_start = 1
        self.frame_end = 1
        self.frame_tot = 1
        self.fps = 24.0

        self.output_dir = ""
        self.bake_dir: Path | None = None

        self.writers: list[tuple[bpy.types.Object, int, int, _PC2Writer, Path]] = []
        self.total_verts = 0
        self.curr: list[float] | None = None

        # PPF per-frame stepping
        self.dt = 0.0
        self.frame_dt = 0.0
        self.steps_per_frame = 1

        # Optional streaming
        self.deformable_slices: list[tuple[str, int, int]] = []
        self.attach_bindings = []
        self.collider_object_names: list[str] = []

        self.next_frame = 0
        self.progress_i = 0
        self.wm = None


class ANDOSIM_ARTEZBUILD_OT_ppf_bake_cache(bpy.types.Operator):
    bl_idname = "andosim_artezbuild.ppf_bake_cache"
    bl_label = "Bake PPF Cache (PC2)"
    bl_options = {"REGISTER"}

    _bake = None

    def execute(self, context):
        backend_or_exc = _try_import_backend()
        if isinstance(backend_or_exc, Exception):
            self.report({"ERROR"}, f"Failed to import ppf_cts_backend: {backend_or_exc}")
            return {"CANCELLED"}

        settings = _get_settings(context)
        if settings is None:
            self.report({"ERROR"}, "Addon settings not initialized")
            return {"CANCELLED"}

        if bool(getattr(settings, "ppf_baking", False)):
            self.report({"WARNING"}, "Bake already running")
            return {"CANCELLED"}

        if bool(getattr(settings, "running", False)):
            self.report({"ERROR"}, "Stop realtime simulation before baking")
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

        state = _PPFBakeState()
        state.running = True
        state.scene = scene
        state.settings = settings
        state.frame_start = frame_start
        state.frame_end = frame_end
        state.frame_tot = int(frame_end - frame_start + 1)
        state.fps = float(fps)
        state.frame_dt = 1.0 / float(state.fps)
        state.output_dir = output_dir
        state.bake_dir = bake_dir

        # Snapshot meshes so Reset Simulation can restore base geometry after bake/cache playback.
        # (Bake itself does not mutate meshes, but it adds a Mesh Cache modifier that overwrites the viewport.)
        try:
            export_for_snapshot = ppf_export.export_ppf_scene_from_roles(context, settings)
            slices_for_snapshot = list(getattr(export_for_snapshot, "deformable_slices", []) or [])
            if slices_for_snapshot:
                ok, err = _ppf_capture_mesh_snapshot(slices_for_snapshot)
                if ok:
                    settings.ppf_has_snapshot = True
        except Exception:
            pass

        # Use substeps per Blender frame instead of forcing dt=1/fps.
        # Forcing dt to a large frame delta can destabilize the solver (PCG failures).
        state.old_dt = float(getattr(settings, "dt", 1e-3))
        dt = float(state.old_dt)
        if not (dt > 0.0):
            dt = 1.0e-3
        if dt > float(state.frame_dt):
            dt = float(state.frame_dt)
        state.dt = float(dt)
        settings.dt = float(state.dt)
        state.steps_per_frame = max(1, int(math.ceil(float(state.frame_dt) / float(state.dt) - 1.0e-12)))

        try:
            try:
                scene.frame_set(frame_start)
            except Exception:
                pass

            export = ppf_export.export_ppf_scene_from_roles(context, settings)
            state.export = export
            for w in getattr(export, "warnings", []) or []:
                self.report({"WARNING"}, str(w))

            slices = list(getattr(export, "deformable_slices", []) or [])
            if not slices:
                self.report({"ERROR"}, "No deformables enabled for PPF (tag objects as Deformable)")
                self._finish(context, state, cancelled=True)
                return {"CANCELLED"}

            state.deformable_slices = list(slices)
            state.collider_object_names = list(getattr(export, "collider_object_names", []) or [])

            # Precompute attach bindings once at start (stable attachment).
            try:
                state.attach_bindings = _compute_attach_bindings(context, list(state.deformable_slices))
            except Exception:
                state.attach_bindings = []

            ok, err = _ppf_capture_mesh_snapshot(slices)
            if not ok:
                self.report({"ERROR"}, str(err or "Failed to snapshot meshes"))
                self._finish(context, state, cancelled=True)
                return {"CANCELLED"}
            settings.ppf_has_snapshot = True

            if int(state.steps_per_frame) > 1:
                self.report(
                    {"INFO"},
                    f"Bake uses {int(state.steps_per_frame)} substeps/frame (dt={state.dt:.3g}, frame_dt={state.frame_dt:.3g})",
                )

            for name, start, count in slices:
                obj = bpy.data.objects.get(name)
                if obj is None or obj.type != "MESH":
                    self.report({"ERROR"}, f"Deformable '{name}' missing or not a mesh")
                    self._finish(context, state, cancelled=True)
                    return {"CANCELLED"}
                if obj.mode != "OBJECT":
                    self.report({"ERROR"}, f"Deformable '{name}' must be in Object Mode")
                    self._finish(context, state, cancelled=True)
                    return {"CANCELLED"}
                if len(obj.data.vertices) != int(count):
                    self.report({"ERROR"}, f"Vertex count changed on '{name}'; apply topology modifiers")
                    self._finish(context, state, cancelled=True)
                    return {"CANCELLED"}

                pc2_path = bake_dir / f"{name}.pc2"
                writer = _PC2Writer(
                    pc2_path,
                    verts_tot=int(count),
                    frame_start=frame_start,
                    frame_tot=state.frame_tot,
                    fps=float(state.fps),
                )
                state.writers.append((obj, int(start), int(count), writer, pc2_path))

            for _obj, start, count, _writer, _pc2_path in state.writers:
                state.total_verts = max(int(state.total_verts), int(start) + int(count))

            state.curr = [0.0] * (int(state.total_verts) * 3)
            for obj, start, _count, _writer, _pc2_path in state.writers:
                mw = obj.matrix_world
                mesh = obj.data
                for i, v in enumerate(mesh.vertices):
                    wco = mw @ v.co
                    sx, sy, sz = ppf_export.blender_to_solver_xyz(float(wco.x), float(wco.y), float(wco.z))
                    j = start + i
                    state.curr[3 * j + 0] = sx
                    state.curr[3 * j + 1] = sy
                    state.curr[3 * j + 2] = sz

            # Sample 0: current local coords at frame_start.
            for obj, _start, count, writer, _pc2_path in state.writers:
                local = [0.0] * (int(count) * 3)
                try:
                    obj.data.vertices.foreach_get("co", local)
                except Exception:
                    for i, v in enumerate(obj.data.vertices):
                        local[3 * i + 0] = float(v.co.x)
                        local[3 * i + 1] = float(v.co.y)
                        local[3 * i + 2] = float(v.co.z)
                writer.write_frame(local)

            state.sess = backend_or_exc.Session(export.scene_path, output_dir)

            try:
                state.wm = context.window_manager
                state.wm.progress_begin(0, state.frame_tot)
            except Exception:
                state.wm = None

            state.next_frame = int(frame_start + 1)
            state.progress_i = 1

            settings.ppf_baking = True

            wm = context.window_manager
            state.timer = wm.event_timer_add(0.01, window=context.window)
            wm.modal_handler_add(self)
            self.__class__._bake = state
            return {"RUNNING_MODAL"}
        except Exception as exc:
            details = _ppf_failure_details(str(output_dir))
            self.report({"ERROR"}, f"Bake failed: {exc}. {details}")
            self._finish(context, state, cancelled=True)
            return {"CANCELLED"}

    def modal(self, context, event):
        state = self.__class__._bake
        if state is None or not getattr(state, "running", False):
            return {"CANCELLED"}

        if event.type == "ESC":
            state.cancel_requested = True

        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        if bool(getattr(state, "cancel_requested", False)):
            self._finish(context, state, cancelled=True)
            self.report({"INFO"}, "Bake cancelled")
            return {"CANCELLED"}

        if int(getattr(state, "next_frame", 0)) > int(getattr(state, "frame_end", 0)):
            self._finish(context, state, cancelled=False)
            self.report({"INFO"}, f"Baked PC2 cache to {state.bake_dir}")
            return {"FINISHED"}

        try:
            frame = int(state.next_frame)
            try:
                state.scene.frame_set(frame)
            except Exception:
                pass

            # Build pin/collider payloads once per Blender frame.
            # We still apply them every solver substep (below) to satisfy the
            # "update colliders every substep" requirement; within a single Blender
            # frame these payloads are typically constant.
            objs = [(obj, int(start), int(count)) for obj, start, count, _writer, _pc2_path in state.writers]
            attach_bindings = list(getattr(state, "attach_bindings", []) or [])
            pin_indices, pin_positions_flat = _ppf_build_pin_targets(context, objs, attach_bindings)
            collider_names = list(getattr(state, "collider_object_names", []) or [])
            collider_verts_flat = _ppf_build_collision_mesh_vertices_flat(context, collider_names)

            curr = state.curr
            if curr is None:
                raise RuntimeError("Missing bake state buffer")

            # Advance the solver by one Blender frame using substeps.
            for _ in range(int(state.steps_per_frame)):
                if bool(getattr(state, "cancel_requested", False)):
                    self._finish(context, state, cancelled=True)
                    self.report({"INFO"}, "Bake cancelled")
                    return {"CANCELLED"}

                # Accuracy-first: apply pins/colliders every substep.
                _ppf_apply_pin_targets(state.sess, pin_indices, pin_positions_flat)
                _ppf_apply_collision_mesh_vertices(state.sess, collider_verts_flat)

                out = state.sess.step(curr)
                if len(out) != len(curr):
                    raise RuntimeError(f"Backend returned {len(out)} floats, expected {len(curr)}")
                curr = list(out)
            state.curr = curr

            for obj, start, count, writer, _pc2_path in state.writers:
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

            if state.wm is not None:
                try:
                    state.wm.progress_update(int(state.progress_i))
                except Exception:
                    pass

            state.next_frame = int(state.next_frame) + 1
            state.progress_i = int(state.progress_i) + 1
            return {"PASS_THROUGH"}
        except Exception as exc:
            output_dir = ""
            try:
                output_dir = str(getattr(state, "output_dir", "") or "")
            except Exception:
                output_dir = ""
            details = _ppf_failure_details(output_dir)
            self.report({"ERROR"}, f"Bake failed: {exc}. {details}")
            self._finish(context, state, cancelled=True)
            return {"CANCELLED"}

    def _finish(self, context, state: _PPFBakeState, *, cancelled: bool) -> None:
        settings = getattr(state, "settings", None)
        if settings is not None:
            try:
                settings.ppf_baking = False
            except Exception:
                pass
            try:
                if state.old_dt is not None:
                    settings.dt = float(state.old_dt)
            except Exception:
                pass

        state.running = False
        if state.timer is not None:
            try:
                context.window_manager.event_timer_remove(state.timer)
            except Exception:
                pass
            state.timer = None

        if state.wm is not None:
            try:
                state.wm.progress_end()
            except Exception:
                pass
            state.wm = None

        if state.sess is not None:
            try:
                state.sess.close()
            except Exception:
                pass
            state.sess = None

        for _obj, _start, _count, writer, _pc2_path in state.writers:
            try:
                writer.close()
            except Exception:
                pass

        if cancelled:
            for _obj, _start, _count, _writer, pc2_path in state.writers:
                try:
                    pc2_path.unlink(missing_ok=True)
                except Exception:
                    pass
        else:
            for obj, _start, _count, _writer, pc2_path in state.writers:
                try:
                    if (not pc2_path.exists()) or pc2_path.stat().st_size <= 32:
                        # PC2 header is 32 bytes; anything <= that means "no frames".
                        raise RuntimeError(f"PC2 cache missing/empty: {pc2_path}")
                except Exception as exc:
                    try:
                        self.report({"ERROR"}, f"Bake produced no usable cache for '{obj.name}': {exc}")
                    except Exception:
                        pass
                    continue
                _ppf_clear_mesh_cache_modifier(obj)
                try:
                    mod = obj.modifiers.new(name="PPF_Cache", type="MESH_CACHE")
                except Exception:
                    continue
                mod.cache_format = "PC2"
                mod.filepath = str(pc2_path)
                mod.time_mode = "FRAME"
                mod.play_mode = "SCENE"
                mod.deform_mode = "OVERWRITE"
                mod.frame_start = int(state.frame_start)
                mod.frame_scale = 1.0

        self.__class__._bake = None


class ANDOSIM_ARTEZBUILD_OT_ppf_cancel_bake(bpy.types.Operator):
    bl_idname = "andosim_artezbuild.ppf_cancel_bake"
    bl_label = "Cancel PPF Bake"
    bl_options = {"REGISTER"}

    def execute(self, context):
        op = ANDOSIM_ARTEZBUILD_OT_ppf_bake_cache
        state = getattr(op, "_bake", None)
        if state is None or not getattr(state, "running", False):
            return {"CANCELLED"}
        state.cancel_requested = True
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


def _ppf_parse_last_time_values(path: Path) -> tuple[float | None, list[str]]:
    """Parse a trace stream with lines like: `<time> <value>`.

    Some streams can write multiple rows with the same simulation time per step.
    This returns all values for the last (time>0) time stamp.
    """

    try:
        text = path.read_text(errors="replace")
    except Exception:
        return None, []

    parsed: list[tuple[float, str]] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) < 2:
            continue
        try:
            t = float(parts[0])
        except Exception:
            continue
        parsed.append((t, parts[1]))

    if not parsed:
        return None, []

    # Prefer last row with time>0 (some streams append summaries at t=0).
    last_t = None
    for t, _v in reversed(parsed):
        if t > 0.0:
            last_t = t
            break
    if last_t is None:
        last_t = parsed[-1][0]

    values = [v for (t, v) in parsed if t == last_t]
    return last_t, values


def _ppf_trace_flag_status(name: str, values: list[str]) -> str | None:
    if not values:
        return None

    seen_ok = False
    seen_fail = False
    for value in values:
        v = value.strip().lower()
        if v in {"1", "1.0", "true"}:
            seen_ok = True
        elif v in {"0", "0.0", "false"}:
            seen_fail = True

    if seen_ok and seen_fail:
        return f"{name}=MIXED"
    if seen_fail:
        return f"{name}=FAIL"
    if seen_ok:
        return f"{name}=OK"
    return f"{name}={values[-1]}"


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

    # Include backend version to help detect Blender caching/old wheels.
    if _PPF_BACKEND_VERSION and _PPF_BACKEND_VERSION != "?":
        summary_bits.append(f"backend={_PPF_BACKEND_VERSION}")

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

        if key in {"linsolve", "intersection"}:
            t, values = _ppf_parse_last_time_values(path)
            if t is not None and t > 0.0:
                last_t = t if last_t is None else max(last_t, t)
            hint = _ppf_trace_flag_status(key, values)
            if hint:
                summary_bits.append(hint)
            continue

        t, v = _ppf_parse_last_sample(path)
        if t is not None and t > 0.0:
            last_t = t if last_t is None else max(last_t, t)
        if v is None:
            continue
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
