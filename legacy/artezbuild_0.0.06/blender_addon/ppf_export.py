import os
import tempfile
from array import array
from dataclasses import dataclass, field

import bpy


# PPF uses Y as the gravity axis by default. Blender is Z-up.
# Map Blender coords -> Solver coords so that solver Y corresponds to Blender Z.
#   solver = (bx, bz, by)
#   blender = (sx, sz, sy)

def blender_to_solver_xyz(x: float, y: float, z: float) -> tuple[float, float, float]:
    return (x, z, y)


def solver_to_blender_xyz(x: float, y: float, z: float) -> tuple[float, float, float]:
    return (x, z, y)


def _write_f64_colmajor_vec3(path: str, verts_xyz: list[tuple[float, float, float]]):
    # Store a 3xN matrix as interleaved columns:
    # [x0, y0, z0, x1, y1, z1, ...]
    buff = array("d")
    for x, y, z in verts_xyz:
        buff.append(float(x))
        buff.append(float(y))
        buff.append(float(z))
    with open(path, "wb") as f:
        buff.tofile(f)


def _write_f32_colmajor_vec3(path: str, verts_xyz: list[tuple[float, float, float]]):
    buff = array("f")
    for x, y, z in verts_xyz:
        buff.append(float(x))
        buff.append(float(y))
        buff.append(float(z))
    with open(path, "wb") as f:
        buff.tofile(f)


def _write_f32_colmajor_vec2(path: str, vec2: list[tuple[float, float]]):
    buff = array("f")
    for a, b in vec2:
        buff.append(float(a))
        buff.append(float(b))
    with open(path, "wb") as f:
        buff.tofile(f)


def _write_u32(path: str, values: list[int]):
    with open(path, "wb") as f:
        array("I", (int(v) for v in values)).tofile(f)


def _write_u64(path: str, values: list[int]):
    with open(path, "wb") as f:
        array("Q", (int(v) for v in values)).tofile(f)


def _write_u64_colmajor_tris(path: str, tris: list[tuple[int, int, int]]):
    # Store a 3xN matrix as interleaved columns (triangle index triplets).
    buff = array("Q")
    for a, b, c in tris:
        buff.append(int(a))
        buff.append(int(b))
        buff.append(int(c))
    with open(path, "wb") as f:
        buff.tofile(f)


def _write_u8(path: str, values: list[int]):
    with open(path, "wb") as f:
        array("B", (int(v) for v in values)).tofile(f)


def _write_f32(path: str, values: list[float]):
    with open(path, "wb") as f:
        array("f", (float(v) for v in values)).tofile(f)


@dataclass
class ExportResult:
    scene_path: str
    deformable_object_names: list[str]
    collider_object_names: list[str]
    deformable_slices: list[tuple[str, int, int]]
    warnings: list[str] = field(default_factory=list)


def _mesh_eval_to_world_tris(obj: bpy.types.Object, depsgraph) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    obj_eval = obj.evaluated_get(depsgraph)
    mesh_eval = obj_eval.to_mesh()
    if mesh_eval is None:
        raise ValueError(f"Object '{obj.name}' has no mesh")

    try:
        mesh_eval.calc_loop_triangles()
        if not getattr(mesh_eval, "loop_triangles", None):
            raise ValueError(f"Object '{obj.name}' has no triangles")

        M = obj_eval.matrix_world
        verts = [(float((M @ v.co).x), float((M @ v.co).y), float((M @ v.co).z)) for v in mesh_eval.vertices]
        tris = [tuple(int(i) for i in tri.vertices) for tri in mesh_eval.loop_triangles]
        return verts, tris
    finally:
        obj_eval.to_mesh_clear()


def _mesh_eval_to_world_edges(obj: bpy.types.Object, depsgraph) -> tuple[list[tuple[float, float, float]], list[tuple[int, int]]]:
    obj_eval = obj.evaluated_get(depsgraph)
    mesh_eval = obj_eval.to_mesh()
    if mesh_eval is None:
        raise ValueError(f"Object '{obj.name}' has no mesh")

    try:
        M = obj_eval.matrix_world
        verts = [(float((M @ v.co).x), float((M @ v.co).y), float((M @ v.co).z)) for v in mesh_eval.vertices]
        edges = [(int(e.vertices[0]), int(e.vertices[1])) for e in mesh_eval.edges]
        return verts, edges
    finally:
        obj_eval.to_mesh_clear()


def _point_segment_closest(p, a, b):
    # Returns (dist2, t) where closest point is a + t*(b-a), clamped to [0,1]
    ax, ay, az = a
    bx, by, bz = b
    px, py, pz = p
    abx = bx - ax
    aby = by - ay
    abz = bz - az
    apx = px - ax
    apy = py - ay
    apz = pz - az
    denom = abx * abx + aby * aby + abz * abz
    if denom <= 1e-20:
        t = 0.0
    else:
        t = (apx * abx + apy * aby + apz * abz) / denom
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
    cx = ax + t * abx
    cy = ay + t * aby
    cz = az + t * abz
    dx = px - cx
    dy = py - cy
    dz = pz - cz
    return (dx * dx + dy * dy + dz * dz, t)


def export_ppf_scene(context, settings, target_obj: bpy.types.Object, collider_objs: list[bpy.types.Object]) -> ExportResult:
    depsgraph = context.evaluated_depsgraph_get()

    target_world_verts, target_tris = _mesh_eval_to_world_tris(target_obj, depsgraph)
    if len(target_world_verts) == 0 or len(target_tris) == 0:
        raise ValueError("Target mesh must have vertices and triangles")

    static_world_verts: list[tuple[float, float, float]] = []
    static_tris: list[tuple[int, int, int]] = []
    collider_names: list[str] = []

    for obj in collider_objs:
        if obj is None or obj.type != "MESH" or obj.name == target_obj.name:
            continue
        v, t = _mesh_eval_to_world_tris(obj, depsgraph)
        if not v or not t:
            continue
        offset = len(static_world_verts)
        static_world_verts.extend(v)
        static_tris.extend([(a + offset, b + offset, c + offset) for (a, b, c) in t])
        collider_names.append(obj.name)

    # Apply axis mapping into solver coordinate system.
    target_solver_verts = [blender_to_solver_xyz(*v) for v in target_world_verts]
    static_solver_verts = [blender_to_solver_xyz(*v) for v in static_world_verts]

    # Create a temp scene folder.
    root = tempfile.mkdtemp(prefix="ppf_scene_", dir=bpy.app.tempdir)
    bin_dir = os.path.join(root, "bin")
    param_dir = os.path.join(bin_dir, "param")
    os.makedirs(param_dir, exist_ok=True)

    n_vert = len(target_solver_verts)
    n_tri = len(target_tris)
    n_static_vert = len(static_solver_verts)
    n_static_tri = len(static_tris)

    # Displacement map: use one entry (0,0,0) and map all verts to it.
    _write_f64_colmajor_vec3(os.path.join(bin_dir, "displacement.bin"), [(0.0, 0.0, 0.0)])
    _write_u32(os.path.join(bin_dir, "vert_dmap.bin"), [0] * n_vert)

    # Core geometry
    _write_f64_colmajor_vec3(os.path.join(bin_dir, "vert.bin"), target_solver_verts)
    _write_f32_colmajor_vec3(os.path.join(bin_dir, "vel.bin"), [(0.0, 0.0, 0.0)] * n_vert)
    _write_f32_colmajor_vec3(os.path.join(bin_dir, "color.bin"), [(0.0, 0.0, 0.0)] * n_vert)

    _write_u64_colmajor_tris(os.path.join(bin_dir, "tri.bin"), target_tris)

    # Optional static collision mesh
    if n_static_vert > 0 and n_static_tri > 0:
        _write_u32(os.path.join(bin_dir, "static_vert_dmap.bin"), [0] * n_static_vert)
        _write_f64_colmajor_vec3(os.path.join(bin_dir, "static_vert.bin"), static_solver_verts)
        _write_u64_colmajor_tris(os.path.join(bin_dir, "static_tri.bin"), static_tris)
        _write_f32_colmajor_vec3(os.path.join(bin_dir, "static_color.bin"), [(0.0, 0.0, 0.0)] * n_static_vert)

    # Element material parameters (constant arrays for now)
    model_map = {
        "arap": 0,
        "stvk": 1,
        "baraff-witkin": 2,
        "snhk": 3,
    }
    tri_model = model_map.get(getattr(settings, "tri_model", "baraff-witkin"), 2)

    tri_density = float(getattr(settings, "tri_density", 1.0))
    tri_young = float(getattr(settings, "tri_young_mod", 100.0))
    tri_poiss = float(getattr(settings, "tri_poiss_rat", 0.35))
    tri_bend = float(getattr(settings, "tri_bend", 2.0))
    tri_shrink = float(getattr(settings, "tri_shrink", 1.0))
    tri_gap = float(getattr(settings, "tri_contact_gap", 1e-3))
    tri_offset = float(getattr(settings, "tri_contact_offset", 0.0))
    tri_strain = float(getattr(settings, "tri_strain_limit", 0.0))
    tri_friction = float(getattr(settings, "tri_friction", 0.0))

    _write_u8(os.path.join(param_dir, "tri-model.bin"), [tri_model] * n_tri)
    _write_f32(os.path.join(param_dir, "tri-density.bin"), [tri_density] * n_tri)
    _write_f32(os.path.join(param_dir, "tri-young-mod.bin"), [tri_young] * n_tri)
    _write_f32(os.path.join(param_dir, "tri-poiss-rat.bin"), [tri_poiss] * n_tri)
    _write_f32(os.path.join(param_dir, "tri-bend.bin"), [tri_bend] * n_tri)
    _write_f32(os.path.join(param_dir, "tri-shrink.bin"), [tri_shrink] * n_tri)
    _write_f32(os.path.join(param_dir, "tri-contact-gap.bin"), [tri_gap] * n_tri)
    _write_f32(os.path.join(param_dir, "tri-contact-offset.bin"), [tri_offset] * n_tri)
    _write_f32(os.path.join(param_dir, "tri-strain-limit.bin"), [tri_strain] * n_tri)
    _write_f32(os.path.join(param_dir, "tri-friction.bin"), [tri_friction] * n_tri)

    if n_static_tri > 0:
        static_gap = float(getattr(settings, "static_contact_gap", 1e-3))
        static_offset = float(getattr(settings, "static_contact_offset", 0.0))
        static_friction = float(getattr(settings, "static_friction", 0.0))
        _write_f32(os.path.join(param_dir, "static-contact-gap.bin"), [static_gap] * n_static_tri)
        _write_f32(os.path.join(param_dir, "static-contact-offset.bin"), [static_offset] * n_static_tri)
        _write_f32(os.path.join(param_dir, "static-friction.bin"), [static_friction] * n_static_tri)

    # Pins (single block from vertex group on target object)
    pin_indices: list[int] = []
    pull_strength = 0.0
    props = getattr(target_obj, "andosim_artezbuild", None)
    if props and bool(getattr(props, "pin_enabled", False)):
        vg_name = (getattr(props, "pin_vertex_group", "") or "").strip()
        if vg_name and vg_name in target_obj.vertex_groups:
            vg_index = target_obj.vertex_groups[vg_name].index
            for v in target_obj.data.vertices:
                w = 0.0
                for g in v.groups:
                    if g.group == vg_index:
                        w = float(g.weight)
                        break
                if w > 0.0:
                    pin_indices.append(int(v.index))
            pull_strength = float(getattr(props, "pin_pull_strength", 0.0))

    # info.toml
    info_path = os.path.join(root, "info.toml")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("[count]\n")
        f.write(f"vert = {n_vert}\n")
        f.write("rod = 0\n")
        f.write(f"tri = {n_tri}\n")
        f.write("tet = 0\n")
        f.write(f"static_vert = {n_static_vert}\n")
        f.write(f"static_tri = {n_static_tri}\n")
        f.write(f"pin_block = {1 if pin_indices else 0}\n")
        f.write("wall = 0\n")
        f.write("sphere = 0\n")
        f.write("stitch = 0\n")
        f.write("rod_vert_start = 0\n")
        f.write("rod_vert_end = 0\n")
        f.write("shell_vert_start = 0\n")
        f.write(f"shell_vert_end = {n_vert}\n")
        f.write("rod_count = 0\n")
        f.write(f"shell_count = {n_tri}\n")

        if pin_indices:
            f.write("\n")
            f.write("[pin-0]\n")
            f.write("operation_count = 0\n")
            f.write(f"pin = {len(pin_indices)}\n")
            f.write(f"pull = {float(pull_strength)}\n")

    # param.toml: must use underscore keys (serde into SimArgs)
    frames = int(context.scene.frame_end - context.scene.frame_start + 1)
    param_path = os.path.join(root, "param.toml")
    dt = float(getattr(settings, "dt", 1e-3))
    fps = float(getattr(settings, "solver_fps", 60.0))
    gravity = float(getattr(settings, "gravity", -9.8))
    with open(param_path, "w", encoding="utf-8") as f:
        f.write("[param]\n")
        f.write(f"disable_contact = false\n")
        f.write(f"keep_states = 10\n")
        f.write(f"keep_verts = 0\n")
        f.write(f"dt = {dt}\n")
        f.write(f"fitting = false\n")
        f.write(f"playback = 1.0\n")
        f.write(f"min_newton_steps = 0\n")
        f.write(f"target_toi = 0.25\n")
        f.write(f"air_friction = 0.2\n")
        f.write(f"line_search_max_t = 1.25\n")
        f.write(f"constraint_ghat = 0.001\n")
        f.write(f"constraint_tol = 0.01\n")
        f.write(f"fps = {fps}\n")
        f.write(f"cg_max_iter = 10000\n")
        f.write(f"cg_tol = 0.001\n")
        f.write(f"ccd_eps = 1e-7\n")
        f.write(f"ccd_reduction = 0.01\n")
        f.write(f"ccd_max_iter = 4096\n")
        f.write(f"max_dx = 1.0\n")
        f.write(f"eiganalysis_eps = 0.01\n")
        f.write(f"friction_eps = 1e-5\n")
        f.write(f"csrmat_max_nnz = 10000000\n")
        f.write(f"bvh_alloc_factor = 2\n")
        f.write(f"frames = {frames}\n")
        f.write(f"auto_save = 0\n")
        f.write('barrier = "cubic"\n')
        f.write(f"stitch_stiffness = 1.0\n")
        f.write(f"air_density = 0.001\n")
        f.write(f"isotropic_air_friction = 0.0\n")
        f.write(f"gravity = {gravity}\n")
        f.write(f"wind = 0.0\n")
        f.write(f"wind_dim = 0\n")
        f.write(f"include_face_mass = false\n")
        f.write(f"fix_xz = 0.0\n")
        f.write(f"fake_crash_frame = -1\n")

    if pin_indices:
        _write_u64(os.path.join(bin_dir, "pin-ind-0.bin"), pin_indices)

    return ExportResult(
        scene_path=root,
        deformable_object_names=[target_obj.name],
        collider_object_names=collider_names,
        deformable_slices=[(target_obj.name, 0, n_vert)],
    )


def export_ppf_scene_from_roles(context, settings) -> ExportResult:
    """Export a PPF scene by reading per-object role tags.

    Deformables are concatenated into one vertex/tri set so they can self-contact and interact.
    Static colliders are concatenated into static_vert/static_tri.
    """

    depsgraph = context.evaluated_depsgraph_get()

    deformables: list[bpy.types.Object] = []
    colliders: list[bpy.types.Object] = []

    for obj in context.scene.objects:
        if obj is None or obj.type != "MESH":
            continue
        props = getattr(obj, "andosim_artezbuild", None)
        if not props or not getattr(props, "enabled", False):
            continue

        role = getattr(props, "role", "DEFORMABLE")
        if role == "DEFORMABLE":
            deformables.append(obj)
        elif role == "STATIC_COLLIDER":
            colliders.append(obj)

    if not deformables:
        raise ValueError("No deformable objects are enabled for PPF")

    # Deterministic order for stable results.
    deformables = sorted(deformables, key=lambda o: o.name)
    colliders = sorted(colliders, key=lambda o: o.name)

    deformable_slices: list[tuple[str, int, int]] = []
    all_def_verts: list[tuple[float, float, float]] = []
    all_def_tris: list[tuple[int, int, int]] = []
    deformable_names: list[str] = []

    # Per-triangle material arrays (match upstream naming: tri-*.bin)
    tri_model_arr: list[int] = []
    tri_density_arr: list[float] = []
    tri_young_arr: list[float] = []
    tri_poiss_arr: list[float] = []
    tri_bend_arr: list[float] = []
    tri_shrink_arr: list[float] = []
    tri_gap_arr: list[float] = []
    tri_offset_arr: list[float] = []
    tri_strain_arr: list[float] = []
    tri_friction_arr: list[float] = []

    # Pins aggregated across deformables (export as one pin block)
    pin_indices: list[int] = []
    pin_pull_strength = 0.0

    # Stitches: list of (src, e0, e1) + (unused, weight)
    stitch_ind: list[tuple[int, int, int]] = []
    stitch_w: list[tuple[float, float]] = []

    for obj in deformables:
        world_verts, tris = _mesh_eval_to_world_tris(obj, depsgraph)
        if not world_verts or not tris:
            raise ValueError(f"Deformable '{obj.name}' has no triangles")
        base = len(all_def_verts)
        all_def_verts.extend(world_verts)
        all_def_tris.extend([(a + base, b + base, c + base) for (a, b, c) in tris])

        # Per-object params override.
        oprops = getattr(obj, "andosim_artezbuild", None)
        use_obj = bool(oprops and getattr(oprops, "use_object_params", False))
        model_map = {"arap": 0, "stvk": 1, "baraff-witkin": 2, "snhk": 3}
        tri_model = model_map.get(
            (getattr(oprops, "tri_model", None) if use_obj else getattr(settings, "tri_model", "baraff-witkin")),
            2,
        )
        tri_density = float(getattr(oprops, "tri_density", 1.0) if use_obj else getattr(settings, "tri_density", 1.0))
        tri_young = float(getattr(oprops, "tri_young_mod", 100.0) if use_obj else getattr(settings, "tri_young_mod", 100.0))
        tri_poiss = float(getattr(oprops, "tri_poiss_rat", 0.35) if use_obj else getattr(settings, "tri_poiss_rat", 0.35))
        tri_bend = float(getattr(oprops, "tri_bend", 2.0) if use_obj else getattr(settings, "tri_bend", 2.0))
        tri_shrink = float(getattr(oprops, "tri_shrink", 1.0) if use_obj else getattr(settings, "tri_shrink", 1.0))
        tri_gap = float(getattr(oprops, "tri_contact_gap", 1e-3) if use_obj else getattr(settings, "tri_contact_gap", 1e-3))
        tri_offset = float(getattr(oprops, "tri_contact_offset", 0.0) if use_obj else getattr(settings, "tri_contact_offset", 0.0))
        tri_strain = float(getattr(oprops, "tri_strain_limit", 0.0) if use_obj else getattr(settings, "tri_strain_limit", 0.0))
        tri_friction = float(getattr(oprops, "tri_friction", 0.0) if use_obj else getattr(settings, "tri_friction", 0.0))

        for _ in range(len(tris)):
            tri_model_arr.append(int(tri_model))
            tri_density_arr.append(float(tri_density))
            tri_young_arr.append(float(tri_young))
            tri_poiss_arr.append(float(tri_poiss))
            tri_bend_arr.append(float(tri_bend))
            tri_shrink_arr.append(float(tri_shrink))
            tri_gap_arr.append(float(tri_gap))
            tri_offset_arr.append(float(tri_offset))
            tri_strain_arr.append(float(tri_strain))
            tri_friction_arr.append(float(tri_friction))

        # Pins (vertex group) -> global indices
        if oprops and bool(getattr(oprops, "pin_enabled", False)):
            vg_name = (getattr(oprops, "pin_vertex_group", "") or "").strip()
            if vg_name and vg_name in obj.vertex_groups:
                vg_index = obj.vertex_groups[vg_name].index
                for v in obj.data.vertices:
                    w = 0.0
                    for g in v.groups:
                        if g.group == vg_index:
                            w = float(g.weight)
                            break
                    if w > 0.0:
                        pin_indices.append(int(base + v.index))
                pin_pull_strength = max(pin_pull_strength, float(getattr(oprops, "pin_pull_strength", 0.0)))

        deformable_slices.append((obj.name, base, len(world_verts)))
        deformable_names.append(obj.name)

    # Second pass: build stitches now that all global vertex offsets are known.
    name_to_offset = {name: int(start) for (name, start, _count) in deformable_slices}

    for obj in deformables:
        oprops = getattr(obj, "andosim_artezbuild", None)
        if not (oprops and bool(getattr(oprops, "stitch_enabled", False))):
            continue
        target_obj = getattr(oprops, "stitch_target_object", None)
        if target_obj is None or target_obj.type != "MESH":
            continue

        tprops = getattr(target_obj, "andosim_artezbuild", None)
        if not (tprops and bool(getattr(tprops, "enabled", False)) and getattr(tprops, "role", "DEFORMABLE") == "DEFORMABLE"):
            continue

        src_offset = name_to_offset.get(obj.name)
        tgt_offset = name_to_offset.get(target_obj.name)
        if src_offset is None or tgt_offset is None:
            continue

        src_group = (getattr(oprops, "stitch_source_vertex_group", "") or "").strip()
        tgt_group = (getattr(oprops, "stitch_target_vertex_group", "") or "").strip()
        max_dist = float(getattr(oprops, "stitch_max_distance", 0.0))
        max_dist2 = max_dist * max_dist

        src_vg_index = None
        if src_group and src_group in obj.vertex_groups:
            src_vg_index = obj.vertex_groups[src_group].index

        tgt_vg_index = None
        if tgt_group and tgt_group in target_obj.vertex_groups:
            tgt_vg_index = target_obj.vertex_groups[tgt_group].index

        if src_vg_index is None:
            continue

        tgt_world_verts, tgt_edges = _mesh_eval_to_world_edges(target_obj, depsgraph)

        if tgt_vg_index is not None:
            in_group = [False] * len(target_obj.data.vertices)
            for v in target_obj.data.vertices:
                w = 0.0
                for g in v.groups:
                    if g.group == tgt_vg_index:
                        w = float(g.weight)
                        break
                if w > 0.0:
                    in_group[int(v.index)] = True
            candidate_edges = [e for e in tgt_edges if in_group[e[0]] and in_group[e[1]]]
        else:
            candidate_edges = tgt_edges

        if not candidate_edges:
            continue

        mw = obj.matrix_world
        for v in obj.data.vertices:
            w = 0.0
            for g in v.groups:
                if g.group == src_vg_index:
                    w = float(g.weight)
                    break
            if w <= 0.0:
                continue

            p = mw @ v.co
            p = (float(p.x), float(p.y), float(p.z))

            best = None
            for e0, e1 in candidate_edges:
                dist2, t = _point_segment_closest(p, tgt_world_verts[e0], tgt_world_verts[e1])
                if best is None or dist2 < best[0]:
                    best = (dist2, int(e0), int(e1), float(t))
            if best is None:
                continue
            dist2, e0, e1, t = best
            if max_dist2 > 0.0 and dist2 > max_dist2:
                continue

            stitch_ind.append((int(src_offset + v.index), int(tgt_offset + e0), int(tgt_offset + e1)))
            stitch_w.append((0.0, float(t)))

    static_world_verts: list[tuple[float, float, float]] = []
    static_tris: list[tuple[int, int, int]] = []
    static_gap_arr: list[float] = []
    static_offset_arr: list[float] = []
    static_friction_arr: list[float] = []
    collider_names: list[str] = []
    for obj in colliders:
        world_verts, tris = _mesh_eval_to_world_tris(obj, depsgraph)
        if not world_verts or not tris:
            continue
        base = len(static_world_verts)
        static_world_verts.extend(world_verts)
        static_tris.extend([(a + base, b + base, c + base) for (a, b, c) in tris])

        oprops = getattr(obj, "andosim_artezbuild", None)
        use_obj = bool(oprops and getattr(oprops, "use_object_params", False))
        gap = float(getattr(oprops, "static_contact_gap", 1e-3) if use_obj else getattr(settings, "static_contact_gap", 1e-3))
        offset = float(getattr(oprops, "static_contact_offset", 0.0) if use_obj else getattr(settings, "static_contact_offset", 0.0))
        friction = float(getattr(oprops, "static_friction", 0.0) if use_obj else getattr(settings, "static_friction", 0.0))
        for _ in range(len(tris)):
            static_gap_arr.append(float(gap))
            static_offset_arr.append(float(offset))
            static_friction_arr.append(float(friction))

        collider_names.append(obj.name)

    # Export using the same low-level writer, but with concatenated arrays.
    target_solver_verts = [blender_to_solver_xyz(*v) for v in all_def_verts]
    target_tris = all_def_tris
    static_solver_verts = [blender_to_solver_xyz(*v) for v in static_world_verts]

    warnings: list[str] = []
    # Guard/warning: upstream CUDA contact code can assert if two deformables start with
    # perfectly coincident points (distance == 0). Detect likely cases and warn early.
    if len(deformable_slices) > 1 and len(target_solver_verts) > 0:
        eps = 1e-9
        key_to_ref: dict[tuple[int, int, int], tuple[str, int]] = {}
        coincident = 0
        samples: list[tuple[str, int, str, int]] = []
        early_stop = False
        for obj_name, start, count in deformable_slices:
            if early_stop:
                break
            for local_i in range(int(count)):
                gi = int(start) + int(local_i)
                sx, sy, sz = target_solver_verts[gi]
                key = (int(round(sx / eps)), int(round(sy / eps)), int(round(sz / eps)))
                ref = key_to_ref.get(key)
                if ref is None:
                    key_to_ref[key] = (obj_name, int(local_i))
                else:
                    other_name, other_local_i = ref
                    if other_name != obj_name:
                        coincident += 1
                        if len(samples) < 3:
                            samples.append((other_name, other_local_i, obj_name, int(local_i)))
                        if coincident >= 1000 and len(samples) >= 3:
                            early_stop = True
                            break

        if coincident > 0:
            example = ""
            if samples:
                a, ai, b, bi = samples[0]
                example = f" (e.g. {a}[{ai}] and {b}[{bi}] share the same position)"
            tail = " (scan stopped early)" if early_stop else ""
            warnings.append(
                "PPF export warning: detected coincident vertices between deformables"
                f"{example}. This can trigger an upstream CUDA contact assertion; separate meshes slightly." + tail
            )

    root = tempfile.mkdtemp(prefix="ppf_scene_", dir=bpy.app.tempdir)
    bin_dir = os.path.join(root, "bin")
    param_dir = os.path.join(bin_dir, "param")
    os.makedirs(param_dir, exist_ok=True)

    n_vert = len(target_solver_verts)
    n_tri = len(target_tris)
    n_static_vert = len(static_solver_verts)
    n_static_tri = len(static_tris)
    n_stitch = len(stitch_ind)

    _write_f64_colmajor_vec3(os.path.join(bin_dir, "displacement.bin"), [(0.0, 0.0, 0.0)])
    _write_u32(os.path.join(bin_dir, "vert_dmap.bin"), [0] * n_vert)

    _write_f64_colmajor_vec3(os.path.join(bin_dir, "vert.bin"), target_solver_verts)
    _write_f32_colmajor_vec3(os.path.join(bin_dir, "vel.bin"), [(0.0, 0.0, 0.0)] * n_vert)
    _write_f32_colmajor_vec3(os.path.join(bin_dir, "color.bin"), [(0.0, 0.0, 0.0)] * n_vert)
    _write_u64_colmajor_tris(os.path.join(bin_dir, "tri.bin"), target_tris)

    if n_static_vert > 0 and n_static_tri > 0:
        _write_u32(os.path.join(bin_dir, "static_vert_dmap.bin"), [0] * n_static_vert)
        _write_f64_colmajor_vec3(os.path.join(bin_dir, "static_vert.bin"), static_solver_verts)
        _write_u64_colmajor_tris(os.path.join(bin_dir, "static_tri.bin"), static_tris)
        _write_f32_colmajor_vec3(
            os.path.join(bin_dir, "static_color.bin"),
            [(0.0, 0.0, 0.0)] * n_static_vert,
        )

    if n_stitch > 0:
        _write_u64_colmajor_tris(os.path.join(bin_dir, "stitch_ind.bin"), stitch_ind)
        _write_f32_colmajor_vec2(os.path.join(bin_dir, "stitch_w.bin"), stitch_w)

    if len(tri_model_arr) != n_tri:
        raise RuntimeError("Internal error: tri param arrays length mismatch")
    _write_u8(os.path.join(param_dir, "tri-model.bin"), tri_model_arr)
    _write_f32(os.path.join(param_dir, "tri-density.bin"), tri_density_arr)
    _write_f32(os.path.join(param_dir, "tri-young-mod.bin"), tri_young_arr)
    _write_f32(os.path.join(param_dir, "tri-poiss-rat.bin"), tri_poiss_arr)
    _write_f32(os.path.join(param_dir, "tri-bend.bin"), tri_bend_arr)
    _write_f32(os.path.join(param_dir, "tri-shrink.bin"), tri_shrink_arr)
    _write_f32(os.path.join(param_dir, "tri-contact-gap.bin"), tri_gap_arr)
    _write_f32(os.path.join(param_dir, "tri-contact-offset.bin"), tri_offset_arr)
    _write_f32(os.path.join(param_dir, "tri-strain-limit.bin"), tri_strain_arr)
    _write_f32(os.path.join(param_dir, "tri-friction.bin"), tri_friction_arr)

    if n_static_tri > 0:
        if len(static_gap_arr) != n_static_tri:
            raise RuntimeError("Internal error: static param arrays length mismatch")
        _write_f32(os.path.join(param_dir, "static-contact-gap.bin"), static_gap_arr)
        _write_f32(os.path.join(param_dir, "static-contact-offset.bin"), static_offset_arr)
        _write_f32(os.path.join(param_dir, "static-friction.bin"), static_friction_arr)

    info_path = os.path.join(root, "info.toml")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("[count]\n")
        f.write(f"vert = {n_vert}\n")
        f.write("rod = 0\n")
        f.write(f"tri = {n_tri}\n")
        f.write("tet = 0\n")
        f.write(f"static_vert = {n_static_vert}\n")
        f.write(f"static_tri = {n_static_tri}\n")
        f.write(f"pin_block = {1 if pin_indices else 0}\n")
        f.write("wall = 0\n")
        f.write("sphere = 0\n")
        f.write(f"stitch = {n_stitch}\n")
        f.write("rod_vert_start = 0\n")
        f.write("rod_vert_end = 0\n")
        f.write("shell_vert_start = 0\n")
        f.write(f"shell_vert_end = {n_vert}\n")
        f.write("rod_count = 0\n")
        f.write(f"shell_count = {n_tri}\n")

        if pin_indices:
            f.write("\n")
            f.write("[pin-0]\n")
            f.write("operation_count = 0\n")
            f.write(f"pin = {len(pin_indices)}\n")
            f.write(f"pull = {float(pin_pull_strength)}\n")

    frames = int(context.scene.frame_end - context.scene.frame_start + 1)
    param_path = os.path.join(root, "param.toml")
    dt = float(getattr(settings, "dt", 1e-3))
    fps = float(getattr(settings, "solver_fps", 60.0))
    gravity = float(getattr(settings, "gravity", -9.8))
    with open(param_path, "w", encoding="utf-8") as f:
        f.write("[param]\n")
        f.write(f"disable_contact = false\n")
        f.write(f"keep_states = 10\n")
        f.write(f"keep_verts = 0\n")
        f.write(f"dt = {dt}\n")
        f.write(f"fitting = false\n")
        f.write(f"playback = 1.0\n")
        f.write(f"min_newton_steps = 0\n")
        f.write(f"target_toi = 0.25\n")
        f.write(f"air_friction = 0.2\n")
        f.write(f"line_search_max_t = 1.25\n")
        f.write(f"constraint_ghat = 0.001\n")
        f.write(f"constraint_tol = 0.01\n")
        f.write(f"fps = {fps}\n")
        f.write(f"cg_max_iter = 10000\n")
        f.write(f"cg_tol = 0.001\n")
        f.write(f"ccd_eps = 1e-7\n")
        f.write(f"ccd_reduction = 0.01\n")
        f.write(f"ccd_max_iter = 4096\n")
        f.write(f"max_dx = 1.0\n")
        f.write(f"eiganalysis_eps = 0.01\n")
        f.write(f"friction_eps = 1e-5\n")
        f.write(f"csrmat_max_nnz = 10000000\n")
        f.write(f"bvh_alloc_factor = 2\n")
        f.write(f"frames = {frames}\n")
        f.write(f"auto_save = 0\n")
        f.write('barrier = "cubic"\n')
        f.write(f"stitch_stiffness = 1.0\n")
        f.write(f"air_density = 0.001\n")
        f.write(f"isotropic_air_friction = 0.0\n")
        f.write(f"gravity = {gravity}\n")
        f.write(f"wind = 0.0\n")
        f.write(f"wind_dim = 0\n")
        f.write(f"include_face_mass = false\n")
        f.write(f"fix_xz = 0.0\n")
        f.write(f"fake_crash_frame = -1\n")

    if pin_indices:
        _write_u64(os.path.join(bin_dir, "pin-ind-0.bin"), pin_indices)

    return ExportResult(
        scene_path=root,
        deformable_object_names=deformable_names,
        collider_object_names=collider_names,
        deformable_slices=deformable_slices,
        warnings=warnings,
    )
