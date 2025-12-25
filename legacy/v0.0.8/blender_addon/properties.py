import bpy


_PPF_MATERIAL_PRESETS = {
    "CLOTH": {
        "label": "Cloth",
        "tri_density": 1100.0,
        "tri_young_mod": 3.0e5,
        "tri_poiss_rat": 0.35,
        "tri_bend": 2.0,
        "tri_shrink": 1.0,
        "tri_friction": 0.4,
        "static_friction": 0.4,
    },
    "RUBBER": {
        "label": "Rubber",
        "tri_density": 1200.0,
        "tri_young_mod": 2.5e6,
        "tri_poiss_rat": 0.45,
        "tri_bend": 4.0,
        "tri_shrink": 1.0,
        "tri_friction": 0.8,
        "static_friction": 0.8,
    },
    "LEATHER": {
        "label": "Leather",
        "tri_density": 950.0,
        "tri_young_mod": 5.0e6,
        "tri_poiss_rat": 0.40,
        "tri_bend": 3.0,
        "tri_shrink": 1.0,
        "tri_friction": 0.6,
        "static_friction": 0.6,
    },
    "SILK": {
        "label": "Silk",
        "tri_density": 1300.0,
        "tri_young_mod": 5.0e5,
        "tri_poiss_rat": 0.35,
        "tri_bend": 1.0,
        "tri_shrink": 1.0,
        "tri_friction": 0.2,
        "static_friction": 0.2,
    },
    "METAL": {
        "label": "Metal",
        "tri_density": 7800.0,
        "tri_young_mod": 5.0e8,
        "tri_poiss_rat": 0.30,
        "tri_bend": 10.0,
        "tri_shrink": 1.0,
        "tri_friction": 0.3,
        "static_friction": 0.3,
    },
}


def _apply_ppf_material_preset(self, _context):
    preset_key = getattr(self, "ppf_material_preset", "CUSTOM")
    if preset_key == "CUSTOM":
        return

    preset = _PPF_MATERIAL_PRESETS.get(preset_key)
    if not preset:
        return

    for key, value in preset.items():
        if key == "label":
            continue
        if hasattr(self, key):
            setattr(self, key, value)


class AndoSimArtezbuildSettings(bpy.types.PropertyGroup):
    ppf_material_preset: bpy.props.EnumProperty(
        name="Preset",
        description="Quick material presets for PPF shell/static friction parameters",
        items=[
            (key, data["label"], "") for key, data in _PPF_MATERIAL_PRESETS.items()
        ]
        + [("CUSTOM", "Custom", "Do not override values")],
        default="CUSTOM",
        update=_apply_ppf_material_preset,
    )

    auto_export: bpy.props.BoolProperty(
        name="Auto Export",
        description="When enabled, export the current Blender scene to a temporary PPF scene folder on Run",
        default=True,
    )

    use_selected_colliders: bpy.props.BoolProperty(
        name="Use Selected Colliders",
        description="Treat other selected mesh objects (besides Target) as static collision meshes",
        default=True,
    )

    scene_path: bpy.props.StringProperty(
        name="PPF Scene",
        description="Path to a PPF scene folder",
        subtype="DIR_PATH",
        default="",
    )

    output_dir: bpy.props.StringProperty(
        name="Output Dir",
        description="Output directory used by the backend (logs, etc.)",
        subtype="DIR_PATH",
        default="",
    )

    target_object: bpy.props.PointerProperty(
        name="Target",
        description="Mesh object to drive with the solver",
        type=bpy.types.Object,
    )

    dt: bpy.props.FloatProperty(
        name="dt",
        description="Solver timestep (seconds)",
        default=1e-3,
        min=1e-6,
        max=1.0,
        precision=6,
    )

    solver_fps: bpy.props.FloatProperty(
        name="Solver FPS",
        description="PPF internal FPS (used for frame/BVH cadence)",
        default=60.0,
        min=1.0,
        max=240.0,
    )

    gravity: bpy.props.FloatProperty(
        name="Gravity",
        description="Gravity magnitude (mapped into PPF's gravity axis)",
        default=-9.8,
        soft_min=-50.0,
        soft_max=0.0,
    )

    tri_model: bpy.props.EnumProperty(
        name="Tri Model",
        description="Deformation model for shell triangles",
        items=[
            ("arap", "arap", ""),
            ("stvk", "stvk", ""),
            ("baraff-witkin", "baraff-witkin", ""),
            ("snhk", "snhk", ""),
        ],
        default="baraff-witkin",
    )

    tri_density: bpy.props.FloatProperty(
        name="Tri Density",
        description="Shell density (per area)",
        default=1.0,
        min=1e-8,
    )
    tri_young_mod: bpy.props.FloatProperty(
        name="Tri Young's Mod",
        description="Shell Young's modulus",
        default=100.0,
        min=1e-8,
    )
    tri_poiss_rat: bpy.props.FloatProperty(
        name="Tri Poisson",
        description="Shell Poisson ratio",
        default=0.35,
        min=1e-6,
        max=0.499,
    )
    tri_bend: bpy.props.FloatProperty(
        name="Tri Bend",
        description="Shell bending stiffness",
        default=2.0,
        min=0.0,
    )
    tri_shrink: bpy.props.FloatProperty(
        name="Tri Shrink",
        description="Shell shrink factor (<= 1.0)",
        default=1.0,
        min=1e-6,
        max=1.0,
    )
    tri_contact_gap: bpy.props.FloatProperty(
        name="Tri Contact Gap",
        description="Contact gap distance",
        default=1e-3,
        min=1e-9,
    )
    tri_contact_offset: bpy.props.FloatProperty(
        name="Tri Contact Offset",
        description="Contact offset distance",
        default=0.0,
    )
    tri_strain_limit: bpy.props.FloatProperty(
        name="Tri Strain Limit",
        description="Strain limit (0 disables)",
        default=0.0,
        min=0.0,
    )
    tri_friction: bpy.props.FloatProperty(
        name="Tri Friction",
        description="Friction coefficient",
        default=0.0,
        min=0.0,
    )

    static_contact_gap: bpy.props.FloatProperty(
        name="Static Contact Gap",
        description="Static collider contact gap",
        default=1e-3,
        min=1e-9,
    )
    static_contact_offset: bpy.props.FloatProperty(
        name="Static Contact Offset",
        description="Static collider contact offset",
        default=0.0,
    )
    static_friction: bpy.props.FloatProperty(
        name="Static Friction",
        description="Static collider friction coefficient",
        default=0.0,
        min=0.0,
    )

    fps: bpy.props.IntProperty(
        name="FPS",
        description="Simulation step frequency (timer interval)",
        default=30,
        min=1,
        max=240,
    )

    running: bpy.props.BoolProperty(
        name="Running",
        description="Internal state flag",
        default=False,
        options={"HIDDEN"},
    )
