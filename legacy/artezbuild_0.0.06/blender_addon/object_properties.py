import bpy


class AndoSimArtezbuildObjectSettings(bpy.types.PropertyGroup):
    enabled: bpy.props.BoolProperty(
        name="Enable PPF",
        description="Include this object in the PPF export",
        default=False,
    )

    role: bpy.props.EnumProperty(
        name="Role",
        description="How this object participates in the PPF simulation",
        items=[
            ("DEFORMABLE", "Deformable", "Simulated deformable shell"),
            ("STATIC_COLLIDER", "Static Collider", "Static collision mesh"),
            ("IGNORE", "Ignore", "Ignored by exporter"),
        ],
        default="DEFORMABLE",
    )

    use_object_params: bpy.props.BoolProperty(
        name="Override Scene Params",
        description="Use per-object material/collision parameters instead of the scene defaults",
        default=False,
    )

    tri_model: bpy.props.EnumProperty(
        name="Shell Model",
        description="Constitutive model for this object's shell triangles",
        items=[
            ("arap", "ARAP", "As-rigid-as-possible"),
            ("stvk", "StVK", "St. Venant-Kirchhoff"),
            ("baraff-witkin", "Baraff-Witkin", "Baraff-Witkin shell"),
            ("snhk", "SNHK", "Stable Neo-Hookean"),
        ],
        default="baraff-witkin",
    )

    tri_density: bpy.props.FloatProperty(name="Density", default=1.0, min=0.0)
    tri_young_mod: bpy.props.FloatProperty(name="Young's Modulus", default=100.0, min=0.0)
    tri_poiss_rat: bpy.props.FloatProperty(name="Poisson Ratio", default=0.35, min=0.0, max=0.499)
    tri_bend: bpy.props.FloatProperty(name="Bend", default=2.0, min=0.0)
    tri_shrink: bpy.props.FloatProperty(name="Shrink", default=1.0, min=0.0)
    tri_contact_gap: bpy.props.FloatProperty(name="Contact Gap", default=1e-3, min=0.0)
    tri_contact_offset: bpy.props.FloatProperty(name="Contact Offset", default=0.0)
    tri_strain_limit: bpy.props.FloatProperty(name="Strain Limit", default=0.0, min=0.0)
    tri_friction: bpy.props.FloatProperty(name="Friction", default=0.0, min=0.0)

    static_contact_gap: bpy.props.FloatProperty(name="Static Contact Gap", default=1e-3, min=0.0)
    static_contact_offset: bpy.props.FloatProperty(name="Static Contact Offset", default=0.0)
    static_friction: bpy.props.FloatProperty(name="Static Friction", default=0.0, min=0.0)

    pin_enabled: bpy.props.BoolProperty(
        name="Pins",
        description="Export pinned vertices from the specified vertex group as upstream pin constraints",
        default=False,
    )

    pin_vertex_group: bpy.props.StringProperty(
        name="Pin Vertex Group",
        description="Vertex group name to use for pins (weight > 0)",
        default="PPF_PIN",
    )

    pin_pull_strength: bpy.props.FloatProperty(
        name="Pull Strength",
        description="If > 0, pins act as pull constraints; if 0, pins are fixed",
        default=0.0,
        min=0.0,
    )

    stitch_enabled: bpy.props.BoolProperty(
        name="Stitches",
        description="Export upstream stitch constraints from this object to a target deformable",
        default=False,
    )

    stitch_target_object: bpy.props.PointerProperty(
        name="Stitch Target",
        description="Target deformable object whose edges will be stitched to",
        type=bpy.types.Object,
    )

    stitch_source_vertex_group: bpy.props.StringProperty(
        name="Source Group",
        description="Vertex group on this object (weight > 0) defining stitch source vertices",
        default="PPF_STITCH",
    )

    stitch_target_vertex_group: bpy.props.StringProperty(
        name="Target Group",
        description="Optional vertex group on target object (weight > 0) to restrict candidate edges; leave empty for all",
        default="",
    )

    stitch_max_distance: bpy.props.FloatProperty(
        name="Max Distance",
        description="Maximum world-space distance for creating a stitch; pairs beyond this are skipped",
        default=0.05,
        min=0.0,
    )
