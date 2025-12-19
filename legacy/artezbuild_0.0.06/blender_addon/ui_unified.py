import bpy

from . import get_backend


_BACKEND_ANDO = "ANDO"
_BACKEND_PPF = "PPF"


class ANDOSIM_ARTEZBUILD_PT_main(bpy.types.Panel):
    bl_label = "AndoSim ArteZbuild"
    bl_idname = "ANDOSIM_ARTEZBUILD_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AndoSim"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        backend = get_backend(context)
        backend_label = "Ando Core" if backend == _BACKEND_ANDO else "PPF Contact Solver"

        box = layout.box()
        row = box.row(align=True)
        row.label(text="Solver Backend", icon="MOD_PHYSICS")
        row.label(text=backend_label)

        # Note: detailed UI lives in the vendored AndoSim panels (ANDO_PT_*),
        # and in the PPF panel below.


class ANDOSIM_ARTEZBUILD_PT_backend(bpy.types.Panel):
    bl_label = "Backend"
    bl_idname = "ANDOSIM_ARTEZBUILD_PT_backend"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AndoSim"
    bl_parent_id = "ANDOSIM_ARTEZBUILD_PT_main"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        addon = None
        try:
            addon = context.preferences.addons.get(__package__)
        except Exception:
            addon = None

        if not addon or not getattr(addon, "preferences", None):
            layout.label(text="Add-on preferences not available", icon="ERROR")
            return

        layout.prop(addon.preferences, "solver_backend")


class ANDOSIM_ARTEZBUILD_PT_ppf_panel(bpy.types.Panel):
    bl_label = "PPF"
    bl_idname = "ANDOSIM_ARTEZBUILD_PT_ppf_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AndoSim"
    bl_parent_id = "ANDOSIM_ARTEZBUILD_PT_main"

    @classmethod
    def poll(cls, context):
        return get_backend(context) == _BACKEND_PPF

    def draw(self, context):
        from .ui import _try_import_ppf_backend

        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        settings = getattr(context.scene, "andosim_artezbuild", None)
        if settings is None:
            layout.label(text="PPF settings not initialized", icon="ERROR")
            return

        backend_or_exc = _try_import_ppf_backend()
        status = layout.box()
        status.label(text="Backend", icon="SETTINGS")
        if isinstance(backend_or_exc, Exception):
            row = status.row()
            row.alert = True
            row.label(text=f"ppf_cts_backend missing: {backend_or_exc}", icon="ERROR")
        else:
            version = getattr(backend_or_exc, "__version__", "?")
            status.label(text=f"ppf_cts_backend OK: {version}", icon="CHECKMARK")

        controls = layout.row(align=True)
        controls.operator("andosim_artezbuild.ppf_run", text="Run", icon="PLAY")
        controls.operator("andosim_artezbuild.ppf_stop", text="Stop", icon="PAUSE")

        general = layout.box()
        general.label(text="Scene", icon="SCENE_DATA")
        general.prop(settings, "auto_export")
        general.prop(settings, "use_selected_colliders")
        general.prop(settings, "scene_path")
        general.prop(settings, "output_dir")
        general.prop(settings, "target_object")
        general.prop(settings, "fps")

        box = layout.box()
        box.label(text="Material Preset", icon="MATERIAL")
        box.prop(settings, "ppf_material_preset")

        sim = layout.box()
        sim.label(text="Simulation", icon="TIME")
        sim.prop(settings, "dt")
        sim.prop(settings, "solver_fps")
        sim.prop(settings, "gravity")

        shell = layout.box()
        shell.label(text="Shell", icon="MOD_CLOTH")
        shell.prop(settings, "tri_model")
        shell.prop(settings, "tri_density")
        shell.prop(settings, "tri_young_mod")
        shell.prop(settings, "tri_poiss_rat")
        shell.prop(settings, "tri_bend")
        shell.prop(settings, "tri_shrink")
        shell.prop(settings, "tri_contact_gap")
        shell.prop(settings, "tri_contact_offset")
        shell.prop(settings, "tri_strain_limit")
        shell.prop(settings, "tri_friction")

        stat = layout.box()
        stat.label(text="Static Collider", icon="CUBE")
        stat.prop(settings, "static_contact_gap")
        stat.prop(settings, "static_contact_offset")
        stat.prop(settings, "static_friction")

        obj = context.object
        obj_box = layout.box()
        obj_box.label(text="Active Object", icon="OUTLINER_OB_MESH")
        if obj is None:
            obj_box.label(text="Select a mesh object to edit PPF role/pins", icon="INFO")
            return
        if obj.type != "MESH":
            obj_box.label(text=f"Active object '{obj.name}' is not a mesh", icon="INFO")
            return

        props = getattr(obj, "andosim_artezbuild", None)
        if props is None:
            obj_box.label(text="Object settings not initialized", icon="ERROR")
            return

        col = obj_box.column(align=True)
        col.prop(props, "enabled")
        row = col.row(align=True)
        row.enabled = bool(getattr(props, "enabled", False))
        row.prop(props, "role")

        if not bool(getattr(props, "enabled", False)) or getattr(props, "role", "IGNORE") == "IGNORE":
            obj_box.label(text="Enable the object to include it in export", icon="INFO")
            return

        role = getattr(props, "role", "DEFORMABLE")

        # Per-object overrides
        obj_box.separator()
        obj_box.prop(props, "use_object_params")
        if bool(getattr(props, "use_object_params", False)):
            if role == "DEFORMABLE":
                o = obj_box.box()
                o.label(text="Deformable Override", icon="MOD_CLOTH")
                o.prop(props, "tri_model")
                o.prop(props, "tri_density")
                o.prop(props, "tri_young_mod")
                o.prop(props, "tri_poiss_rat")
                o.prop(props, "tri_bend")
                o.prop(props, "tri_shrink")
                o.prop(props, "tri_contact_gap")
                o.prop(props, "tri_contact_offset")
                o.prop(props, "tri_strain_limit")
                o.prop(props, "tri_friction")
            elif role == "STATIC_COLLIDER":
                o = obj_box.box()
                o.label(text="Static Collider Override", icon="CUBE")
                o.prop(props, "static_contact_gap")
                o.prop(props, "static_contact_offset")
                o.prop(props, "static_friction")

        # Pins / stitches only apply to deformables.
        if role == "DEFORMABLE":
            pins = obj_box.box()
            pins.label(text="Pins", icon="PINNED")
            pins.prop(props, "pin_enabled")
            if bool(getattr(props, "pin_enabled", False)):
                pins.prop(props, "pin_vertex_group")
                pins.prop(props, "pin_pull_strength")

            stitch = obj_box.box()
            stitch.label(text="Stitches", icon="LINKED")
            stitch.prop(props, "stitch_enabled")
            if bool(getattr(props, "stitch_enabled", False)):
                stitch.prop(props, "stitch_target_object")
                stitch.prop(props, "stitch_source_vertex_group")
                stitch.prop(props, "stitch_target_vertex_group")
                stitch.prop(props, "stitch_max_distance")


_classes = (
    ANDOSIM_ARTEZBUILD_PT_main,
    ANDOSIM_ARTEZBUILD_PT_backend,
    ANDOSIM_ARTEZBUILD_PT_ppf_panel,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
