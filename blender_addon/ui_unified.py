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

        def _section(parent, prop_name: str, title: str, icon: str):
            box = parent.box()
            row = box.row(align=True)
            opened = bool(getattr(settings, prop_name, False))
            row.prop(settings, prop_name, text="", emboss=False, icon="TRIA_DOWN" if opened else "TRIA_RIGHT")
            row.label(text=title, icon=icon)
            return box, opened

        realtime_box, realtime_open = _section(layout, "ppf_ui_show_realtime", "Realtime", "TIME")
        if realtime_open:
            controls = realtime_box.row(align=True)
            controls.operator("andosim_artezbuild.ppf_run", text="Run", icon="PLAY")
            controls.operator("andosim_artezbuild.ppf_stop", text="Stop", icon="PAUSE")
            reset_row = controls.row(align=True)
            reset_row.enabled = bool(getattr(settings, "ppf_has_snapshot", False)) and not bool(
                getattr(settings, "ppf_baking", False)
            )
            reset_row.operator("andosim_artezbuild.ppf_reset_simulation", text="Reset Simulation", icon="LOOP_BACK")

        bake_box, bake_open = _section(layout, "ppf_ui_show_bake", "Bake", "FILE_TICK")
        if bake_open:
            bake_box.use_property_split = True
            bake_box.use_property_decorate = False
            bake_box.prop(settings, "auto_export")
            bake_box.prop(settings, "use_selected_colliders")
            bake_box.prop(settings, "scene_path")
            bake_box.prop(settings, "output_dir")
            bake_box.prop(settings, "target_object")
            bake_box.prop(settings, "fps")

            bake = bake_box.row(align=True)
            if bool(getattr(settings, "ppf_baking", False)):
                bake.operator("andosim_artezbuild.ppf_cancel_bake", text="Cancel", icon="CANCEL")
            else:
                bake.operator("andosim_artezbuild.ppf_bake_cache", text="Bake Cache", icon="FILE_TICK")

        settings_box, settings_open = _section(layout, "ppf_ui_show_settings", "Settings", "SETTINGS")
        if settings_open:
            box = settings_box.box()
            box.label(text="Material Preset", icon="MATERIAL")
            box.prop(settings, "ppf_material_preset")

            sim_box, sim_open = _section(settings_box, "ppf_ui_show_sim", "Simulation", "TIME")
            if sim_open:
                sim_box.prop(settings, "dt")
                sim_box.prop(settings, "solver_fps")
                sim_box.prop(settings, "gravity")

            shell_box, shell_open = _section(settings_box, "ppf_ui_show_shell", "Shell", "MOD_CLOTH")
            if shell_open:
                shell_box.prop(settings, "tri_model")
                shell_box.prop(settings, "tri_density")
                shell_box.prop(settings, "tri_young_mod")
                shell_box.prop(settings, "tri_poiss_rat")
                shell_box.prop(settings, "tri_bend")
                shell_box.prop(settings, "tri_shrink")
                shell_box.prop(settings, "tri_contact_gap")
                shell_box.prop(settings, "tri_contact_offset")
                shell_box.prop(settings, "tri_strain_limit")
                shell_box.prop(settings, "tri_friction")

            stat_box, stat_open = _section(settings_box, "ppf_ui_show_static", "Static Collider", "CUBE")
            if stat_open:
                stat_box.prop(settings, "static_contact_gap")
                stat_box.prop(settings, "static_contact_offset")
                stat_box.prop(settings, "static_friction")

            obj_box, obj_open = _section(settings_box, "ppf_ui_show_active_object", "Active Object", "OUTLINER_OB_MESH")
            if obj_open:
                obj = context.object
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

                override_box, override_open = _section(obj_box, "ppf_ui_show_object_override", "Object Override", "MODIFIER")
                if override_open:
                    override_box.prop(props, "use_object_params")
                    if bool(getattr(props, "use_object_params", False)):
                        if role == "DEFORMABLE":
                            o = override_box.box()
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
                            o = override_box.box()
                            o.label(text="Static Collider Override", icon="CUBE")
                            o.prop(props, "static_contact_gap")
                            o.prop(props, "static_contact_offset")
                            o.prop(props, "static_friction")

                # Pins / stitches only apply to deformables.
                if role == "DEFORMABLE":
                    pins_box, pins_open = _section(obj_box, "ppf_ui_show_pins", "Pins", "PINNED")
                    if pins_open:
                        row = pins_box.row(align=True)
                        row.operator("andosim_artezbuild.ppf_create_pin_handle", text="Create Pin Handle")
                        row.operator("andosim_artezbuild.ppf_clear_grab", text="Clear Handles")

                        prep_box, prep_open = _section(pins_box, "ppf_ui_show_cloth_prep", "Cloth Prep", "MOD_REMESH")
                        if prep_open:
                            prep_box.prop(settings, "cloth_prep_voxel_size")
                            prep_box.prop(settings, "cloth_prep_adaptivity")
                            prep_box.prop(settings, "cloth_prep_merge_distance")
                            prep_box.prop(settings, "cloth_prep_auto_voxel_from_shell")
                            prep_box.prop(settings, "cloth_prep_report_mesh_quality")
                            prep_box.prop(settings, "cloth_prep_contact_gap_mode")
                            if getattr(settings, "cloth_prep_contact_gap_mode", "OFF") != "OFF":
                                prep_box.prop(settings, "cloth_prep_contact_gap_factor")
                            prep_box.prop(settings, "cloth_prep_dt_mode")
                            if getattr(settings, "cloth_prep_dt_mode", "OFF") != "OFF":
                                prep_box.prop(settings, "cloth_prep_dt_max_gravity_disp_frac")

                            run_row = prep_box.row(align=True)
                            op = run_row.operator(
                                "andosim_artezbuild.prepare_cloth_mesh",
                                text="Prepare Cloth Mesh",
                                icon="MOD_REMESH",
                            )
                            # Drive the operator from persistent UI settings.
                            try:
                                op.voxel_size = float(getattr(settings, "cloth_prep_voxel_size", 0.0))
                                op.adaptivity = float(getattr(settings, "cloth_prep_adaptivity", 0.0))
                                op.merge_distance = float(getattr(settings, "cloth_prep_merge_distance", 1.0e-6))
                                op.auto_voxel_from_shell = bool(getattr(settings, "cloth_prep_auto_voxel_from_shell", True))
                                op.report_mesh_quality = bool(getattr(settings, "cloth_prep_report_mesh_quality", True))
                                op.contact_gap_mode = str(getattr(settings, "cloth_prep_contact_gap_mode", "SUGGEST"))
                                op.contact_gap_factor = float(getattr(settings, "cloth_prep_contact_gap_factor", 0.1))
                                op.dt_mode = str(getattr(settings, "cloth_prep_dt_mode", "SUGGEST"))
                                op.dt_max_gravity_disp_frac = float(getattr(settings, "cloth_prep_dt_max_gravity_disp_frac", 0.1))
                            except Exception:
                                pass

                        pins_box.prop(props, "pin_enabled")
                        if bool(getattr(props, "pin_enabled", False)):
                            pins_box.prop(props, "pin_vertex_group")
                            pins_box.prop(props, "pin_pull_strength")

                        pins_box.separator(factor=0.5)
                        pins_box.prop(props, "attach_enabled")
                        if bool(getattr(props, "attach_enabled", False)):
                            pins_box.prop(props, "attach_target_object")
                            pins_box.prop(props, "attach_vertex_group")
                            pins_box.prop(props, "attach_pull_strength")

                        stitch = pins_box.box()
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
