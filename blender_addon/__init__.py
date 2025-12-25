bl_info = {
    "name": "ArzteZ PPF solver",
    "author": "Moritz",
    "version": (0, 0, 9, 3),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > AndoSim",
    "description": "Unified Ando (CPU) + PPF (GPU) simulation (PPF core integrated)",
    "category": "Physics",
}

import bpy

from .operators import (
    ANDOSIM_ARTEZBUILD_OT_ppf_run,
    ANDOSIM_ARTEZBUILD_OT_ppf_stop,
    ANDOSIM_ARTEZBUILD_OT_ppf_reset_simulation,
    ANDOSIM_ARTEZBUILD_OT_ppf_bake_cache,
    ANDOSIM_ARTEZBUILD_OT_ppf_cancel_bake,
    ANDOSIM_ARTEZBUILD_OT_ppf_create_pin_handle,
    ANDOSIM_ARTEZBUILD_OT_ppf_grab_pin,
    ANDOSIM_ARTEZBUILD_OT_ppf_clear_grab,
    ANDOSIM_ARTEZBUILD_OT_prepare_cloth_mesh,
)
from .object_properties import AndoSimArtezbuildObjectSettings
from .properties import AndoSimArtezbuildSettings


_BACKEND_ANDO = "ANDO"
_BACKEND_PPF = "PPF"

_CLOTH_PREP_DUPLICATE = "DUPLICATE"
_CLOTH_PREP_DESTRUCTIVE = "DESTRUCTIVE"


class AndoSimArtezbuildPrefs(bpy.types.AddonPreferences):
    bl_idname = __package__

    solver_backend: bpy.props.EnumProperty(
        name="Solver backend",
        items=[(_BACKEND_ANDO, "Ando Core", ""), (_BACKEND_PPF, "PPF Contact Solver", "")],
        default=_BACKEND_PPF,
    )

    cloth_prep_mode: bpy.props.EnumProperty(
        name="Cloth Prep Mode",
        description="Whether 'Prepare Cloth Mesh' duplicates the object (recommended) or modifies it in-place",
        items=[
            (_CLOTH_PREP_DUPLICATE, "Duplicate (Sim Workflow)", "Create a new *_SIM mesh and keep the original"),
            (_CLOTH_PREP_DESTRUCTIVE, "Destructive", "Remesh the active object in-place"),
        ],
        default=_CLOTH_PREP_DUPLICATE,
    )

    def draw(self, ctx):
        layout = self.layout
        layout.prop(self, "solver_backend")

        box = layout.box()
        box.label(text="Cloth Mesh Prep")
        box.prop(self, "cloth_prep_mode")


def get_backend(ctx) -> str:
    try:
        addon = ctx.preferences.addons.get(__package__)
    except AttributeError:
        return _BACKEND_PPF
    if not addon:
        return _BACKEND_PPF
    return getattr(addon.preferences, "solver_backend", _BACKEND_PPF)


from . import ui_unified

from .ando import properties as ando_properties
from .ando import ui as ando_ui
from .ando import operators as ando_operators
from .ando import parameter_update as ando_parameter_update

_classes = (
    AndoSimArtezbuildPrefs,
    ANDOSIM_ARTEZBUILD_OT_ppf_run,
    ANDOSIM_ARTEZBUILD_OT_ppf_stop,
    ANDOSIM_ARTEZBUILD_OT_ppf_reset_simulation,
    ANDOSIM_ARTEZBUILD_OT_ppf_bake_cache,
    ANDOSIM_ARTEZBUILD_OT_ppf_cancel_bake,
    ANDOSIM_ARTEZBUILD_OT_ppf_create_pin_handle,
    ANDOSIM_ARTEZBUILD_OT_ppf_grab_pin,
    ANDOSIM_ARTEZBUILD_OT_ppf_clear_grab,
    ANDOSIM_ARTEZBUILD_OT_prepare_cloth_mesh,
    AndoSimArtezbuildObjectSettings,
    AndoSimArtezbuildSettings,
)


_addon_keymaps = []


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.andosim_artezbuild = bpy.props.PointerProperty(type=AndoSimArtezbuildSettings)
    bpy.types.Object.andosim_artezbuild = bpy.props.PointerProperty(type=AndoSimArtezbuildObjectSettings)

    ui_unified.register()

    # Register vendored AndoSim functionality under its original property names.
    ando_properties.register()
    ando_operators.register()
    ando_parameter_update.register()
    ando_ui.register()

    # Keymaps in 3D View:
    # - Ctrl+G: start Grab Pin
    # - Ctrl+Alt+Shift+LMB: clear grab
    wm = bpy.context.window_manager
    kc = getattr(wm.keyconfigs, "addon", None)
    if kc is not None:
        km = kc.keymaps.new(name="3D View", space_type="VIEW_3D")

        kmi = km.keymap_items.new(
            "andosim_artezbuild.ppf_create_pin_handle",
            type="G",
            value="PRESS",
            ctrl=True,
        )
        _addon_keymaps.append((km, kmi))

        kmi = km.keymap_items.new(
            "andosim_artezbuild.ppf_clear_grab",
            type="LEFTMOUSE",
            value="PRESS",
            ctrl=True,
            alt=True,
            shift=True,
        )
        _addon_keymaps.append((km, kmi))


def unregister():
    # Remove keymaps first.
    global _addon_keymaps
    for km, kmi in _addon_keymaps:
        try:
            km.keymap_items.remove(kmi)
        except Exception:
            pass
    _addon_keymaps.clear()

    try:
        ui_unified.unregister()
    except Exception:
        pass
    # Unregister vendored AndoSim functionality first (it may reference Blender types).
    try:
        ando_ui.unregister()
    except Exception:
        pass
    try:
        ando_parameter_update.unregister()
    except Exception:
        pass
    try:
        ando_operators.unregister()
    except Exception:
        pass
    try:
        ando_properties.unregister()
    except Exception:
        pass

    if hasattr(bpy.types.Scene, "andosim_artezbuild"):
        del bpy.types.Scene.andosim_artezbuild

    if hasattr(bpy.types.Object, "andosim_artezbuild"):
        del bpy.types.Object.andosim_artezbuild

    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
