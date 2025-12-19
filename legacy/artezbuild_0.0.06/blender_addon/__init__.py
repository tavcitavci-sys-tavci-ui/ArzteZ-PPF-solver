bl_info = {
    "name": "AndoSim ArteZbuild",
    "author": "Moritz",
    "version": (0, 0, 6),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > AndoSim",
    "description": "Unified Ando (CPU) + PPF (GPU) simulation (PPF core integrated)",
    "category": "Physics",
}

import bpy

from .operators import ANDOSIM_ARTEZBUILD_OT_ppf_run, ANDOSIM_ARTEZBUILD_OT_ppf_stop
from .object_properties import AndoSimArtezbuildObjectSettings
from .properties import AndoSimArtezbuildSettings


_BACKEND_ANDO = "ANDO"
_BACKEND_PPF = "PPF"


class AndoSimArtezbuildPrefs(bpy.types.AddonPreferences):
    bl_idname = __package__

    solver_backend: bpy.props.EnumProperty(
        name="Solver backend",
        items=[(_BACKEND_ANDO, "Ando Core", ""), (_BACKEND_PPF, "PPF Contact Solver", "")],
        default=_BACKEND_PPF,
    )

    def draw(self, ctx):
        layout = self.layout
        layout.prop(self, "solver_backend")


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
    AndoSimArtezbuildObjectSettings,
    AndoSimArtezbuildSettings,
)


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


def unregister():
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
