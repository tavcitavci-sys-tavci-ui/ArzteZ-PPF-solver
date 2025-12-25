import bpy
import sys


def _try_import_ppf_backend():
    try:
        import ppf_cts_backend  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        py = sys.version_info
        py_tag = f"cp{py.major}{py.minor}"
        # The bundled wheel in this repo/release is currently built for CPython 3.11.
        # Newer Blender versions may ship a different Python (e.g. cp312), in which case
        # the wheel won't install/import and users end up with a confusing ModuleNotFoundError.
        if (py.major, py.minor) != (3, 11):
            return RuntimeError(
                f"{exc} (Blender Python is {py.major}.{py.minor} / {py_tag}; "
                "this extension bundles a cp311 wheel. Build/bundle a matching wheel "
                "for your Blender Python, or use a Blender build with Python 3.11.)"
            )
        return exc
    return ppf_cts_backend
