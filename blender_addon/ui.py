import bpy


def _try_import_ppf_backend():
    try:
        import ppf_cts_backend  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        return exc
    return ppf_cts_backend
