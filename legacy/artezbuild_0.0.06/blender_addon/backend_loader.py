def import_backend():
    # Blender Extensions installs wheels declared in blender_manifest.toml.
    # Import directly without sys.path manipulation.
    import ppf_cts_backend  # type: ignore

    return ppf_cts_backend
