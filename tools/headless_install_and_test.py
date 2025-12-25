import argparse
import math
import os
import sys
import tempfile
from pathlib import Path


def _require(obj, attr: str, label: str):
	if not hasattr(obj, attr):
		raise RuntimeError(f"Missing expected attribute: {label}.{attr}")


def _finite(values, label: str):
	for i, v in enumerate(values):
		if not math.isfinite(float(v)):
			raise RuntimeError(f"Non-finite {label}[{i}]={v}")


def _install_extension_zip(zip_path: str) -> None:
	import bpy  # type: ignore

	zp = str(Path(zip_path).expanduser().resolve())
	if not Path(zp).exists():
		raise FileNotFoundError(zp)

	# Blender requires 'online access' enabled for Extensions operations
	# even when installing from a local zip.
	try:
		bpy.context.preferences.system.use_online_access = True
	except Exception as exc:
		print("[INSTALL] Warning: could not enable online access:", repr(exc))

	# Target the user extensions repository explicitly.
	repo_dir = None
	try:
		ext = bpy.context.preferences.extensions
		for r in getattr(ext, "repos", []) or []:
			if getattr(r, "module", "") == "user_default":
				repo_dir = str(getattr(r, "directory", "") or "")
				break
	except Exception:
		repo_dir = None
	if not repo_dir:
		try:
			import bpy.utils  # type: ignore

			base = bpy.utils.user_resource("EXTENSIONS")
			if base:
				repo_dir = str(Path(base) / "user_default")
		except Exception:
			repo_dir = None

	file_url = Path(zp).as_uri()
	print("[INSTALL] Installing extension from:", zp)
	print("[INSTALL] Using repo_directory:", repo_dir or "<auto>")

	# In some Blender versions/background contexts, package_install is strict about URL parsing.
	# Prefer package_install for proper manifest/id handling, but fall back to package_install_files.
	res = None
	try:
		kwargs = {
			"url": file_url,
			"pkg_id": "andosim_artezbuild",
			"enable_on_install": True,
			"do_legacy_replace": True,
		}
		if repo_dir:
			kwargs["repo_directory"] = repo_dir
		res = bpy.ops.extensions.package_install(**kwargs)
		print("[INSTALL] package_install result:", res)
	except Exception as exc:
		print("[INSTALL] package_install failed; trying package_install_files:", repr(exc))
		kwargs2 = {
			"filepath": zp,
			"enable_on_install": True,
			"overwrite": True,
		}
		try:
			# Prefer explicit user repo if available.
			kwargs2["repo"] = "user_default"
		except Exception:
			pass
		res = bpy.ops.extensions.package_install_files(**kwargs2)
		print("[INSTALL] package_install_files result:", res)

	try:
		bpy.ops.wm.save_userpref()
	except Exception as exc:
		print("[INSTALL] Warning: failed to save user prefs:", repr(exc))


def _ensure_addon_registered(module_name: str):
	import bpy  # type: ignore

	# In Blender background mode, the extensions repo is not always appended to sys.path.
	# Add it explicitly so we can import the installed extension as Blender would in the UI.
	try:
		import bpy.utils  # type: ignore

		base = bpy.utils.user_resource("EXTENSIONS")
		if base:
			for p in (base, str(Path(base) / "user_default"), str(Path(base) / "user_default" / module_name)):
				if p and p not in sys.path:
					sys.path.insert(0, p)
	except Exception as exc:
		print("[INSTALL] Warning: could not add extensions to sys.path:", repr(exc))

	mod = __import__(module_name)
	if not hasattr(bpy.types.Scene, "andosim_artezbuild"):
		print("[INSTALL] Scene properties missing; calling register()")
		mod.register()
	return mod


def _smoke_ppf_step(outdir: str, module_name: str) -> None:
	import bpy  # type: ignore

	bpy.ops.wm.read_factory_settings(use_empty=True)

	# Factory settings can disable/unregister addons; re-register ours.
	_ensure_addon_registered(module_name)

	bpy.ops.mesh.primitive_grid_add(x_subdivisions=25, y_subdivisions=25, size=1.0)
	deform = bpy.context.active_object
	deform.name = "Deform"

	bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -0.6))
	coll = bpy.context.active_object
	coll.name = "Collider"

	dprops = getattr(deform, "andosim_artezbuild")
	dprops.enabled = True
	dprops.role = "DEFORMABLE"

	cprops = getattr(coll, "andosim_artezbuild")
	cprops.enabled = True
	cprops.role = "STATIC_COLLIDER"

	settings = bpy.context.scene.andosim_artezbuild
	settings.auto_export = True
	settings.use_selected_colliders = False
	settings.output_dir = outdir

	import andosim_artezbuild.ppf_export as ppf_export  # type: ignore

	export = ppf_export.export_ppf_scene_from_roles(bpy.context, settings)
	print("[PPF] scene_path:", export.scene_path)

	slices = list(getattr(export, "deformable_slices", []) or [])
	if not slices:
		raise RuntimeError("PPF export produced no deformable_slices")

	total_verts = 0
	objs = []
	for name, start, count in slices:
		obj = bpy.data.objects.get(name)
		if obj is None or getattr(obj, "type", None) != "MESH":
			raise RuntimeError(f"Missing deformable object '{name}'")
		if len(obj.data.vertices) != int(count):
			raise RuntimeError(f"Vertex count changed on '{name}'")
		start_i = int(start)
		count_i = int(count)
		objs.append((obj, start_i, count_i))
		total_verts = max(total_verts, start_i + count_i)

	curr = [0.0] * (total_verts * 3)
	for obj, start, _count in objs:
		mw = obj.matrix_world
		for i, v in enumerate(obj.data.vertices):
			w = mw @ v.co
			sx, sy, sz = ppf_export.blender_to_solver_xyz(float(w.x), float(w.y), float(w.z))
			j = start + i
			curr[3 * j + 0] = sx
			curr[3 * j + 1] = sy
			curr[3 * j + 2] = sz

	_finite(curr, "curr")

	import ppf_cts_backend  # type: ignore

	sess = ppf_cts_backend.Session(export.scene_path, outdir)
	try:
		out = sess.step(curr)
	finally:
		try:
			sess.close()
		except Exception:
			pass

	print("[PPF] step ok; outlen:", len(out))
	_finite(out, "out")


def _smoke_ando_core() -> None:
	from andosim_artezbuild.ando import _core_loader  # type: ignore

	core = _core_loader.get_core_module(strict=True, context="headless install+test")
	if core is None:
		raise RuntimeError("ando_barrier_core could not be loaded")

	for attr in ("barrier_energy", "barrier_gradient", "barrier_hessian"):
		_require(core, attr, "ando_barrier_core")

	e = core.barrier_energy(0.5, 1.0, 100.0)
	g = core.barrier_gradient(0.5, 1.0, 100.0)
	h = core.barrier_hessian(0.5, 1.0, 100.0)
	print("[ANDO] barrier_energy/gradient/hessian:", e, g, h)


def _smoke_reset_restores_mesh(outdir: str, module_name: str) -> None:
	import bpy  # type: ignore

	bpy.ops.wm.read_factory_settings(use_empty=True)

	_ensure_addon_registered(module_name)

	bpy.ops.mesh.primitive_grid_add(x_subdivisions=8, y_subdivisions=8, size=1.0)
	obj = bpy.context.active_object
	obj.name = "Deform"

	props = getattr(obj, "andosim_artezbuild")
	props.enabled = True
	props.role = "DEFORMABLE"

	settings = bpy.context.scene.andosim_artezbuild
	settings.output_dir = outdir

	import andosim_artezbuild.operators as ops_mod  # type: ignore

	coords0 = [0.0] * (len(obj.data.vertices) * 3)
	obj.data.vertices.foreach_get("co", coords0)

	ok, err = ops_mod._ppf_capture_mesh_snapshot([(obj.name, 0, len(obj.data.vertices))])
	if not ok:
		raise RuntimeError(f"Snapshot failed: {err}")
	settings.ppf_has_snapshot = True

	coords1 = coords0[:]
	for i in range(0, len(coords1), 3):
		coords1[i + 2] += 0.123
	obj.data.vertices.foreach_set("co", coords1)
	obj.data.update()

	mod = obj.modifiers.new(name="PPF_Cache", type="MESH_CACHE")
	mod.cache_format = "PC2"
	mod.filepath = str(Path(outdir) / "dummy.pc2")

	res = bpy.ops.andosim_artezbuild.ppf_reset_simulation()
	print("[RESET] op result:", res)

	coords2 = [0.0] * (len(obj.data.vertices) * 3)
	obj.data.vertices.foreach_get("co", coords2)

	if coords2 != coords0:
		raise RuntimeError("Reset did not restore vertex positions")
	if obj.modifiers.get("PPF_Cache") is not None:
		raise RuntimeError("Reset did not remove PPF_Cache modifier")


def main(argv: list[str]) -> int:
	parser = argparse.ArgumentParser(description="Install AndoSim ArteZbuild extension zip and run headless smoke tests")
	parser.add_argument("--zip", required=True, help="Path to andosim_artezbuild-*.zip")
	parser.add_argument("--outdir", default=os.environ.get("OUTDIR", ""), help="Output directory for traces")
	args = parser.parse_args(argv)

	outdir = (args.outdir or "").strip() or tempfile.mkdtemp(prefix="andosim_headless_")
	Path(outdir).mkdir(parents=True, exist_ok=True)
	print("[RUN] outdir:", outdir)

	_install_extension_zip(args.zip)

	module_name = "andosim_artezbuild"
	_ensure_addon_registered(module_name)

	try:
		import ppf_cts_backend  # type: ignore

		print("[PPF] ppf_cts_backend version:", getattr(ppf_cts_backend, "__version__", "?"))
	except Exception as exc:
		raise RuntimeError(f"ppf_cts_backend not importable after install: {exc}")

	import bpy  # type: ignore

	settings = getattr(bpy.context.scene, "andosim_artezbuild", None)
	if settings is None:
		raise RuntimeError("Scene missing andosim_artezbuild settings")
	for attr in ("output_dir", "scene_path", "fps", "dt"):
		_require(settings, attr, "settings")

	_smoke_ando_core()
	_smoke_ppf_step(outdir, module_name)
	_smoke_reset_restores_mesh(outdir, module_name)

	print("[OK] All headless checks passed")
	return 0


if __name__ == "__main__":
	if "--" in sys.argv:
		argv = sys.argv[sys.argv.index("--") + 1 :]
	else:
		argv = []
	raise SystemExit(main(argv))
