use pyo3::prelude::*;

use ppf_contact_solver as ppf;

const VERSION: &str = "0.0.5";

#[pyclass]
struct Session {
    inner: Option<ppf::InProcessSession>,
}

#[pymethods]
impl Session {
    #[new]
    #[pyo3(signature = (scene_path=None, output_dir=None))]
    fn new(scene_path: Option<String>, output_dir: Option<String>) -> PyResult<Self> {
        let scene_path = match scene_path {
            Some(p) if !p.is_empty() => p,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "scene_path is required for now (set PPF_SCENE_PATH in Blender)".to_string(),
                ))
            }
        };

        let output_dir = output_dir.unwrap_or_else(|| "/tmp/ppf_blender_output".to_string());

        let program_args = ppf::ProgramArgs {
            path: scene_path,
            output: output_dir,
            load: 0,
        };

        let inner = ppf::InProcessSession::new(program_args)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

        Ok(Self { inner: Some(inner) })
    }

    /// Step the simulation.
    ///
    /// Current stub behavior: returns the input vertices unchanged.
    ///
    /// Args:
    ///   verts_flat: a flat list [x0,y0,z0,x1,y1,z1,...]
    fn step(&mut self, py: Python<'_>, verts_flat: Vec<f32>) -> PyResult<Vec<f32>> {
        let Some(inner) = self.inner.as_mut() else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Session is closed".to_string(),
            ));
        };

        // Optional: if the incoming vertex buffer matches, use it as the current state.
        // This keeps the Python API compatible with the Blender operator.
        let _ = inner.set_curr_vertices_flat(&verts_flat);

        py.allow_threads(|| {
            inner
                .step_vertices_flat()
                .map_err(pyo3::exceptions::PyRuntimeError::new_err)
        })
    }

    /// Override pin target positions for the current session.
    ///
    /// Args:
    ///   indices: global vertex indices in the exported PPF scene
    ///   positions_flat: flat xyz targets, length == 3 * len(indices), in solver coordinates
    fn set_pin_targets(&mut self, indices: Vec<u32>, positions_flat: Vec<f32>) -> PyResult<()> {
        let Some(inner) = self.inner.as_mut() else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Session is closed".to_string(),
            ));
        };
        inner
            .set_pin_targets_flat(&indices, &positions_flat)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        Ok(())
    }

    fn clear_pin_targets(&mut self) -> PyResult<()> {
        let Some(inner) = self.inner.as_mut() else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Session is closed".to_string(),
            ));
        };
        inner.clear_pin_targets();
        Ok(())
    }

    /// Override collision mesh (static colliders) vertex positions.
    ///
    /// Args:
    ///   verts_flat: flat xyz positions for the exported collision mesh, in solver coordinates.
    fn set_collision_mesh_vertices(&mut self, verts_flat: Vec<f32>) -> PyResult<()> {
        let Some(inner) = self.inner.as_mut() else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Session is closed".to_string(),
            ));
        };

        inner
            .set_collision_mesh_vertices_flat(&verts_flat)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        Ok(())
    }

    fn clear_collision_mesh_vertices(&mut self) -> PyResult<()> {
        let Some(inner) = self.inner.as_mut() else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Session is closed".to_string(),
            ));
        };
        inner.clear_collision_mesh_vertices();
        Ok(())
    }

    fn close(&mut self) {
        if let Some(mut inner) = self.inner.take() {
            inner.close();
        }
    }
}

#[pymodule]
fn _core(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", VERSION)?;
    m.add_class::<Session>()?;
    // For convenience: also expose a top-level version() call later.
    let _ = py;
    Ok(())
}
