use pyo3::prelude::*;

mod action_mapper;
mod observation;
mod step_result;
mod vec_env;
mod spectator;

/// Native module for shogi-gym RL environments.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // m.add_class::<action_mapper::DefaultActionMapper>()?;
    // m.add_class::<observation::DefaultObservationGenerator>()?;
    // m.add_class::<vec_env::VecEnv>()?;
    // m.add_class::<spectator::SpectatorEnv>()?;
    // m.add_class::<step_result::StepResult>()?;
    Ok(())
}
