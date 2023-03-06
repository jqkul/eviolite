//! Convenience re-export of commonly used items

pub use crate::{
    alg, fitness,
    fitness::MultiObjective,
    hof,
    repro_rng::{random, thread_rng},
    select, stats, Cached, Evolution, Solution,
};

#[cfg(feature = "ndarray")]
pub use crate::{crossover, mutation};

pub use rand::Rng;
