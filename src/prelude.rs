//! Convenience re-export of commonly used items

pub use crate::{
    Solution,
    Evolution,
    Cached,
    alg,
    fitness,
    fitness::MultiObjective,
    hof,
    repro_rng::{
        thread_rng,
        random
    },
    select,
    stats,
};

#[cfg(feature = "ndarray")]
pub use crate::{
    crossover,
    mutation
};

pub use rand::Rng;