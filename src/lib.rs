pub mod alg;
pub mod fitness;
pub mod repro_thread_rng;
pub mod select;
pub mod stats;
pub mod hof;
#[cfg(feature = "ndarray")]
pub use eviolite_ndarray::{mutation, crossover};

mod traits;
pub use traits::{Solution, Fitness, Algorithm};

mod utils;
pub use utils::Cached;

use select::Select;
use stats::GenerationStats;
use utils::NFromFunction;
use hof::HallOfFame;

pub struct Evolution<T, Alg, Sel, Stat, Hof> where T: Solution, Alg: Algorithm, Sel: Select<T>, Stat: GenerationStats<T>, Hof: HallOfFame<T> {
    population: Vec<Cached<T>>,
    algorithm: Alg,
    selector: Sel,
    stats: Vec<Stat>,
    hall_of_fame: Hof
}

impl<T, Alg, Sel, Stat, Hof> Evolution<T, Alg, Sel, Stat, Hof> where T: Solution, Alg: Algorithm, Sel: Select<T>, Stat: GenerationStats<T>, Hof: HallOfFame<T> {
    pub fn new(algorithm: Alg, selector: Sel, hall_of_fame: Hof) -> Self {
        Evolution {
            population: Vec::n_from_function(algorithm.pop_size(), Cached::generate),
            algorithm,
            selector,
            stats: Vec::new(),
            hall_of_fame
        }
    }

    pub fn run(&mut self, ngens: usize) {

    }
}