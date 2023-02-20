mod nsga;
mod tournament;
pub(crate) mod utils;

pub use nsga::rank_nondominated;
pub use nsga::NSGA2;
pub use tournament::Tournament;

use crate::Cached;
use crate::Solution;

pub trait Select<T: Solution> {
    fn select(&self, k: usize, pop: &mut Vec<Cached<T>>);
}
