pub mod utils;
mod tournament;
mod nsga;
pub use tournament::Tournament;

use crate::Solution;

pub trait Select<T: Solution> {
    fn select(&self, n: usize, population: &mut Vec<T>);
}