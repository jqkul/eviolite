//! Selection operators
//! 
//! This module contains the [`Select`] trait, as well as implementations of some common EA selection operators.
//! [`Tournament`] is a good selector to use if you just want to get started quickly,
//! since it works with every kind of algorithm and is pretty simple to understand.
//! 
//! [`Tournament`]: ./struct.Tournament.html

pub(crate) mod nsga;
pub(crate) mod tournament;
pub(crate) mod utils;

pub use nsga::{NSGA2, rank_nondominated, ParetoFronts};
pub use tournament::Tournament;

use crate::Cached;
use crate::Solution;

/// Trait that indicates the ability to select from a population.
pub trait Select<T: Solution> {
    /// Mutate `population` in place, leaving `amount` solutions in it.
    fn select(&self, amount: usize, population: &mut Vec<Cached<T>>);
}

/// Marker trait that indicates a selector uses randomness in its selection.
/// 
/// This is required by some algorithms for correctness, e.g. [`Simple`].
/// 
/// [`Simple`]: ../alg/struct.Simple.html
pub trait Stochastic {}