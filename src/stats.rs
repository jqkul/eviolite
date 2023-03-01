//! Per-generation statistical analysis
//!
//! This module contains the [`GenerationStats`] trait.
//! You can implement this trait for a type to calculate whatever statistics you want about a given generation of solutions.
//! When you call [`.run()`] on your [`Evolution`] instance, you'll receive a `Vec` of statistics about each successive generation.
//!
//! If you just want to get started quickly, this module also includes a few simple implementors.
//! [`GenerationStats`] is also implemented for `()` as a no-op,
//! allowing you to opt-out of calculating any statistics.
//!
//! [`.run()`]: ../struct.Evolution.html#method.run
//! [`Evolution`]: ../struct.Evolution.html

use crate::{fitness::MultiObjective, utils::Cached, Solution};

/// Trait that indicates a type represents statistics about
/// a generation of solutions
pub trait GenerationStats<T: Solution> {
    /// Analyze the generation and generate statistics about it.
    fn analyze(generation: &[Cached<T>]) -> Self;
}

impl<T> GenerationStats<T> for ()
where
    T: Solution,
{
    fn analyze(_: &[Cached<T>]) -> Self {}
}

/// Mean and standard deviation for single-objective fitness
pub struct FitnessBasic {
    mean: f64,
    variance: f64,
}

impl FitnessBasic {
    /// Get the mean of the generation's fitness values.
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Get the variance of the generation's fitness values.
    pub fn variance(&self) -> f64 {
        self.variance
    }

    /// Get the standard deviation of the generation's fitness values.
    pub fn stdev(&self) -> f64 {
        self.variance.sqrt()
    }
}

impl<T> GenerationStats<T> for FitnessBasic
where
    T: Solution,
    T::Fitness: Into<f64>,
{
    fn analyze(generation: &[Cached<T>]) -> Self {
        let mean: f64 = generation.iter().map(|sol| sol.evaluate().into()).sum();
        let variance: f64 = generation
            .iter()
            .map(|sol| (sol.evaluate().into() - mean).powi(2))
            .sum();

        FitnessBasic { mean, variance }
    }
}

/// Mean and standard deviation of each objective in a [`MultiObjective`]
///
/// [`MultiObjective`]: ../fitness/struct.MultiObjective.html
pub struct FitnessBasicMulti<const M: usize> {
    mean: [f64; M],
    variance: [f64; M],
    stdev: [f64; M],
}

impl<const M: usize> FitnessBasicMulti<M> {
    /// Get the mean for each objective.
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// Get the variance for each objective.
    pub fn variance(&self) -> &[f64] {
        &self.variance
    }

    /// Get the standard deviation for each objective.
    pub fn stdev(&self) -> &[f64] {
        &self.stdev
    }
}

impl<T, const M: usize> GenerationStats<T> for FitnessBasicMulti<M>
where
    T: Solution<Fitness = MultiObjective<M>>,
{
    fn analyze(generation: &[Cached<T>]) -> Self {
        let len = generation.len() as f64;
        let mut mean = [0.0f64; M];
        let mut variance = [0.0f64; M];
        let mut stdev = [0.0f64; M];

        for m in 0..M {
            mean[m] = generation
                .iter()
                .map(|ind| Cached::fit(ind, m))
                .sum::<f64>()
                / len;
            variance[m] = generation
                .iter()
                .map(|ind| (Cached::fit(ind, m) - mean[m]).powi(2))
                .sum::<f64>()
                / len;
            stdev[m] = variance[m].sqrt();
        }

        FitnessBasicMulti {
            mean,
            variance,
            stdev,
        }
    }
}
