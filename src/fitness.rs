//! Representation and evaluation of fitness values
//! 
//! This module contains [`MultiObjective`].
//! You should use either `f64` or [`MultiObjective`]
//! as your [`Solution`]'s fitness type for simple applications.
//! 
//! This module also contains [`par_evaluate`], a function that uses
//! [`rayon`]'s parallel iterators to efficiently evaluate a population.
//! 
//! [`Solution`]: ../trait.Solution.html
//! [`MultiObjective`]: ./struct.MultiObjective.html

use std::ops::Deref;

use rayon::prelude::*;

use crate::{Solution, Cached};

/// Type that represents fitness values in multi-objective optimization
/// 
/// This type includes support for weighted fitness values,
/// which can then be collapsed into a single combined fitness.
#[derive(Clone, Copy, Debug)]
pub struct MultiObjective<const M: usize> {
    weighted: [f64; M],
}

impl<const M: usize> From<MultiObjective<M>> for f64 {
    fn from(value: MultiObjective<M>) -> f64 {
        let mut result: f64 = 0.0;
        for i in 0..M {
            result += value.weighted[i];
        }
        result
    }
}

impl<const M: usize> MultiObjective<M> {
    /// Create a new instance of `MultiObjective` that contains exactly `values` with no weighting.
    /// 
    /// Example
    /// =======
    /// ```
    /// # use eviolite::fitness::MultiObjective;
    /// let fit: MultiObjective<3> = MultiObjective::new_unweighted([1.0, 2.0, 3.0]);
    /// assert_eq!(fit[2], 3.0);
    /// ```
    pub fn new_unweighted(values: [f64; M]) -> Self {
        MultiObjective { weighted: values }
    }

    /// Create a builder that produces `MultiObjective` instances weighted by `weights`.
    /// 
    /// Example
    /// =======
    /// ```
    /// # use eviolite::fitness::MultiObjective;
    /// let builder = MultiObjective::weighted_builder([1.0, 2.0]);
    /// let fit: MultiObjective<2> = builder([5.0, 5.0]);
    /// assert_eq!(fit[1], 10.0);
    /// ```
    pub fn weighted_builder(weights: [f64; M]) -> impl Fn([f64; M]) -> Self {
        move |values: [f64; M]| MultiObjective {
            weighted: {
                let mut arr = [0f64; M];
                for i in 0..M {
                    arr[i] = weights[i] * values[i];
                }
                arr
            },
        }
    }
}

impl<const M: usize> Deref for MultiObjective<M> {
    type Target = [f64; M];
    fn deref(&self) -> &Self::Target {
        &self.weighted
    }
}

impl<const M: usize> PartialEq for MultiObjective<M> {
    fn eq(&self, other: &Self) -> bool {
        for i in 0..M {
            if (self[i] - other[i]).abs() > f64::EPSILON {
                return false;
            }
        }
        true
    }
}

impl PartialEq<f64> for MultiObjective<1> {
    fn eq(&self, other: &f64) -> bool {
        (self[0] - other).abs() < f64::EPSILON
    }
}

/// Evaluate the fitness of every solution in a population in parallel.
/// 
/// For good performance, you should only ever evaluate solutions using this function, not
/// using the [`.evaluate()`] method directly.
/// 
/// [`.evaluate()`]: ../trait.Solution.html#tymethod.evaluate
pub fn par_evaluate<T: Solution>(pop: &[Cached<T>]) {
    pop.par_iter().for_each(|ind| {
        ind.evaluate();
    });
}