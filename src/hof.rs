//! Keeping track of the best solutions across generations
//!
//! There's no point in running an evolutionary algorithm if you can't find the solutions
//! that have performed the best. As such, the [`HallOfFame`] trait describes an object that will
//! record each successive generation and update its own records of the best solutions over time.
//!
//! This module also contains a few simple [`HallOfFame`] implementors that should work well for simple applications.

use std::{fmt::Debug, ops::Deref};

use crate::{
    fitness::MultiObjective,
    select::{rank_nondominated, utils::retain_indices},
    Cached, Solution,
};
use itertools::Itertools;

/// A trait that indicates a type can record certain solutions over successive generations.
pub trait HallOfFame<T: Solution> {
    /// Include the solutions of a generation in the record.
    fn record(&mut self, generation: &[Cached<T>]);
}

/// Keeps a ranking of the best solutions across all generations
///
/// This type supports any solution whose fitness can be represented as a single number,
/// enforced by the `T::Fitness: Into<f64>` requirement on its [`HallOfFame`] implementation.
/// [`MultiObjective`] implements `Into<f64>` for convenience, taking weighting into account.
///
/// [`HallOfFame`]: ./trait.HallOfFame.html
/// [`MultiObjective`]: ../fitness/struct.MultiObjective.html
#[derive(Clone)]
pub struct BestN<T: Solution> {
    max: usize,
    best: Vec<Cached<T>>,
}

impl<T: Solution> BestN<T> {
    /// Create a new `BestN` that will hold at most the `max` best solutions
    /// across all generations it records.
    pub fn new(max: usize) -> Self {
        BestN {
            max,
            best: Vec::with_capacity(max),
        }
    }

    /// Get a reference to the stored best solutions, sorted by rank.
    /// E.g. `all()[0]` is the #1 best solution.
    pub fn all(&self) -> &[Cached<T>] {
        &self.best
    }

    /// Get a reference to the solution with the highest fitness
    /// across all recorded generations, if it exists.
    ///
    /// Returns `None` if no solutions are stored.
    pub fn best(&self) -> Option<&Cached<T>> {
        self.best.first()
    }
}

impl<T> HallOfFame<T> for BestN<T>
where
    T: Solution,
    T::Fitness: Into<f64>,
{
    fn record(&mut self, generation: &[Cached<T>]) {
        for ind in generation {
            if let Some(idx) = self.find_index(ind) {
                self.best.insert(idx, ind.clone());
            } else if self.best.len() < self.max {
                self.best.push(ind.clone());
            }
        }
        self.best.truncate(self.max);
    }
}

impl<T: Solution> IntoIterator for BestN<T> {
    type Item = Cached<T>;
    type IntoIter = IntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            inner: self.best.into_iter(),
        }
    }
}

impl<T> Debug for BestN<T>
where
    T: Solution,
    Cached<T>: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.best.iter()).finish()
    }
}

impl<T> Deref for BestN<T>
where
    T: Solution,
{
    type Target = [Cached<T>];
    fn deref(&self) -> &Self::Target {
        &self.best
    }
}

impl<T, F> BestN<T>
where
    T: Solution<Fitness = F>,
    F: Into<f64>,
{
    fn find_index(&self, ind: &Cached<T>) -> Option<usize> {
        let fit = ind.evaluate().into();
        if self.best.is_empty() || fit > self.best[0].evaluate().into() {
            return Some(0);
        }

        for (i, (a, b)) in self.best.iter().tuple_windows().enumerate() {
            if fit > b.evaluate().into() && fit < a.evaluate().into() {
                return Some(i + 1);
            }
        }

        None
    }
}

/// Stores all globally nondominated solutions
///
/// Stores a record of all solutions who are not dominated in the set of all solutions in every generation
/// (also known as a [Pareto front](https://en.wikipedia.org/wiki/Pareto_front)).
/// For more information on how this is calculated, see the documentation for [`rank_nondominated()`].
///
/// [`rank_nondominated()`]: ../select/fn.rank_nondominated.html
#[derive(Clone)]
pub struct BestPareto<T, const M: usize>
where
    T: Solution<Fitness = MultiObjective<M>>,
{
    front: Vec<Cached<T>>,
}

impl<T, const M: usize> BestPareto<T, M>
where
    T: Solution<Fitness = MultiObjective<M>>,
{
    /// Create a new instance of `BestPareto` with no stored solutions.
    pub fn new() -> Self {
        BestPareto {
            front: Default::default(),
        }
    }

    /// Get a reference to the stored list of globally nondominated solutions, in arbitrary order.
    pub fn front(&self) -> &[Cached<T>] {
        &self.front
    }
}

impl<T, const M: usize> Default for BestPareto<T, M>
where
    T: Solution<Fitness = MultiObjective<M>>,
{
    fn default() -> Self {
        BestPareto { front: Vec::new() }
    }
}

impl<T, const M: usize> HallOfFame<T> for BestPareto<T, M>
where
    T: Solution<Fitness = MultiObjective<M>>,
{
    fn record(&mut self, generation: &[Cached<T>]) {
        let pareto = rank_nondominated(generation);
        for (ind, rank) in generation.iter().zip(pareto.ranks.into_iter()) {
            if rank == 0 {
                self.front.push(ind.clone());
            }
        }
        let pareto2 = rank_nondominated(&self.front);
        let indices = (0..self.front.len())
            .filter(|i| pareto2.ranks[*i] == 0)
            .collect();
        retain_indices(&mut self.front, indices);
    }
}

impl<T, const M: usize> IntoIterator for BestPareto<T, M>
where
    T: Solution<Fitness = MultiObjective<M>>,
{
    type Item = Cached<T>;
    type IntoIter = IntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            inner: self.front.into_iter(),
        }
    }
}

impl<T, const M: usize> Debug for BestPareto<T, M>
where
    T: Solution<Fitness = MultiObjective<M>>,
    Cached<T>: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.front.iter()).finish()
    }
}

/// Iterator over the entries in a hall of fame
pub struct IntoIter<T: Solution> {
    inner: std::vec::IntoIter<Cached<T>>,
}

impl<T: Solution> Iterator for IntoIter<T> {
    type Item = Cached<T>;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutils::*;

    macro_rules! pop {
        ($ty:expr, $($val:expr),*) => {
            &[
                $(
                    Cached::new($ty($val))
                ),*
            ]
        };
    }

    #[test]
    fn bestn_size_1() {
        let mut hof: BestN<One> = BestN::new(1);

        hof.record(pop!(One, 1.0, 2.0, 3.0));
        assert_eq!(hof.best.len(), 1);
        assert_eq!(hof.best[0].evaluate(), 3.0);

        hof.record(pop!(One, 1.5, 2.5, 3.5));
        assert_eq!(hof.best[0].evaluate(), 3.5);
    }

    #[test]
    fn bestn_size_3() {
        let mut hof: BestN<One> = BestN::new(3);

        hof.record(pop!(One, 1.0, 2.0, 3.0, 4.0, 5.0));
        assert_eq!(hof.best.len(), 3);
        assert_eq!(hof.best[0].evaluate(), 5.0);
        assert_eq!(hof.best[1].evaluate(), 4.0);
        assert_eq!(hof.best[2].evaluate(), 3.0);

        hof.record(pop!(One, 1.5, 2.5, 3.5, 4.5, 5.5));
        assert_eq!(hof.best.len(), 3);
        assert_eq!(hof.best[0].evaluate(), 5.5);
        assert_eq!(hof.best[1].evaluate(), 5.0);
        assert_eq!(hof.best[2].evaluate(), 4.5);
    }

    #[test]
    fn bestpareto() {
        let mut hof: BestPareto<Foo, 2> = BestPareto::new();

        hof.record(pop!(Foo, [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]));
        assert_eq!(hof.front.len(), 3);

        hof.record(pop!(Foo, [0.6, 0.6], [0.7, 0.7]));
        assert_eq!(hof.front.len(), 3);

        assert!(hof.front.contains(&Cached::new(Foo([0.7, 0.7]))));
        assert!(hof.front.contains(&Cached::new(Foo([1.0, 0.0]))));
        assert!(hof.front.contains(&Cached::new(Foo([0.0, 1.0]))));

        assert!(!hof.front.contains(&Cached::new(Foo([0.5, 0.5]))));
        assert!(!hof.front.contains(&Cached::new(Foo([0.6, 0.6]))));
    }
}
