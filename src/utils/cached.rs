use std::{cell::UnsafeCell, fmt::Debug};

use crate::{fitness::MultiObjective, Solution};

/// A wrapper around a solution that automatically caches the fitness value
///
/// Evaluating the fitness of solutions is nearly always the most computationally intensive
/// part of an evolutionary algorithm. This wrapper type makes it so that that computation
/// will only ever happen once for every distinct individual. It implements [`Solution`] itself,
/// so you can use the exact same interface you would if it weren't there.
pub struct Cached<T: Solution> {
    inner: T,
    fitness: UnsafeCell<Option<T::Fitness>>,
}

impl<T> Solution for Cached<T>
where
    T: Solution,
{
    type Fitness = T::Fitness;

    fn generate() -> Self {
        Cached {
            inner: T::generate(),
            fitness: UnsafeCell::new(None),
        }
    }

    fn evaluate(&self) -> Self::Fitness {
        if let Some(fitness) = unsafe { *self.fitness.get() } {
            fitness
        } else {
            let new_fitness = self.inner.evaluate();
            unsafe {
                *self.fitness.get() = Some(new_fitness);
            }
            new_fitness
        }
    }

    fn crossover(a: &mut Self, b: &mut Self) {
        T::crossover(&mut a.inner, &mut b.inner);
        a.clear_cache();
        b.clear_cache();
    }

    fn mutate(&mut self) {
        self.inner.mutate();
        self.clear_cache();
    }
}

impl<T> Cached<T>
where
    T: Solution,
{
    /// Create a new wrapper around an existing solution.
    pub fn new(individual: T) -> Self {
        Cached {
            inner: individual,
            fitness: UnsafeCell::new(None),
        }
    }

    /// Consumes the `Cached`, returning a tuple of the solution it contained
    /// and an [`Option`] of the fitness value that could have been cached.
    pub fn into_inner(mut self) -> (T, Option<T::Fitness>) {
        (self.inner, *self.fitness.get_mut())
    }

    /// Delete any cached fitness value.
    /// Returns the fitness value that was cached, if it existed.
    ///
    /// **Be careful with this method**;
    /// you should not generally need to use it.
    /// Using it incorrectly can cause evaluations to be repeated
    /// unnecessarily, leading to heavy slowdowns.
    pub fn clear_cache(&mut self) -> Option<T::Fitness> {
        std::mem::replace(self.fitness.get_mut(), None)
    }
}

impl<T> Clone for Cached<T>
where
    T: Solution,
{
    fn clone(&self) -> Self {
        Cached {
            inner: self.inner.clone(),
            fitness: UnsafeCell::new(unsafe { *self.fitness.get() }),
        }
    }
}

impl<T> AsRef<T> for Cached<T>
where
    T: Solution,
{
    fn as_ref(&self) -> &T {
        &self.inner
    }
}

impl<T> PartialEq for Cached<T>
where
    T: Solution + PartialEq,
{
    fn eq(&self, other: &Cached<T>) -> bool {
        self.inner.eq(&other.inner)
    }
}

impl<T> PartialEq<T> for Cached<T>
where
    T: Solution + PartialEq,
{
    fn eq(&self, other: &T) -> bool {
        self.inner.eq(other)
    }
}

impl<T> PartialOrd for Cached<T>
where
    T: Solution + PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.inner.partial_cmp(&other.inner)
    }
}

impl<T> PartialOrd<T> for Cached<T>
where
    T: Solution + PartialOrd,
{
    fn partial_cmp(&self, other: &T) -> Option<std::cmp::Ordering> {
        self.inner.partial_cmp(other)
    }
}

impl<T> Debug for Cached<T>
where
    T: Solution + Debug,
    T::Fitness: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cached")
            .field("solution", &self.inner)
            .field("stored_fitness", unsafe { &*self.fitness.get() })
            .finish()
    }
}

unsafe impl<T: Solution> Sync for Cached<T> {}

impl<T, const M: usize> Cached<T>
where
    T: Solution<Fitness = MultiObjective<M>>,
{
    pub(crate) fn fit(this: &Self, m: usize) -> f64 {
        unsafe { &*this.fitness.get() }.as_ref().unwrap()[m]
    }
}
