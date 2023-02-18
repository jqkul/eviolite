use std::cell::Cell;

use crate::traits::{Fitness, Solution};

#[derive(Clone)]
pub struct LazyWrapper<T: Solution> {
    individual: T,
    fitness: Cell<Option<T::Fitness>>,
}

impl<T> LazyWrapper<T>
where
    T: Solution,
{
    pub fn new(individual: T) -> Self {
        LazyWrapper {
            individual,
            fitness: Cell::new(None),
        }
    }

    pub fn into_inner(self) -> T {
        self.individual
    }
}

impl<T> Solution for LazyWrapper<T>
where
    T: Solution,
{
    type Fitness = T::Fitness;

    fn generate() -> Self {
        LazyWrapper {
            individual: T::generate(),
            fitness: Cell::new(None),
        }
    }

    fn evaluate(&self) -> Self::Fitness {
        if let Some(fitness) = self.fitness.get() {
            fitness
        } else {
            let new_fitness = self.individual.evaluate();
            self.fitness.set(Some(new_fitness));
            new_fitness
        }
    }

    fn crossover(a: &mut Self, b: &mut Self) {
        T::crossover(&mut a.individual, &mut b.individual);
        a.invalidate();
        b.invalidate();
    }

    fn mutate(&mut self) {
        self.individual.mutate();
        self.invalidate();
    }
}

impl<T> LazyWrapper<T> where T: Solution {
    fn invalidate(&self) {
        self.fitness.set(None);
    }
}

unsafe impl<T> Sync for LazyWrapper<T> where T: Solution {}

pub trait IterIndices {
    type Item;
    fn iter_indices<I>(&self, indices: I) -> IndicesIter<Self::Item, I>
    where
        I: Iterator<Item = usize>;
}

impl<T> IterIndices for Vec<T> {
    type Item = T;
    fn iter_indices<I>(&self, indices: I) -> IndicesIter<Self::Item, I>
    where
        I: Iterator<Item = usize>,
    {
        IndicesIter {
            inner: self,
            indices,
        }
    }
}

pub struct IndicesIter<'a, T, I>
where
    I: Iterator<Item = usize>,
{
    inner: &'a [T],
    indices: I,
}

impl<'a, T, I> Iterator for IndicesIter<'a, T, I>
where
    I: Iterator<Item = usize>,
{
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        self.indices.next().map(|idx| &self.inner[idx])
    }
}

pub trait NFromFunction<T> {
    fn n_from_function(n: usize, f: impl Fn() -> T) -> Self;
}

impl<T> NFromFunction<T> for Vec<T> {
    fn n_from_function(n: usize, f: impl Fn() -> T) -> Self {
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(f());
        }
        v
    }
}