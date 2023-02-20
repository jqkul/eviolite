use std::cell::UnsafeCell;

use crate::{
    traits::{Fitness, Solution}, fitness::MultiObjective,
};

pub struct Cached<T: Solution> {
    individual: T,
    fitness: UnsafeCell<Option<T::Fitness>>,
}

impl<T> Solution for Cached<T>
where
    T: Solution,
{
    type Fitness = T::Fitness;

    fn generate() -> Self {
        Cached {
            individual: T::generate(),
            fitness: UnsafeCell::new(None),
        }
    }

    fn evaluate(&self) -> Self::Fitness {
        if let Some(fitness) = unsafe { *self.fitness.get() } {
            fitness
        } else {
            let new_fitness = self.individual.evaluate();
            unsafe {
                *self.fitness.get() = Some(new_fitness);
            }
            new_fitness
        }
    }

    fn crossover(a: &mut Self, b: &mut Self) {
        T::crossover(&mut a.individual, &mut b.individual);
        Cached::invalidate(a);
        Cached::invalidate(b);
    }

    fn mutate(&mut self) {
        self.individual.mutate();
        Cached::invalidate(self);
    }
}

impl<T> Clone for Cached<T>
where
    T: Solution,
{
    fn clone(&self) -> Self {
        Cached {
            individual: self.individual.clone(),
            fitness: UnsafeCell::new(unsafe { *self.fitness.get() }),
        }
    }
}

impl<T> AsRef<T> for Cached<T>
where
    T: Solution,
{
    fn as_ref(&self) -> &T {
        &self.individual
    }
}

unsafe impl<T> Sync for Cached<T> where T: Solution {}

impl<T> Cached<T>
where
    T: Solution,
{
    pub fn new(individual: T) -> Self {
        Cached {
            individual,
            fitness: UnsafeCell::new(None),
        }
    }

    pub fn into_inner(mut this: Self) -> (T, Option<T::Fitness>) {
        (this.individual, *this.fitness.get_mut())
    }
}

impl<T, const M: usize> Cached<T>
where
    T: Solution<Fitness = MultiObjective<M>>,
{
    pub(crate) fn fit(this: &Self, m: usize) -> f64 {
        unsafe { &*this.fitness.get() }.as_ref().unwrap()[m]
    }
}

impl<T> Cached<T>
where
    T: Solution,
{
    fn invalidate(this: &mut Self) {
        *this.fitness.get_mut() = None;
    }
}