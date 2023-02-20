use crate::{select::Select, fitness::MultiObjective};

use super::Cache;

pub trait Solution: Clone + Sync {
    type Fitness: Fitness;
    fn generate() -> Self;
    fn evaluate(&self) -> Self::Fitness;
    fn crossover(a: &mut Self, b: &mut Self);
    fn mutate(&mut self);

    fn collapsed(&self) -> f64 {
        self.evaluate().collapse()
    }
}

pub trait Fitness: Copy {
    fn collapse(&self) -> f64;
}

impl Fitness for f64 {
    fn collapse(&self) -> f64 {
        *self
    }
}

pub trait Algorithm {
    fn step<T, const M: usize>(&self, population: &mut Vec<Cache<T>>, selector: &impl Select<T>, logger: impl FnMut(&[Cache<T>])) where T: Solution;
    fn pop_size(&self) -> usize;
}