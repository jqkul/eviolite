use rand::Rng;
use rayon::prelude::*;

use crate::{select::{Select, find_best}, Individual, utils::LazyWrapper, repro_thread_rng::thread_rng};

pub fn simple<T>(
    initial_pop_size: usize,
    selector: impl Select<LazyWrapper<T>>,
    cxpb: f64,
    mutpb: f64,
    n_generations: usize,
) -> Vec<T>
where
    T: Individual,
{
    let mut population: Vec<LazyWrapper<T>> = Vec::with_capacity(initial_pop_size);
    for _ in 0..initial_pop_size {
        population.push(LazyWrapper::generate());
    }

    for n in 0..n_generations {
        // Evaluate all unevaluated individuals.
        // LazyWrapper will automatically cache the result, so intensive
        // computation can only happen once per distinct individual.
        population.par_iter().for_each(|lw| {lw.evaluate();});
        selector.select(population.len(), &mut population);
        var_and(&mut population, cxpb, mutpb);
    }

    population.into_iter().map(|lw| lw.into_inner()).collect()
}

pub fn var_and<T>(population: &mut Vec<T>, cxpb: f64, mutpb: f64) where T: Individual {
    let mut rng = thread_rng();
    for i in 1..population.len() {
        if rng.gen_bool(cxpb) {
            let (head, tail) = population.split_at_mut(i);
            let a = head.last_mut().unwrap();
            let b = tail.first_mut().unwrap();
            T::crossover(a, b);
        }
    }

    for individual in population.iter_mut() {
        if rng.gen_bool(mutpb) {
            individual.mutate();
        }
    }
}