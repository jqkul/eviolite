use rand::{seq::SliceRandom, Rng};
use rayon::prelude::*;

use crate::{
    fitness::MultiObjective,
    repro_thread_rng::thread_rng,
    select::{rank_nondominated, Select, NSGA2},
    utils::Cached, Solution
};

pub fn simple<T>(
    initial_pop_size: usize,
    selector: impl Select<T>,
    cxpb: f64,
    mutpb: f64,
    n_generations: usize,
) -> Vec<T>
where
    T: Solution,
{
    let mut pop = gen_pop(initial_pop_size);

    for _ in 0..n_generations {
        // Evaluate all unevaluated individuals.
        // LazyWrapper will automatically cache the result, so intensive
        // computation can only happen once per distinct individual.
        pop.par_iter().for_each(|lw| {
            lw.evaluate();
        });

        selector.select(pop.len(), &mut pop);

        var_and(&mut pop, cxpb, mutpb);
    }

    decache(pop)
}

pub fn mu_plus_lambda<T>(
    mu: usize,
    lambda: usize,
    selector: impl Select<T>,
    cxpb: f64,
    mutpb: f64,
    n_generations: usize,
) -> Vec<T>
where
    T: Solution,
{
    let mut pop = gen_pop(mu);

    for _ in 0..n_generations {
        // Generate offspring using var or
        pop.extend_from_slice(&var_or(&pop, lambda, cxpb, mutpb));

        // Evaluate all unevaluated individuals in parallel
        pop.par_iter().for_each(|lw| {
            lw.evaluate();
        });

        selector.select(mu, &mut pop);
    }

    decache(pop)
}

pub fn mu_comma_lambda<T>(
    mu: usize,
    lambda: usize,
    selector: impl Select<T>,
    cxpb: f64,
    mutpb: f64,
    n_generations: usize,
) -> Vec<T>
where
    T: Solution,
{
    assert!(
        lambda >= mu,
        "mu_comma_lambda requires mu < lambda to work correctly"
    );

    let mut pop = gen_pop(mu);

    for _ in 0..n_generations {
        pop = var_or(&pop, lambda, cxpb, mutpb);

        pop.par_iter().for_each(|lw| {
            lw.evaluate();
        });

        selector.select(mu, &mut pop);
    }

    decache(pop)
}

pub fn nsga2<T, const M: usize>(
    pop_size: usize,
    cxpb: f64,
    mutpb: f64,
    n_generations: usize,
) -> Vec<T>
where
    T: Solution<Fitness = MultiObjective<M>>
{
    let mut pop = gen_pop(pop_size);

    for _ in 0..n_generations {
        // Generate offspring using var or
        pop.extend_from_slice(&var_or(&pop, pop_size, cxpb, mutpb));

        // Evaluate all unevaluated individuals in parallel
        pop.par_iter().for_each(|lw| {
            lw.evaluate();
        });

        NSGA2.select(pop_size, &mut pop);
    }

    let rankings = rank_nondominated(&pop).rankings;

    decache(
        pop.into_iter()
            .enumerate()
            .filter(|(i, _)| rankings[*i] == 0)
            .map(|(_, ind)| ind),
    )
}

pub fn var_and<T>(pop: &mut [T], cxpb: f64, mutpb: f64)
where
    T: Solution,
{
    let mut rng = thread_rng();
    for i in 1..pop.len() {
        if rng.gen_bool(cxpb) {
            let (head, tail) = pop.split_at_mut(i);
            let a = head.last_mut().unwrap();
            let b = tail.first_mut().unwrap();
            T::crossover(a, b);
        }
    }

    for individual in pop.iter_mut() {
        if rng.gen_bool(mutpb) {
            individual.mutate();
        }
    }
}

pub fn var_or<T>(pop: &[T], n: usize, cxpb: f64, mutpb: f64) -> Vec<T>
where
    T: Solution,
{
    VarOr::from_slice(pop, n, cxpb, mutpb).collect()
}

pub struct VarOr<'a, T>
where
    T: Solution,
{
    n: usize,
    i: usize,
    cxpb: f64,
    mutpb: f64,
    src: &'a [T],
}

impl<'a, T> VarOr<'a, T>
where
    T: Solution,
{
    fn from_slice(src: &'a [T], n: usize, cxpb: f64, mutpb: f64) -> Self {
        VarOr {
            n,
            i: 0,
            cxpb,
            mutpb,
            src,
        }
    }
}

impl<'a, T> Iterator for VarOr<'a, T>
where
    T: Solution,
{
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= self.n {
            None
        } else {
            self.i += 1;
            let mut rng = thread_rng();
            let choice: f64 = rng.gen();
            Some(if choice < self.cxpb {
                let mut iter = self.src.choose_multiple(&mut rng, 2).cloned();
                let mut a = iter.next().unwrap();
                let mut b = iter.next().unwrap();

                T::crossover(&mut a, &mut b);
                a
            } else if choice < self.cxpb + self.mutpb {
                let mut chosen = self.src.choose(&mut rng).unwrap().clone();
                chosen.mutate();
                chosen
            } else {
                self.src.choose(&mut rng).unwrap().clone()
            })
        }
    }
}

impl<'a, T> ExactSizeIterator for VarOr<'a, T>
where
    T: Solution,
{
    fn len(&self) -> usize {
        self.n - self.i
    }
}

fn gen_pop<T>(size: usize) -> Vec<Cached<T>>
where
    T: Solution,
{
    let mut pop = Vec::with_capacity(size);
    for _ in 0..size {
        pop.push(Cached::generate())
    }
    pop
}

fn decache<T>(wrapped: impl IntoIterator<Item = Cached<T>>) -> Vec<T>
where
    T: Solution,
{
    wrapped
        .into_iter()
        .map(|lw| Cached::into_inner(lw).0)
        .collect()
}
