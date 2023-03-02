//! Pre-built algorithms
//!
//! This module contains the [`Algorithm`] trait and several pre-built algorithms that are commonly used.
//! If you want to get started quickly, using one of the pre-built algorithms is your best bet.

use std::marker::PhantomData;

use rand::{seq::SliceRandom, Rng};

use crate::{
    fitness::{par_evaluate, MultiObjective},
    repro_rng::thread_rng,
    select::{Select, Stochastic},
    utils::Cached,
    Solution,
};

/// A trait that describes the basic functionality of an evolutionary algorithm.
///
/// You can implement this yourself, or use one of the provided algorithms in this module.
pub trait Algorithm<T: Solution> {
    /// Advance one generation.
    ///
    /// The hall of fame and statistics are updated between calls to this method,
    /// so it must leave the population in a state where all the solutions in it
    /// are "presentable."
    ///
    /// [`Evolution`] will automatically evaluate the population between calls,
    /// so all solutions will have cached fitness values when this method is called.
    /// If you add new solutions to the population before selecting,
    /// you will need to manually call [`par_evaluate()`] between those steps
    /// in order to have good performance.
    ///
    /// [`Evolution`]: ../struct.Evolution.html
    /// [`par_evaluate()`]: ./fn.par_evaluate.html
    fn step(&self, population: &mut Vec<Cached<T>>);

    /// Get the desired population size of the algorithm.
    ///
    /// This is used by [`Evolution`] to generate the initial population for a run.
    ///
    /// [`Evolution`]: ../struct.Evolution.html
    fn pop_size(&self) -> usize;
}

/// One of the simplest possible evolutionary algorithms.
///
/// This is a good starting point, especially for single-objective optimization,
/// but it is only recommended as a starting point. You will generally get better
/// results with any other pre-built algorithm in this module.
///
/// The selection operator for this algorithm must implement the [`Stochastic`] trait
/// to show that randomness is involved in its selection. The algorithm selects N
/// solutions from a population of N, so if the selector is not stochastic, it will always
/// just yield the same population.
///
/// Pseudocode
/// ----------
/// A single step of the algorithm does the following:
/// ```notrust
/// select N solutions out of the population of N (this must necessarily result in duplicates)
/// replace the population with that selection
/// apply var_and to the population
/// ```
#[derive(Clone, Debug)]
pub struct Simple<T, S>
where
    T: Solution,
    S: Select<T> + Stochastic,
{
    pop_size: usize,
    cxpb: f64,
    mutpb: f64,
    selector: S,
    _phantom: PhantomData<T>,
}

impl<T, S> Simple<T, S>
where
    T: Solution,
    S: Select<T> + Stochastic,
{
    /// Create a new instance of the `Simple` algorithm with the specified parameters.
    pub fn new(pop_size: usize, cxpb: f64, mutpb: f64, selector: S) -> Self {
        Simple {
            pop_size,
            cxpb,
            mutpb,
            selector,
            _phantom: PhantomData,
        }
    }
}

impl<T, S> Algorithm<T> for Simple<T, S>
where
    T: Solution,
    S: Select<T> + Stochastic,
{
    fn pop_size(&self) -> usize {
        self.pop_size
    }

    fn step(&self, population: &mut Vec<Cached<T>>) {
        debug_assert_eq!(self.pop_size, population.len());

        self.selector.select(self.pop_size, population);

        var_and(population, self.cxpb, self.mutpb);
    }
}

/// Implementation of the (μ + λ) evolutionary algorithm.
///
/// Pseudocode
/// ----------
/// A single step of the algorithm does the following:
/// ```notrust
/// generate λ offspring using gen_or
/// evaluate the offspring
/// add the offspring to the population
/// select μ solutions out of the population of μ + λ
/// replace the population with that selection
/// ```
#[derive(Clone, Debug)]
pub struct MuPlusLambda<T, S>
where
    T: Solution,
    S: Select<T>,
{
    mu: usize,
    lambda: usize,
    cxpb: f64,
    mutpb: f64,
    selector: S,
    _phantom: PhantomData<T>,
}

impl<T, S> MuPlusLambda<T, S>
where
    T: Solution,
    S: Select<T>,
{
    /// Create a new instance of the `MuPlusLambda` algorithm with the specified parameters.
    pub fn new(mu: usize, lambda: usize, cxpb: f64, mutpb: f64, selector: S) -> Self {
        MuPlusLambda {
            mu,
            lambda,
            cxpb,
            mutpb,
            selector,
            _phantom: PhantomData,
        }
    }
}

impl<T, S> Algorithm<T> for MuPlusLambda<T, S>
where
    T: Solution,
    S: Select<T>,
{
    fn pop_size(&self) -> usize {
        self.mu
    }

    fn step(&self, population: &mut Vec<Cached<T>>) {
        population.append(&mut gen_or(population, self.lambda, self.cxpb, self.mutpb));

        par_evaluate(population);

        self.selector.select(self.mu, population);
    }
}

/// Implementation of the (μ, λ) evolutionary algorithm.
///
/// Pseudocode
/// ----------
/// A single step of the algorithm does the following:
/// ```notrust
/// generate λ offspring using gen_or
/// evaluate the offspring
/// replace the population with the offspring
/// select μ solutions out of the population of λ
/// make that selection the new population
/// ```
#[derive(Clone, Debug)]
pub struct MuCommaLambda<T, S>
where
    T: Solution,
    S: Select<T>,
{
    mu: usize,
    lambda: usize,
    cxpb: f64,
    mutpb: f64,
    selector: S,
    _phantom: PhantomData<T>,
}

impl<T, S> MuCommaLambda<T, S>
where
    T: Solution,
    S: Select<T>,
{
    /// Create a new instance of the `MuPlusLambda` algorithm with the specified parameters.
    ///
    /// Panics
    /// ======
    /// Panics if `mu > lambda`. The algorithm requires μ to be less than or equal to λ to work,
    /// since it selects μ solutions out of a population of λ.
    pub fn new(mu: usize, lambda: usize, cxpb: f64, mutpb: f64, selector: S) -> Self {
        if mu > lambda {
            panic!("(μ, λ) requires μ < λ");
        }
        MuCommaLambda {
            mu,
            lambda,
            cxpb,
            mutpb,
            selector,
            _phantom: PhantomData,
        }
    }
}

impl<T, S> Algorithm<T> for MuCommaLambda<T, S>
where
    T: Solution,
    S: Select<T>,
{
    fn pop_size(&self) -> usize {
        self.mu
    }

    fn step(&self, population: &mut Vec<Cached<T>>) {
        *population = gen_or(population, self.lambda, self.cxpb, self.mutpb);

        par_evaluate(population);

        self.selector.select(self.mu, population);
    }
}

/// An implementation of the NSGA-II evolutionary algorithm.
///
/// For more information about NSGA-II, see the documentation for
/// [`select::NSGA2`].
///
/// [`select::NSGA2`]: ../select/struct.NSGA2.html
#[derive(Clone, Debug)]
pub struct NSGA2 {
    pop_size: usize,
    cxpb: f64,
    mutpb: f64,
}

impl NSGA2 {
    /// Create a new instance of the `NSGA2` algorithm with the specified parameters.
    pub fn new(pop_size: usize, cxpb: f64, mutpb: f64) -> Self {
        NSGA2 {
            pop_size,
            cxpb,
            mutpb,
        }
    }
}

impl<T, const M: usize> Algorithm<T> for NSGA2
where
    T: Solution<Fitness = MultiObjective<M>>,
{
    fn pop_size(&self) -> usize {
        self.pop_size
    }

    fn step(&self, population: &mut Vec<Cached<T>>) {
        population.append(&mut gen_or(
            population,
            self.pop_size,
            self.cxpb,
            self.mutpb,
        ));

        par_evaluate(population);

        crate::select::NSGA2.select(self.pop_size, population);
    }
}

/// Vary a population in place.
///
/// This function has the potential to apply both crossover *and* mutation
/// to the same solution, hence the name.
///
/// Pseudocode
/// ----------
/// ```notrust
/// for each solution in the population:
///     if a random check of chance cxpb passes:
///         apply crossover between the solution and the one adjacent to it
///     if a random check of chance mutpb passes:
///         apply mutation to the solution
/// ```
pub fn var_and<T>(pop: &mut [T], cxpb: f64, mutpb: f64)
where
    T: Solution,
{
    let mut rng = thread_rng();
    for i in 0..pop.len() {
        if i != 0 && rng.gen_bool(cxpb) {
            let (head, tail) = pop.split_at_mut(i);
            let a = head.last_mut().unwrap();
            let b = tail.first_mut().unwrap();
            T::crossover(a, b);
        }

        if rng.gen_bool(mutpb) {
            pop[i].mutate();
        }
    }
}

/// Generate offspring from a population.
///
/// This function only ever applies crossover *or* mutation to a solution, hence the name.
///
/// Pseudocode
/// ----------
/// ```notrust
/// do n_offspring times:
///     randomly choose one operation from crossover, mutate, or clone
///     if crossover is chosen:
///         randomly choose two solutions from the population and clone them
///         apply crossover between the clones
///         add one of the clones (arbitrary) to the offspring
///     if mutate is chosen:
///         randomly choose a solution from the population and clone it
///         apply mutation to the clone
///         add the clone to the offspring
///     if clone is chosen:
///         randomly choose a solution from the population and clone it
///         add the clone to the offspring     
/// ```
///
/// The probabilities of crossover, mutate, and clone being chosen each iteration are
/// `cxpb`, `mutpb`, and `1 - (cxpb + mutpb)` respectively.
pub fn gen_or<T: Solution>(pop: &[T], n_offspring: usize, cxpb: f64, mutpb: f64) -> Vec<T> {
    let mut offspring: Vec<T> = Vec::with_capacity(n_offspring);
    for _ in 0..n_offspring {
        let mut rng = thread_rng();
        let choice: f64 = rng.gen();
        offspring.push(if choice < cxpb {
            let mut iter = pop.choose_multiple(&mut rng, 2).cloned();
            let mut a = iter.next().unwrap();
            let mut b = iter.next().unwrap();

            T::crossover(&mut a, &mut b);
            a
        } else if choice < cxpb + mutpb {
            let mut chosen = pop.choose(&mut rng).unwrap().clone();
            chosen.mutate();
            chosen
        } else {
            pop.choose(&mut rng).unwrap().clone()
        });
    }

    offspring
}
