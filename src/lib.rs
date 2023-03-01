#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! Tools for evolutionary algorithms
//!
//! Eviolite is a set of tools and algorithms for evolutionary computing in Rust.
//! It is written in a performance-minded, minimal-copy style,
//! and uses [`rayon`] to parallelize the most computationally intensive parts.
//! It also includes a drop-in replacement for [`rand`]'s `thread_rng`
//! that is fully reproducible and can be seeded from an environment variable.
//!
//! The general workflow is to implement [`Solution`] for a type you wish to optimize,
//! construct an instance of [`Evolution`], and call one of its `run_` methods.
//!
//! Features
//! ========
//! The `ndarray` crate feature enables the [`crossover`] and [`mutation`] modules,
//! which contain helpful functions for using Eviolite alongside the [`ndarray`] crate.
//!
//! [`.run()`]: ./struct.Evolution.html#method.run

pub mod alg;
pub mod fitness;
pub mod hof;
pub mod repro_rng;
pub mod select;
pub mod stats;

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
pub mod crossover;
#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
pub mod mutation;

mod utils;

pub use utils::Cached;

pub(crate) mod testutils;

use alg::Algorithm;
use fitness::par_evaluate;
use hof::HallOfFame;
use stats::GenerationStats;
use utils::NFromFunction;

/// A trait that allows a type to be optimized using an evolutionary algorithm.
///
/// The entire crate is generic over this trait; you must implement it on a type to get anything done.
pub trait Solution: Clone + Sync {
    /// The type that represents this solution's fitness.
    /// For most applications, you'll want to use `f64` or [`MultiObjective`] for this,
    /// but you can use any type you want.
    ///
    /// [`MultiObjective`]: ./fitness/struct.MultiObjective.html
    type Fitness: Copy;

    /// Randomly generate a new solution.
    fn generate() -> Self;

    /// Evaluate the fitness of the solution.
    ///
    /// The return value of this method is cached by [`Cached`],
    /// so it must always return the exact same value for a given solution.
    fn evaluate(&self) -> Self::Fitness;

    /// Crossover operator.
    ///
    /// This is the operation analogous to reproduction in real-life evolution.
    /// This method must modify two solutions in such a way that
    /// afterward they both carry information from both of the inputs.
    /// It is recommended that this operation be stochastic, i.e.
    /// two applications of `crossover(&mut a, &mut b)` would produce different results
    /// for the same initial `a` and `b`.
    fn crossover(a: &mut Self, b: &mut Self);

    /// Mutation operator.
    ///
    /// This is the operation analogous to DNA mutation in real-life evolution.
    /// This method must randomly change some aspects of the solution.
    /// Ideally it should not change it so drastically that the entire "character"
    /// of the solution is different afterward, but drastically enough
    /// that it will produce notably different results when evaluated.
    fn mutate(&mut self);
}

/// A single run of an evolutionary algorithm.
pub struct Evolution<T, Alg, Hof, Stat>
where
    T: Solution,
    Alg: Algorithm<T>,
    Hof: HallOfFame<T>,
    Stat: GenerationStats<T>,
{
    population: Vec<Cached<T>>,
    algorithm: Alg,
    hall_of_fame: Hof,
    stats: Vec<Stat>,
}

impl<T, Alg, Hof, Stat> Evolution<T, Alg, Hof, Stat>
where
    T: Solution,
    Alg: Algorithm<T>,
    Hof: HallOfFame<T>,
    Stat: GenerationStats<T>,
{
    /// Create a new [`Evolution`] with the specified algorithm and hall of fame.
    pub fn new(algorithm: Alg, hall_of_fame: Hof) -> Self {
        Evolution {
            population: Vec::n_from_function(algorithm.pop_size(), Cached::generate),
            algorithm,
            hall_of_fame,
            stats: Vec::new(),
        }
    }

    /// Run the algorithm for `n_gens` generations. Consumes the `Evolution` instance.
    ///
    /// Returns
    /// =======
    /// Returns an instance of [`Log`] containing the hall of fame and collected statistics for the run.
    ///
    /// [`Log`]: ./struct.Log.html
    pub fn run_for(self, n_gens: usize) -> Log<T, Hof, Stat> {
        self.run_for_with(n_gens, |_| {})
    }

    /// Run the algorithm until the provided `predicate` closure returns `true`.
    ///  
    /// [`Log`]: ./struct.Log.html
    pub fn run_until<F>(self, predicate: F) -> Log<T, Hof, Stat>
    where
        F: FnMut(Generation<T, Hof, Stat>) -> bool,
    {
        self.run_until_with(predicate, |_| {})
    }

    /// Run the algorithm for `n_gens` generations, calling the provided closure for each generation.
    /// This can be used to hook into external logging, a progress bar, or anything else
    /// that you want to execute interleaved with the algorithm.
    ///
    /// The closure is passed three arguments:
    /// - the generation number (starting from 0)
    /// - an immutable slice of that generation's population
    /// - a reference to this run's `Log` instance
    pub fn run_for_with<F>(mut self, n_gens: usize, mut callback: F) -> Log<T, Hof, Stat>
    where
        F: FnMut(Generation<T, Hof, Stat>),
    {
        for generation in 0..n_gens {
            par_evaluate(&self.population);
            self.hall_of_fame.record(&self.population);
            let stat = Stat::analyze(&self.population);
            callback(Generation {
                gen: generation,
                pop: &self.population,
                hall_of_fame: &self.hall_of_fame,
                stats: &stat,
            });
            self.stats.push(stat);

            self.algorithm.step(&mut self.population);
        }

        Log {
            hall_of_fame: self.hall_of_fame,
            stats: self.stats,
            final_population: self.population,
        }
    }

    /// Run the algorithm until the provided `predicate` closure returns `true`,
    /// calling the provided `callback` closure for each generation.
    /// Works the same way as [`.run_until()`] and [`.run_for_with()`].
    ///
    /// [`.run_until()`]: ./struct.Evolution.html#method.run_until
    /// [`.run_for_with()`]: .struct.Evolution.html#method.run_for_with
    pub fn run_until_with<F, G>(mut self, mut predicate: F, mut callback: G) -> Log<T, Hof, Stat>
    where
        F: FnMut(Generation<T, Hof, Stat>) -> bool,
        G: FnMut(Generation<T, Hof, Stat>),
    {
        let mut generation = 0;
        let mut stat: Stat;

        par_evaluate(&self.population);
        self.hall_of_fame.record(&self.population);
        stat = Stat::analyze(&self.population);

        while !predicate(Generation {
            gen: generation,
            pop: &self.population,
            hall_of_fame: &self.hall_of_fame,
            stats: &stat,
        }) {
            callback(Generation {
                gen: generation,
                pop: &self.population,
                hall_of_fame: &self.hall_of_fame,
                stats: &stat,
            });
            self.stats.push(stat);

            generation += 1;

            self.algorithm.step(&mut self.population);
            par_evaluate(&self.population);
            self.hall_of_fame.record(&self.population);
            stat = Stat::analyze(&self.population);
        }

        Log {
            hall_of_fame: self.hall_of_fame,
            stats: self.stats,
            final_population: self.population,
        }
    }
}

/// Container type for the results of a run
pub struct Log<T, Hof, Stat>
where
    T: Solution,
    Hof: HallOfFame<T>,
    Stat: GenerationStats<T>,
{
    /// The population after the run.
    pub final_population: Vec<Cached<T>>,
    /// The hall of fame that has recorded every solution in every generation of the run.
    pub hall_of_fame: Hof,
    /// Statistics for each generation.
    pub stats: Vec<Stat>,
}

/// Container type passed to callbacks
#[derive(Clone, Copy)]
pub struct Generation<'a, T, Hof, Stat>
where
    T: Solution,
    Hof: HallOfFame<T>,
    Stat: GenerationStats<T>,
{
    /// The index of the generation this instance refers to, i.e. the first generation
    /// before any evolution takes place would have `gen == 0`.
    pub gen: usize,
    /// A reference to the population as of the current generation.
    pub pop: &'a [Cached<T>],
    /// A reference to the hall of fame as of the current generation.
    pub hall_of_fame: &'a Hof,
    /// The calculated statistics for the generation this instance refers to.
    pub stats: &'a Stat,
}
