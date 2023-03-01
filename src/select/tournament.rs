use std::cmp::Ordering;

use rand::seq::index::sample;

use crate::repro_rng::thread_rng;
use crate::select::{utils::*, Select};
use crate::{Cached, Solution};

use super::Stochastic;

/// Simple tournament selection
///
/// This type's `.select()` method runs a series of tournaments.
/// For each tournament, it randomly chooses `round_size` solutions from the population
/// and chooses the one with the highest fitness.
/// The new population is then composed of the winners.
#[derive(Clone, Copy)]
pub struct Tournament {
    round_size: usize,
}

impl Stochastic for Tournament {}

impl Tournament {
    /// Create a new `Tournament` with the provided round size.
    ///
    /// # Panics
    ///
    /// Panics if `round_size` is 0 as this leads to an invalid state.
    pub fn new(round_size: usize) -> Self {
        if round_size == 0 {
            panic!("Tournament needs at least one participant per round");
        }
        Tournament { round_size }
    }

    /// Get this `Tournament`'s round size.
    pub fn round_size(&self) -> usize {
        self.round_size
    }

    /// Run a single round on `population`, using `cmp` to compare solutions.
    ///
    /// Returns a reference to the winner.
    pub fn round<'a, T: Solution>(
        &self,
        population: &'a [Cached<T>],
        cmp: impl Fn(&Cached<T>, &Cached<T>) -> Ordering,
    ) -> &'a Cached<T> {
        &population[self.round_idx(population, cmp)]
    }
}

impl<T, F> Select<T> for Tournament
where
    T: Solution<Fitness = F>,
    F: Into<f64>,
{
    fn select(&self, n_rounds: usize, pop: &mut Vec<Cached<T>>) {
        let mut winners: Vec<usize> = Vec::with_capacity(n_rounds);

        // Run `n_rounds` rounds. Each round does the following:
        // - randomly sample `round_size` distinct individuals from the population
        // - choose the individual with the highest fitness as the winner
        // - append the winner's index to `winners`
        for _ in 0..n_rounds {
            winners.push(self.round_idx(pop, |a, b| {
                f64::partial_cmp(&a.evaluate().into(), &b.evaluate().into()).unwrap()
            }));
        }

        // Delete every individual that didn't win a tournament
        retain_indices(pop, winners);
    }
}

impl Tournament {
    pub(crate) fn round_idx<T: Solution>(
        &self,
        pop: &[Cached<T>],
        cmp: impl Fn(&Cached<T>, &Cached<T>) -> Ordering,
    ) -> usize {
        let mut rng = thread_rng();
        let mut participants = sample(&mut rng, pop.len(), self.round_size).into_iter();
        let mut curr_max = participants.next().unwrap();
        for idx in participants {
            if cmp(&pop[idx], &pop[curr_max]).is_gt() {
                curr_max = idx;
            }
        }
        curr_max
    }
}
