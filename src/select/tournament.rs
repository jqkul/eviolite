use rand::seq::index::sample;

use crate::pop::{Population, Visitor};
use crate::{Fitness, Solution};
use crate::repro_thread_rng::thread_rng;
use crate::select::{Select, utils::*};

#[derive(Clone, Copy)]
pub struct Tournament {
    pub round_size: usize,
}

/// Selects the best individual out of `round_size` randomly chosen distinct individuals,
/// `n_rounds` times. Applying this to a population will always leave `n_rounds` members.
impl Tournament {
    pub fn new(round_size: usize) -> Self {
        if round_size == 0 {
            panic!("Tournament needs at least one participant per round");
        }
        Tournament {
            round_size,
        }
    }
}

impl<T> Select<T> for Tournament
where
    T: Solution,
{
    fn select(&self, n_rounds: usize, population: &mut Vec<T>) {
        let mut rng = thread_rng();
        let len = population.len();
        let mut winners: Vec<usize> = Vec::with_capacity(n_rounds);

        // Run `n_rounds` rounds. Each round does the following:
        // - randomly sample `round_size` distinct individuals from the population
        // - choose the individual with the highest fitness as the winner
        // - append the winner's index to `winners`
        for _ in 0..n_rounds {
            winners.push({
                let mut participants = sample(&mut rng, len, self.round_size).into_iter();
                let mut curr_max = participants.next().unwrap();
                for idx in participants {
                    if population[idx].collapsed() > population[curr_max].collapsed() {
                        curr_max = idx;
                    }
                }
                curr_max
            });
        }

        // Delete every individual that didn't win a tournament
        retain_indices(population, winners);
    }
}