use rand::seq::index::sample;

use crate::pop::{Population, Visitor};
use crate::{Fitness, Individual};
use crate::repro_thread_rng::thread_rng;

pub trait Select<T: Individual> {
    fn select(&self, n: usize, population: &mut Vec<T>);
}

// impl<T, S> Visitor<T> for S
// where
//     T: Individual,
//     S: Select<T>,
// {
//     fn visit(&mut self, population: &mut Vec<T>) {
//         self.select(population);
//     }
// }

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
    T: Individual,
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

pub fn find_best<T>(pop: &[T]) -> &T where T: Individual {
    let mut best_idx: usize = 0;
    for i in 1..pop.len() {
        if pop[i].collapsed() > pop[best_idx].collapsed() {
            best_idx = i;
        }
    }
    &pop[best_idx]
}

// Mutate `vec` in place, keeping only the elements at the positions
// specified by `indices`. Clones elements only for duplicate indices.
fn retain_indices<T>(vec: &mut Vec<T>, mut indices: Vec<usize>)
where
    T: Clone,
{
    let n_indices = indices.len();
    indices.sort();

    vec.swap(indices[0], 0);
    let mut swap_to: usize = 1;
    let mut i: usize = 1;
    while i < indices.len() {
        if indices[i] == indices[i - 1] {
            indices.push(vec.len());
            vec.push(vec[swap_to - 1].clone());
        } else {
            vec.swap(indices[i], swap_to);
            swap_to += 1;
        }
        i += 1;
    }
    vec.truncate(n_indices);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retain_indices() {
        let mut myvec = vec!['a', 'b', 'c', 'd', 'e', 'f', 'g'];
        retain_indices(&mut myvec, vec![5, 4, 5, 1]);
        myvec.sort();
        assert_eq!(myvec, vec!['b', 'e', 'f', 'f']);

        let mut myvec2 = vec!['a', 'b', 'c', 'd', 'e', 'f', 'g'];
        retain_indices(&mut myvec2, vec![2, 4, 1, 2, 1, 2, 1]);
        myvec2.sort();
        assert_eq!(myvec2, vec!['b', 'b', 'b', 'c', 'c', 'c', 'e']);

        let mut myvec3 = vec!['a', 'b', 'c', 'd', 'e', 'f', 'g'];
        retain_indices(&mut myvec3, vec![0; 10]);
        assert_eq!(myvec3, vec!['a'; 10]);
    }
}
