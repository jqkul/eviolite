use std::fmt::Debug;

use crate::{
    Solution,
    Cached,
    fitness::MultiObjective,
    select::{utils::retain_indices, Select},
};


/// NSGA-II selection operator
/// 
/// This struct implements the NSGA-II selection algorithm[^1].
/// This algorithm is designed for multi-objective optimization,
/// and as such only works with solutions whose fitness is a [`MultiObjective`].
/// 
/// [^1]: Deb, Pratap, Agarwal, & Meyarivan.
/// "A fast and elitist multiobjective genetic algorithm: NSGA-II."
/// 2002. <https://doi.org/10.1109/4235.996017>
pub struct NSGA2;

impl<T, const M: usize> Select<T> for NSGA2
where
    T: Solution<Fitness = MultiObjective<M>>,
{
    fn select(&self, k: usize, pop: &mut Vec<Cached<T>>) {
        let indices = self.select_indices(k, pop).0;
        retain_indices(pop, indices);
    }
}

impl NSGA2 {
    pub(crate) fn select_indices<T, const M: usize>(&self, n: usize, pop: &[Cached<T>]) -> (Vec<usize>, ParetoFronts) where T: Solution<Fitness = MultiObjective<M>> {
        debug_assert!(n <= pop.len());

        let pareto = rank_nondominated(pop);

        let mut indices: Vec<usize> = (0..pop.len()).collect();
        indices.sort_unstable_by_key(|&i| pareto.ranks[i]);

        let mut selected: Vec<usize> = Vec::with_capacity(n);

        // Find the ranks that will completely fit in n,
        let mut curr_rank: usize = 0;
        let mut count_sum: usize = 0;
        while count_sum + pareto.counts[curr_rank] < n {
            count_sum += pareto.counts[curr_rank];
            curr_rank += 1;
        }
        // Add the complete ranks to the selection, draining them from the main indices vec
        selected.extend(indices.drain(..count_sum));

        // Cut off the ranks we're not using any of
        indices.truncate(pareto.counts[curr_rank]);

        sort_by_crowding_distance(&mut indices, pop);

        selected.extend_from_slice(&indices[..n - count_sum]);

        (selected, pareto)
    }
}

/// A representation of the nondominated ranks of a population
/// 
/// The set of solutions with a given nondominated rank are also known as a
/// [Pareto front](https://en.wikipedia.org/wiki/Pareto_front),
/// hence the name.
pub struct ParetoFronts {
    /// Each of the members' nondominated ranks
    pub ranks: Vec<usize>,
    /// The size of each rank in order
    pub counts: Vec<usize>,
}

impl ParetoFronts {
    fn new(popsize: usize) -> Self {
        ParetoFronts {
            ranks: vec![0; popsize],
            counts: Vec::new(),
        }
    }

    fn add_ranking(&mut self, idx: usize, rank: usize) {
        self.ranks[idx] = rank;
        if let Some(count) = self.counts.get_mut(rank) {
            *count += 1;
        } else {
            self.counts.resize(rank + 1, 0);
            self.counts[rank] = 1;
        }
    }
}

/// Determine the nondominated rank of every solution in a population
/// 
/// The nondominated rank is a metric used in multi-objective optimization.
/// A solution *dominates* another if it outperforms it in every objective.
/// If a solution is not dominated by any other solution in the population,
/// it is assigned a rank of 0. If a solution is not dominated by any other
/// solution in the set of solutions whose rank is not 0, it is assigned a
/// rank of 1. This continues recursively until every solution in the
/// population has an associated rank.
/// 
/// This function implements the Best Order Sort algorithm for nondominated ranking[^1].
/// 
/// [^1]: Roy, Islam, & Deb.
/// "Best Order Sort: A New Algorithm to Non-dominated Sorting for Evolutionary Multi-objective Optimization."
/// 2016. <https://doi.org/10.1145/2908961.2931684>
pub fn rank_nondominated<T, const M: usize>(pop: &[T]) -> ParetoFronts
where
    T: Solution<Fitness = MultiObjective<M>>,
{
    // Algorithm 1: Initialization
    let popsize = pop.len();
    let mut l = vec![vec![Vec::<usize>::new(); M]; popsize];
    let mut c = vec![(0..M).collect::<Vec<_>>(); popsize];
    let mut is_ranked = vec![false; popsize];
    let mut solutions_completed: usize = 0;
    let mut rank_count: usize = 1;
    let mut pareto = ParetoFronts::new(popsize);

    let mut q: Vec<Vec<usize>> = Vec::with_capacity(M);
    for j in 0..M {
        q.push({
            let mut q_j = (0..popsize).collect::<Vec<_>>();
            q_j.sort_unstable_by(|&a, &b| {
                f64::total_cmp(&pop[b].evaluate()[j], &pop[a].evaluate()[j])
            });
            q_j
        });
    }

    // Algorithm 2: Main Loop
    for i in 0..popsize {
        for j in 0..M {
            let s = q[j][i];
            c[s].retain(|&k| k != j);
            if is_ranked[s] {
                l[pareto.ranks[s]][j].push(s);
            } else {
                // Algorithm 3: FindRank
                let mut done = false;
                for k in 0..rank_count {
                    let mut check = false;
                    for &t in l[k][j].iter() {
                        check = cmp_dom(&pop[s], &pop[t]) == DomOrdering::BOverA;
                        if check {
                            break;
                        }
                    }
                    if !check {
                        pareto.add_ranking(s, k);
                        done = true;
                        l[k][j].push(s);
                        break;
                    }
                }
                if !done {
                    pareto.add_ranking(s, rank_count);
                    l[rank_count][j].push(s);
                    rank_count += 1;
                }

                is_ranked[s] = true;
                solutions_completed += 1;
            }
        }
        if solutions_completed == popsize {
            break;
        }
    }

    pareto
}

pub fn sort_by_crowding_distance<T, const M: usize>(front: &mut [usize], pop: &[Cached<T>])
where
    T: Solution<Fitness = MultiObjective<M>>,
{
    let fit = |idx: usize, m: usize| Cached::fit(&pop[idx], m);

    let frontsize = front.len();
    let mut distances: Vec<f64> = vec![0.0; frontsize];
    let mut front_enumerated = {
        let mut v: Vec<(usize, usize)> = Vec::with_capacity(frontsize);
        for i in 0..frontsize {
            v.push((i, front[i]));
        }
        v
    };
    for m in 0..M {
        front_enumerated
            .sort_unstable_by(|(_, a), (_, b)| f64::total_cmp(&fit(*a, m), &fit(*b, m)));
        let min_fit = fit(front_enumerated[0].1, m);
        let max_fit = fit(front_enumerated[frontsize - 1].1, m);
        let fit_range = max_fit - min_fit;
        distances[front_enumerated[0].0] = f64::INFINITY;
        distances[front_enumerated.last().unwrap().0] = f64::INFINITY;
        for i in 1..frontsize - 1 {
            let (j, _) = front_enumerated[i];
            let prev_fit = fit(front_enumerated[i - 1].1, m);
            let next_fit = fit(front_enumerated[i + 1].1, m);
            distances[j] += (next_fit - prev_fit) / fit_range;
        }
    }

    front_enumerated
        .sort_unstable_by(|(i, _), (j, _)| f64::total_cmp(&distances[*j], &distances[*i]));

    for i in 0..frontsize {
        front[i] = front_enumerated[i].1;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DomOrdering {
    AOverB,
    BOverA,
    Neither,
}

fn cmp_dom<T, const M: usize>(a: &T, b: &T) -> DomOrdering
where
    T: Solution<Fitness = MultiObjective<M>>,
{
    cmp_dom_f64_slices(&a.evaluate(), &b.evaluate())
}

fn cmp_dom_f64_slices<const M: usize>(a: &[f64; M], b: &[f64; M]) -> DomOrdering {
    let mut a_win = false;
    let mut b_win = false;
    for i in 0..M {
        if b[i] > a[i] {
            b_win = true;
        // no need for another condition here because
        // floats are absurdly unlikely to compare equal
        } else {
            a_win = true;
        }
    }
    if a_win && !b_win {
        DomOrdering::AOverB
    } else if b_win && !a_win {
        DomOrdering::BOverA
    } else {
        DomOrdering::Neither
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use crate::{
        testutils::*,
        utils::NFromFunction,
    };

    use super::*;

    #[test]
    fn test_rank_nondominated() {
        let pop = vec![
            Foo([0.6, 0.6]),
            Foo([0.0, 1.0]),
            Foo([0.75, 0.25]),
            Foo([0.25, 0.75]),
            Foo([1.0, 0.0]),
            Foo([0.9, 0.9]),
        ];

        let pareto = rank_nondominated(&pop);
        let rankings = pareto.ranks;
        let counts = pareto.counts;

        assert_eq!(rankings, vec![1, 0, 1, 1, 0, 0]);
        assert_eq!(counts, vec![3, 3]);
    }

    #[test]
    fn test_cmp_dom() {
        use DomOrdering::*;

        let arr1 = [5.0f64, 5.0, 5.0];
        let arr2 = [-2.0f64, 3.0, 4.9];
        let arr3 = [-1.9f64, 2.0, 3.1];

        assert_eq!(cmp_dom_f64_slices(&arr1, &arr2), AOverB);
        assert_eq!(cmp_dom_f64_slices(&arr3, &arr1), BOverA);
        assert_eq!(cmp_dom_f64_slices(&arr2, &arr3), Neither);
    }

    // The following tests will always pass. They are intended for use with
    // --nocapture, producing human-readable output for sanity checking. They
    // are not part of the automated testing process.

    #[test]
    fn test_best_order_sort_1d() {
        let pop = Vec::n_from_function(100, One::generate);

        let rankings = rank_nondominated(&pop).ranks;
        let mut ranked: Vec<_> = rankings.into_iter().zip(pop.into_iter()).collect();
        ranked.sort_unstable_by_key(|(rank, _)| *rank);
        for (rank, members) in &ranked.into_iter().group_by(|(rank, _)| *rank) {
            print!("rank {}: ", rank);
            for (_, x) in members {
                print!("{:.3} ", x.0);
            }
            println!();
        }
    }

    #[test]
    fn test_best_order_sort_3d() {
        let pop = Vec::n_from_function(1000, Bar::generate);

        let rankings = rank_nondominated(&pop).ranks;
        let mut ranked: Vec<_> = rankings.into_iter().zip(pop.into_iter()).collect();
        ranked.sort_unstable_by_key(|(rank, _)| *rank);
        for (rank, members) in &ranked.into_iter().group_by(|(rank, _)| *rank) {
            print!("rank {}: ", rank);
            for (_, bar) in members {
                print!("({:.3}, {:.3}, {:.3}) ", bar.0[0], bar.0[1], bar.0[2]);
            }
            println!();
        }
    }
}
