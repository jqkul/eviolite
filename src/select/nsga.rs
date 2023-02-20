use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::Debug;
use std::{ops::Mul, path::Path};

use rayon::slice::ParallelSliceMut;

use crate::Cache;
use crate::fitness::MultiObjective;
use crate::{
    Fitness,
    select::{utils::retain_indices, Select},
    Solution,
};

pub struct NSGA2;

impl<T, const M: usize> Select<T> for NSGA2
where
    T: Solution<Fitness = MultiObjective<M>>,
{
    fn select(&self, n: usize, pop: &mut Vec<Cache<T>>) {
        debug_assert!(n <= pop.len());

        let pareto = rank_nondominated(pop);

        let mut indices: Vec<usize> = (0..pop.len()).collect();
        indices.sort_unstable_by_key(|&i| pareto.rankings[i]);

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

        sort_by_crowding_distance(&mut indices, &pop);

        selected.extend_from_slice(&indices[..n - count_sum]);

        retain_indices(pop, selected);
    }
}

pub struct ParetoFronts {
    pub rankings: Vec<usize>,
    pub counts: Vec<usize>,
}

impl ParetoFronts {
    fn new(popsize: usize) -> Self {
        ParetoFronts {
            rankings: vec![0; popsize],
            counts: Vec::new(),
        }
    }

    fn add_ranking(&mut self, idx: usize, rank: usize) {
        self.rankings[idx] = rank;
        if let Some(count) = self.counts.get_mut(rank) {
            *count += 1;
        } else {
            self.counts.resize(rank + 1, 0);
            self.counts[rank] = 1;
        }
    }
}

// Implements the Best Order Sort algorithm for nondominated sorting from [Roy2016]
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
                l[pareto.rankings[s]][j].push(s);
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

fn sort_by_crowding_distance<T, const M: usize>(front: &mut [usize], pop: &[Cache<T>])
where
    T: Solution<Fitness = MultiObjective<M>>,
{
    let fit = |idx: usize, m: usize| Cache::fit(&pop[idx], m);

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
    use rand::{seq::SliceRandom, Rng};

    use crate::{
        repro_thread_rng::{random, thread_rng},
        utils::NFromFunction,
    };

    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Foo([f64; 2]);
    impl Solution for Foo {
        type Fitness = MultiObjective<2>;

        fn generate() -> Self {
            let mut rng = thread_rng();
            Foo([rng.gen(), rng.gen()])
        }

        fn evaluate(&self) -> Self::Fitness {
            MultiObjective::unweighted(self.0)
        }

        fn crossover(_: &mut Self, _: &mut Self) {
            unreachable!()
        }
        fn mutate(&mut self) {
            unreachable!()
        }
    }

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
        let rankings = pareto.rankings;
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
        let pop = Vec::n_from_function(100, f64::generate);

        let rankings = rank_nondominated(&pop).rankings;
        let mut ranked: Vec<_> = rankings.into_iter().zip(pop.into_iter()).collect();
        ranked.sort_unstable_by_key(|(rank, _)| *rank);
        for (rank, members) in &ranked.into_iter().group_by(|(rank, _)| *rank) {
            print!("rank {}: ", rank);
            for (_, x) in members {
                print!("{:.3} ", x);
            }
            println!();
        }
    }

    #[test]
    fn test_best_order_sort_3d() {
        let pop = Vec::n_from_function(1000, Bar::generate);

        let rankings = rank_nondominated(&pop).rankings;
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

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Bar([f64; 3]);
    impl Solution for Bar {
        type Fitness = MultiObjective<3>;

        fn generate() -> Self {
            let mut rng = thread_rng();
            Bar([-rng.gen::<f64>(), -rng.gen::<f64>(), -rng.gen::<f64>()])
        }

        fn evaluate(&self) -> Self::Fitness {
            MultiObjective::unweighted(self.0)
        }

        fn crossover(_: &mut Self, _: &mut Self) {
            unreachable!()
        }
        fn mutate(&mut self) {
            unreachable!()
        }
    }

    impl Solution for f64 {
        type Fitness = MultiObjective<1>;

        fn generate() -> Self {
            random()
        }

        fn evaluate(&self) -> Self::Fitness {
            MultiObjective::unweighted([*self])
        }

        fn crossover(_: &mut Self, _: &mut Self) {
            unreachable!()
        }
        fn mutate(&mut self) {
            unreachable!()
        }
    }
}
