use std::{ops::Mul, path::Path};
use std::fmt::Debug;

use crate::{
    Solution,
    fitness::MultiObjective,
    select::utils::bring_indices_to_front, Fitness
};

// Implements the Best Order Sort algorithm for nondominated sorting from [Roy2016]
pub fn rank_nondominated_best_order<T, const M: usize>(pop: &[T]) -> Vec<usize> where T: Debug + Solution<Fitness = MultiObjective<M>> {
    // Algorithm 1: Initialization
    let popsize = pop.len();
    let mut l = vec![vec![Vec::<usize>::new(); M]; popsize];
    let mut c = vec![(0..M).collect::<Vec<_>>(); popsize];
    let mut is_ranked = vec![false; popsize];
    let mut solutions_completed: usize = 0;
    let mut rank_count: usize = 1;
    let mut rank = vec![0usize; popsize];

    let mut q: Vec<Vec<usize>> = Vec::with_capacity(M);
    for j in 0..M {
        q.push({
            let mut q_j = (0..popsize).collect::<Vec<_>>();
            q_j.sort_by(|&a, &b|
                f64::total_cmp(
                    &pop[b].evaluate().weighted[j],
                    &pop[a].evaluate().weighted[j]
                )
            );
            q_j
        });
    }

    // Algorithm 2: Main Loop
    for i in 0..popsize {
        for j in 0..M {
            let s = q[j][i];
            println!("main loop: s = {}", s);
            c[s].retain(|&k| k != j);
            if is_ranked[s] {
                println!("{} is already ranked", s);
                l[rank[s]][j].push(s);
            } else {
                // Algorithm 3: FindRank
                let mut done = false;
                for k in 0..rank_count {
                    let mut check = false;
                    for &t in l[k][j].iter() {
                        check = cmp_dom(&pop[s], &pop[t]) == DomOrdering::BOverA;
                        println!("comparing s={} -> {:?} to t={} -> {:?}; check = {}", s, &pop[s], t, &pop[t], check);
                        if check {
                            break;
                        }
                    }
                    if !check {
                        rank[s] = k;
                        done = true;
                        l[rank[s]][j].push(s);
                        println!("found {} to be in rank {}", s, rank[s]);
                        break;
                    }
                }
                if !done {
                    rank[s] = rank_count;
                    rank_count += 1;
                    l[rank[s]][j].push(s);
                    println!("creating new rank {} for {}", rank[s], s);
                }

                is_ranked[s] = true;
                solutions_completed += 1;
            }
        }
        if solutions_completed == popsize {
            break;
        }
    }

    rank
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DomOrdering {
    AOverB,
    BOverA,
    Neither
}

fn cmp_dom<T, const M: usize>(a: &T, b: &T) -> DomOrdering where T: Solution<Fitness = MultiObjective<M>> {
    cmp_dom_f64_slices(&a.evaluate().weighted, &b.evaluate().weighted)
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
    use rand::{seq::SliceRandom, Rng};

    use crate::{repro_thread_rng::thread_rng, utils::NFromFunction};

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

        fn crossover(_: &mut Self, _: &mut Self) { unreachable!() }
        fn mutate(&mut self) { unreachable!() }
    }

    #[test]
    fn test_rank_nondominated() {
        let pop = vec![
            Foo([0.6, 0.6]),   Foo([0.0, 1.0]), Foo([0.75, 0.25]),
            Foo([0.25, 0.75]), Foo([1.0, 0.0]), Foo([0.9, 0.9]),
        ];
        
        let rank = rank_nondominated_best_order(&pop);

        assert_eq!(rank, vec![1, 0, 1, 1, 0, 0]);
    }

    // #[test]
    // fn test_best_order_sort() {
    //     let pop = Vec::n_from_function(25, Foo::generate);

    //     let rankings = rank_nondominated_best_order(&pop);
    //     let mut ranked: Vec<_> = rankings.into_iter().zip(pop.into_iter()).collect();
    //     ranked.sort_unstable_by_key(|(rank, _)| *rank);
    //     for (rank, foo) in ranked.into_iter() {
    //         println!("{}: ({:.3}, {:.3})", rank, foo.0[0], foo.0[1]);
    //     }
    // }

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
}