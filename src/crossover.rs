//! Commonly used crossover operators for ndarrays
//! 
//! This module contains functions that apply some commonly used crossover operators
//! to two ndarrays of arbitrary shape and containing an arbitrary type.
//! All operations are done in place, and cloning only occurs when it is strictly necessary,
//! i.e. when both offspring need to own the same information from one of the parents.

use std::{mem, collections::HashSet};

use ndarray::prelude::*;

use rand::Rng;
use rand::seq::IteratorRandom;

use crate::repro_rng::thread_rng;

/// Swap one random element
/// 
/// Swap one randomly chosen elements between two arrays with the same shape.
pub fn swap_one<T, D>(array1: &mut Array<T, D>, array2: &mut Array<T, D>) where D: Dimension {
    debug_assert_eq!(array1.len(), array2.len());

    let target = thread_rng().gen_range(0..array1.len());
    let mut i: usize = 0;

    azip!((a in array1, b in array2) {
        if i == target {
            mem::swap(a, b);
        }
        i += 1;
    });
}

/// Swap any number of random elements
/// 
/// Swap `n_swaps` randomly chosen elements between two arrays with the same shape.
/// 
/// Panics
/// ======
/// Panics if `n_swaps > size`, where `size` is the length of the input arrays irrespective
/// of shape, as this makes it impossible to choose `n_swaps` distinct elements. 
pub fn swap_n<T, D>(n_swaps: usize, array1: &mut Array<T, D>, array2: &mut Array<T, D>) where D: Dimension {
    debug_assert_eq!(array1.len(), array2.len());

    assert!(n_swaps <= array1.len(), "n_swaps must be less than or equal to the array size");
    
    let targets: HashSet<usize> = {
        let mut hs = HashSet::with_capacity(n_swaps);
        for target in (0..array1.len()).choose_multiple(&mut thread_rng(), n_swaps) {
            debug_assert!(hs.insert(target));
        }
        hs
    };
    let mut i: usize = 0;

    azip!((a in array1, b in array2) {
        if targets.contains(&i) {
            mem::swap(a, b);
        }
        i += 1;
    });
}

/// Apply a random chance of `indpb` to swap each element in two arrays with the same shape.
pub fn swap_each_random<T, D>(indpb: f64, array1: &mut Array<T, D>, array2: &mut Array<T, D>) where D: Dimension {
    let mut rng = thread_rng();
    azip!((a in array1, b in array2) {
        if rng.gen_bool(indpb) {
            mem::swap(a, b);
        }
    })
}

/// Perform standard
/// [uniform crossover](https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)#Uniform_crossover),
/// also called discrete recombination.
/// 
/// This function does the same thing as [`uniform_with_ratio`], but with a standard mixing ratio of `0.5`,
/// which gives an equal probability of taking from either input array.
/// 
/// [`uniform_with_ratio`]: ./fn.uniform_with_ratio.html
pub fn uniform<T, D>(array1: &mut Array<T, D>, array2: &mut Array<T, D>) where T: Clone, D: Dimension {
    uniform_with_ratio(0.5, array1, array2);
}

/// Perform [uniform crossover](https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)#Uniform_crossover),
/// also called discrete recombination, with a custom mixing ratio.
/// 
/// This function walks through each pair `(a, b)` of elements in the input arrays.
/// It rolls an independent chance for both `a` and `b` to decide whether they should
/// be kept as they are, or set to the other one.
/// 
/// `mixing_ratio` represents the probability of choosing from `array1` vs `array2`. 
/// Values closer to zero will make the process more likely to take from `array1`,
/// while values closer to one will make it more likely to take from `array2`.
/// 
/// Panics
/// ======
/// Panics if `mixing_ratio < 0` or `mixing_ratio > 1`.
pub fn uniform_with_ratio<T, D>(mixing_ratio: f64, array1: &mut Array<T, D>, array2: &mut Array<T, D>) where T: Clone, D: Dimension {
    let mut rng = thread_rng();
    azip!((a in array1, b in array2) {
        // false represents taking from a, true represents taking from b
        let a_choice = rng.gen_bool(mixing_ratio);
        let b_choice = rng.gen_bool(mixing_ratio);
        if a_choice && !b_choice {
            // if a chose b and b chose a, swap them efficiently
            mem::swap(a, b);
        } else if !a_choice && !b_choice {
            // if both chose a, put a clone of a into b
            b.clone_from(&a);
        } else if a_choice && b_choice {
            // if both chose b, put a clone of b into a
            a.clone_from(&b);
        } // if a chose a and b chose b, we don't need to do anything
    });
}

/// One-point crossover
/// 
/// Randomly chooses a pivot index in the range `0..size - 1`,
/// where `size` is the input arrays' size irrespective of shape,
/// and swaps the arrays' elements for every element after the pivot.
/// 
/// Panics
/// ======
/// Panics if `size <= 1`, as this operation makes no sense on empty or single-element arrays.
pub fn one_point<T, D>(array1: &mut Array<T, D>, array2: &mut Array<T, D>) where D: Dimension {
    n_point(1, array1, array2);
}

/// Two-point crossover
/// 
/// Randomly chooses two distinct pivot indices in the range `0..size - 1`,
/// where `size` is the input arrays' size irrespective of shape,
/// and swaps the arrays' elements for every element between the two pivots.
/// 
/// Panics
/// ======
/// Panics if `size <= 2`, as this operation makes no sense on arrays with less than three elements.
pub fn two_point<T, D>(array1: &mut Array<T, D>, array2: &mut Array<T, D>) where D: Dimension {
    n_point(2, array1, array2);
}

/// *n*-point crossover
/// 
/// Randomly chooses `n_pivots` distinct pivot indices in the range `0..size - 1`,
/// where `size` is the input arrays' size irrespective of shape, 
/// and swaps the arrays' elements for every element between successive pairs of pivots.
/// 
/// Panics
/// ======
/// Panics if `n_pivots >= size`, as this makes it impossible to choose `n_pivots` distinct pivots.
pub fn n_point<T, D>(n_pivots: usize, array1: &mut Array<T, D>, array2: &mut Array<T, D>) where D: Dimension {
    debug_assert_eq!(array1.len(), array2.len());

    assert!(n_pivots < array1.len(), "n_pivots must be less than the array size");

    let pivots: HashSet<usize> = {
        let mut hs = HashSet::with_capacity(n_pivots);
        for pivot in (0..array1.len() - 1).choose_multiple(&mut thread_rng(), n_pivots) {
            hs.insert(pivot);
        }
        hs
    };
    let mut i: usize = 0;
    let mut swap = false;

    azip!((a in array1, b in array2) {
        if swap {
            mem::swap(a, b);
        }
        if pivots.contains(&i) {
            swap = !swap;
        }
        i += 1;
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swap_one() {
        let mut a = Array::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
        let mut b = -a.clone();

        swap_one(&mut a, &mut b);

        assert_eq!(count_neg(&a), 1);
        assert_eq!(count_neg(&b), 8);
    }

    #[test]
    fn test_swap_n() {
        let mut a = Array::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
        let mut b = -a.clone();

        let n = thread_rng().gen_range(0..=9);

        swap_n(n, &mut a, &mut b);

        assert_eq!(count_neg(&a), n);
        assert_eq!(count_neg(&b), 9 - n);
    }

    fn count_neg<D: Dimension>(arr: &Array<f64, D>) -> usize {
        arr.mapv(|x| if x < 0.0 {1} else {0}).sum()
    }

    #[test]
    fn test_n_point() {
        let mut a = Array::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
        let mut b = -a.clone();

        n_point(3, &mut a, &mut b);
    }
}