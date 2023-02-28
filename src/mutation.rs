//! Commonly used mutation operators for ndarrays
//! 
//! This module contains functions that apply some commonly used mutation operators
//! to ndarrays of arbitrary shape.

use ndarray::prelude::*;

use rand::Rng;
use rand_distr::{Distribution, Uniform, StandardNormal};

use num_traits::Float;

use crate::repro_rng::thread_rng;

/// Apply Gaussian noise to random elements.
/// 
/// This function does a random roll of probability `indpb` for each element in the input array.
/// If the roll succeeds, it adds noise to that element, drawn from a Gaussian/normal distribution 
/// with mean 0 and standard deviation `stdev`.
/// 
/// Panics
/// ======
/// Panics if `stdev` is infinite, `NaN`, or negative.
/// 
/// Also panics if adding noise to an element would cause it to overflow or underflow,
/// though this is pretty unlikely for most use cases.
pub fn gaussian<D, F>(arr: &mut Array<F, D>, indpb: f64, stdev: F) where F: Float + std::ops::AddAssign + std::fmt::Debug, D: Dimension, StandardNormal: Distribution<F> {
    assert!(stdev.is_finite() && stdev >= F::zero(), "{:?} is not a valid standard deviation", stdev);

    let mut rng = thread_rng();
    arr.map_inplace(|elem| {
        if rng.gen_bool(indpb) {
            *elem += stdev * StandardNormal.sample(&mut rng);
        }
    })
}

/// Apply Gaussian noise to random elements with different parameters for each element.
/// 
/// This function does the same thing as [`gaussian()`], but with
/// a different mutation probability and standard deviation for
/// each element in the array.
/// 
/// This allows you to customize how much each element can be mutated,
/// as well as make some array elements unable to be mutated
/// by setting the corresponding elements in the `stdevs` array to zero.
/// 
/// Panics
/// ======
/// Panics if any element of `stdevs` is infinite, `NaN`, or negative.
/// 
/// Also panics if adding noise to an element would cause it to overflow,
/// though this is pretty unlikely for most use cases.
pub fn gaussian_with<F, D>(arr: &mut Array<F, D>, probabilities: &Array<f64, D>, stdevs: &Array<F, D>) where F: Float + std::ops::AddAssign + std::fmt::Debug, D: Dimension, StandardNormal: Distribution<F> {
    let mut rng = thread_rng();
    azip!((elem in arr, &stdev in stdevs, &indpb in probabilities) {
        assert!(stdev.is_finite() && stdev >= F::zero(), "{:?} is not a valid standard deviation", stdev);
        if rng.gen_bool(indpb) {
            *elem += stdev * StandardNormal.sample(&mut rng);
        }
    });
}

/// Randomly swap some elements of an array.
/// 
/// This function does a random roll of probability `indpb` for each element in the input array.
/// If the roll succeeds, it randomly chooses another element from the input array and swaps the two.
/// 
/// Panics
/// ======
/// Panics if the input array is not contiguous.
/// This limitation may be removed in a future release.
pub fn shuffle<T, D>(arr: &mut Array<T, D>, indpb: f64) where D: Dimension {    
    if let Some(slice) = arr.as_slice_memory_order_mut() {
        let mut rng = thread_rng();
        let len = slice.len();
        let distr = Uniform::new(0, len);
        for i in 0..slice.len() {
            if rng.gen_bool(indpb) {
                slice.swap(i, distr.sample(&mut rng));
            }
        }
    } else {
        panic!("array passed to shuffle must be contiguous");
    }
}