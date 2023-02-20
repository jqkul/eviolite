use std::mem::MaybeUninit;
use std::ops::Deref;
use std::ptr::read;

use crate::Fitness;

#[derive(Clone, Copy, PartialEq)]
pub struct MultiObjective<const M: usize> {
    pub weighted: [f64; M],
}

impl<const M: usize> Fitness for MultiObjective<M> {
    fn collapse(&self) -> f64 {
        let mut result: f64 = 0.0;
        for i in 0..M {
            result += self.weighted[i];
        }
        result
    }
}

impl<const M: usize> MultiObjective<M> {
    pub fn builder(weights: [f64; M]) -> impl Fn([f64; M]) -> Self {
        move |values: [f64; M]| MultiObjective {
            weighted: {
                // Using MaybeUninit is slightly faster than writing zeros, especially for large M.
                let mut arr: [MaybeUninit<f64>; M] = unsafe { MaybeUninit::uninit().assume_init() };
                for i in 0..M {
                    arr[i].write(weights[i] * values[i]);
                }
                // This is a workaround for mem::transmute() not working on arrays of generic size.
                // It's completely safe because every element in the array was just initialized.
                let arr_ptr = &arr as *const [MaybeUninit<f64>; M] as *const [f64; M];
                unsafe { read(arr_ptr) }
            },
        }
    }

    pub fn unweighted(values: [f64; M]) -> Self {
        MultiObjective { weighted: values }
    }
}

impl<const M: usize> Deref for MultiObjective<M> {
    type Target = [f64; M];
    fn deref(&self) -> &Self::Target {
        &self.weighted
    }
}
