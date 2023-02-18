use std::mem::MaybeUninit;
use std::ptr::read;

use crate::traits::Fitness;

impl Fitness for f64 {
    fn collapse(&self) -> f64 {
        *self
    }
}

#[derive(Clone, Copy)]
pub struct MultiObjective<const N: usize> {
    pub weighted: [f64; N],
}

impl<const N: usize> MultiObjective<N> {
    pub fn builder(weights: [f64; N]) -> impl Fn([f64; N]) -> Self {
        move |values: [f64; N]| MultiObjective {
            weighted: {
                // Using MaybeUninit is slightly faster than writing zeros, especially for large N.
                let mut arr: [MaybeUninit<f64>; N] = unsafe {
                    MaybeUninit::uninit().assume_init()
                };
                for i in 0..N {
                    arr[i].write(weights[i] * values[i]);
                }
                // This is a workaround for mem::transmute() not working on arrays of generic size.
                // It's completely safe because every element in the array was just initialized.
                let arr_ptr = &arr as *const [MaybeUninit<f64>; N] as *const [f64; N];
                unsafe { read(arr_ptr) }
            }
        }
    }

    pub fn unweighted(values: [f64; N]) -> Self {
        MultiObjective { weighted: values }
    }
}

impl<const N: usize> Fitness for MultiObjective<N> {
    fn collapse(&self) -> f64 {
        let mut result: f64 = 0.0;
        for i in 0..N {
            result += self.weighted[i];
        }
        result
    }
}
