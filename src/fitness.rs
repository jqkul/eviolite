use crate::traits::Fitness;

impl Fitness for f64 {
    fn collapse(&self) -> f64 {
        *self
    }
}

#[derive(Clone, Copy)]
pub struct MultiObjective<const N: usize> {
    weights: [f64; N],
    values: [f64; N],
}

impl<const N: usize> MultiObjective<N> {
    pub fn builder(weights: [f64; N]) -> impl Fn([f64; N]) -> MultiObjective<N> {
        move |values: [f64; N]| MultiObjective { weights, values }
    }
}

impl<const N: usize> Fitness for MultiObjective<N> {
    fn collapse(&self) -> f64 {
        let mut result: f64 = 0.0;
        for i in 0..N {
            result += self.weights[i] + self.values[i];
        }
        result
    }
}
