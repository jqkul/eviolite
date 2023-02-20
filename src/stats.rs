use crate::{fitness::MultiObjective, utils::Cached, Solution};

pub trait GenerationStats<T: Solution> {
    fn analyze(generation: &[Cached<T>]) -> Self;
}

impl<T> GenerationStats<T> for () where T: Solution {
    fn analyze(generation: &[Cached<T>]) -> Self {
        ()
    }
}

pub struct FitnessBasicMulti<const M: usize> {
    mean: [f64; M],
    variance: [f64; M],
    stdev: [f64; M],
}

impl<const M: usize> FitnessBasicMulti<M> {
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    pub fn variance(&self) -> &[f64] {
        &self.variance
    }

    pub fn stdev(&self) -> &[f64] {
        &self.stdev
    }
}

impl<T, const M: usize> GenerationStats<T> for FitnessBasicMulti<M> where T: Solution<Fitness = MultiObjective<M>>
{
    fn analyze(generation: &[Cached<T>]) -> Self {
        let len = generation.len() as f64;
        let mut mean = [0.0f64; M];
        let mut variance = [0.0f64; M];
        let mut stdev = [0.0f64; M];

        for m in 0..M {
            mean[m] = generation.iter().map(|ind| Cached::fit(ind, m)).sum::<f64>() / len;
            variance[m] = generation
                .iter()
                .map(|ind| (Cached::fit(ind, m) - mean[m]).powi(2))
                .sum::<f64>()
                / len;
            stdev[m] = variance[m].sqrt();
        }

        FitnessBasicMulti {
            mean,
            variance,
            stdev,
        }
    }
}
