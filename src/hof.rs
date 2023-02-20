use crate::{Solution, Cached, Fitness};
use itertools::Itertools;

pub trait HallOfFame<T: Solution> {
    fn record(&mut self, generation: &[Cached<T>]);
}

impl<T: Solution> HallOfFame<T> for () {
    fn record(&mut self, _: &[Cached<T>]) { }
}

pub struct BestN<T: Solution> {
    max: usize,
    best: Vec<Cached<T>>
}

impl<T: Solution> BestN<T> {
    pub fn new(max: usize) -> Self {
        BestN { max, best: Vec::with_capacity(max) } 
    }
}

impl<T: Solution> HallOfFame<T> for BestN<T> {
    fn record(&mut self, generation: &[Cached<T>]) {
        for ind in generation {
            if let Some(idx) = self.find_index(ind) {
                self.best.insert(idx, ind.clone());
            } else if self.best.len() < self.max {
                self.best.push(ind.clone());
            }
        }
        self.best.truncate(self.max);
    }
}

impl<T: Solution> BestN<T> {
    fn find_index(&self, ind: &Cached<T>) -> Option<usize> {
        let fit = ind.evaluate().collapse();
        if self.best.is_empty() || fit > self.best[0].evaluate().collapse() {
            return Some(0);
        }

        for (i, (a, b)) in self.best.iter().tuple_windows().enumerate() {
            if fit > b.evaluate().collapse() && fit < a.evaluate().collapse() {
                return Some(i+1);
            }
        }

        None
    }
}