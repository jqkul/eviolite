use crate::{Solution, Cache};

pub trait HallOfFame<T: Solution> {
    fn record(&mut self, generation: &[Cache<T>]);
}

pub struct BestN<T: Solution> {
    max: usize,
    best: Vec<Cache<T>>
}

impl<T: Solution> BestN<T> {
    pub fn new(n: usize) -> Self {
        BestN { max: n, best: Vec::with_capacity(n) } 
    }
}

impl<T: Solution> HallOfFame<T> for BestN<T> {
    fn record(&mut self, generation: &[Cache<T>]) {
        
    }
}

// impl<T: Solution> BestN<T> {
//     fn find_index(&self, ind: &Cache<T>) -> Option<usize> {
        
//     }
// }