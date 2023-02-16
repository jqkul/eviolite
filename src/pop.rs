use crate::traits::Individual;
use crate::utils::LazyWrapper;

pub type Population<T> = Vec<LazyWrapper<T>>;

pub trait Visitor<T: Individual> {
    fn visit(&mut self, population: &mut Vec<T>);
}

#[allow(dead_code)]

unsafe fn main() {}
