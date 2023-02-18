use crate::traits::Solution;
use crate::utils::LazyWrapper;

pub type Population<T> = Vec<LazyWrapper<T>>;

pub trait Visitor<T: Solution> {
    fn visit(&mut self, population: &mut Vec<T>);
}

#[allow(dead_code)]

unsafe fn main() {}
