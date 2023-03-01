use rand::Rng;

use crate::{
    fitness::MultiObjective,
    repro_rng::{random, thread_rng},
    Solution,
};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct One(pub f64);

impl Solution for One {
    type Fitness = MultiObjective<1>;

    fn generate() -> Self {
        One(random())
    }

    fn evaluate(&self) -> Self::Fitness {
        MultiObjective::new_unweighted([self.0])
    }

    fn crossover(_: &mut Self, _: &mut Self) {
        unreachable!()
    }
    fn mutate(&mut self) {
        unreachable!()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Foo(pub [f64; 2]);

impl Solution for Foo {
    type Fitness = MultiObjective<2>;

    fn generate() -> Self {
        let mut rng = thread_rng();
        Foo([rng.gen(), rng.gen()])
    }

    fn evaluate(&self) -> Self::Fitness {
        MultiObjective::new_unweighted(self.0)
    }

    fn crossover(_: &mut Self, _: &mut Self) {
        unreachable!()
    }
    fn mutate(&mut self) {
        unreachable!()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bar(pub [f64; 3]);

impl Solution for Bar {
    type Fitness = MultiObjective<3>;

    fn generate() -> Self {
        let mut rng = thread_rng();
        Bar([-rng.gen::<f64>(), -rng.gen::<f64>(), -rng.gen::<f64>()])
    }

    fn evaluate(&self) -> Self::Fitness {
        MultiObjective::new_unweighted(self.0)
    }

    fn crossover(_: &mut Self, _: &mut Self) {
        unreachable!()
    }
    fn mutate(&mut self) {
        unreachable!()
    }
}
