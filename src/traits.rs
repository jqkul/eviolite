pub trait Solution: Clone + Sync {
    type Fitness: Fitness;
    fn generate() -> Self;
    fn evaluate(&self) -> Self::Fitness;
    fn crossover(a: &mut Self, b: &mut Self);
    fn mutate(&mut self);

    fn collapsed(&self) -> f64 {
        self.evaluate().collapse()
    }
}

pub trait Fitness: Copy {
    fn collapse(&self) -> f64;
}
