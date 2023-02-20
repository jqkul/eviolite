use std::ops::{Deref, DerefMut};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use eviolite::{alg::var_and, repro_thread_rng::thread_rng, Solution};
use rand::Rng;

#[derive(Clone, Copy)]
struct Matrix3x3([[f64; 3]; 3]);

impl Deref for Matrix3x3 {
    type Target = [[f64; 3]; 3];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Matrix3x3 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Solution for Matrix3x3 {
    type Fitness = f64;

    fn generate() -> Self {
        let mut rng = thread_rng();
        Matrix3x3([
            [rng.gen(), rng.gen(), rng.gen()],
            [rng.gen(), rng.gen(), rng.gen()],
            [rng.gen(), rng.gen(), rng.gen()],
        ])
    }

    fn evaluate(&self) -> Self::Fitness {
        unreachable!()
    }

    fn crossover(a: &mut Self, b: &mut Self) {
        let mut rng = thread_rng();
        // sampling a usize directly is bad for portability, so we sample a u32 and cast
        let (i, j) = (
            rng.gen_range::<u32, _>(0..=2) as usize,
            rng.gen_range::<u32, _>(0..=2) as usize,
        );
        (a[i][i], b[i][j]) = (b[i][j], a[i][j]);
    }

    fn mutate(&mut self) {
        let mut rng = thread_rng();
        let (i, j) = (
            rng.gen_range::<u32, _>(0..=2) as usize,
            rng.gen_range::<u32, _>(0..=2) as usize,
        );
        self[i][j] = (self[i][j] + rng.gen::<f64>()) / 2.0;
    }
}

pub fn bench_var_and(c: &mut Criterion) {
    let mut pop = Vec::with_capacity(500);
    for _ in 0..500 {
        pop.push(Matrix3x3::generate());
    }
    c.bench_function("var_and 3x3 500 0.5 0.5", |b| {
        b.iter(|| {
            var_and(&mut pop, 0.5, 0.5);
            black_box(&pop);
        })
    });
}

criterion_group!(grp_var_and, bench_var_and);
criterion_main!(grp_var_and);
