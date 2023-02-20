use criterion::{black_box, criterion_group, criterion_main, Criterion};

use num::traits::ToPrimitive;
use num::Rational32;

use rand::Rng;

use eviolite::{
    alg, fitness::MultiObjective, repro_thread_rng::thread_rng, select::Tournament, Solution,
};

const TARGET: f64 = std::f64::consts::PI;

#[derive(Clone, Debug)]
struct Fraction(Rational32);

impl Solution for Fraction {
    type Fitness = MultiObjective<1>;

    fn generate() -> Self {
        let mut rng = thread_rng();
        Fraction(Rational32::new(
            rng.gen_range(1..=100000),
            rng.gen_range(1..=100000),
        ))
    }

    fn evaluate(&self) -> Self::Fitness {
        let numer = *self.0.numer();
        let denom = *self.0.denom();
        MultiObjective::unweighted([
            if numer < 0 || denom < 0 || numer > 100000 || denom > 100000 {
                -1000000f64
            } else {
                -(self.0.to_f64().unwrap() - TARGET).abs()
            },
        ])
    }

    fn crossover(a: &mut Self, b: &mut Self) {
        let (&min_num, &max_num) = minmax(a.0.numer(), b.0.numer());
        let (&min_den, &max_den) = minmax(a.0.denom(), b.0.denom());

        let mut rng = thread_rng();
        a.0 = Rational32::new(
            rng.gen_range(min_num..=max_num),
            rng.gen_range(min_den..=max_den),
        );
        b.0 = Rational32::new(
            rng.gen_range(min_num..=max_num),
            rng.gen_range(min_den..=max_den),
        );
    }

    fn mutate(&mut self) {
        let mut rng = thread_rng();
        if rng.gen_bool(0.5) {
            self.0 = Rational32::new(*self.0.numer() + rng.gen_range(-2..=2), *self.0.denom());
        } else {
            self.0 = Rational32::new(
                *self.0.numer(),
                (*self.0.denom() + rng.gen_range(-2..=2)).clamp(1, i32::MAX),
            );
        }
    }
}

fn minmax<T>(a: T, b: T) -> (T, T)
where
    T: Ord,
{
    if b < a {
        (b, a)
    } else {
        (a, b)
    }
}

pub fn bench_pi_frac(c: &mut Criterion) {
    let selector = Tournament::new(3);

    c.bench_function("pi_frac simple", |b| {
        b.iter_with_large_drop(|| {
            let finalpop: Vec<Fraction> = alg::simple(1000, selector, 0.5, 0.5, 10);
            finalpop
        })
    });

    c.bench_function("pi_frac nsga2", |b| {
        b.iter_with_large_drop(|| {
            let finalpop: Vec<Fraction> = alg::nsga2(1000, 0.5, 0.5, 10);
            finalpop
        })
    });
}

criterion_group! {
    name = grp_pi_frac;
    config = {
        Criterion::default().sample_size(100)
    };
    targets = bench_pi_frac
}
criterion_main!(grp_pi_frac);
