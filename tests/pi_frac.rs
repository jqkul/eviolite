use std::cmp::min;
use std::fmt::Display;

use num::{rational::Ratio, Rational32};
use num::traits::ToPrimitive;

use rand::{Rng};

use eviolite::{select::{Tournament, find_best}, Individual, alg, repro_thread_rng::thread_rng};

const TARGET: f64 = std::f64::consts::PI;

#[derive(Clone, Debug)]
struct Fraction(Rational32);

impl Individual for Fraction {
    type Fitness = f64;

    fn generate() -> Self {
        let mut rng = thread_rng();
        Fraction(Rational32::new(rng.gen_range(1..=100000), rng.gen_range(1..=100000)))
    }

    fn evaluate(&self) -> Self::Fitness {
        let numer = *self.0.numer();
        let denom = *self.0.denom();
        if numer < 0 || denom < 0 || numer > 100000 || denom > 100000 {
            -1000000f64
        } else {
            -(self.0.to_f64().unwrap() - TARGET).abs()
        }
    }

    fn crossover(a: &mut Self, b: &mut Self) {
        let (&min_num, &max_num) = minmax(a.0.numer(), b.0.numer());
        let (&min_den, &max_den) = minmax(a.0.denom(), b.0.denom());

        let mut rng = thread_rng();
        a.0 = Rational32::new(rng.gen_range(min_num..=max_num), rng.gen_range(min_den..=max_den));
        b.0 = Rational32::new(rng.gen_range(min_num..=max_num), rng.gen_range(min_den..=max_den));
    }

    fn mutate(&mut self) {
        let mut rng = thread_rng();
        if rng.gen_bool(0.5) {
            self.0 = Rational32::new(*self.0.numer() + rng.gen_range(-2..=2), *self.0.denom());
        } else {
            self.0 = Rational32::new(*self.0.numer(), (*self.0.denom() + rng.gen_range(-2..=2)).clamp(1, i32::MAX));
        }
    }
}

impl Display for Fraction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.0.numer(), self.0.denom())
    }
}

#[test]
fn main() {
    let selector = Tournament::new(3);

    let finalpop: Vec<Fraction> = alg::simple(1000, selector, 0.5, 0.5, 1000);

    let best = find_best(&finalpop);

    println!("{} = {}", best, best.0.to_f64().unwrap());
}

fn minmax<T>(a: T, b: T) -> (T, T) where T: Ord {
    if b < a {
        (b, a)
    } else {
        (a, b)
    }
}