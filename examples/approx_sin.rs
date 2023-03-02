use eviolite::prelude::*;

use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use std::f64::consts::FRAC_PI_2;

lazy_static::lazy_static! {
    // create a static array of 100 points between 0 and π/2
    // we'll use these points to check how good our polynomials are at approximating sine
    static ref TEST_POINTS: Array1<f64> = Array1::range(0., FRAC_PI_2, FRAC_PI_2 / 100.);
}

#[derive(Clone)]
struct Polynomial(Array1<f64>);

impl Polynomial {
    fn apply(&self, x: f64) -> f64 {
        self.0[0]
        + self.0[1] * x
        + self.0[2] * x.powi(2)
        + self.0[3] * x.powi(3)
    }
}

impl Solution for Polynomial {
    type Fitness = f64;

    fn generate() -> Self {
        // create a random array of four f64s between 0 and 1
        // these represent the coefficients a, b, c, d in a + bx + cx² + dx³
        Polynomial(Array1::random_using(4, Uniform::new_inclusive(0.0, 1.0), &mut thread_rng()))
    }

    fn evaluate(&self) -> Self::Fitness {
        // for each test point, calculate the absolute difference between our polynomial and sine
        // then take the mean of those errors, and negate it so higher values are better
        -TEST_POINTS.mapv(
            |x| (self.apply(x) - x.sin()).abs()
        ).mean().unwrap()
    }

    fn crossover(a: &mut Self, b: &mut Self) {
        crossover::one_point(&mut a.0, &mut b.0);
    }

    fn mutate(&mut self) {
        mutation::gaussian(&mut self.0, 0.5, 0.1);
    }
}

fn main() {
    let evo: Evolution<Polynomial, _, _, ()> = Evolution::with_resets(
        // using the (μ + λ) algorithm
        alg::MuPlusLambda::new(
            // population size (μ)
            1000,
            // offspring size (λ)
            1000,
            // crossover chance (cxpb)
            0.5,
            // mutation chance (mutpb)
            0.2,
            // selection operator
            select::Tournament::new(10)
        ),

        // a hall of fame that will track the single best polynomial we've found
        hof::BestN::new(1),

        // completely reset the algorithm every 25000 generations
        25000
    );

    let start = std::time::Instant::now();
    // run the algorithm until we have a polynomial that's accurate to 3 decimal places on average
    let log = evo.run_until(
        |gen| -gen.hall_of_fame[0].evaluate() < 0.001
    );
    let time = start.elapsed();

    let (best, _) = log.hall_of_fame[0].clone().into_inner();

    println!("found in {:.3} secs: sin(x) ≈ {:.3} + {:.3}x + {:.3}x² + {:.3}x³",
        time.as_secs_f64(), best.0[0], best.0[1], best.0[2], best.0[3]
    );
}