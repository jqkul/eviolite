use eviolite::prelude::*;

const TARGET: f64 = std::f64::consts::PI;

const POPSIZE: usize = 1000;
const RESET_INTERVAL: usize = 500;
const NGENS: usize = 5000;

fn main() {
    let evo: Evolution<Fraction, _, _, stats::FitnessBasic> = Evolution::with_resets(
        alg::MuPlusLambda::new(POPSIZE, POPSIZE, 0.5, 0.1, select::Tournament::new(10)),
        hof::BestN::new(1),
        RESET_INTERVAL
    );

    let log = evo.run_for(NGENS);

    let (best, _) = log.hall_of_fame[0].clone().into_inner();

    println!("{}/{} = {}", best.0, best.1, best.divide());
}

#[derive(Clone)]
struct Fraction(u64, u64);

impl Solution for Fraction {
    type Fitness = f64;
    fn generate() -> Self {
        let mut rng = thread_rng();
        Fraction(rng.gen_range(1..100000000), rng.gen_range(1..100000000))
    }

    fn evaluate(&self) -> Self::Fitness {
        -(self.divide() - TARGET).abs()
    }

    fn crossover(a: &mut Self, b: &mut Self) {
        let (num_min, num_max) = if a.0 < b.0 { (a.0, b.0) } else { (b.0, a.0) };
        let (den_min, den_max) = if a.1 < b.1 { (a.1, b.1) } else { (b.1, a.1) };

        let mut rng = thread_rng();

        *a = Fraction(
            rng.gen_range(num_min..=num_max),
            rng.gen_range(den_min..=den_max),
        );
        a.normalize();

        *b = Fraction(
            rng.gen_range(num_min..=num_max),
            rng.gen_range(den_min..=den_max),
        );
        b.normalize();
    }

    fn mutate(&mut self) {
        let mut rng = thread_rng();
        let multiplier = rng.gen_range(0..=3);
        self.0 = self.0 * multiplier + multiplier;
        self.1 = self.1 * multiplier + multiplier;
        self.normalize();
    }
}

impl Fraction {
    fn normalize(&mut self) {
        if self.0 == 0 {
            self.0 = 1;
        }
        if self.1 == 0 {
            self.1 = 1;
        }
        let gcd = num::integer::gcd(self.0, self.1);
        self.0 /= gcd;
        self.1 /= gcd;
    }

    fn divide(&self) -> f64 {
        self.0 as f64 / self.1 as f64
    }
}
