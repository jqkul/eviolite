//! Reproducible and globally seedable version of [`rand`]'s `thread_rng`
//! 
//! This module contains a drop-in replacement for [`random`][rand::random] and [`thread_rng`][rand::thread_rng] from the [`rand`] crate.
//! This version uses a faster, non-cryptographically-secure PRNG ([`Xoshiro256StarStar`][rand_xoshiro::Xoshiro256StarStar] from the [`rand_xoshiro`] crate),
//! and never re-seeds it from an external source, making results using it fully 
//! reproducible by supplying the same seed as a previous run.
//! To use it, just use this module's [`random`] and [`thread_rng`] instead of [`rand`]'s version
//! every time you need to generate a random number.
//! 
//! Notes on reproducibility
//! ------------------------
//! When the RNG is initialized, the program will read the environment variable `EVIOLITE_SEED`
//! and attempt to parse its contents as a `u64`. If it succeeds, it will seed the RNG with the result.
//! If it fails, either in reading `EVIOLITE_SEED` or in parsing it as a `u64`, it will seed itself with
//! a random number provided by the OS, and print the seed it used to standard error.
//! 
//! If you want to reproduce a run, **make sure to copy the seed from standard error and keep it.**
//! In addition, **make sure never to use randomness in your [`Solution`]'s [`evaluate()`] method.**
//! Any sane fitness evaluation shouldn't be random, so this shouldn't be much of a limitation.
//! 
//! [`random`]: ./fn.random.html
//! [`thread_rng`]: ./fn.thread_rng.html
//! [`Solution`]: ../trait.Solution.html
//! [`evaluate()`]: ../trait.Solution.html#tymethod.evaluate

use std::cell::UnsafeCell;
use std::rc::Rc;

use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::rngs::OsRng;
use rand::Rng;
use rand::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;

const SEED_ENV_VAR_NAME: &str = "EVIOLITE_SEED";

/// A reference to the thread-local reproducible RNG
/// 
/// This type works exactly the same as [`rand`]'s [`ThreadRng`][rand::rngs::ThreadRng],
/// except that it can be seeded from an environment variable and uses a faster RNG.
/// See the [module-level documentation][./index.html] for further information.
pub struct ReproThreadRng {
    rng: Rc<UnsafeCell<Xoshiro256StarStar>>,
}

thread_local! {
    static THREAD_RNG_KEY: Rc<UnsafeCell<Xoshiro256StarStar>> = {
        let seed: u64 = match std::env::var(SEED_ENV_VAR_NAME).map(|s| s.parse::<u64>()) {
            Ok(Ok(seed)) => seed,
            _ => {
                let seed = OsRng.next_u64();
                eprintln!("eviolite: unable to read preset RNG seed from environment variable {}\neviolite: using OS-generated seed {}", SEED_ENV_VAR_NAME, seed);
                seed
            }
        };

        let rng = Xoshiro256StarStar::seed_from_u64(seed);

        Rc::new(UnsafeCell::new(rng))
    }
}

/// Generate a random value using the reproducible thread-local RNG.
/// 
/// This function works exactly the same as [`rand`]'s [`random()`][rand::random];
/// see that documentation for further information.
pub fn random<T>() -> T
where
    Standard: Distribution<T>,
{
    thread_rng().gen()
}

/// Retrieve the lazily-initialized reproducible thread-local RNG.
/// 
/// This function works exactly the same as [`rand`]'s [`thread_rng()`][rand::thread_rng],
/// except that it can be seeded from an environment variable and uses a faster RNG.
/// See the [module-level documentation][./index.html] for further information.
pub fn thread_rng() -> ReproThreadRng {
    Default::default()
}

impl Default for ReproThreadRng {
    fn default() -> Self {
        let rng = THREAD_RNG_KEY.with(|t| t.clone());
        ReproThreadRng { rng }
    }
}

impl RngCore for ReproThreadRng {
    #[inline(always)]
    fn next_u32(&mut self) -> u32 {
        let rng = unsafe { &mut *self.rng.get() };
        rng.next_u32()
    }

    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        let rng = unsafe { &mut *self.rng.get() };
        rng.next_u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let rng = unsafe { &mut *self.rng.get() };
        rng.fill_bytes(dest);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        let rng = unsafe { &mut *self.rng.get() };
        rng.try_fill_bytes(dest)
    }
}
