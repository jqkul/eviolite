/// This file is a re-implementation of the `thread_rng` implementation from `rand`,
/// but specialized to avoid re-seeding and enable a single seed applied through the whole program.

use std::rc::Rc;
use std::cell::UnsafeCell;

use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::{SeedableRng, RngCore};
use rand::rngs::OsRng;
use rand::Rng;
use rand_xoshiro::Xoshiro256StarStar;

pub struct ReproThreadRng {
    rng: Rc<UnsafeCell<Xoshiro256StarStar>>
}

thread_local! {
    static THREAD_RNG_KEY: Rc<UnsafeCell<Xoshiro256StarStar>> = {
        let seed: u64 = match std::env::var("MENDEL_SEED").map(|s| s.parse::<u64>()) {
            Ok(Ok(seed)) => seed,
            _ => {
                let seed = OsRng.next_u64();
                eprintln!("mendel: unable to read preset RNG seed from environment variable MENDEL_SEED\nmendel: using OS-generated seed {}", seed);
                seed
            }
        };

        let rng = Xoshiro256StarStar::seed_from_u64(seed);

        Rc::new(UnsafeCell::new(rng))
    }
}

pub fn random<T>() -> T where Standard: Distribution<T> {
    thread_rng().gen()
}

pub fn thread_rng() -> ReproThreadRng {
    let rng = THREAD_RNG_KEY.with(|t| t.clone());
    ReproThreadRng { rng }
}

impl Default for ReproThreadRng {
    fn default() -> Self {
        thread_rng()
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