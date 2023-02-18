pub mod alg;
pub mod fitness;
pub mod pop;
pub mod select;
pub mod repro_thread_rng;
mod traits;
pub use traits::{Fitness, Solution};

mod utils;

#[cfg(test)]
mod tests {
    use super::*;
}
