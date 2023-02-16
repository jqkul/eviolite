pub mod alg;
pub mod fitness;
pub mod pop;
pub mod select;
mod traits;
pub use traits::{Fitness, Individual};

mod utils;

#[cfg(test)]
mod tests {
    use super::*;
}
