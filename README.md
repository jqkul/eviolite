# Eviolite
[![Published on crates.io!](https://img.shields.io/crates/v/eviolite?style=flat-square)](https://crates.io/crates/eviolite)
[![Read the docs on docs.rs!](https://img.shields.io/docsrs/eviolite?style=flat-square)](https://docs.rs/eviolite)

Eviolite is a set of tools and algorithms for using evolutionary algorithms in 
Rust. It is written in a performance-minded, minimal-copy style, and uses 
[rayon](https://crates.io/crates/rayon) to parallelize the most 
computationally intensive parts. It also includes a drop-in replacement for 
[rand](https://crates.io/crates/rand)'s `thread_rng` that is fully 
reproducible and can be seeded from an environment variable. This means that 
if you get a run you like, you can share that seed with someone else alongside 
your program and they will be guaranteed to get the same output you got.

### Getting Started
Add the following to your `Cargo.toml`: 

```toml
eviolite = "0.1"
```

Then continue reading for a simple example, or take a look at 
[the docs](https://docs.rs/eviolite) if you're not the tutorial type.

## Example
Let's go step by step over a complete program that uses a genetic algorithm to 
find a polynomial that approximates the sine function. This program is nice 
and short, but nicely demonstrates the general workflow of using Eviolite.

First, we'll import everything we need from Eviolite:

```rust
use eviolite::prelude::*;
```

Then import a few other things that'll be helpful:
```rust
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use std::f64::consts::FRAC_PI_2;
```

### Defining Our Solution Type
```rust
#[derive(Clone)]
struct Polynomial(Array1<f64>);
```
The elements of this array will represent the coefficients `a`, `b`, `c`, and 
`d` of the polynomial `a + bx + cx² + dx³`. Now let's define a method to 
evaluate our polynomial for a given `x`:

```rust
impl Polynomial {
    fn apply(&self, x: f64) -> f64 {
        self.0[0]
        + self.0[1] * x
        + self.0[2] * x.powi(2)
        + self.0[3] * x.powi(3)
    }
}
```

Next, we'll create an array of test points. We can't evaluate how good our 
approximation is at *every* point, since that would take forever, so we'll use 
100 evenly spaced points between 0 and &pi;/2. We'll wrap this in a 
`lazy_static` so it's globally accessible.
```rust
lazy_static::lazy_static! {
    static ref TEST_POINTS: Array1<f64> = Array1::range(0., FRAC_PI_2, FRAC_PI_2 / 100.);
}
```

### Implementing `Solution`
Now we're ready to implement the `Solution` trait:
```rust
impl Solution for Polynomial {
```

First, we define what type our fitness values will be. For now, we'll keep it 
simple, and say that how good an approximation is can be represented as a 
single `f64`:
```rust
type Fitness = f64;
```

Next, we need to define the four genetic algorithm primitives:
***generation***, ***evaluation***, ***crossover***, and ***mutation***. 

#### Generation
Eviolite needs to be able to create a population of randomly generated 
individuals as a starting point for the algorithm. We'll use the handy 
[`random_using`](https://docs.rs/ndarray-rand/*/ndarray_rand/trait.RandomExt.html#tymethod.random_using)
method from 
[ndarray-rand](https://crates.io/crates/ndarray-rand) 
to generate four random `f64`s between 0 and 1 with Eviolite's reproducible RNG.
These numbers will represent the coefficients `a`, `b`, `c`, and `d` in our 
polynomial `a + bx + cx² + dx³`.
```rust
fn generate() -> Polynomial {
        Polynomial(Array1::random_using(4, Uniform::new_inclusive(0.0, 1.0), &mut thread_rng()))
}
```

#### Evaluation
We need to be able to tell the algorithm how good a given polynomial is at 
approximating `sin`. We'll do this by evaluating our polynomial for every test 
point `x` and taking the absolute difference between that value and `sin(x)`, 
then averaging all of those differences to get the mean error across all test 
points.
```rust
fn evaluate(&self) -> f64 {
        -TEST_POINTS.mapv(
            |x| (self.apply(x) - x.sin()).abs()
        ).mean().unwrap()
}
```
Genetic algorithms assume that a higher fitness value is better, so we stick a 
negative sign in front.

> **Activity:** Think of other ways we could measure the fitness of these
> approximations!

#### Crossover
If you were to compare genetic algorithms to real-life selective breeding, this
would be the breeding part. We need to mix together the coefficients of two
polynomials so that they both get some information from the other while
retaining some of their own. We'll just use a simple
[one-point crossover](https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)#One-point_crossover),
which there's conveniently a built-in for:
```rust
fn crossover(a: &mut Self, b: &mut Self) {
        crossover::one_point(&mut a.0, &mut b.0);
}
```

#### Mutation
Just like in real-life evolution, this is where "new ideas" come from.
We need to randomly modify a polynomial just a little bit.
One way to do this is by adding some
[Gaussian noise](https://en.wikipedia.org/wiki/Gaussian_noise)
to the coefficients.
Fortunately, there's a built-in for that.
We'll apply a 50% chance to mutate each coefficient,
and add noise with a standard deviation of 0.1
to the ones that are chosen for mutation.
```rust
fn mutate(&mut self) {
        mutation::gaussian(&mut self.0, 0.5, 0.1);
}
```

And that's it! We've implemented `Solution`, and that means Eviolite has
everything it needs to evolve a polynomial to approximate `sin(x)`.
Now we can write our `main()` function and actually get a result.

### Running an Evolution
Let's create our
[`Evolution`](https://docs.rs/eviolite/*/eviolite/struct.Evolution.html)
instance. We'll use the
[*`(μ + λ)`*](https://docs.rs/eviolite/*/eviolite/alg/struct.MuPlusLambda.html)
algorithm, with some somewhat arbitrarily chosen parameters. We'll pass it a 
[`Tournament`](https://docs.rs/eviolite/*/eviolite/select/struct.Tournament.html)
selector, which will repeatedly choose 10 polynomials at random and pick the
best. We'll also create a
[`BestN`](https://docs.rs/eviolite/*/eviolite/hof/struct.BestN.html)
hall of fame with a size of 1, which will automatically track the best
polynomial that's been found so far. Finally, we'll completely delete the 
population and generate a fresh one every 25,000 generations as a failsafe in 
case the algorithm gets stuck. In `main()`:
```rust
let evo: Evolution<Polynomial, _, _, ()> = Evolution::with_resets(
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

    hof::BestN::new(1),

    25000
);
```

> **Activity:** Try playing around with these parameters to see how they affect
> the evolution!

Now we just have to run it! We'll have the evolution run until it finds a
polynomial whose mean error across all the test points is less than 0.001.
We'll also time it for good measure:
```rust
let start = std::time::Instant::now();

let log = evo.run_until(
    |gen| -gen.hall_of_fame[0].evaluate() < 0.001
);

let time = start.elapsed();
```

[`.run_until()`](https://docs.rs/eviolite/*/eviolite/struct.Evolution.html#method.run_until)
takes a closure that receives information about the current state of the run
and decides whether to stop it, so we just check if the best solution we've
seen has a mean error less than 0.001.

Now, let's extract our shiny new less-than-0.001-average-error polynomial from
the run log and print it to console:
```rust
let (best, _) = log.hall_of_fame[0].clone().into_inner();

println!("found in {:.3} secs: sin(x) ≈ {:.3} + {:.3}x + {:.3}x² + {:.3}x³",
    time.as_secs_f64(), best.0[0], best.0[1], best.0[2], best.0[3]
);
```

And that's everything! See
[`examples/approx_sin.rs`](https://github.com/jqkul/eviolite/blob/main/examples/approx_sin.rs)
for the full code.

When you compile and run this example (on release build - genetic algorithms
are very computationally intensive!), after a moment you should get an output
that looks a bit like this:
```
found in 3.982 secs: sin(x) ≈ 0.001 + 1.017x + -0.059x² + -0.118x³
```
> **Activity:** Set the environment variable `EVIOLITE_SEED` to `1175913497836025702` and run the program again to get the exact same output as above! (Except the run time, of course.)

The
[Taylor series expansion](https://en.wikipedia.org/wiki/Taylor_series#Trigonometric_functions)
for `sin(x)` starts `x - (x³ / 3!) + ...`, so it's a nice sanity check to see
that the coefficients roughly match 0, 1, 0, and -1/6. 