# BFGS

[![Rust](https://github.com/alessandrocasalino/r-bfgs/actions/workflows/rust.yml/badge.svg)](https://github.com/alessandrocasalino/r-bfgs/actions/workflows/rust.yml)

A BFGS implementation in Rust using BLAS routines.

## Example

```rust
// Import r-bfgs library
use bfgs;
use bfgs::settings::{LineSearchAlg, MinimizationAlg};

// Create the settings with default parameters
let mut settings: bfgs::settings::Settings = Default::default();
// And eventually change some of the settings
settings.minimization = MinimizationAlg::Lbfgs;
settings.line_search = LineSearchAlg::Backtracking;

// Function to be minimized
let function = |x: &[f64], g: &[f64], f: &mut f64, d: i32| {
    *f = 0.;
    for v in x {
      *f += v * v;
    }
};
///
// Set the starting point
let x = vec![0., -1.];
// Find the minimum
let result = bfgs::get_minimum(&function, &x, &settings);
// Check if the result is found
assert!(result.is_ok(), "Result not found: {}", result.err().unwrap());
// Access the results
println!("Minimum energy: {}", result.as_ref().unwrap().f);
println!("Position of the minimum: {:?}", result.as_ref().unwrap().x);
println!("Number of iterations: {}", result.as_ref().unwrap().iter);
```

See the tests for more examples, e.g., for the `get_minimum` with provided gradient.
