// Import r-bfgs library
use bfgs::settings::{MinimizationAlgorithm, LineSearchAlgorithm};

fn main() {

    // Create the settings with default parameters
    let mut settings: bfgs::settings::Settings = Default::default();
    // And eventually change some of the settings
    settings.minimization = MinimizationAlgorithm::Lbfgs;
    settings.line_search = LineSearchAlgorithm::Backtracking;

    // Function to be minimized
    let booth = |r: &[f64], _g: &[f64], f: &mut f64, _d: i32| {
        let x = r[0];
        let y = r[1];
        *f = (x + 2. * y - 7.) * (x + 2. * y - 7.) + (2. * x + y - 5.) * (2. * x + y - 5.);
    };


    // Set the starting point
    let x = vec![0., -1.];
    // Find the minimum
    let result = bfgs::get_minimum(&booth, &x, &settings);
    // Check if the result is found
    assert!(result.is_ok(), "Result not found: {}", result.err().unwrap());
    // Access the results
    println!("Minimum energy: {}", result.as_ref().unwrap().f);
    println!("Position of the minimum: {:?}", result.as_ref().unwrap().x);
    println!("Number of iterations: {}", result.as_ref().unwrap().iter);
    println!("Number of function evaluations: {}", result.as_ref().unwrap().eval);
}