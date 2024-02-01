extern crate cblas;
extern crate blas_src;

// Public modules
pub mod settings;

// Private modules
mod bfgs;
mod lbfgs;
mod line_search;
mod exit_condition;
mod log;
mod gradient_descent;

use crate::settings::Settings;
use crate::settings::MinimizationAlg;

/// Calculates the minimum of a function using the BFGS algorithm.
///
/// The BFGS algorithm is an iterative optimization algorithm used to
/// find the minimum of a function. This implementation uses the BLAS
/// library for linear algebra computations.
///
/// # Arguments
///
/// * `fn_function` - A function representing the function to minimize.
///    It takes in the current position `x`, the gradient vector `g`,
///    the function value `f`, and the dimension size `d` as arguments.
///    The closure is expected to update `f` with the current values
///    at `x`.
///    This can be a function (passed by &) or a closure.
///
/// * `x` - A mutable reference to the initial position vector.
///
/// * `settings` - The settings for the BFGS algorithm.
///
/// The gradient is estimated with a centered finite difference
/// method.
///
/// # Returns
///
/// The minimum energy value if the algorithm converges within the
/// maximum number of iterations specified in the `settings`. Returns
/// `Err` if the algorithm does not converge.
///
/// # Examples
///
/// ```no_run
/// // Import r-bfgs library
/// use bfgs;
/// use bfgs::settings::{LineSearchAlg, MinimizationAlg};
///
/// // Create the settings with default parameters
/// let mut settings: bfgs::settings::Settings = Default::default();
/// // And eventually change some of the settings
/// settings.minimization = MinimizationAlg::Lbfgs;
/// settings.line_search = LineSearchAlg::Backtracking;
///
/// // Function to be minimized
/// let function = |x: &[f64], g: &[f64], f: &mut f64, d: i32| {
///     *f = 0.;
///     for v in x {
///       *f += v * v;
///     }
/// };
///
/// // Set the starting point
/// let mut x = vec![0., -1.];
/// // Find the minimum
/// let result = bfgs::get_minimum(&function, &mut x, &settings);
/// // Check if the result is found
/// assert!(result.is_ok(), "Result not found: {}", result.err().unwrap());
/// ```
#[allow(non_snake_case)]
pub fn get_minimum<Function>(fn_function: &Function, x: &mut Vec<f64>, settings: &Settings)
    -> Result<f64, &'static str>
    where
        Function: Fn(&[f64], &[f64], &mut f64, i32)
{
    // Default gradient
    let gf = |x: &[f64], g: &mut [f64], _f: &f64, d: i32| {
        // Finite difference derivative
        let h = 1e-5;
        let mut x_for: Vec<f64> = x.to_vec();
        let mut x_bck: Vec<f64> = x.to_vec();
        for i in 0..d {
            let mut f1 = 0.;
            let mut f2 = 0.;
            x_for[i as usize] += h;
            x_bck[i as usize] -= h;
            fn_function(&x_bck, g, &mut f1, d);
            fn_function(&x_for, g, &mut f2, d);
            g[i as usize] = (f2 - f1) / (2. * h);
            x_for[i as usize] -= h;
            x_bck[i as usize] += h;
        }
    };

    do_bfgs(fn_function, &gf, x, settings)
}

/// Calculates the minimum of a function using the BFGS algorithm,
/// specifying a function for the gradient.
///
/// The BFGS algorithm is an iterative optimization algorithm used to
/// find the minimum of a function. This implementation uses the BLAS
/// library for linear algebra computations.
///
/// # Arguments
///
/// * `fn_function` - A function representing the function to minimize.
///    It takes in the current position `x`, the gradient vector `g`,
///    the function value `f`, and the dimension size `d` as arguments.
///    The closure is expected to update `f` with the current values
///    at `x`.
///    This can be a function (passed by &) or a closure.
///
/// * `fn_gradient` - A function representing the gradient function.
///    It takes in the current position `x`, the gradient vector `g`, the
///    function value `f`, and the dimension size `d` as arguments. The
///    closure is expected to update `g` with the current value at `x`.
///    This can be a function (passed by &) or a closure.
///
/// * `x` - A mutable reference to the initial position vector.
///
/// * `settings` - The settings for the BFGS algorithm.
///
/// # Returns
///
/// The minimum energy value if the algorithm converges within the
/// maximum number of iterations specified in the `settings`. Returns
/// `Err` if the algorithm does not converge.
///
/// # Examples
///
/// ```no_run
/// // Import r-bfgs library
/// use bfgs;
/// use bfgs::settings::{LineSearchAlg, MinimizationAlg};
///
/// // Create the settings with default parameters
/// let mut settings: bfgs::settings::Settings = Default::default();
/// // And eventually change some of the settings
/// settings.minimization = MinimizationAlg::Lbfgs;
/// settings.line_search = LineSearchAlg::Backtracking;
///
/// // Function to be minimized
/// let function = |x: &[f64], g: &[f64], f: &mut f64, _d: i32| {
///     *f = 0.;
///     for v in x {
///       *f += v * v;
///     }
/// };
///
/// // Gradient
/// let gradient = |x: &[f64], g: &mut [f64], f: &f64, d: i32| {
///     for i in 0..d as usize {
///       g[i] = 2. * x[i];
///     }
/// };
///
/// // Set the starting point
/// let mut x = vec![0., -1.];
/// // Find the minimum
/// let result = bfgs::get_minimum_with_gradient(&function, &gradient, &mut x, &settings);
/// // Check if the result is found
/// assert!(result.is_ok(), "Result not found: {}", result.err().unwrap());
/// ```
#[allow(non_snake_case)]
pub fn get_minimum_with_gradient<Function, Gradient>(fn_function: &Function, fn_gradient: &Gradient, x: &mut Vec<f64>, settings: &Settings)
                                                     -> Result<f64, &'static str>
    where
        Function: Fn(&[f64], &[f64], &mut f64, i32),
        Gradient: Fn(&[f64], &mut [f64], &f64, i32)
{
    do_bfgs(fn_function, fn_gradient, x, settings)
}

fn do_bfgs<Function, Gradient>(fn_function: &Function, fn_gradient: &Gradient, x: &mut Vec<f64>, settings: &Settings)
    -> Result<f64, &'static str>
    where
        Function: Fn(&[f64], &[f64], &mut f64, i32),
        Gradient: Fn(&[f64], &mut [f64], &f64, i32)
{
    // Check value of settings
    if settings.mu > settings.eta {
        return Err("mu can not be bigger than eta")
    }

    // Handle different minimization methods
    match settings.minimization {
        MinimizationAlg::GradientDescent => {
            use crate::gradient_descent::gradient_descent;
            gradient_descent(fn_function, fn_gradient, x, settings)
        }
        MinimizationAlg::Bfgs => {
            use crate::bfgs::bfgs;
            bfgs(fn_function, fn_gradient, x, settings)
        }
        MinimizationAlg::Lbfgs => {
            use crate::lbfgs::lbfgs;
            lbfgs(fn_function, fn_gradient, x, settings)
        }
        MinimizationAlg::BfgsBackup => {
            use crate::bfgs::bfgs;
            let r = bfgs(fn_function, fn_gradient, x, settings);
            match r {
                Ok(f) => Ok(f),
                Err(_e) => {
                    use crate::lbfgs::lbfgs;
                    lbfgs(fn_function, fn_gradient, x, settings)
                }
            }
        }
    }
}
