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
/// * `ef` - A closure representing the energy function.
///    It takes in the current position `x`, the gradient vector `g`,
///    the energy value `f`, and the dimension size `d` as arguments.
///    The closure is expected to update `f` with the current values
///    at `x`.
///
/// * `gf` - A closure representing the gradient function. It takes in
///    the current position `x`, the gradient vector `g`, the energy
///    value `f`, and the dimension size `d` as arguments. The closure
///    is expected to update `g` with the current value at `x`.
///
/// * `x` - A mutable reference to the initial position vector.
///
/// * `d` - The dimension size.
///
/// * `settings` - The settings for the BFGS algorithm.
///
/// # Returns
///
/// The minimum energy value if the algorithm converges within the
/// maximum number of iterations specified in the `settings`. Returns
/// `None` if the algorithm does not converge.
#[allow(non_snake_case)]
pub fn get_minimum<Ef, Gf>(ef: &Ef, gf: &Gf, x: &mut Vec<f64>, d: i32, settings: &Settings)
                           -> Option<f64>
    where
        Ef: Fn(&Vec<f64>, &Vec<f64>, &mut f64, i32),
        Gf: Fn(&Vec<f64>, &mut Vec<f64>, &f64, i32)
{
    // Check value of settings
    if settings.mu > settings.eta {
        println!("ERROR: mu can not be bigger than eta");
        return None;
    }

    match settings.minimization {
        MinimizationAlg::Bfgs => {
            use crate::bfgs::bfgs;
            bfgs(ef, gf, x, d, settings)
        }
        MinimizationAlg::Lbfgs => {
            use crate::lbfgs::lbfgs;
            lbfgs(ef, gf, x, d, settings)
        }
        MinimizationAlg::BfgsBackup => {
            use crate::bfgs::bfgs;
            let r = bfgs(ef, gf, x, d, settings);
            match r {
                Some(f) => Some(f),
                None => {
                    use crate::lbfgs::lbfgs;
                    match lbfgs(ef, gf, x, d, settings) {
                        Some(f) => Some(f),
                        None => None
                    }
                }
            }
        }
    }
}
