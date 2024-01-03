extern crate cblas;
extern crate blas_src;

// Public modules
pub mod settings;

// Private modules
mod bfgs;
mod line_search;
mod log;
mod exit_condition;

use settings::Settings;
use crate::bfgs::bfgs;
use crate::settings::MinimizationAlg;

#[allow(dead_code)]
struct Point {
    a: f64,
    f: f64,
    d: f64,
}

#[allow(non_snake_case)]
fn Hessian(H: &mut Vec<f64>, s: &Vec<f64>, y: &Vec<f64>, I: &Vec<f64>, B: &mut Vec<f64>, C: &mut Vec<f64>,
           d: i32, layout: cblas::Layout, part: cblas::Part) {
    let rho: f64 = 1. / unsafe { cblas::ddot(d, y, 1, s, 1) };

    // Set B to the identity
    unsafe { cblas::dcopy(d * d, I, 1, &mut *B, 1); }

    unsafe { cblas::dger(layout, d, d, -rho, y, 1, s, 1, &mut *B, d); }

    // The first matrix multiplication have one symmetric matrix
    unsafe { cblas::dsymm(layout, cblas::Side::Left, part, d, d, 1., H, d, B, d, 0., &mut *C, d); }

    // Flush the value of the Hessian to 0
    unsafe { cblas::dscal(d * d, 0., H, 1); }
    unsafe { cblas::dger(layout, d, d, rho, s, 1, s, 1, H, d); }
    // Since no matrix is symmetric, gemm is used
    unsafe {
        cblas::dgemm(layout, cblas::Transpose::Ordinary, cblas::Transpose::None, d, d, d, 1., B, d, C, d,
                     1., H, d);
    }
}

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
/// * `f` - A mutable reference to the initial energy value.
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
pub fn get_minimum<Ef, Gf>(ef: &Ef, gf: &Gf, x: &mut Vec<f64>, d: i32, settings: Settings)
                           -> Option<f64>
    where
        Ef: Fn(&Vec<f64>, &Vec<f64>, &mut f64, i32),
        Gf: Fn(&Vec<f64>, &mut Vec<f64>, &f64, i32)
{
    match settings.minimization {
        MinimizationAlg::Bfgs => {
            bfgs(ef, gf, x, d, settings)
        }
        _ => None
    }
}
