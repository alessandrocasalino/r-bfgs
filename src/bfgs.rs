use crate::{exit_condition, line_search, MinimizationResult};
use crate::settings::Settings;

#[allow(non_snake_case, clippy::too_many_arguments)]
fn Hessian(H: &mut [f64], s: &[f64], y: &[f64], I: &[f64], B: &mut Vec<f64>, C: &mut Vec<f64>,
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

#[allow(non_snake_case)]
pub fn bfgs<Function, Gradient>(fn_function: &Function, fn_gradient: &Gradient, x0: &[f64], settings: &Settings)
    -> Result<MinimizationResult, &'static str>
    where
        Function: Fn(&[f64], &[f64], &mut f64, i32),
        Gradient: Fn(&[f64], &mut [f64], &f64, i32)
{
    // Settings
    let iter_max = settings.iter_max;
    // BLAS definitions
    let layout = settings.layout;
    let part = settings.part;
    // Verbose (log)
    let verbose = settings.verbose;

    // Position vector
    let mut x = x0.to_owned();

    // Get the dimension
    let d = x.len() as i32;

    // Function update evaluations
    let mut eval: usize = 0;

    // Energy definition
    let mut f: f64 = 0.;

    // Gradient definition
    let mut g: Vec<f64> = vec![0.; d as usize];

    // Update energy and gradient
    fn_function(&x, &g, &mut f, d);
    fn_gradient(&x, &mut g, &f, d);
    eval += 1;

    // Hessian estimation
    let mut I: Vec<f64> = vec![0.; (d * d) as usize];
    let mut H: Vec<f64> = vec![0.; (d * d) as usize];
    for i in 0..d {
        I[(i * (d + 1)) as usize] = 1.;
    }
    unsafe { cblas::dcopy(d * d, &I, 1, &mut H, 1); }


    // Search direction
    let mut p: Vec<f64> = vec![0.; d as usize];
    // Difference x_k+1 - x_k
    let mut s: Vec<f64> = vec![0.; d as usize];
    // Difference of gradients
    let mut y: Vec<f64> = vec![0.; d as usize];

    // Temporary vectors for line_search routine
    let mut x_new: Vec<f64> = vec![0.; d as usize];

    // Line search element coefficient
    let mut a: f64 = 1.;
    // Temporary value of f
    let mut f_old: f64;

    // Temporary matrices for Hessian computation
    let mut B: Vec<f64> = vec![0.; (d * d) as usize];
    let mut C: Vec<f64> = vec![0.; (d * d) as usize];

    //Iteration number
    let mut k: usize = 0;

    // Main loop
    loop {
        // Stop if reaching the maximum number of iterations requested
        if k >= iter_max {
            return Err("Maximum number of iterations reached")
        }
        k += 1;

        // Store current values
        unsafe { cblas::dcopy(d, &x, 1, &mut s, 1); }
        unsafe { cblas::dcopy(d, &g, 1, &mut y, 1); }
        f_old = f;

        // Store current values
        unsafe { cblas::dsymv(layout, part, d, -1., &H, d, &g, 1, 0., &mut p, 1); }

        // Save the value of Phi_0 to be used for both line_search
        let phi_0: line_search::Point = line_search::Point { a: 0., f, d: unsafe { cblas::ddot(d, &g, 1, &p, 1) } };

        // Perform line search (updating a)
        if !line_search::line_search(&fn_function, &fn_gradient, &phi_0, &p, &mut x, &mut x_new, &mut g, &mut f, &mut a, d, k, settings, &mut eval) {
            return Err("Line search not converging");
        }

        // Update x with the new values of a
        unsafe { cblas::dcopy(d, &x_new, 1, &mut x, 1); }

        // Compute -s and -y
        unsafe { cblas::daxpy(d, -1., &x, 1, &mut s, 1); }
        unsafe { cblas::daxpy(d, -1., &g, 1, &mut y, 1); }

        // Normalize the Hessian at first iteration
        if k == 1 {
            let ynorm: f64 = unsafe { cblas::dnrm2(d, &y, 1) };
            for i in 0..d {
                H[(i * (d + 1)) as usize] *=
                    unsafe { cblas::ddot(d, &y, 1, &s, 1) } / (ynorm * ynorm);
            }
        }

        // Compute the Hessian
        Hessian(&mut H, &s, &y, &I, &mut B, &mut C, d, layout, part);

        if verbose {
            crate::log::print_log(&x, &g, &p, &y, &s, f, f_old, k, a, d, eval);
        };

        // Exit condition
        if !exit_condition::evaluate(&x, &g, f, f_old, d, settings) {
            break;
        }
    }

    Ok(MinimizationResult{f, x: x.to_vec(), iter: k, eval})
}