use crate::{MinimizationResult, Settings};

pub(crate) fn gradient_descent<Function, Gradient>(fn_function: &Function, fn_gradient: &Gradient,
                                                   x0: &[f64], settings: &Settings)
                                                   -> Result<MinimizationResult, &'static str>
    where
        Function: Fn(&[f64], &[f64], &mut f64, i32),
        Gradient: Fn(&[f64], &mut [f64], &f64, i32)
{
    // Settings
    let iter_max = settings.iter_max;

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

    // Iteration counter
    let mut iter: usize = 0;

    // Learning rate
    let alpha = 0.1;

    // Iteration
    while iter < iter_max {
        // Update the iteration counter
        iter += 1;

        // Update the position
        unsafe { cblas::daxpy(d, -alpha, &g, 1, &mut x, 1); }

        // Update energy and gradient
        fn_function(&x, &g, &mut f, d);
        fn_gradient(&x, &mut g, &f, d);
        eval += 1;

        let g_norm = unsafe { cblas::dnrm2(d, &g, 1) };
        if g_norm < settings.gtol {
            return Ok(MinimizationResult { f, x: x.to_vec(), iter, eval });
        }
    }

    Err("Maximum number of iterations reached")
}


