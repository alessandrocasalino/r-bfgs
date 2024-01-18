use std::collections::VecDeque;
use crate::{exit_condition, line_search};
use crate::lbfgs::lbfgs_deque::{HistoryPoint, fifo_operation};
use crate::settings::Settings;

mod lbfgs_deque;

/// Routine to find the direction vector for L-BFGS routine as in
/// Jorge Nocedal Stephen J. Wright "Numerical Optimization" (2nd Edition)
/// Algorithm 7.4 (L-BFGS two-loop recursion)
#[allow(non_snake_case)]
fn search_direction(history: &VecDeque<HistoryPoint>, p: &mut Vec<f64>, g: &Vec<f64>, H: &Vec<f64>, alpha: &mut Vec<f64>, d: i32, settings: &Settings) {
    let m = history.len();

    // If no points in history, estimate the Hessian
    if m < 1 {
        unsafe{ cblas::dcopy(d, g, 1, p, 1); }
        unsafe{ cblas::dscal(d, -1., p, 1); }
        return
    }

    // Temporary variables for two-loops
    let mut beta : f64;

    // Starting value for p (current gradient)
    unsafe{ cblas::dcopy(d, g, 1, p, 1); }

    // Forward loop
    for i in 0..m {
        let h = &history[i];
        alpha[i] = unsafe{ -cblas::ddot(d, &*h.s, 1, p, 1) /
            cblas::ddot(d, &*h.y, 1, &*h.s, 1) };
        unsafe{ cblas::daxpy(d, alpha[i], &*h.y, 1, p, 1); }
    }

    // Compute first version of p with gamma
    // If M1QN3 is not used, the "Hessian" is considered a scalar
    if !settings.m1qn3 {
        let h = &history[0];
        let y = unsafe{ cblas::dnrm2(d, &*h.y, 1)};
        unsafe{ cblas::dscal(d, -cblas::ddot(d, &*h.y, 1, &*h.s, 1) / (y * y), p, 1); }
    } else {
        for i in 0..d {
            p[i as usize] = -H[i as usize] * p[i as usize];
        }
    }

    // Backward loop
    for i in 0..m {
        let h = &history[m - 1 - i];
        beta = unsafe{ cblas::ddot(d, &*h.y, 1, p, 1) /
            cblas::ddot(d, &*h.y, 1, &*h.s, 1) };
        unsafe{ cblas::daxpy(d, -beta + alpha[m - 1 - i], &*h.s, 1, p, 1); }
    }
}

/// Algorithm M1QN3 B2 to update the Hessian from
/// Gilbert, J.C., Lemaréchal, C. Some numerical experiments with variable-storage
/// quasi-Newton algorithms. Mathematical Programming 45, 407–435 (1989).
/// https://doi.org/10.1007/BF01589113 In particular, equation (4.9) for the diagonal
/// Hessian. Since we are using a vector to store the diagonal part, we can not use blas
/// for all computations.
#[allow(non_snake_case)]
fn Hessian(H: &mut Vec<f64>, s: &Vec<f64>, y: &Vec<f64>, d: i32) {
    let mut dinvss = 0.;
    let mut dyy = 0.;

    let ys = unsafe {cblas::ddot(d, y, 1, s, 1)};

    for i in 0..d {
        dinvss += s[i as usize] * s[i as usize] / H[i as usize];
        dyy += y[i as usize] * y[i as usize] * H[i as usize];
    }

    for i in 0..d {
        H[i as usize] = 1./(dyy / (ys * H[i as usize]) + y[i as usize] * y[i as usize] / ys -
            dyy * s[i as usize] * s[i as usize] / (ys * dinvss * H[i as usize] * H[i as usize]));
    }
}

/// L-BFGS routine as in Jorge Nocedal Stephen J. Wright "Numerical Optimization" (2nd
/// Edition) Algorithm 7.4 (L-BFGS Method). The approximation employed is to consider a
/// diagonal Hessian with the same elements, i.e., an identity matrix multiplied by a
/// constant, as in equation (7.20)
#[allow(non_snake_case)]
pub fn lbfgs<Ef, Gf>(ef: &Ef, gf: &Gf, x: &mut Vec<f64>, settings: &Settings)
                    -> Option<f64>
    where
        Ef: Fn(&Vec<f64>, &Vec<f64>, &mut f64, i32),
        Gf: Fn(&Vec<f64>, &mut Vec<f64>, &f64, i32)
{
    // Settings
    let iter_max = settings.iter_max;
    // Verbose (log)
    let verbose = settings.verbose;

    // Get the dimension
    let d=x.len() as i32;

    // Function update evaluations
    let mut eval: usize = 0;

    // Energy definition
    let mut f: f64 = 0.;

    // Gradient definition
    let mut g: Vec<f64> = vec![0.; d as usize];

    // Update energy and gradient
    ef(x, &g, &mut f, d);
    gf(x, &mut g, &f, d);
    eval += 1;

    // Hessian estimation
    let mut H: Vec<f64> = vec![1.; d as usize];

    // Search direction
    let mut p: Vec<f64> = vec![0.; d as usize];

    // Temporary vectors for line_search routine
    let mut x_new: Vec<f64> = vec![0.; d as usize];

    // Line search element coefficient
    let mut a: f64 = 1.;
    // Temporary value of f
    let mut f_old: f64;

    // Queue for L-BFGS history
    let mut history: VecDeque<HistoryPoint> = VecDeque::new();

    // Temporary vector for L-BFGS two-loop recursion
    let mut alpha = vec![0.; settings.history_depth];

    //Iteration number
    let mut k: usize = 0;

    // Main loop
    loop {
        // Stop if reaching the maximum number of iterations requested
        if k >= iter_max {
            return None;
        }
        k += 1;

        // Difference x_k+1 - x_k
        let mut s: Vec<f64> = vec![0.; d as usize];
        // Difference of gradients
        let mut y: Vec<f64> = vec![0.; d as usize];

        // Store current values
        unsafe { cblas::dcopy(d, &*x, 1, &mut *s, 1); }
        unsafe { cblas::dcopy(d, &*g, 1, &mut *y, 1); }
        f_old = f;

        // Store current values
        search_direction(&history, &mut p, &g, &H, &mut alpha, d, &settings);

        // Save the value of Phi_0 to be used for both line_search
        let phi_0: line_search::Point = line_search::Point { a: 0., f: f, d: unsafe { cblas::ddot(d, &*g, 1, &mut *p, 1) } };

        // Perform line search (updating a)
        if !line_search::line_search(&ef, &gf, &phi_0, &p, x, &mut x_new, &mut g, &mut f, &mut a, d, k, &settings, &mut eval) {
            eprintln!("ERROR: Line search not converging");
            return None;
        }

        // Update x with the new values of a
        unsafe { cblas::dcopy(d, &*x_new, 1, &mut *x, 1); }

        // Compute -s and -y
        unsafe { cblas::daxpy(d, -1., &*x, 1, &mut *s, 1); }
        unsafe { cblas::daxpy(d, -1., &*g, 1, &mut *y, 1); }

        // Compute the Hessian
        if settings.m1qn3 {
            Hessian(&mut H, &s, &y, d);
        }

        if verbose {
            crate::log::print_log(x, &g, &p, &y, &s, f, f_old, k, a, d, eval);
        };

        // Store in deque
        fifo_operation(&mut history, s, y, &settings);

        // Exit condition
        if !exit_condition::evaluate(&x, &g, f, f_old, d, &settings) {
            break;
        }
    }

    Some(f)
}