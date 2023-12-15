extern crate cblas;
extern crate blas_src;

mod bfgs {

    use log::{error, trace};

    pub struct Settings {
        // Exit conditions
        ftol: f64,
        gtol: f64,
        gmax: f64,

        // Maximum number of iterations before stopping with "no convergence" error
        iter_max: usize,

        // Sufficient decrease constant (Armijo rule)
        mu: f64,
        // Curvature condition constant
        eta: f64,

        // Verbosity
        pub(crate) verbose: bool,

        // Layout with Row major
        layout: cblas::Layout,
        // Use only the upper matrix part
        part: cblas::Part,
    }

    // Default trait for Settings
    impl Default for Settings {
        fn default() -> Settings {
            Settings {
                ftol: 1e-6,
                gtol: 1e-14,
                gmax: 1e-14,
                iter_max: 10000,
                mu: 1e-4,
                eta: 0.9,
                verbose: false,
                layout: cblas::Layout::RowMajor,
                part: cblas::Part::Upper
            }
        }
    }

    #[allow(dead_code)]
    struct Point {
        a : f64,
        f : f64,
        d : f64,
    }

    fn exit_condition(x: &Vec<f64>, g: &Vec<f64>, f: f64, f_old: f64, d: i32, settings: &Settings) -> bool {
        let g_norm = unsafe { cblas::dnrm2(d, g, 1) };
        let x_norm = unsafe { cblas::dnrm2(d, x, 1) };
        let x_norm = f64::max(x_norm, 1.);

        let g_el_max = num::abs(g[ unsafe { cblas::idamax(d, g, 1) } as usize]);

        g_norm/x_norm > settings.gtol &&
            g_el_max > settings.gmax &&
            (f_old - f) / f64::max(f64::max(num::abs(f_old), num::abs(f)),1.) > settings.ftol
    }

    fn line_search<Ef, Gf> (mut ef: Ef, mut gf: Gf, p : &Vec<f64>, x : &mut Vec<f64>, x_new : &mut Vec<f64>,
                            g : &mut Vec<f64>, f: &mut f64, a: &mut f64, d: i32, k_out: usize, settings: &Settings)
        -> bool
        where
            Ef: FnMut(&mut Vec<f64>, &mut Vec<f64>, &mut f64, i32),
            Gf: FnMut(&mut Vec<f64>, &mut Vec<f64>, &mut f64, i32)
    {
        // Import settings
        let mu = settings.mu;
        let eta = settings.eta;

        // Estimate the value of a from the gradient
        if k_out != 0 {
            *a = f64::min(1., 1./unsafe {cblas::dnrm2(d, g, 1) });
        }

        // Set energy to current value
        let fx = *f;

        let nabla_dot_p = unsafe { cblas::ddot(d, g, 1, p, 1) };

        while *f >= fx + mu * *a * nabla_dot_p || unsafe { cblas::ddot(d, g, 1, p, 1) <= eta * nabla_dot_p } {
            unsafe{ cblas::dcopy(d, x, 1, x_new, 1) };
            unsafe{ cblas::daxpy(d, *a, p, 1, x_new, 1) };

            // Update energy and gradient
            ef(x_new, g, f, d);
            gf(x_new, g, f, d);

            *a = *a * 0.5;

            if *a < 1e-20 {
                error!("Can not find a suitable value for a");
                return false
            }
        }

        true
    }

    #[allow(non_snake_case)]
    fn Hessian(H : &mut Vec<f64>, s : &Vec<f64>, y : &Vec<f64>, I : &Vec<f64>, B : &mut Vec<f64>, C : &mut Vec<f64>,
               d : i32, layout: cblas::Layout, part: cblas::Part) {
        let rho : f64 = 1./unsafe {cblas::ddot(d, y, 1, s, 1)};

        // Set B to the identity
        unsafe { cblas::dcopy(d * d, I, 1, &mut *B, 1); }

        unsafe { cblas::dger(layout, d, d, -rho, y, 1, s, 1, &mut *B, d); }

        // The first matrix multiplication have one symmetric matrix
        unsafe { cblas::dsymm(layout, cblas::Side::Left, part, d, d, 1., H, d, B, d, 0., &mut *C, d); }

        // Flush the value of the Hessian to 0
        unsafe { cblas::dscal(d * d, 0., H, 1); }
        unsafe { cblas::dger(layout, d, d, rho, s, 1, s, 1, H, d); }
        // Since no matrix is symmetric, gemm is used
        unsafe { cblas::dgemm(layout, cblas::Transpose::Ordinary, cblas::Transpose::None, d, d, d, 1., B, d, C, d,
                              1., H, d); }
    }

    fn log(x : &Vec<f64>, g : &Vec<f64>, p : &Vec<f64>, y : &Vec<f64>, s : &Vec<f64>, f: f64, f_old: f64, k : usize, a: f64, d: i32) {
        trace!("--- Iteration {k}");
        trace!("                         Exit condition  ||g||/max(1,||x||) : {}",
            unsafe { cblas::dnrm2(d, g, 1) / f64::max(cblas::dnrm2(d, x, 1), 1.) });
        trace!("                  Exit condition  max(|g_i|, i = 1, ..., n) : {}",
            num::abs(g[ unsafe { cblas::idamax(d, g, 1) } as usize]));
        trace!("     Exit condition  (f^k - f^(k+1))/max(|f^k|,|f^(k+1)|,1) : {}",
            (f_old - f) / f64::max(f64::max(num::abs(f_old), num::abs(f)),1.));
        trace!("                      Mean value of the displacement vector : {}",
            unsafe { cblas::dasum(d, p, 1) } / d as f64);
        trace!("                Mean position difference from previous step : {}",
            unsafe { cblas::dasum(d, y, 1) } / d as f64 );
        trace!("                Mean gradient difference from previous step : {}",
            unsafe { cblas::dasum(d, s, 1) } / d as f64 );
        trace!("                                                          a : {}",
            a);
        trace!("                                                    s^T * y : {}",
            unsafe { cblas::ddot(d, y, 1, s, 1) });
    }

    #[allow(non_snake_case)]
    pub fn get_minimum<Ef, Gf> (mut ef: Ef, mut gf: Gf, x: &mut Vec<f64>, f: &mut f64, d : i32, settings: Settings)
        -> Option<f64>
        where
            Ef: FnMut(&mut Vec<f64>, &mut Vec<f64>, &mut f64, i32),
            Gf: FnMut(&mut Vec<f64>, &mut Vec<f64>, &mut f64, i32)
    {
        // Settings
        let iter_max = settings.iter_max;

        // BLAS definitions
        let layout = settings.layout;
        let part = settings.part;

        // Verbose (log)
        let verbose = settings.verbose;

        // Gradient definition
        let mut g: Vec<f64> = vec![0.; d as usize];

        // Update energy and gradient
        ef(x, &mut g, f, d);
        gf(x, &mut g, f, d);

        // Hessian estimation
        let mut I: Vec<f64> = vec![0.; (d * d) as usize];
        let mut H: Vec<f64> = vec![0.; (d * d) as usize];
        for i in 0..d {
            I[(i * (d + 1)) as usize] = 1.;
        }
        unsafe { cblas::dcopy(d * d, &*I, 1, &mut *H, 1); }

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
        let mut B: Vec<f64> = vec![0.; d as usize];
        let mut C: Vec<f64> = vec![0.; d as usize];

        //Iteration number
        let mut k: usize = 0;

        // Main loop
        loop {
            // Stop if reaching the maximum number of iterations requested
            if k >= iter_max {
                return None;
            }
            k += 1;

            // Store current values
            unsafe { cblas::dcopy(d, &*x, 1, &mut *s, 1); }
            unsafe { cblas::dcopy(d, &*g, 1, &mut *y, 1); }
            f_old = *f;

            // Store current values
            unsafe { cblas::dsymv(layout, part, d, -1., &*H, d, &*g, 1, 0., &mut *p, 1); }

            // Save the value of Phi_0 to be used for both line_search
            let _phi_0: Point = Point { a: 0., f: *f, d: unsafe { cblas::ddot(d, &*g, 1, &mut *p, 1) } };
            /* Find a according to Wolfe's condition:
             * - more_thuente: check if this can be used to find a (if yes use that a value)
             * - backtracking: otherwise evaluate the second with starting a from the first
             * This ensures that most of the steps have a = a_max = 1
             * NOTE: line_search also updates f
             */
            if !line_search(&mut ef, &mut gf, &p, x, &mut x_new, &mut g, f, &mut a, d, k, &settings) {
                return None
            }

            // Update x with the new values of a
            unsafe { cblas::dcopy(d, &*x_new, 1, &mut *x, 1); }

            // Compute -s and -y
            unsafe { cblas::daxpy(d, -1., &*x, 1, &mut *s, 1); }
            unsafe { cblas::daxpy(d, -1., &*g, 1, &mut *y, 1); }

            // Normalize the Hessian at first iteration
            if k == 1 {
                let ynorm: f64 = unsafe { cblas::dnrm2(d, &*y, 1) };
                for i in 0..d {
                    H[(i * (d + 1)) as usize] *=
                        unsafe { cblas::ddot(d, &*y, 1, &*s, 1) } / (ynorm * ynorm);
                }
            }

            // Compute the Hessian
            Hessian(&mut H, &s, &y, &I, &mut B, &mut C, d, layout, part);

            if verbose {
                log(x, &g, &p, &y, &s, *f, f_old, k, a, d)
            };

            // Exit condition
            if exit_condition(&x, &g, *f, f_old, d, &settings) {
                break;
            }
        }

        Some(2.)
    }

}



#[cfg(test)]
mod tests {
    use crate::bfgs;
    //use super::*;

    #[test]
    fn bfgs() {

        // Create settings with default parameters
        let mut settings : bfgs::Settings = Default::default();

        settings.verbose = true;

        let d: i32 = 10;
        let mut f=0.;
        let mut x = vec![1.; d as usize];

        let ef = |x: &mut Vec<f64>, _g:&mut Vec<f64>, f:&mut f64, d: i32| {
            *f = unsafe { cblas::ddot(d, x, 1, x, 1) }
        };
        let gf = |x: &mut Vec<f64>, g:&mut Vec<f64>, _f:&mut f64, d: i32| {
            for i in 0..d {
                g[i as usize] = 2. * x[i as usize];
            }
        };

        let result = bfgs::get_minimum(ef, gf, &mut x, &mut f, d, settings);
        println!("{:?}", result);
        //assert_eq!(result, Some(4.));
    }
}

