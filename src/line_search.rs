use std::mem;
use crate::settings::{LineSearchAlg, Settings};

/// Simple line search according to Wolfe's condition
pub(crate) fn line_search_simple<Ef, Gf>(ef: &Ef, gf: &Gf, p: &Vec<f64>, x: &mut Vec<f64>, x_new: &mut Vec<f64>,
                                         g: &mut Vec<f64>, f: &mut f64, a: &mut f64, d: i32, k_out: usize, settings: &Settings, eval: &mut usize)
                                         -> bool
    where
        Ef: Fn(&Vec<f64>, &Vec<f64>, &mut f64, i32),
        Gf: Fn(&Vec<f64>, &mut Vec<f64>, &f64, i32)
{
    // Import settings
    let mu = settings.mu;
    let eta = settings.eta;

    // Estimate the value of a from the gradient
    if k_out > 1 && settings.estimate_a {
        let nrm2 = unsafe { cblas::dnrm2(d, g, 1) };
        *a = f64::min(1., 1. / nrm2.sqrt());
    } else {
        *a = 1.;
    }

    let mut a_init = *a;

    // Set energy to current value
    let fx = *f;

    let nabla_dot_p = unsafe { cblas::ddot(d, g, 1, p, 1) };

    loop {
        unsafe { cblas::dcopy(d, x, 1, x_new, 1) };
        unsafe { cblas::daxpy(d, *a, p, 1, x_new, 1) };

        // Update energy and gradient
        ef(x_new, g, f, d);
        gf(x_new, g, f, d);
        *eval += 1;

        if *f <= fx + mu * *a * nabla_dot_p && unsafe { cblas::ddot(d, g, 1, p, 1) } >= eta * nabla_dot_p {
            break;
        }

        *a = *a * 0.5;

        // Try to increase the value of initial a or stop the process if the value of a is too low
        if *a < 1e-6 {
            if a_init < 1000. {
                // Before throwing an error, try to increase the initial value of a
                a_init *= 10.;
                *a = a_init;
            } else {
                eprintln!("ERROR: Can not find a suitable value for a");
                return false;
            }
        }
    }

    unsafe { cblas::dcopy(d, x_new, 1, x, 1) };

    true
}

/// Function point
#[derive(Debug, Copy, Clone)]
pub(crate) struct Point {
    /// Argument
    pub(crate) a: f64,
    /// Value
    pub(crate) f: f64,
    /// Derivative
    pub(crate) d: f64,
}

fn sufficient_decrease(phi_0: &Point, phi_a: &Point, mu: f64) -> bool {
    phi_a.f <= phi_0.f + mu * phi_a.a * phi_0.d
}

fn curvature_condition(phi_0: &Point, phi_a: &Point, eta: f64) -> bool {
    phi_a.d.abs() <= eta * phi_0.d.abs()
}

/// Minimizer of the cubic function that interpolates f(a), f'(a), f(b), f'(b) within the
/// given interval
fn find_cubic_minimizer(mut a: f64, mut fa: f64, mut ga: f64, mut b: f64, mut fb: f64, mut gb: f64) -> f64 {
    if a > b {
        mem::swap(&mut a, &mut b);
        mem::swap(&mut fa, &mut fb);
        mem::swap(&mut ga, &mut gb);
    }
    let z = 3. * (fa - fb) / (b - a) + ga - gb;
    let d = z * z - ga * gb;
    if d <= 0. {
        // No minumum in the interval, +inf here because of the linesearch nature
        return f64::MAX;
    }
    // This code assumes a<b; negate this value if b<a
    let w = d.sqrt();
    b - (b - a) * (gb + w - z) / (gb - ga + 2. * w)
}

fn find_quadratic_minimizer_3(a: f64, fa: f64, ga: f64, b: f64, fb: f64) -> f64 {
    a + (((b - a) * (b - a)) * ga) / (2. * (fa - fb + (b - a) * ga))
}

/// Minimizer of the quadratic function that interpolates f(a), f'(a), f(b) within the
/// given interval
fn find_quadratic_minimizer_2(a: f64, ga: f64, b: f64, gb: f64) -> f64 {
    b + ((b - a) * gb) / (ga - gb)
}

fn trial_value(l: &Point, t: &Point, u: &Point, bracketed: bool) -> (f64, u32) {
    let mut ac = find_cubic_minimizer(l.a, l.f, l.d, t.a, t.f, t.d);

    // Case 1: a higher function value. The minimum is bracketed.
    if t.f > l.f {
        let aq = find_quadratic_minimizer_3(l.a, l.f, l.d, t.a, t.f);
        let res = if (ac-l.a).abs() < (aq-l.a).abs() { ac } else { (aq+ac)/2. };
        return (res, 1)
    }

    // Case 2: A lower function value and derivatives of opposite sign. The minimum is
    // bracketed.
    let ar = find_quadratic_minimizer_2(l.a, l.d, t.a, t.d);
    if (l.d > 0. && t.d < 0.) || (l.d < 0. && t.d > 0.) {
        let res = if (ac - t.a).abs() >= (ar - t.a).abs() {ac} else {ar};
        return (res, 2);
    }

    // Case 3: A lower function value, derivatives of the same sign, and the magnitude of
    // the derivative decreases.
    if t.d.abs() <= l.d.abs() {
        // The cubic function may not have a minimizer; moreover, even if it exists, it
        // can be in the wrong direction; fix it
        if (l.a < u.a && ac <= t.a) || (l.a > u.a && ac >= t.a) {
            ac = u.a
        };
        let res = if bracketed { if (ac - t.a).abs() < (ar - t.a).abs() {ac} else {ar}}
            else {if (ac - t.a).abs() > (ar - t.a).abs() {ac} else {ar}};
        return (res, 3);
    }

    // Case 4: A lower function value, derivatives of the same sign, and the magnitude of
    // the derivative does not decrease.
    let res = if bracketed {find_cubic_minimizer(t.a, t.f, t.d, u.a, u.f, u.d)} else {u.a};
    (res, 4)
}

/// Update the value of the energy and the gradient in the Point structure
pub(crate) fn update_function<Ef, Gf>(ef: &Ef, gf: &Gf, p: &Vec<f64>, x: &mut Vec<f64>, x_new: &mut Vec<f64>,
                                      g: &mut Vec<f64>, f: &mut f64, a: &mut f64, d: i32, phi: &mut Point, eval: &mut usize)
    where
        Ef: Fn(&Vec<f64>, &Vec<f64>, &mut f64, i32),
        Gf: Fn(&Vec<f64>, &mut Vec<f64>, &f64, i32) {
    unsafe{ cblas::dcopy(d, x, 1, x_new, 1)};
    unsafe{ cblas::daxpy(d, *a, p, 1, x_new, 1)};

    ef(x_new, g, f, d);
    gf(x_new, g, f, d);
    *eval += 1;

    *phi = Point{a: *a, f: *f, d : unsafe{ cblas::ddot(d, g, 1, p, 1)}};
}

/// Algorithm for efficient line search from J. J. More and D. J. Thuente, Line search algorithms with guaranteed sufficient decrease, ACM Transactions on Mathematical Software, 20 (1994)
pub(crate) fn line_search_more_thuente<Ef, Gf>(phi_s: &Point, ef: &Ef, gf: &Gf, p: &Vec<f64>, x: &mut Vec<f64>, x_new: &mut Vec<f64>,
                                               g: &mut Vec<f64>, f: &mut f64, a: &mut f64, d: i32, k_out: usize, settings: &Settings, eval: &mut usize, iter_max: usize)
                                               -> bool
    where
        Ef: Fn(&Vec<f64>, &Vec<f64>, &mut f64, i32),
        Gf: Fn(&Vec<f64>, &mut Vec<f64>, &f64, i32)
{
    // Estimate the value of a from the gradient
    *a = if k_out > 0 {1.} else {f64::min(1., 1./unsafe{ cblas::dnrm2(d, g, 1) })};

    // Import settings
    let mu = settings.mu;
    let eta = settings.eta;

    // Save the initial step
    let init_step = *a;

    // Use function psi instead if phi
    let stage1 = true;
    let bracketed = false;

    // Extract starting point from input
    let mut phi_0: Point = (*phi_s).clone();
    // Change the sign of the derivative to be consistent with paper notation
    phi_0.d *= -1.;

    // Temporary points
    let mut phi_l: Point = phi_0.clone();
    let mut phi_u = Point { a: 0., f: 0., d: 0.};
    let mut phi_j = Point { a: 0., f: 0., d: 0.};

    let width_prev = f64::MAX;

    let k: usize = 1;

    while k <= iter_max {
        if!bracketed {
            phi_u.a = *a + 4. * (*a-phi_l.a);
        }
        update_function(ef, gf, p, x, x_new, g, f, a, d, &mut phi_j, eval);
        // Change the sign of the derivative to be consistent with paper notation
        phi_j.d *= -1.;

        if sufficient_decrease(&phi_0, &phi_j, mu) && curvature_condition(&phi_0, &phi_j, eta) {
            return true;
        }

        if !phi_l.a.is_finite() || !phi_l.f.is_finite() || !phi_l.d.is_finite() ||
            !phi_j.a.is_finite() || !phi_j.f.is_finite() || !phi_j.d.is_finite() ||
            !phi_u.a.is_finite() || !phi_u.f.is_finite() || !phi_u.d.is_finite() ||
            ( (phi_l.a >= phi_u.a || phi_l.a >= phi_j.a || phi_j.a >= phi_u.a || phi_l.d >= 0.) &&
                (phi_l.a <= phi_u.a || phi_l.a <= phi_j.a || phi_j.a <= phi_u.a || phi_l.d <= 0.)) {
            break;
        }

        let psi = |phi_a: &Point| -> Point{
            Point{
                a: phi_a.a,
                f: phi_a.f - phi_0.f - mu*phi_0.d*phi_a.a,
                d: phi_a.d - mu*phi_0.d,
            }
        };

        // Decide if we want to switch to using a "Modified Updating Algorithm" (shown
        // after theorem 3.2 in the paper) by switching from using function psi to using
        // function phi. The decision follows the logic in the paragraph right before
        // theorem 3.3 in the paper.
    }

    false
}

/// A line search using More-Thuente and backtracking algorithm (faster, but might fail in some cases)
pub(crate) fn line_search_backtracking<Ef, Gf>(ef: &Ef, gf: &Gf, p: &Vec<f64>, x: &mut Vec<f64>, x_new: &mut Vec<f64>,
                                               g: &mut Vec<f64>, f: &mut f64, a: &mut f64, d: i32, k_out: usize, settings: &Settings, eval: &mut usize)
                                               -> bool
    where
        Ef: Fn(&Vec<f64>, &Vec<f64>, &mut f64, i32),
        Gf: Fn(&Vec<f64>, &mut Vec<f64>, &f64, i32)
{
    false
}

pub(crate) fn line_search<Ef, Gf>(ef: &Ef, gf: &Gf, p: &Vec<f64>, x: &mut Vec<f64>, x_new: &mut Vec<f64>,
                                  g: &mut Vec<f64>, f: &mut f64, a: &mut f64, d: i32, k_out: usize, settings: &Settings, eval: &mut usize)
                                  -> bool
    where
        Ef: Fn(&Vec<f64>, &Vec<f64>, &mut f64, i32),
        Gf: Fn(&Vec<f64>, &mut Vec<f64>, &f64, i32)
{
    match settings.line_search {
        LineSearchAlg::Simple => {
            line_search_simple(&ef, &gf, &p, x, x_new, g, f, a, d, k_out, &settings, eval)
        }
        LineSearchAlg::Backtracking => {
            //line_search_backtracking(&ef, &gf, &p, x, x_new, g, f, a, d, k_out, &settings, eval);
            println!("ERROR: Backtracking line_search is not implemented yet");
            return false;
        }
    }
}
