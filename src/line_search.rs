use log::error;
use crate::Settings;

pub(crate) fn line_search<Ef, Gf>(ef: &Ef, gf: &Gf, p: &Vec<f64>, x: &mut Vec<f64>, x_new: &mut Vec<f64>,
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
        *a = f64::min(1., 1. / nrm2);
    } else {
        *a = 1.;
    }

    // Set energy to current value
    let fx = *f;

    let nabla_dot_p = unsafe { cblas::ddot(d, g, 1, p, 1) };

    loop {
        *a = *a * 0.5;

        unsafe { cblas::dcopy(d, x, 1, x_new, 1) };
        unsafe { cblas::daxpy(d, *a, p, 1, x_new, 1) };

        // Update energy and gradient
        ef(x_new, g, f, d);
        gf(x_new, g, f, d);
        *eval += 1;

        if !(*f >= fx + mu * *a * nabla_dot_p || unsafe { cblas::ddot(d, g, 1, p, 1) } <= eta * nabla_dot_p) {
            break;
        }

        if *a < 1e-20 {
            error!("Can not find a suitable value for a");
            return false;
        }
    }

    unsafe { cblas::dcopy(d, x_new, 1, x, 1) };

    true
}