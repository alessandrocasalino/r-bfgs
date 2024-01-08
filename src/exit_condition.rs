use crate::settings::Settings;

pub(crate) fn evaluate(x: &Vec<f64>, g: &Vec<f64>, f: f64, f_old: f64, d: i32, settings: &Settings) -> bool {
    let g_norm = unsafe { cblas::dnrm2(d, g, 1) };
    let x_norm = unsafe { cblas::dnrm2(d, x, 1) };

    let idmax = unsafe { cblas::idamax(d, g, 1) };
    let g_el_max = num::abs(g[idmax as usize]);

    let condition1 = g_norm / f64::max(x_norm, 1.) > settings.gtol;
    let condition2 = g_el_max > settings.gmax;
    let condition3 = (f_old - f) / f64::max(f64::max(num::abs(f_old), num::abs(f)), 1.) > settings.ftol;

    condition1 && condition2 && condition3
}