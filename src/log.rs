#[allow(clippy::too_many_arguments)]
pub(crate) fn print_log(x: &[f64], g: &[f64], p: &[f64], y: &[f64], s: &[f64], f: f64, f_old: f64, k: usize, a: f64,
                        d: i32, eval: usize) {
    println!("--- Iteration {k}");
    println!("                                       Function evaluations : {}", eval);
    println!("                         Exit condition  ||g||/max(1,||x||) : {}",
             unsafe { cblas::dnrm2(d, g, 1) / f64::max(cblas::dnrm2(d, x, 1), 1.) });
    println!("                  Exit condition  max(|g_i|, i = 1, ..., n) : {}",
             num::abs(g[unsafe { cblas::idamax(d, g, 1) } as usize]));
    println!("     Exit condition  (f^k - f^(k+1))/max(|f^k|,|f^(k+1)|,1) : {}",
             (f_old - f) / f64::max(f64::max(num::abs(f_old), num::abs(f)), 1.));
    println!("                      Mean value of the displacement vector : {}",
             unsafe { cblas::dasum(d, p, 1) } / d as f64);
    println!("                Mean position difference from previous step : {}",
             unsafe { cblas::dasum(d, y, 1) } / d as f64);
    println!("                Mean gradient difference from previous step : {}",
             unsafe { cblas::dasum(d, s, 1) } / d as f64);
    println!("                                                          a : {}",
             a);
    println!("                                                    s^T * y : {}",
             unsafe { cblas::ddot(d, y, 1, s, 1) });
}