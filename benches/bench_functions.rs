// Note: allow(unused) flag to avoid compiler warning

#[allow(unused)]
// Global minimum: [0., 0.., ...]
pub fn sphere(x: &[f64], _g: &[f64], f: &mut f64, _d: i32) {
    *f = 0.;
    for v in x {
        *f += v * v;
    }
}
