// Note: allow(unused) flag to avoid compiler warning

#[allow(unused)]
// Global minimum: [0., 0.., ...]
pub fn sphere(x: &[f64], _g: &[f64], f: &mut f64, _d: i32) {
    *f = 0.;
    for v in x {
        *f += v * v;
    }
}

#[allow(unused)]
// Global minimum: [0., -1.]
pub fn booth(r: &[f64], _g: &[f64], f: &mut f64, _d: i32) {
    let x = r[0];
    let y = r[1];
    *f = (x + 2. * y - 7.) * (x + 2. * y - 7.) + (2. * x + y - 5.) * (2. * x + y - 5.);
}
