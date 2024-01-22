// allow(unused) flag to avoid compiler warning

#[allow(unused)]
// Global minimum: [0., 0.., ...]
pub fn sphere(x: &Vec<f64>, _g: &Vec<f64>, f: &mut f64, d: i32) {
    *f = 0.;
    for i in 0..d as usize {
        *f += x[i] * x[i];
    }
}

#[allow(unused)]
// Global minimum: [1., 1., ...]
pub fn rosenbrock(x: &Vec<f64>, _g: &Vec<f64>, f: &mut f64, d: i32) {
    *f = 0.;
    for i in 0..(d - 1) as usize {
        let t1 = x[i + 1] - x[i] * x[i];
        let t2 = 1. - x[i];
        *f += 100. * t1 * t1 + t2 * t2;
    }
}

#[allow(unused)]
// Global minima: [-2.805118, 3.131312] [3.,2.] [-3.779310, -3.283186] [3.584428, -1.848126]
pub fn himmelblau(x: &Vec<f64>, _g: &Vec<f64>, f: &mut f64, _d: i32) {
    *f = (x[0] * x[0] + x[1] - 11.) * (x[0] * x[0] + x[1] - 11.) + (x[0] + x[1] * x[1] - 7.) * (x[0] + x[1] * x[1] - 7.);
}

#[allow(unused)]
// Global minimum: [0., 0.]
// Local minimum: [-1.74755, 0.873776]
pub fn three_hump_camel(x: &Vec<f64>, _g: &Vec<f64>, f: &mut f64, _d: i32) {
    *f = 2. * x[0] * x[0] - 1.05 * x[0] * x[0] * x[0] * x[0] + x[0] * x[0] * x[0] * x[0] * x[0] * x[0] / 6. + x[0] * x[1] + x[1] * x[1];
}

#[allow(unused)]
// Global minimum: [-0.54719, -1.54719]
pub fn mccormick(x: &Vec<f64>, _g: &Vec<f64>, f: &mut f64, _d: i32) {
    *f = f64::sin(x[0] + x[1]) + (x[0] - x[1]) * (x[0] - x[1]) - 1.5 * x[0] + 2.5 * x[1] + 1.;
}

#[allow(unused)]
// Global minimum: [-2.903524, -2.903524, ...]
// Local minima with 2.7468 on any of the elements of the resulting vector, e.g. [-2.903534, 2.7468]
pub fn styblinki_tang(x: &Vec<f64>, _g: &Vec<f64>, f: &mut f64, d: i32) {
    *f = 0.;
    for i in 0..(d as usize) {
        *f += x[i] * x[i] * x[i] * x[i] - 16. * x[i] * x[i] + 5. * x[i];
    }
    *f *= 0.5;
}

#[allow(unused)]
// Global minimum: [3., 0.5]
pub fn beale(x: &Vec<f64>, _g: &Vec<f64>, f: &mut f64, _d: i32) {
    *f = (1.5 - x[0] + x[0] * x[1]) * (1.5 - x[0] + x[0] * x[1]) +
        (2.25 - x[0] + x[0] * x[1] * x[1]) * (2.25 - x[0] + x[0] * x[1] * x[1]) +
        (2.625 - x[0] + x[0] * x[1] * x[1] * x[1]) * (2.625 - x[0] + x[0] * x[1] * x[1] * x[1]);
}

#[allow(unused)]
// Global minimum: [0., -1.]
// Local minima: [1.2, 0.8] [1.8, 0.2] [-0.6, -0.4]
pub fn goldstein_price(r: &Vec<f64>, _g: &Vec<f64>, f: &mut f64, _d: i32) {
    let x = r[0];
    let y = r[1];
    *f = (1. + (x + y + 1.) * (x + y + 1.) * (19. - 14. * x + 3. * x * x - 14. * y + 6. * x * y + 3. * y * y)) *
        (30. + (2. * x - 3. * y) * (2. * x - 3. * y) * (18. - 32. * x + 12. * x * x + 48. * y - 36. * x * y + 27. * y * y));
}

#[allow(unused)]
// Global minimum: [0., -1.]
pub fn booth(r: &Vec<f64>, _g: &Vec<f64>, f: &mut f64, _d: i32) {
    let x = r[0];
    let y = r[1];
    *f = (x + 2. * y - 7.) * (x + 2. * y - 7.) + (2. * x + y - 5.) * (2. * x + y - 5.);
}

#[allow(unused)]
// Global minimum: [0., 0.]
pub fn matyas(r: &Vec<f64>, _g: &Vec<f64>, f: &mut f64, _d: i32) {
    let x = r[0];
    let y = r[1];
    *f = 0.26 * (x *x + y*y) - 0.48*x*y;
}
