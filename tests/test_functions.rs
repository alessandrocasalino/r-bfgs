// allow(unused) flag to avoid compiler warning

#[allow(unused)]
pub(crate) fn test_function_sphere(x: &Vec<f64>, _g: &Vec<f64>, f: &mut f64, d: i32) {
    *f = 0.;
    for i in 0..d as usize {
        *f += x[i] * x[i];
    }
}
