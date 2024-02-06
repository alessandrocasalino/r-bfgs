// allow(unused) flag to avoid compiler warning
#[allow(unused)]
// Fix for zero value
pub fn check_result(x: &[f64], cmp: &[f64]) {
    let d = x.len();
    for i in 0..d {
        assert!((x[i] - cmp[i]).abs() < 0.001);
    }
}
