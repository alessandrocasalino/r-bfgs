use std::path::Path;

use bfgs::settings::MinimizationAlgorithm;

// Global minimum: [1., 3.]
fn booth(r: &[f64], _g: &[f64], f: &mut f64, _d: i32) {
    let x = r[0];
    let y = r[1];
    *f = (x + 2. * y - 7.) * (x + 2. * y - 7.) + (2. * x + y - 5.) * (2. * x + y - 5.);
}

fn main() {
    let mut settings: bfgs::settings::Settings = Default::default();
    settings.minimization = MinimizationAlgorithm::Bfgs;
    settings.save_history = true;

    let x = vec![2., 3.];
    let result = bfgs::get_minimum(&booth, &x, &settings).expect("Result not found");
    let cmp = vec![1., 3.];
    float_eq::assert_float_eq!(result.x, cmp, rmax_all <= 0.01);

    let path = Path::new("test.png");
    result.plot_points(path).expect("Plot error");
}
