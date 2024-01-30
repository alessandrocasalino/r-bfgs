use rand::{Rng, thread_rng};
use bfgs::settings::MinimizationAlg;

mod test_functions;
mod test_utils;

#[test]
fn test_sphere_function() {
    use bfgs;

    // Create settings with default parameters
    let mut settings: bfgs::settings::Settings = Default::default();
    // Select the minimization algorithm
    settings.minimization = MinimizationAlg::GradientDescent;
    settings.iter_max = 1000;

    let dims = vec![2, 6, 20, 100, 1000];

    for d in dims {
        let mut x = vec![(); d].into_iter().map(|_| thread_rng().gen_range(-10.0..10.0)).collect();
        let result = bfgs::get_minimum(&test_functions::sphere, &mut x, &settings);
        assert_ne!(result, None, "Result not found");
        let cmp = vec![0.; d];
        test_utils::check_result(x, cmp);
    }
}
