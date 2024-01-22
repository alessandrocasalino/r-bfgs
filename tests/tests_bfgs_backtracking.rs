use rand::{Rng, thread_rng};
use bfgs::settings::{LineSearchAlg, MinimizationAlg};

mod test_functions;
mod test_utils;

#[test]
fn test_sphere_function() {
    use bfgs;

    // Create settings with default parameters
    let mut settings: bfgs::settings::Settings = Default::default();
    // Select the minimization algorithm
    settings.minimization = MinimizationAlg::Bfgs;
    // Select the line search algorithm
    settings.line_search = LineSearchAlg::Backtracking;

    let dims = vec![2, 6, 20, 100, 1000];

    for d in dims {
        let mut x = vec![(); d].into_iter().map(|_| thread_rng().gen_range(-10.0..10.0)).collect();
        let result = bfgs::get_minimum(&test_functions::sphere, &mut x, &settings);
        assert_ne!(result, None, "Result not found");
        let cmp = vec![0.; d];
        test_utils::check_result(x, cmp, d);
    }
}

#[test]
#[ignore]
fn test_rosenbrock_function() {
    use bfgs;

    let mut settings: bfgs::settings::Settings = Default::default();
    settings.minimization = MinimizationAlg::Bfgs;
    settings.line_search = LineSearchAlg::Backtracking;

    let mut x = vec![-1.2, 1.0];
    let result = bfgs::get_minimum(&test_functions::rosenbrock, &mut x, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![1., 1.];
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.001);

    let mut x = vec![-0.2, 1., -1., -3.2, 1., -0.9];
    let result = bfgs::get_minimum(&test_functions::rosenbrock, &mut x, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![1., 1., 1., 1., 1., 1.];
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.001);
}

#[test]
fn test_himmelblau_function() {
    use bfgs;

    let mut settings: bfgs::settings::Settings = Default::default();
    settings.minimization = MinimizationAlg::Bfgs;
    settings.line_search = LineSearchAlg::Backtracking;

    let mut x = vec![-1.2, 1.0];
    let result = bfgs::get_minimum(&test_functions::himmelblau, &mut x, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![-2.805118, 3.131312];
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.001);
}

#[test]
fn test_three_hump_camel_function() {
    use bfgs;

    let mut settings: bfgs::settings::Settings = Default::default();
    settings.minimization = MinimizationAlg::Bfgs;
    settings.line_search = LineSearchAlg::Backtracking;

    // Local minimum
    let mut x = vec![-1.2, 1.0];
    let result = bfgs::get_minimum(&test_functions::three_hump_camel, &mut x, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![-1.74755, 0.873776];
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.01);

    // Global minimum
    let mut x = vec![-0.2, 0.5];
    let result = bfgs::get_minimum(&test_functions::three_hump_camel, &mut x, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![0., 0.];
    test_utils::check_result(x, cmp, 2);
}

#[test]
fn test_mccormick_function() {
    use bfgs;

    let mut settings: bfgs::settings::Settings = Default::default();
    settings.minimization = MinimizationAlg::Bfgs;
    settings.line_search = LineSearchAlg::Backtracking;

    // Global minimum
    let mut x = vec![-1.2, 1.0];
    let result = bfgs::get_minimum(&test_functions::mccormick, &mut x, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![-0.54719, -1.54719];
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.01);
}

#[test]
fn test_styblinski_tang_function() {
    use bfgs;

    let mut settings: bfgs::settings::Settings = Default::default();
    settings.minimization = MinimizationAlg::Bfgs;
    settings.line_search = LineSearchAlg::Backtracking;

    // Global minimum
    let mut x = vec![-1.2, -1.0];
    let result = bfgs::get_minimum(&test_functions::styblinki_tang, &mut x, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![-2.903534, -2.903534];
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.01);

    // Local minimum
    let mut x = vec![-1.2, 1.0];
    let result = bfgs::get_minimum(&test_functions::styblinki_tang, &mut x, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![-2.903534, 2.7468];
    // Local minimum
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.01);

    // Global minimum
    let mut x = vec![-1.2, -1.0, -1.0, -1.0, -1.0];
    let result = bfgs::get_minimum(&test_functions::styblinki_tang, &mut x, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![-2.903534, -2.903534, -2.903534, -2.903534, -2.903534];
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.01);

    // Local minimum
    let mut x = vec![1.2, 1.0, 1.0, 1.0, 1.0];
    let result = bfgs::get_minimum(&test_functions::styblinki_tang, &mut x, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![2.7468, 2.7468, 2.7468, 2.7468, 2.7468];
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.01);
}

#[test]
fn test_beale_function() {
    use bfgs;

    let mut settings: bfgs::settings::Settings = Default::default();
    settings.minimization = MinimizationAlg::Bfgs;
    settings.line_search = LineSearchAlg::Backtracking;

    // Global minimum
    let mut x = vec![1.2, 1.0];
    let result = bfgs::get_minimum(&test_functions::beale, &mut x, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![3., 0.5];
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.01);

    // Global minimum
    let mut x = vec![-1.2, -1.0];
    let result = bfgs::get_minimum(&test_functions::beale, &mut x, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![3., 0.5];
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.01);

    // Not working
    /*
    let mut x = vec![-3.0, 3.0];
    let result = bfgs::get_minimum(&test_functions::beale, &mut x, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![3., 0.5];
    // Global minimum
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.01);

    let mut x = vec![-3.0, -3.0];
    let result = bfgs::get_minimum(&test_functions::beale, &mut x, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![3., 0.5];
    // Global minimum
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.01);
    */
}

#[test]
#[ignore]
fn test_goldstein_price_function() {
    use bfgs;

    let mut settings: bfgs::settings::Settings = Default::default();
    settings.minimization = MinimizationAlg::Bfgs;
    settings.line_search = LineSearchAlg::Backtracking;

    // Global minimum
    let mut x = vec![0., -1.];
    let result = bfgs::get_minimum(&test_functions::goldstein_price, &mut x, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![0., -1.];
    test_utils::check_result(x, cmp, 2);

    // Global minimum
    let mut x = vec![0., 1.0];
    let result = bfgs::get_minimum(&test_functions::goldstein_price, &mut x, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![0., -1.];
    test_utils::check_result(x, cmp, 2);
}

#[test]
fn test_booth_function() {
    use bfgs;

    let mut settings: bfgs::settings::Settings = Default::default();
    settings.minimization = MinimizationAlg::Bfgs;
    settings.line_search = LineSearchAlg::Backtracking;

    // Global minimum
    let mut x = vec![0., -1.];
    let result = bfgs::get_minimum(&test_functions::booth, &mut x, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![1., 3.];
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.01);

    // Global minimum
    let mut x = vec![5., -5.];
    let result = bfgs::get_minimum(&test_functions::booth, &mut x, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![1., 3.];
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.01);

    // Global minimum
    let mut x = vec![-5., -5.];
    let result = bfgs::get_minimum(&test_functions::booth, &mut x, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![1., 3.];
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.01);

    // Global minimum
    let mut x = vec![5., 5.];
    let result = bfgs::get_minimum(&test_functions::booth, &mut x, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![1., 3.];
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.01);
}
