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
        let x: Vec<f64> = vec![(); d].into_iter().map(|_| thread_rng().gen_range(-10.0..10.0)).collect();
        let result = bfgs::get_minimum(&test_functions::sphere, &x, &settings);
        assert!(result.is_ok(), "Result not found: {}", result.err().unwrap());
        let cmp = vec![0.; d];
        test_utils::check_result(&result.unwrap().x, &cmp);
    }
}

#[test]
#[ignore]
fn test_rosenbrock_function() {
    use bfgs;

    let mut settings: bfgs::settings::Settings = Default::default();
    settings.minimization = MinimizationAlg::Bfgs;
    settings.line_search = LineSearchAlg::Backtracking;

    let x = vec![-1.2, 1.0];
    let result = bfgs::get_minimum(&test_functions::rosenbrock, &x, &settings);
    assert!(result.is_ok(), "Result not found: {}", result.err().unwrap());
    let cmp = vec![1., 1.];
    float_eq::assert_float_eq!(result.unwrap().x, cmp, rmax_all <= 0.001);

    let x = vec![-0.2, 1., -1., -3.2, 1., -0.9];
    let result = bfgs::get_minimum(&test_functions::rosenbrock, &x, &settings);
    assert!(result.is_ok(), "Result not found: {}", result.err().unwrap());
    let cmp = vec![1., 1., 1., 1., 1., 1.];
    float_eq::assert_float_eq!(result.unwrap().x, cmp, rmax_all <= 0.001);
}

#[test]
fn test_himmelblau_function() {
    use bfgs;

    let mut settings: bfgs::settings::Settings = Default::default();
    settings.minimization = MinimizationAlg::Bfgs;
    settings.line_search = LineSearchAlg::Backtracking;

    let x = vec![-1.2, 1.0];
    let result = bfgs::get_minimum(&test_functions::himmelblau, &x, &settings);
    assert!(result.is_ok(), "Result not found: {}", result.err().unwrap());
    let cmp = vec![-2.805118, 3.131312];
    float_eq::assert_float_eq!(result.unwrap().x, cmp, rmax_all <= 0.001);
}

#[test]
fn test_three_hump_camel_function() {
    use bfgs;

    let mut settings: bfgs::settings::Settings = Default::default();
    settings.minimization = MinimizationAlg::Bfgs;
    settings.line_search = LineSearchAlg::Backtracking;

    // Local minimum
    let x = vec![-1.2, 1.0];
    let result = bfgs::get_minimum(&test_functions::three_hump_camel, &x, &settings);
    assert!(result.is_ok(), "Result not found: {}", result.err().unwrap());
    let cmp = vec![-1.74755, 0.873776];
    float_eq::assert_float_eq!(result.unwrap().x, cmp, rmax_all <= 0.01);

    // Global minimum
    let x = vec![-0.2, 0.5];
    let result = bfgs::get_minimum(&test_functions::three_hump_camel, &x, &settings);
    assert!(result.is_ok(), "Result not found: {}", result.err().unwrap());
    let cmp = vec![0., 0.];
    test_utils::check_result(&result.unwrap().x, &cmp);
}

#[test]
fn test_mccormick_function() {
    use bfgs;

    let mut settings: bfgs::settings::Settings = Default::default();
    settings.minimization = MinimizationAlg::Bfgs;
    settings.line_search = LineSearchAlg::Backtracking;

    // Global minimum
    let x = vec![-1.2, 1.0];
    let result = bfgs::get_minimum(&test_functions::mccormick, &x, &settings);
    assert!(result.is_ok(), "Result not found: {}", result.err().unwrap());
    let cmp = vec![-0.54719, -1.54719];
    float_eq::assert_float_eq!(result.unwrap().x, cmp, rmax_all <= 0.01);
}

#[test]
fn test_styblinski_tang_function() {
    use bfgs;

    let mut settings: bfgs::settings::Settings = Default::default();
    settings.minimization = MinimizationAlg::Bfgs;
    settings.line_search = LineSearchAlg::Backtracking;

    // Global minimum
    let x = vec![-1.2, -1.0];
    let result = bfgs::get_minimum(&test_functions::styblinki_tang, &x, &settings);
    assert!(result.is_ok(), "Result not found: {}", result.err().unwrap());
    let cmp = vec![-2.903534, -2.903534];
    float_eq::assert_float_eq!(result.unwrap().x, cmp, rmax_all <= 0.01);

    // Local minimum
    let x = vec![-1.2, 1.0];
    let result = bfgs::get_minimum(&test_functions::styblinki_tang, &x, &settings);
    assert!(result.is_ok(), "Result not found: {}", result.err().unwrap());
    let cmp = vec![-2.903534, 2.7468];
    // Local minimum
    float_eq::assert_float_eq!(result.unwrap().x, cmp, rmax_all <= 0.01);

    // Global minimum
    let x = vec![-1.2, -1.0, -1.0, -1.0, -1.0];
    let result = bfgs::get_minimum(&test_functions::styblinki_tang, &x, &settings);
    assert!(result.is_ok(), "Result not found: {}", result.err().unwrap());
    let cmp = vec![-2.903534, -2.903534, -2.903534, -2.903534, -2.903534];
    float_eq::assert_float_eq!(result.unwrap().x, cmp, rmax_all <= 0.01);

    // Local minimum
    let x = vec![1.2, 1.0, 1.0, 1.0, 1.0];
    let result = bfgs::get_minimum(&test_functions::styblinki_tang, &x, &settings);
    assert!(result.is_ok(), "Result not found: {}", result.err().unwrap());
    let cmp = vec![2.7468, 2.7468, 2.7468, 2.7468, 2.7468];
    float_eq::assert_float_eq!(result.unwrap().x, cmp, rmax_all <= 0.01);
}

#[test]
fn test_beale_function() {
    use bfgs;

    let mut settings: bfgs::settings::Settings = Default::default();
    settings.minimization = MinimizationAlg::Bfgs;
    settings.line_search = LineSearchAlg::Backtracking;

    // Global minimum
    let x = vec![1.2, 1.0];
    let result = bfgs::get_minimum(&test_functions::beale, &x, &settings);
    assert!(result.is_ok(), "Result not found: {}", result.err().unwrap());
    let cmp = vec![3., 0.5];
    float_eq::assert_float_eq!(result.unwrap().x, cmp, rmax_all <= 0.01);

    // Global minimum
    let x = vec![-1.2, -1.0];
    let result = bfgs::get_minimum(&test_functions::beale, &x, &settings);
    assert!(result.is_ok(), "Result not found: {}", result.err().unwrap());
    let cmp = vec![3., 0.5];
    float_eq::assert_float_eq!(result.unwrap().x, cmp, rmax_all <= 0.01);

    // Not working
    /*
    let x = vec![-3.0, 3.0];
    let result = bfgs::get_minimum(&test_functions::beale, &x, &settings);
    assert!(result.is_ok(), "Result not found: {}", result.err().unwrap());
    let cmp = vec![3., 0.5];
    // Global minimum
    float_eq::assert_float_eq!(result.unwrap().x, cmp, rmax_all <= 0.01);

    let x = vec![-3.0, -3.0];
    let result = bfgs::get_minimum(&test_functions::beale, &x, &settings);
    assert!(result.is_ok(), "Result not found: {}", result.err().unwrap());
    let cmp = vec![3., 0.5];
    // Global minimum
    float_eq::assert_float_eq!(result.unwrap().x, cmp, rmax_all <= 0.01);
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
    let x = vec![0., -1.];
    let result = bfgs::get_minimum(&test_functions::goldstein_price, &x, &settings);
    assert!(result.is_ok(), "Result not found: {}", result.err().unwrap());
    let cmp = vec![0., -1.];
    test_utils::check_result(&result.unwrap().x, &cmp);

    // Global minimum
    let x = vec![0., 1.0];
    let result = bfgs::get_minimum(&test_functions::goldstein_price, &x, &settings);
    assert!(result.is_ok(), "Result not found: {}", result.err().unwrap());
    let cmp = vec![0., -1.];
    test_utils::check_result(&result.unwrap().x, &cmp);
}

#[test]
fn test_booth_function() {
    use bfgs;

    let mut settings: bfgs::settings::Settings = Default::default();
    settings.minimization = MinimizationAlg::Bfgs;
    settings.line_search = LineSearchAlg::Backtracking;

    // Global minimum
    let x = vec![0., -1.];
    let result = bfgs::get_minimum(&test_functions::booth, &x, &settings);
    assert!(result.is_ok(), "Result not found: {}", result.err().unwrap());
    let cmp = vec![1., 3.];
    float_eq::assert_float_eq!(result.unwrap().x, cmp, rmax_all <= 0.01);

    // Global minimum
    let x = vec![5., -5.];
    let result = bfgs::get_minimum(&test_functions::booth, &x, &settings);
    assert!(result.is_ok(), "Result not found: {}", result.err().unwrap());
    let cmp = vec![1., 3.];
    float_eq::assert_float_eq!(result.unwrap().x, cmp, rmax_all <= 0.01);

    // Global minimum
    let x = vec![-5., -5.];
    let result = bfgs::get_minimum(&test_functions::booth, &x, &settings);
    assert!(result.is_ok(), "Result not found: {}", result.err().unwrap());
    let cmp = vec![1., 3.];
    float_eq::assert_float_eq!(result.unwrap().x, cmp, rmax_all <= 0.01);

    // Global minimum
    let x = vec![5., 5.];
    let result = bfgs::get_minimum(&test_functions::booth, &x, &settings);
    assert!(result.is_ok(), "Result not found: {}", result.err().unwrap());
    let cmp = vec![1., 3.];
    float_eq::assert_float_eq!(result.unwrap().x, cmp, rmax_all <= 0.01);
}

#[test]
#[ignore]
fn test_matyas_function() {
    use bfgs;

    let mut settings: bfgs::settings::Settings = Default::default();
    settings.minimization = MinimizationAlg::Bfgs;
    settings.line_search = LineSearchAlg::Backtracking;

    // Global minimum
    let x = vec![0., -1.];
    let result = bfgs::get_minimum(&test_functions::matyas, &x, &settings);
    assert!(result.is_ok(), "Result not found: {}", result.err().unwrap());
    let cmp = vec![0., 0.];
    test_utils::check_result(&result.unwrap().x, &cmp);
}
