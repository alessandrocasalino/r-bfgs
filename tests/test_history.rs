mod test_functions;
mod test_utils;

#[test]
fn test_history_bfgs () {
    use bfgs;
    use bfgs::settings::MinimizationAlgorithm;
    use bfgs::settings::LineSearchAlgorithm;

    let mut settings: bfgs::settings::Settings = Default::default();
    settings.minimization = MinimizationAlgorithm::Bfgs;
    settings.line_search = LineSearchAlgorithm::Simple;
    settings.save_history = true;

    // Global minimum
    let x = vec![0., -10.];
    let result = bfgs::get_minimum(&test_functions::booth, &x, &settings);
    assert!(result.is_ok(), "Result not found: {}", result.err().unwrap());
    let cmp = vec![1., 3.];
    test_utils::check_result(&result.as_ref().unwrap().x, &cmp);

    let mut f = result.as_ref().unwrap().history.first().unwrap().f;
    for (i, point) in result.as_ref().unwrap().history.iter().enumerate() {
        if i > 0 {
            assert!(f > point.f);
            f = point.f;
        }
    }
}

#[test]
fn test_history_lbfgs () {
    use bfgs;
    use bfgs::settings::MinimizationAlgorithm;
    use bfgs::settings::LineSearchAlgorithm;

    let mut settings: bfgs::settings::Settings = Default::default();
    settings.minimization = MinimizationAlgorithm::Lbfgs;
    settings.line_search = LineSearchAlgorithm::Simple;
    settings.save_history = true;

    // Global minimum
    let x = vec![0., -1.];
    let result = bfgs::get_minimum(&test_functions::booth, &x, &settings);
    assert!(result.is_ok(), "Result not found: {}", result.err().unwrap());
    let cmp = vec![1., 3.];
    test_utils::check_result(&result.as_ref().unwrap().x, &cmp);

    let mut f = result.as_ref().unwrap().history.first().unwrap().f;
    for (i, point) in result.as_ref().unwrap().history.iter().enumerate() {
        if i > 0 {
            assert!(f > point.f);
            f = point.f;
        }
    }
}

#[test]
fn test_history_bfgs_backtracking () {
    use bfgs;
    use bfgs::settings::MinimizationAlgorithm;
    use bfgs::settings::LineSearchAlgorithm;

    let mut settings: bfgs::settings::Settings = Default::default();
    settings.minimization = MinimizationAlgorithm::Bfgs;
    settings.line_search = LineSearchAlgorithm::Backtracking;
    settings.save_history = true;

    // Global minimum
    let x = vec![0., -1.];
    let result = bfgs::get_minimum(&test_functions::booth, &x, &settings);
    assert!(result.is_ok(), "Result not found: {}", result.err().unwrap());
    let cmp = vec![1., 3.];
    test_utils::check_result(&result.as_ref().unwrap().x, &cmp);

    let mut f = result.as_ref().unwrap().history.first().unwrap().f;
    for (i, point) in result.as_ref().unwrap().history.iter().enumerate() {
        if i > 0 {
            assert!(f > point.f);
            f = point.f;
        }
    }
}

#[test]
fn test_history_lbfgs_backtracking () {
    use bfgs;
    use bfgs::settings::MinimizationAlgorithm;
    use bfgs::settings::LineSearchAlgorithm;

    let mut settings: bfgs::settings::Settings = Default::default();
    settings.minimization = MinimizationAlgorithm::Lbfgs;
    settings.line_search = LineSearchAlgorithm::Backtracking;
    settings.save_history = true;

    // Global minimum
    let x = vec![0., -1.];
    let result = bfgs::get_minimum(&test_functions::booth, &x, &settings);
    assert!(result.is_ok(), "Result not found: {}", result.err().unwrap());
    let cmp = vec![1., 3.];
    test_utils::check_result(&result.as_ref().unwrap().x, &cmp);

    let mut f = result.as_ref().unwrap().history.first().unwrap().f;
    for (i, point) in result.as_ref().unwrap().history.iter().enumerate() {
        if i > 0 {
            assert!(f > point.f);
            f = point.f;
        }
    }
}
