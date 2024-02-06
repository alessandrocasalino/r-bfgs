mod test_functions;
mod test_utils;

#[test]
fn test_history_bfgs () {
    use bfgs;
    use bfgs::settings::MinimizationAlg;
    use bfgs::settings::LineSearchAlg;

    let mut settings: bfgs::settings::Settings = Default::default();
    settings.minimization = MinimizationAlg::Bfgs;
    settings.line_search = LineSearchAlg::Simple;
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
    use bfgs::settings::MinimizationAlg;
    use bfgs::settings::LineSearchAlg;

    let mut settings: bfgs::settings::Settings = Default::default();
    settings.minimization = MinimizationAlg::Lbfgs;
    settings.line_search = LineSearchAlg::Simple;
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
    use bfgs::settings::MinimizationAlg;
    use bfgs::settings::LineSearchAlg;

    let mut settings: bfgs::settings::Settings = Default::default();
    settings.minimization = MinimizationAlg::Bfgs;
    settings.line_search = LineSearchAlg::Backtracking;
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
    use bfgs::settings::MinimizationAlg;
    use bfgs::settings::LineSearchAlg;

    let mut settings: bfgs::settings::Settings = Default::default();
    settings.minimization = MinimizationAlg::Lbfgs;
    settings.line_search = LineSearchAlg::Backtracking;
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
