use rand::{Rng, thread_rng};
use bfgs::settings::{LineSearchAlg, MinimizationAlg};

#[test]
fn test_sphere_function() {
    use bfgs;

    // Create settings with default parameters
    let mut settings: bfgs::settings::Settings = Default::default();

    settings.minimization = MinimizationAlg::Bfgs;
    settings.line_search = LineSearchAlg::Backtracking;
    settings.verbose = false;

    let ef = |x: &Vec<f64>, _g: &Vec<f64>, f: &mut f64, d: i32| {
        *f = 0.;
        for i in 0..d as usize {
            *f += x[i] * x[i];
        }
    };
    let gf = |x: &Vec<f64>, g: &mut Vec<f64>, _f: &f64, d: i32| {
        // Finite difference derivative
        let h = 1e-5;
        let mut x_for = x.clone();
        let mut x_bck = x.clone();
        for i in 0..d {
            let mut f1 = 0.;
            let mut f2 = 0.;
            x_for[i as usize] += h;
            x_bck[i as usize] -= h;
            ef(&x_bck, g, &mut f1, d);
            ef(&x_for, g, &mut f2, d);
            g[i as usize] = (f2 - f1) / (2. * h);
            x_for[i as usize] -= h;
            x_bck[i as usize] += h;
        }
    };

    // Fix for zero value
    let check_result = |x: Vec<f64>, cmp: Vec<f64>, d: usize| {
        for i in 0..d {
            assert!((x[i] - cmp[i]).abs() < 0.001);
        }
    };

    let dims = vec![2, 6, 20, 100, 1000];

    for d in dims {
        let mut x = vec![(); d].into_iter().map(|_| thread_rng().gen_range(-10.0..10.0)).collect();
        let result = bfgs::get_minimum(&ef, &gf, &mut x, d as i32, &settings);
        assert_ne!(result, None, "Result not found");
        let cmp = vec![0.; d];
        check_result(x, cmp, d);
    }
}

#[test]
fn test_rosenbrock_function() {
    use bfgs;

    // Create settings with default parameters
    let mut settings: bfgs::settings::Settings = Default::default();

    settings.minimization = MinimizationAlg::Bfgs;
    settings.line_search = LineSearchAlg::Backtracking;
    settings.verbose = false;

    let d: i32 = 2;

    let ef = |x: &Vec<f64>, _g: &Vec<f64>, f: &mut f64, d: i32| {
        *f = 0.;
        for i in 0..(d - 1) as usize {
            let t1 = x[i + 1] - x[i] * x[i];
            let t2 = 1. - x[i];
            *f += 100. * t1 * t1 + t2 * t2;
        }
    };
    let gf = |x: &Vec<f64>, g: &mut Vec<f64>, _f: &f64, d: i32| {
        // Finite difference derivative
        let h = 1e-5;
        let mut x_for = x.clone();
        let mut x_bck = x.clone();
        for i in 0..d {
            let mut f1 = 0.;
            let mut f2 = 0.;
            x_for[i as usize] += h;
            x_bck[i as usize] -= h;
            ef(&x_bck, g, &mut f1, d);
            ef(&x_for, g, &mut f2, d);
            g[i as usize] = (f2 - f1) / (2. * h);
            x_for[i as usize] -= h;
            x_bck[i as usize] += h;
        }
    };

    let mut x = vec![-1.2, 1.0];
    let result = bfgs::get_minimum(&ef, &gf, &mut x, d, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![1., 1.];
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.001);

    let d: i32 = 6;
    let mut x = vec![-0.2, 1., -1., -3.2, 1., -0.9];
    let result = bfgs::get_minimum(&ef, &gf, &mut x, d, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![1., 1., 1., 1., 1., 1.];
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.001);
}

#[test]
fn test_himmelblau_function() {
    use bfgs;

    // Create settings with default parameters
    let mut settings: bfgs::settings::Settings = Default::default();

    settings.minimization = MinimizationAlg::Bfgs;
    settings.line_search = LineSearchAlg::Backtracking;
    settings.verbose = false;

    let d: i32 = 2;

    let ef = |x: &Vec<f64>, _g: &Vec<f64>, f: &mut f64, _d: i32| {
        *f = (x[0] * x[0] + x[1] - 11.) * (x[0] * x[0] + x[1] - 11.) + (x[0] + x[1] * x[1] - 7.) * (x[0] + x[1] * x[1] - 7.);
    };
    let gf = |x: &Vec<f64>, g: &mut Vec<f64>, _f: &f64, d: i32| {
        // Finite difference derivative
        let h = 1e-5;
        let mut x_for = x.clone();
        let mut x_bck = x.clone();
        for i in 0..d {
            let mut f1 = 0.;
            let mut f2 = 0.;
            x_for[i as usize] += h;
            x_bck[i as usize] -= h;
            ef(&x_bck, g, &mut f1, d);
            ef(&x_for, g, &mut f2, d);
            g[i as usize] = (f2 - f1) / (2. * h);
            x_for[i as usize] -= h;
            x_bck[i as usize] += h;
        }
    };

    let mut x = vec![-1.2, 1.0];
    let result = bfgs::get_minimum(&ef, &gf, &mut x, d, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![-2.805118, 3.131312];
    // Other possibilities [3.,2.] [-3.779310, -3.283186] [3.584428, -1.848126]
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.001);
}

#[test]
fn test_three_hump_camel_function() {
    use bfgs;

    // Create settings with default parameters
    let mut settings: bfgs::settings::Settings = Default::default();

    settings.minimization = MinimizationAlg::Bfgs;
    settings.line_search = LineSearchAlg::Backtracking;
    settings.verbose = false;

    let d: i32 = 2;

    let ef = |x: &Vec<f64>, _g: &Vec<f64>, f: &mut f64, _d: i32| {
        *f = 2. * x[0] * x[0] - 1.05 * x[0] * x[0] * x[0] * x[0] + x[0] * x[0] * x[0] * x[0] * x[0] * x[0] / 6. + x[0] * x[1] + x[1] * x[1];
    };
    let gf = |x: &Vec<f64>, g: &mut Vec<f64>, _f: &f64, d: i32| {
        // Finite difference derivative
        let h = 1e-5;
        let mut x_for = x.clone();
        let mut x_bck = x.clone();
        for i in 0..d {
            let mut f1 = 0.;
            let mut f2 = 0.;
            x_for[i as usize] += h;
            x_bck[i as usize] -= h;
            ef(&x_bck, g, &mut f1, d);
            ef(&x_for, g, &mut f2, d);
            g[i as usize] = (f2 - f1) / (2. * h);
            x_for[i as usize] -= h;
            x_bck[i as usize] += h;
        }
    };

    let mut x = vec![-1.2, 1.0];
    let result = bfgs::get_minimum(&ef, &gf, &mut x, d, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![-1.74755, 0.873776];
    // Local minimum
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.01);

    let mut x = vec![-0.2, 0.5];
    let result = bfgs::get_minimum(&ef, &gf, &mut x, d, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![0., 0.]; // Global minimum
    // Fix for vector
    for i in 0..d {
        assert!(f64::abs(x[i as usize] - cmp[i as usize]) < 0.001);
    }
}

#[test]
fn test_mccoormic_function() {
    use bfgs;

    // Create settings with default parameters
    let mut settings: bfgs::settings::Settings = Default::default();

    settings.minimization = MinimizationAlg::Bfgs;
    settings.line_search = LineSearchAlg::Backtracking;
    settings.verbose = false;

    let d: i32 = 2;

    let ef = |x: &Vec<f64>, _g: &Vec<f64>, f: &mut f64, _d: i32| {
        *f = f64::sin(x[0] + x[1]) + (x[0] - x[1]) * (x[0] - x[1]) - 1.5 * x[0] + 2.5 * x[1] + 1.;
    };
    let gf = |x: &Vec<f64>, g: &mut Vec<f64>, _f: &f64, d: i32| {
        // Finite difference derivative
        let h = 1e-5;
        let mut x_for = x.clone();
        let mut x_bck = x.clone();
        for i in 0..d {
            let mut f1 = 0.;
            let mut f2 = 0.;
            x_for[i as usize] += h;
            x_bck[i as usize] -= h;
            ef(&x_bck, g, &mut f1, d);
            ef(&x_for, g, &mut f2, d);
            g[i as usize] = (f2 - f1) / (2. * h);
            x_for[i as usize] -= h;
            x_bck[i as usize] += h;
        }
    };

    let mut x = vec![-1.2, 1.0];
    let result = bfgs::get_minimum(&ef, &gf, &mut x, d, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![-0.54719, -1.54719];
    // Local minimum
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.01);
}

#[test]
fn test_styblinski_tang_function() {
    use bfgs;

    // Create settings with default parameters
    let mut settings: bfgs::settings::Settings = Default::default();

    settings.minimization = MinimizationAlg::Bfgs;
    settings.line_search = LineSearchAlg::Backtracking;
    settings.verbose = false;

    let ef = |x: &Vec<f64>, _g: &Vec<f64>, f: &mut f64, d: i32| {
        *f = 0.;
        for i in 0..(d as usize) {
            *f += x[i] * x[i] * x[i] * x[i] - 16. * x[i] * x[i] + 5. * x[i];
        }
        *f *= 0.5;
    };
    let gf = |x: &Vec<f64>, g: &mut Vec<f64>, _f: &f64, d: i32| {
        // Finite difference derivative
        let h = 1e-5;
        let mut x_for = x.clone();
        let mut x_bck = x.clone();
        for i in 0..d {
            let mut f1 = 0.;
            let mut f2 = 0.;
            x_for[i as usize] += h;
            x_bck[i as usize] -= h;
            ef(&x_bck, g, &mut f1, d);
            ef(&x_for, g, &mut f2, d);
            g[i as usize] = (f2 - f1) / (2. * h);
            x_for[i as usize] -= h;
            x_bck[i as usize] += h;
        }
    };

    let d: i32 = 2;
    let mut x = vec![-1.2, -1.0];
    let result = bfgs::get_minimum(&ef, &gf, &mut x, d, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![-2.903534, -2.903534];
    // Global minimum
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.01);

    let d: i32 = 2;
    let mut x = vec![-1.2, 1.0];
    let result = bfgs::get_minimum(&ef, &gf, &mut x, d, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![-2.903534, 2.7468];
    // Local minimum
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.01);

    let d: i32 = 5;
    let mut x = vec![-1.2, -1.0, -1.0, -1.0, -1.0];
    let result = bfgs::get_minimum(&ef, &gf, &mut x, d, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![-2.903534, -2.903534, -2.903534, -2.903534, -2.903534];
    // Global minimum
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.01);

    let d: i32 = 5;
    let mut x = vec![1.2, 1.0, 1.0, 1.0, 1.0];
    let result = bfgs::get_minimum(&ef, &gf, &mut x, d, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![2.7468, 2.7468, 2.7468, 2.7468, 2.7468];
    // Local minimum
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.01);
}

#[test]
fn test_beale_function() {
    use bfgs;

    // Create settings with default parameters
    let mut settings: bfgs::settings::Settings = Default::default();

    settings.minimization = MinimizationAlg::Bfgs;
    settings.line_search = LineSearchAlg::Backtracking;
    settings.verbose = false;

    let ef = |x: &Vec<f64>, _g: &Vec<f64>, f: &mut f64, _d: i32| {
        *f = (1.5 - x[0] + x[0] * x[1]) * (1.5 - x[0] + x[0] * x[1]) +
            (2.25 - x[0] + x[0] * x[1] * x[1]) * (2.25 - x[0] + x[0] * x[1] * x[1]) +
            (2.625 - x[0] + x[0] * x[1] * x[1] * x[1]) * (2.625 - x[0] + x[0] * x[1] * x[1] * x[1]);
    };
    let gf = |x: &Vec<f64>, g: &mut Vec<f64>, _f: &f64, d: i32| {
        // Finite difference derivative
        let h = 1e-5;
        let mut x_for = x.clone();
        let mut x_bck = x.clone();
        for i in 0..d {
            let mut f1 = 0.;
            let mut f2 = 0.;
            x_for[i as usize] += h;
            x_bck[i as usize] -= h;
            ef(&x_bck, g, &mut f1, d);
            ef(&x_for, g, &mut f2, d);
            g[i as usize] = (f2 - f1) / (2. * h);
            x_for[i as usize] -= h;
            x_bck[i as usize] += h;
        }
    };

    let d: i32 = 2;
    let mut x = vec![1.2, 1.0];
    let result = bfgs::get_minimum(&ef, &gf, &mut x, d, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![3., 0.5];
    // Global minimum
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.01);

    let d: i32 = 2;
    let mut x = vec![-1.2, -1.0];
    let result = bfgs::get_minimum(&ef, &gf, &mut x, d, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![3., 0.5];
    // Global minimum
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.01);

    // FIXME: not working
    /*let d: i32 = 2;
    let mut x = vec![-3.0, 3.0];
    let result = bfgs::get_minimum(&ef, &gf, &mut x, d, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![3., 0.5];
    // Global minimum
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.01);

    let d: i32 = 2;
    let mut x = vec![-3.0, -3.0];
    let result = bfgs::get_minimum(&ef, &gf, &mut x, d, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![3., 0.5];
    // Global minimum
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.01);*/
}

#[test]
fn test_goldstein_price_function() {
    use bfgs;

    // Create settings with default parameters
    let mut settings: bfgs::settings::Settings = Default::default();

    settings.minimization = MinimizationAlg::Bfgs;
    settings.line_search = LineSearchAlg::Backtracking;
    settings.verbose = false;

    let ef = |r: &Vec<f64>, _g: &Vec<f64>, f: &mut f64, _d: i32| {
        let x = r[0];
        let y = r[1];
        *f = (1. + (x + y + 1.) * (x + y + 1.) * (19. - 14. * x + 3. * x * x - 14. * y + 6. * x * y + 3. * y * y)) *
            (30. + (2. * x - 3. * y) * (2. * x - 3. * y) * (18. - 32. * x + 12. * x * x + 48. * y - 36. * x * y + 27. * y * y));
    };
    let gf = |x: &Vec<f64>, g: &mut Vec<f64>, _f: &f64, d: i32| {
        // Finite difference derivative
        let h = 1e-5;
        let mut x_for = x.clone();
        let mut x_bck = x.clone();
        for i in 0..d {
            let mut f1 = 0.;
            let mut f2 = 0.;
            x_for[i as usize] += h;
            x_bck[i as usize] -= h;
            ef(&x_bck, g, &mut f1, d);
            ef(&x_for, g, &mut f2, d);
            g[i as usize] = (f2 - f1) / (2. * h);
            x_for[i as usize] -= h;
            x_bck[i as usize] += h;
        }
    };

    // Fix for zero value
    let check_result = |x: Vec<f64>, cmp: Vec<f64>, d: usize| {
        for i in 0..d {
            assert!((x[i] - cmp[i]).abs() < 0.001);
        }
    };

    // Local minima: (1.2, 0.8) (1.8, 0.2) (-0.6, -0.4)
    // Global minimum: (0.0, -1.0)

    let d: i32 = 2;
    let mut x = vec![0., -1.];
    let result = bfgs::get_minimum(&ef, &gf, &mut x, d, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![0., -1.]; // Global minimum
    check_result(x, cmp, d as usize);

    let d: i32 = 2;
    let mut x = vec![0., 1.0];
    let result = bfgs::get_minimum(&ef, &gf, &mut x, d, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![0., -1.]; // Global minimum
    check_result(x, cmp, d as usize);
}

#[test]
fn test_booth_function() {
    use bfgs;

    // Create settings with default parameters
    let mut settings: bfgs::settings::Settings = Default::default();

    settings.minimization = MinimizationAlg::Bfgs;
    settings.line_search = LineSearchAlg::Backtracking;
    settings.verbose = false;

    let ef = |r: &Vec<f64>, _g: &Vec<f64>, f: &mut f64, _d: i32| {
        let x = r[0];
        let y = r[1];
        *f = (x + 2. * y - 7.) * (x + 2. * y - 7.) + (2. * x + y - 5.) * (2. * x + y - 5.);
    };
    let gf = |x: &Vec<f64>, g: &mut Vec<f64>, _f: &f64, d: i32| {
        // Finite difference derivative
        let h = 1e-5;
        let mut x_for = x.clone();
        let mut x_bck = x.clone();
        for i in 0..d {
            let mut f1 = 0.;
            let mut f2 = 0.;
            x_for[i as usize] += h;
            x_bck[i as usize] -= h;
            ef(&x_bck, g, &mut f1, d);
            ef(&x_for, g, &mut f2, d);
            g[i as usize] = (f2 - f1) / (2. * h);
            x_for[i as usize] -= h;
            x_bck[i as usize] += h;
        }
    };

    let d: i32 = 2;
    let mut x = vec![0., -1.];
    let result = bfgs::get_minimum(&ef, &gf, &mut x, d, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![1., 3.]; // Global minimum
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.01);

    let d: i32 = 2;
    let mut x = vec![5., -5.];
    let result = bfgs::get_minimum(&ef, &gf, &mut x, d, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![1., 3.]; // Global minimum
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.01);

    let d: i32 = 2;
    let mut x = vec![-5., -5.];
    let result = bfgs::get_minimum(&ef, &gf, &mut x, d, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![1., 3.]; // Global minimum
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.01);

    let d: i32 = 2;
    let mut x = vec![5., 5.];
    let result = bfgs::get_minimum(&ef, &gf, &mut x, d, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![1., 3.]; // Global minimum
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.01);
}

/*#[test]
fn test_bulkin_function() {
    use bfgs;

    // Create settings with default parameters
    let mut settings: bfgs::settings::Settings = Default::default();

    settings.minimization = MinimizationAlg::Bfgs;
    settings.minimization = MinimizationAlg::Bfgs;
    settings.verbose = false;

    let ef = |r: &Vec<f64>, _g: &Vec<f64>, f: &mut f64, _d: i32| {
        let x = r[0];
        let y = r[1];
        *f = 100. * (y - 0.01*x*x).abs().sqrt() + 0.01 * (x + 10.).abs();
    };
    let gf = |x: &Vec<f64>, g: &mut Vec<f64>, _f: &f64, d: i32| {
        // Finite difference derivative
        let h = 1e-5;
        let mut x_for = x.clone();
        let mut x_bck = x.clone();
        for i in 0..d {
            let mut f1 = 0.;
            let mut f2 = 0.;
            x_for[i as usize] += h;
            x_bck[i as usize] -= h;
            ef(&x_bck, g, &mut f1, d);
            ef(&x_for, g, &mut f2, d);
            g[i as usize] = (f2 - f1) / (2. * h);
            x_for[i as usize] -= h;
            x_bck[i as usize] += h;
        }
    };

    let d: i32 = 2;
    let mut x = vec![-10., 1.];
    let result = bfgs::get_minimum(&ef, &gf, &mut x, d, &settings);
    assert_ne!(result, None, "Result not found");
    let cmp = vec![-10., 1.]; // Global minimum
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.01);
}*/
