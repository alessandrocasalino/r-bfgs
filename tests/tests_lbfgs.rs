use bfgs::settings::MinimizationAlg;

#[test]
fn test_simple_function() {
    use bfgs;

    // Create settings with default parameters
    let mut settings: bfgs::settings::Settings = Default::default();

    settings.minimization = MinimizationAlg::Lbfgs;
    settings.verbose = false;
    settings.estimate_a = false;

    let d: i32 = 2;

    let ef = |x: &Vec<f64>, _g: &Vec<f64>, f: &mut f64, d: i32| {
        *f = 0.;
        for i in 0..d - 1 {
            *f += 100. * (x[(i + 1) as usize] - x[i as usize] * x[i as usize]) *
                (x[(i + 1) as usize] - x[i as usize] * x[i as usize]) + (x[i as usize] - 1.) * (x[i as usize] - 1.);
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
    let _result = bfgs::get_minimum(&ef, &gf, &mut x, d, &settings);
    let cmp = vec![1., 1.];
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.001);
}

#[test]
fn test_himmelblau_function() {
    use bfgs;

    // Create settings with default parameters
    let mut settings: bfgs::settings::Settings = Default::default();

    settings.minimization = MinimizationAlg::Lbfgs;
    settings.verbose = false;
    settings.estimate_a = false;

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
    let _result = bfgs::get_minimum(&ef, &gf, &mut x, d, &settings);
    let cmp = vec![-2.805118, 3.131312];
    // Other possibilities [3.,2.] [-3.779310, -3.283186] [3.584428, -1.848126]
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.001);
}

#[test]
fn test_three_hump_camel_function() {
    use bfgs;

    // Create settings with default parameters
    let mut settings: bfgs::settings::Settings = Default::default();

    settings.minimization = MinimizationAlg::Lbfgs;
    settings.verbose = false;
    settings.estimate_a = false;

    let d: i32 = 2;

    let ef = |x: &Vec<f64>, _g: &Vec<f64>, f: &mut f64, _d: i32| {
        *f = 2. * x[0] * x[0] - 1.05 * x[0] * x[0] * x[0] * x[0] + x[0] * x[0] * x[0] * x[0] * x[0] * x[0]/6. + x[0] * x[1] + x[1] * x[1];
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
    let _result = bfgs::get_minimum(&ef, &gf, &mut x, d, &settings);
    let cmp = vec![-1.74755, 0.873776]; // Local minimum
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.001);

    let mut x = vec![-0.2, 0.5];
    let _result = bfgs::get_minimum(&ef, &gf, &mut x, d, &settings);
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

    settings.minimization = MinimizationAlg::Lbfgs;
    settings.verbose = false;
    settings.estimate_a = false;

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
    let _result = bfgs::get_minimum(&ef, &gf, &mut x, d, &settings);
    let cmp = vec![-0.54719, -1.54719]; // Local minimum
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.01);
}

#[test]
fn test_styblinski_tang_function() {
    use bfgs;

    // Create settings with default parameters
    let mut settings: bfgs::settings::Settings = Default::default();

    settings.minimization = MinimizationAlg::Lbfgs;
    settings.verbose = false;
    settings.estimate_a = false;

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
    let _result = bfgs::get_minimum(&ef, &gf, &mut x, d, &settings);
    let cmp = vec![-2.903534, -2.903534]; // Global minimum
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.01);

    let d: i32 = 2;
    let mut x = vec![-1.2, 1.0];
    let _result = bfgs::get_minimum(&ef, &gf, &mut x, d, &settings);
    let cmp = vec![-2.903534, 2.7468]; // Local minimum
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.01);

    let d: i32 = 5;
    let mut x = vec![-1.2, -1.0, -1.0, -1.0, -1.0];
    let _result = bfgs::get_minimum(&ef, &gf, &mut x, d, &settings);
    let cmp = vec![-2.903534, -2.903534, -2.903534, -2.903534, -2.903534]; // Global minimum
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.01);

    let d: i32 = 5;
    let mut x = vec![1.2, 1.0, 1.0, 1.0, 1.0];
    let _result = bfgs::get_minimum(&ef, &gf, &mut x, d, &settings);
    let cmp = vec![2.7468, 2.7468, 2.7468, 2.7468, 2.7468]; // Local minimum
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.01);
}
