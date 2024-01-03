use bfgs::settings::MinimizationAlg;

#[test]
fn test_simple_function() {
    use bfgs;

    // Create settings with default parameters
    let mut settings: bfgs::settings::Settings = Default::default();

    settings.minimization = MinimizationAlg::Bfgs;
    settings.verbose = false;
    settings.estimate_a = false;

    let d: i32 = 2;
    let mut x = vec![-1.2, 1.0];

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

    let _result = bfgs::get_minimum(&ef, &gf, &mut x, d, &settings);
    let cmp = vec![1., 1.];
    float_eq::assert_float_eq!(x, cmp, rmax_all <= 0.1);
}
