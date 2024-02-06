use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use bfgs::settings::MinimizationAlg;

mod bench_functions;

use bench_functions::{sphere, booth};

fn bench_gradient_descent_sphere(c: &mut Criterion) {
    let dims = vec![2, 6, 20, 60, 200];

    let mut settings: bfgs::settings::Settings = Default::default();
    settings.minimization = MinimizationAlg::GradientDescent;

    let mut group = c.benchmark_group("gradient_descent");

    for d in dims {
        group.bench_with_input(BenchmarkId::from_parameter(d), &d, |b, &d| {
            b.iter(|| bfgs::get_minimum(&sphere, black_box(&vec![1.7; d]), &settings));
        });
    }

    group.finish();
}

fn bench_gradient_descent_booth(c: &mut Criterion) {
    let dims = vec![2];

    let mut settings: bfgs::settings::Settings = Default::default();
    settings.minimization = MinimizationAlg::GradientDescent;

    let mut group = c.benchmark_group("gradient_descent");

    for d in dims {
        group.bench_with_input(BenchmarkId::from_parameter(d), &d, |b, &d| {
            b.iter(||
                {
                    let result = bfgs::get_minimum(&booth, black_box(&vec![1.7; d]), &settings);
                    let cmp = vec![1., 3.];
                    float_eq::assert_float_eq!(result.unwrap().x, cmp, rmax_all <= 0.01);
                }
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_gradient_descent_sphere, bench_gradient_descent_booth);
criterion_main!(benches);
