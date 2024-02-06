use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use bfgs::settings::{LineSearchAlg, MinimizationAlg};

// Global minimum: [0., 0.., ...]
mod bench_functions;

use bench_functions::sphere;
use crate::bench_functions::booth;

fn bench_bfgs_backtracking_sphere(c: &mut Criterion) {
    let dims = vec![2, 6, 20, 60, 200];

    let mut settings: bfgs::settings::Settings = Default::default();
    settings.minimization = MinimizationAlg::Bfgs;
    settings.line_search = LineSearchAlg::Backtracking;

    let mut group = c.benchmark_group("bfgs_backtracking_sphere");

    for d in dims {
        group.bench_with_input(BenchmarkId::from_parameter(d), &d, |b, &d| {
            b.iter(|| bfgs::get_minimum(&sphere, black_box(&vec![1.7; d]), &settings));
        });
    }

    group.finish();
}

fn bench_bfgs_backtracking_booth(c: &mut Criterion) {
    let dims = vec![2];

    let mut settings: bfgs::settings::Settings = Default::default();
    settings.minimization = MinimizationAlg::Bfgs;
    settings.line_search = LineSearchAlg::Backtracking;

    let mut group = c.benchmark_group("bfgs_backtracking_booth");

    for d in dims {
        group.bench_with_input(BenchmarkId::from_parameter(d), &d, |b, &d| {
            b.iter(|| {
                let result = bfgs::get_minimum(&booth, black_box(&vec![1.7; d]), &settings);
                let cmp = vec![1., 3.];
                float_eq::assert_float_eq!(result.unwrap().x, cmp, rmax_all <= 0.01);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_bfgs_backtracking_sphere, bench_bfgs_backtracking_booth);
criterion_main!(benches);
