use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use bfgs::settings::{LineSearchAlg, MinimizationAlg};

// Global minimum: [0., 0.., ...]
mod bench_functions;

use bench_functions::sphere;

fn bench_bfgs_backtracking(c: &mut Criterion) {
    let dims = vec![2, 6, 20, 60, 200];

    let mut settings: bfgs::settings::Settings = Default::default();
    settings.minimization = MinimizationAlg::Bfgs;
    settings.line_search = LineSearchAlg::Backtracking;

    let mut group = c.benchmark_group("bfgs_backtracking");

    for d in dims {
        group.bench_with_input(BenchmarkId::from_parameter(d), &d, |b, &d| {
            b.iter(|| bfgs::get_minimum(&sphere, black_box(&vec![1.7; d]), &settings));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_bfgs_backtracking);
criterion_main!(benches);
