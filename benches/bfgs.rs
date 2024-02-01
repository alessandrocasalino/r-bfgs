use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use bfgs::settings::MinimizationAlg;

mod bench_functions;

use bench_functions::sphere;

fn bench_bfgs(c: &mut Criterion) {
    let dims = vec![2, 6, 20, 60, 200];

    let mut settings: bfgs::settings::Settings = Default::default();
    settings.minimization = MinimizationAlg::Bfgs;

    let mut group = c.benchmark_group("bfgs");

    for d in dims {
        group.bench_with_input(BenchmarkId::from_parameter(d), &d, |b, &d| {
            b.iter(|| bfgs::get_minimum(&sphere, black_box(&vec![1.7; d]), &settings));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_bfgs);
criterion_main!(benches);
