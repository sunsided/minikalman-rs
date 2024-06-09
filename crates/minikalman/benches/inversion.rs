use criterion::{black_box, criterion_group, criterion_main, Criterion};
use minikalman::MatrixData;

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("invert_lower", |bencher| {
        let mut a_buf = [1.0f32, 0.0, 0.0, -2.0, 1.0, 0.0, 3.5, -2.5, 1.0];
        let a = MatrixData::<3, 3>::new(&mut a_buf);

        let mut inv_buf = [0f32; 3 * 3];
        let mut inv = MatrixData::<3, 3>::new(&mut inv_buf);

        bencher.iter(|| a.invert_l_cholesky(black_box(&mut inv)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
