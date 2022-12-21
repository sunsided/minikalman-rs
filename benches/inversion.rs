use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kalman::Matrix;

fn criterion_benchmark(c: &mut Criterion) {
    let mut a_buf = [1.0f32, 0.0, 0.0, -2.0, 1.0, 0.0, 3.5, -2.5, 1.0];
    let a = Matrix::new(3, 3, &mut a_buf);

    let mut inv_buf = [0f32; 3 * 3];
    let mut inv = Matrix::new(3, 3, &mut inv_buf);

    c.bench_function("invert_lower", |b| {
        b.iter(|| a.invert_lower(black_box(&mut inv)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
