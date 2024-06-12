use criterion::{black_box, criterion_group, criterion_main, Criterion};
use minikalman::matrix::{MatrixData, SquareMatrix};

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("invert_lower (ref)", |bencher| {
        let a_buf = [1.0f32, 0.0, 0.0, -2.0, 1.0, 0.0, 3.5, -2.5, 1.0];
        let a = MatrixData::new_ref::<3, 3, f32>(&a_buf);

        let mut inv_buf = [0f32; 3 * 3];
        let mut inv = MatrixData::new_mut::<3, 3, f32>(&mut inv_buf);

        bencher.iter(|| a.invert_l_cholesky(black_box(&mut inv)))
    });

    c.bench_function("invert_lower (owned)", |bencher| {
        let a_buf = [1.0f32, 0.0, 0.0, -2.0, 1.0, 0.0, 3.5, -2.5, 1.0];
        let a = MatrixData::new_array::<3, 3, 9, f32>(a_buf);

        let inv_buf = [0f32; 3 * 3];
        let mut inv = MatrixData::new_array::<3, 3, 9, f32>(inv_buf);

        bencher.iter(|| a.invert_l_cholesky(black_box(&mut inv)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
