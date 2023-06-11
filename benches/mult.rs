use criterion::{black_box, criterion_group, criterion_main, Criterion};
use minikalman::Matrix;
use rand::Rng;

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("mult 2x3 x 3x2", |bencher| {
        let mut a_buf = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = Matrix::new(2, 3, &mut a_buf);

        let mut b_buf = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = Matrix::new(3, 2, &mut b_buf);

        let mut c_buf = [0f32; 3 * 3];
        let mut c = Matrix::new(3, 3, &mut c_buf);

        bencher.iter(|| Matrix::mult(black_box(&a), black_box(&b), black_box(&mut c)))
    });

    c.bench_function("mult_buffered 2x3 x 3x2", |bencher| {
        let mut a_buf = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = Matrix::new(2, 3, &mut a_buf);

        let mut b_buf = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = Matrix::new(3, 2, &mut b_buf);

        let mut c_buf = [0f32; 3 * 3];
        let mut c = Matrix::new(3, 3, &mut c_buf);

        let mut aux = [0f32; 3 * 1];

        bencher.iter(|| {
            Matrix::mult_buffered(
                black_box(&a),
                black_box(&b),
                black_box(&mut c),
                black_box(&mut aux),
            )
        })
    });

    c.bench_function("mult 3x3 x 3x2", |bencher| {
        let mut a_buf = [1.0f32, 0.0, 0.0, -2.0, 1.0, 0.0, 3.5, -2.5, 1.0];
        let a = Matrix::new(3, 3, &mut a_buf);

        let mut b_buf = [10.0, 11.0, 20.0, 21.0, 30.0, 31.0];
        let b = Matrix::new(3, 2, &mut b_buf);

        let mut c_buf = [0f32; 3 * 3];
        let mut c = Matrix::new(3, 3, &mut c_buf);

        bencher.iter(|| Matrix::mult(black_box(&a), black_box(&b), black_box(&mut c)))
    });

    c.bench_function("mult_buffered 3x3 x 3x2", |bencher| {
        let mut a_buf = [1.0f32, 0.0, 0.0, -2.0, 1.0, 0.0, 3.5, -2.5, 1.0];
        let a = Matrix::new(3, 3, &mut a_buf);

        let mut b_buf = [10.0, 11.0, 20.0, 21.0, 30.0, 31.0];
        let b = Matrix::new(3, 2, &mut b_buf);

        let mut c_buf = [0f32; 3 * 3];
        let mut c = Matrix::new(3, 3, &mut c_buf);

        let mut aux = [0f32; 3 * 1];

        bencher.iter(|| {
            Matrix::mult_buffered(
                black_box(&a),
                black_box(&b),
                black_box(&mut c),
                black_box(&mut aux),
            )
        })
    });

    c.bench_function("mult 3x16 x 16x3", |bencher| {
        let mut a_buf: Vec<f32> = rand::thread_rng()
            .sample_iter(rand::distributions::Standard)
            .take(3 * 16)
            .collect();
        let a = Matrix::new(3, 16, &mut a_buf);

        let mut b_buf: Vec<f32> = rand::thread_rng()
            .sample_iter(rand::distributions::Standard)
            .take(16 * 3)
            .collect();
        let b = Matrix::new(16, 3, &mut b_buf);

        let mut c_buf = [0f32; 3 * 3];
        let mut c = Matrix::new(3, 3, &mut c_buf);

        bencher.iter(|| Matrix::mult(black_box(&a), black_box(&b), black_box(&mut c)))
    });

    c.bench_function("mult_buffered 3x16 x 16x3", |bencher| {
        let mut a_buf: Vec<f32> = rand::thread_rng()
            .sample_iter(rand::distributions::Standard)
            .take(3 * 16)
            .collect();
        let a = Matrix::new(3, 16, &mut a_buf);

        let mut b_buf: Vec<f32> = rand::thread_rng()
            .sample_iter(rand::distributions::Standard)
            .take(16 * 3)
            .collect();
        let b = Matrix::new(16, 3, &mut b_buf);

        let mut c_buf = [0f32; 3 * 3];
        let mut c = Matrix::new(3, 3, &mut c_buf);

        let mut aux = [0f32; 16 * 1];

        bencher.iter(|| {
            Matrix::mult_buffered(
                black_box(&a),
                black_box(&b),
                black_box(&mut c),
                black_box(&mut aux),
            )
        })
    });

    c.bench_function("mult_transb 2x3 x 2x3'", |bencher| {
        let mut a_buf = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = Matrix::new(2, 3, &mut a_buf);

        let mut b_buf = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let b = Matrix::new(2, 3, &mut b_buf);

        let mut c_buf = [0f32; 3 * 3];
        let mut c = Matrix::new(3, 3, &mut c_buf);

        bencher.iter(|| Matrix::mult_transb(black_box(&a), black_box(&b), black_box(&mut c)))
    });

    c.bench_function("multscale_transb 2x3 x 2x3' * 1.0", |bencher| {
        let mut a_buf = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = Matrix::new(2, 3, &mut a_buf);

        let mut b_buf = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let b = Matrix::new(2, 3, &mut b_buf);

        let mut c_buf = [0f32; 3 * 3];
        let mut c = Matrix::new(3, 3, &mut c_buf);

        bencher.iter(|| {
            Matrix::multscale_transb(
                black_box(&a),
                black_box(&b),
                black_box(1.0),
                black_box(&mut c),
            )
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
