//! A plain `f64` textbook filter, the yardstick for the agreement check.
//!
//! Deliberately independent of every crate under test: dense loops, Gauss–Jordan inverse of
//! `S`, and the Joseph form `P = (I−KH)·P·(I−KH)ᵀ + K·R·Kᵀ`.

use crate::{Filter, Model, Problem, radar_h, radar_jacobian};

/// `C = A·B` for row-major `A` (r×k) and `B` (k×c).
pub fn mul(a: &[f64], b: &[f64], r: usize, k: usize, c: usize) -> Vec<f64> {
    let mut out = vec![0.0; r * c];
    for i in 0..r {
        for j in 0..c {
            out[i * c + j] = (0..k).map(|l| a[i * k + l] * b[l * c + j]).sum();
        }
    }
    out
}

/// Transpose of a row-major `r×c` matrix.
pub fn tr(a: &[f64], r: usize, c: usize) -> Vec<f64> {
    let mut out = vec![0.0; r * c];
    for i in 0..r {
        for j in 0..c {
            out[j * r + i] = a[i * c + j];
        }
    }
    out
}

/// Inverse of a small `n×n` matrix by Gauss–Jordan with partial pivoting.
pub fn inv(a: &[f64], n: usize) -> Vec<f64> {
    let mut a = a.to_vec();
    let mut b = vec![0.0; n * n];
    for i in 0..n {
        b[i * n + i] = 1.0;
    }
    for c in 0..n {
        let p = (c..n)
            .max_by(|&x, &y| a[x * n + c].abs().total_cmp(&a[y * n + c].abs()))
            .unwrap();
        for k in 0..n {
            a.swap(c * n + k, p * n + k);
            b.swap(c * n + k, p * n + k);
        }
        let d = a[c * n + c];
        assert!(d.abs() > 1e-300, "singular S in the reference");
        for k in 0..n {
            a[c * n + k] /= d;
            b[c * n + k] /= d;
        }
        for r in 0..n {
            if r != c {
                let f = a[r * n + c];
                for k in 0..n {
                    a[r * n + k] -= f * a[c * n + k];
                    b[r * n + k] -= f * b[c * n + k];
                }
            }
        }
    }
    b
}

/// The reference filter.
pub struct Reference {
    p: Problem,
    /// State.
    pub x: Vec<f64>,
    /// Covariance, row-major.
    pub P: Vec<f64>,
    /// Last gain, `n×m`.
    pub K: Vec<f64>,
}

impl Reference {
    /// Starts at the problem's `x0`, `P0`.
    pub fn new(p: &Problem) -> Self {
        Self {
            x: p.x0.clone(),
            P: p.P0.clone(),
            K: vec![0.0; p.n * p.m],
            p: p.clone(),
        }
    }
}

impl Filter for Reference {
    type T = f64;

    fn predict(&mut self) {
        let n = self.p.n;
        self.x = mul(&self.p.F, &self.x, n, n, 1);
        let fp = mul(&self.p.F, &self.P, n, n, n);
        let fpf = mul(&fp, &tr(&self.p.F, n, n), n, n, n);
        self.P = fpf.iter().zip(&self.p.Q).map(|(a, b)| a + b).collect();
    }

    fn correct(&mut self, z: &[f64]) {
        let (n, m) = (self.p.n, self.p.m);
        let (H, hx) = match self.p.model {
            Model::Linear => (self.p.H.clone(), mul(&self.p.H, &self.x, m, n, 1)),
            Model::Radar => (radar_jacobian(&self.x).to_vec(), radar_h(&self.x).to_vec()),
        };
        let y: Vec<f64> = z.iter().zip(&hx).map(|(a, b)| a - b).collect();
        let Ht = tr(&H, m, n);
        let pht = mul(&self.P, &Ht, n, n, m);
        let S: Vec<f64> = mul(&H, &pht, m, n, m)
            .iter()
            .zip(&self.p.R)
            .map(|(a, b)| a + b)
            .collect();
        let K = mul(&pht, &inv(&S, m), n, m, m);
        let ky = mul(&K, &y, n, m, 1);
        for i in 0..n {
            self.x[i] += ky[i];
        }
        let mut ikh = mul(&K, &H, n, m, n);
        for v in ikh.iter_mut() {
            *v = -*v;
        }
        for i in 0..n {
            ikh[i * n + i] += 1.0;
        }
        let a = mul(&mul(&ikh, &self.P, n, n, n), &tr(&ikh, n, n), n, n, n);
        let krk = mul(&mul(&K, &self.p.R, n, m, m), &tr(&K, n, m), n, m, n);
        self.P = a.iter().zip(&krk).map(|(a, b)| a + b).collect();
        self.K = K;
    }

    fn state(&self) -> Vec<f64> {
        self.x.clone()
    }

    fn covariance(&self) -> Vec<f64> {
        self.P.clone()
    }

    fn restore(&mut self, x: &[f64], p: &[f64]) {
        self.x = x.to_vec();
        self.P = p.to_vec();
    }
}

/// A fixed-gain observer recursion `x ← F·x; x ← x + K·(z − H·x)`, the yardstick for the
/// Luenberger observer.
pub struct FixedGainReference {
    /// The problem.
    pub p: Problem,
    /// State.
    pub x: Vec<f64>,
    /// Gain, `n×m`.
    pub K: Vec<f64>,
}

impl FixedGainReference {
    /// One predict + correct step.
    pub fn step(&mut self, z: &[f64]) {
        let (n, m) = (self.p.n, self.p.m);
        self.x = mul(&self.p.F, &self.x, n, n, 1);
        let hx = mul(&self.p.H, &self.x, m, n, 1);
        let y: Vec<f64> = z.iter().zip(&hx).map(|(a, b)| a - b).collect();
        let ky = mul(&self.K, &y, n, m, 1);
        for i in 0..n {
            self.x[i] += ky[i];
        }
    }
}
