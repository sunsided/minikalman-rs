//! The benchmark problems, defined once as row-major `f64` arrays.

/// Linear measurement `z = H·x` or the nonlinear radar `z = [range, bearing]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Model {
    /// `z = H·x`.
    Linear,
    /// `z = [√(px² + py²), atan2(py, px)]`, the extended Kalman filter case.
    Radar,
}

/// Number of warm-up steps before the snapshot the benchmarks restore to.
pub const WARMUP: usize = 64;
/// Number of measurements after the snapshot; benchmarks cycle through them.
pub const LAP: usize = 256;

/// One estimation problem: model matrices, the start, and the measurement sequence.
#[derive(Clone, Debug)]
pub struct Problem {
    /// Short name used in benchmark ids.
    pub name: &'static str,
    /// States.
    pub n: usize,
    /// Measurements.
    pub m: usize,
    /// Time step.
    pub dt: f64,
    /// Measurement model.
    pub model: Model,
    /// State transition, `n×n`.
    pub F: Vec<f64>,
    /// Process noise, `n×n`, positive definite.
    pub Q: Vec<f64>,
    /// Observation matrix, `m×n` (unused for the radar).
    pub H: Vec<f64>,
    /// Measurement noise, `m×m`.
    pub R: Vec<f64>,
    /// Initial state.
    pub x0: Vec<f64>,
    /// Initial covariance, `n×n`.
    pub P0: Vec<f64>,
    /// `WARMUP + LAP` measurements.
    pub zs: Vec<Vec<f64>>,
}

/// A deterministic standard-normal sequence (xorshift64* and Box–Muller).
struct Gauss(u64);

impl Gauss {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let v = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((v >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn next(&mut self) -> f64 {
        let (u1, u2) = (self.uniform(), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
    }
}

/// `F` and `Q` of a `d`-dimensional constant-velocity model, state `[p₁…p_d, v₁…v_d]`,
/// with the continuous white-noise-acceleration `Q` of intensity `q` (positive definite).
fn constant_velocity(d: usize, dt: f64, q: f64) -> (Vec<f64>, Vec<f64>) {
    let n = 2 * d;
    let mut F = vec![0.0; n * n];
    let mut Q = vec![0.0; n * n];
    for i in 0..d {
        F[i * n + i] = 1.0;
        F[i * n + i + d] = dt;
        F[(i + d) * n + i + d] = 1.0;
        Q[i * n + i] = q * dt.powi(3) / 3.0;
        Q[i * n + i + d] = q * dt.powi(2) / 2.0;
        Q[(i + d) * n + i] = q * dt.powi(2) / 2.0;
        Q[(i + d) * n + i + d] = q * dt;
    }
    (F, Q)
}

/// The truth trajectory: constant velocity with a slow sinusoidal acceleration.
fn truth(d: usize, dt: f64, steps: usize, p0: &[f64], v0: &[f64]) -> Vec<Vec<f64>> {
    let (mut p, mut v) = (p0.to_vec(), v0.to_vec());
    let mut out = Vec::with_capacity(steps);
    for k in 0..steps {
        for i in 0..d {
            let a = 0.2 * ((k as f64) * dt * (0.7 + 0.3 * i as f64)).sin();
            p[i] += v[i] * dt + 0.5 * a * dt * dt;
            v[i] += a * dt;
        }
        out.push(p.clone());
    }
    out
}

impl Problem {
    /// A `2·d`-state constant-velocity tracker that measures the `d` positions.
    ///
    /// `d = 1, 2, 3` gives the 2-state/1-measurement, 4-state/2-measurement and
    /// 6-state/3-measurement problems.
    pub fn linear(d: usize) -> Self {
        let (n, m, dt) = (2 * d, d, 0.1);
        let (F, Q) = constant_velocity(d, dt, 0.5);
        let mut H = vec![0.0; m * n];
        let mut R = vec![0.0; m * m];
        for i in 0..m {
            H[i * n + i] = 1.0;
            R[i * m + i] = 0.25 + 0.05 * i as f64;
            for j in 0..i {
                // Mildly correlated noise, so a full R is exercised.
                R[i * m + j] = 0.05;
                R[j * m + i] = 0.05;
            }
        }
        let p_true: Vec<f64> = (0..d).map(|i| 1.0 + i as f64).collect();
        let v_true: Vec<f64> = (0..d).map(|i| 0.5 - 0.2 * i as f64).collect();
        let traj = truth(d, dt, WARMUP + LAP, &p_true, &v_true);
        let mut g = Gauss(0x9E37_79B9_7F4A_7C15 ^ d as u64);
        let zs = traj
            .iter()
            .map(|p| {
                (0..m)
                    .map(|i| p[i] + R[i * m + i].sqrt() * g.next())
                    .collect()
            })
            .collect();
        let mut x0 = vec![0.0; n];
        let mut P0 = vec![0.0; n * n];
        for i in 0..d {
            x0[i] = p_true[i] + 0.5;
            x0[i + d] = 0.0;
        }
        for i in 0..n {
            P0[i * n + i] = 4.0;
        }
        let name = match d {
            1 => "kf_2x1",
            2 => "kf_4x2",
            3 => "kf_6x3",
            _ => "kf",
        };
        Self {
            name,
            n,
            m,
            dt,
            model: Model::Linear,
            F,
            Q,
            H,
            R,
            x0,
            P0,
            zs,
        }
    }

    /// The 4-state, 2-measurement range–bearing radar, for the extended Kalman filters.
    ///
    /// The target stays in the first quadrant (bearing ≈ 0.4–0.5 rad), far from the ±π
    /// cut, so no crate needs to wrap the bearing innovation.
    pub fn radar() -> Self {
        let (d, n, m, dt) = (2, 4, 2, 0.1);
        let (F, Q) = constant_velocity(d, dt, 0.5);
        let R = vec![0.1 * 0.1, 0.0, 0.0, 0.01 * 0.01];
        let p_true = [10.0, 5.0];
        let v_true = [1.0, 0.5];
        let traj = truth(d, dt, WARMUP + LAP, &p_true, &v_true);
        let mut g = Gauss(0xD1B5_4A32_D192_ED03);
        let zs = traj
            .iter()
            .map(|p| {
                let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
                let b = p[1].atan2(p[0]);
                vec![r + 0.1 * g.next(), b + 0.01 * g.next()]
            })
            .collect();
        let x0 = vec![10.5, 4.5, 0.0, 0.0];
        let mut P0 = vec![0.0; n * n];
        for i in 0..n {
            P0[i * n + i] = 1.0;
        }
        Self {
            name: "ekf_radar_4x2",
            n,
            m,
            dt,
            model: Model::Radar,
            F,
            Q,
            H: vec![0.0; m * n],
            R,
            x0,
            P0,
            zs,
        }
    }
}
