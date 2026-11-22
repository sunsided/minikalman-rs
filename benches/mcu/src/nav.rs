//! Error-state models of an attitude filter (6 states) and a GNSS/INS filter (12 states).
//!
//! `no_std`, `f32`, no allocation; trigonometry from `libm`, square roots from minikalman's
//! `square_root` (so the `unsafe` feature decides between `vsqrt.f32` and `libm::sqrtf`).
//! The model is the one of `crates/minikalman/examples/attitude_eskf.rs` (#184), written as
//! 3×3 blocks placed at an offset, so the same functions serve both state layouts:
//!
//! ```text
//! attitude:  δx = (δθ, δb_g)               attitude block at 0
//! GNSS/INS:  δx = (δp, δv, δθ, δb_g)       attitude block at 6
//! ```
//!
//! The world frame has z up and gravity `(0, 0, −9.81)`, so the accelerometer at rest measures
//! `f = Rᵀ·(0, 0, 9.81)`. `δθ` is the local (body-frame) attitude error, `q = q̂ ⊗ Exp(δθ)`.
//! This module holds only the model, not the benchmark, so the GNSS/INS example (#186) can
//! share it.

use minikalman::prelude::*;

/// A 3×3 matrix as rows.
pub type Mat3 = [[f32; 3]; 3];

/// A 3-vector.
pub type Vec3 = [f32; 3];

/// Magnitude of gravity in m/s².
pub const GRAVITY: f32 = 9.81;

/// Largest rotation angle in radians for which [`Quat::exp`] uses its series; the first
/// omitted terms are below `θ⁶/46 080 ≈ 3·10⁻¹³` there.
pub const SERIES_ANGLE: f32 = 0.05;

/// Offset of the attitude error `δθ` in the attitude state.
pub const ATT_ATTITUDE: usize = 0;

/// Offset of the position error `δp` in the GNSS/INS state.
pub const INS_POSITION: usize = 0;

/// Offset of the velocity error `δv` in the GNSS/INS state.
pub const INS_VELOCITY: usize = 3;

/// Offset of the attitude error `δθ` in the GNSS/INS state.
pub const INS_ATTITUDE: usize = 6;

/// Offset of the gyro bias error `δb_g` in the GNSS/INS state.
pub const INS_BIAS: usize = 9;

/// A unit quaternion `w + x·i + y·j + z·k` (Hamilton convention), body to world.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat {
    /// The scalar part.
    pub w: f32,
    /// The vector part.
    pub v: Vec3,
}

impl Quat {
    /// The identity rotation.
    pub const IDENTITY: Quat = Quat {
        w: 1.0,
        v: [0.0; 3],
    };

    /// Returns the Hamilton product `self ⊗ other`.
    pub fn mul(&self, other: &Quat) -> Quat {
        let (a, b) = (self.v, other.v);
        let c = cross(a, b);

        Quat {
            w: self.w * other.w - dot(a, b),
            v: core::array::from_fn(|i| self.w * b[i] + other.w * a[i] + c[i]),
        }
    }

    /// Returns the inverse of a unit quaternion.
    pub fn conjugate(&self) -> Quat {
        Quat {
            w: self.w,
            v: self.v.map(|x| -x),
        }
    }

    /// Returns the quaternion scaled to unit length, which `f32` products slowly lose.
    pub fn normalized(&self) -> Quat {
        let n = (self.w * self.w + dot(self.v, self.v)).square_root();

        Quat {
            w: self.w / n,
            v: self.v.map(|x| x / n),
        }
    }

    /// Returns `Exp(θ)`: the rotation by `|θ|` radians about `θ/|θ|`.
    ///
    /// Below [`SERIES_ANGLE`] it uses the Taylor series of `cos(|θ|/2)` and `sin(|θ|/2)/|θ|`
    /// to the fourth order, exact in `f32` there. Gyro increments at a few hundred hertz and
    /// error injections are always that small. `libm::sinf` and `libm::cosf` compute in
    /// `f64`, which is soft float on a Cortex-M4F (thousands of instructions per pair).
    pub fn exp(theta: Vec3) -> Quat {
        let a2 = dot(theta, theta);
        if a2 < SERIES_ANGLE * SERIES_ANGLE {
            let scale = 0.5 - a2 / 48.0 + a2 * a2 / 3840.0;
            return Quat {
                w: 1.0 - a2 / 8.0 + a2 * a2 / 384.0,
                v: theta.map(|x| x * scale),
            };
        }

        let angle = a2.square_root();
        let half = 0.5 * angle;
        let scale = libm::sinf(half) / angle;

        Quat {
            w: libm::cosf(half),
            v: theta.map(|x| x * scale),
        }
    }

    /// Returns the rotation matrix `R` (body to world), rows.
    pub fn rotation(&self) -> Mat3 {
        let Quat { w, v: [x, y, z] } = *self;

        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ]
    }
}

/// Returns `Rᵀ·v`: a world-frame vector in the body frame.
pub fn transpose_mul(r: &Mat3, v: Vec3) -> Vec3 {
    core::array::from_fn(|c| (0..3).map(|k| r[k][c] * v[k]).sum())
}

/// Returns `R·v`: a body-frame vector in the world frame.
pub fn mat_mul(r: &Mat3, v: Vec3) -> Vec3 {
    r.map(|row| dot(row, v))
}

/// Returns the yaw `ψ = atan2(R₁₀, R₀₀)` of `R` in `(−π, π]` radians.
pub fn yaw(r: &Mat3) -> f32 {
    libm::atan2f(r[1][0], r[0][0])
}

/// Returns the skew-symmetric matrix `[v]×`, so that `[v]×·w = v × w`.
pub fn skew([x, y, z]: Vec3) -> Mat3 {
    [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]]
}

/// Returns `a × b`.
pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Returns `a · b`.
pub fn dot(a: Vec3, b: Vec3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Returns `|v|`.
pub fn norm(v: Vec3) -> f32 {
    dot(v, v).square_root()
}

/// Returns the attitude block `Exp(u)ᵀ` of `F` for the gyro increment `u = (ω_m − b̂)·Δt`.
///
/// The bias block of `F` is `−Δt·I` to its right and `I` on the diagonal.
pub fn attitude_transition(u: Vec3) -> Mat3 {
    let r = Quat::exp(u).rotation();
    core::array::from_fn(|i| core::array::from_fn(|j| r[j][i]))
}

/// Returns the velocity–attitude block `−R̂·[f]×·Δt` of the GNSS/INS `F` for the body-frame
/// specific force `f` (m/s²).
pub fn velocity_attitude_transition(r: &Mat3, f: Vec3, dt: f32) -> Mat3 {
    let s = skew(f);
    core::array::from_fn(|i| {
        core::array::from_fn(|j| -dt * (r[i][0] * s[0][j] + r[i][1] * s[1][j] + r[i][2] * s[2][j]))
    })
}

/// Returns the tilt Jacobian `[f̂]×` of the specific force `f̂ = R̂ᵀ·g` predicted in the body frame.
pub fn gravity_jacobian(predicted: Vec3) -> Mat3 {
    skew(predicted)
}

/// Returns the row `∂ψ/∂δθ` of the yaw `ψ = atan2(R₁₀, R₀₀)` for the attitude `R̂`.
///
/// Singular at ±90° pitch, where the yaw is undefined.
pub fn yaw_jacobian(r: &Mat3) -> Vec3 {
    let c = r[0][0] * r[0][0] + r[1][0] * r[1][0];
    let world = [-r[0][0] * r[2][0] / c, -r[1][0] * r[2][0] / c, 1.0];

    // h_world · R̂, since the world-frame error is φ = R̂·δθ.
    core::array::from_fn(|j| world[0] * r[0][j] + world[1] * r[1][j] + world[2] * r[2][j])
}

/// Returns the attitude block `I − ½[δθ]×` of the reset Jacobian `G`; every other block is `I`.
pub fn reset_jacobian(dtheta: Vec3) -> Mat3 {
    let s = skew(dtheta);
    core::array::from_fn(|r| core::array::from_fn(|c| if r == c { 1.0 } else { -0.5 * s[r][c] }))
}

/// Replaces `P` by `G·P·Gᵀ` where `G` is the identity except for the 3×3 block `g` at
/// `(offset, offset)`: 18·N multiplications instead of the N⁴ of a full product.
pub fn reset_covariance<const N: usize>(
    p: &mut impl MatrixMut<N, N, f32>,
    offset: usize,
    g: &Mat3,
) {
    // Rows: P[o..o+3, :] ← g·P[o..o+3, :].
    for c in 0..N {
        let col: Vec3 = core::array::from_fn(|k| p.get_at(offset + k, c));
        (0..3).for_each(|r| p.set_at(offset + r, c, dot(g[r], col)));
    }

    // Columns: P[:, o..o+3] ← P[:, o..o+3]·gᵀ.
    for r in 0..N {
        let row: Vec3 = core::array::from_fn(|k| p.get_at(r, offset + k));
        (0..3).for_each(|c| p.set_at(r, offset + c, dot(g[c], row)));
    }
}

/// Writes a 3×3 block into `m` with its top-left corner at `(row, col)`.
pub fn write_block<const R: usize, const C: usize>(
    m: &mut impl MatrixMut<R, C, f32>,
    row: usize,
    col: usize,
    block: &Mat3,
) {
    for (r, values) in block.iter().enumerate() {
        for (c, value) in values.iter().enumerate() {
            m.set_at(row + r, col + c, *value);
        }
    }
}

/// Writes `v` on the diagonal of the 3×3 block at `(offset, offset)` of `m`.
pub fn write_diagonal<const N: usize>(m: &mut impl MatrixMut<N, N, f32>, offset: usize, v: f32) {
    (0..3).for_each(|i| m.set_at(offset + i, offset + i, v));
}

/// Moves the error `δθ` (at `offset` of `δx`) into the nominal attitude.
pub fn inject_attitude(attitude: &Quat, dtheta: Vec3) -> Quat {
    attitude.mul(&Quat::exp(dtheta)).normalized()
}

/// Reads the 3-vector at `offset` of a state vector.
pub fn read3<const N: usize>(x: &impl Matrix<N, 1, f32>, offset: usize) -> Vec3 {
    core::array::from_fn(|i| x.get_at(offset + i, 0))
}
