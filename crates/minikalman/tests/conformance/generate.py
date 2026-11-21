#!/usr/bin/env python3
"""Generates the reference vectors of the minikalman conformance suite.

Every number in `reference.rs` comes from this script, which implements the filters
independently with NumPy and SciPy. The crate itself is never used to produce a reference.

    python3 crates/minikalman/tests/conformance/generate.py
    git diff crates/minikalman/tests/conformance/reference.rs

The script also checks itself: the Riccati residual of the steady-state solution, and the
RTS smoother against the batch least-squares smoother. It fails if either disagrees.

Noise comes from `numpy.random.RandomState`, whose stream NumPy keeps stable across versions,
shaped by Cholesky factors, which are unique. On the same NumPy/LAPACK stack a regenerated
file is byte-identical; elsewhere the last digits may differ, and the portable check is that
the suite still passes against the regenerated file.
"""

import pathlib
import sys

import numpy as np
import scipy
from scipy.linalg import solve_discrete_are
from scipy.stats import chi2

try:
    from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter
except ImportError:  # optional cross-check
    UnscentedKalmanFilter = None

OUT = pathlib.Path(__file__).with_name("reference.rs")

# Monte Carlo configuration of the statistical consistency checks.
MC_RUNS = 100
MC_STEPS = 50
# Total false-failure probability of each consistency test for a correct filter.
MC_ALPHA = 1e-3
# Steps at which the run-averaged NEES is checked (Bonferroni-corrected).
MC_NEES_STEPS = [20, 30, 40, 50]


def cv_model(dt, q):
    """Constant velocity with continuous white-noise acceleration (full-rank `Q`)."""
    A = np.array([[1.0, dt], [0.0, 1.0]])
    Q = q * np.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]])
    return A, Q


def gaussian(rng, mean, cov):
    """Draws from `N(mean, cov)` through the Cholesky factor of `cov`.

    `RandomState.multivariate_normal` factorizes with an SVD, whose singular-vector signs can
    differ between LAPACK builds; a Cholesky factor is unique, so the draw is portable.
    """
    return mean + np.linalg.cholesky(cov) @ rng.standard_normal(cov.shape[0])


def simulate_linear(rng, A, Q, H, R, x0, P0, steps):
    """Draws a truth trajectory and its measurements; `x_k = A x_k-1 + w`, `z_k = H x_k + v`."""
    x = gaussian(rng, x0, P0)
    truth, zs = [], []
    for _ in range(steps):
        x = A @ x + gaussian(rng, 0.0, Q)
        z = H @ x + gaussian(rng, 0.0, R)
        truth.append(x)
        zs.append(z)
    return np.array(truth), np.array(zs)


def kalman(A, Q, H, R, x0, P0, zs):
    """Plain Kalman filter, predict then correct per step. Returns priors and posteriors."""
    x, P = x0.copy(), P0.copy()
    xp, Pp, xf, Pf = [], [], [], []
    for z in zs:
        x = A @ x
        P = A @ P @ A.T + Q
        xp.append(x)
        Pp.append(P)
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ (z - H @ x)
        P = (np.eye(len(x)) - K @ H) @ P
        xf.append(x)
        Pf.append(P)
    return np.array(xp), np.array(Pp), np.array(xf), np.array(Pf)


def rts(A, xp, Pp, xf, Pf):
    """Rauch-Tung-Striebel smoother over a filtered history."""
    xs, Ps = xf.copy(), Pf.copy()
    for k in range(len(xf) - 2, -1, -1):
        C = Pf[k] @ A.T @ np.linalg.inv(Pp[k + 1])
        xs[k] = xf[k] + C @ (xs[k + 1] - xp[k + 1])
        Ps[k] = Pf[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ C.T
    return xs, Ps


def batch_smoother(A, Q, H, R, x0, P0, zs):
    """Smoothed marginals of x_1..x_K from the joint Gaussian over x_0..x_K, solved at once."""
    n, K = len(x0), len(zs)
    N = n * (K + 1)
    info = np.zeros((N, N))
    vec = np.zeros(N)
    P0i, Qi, Ri = np.linalg.inv(P0), np.linalg.inv(Q), np.linalg.inv(R)

    def blk(i):
        return slice(i * n, (i + 1) * n)

    info[blk(0), blk(0)] += P0i
    vec[blk(0)] += P0i @ x0
    for k in range(1, K + 1):
        # Dynamics residual x_k - A x_k-1, with weight Q^-1.
        J = np.zeros((n, N))
        J[:, blk(k)] = np.eye(n)
        J[:, blk(k - 1)] = -A
        info += J.T @ Qi @ J
        # Measurement residual z_k - H x_k, with weight R^-1.
        info[blk(k), blk(k)] += H.T @ Ri @ H
        vec[blk(k)] += H.T @ Ri @ zs[k - 1]
    cov = np.linalg.inv(info)
    mean = cov @ vec
    xs = np.array([mean[blk(k)] for k in range(1, K + 1)])
    Ps = np.array([cov[blk(k), blk(k)] for k in range(1, K + 1)])
    return xs, Ps


def radar_h(x, radar):
    """Range and bearing of the target from the radar."""
    dx, dy = x[0] - radar[0], x[1] - radar[1]
    return np.array([np.hypot(dx, dy), np.arctan2(dy, dx)])


def radar_jacobian(x, radar):
    """Jacobian of `radar_h` at `x`."""
    dx, dy = x[0] - radar[0], x[1] - radar[1]
    r2 = dx * dx + dy * dy
    r = np.sqrt(r2)
    return np.array([[dx / r, dy / r, 0.0, 0.0], [-dy / r2, dx / r2, 0.0, 0.0]])


def ekf_radar(A, Q, R, radar, x0, P0, zs):
    """Extended Kalman filter with a linear transition and the range-bearing observation."""
    x, P = x0.copy(), P0.copy()
    xf, Pf = [], []
    for z in zs:
        x = A @ x
        P = A @ P @ A.T + Q
        H = radar_jacobian(x, radar)
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ (z - radar_h(x, radar))
        P = (np.eye(4) - K @ H) @ P
        xf.append(x)
        Pf.append(P)
    return np.array(xf), np.array(Pf)


def pendulum_f(x, dt, g_over_l):
    """One explicit Euler step of a frictionless pendulum `[angle, angular rate]`."""
    return np.array([x[0] + x[1] * dt, x[1] - g_over_l * np.sin(x[0]) * dt])


def pendulum_h(x, length):
    """Horizontal position of the bob."""
    return np.array([length * np.sin(x[0])])


def ukf(f, h, Q, R, x0, P0, zs, alpha, beta, kappa):
    """Unscented Kalman filter (Wan and van der Merwe weights).

    Follows the convention also used by filterpy: the correction passes the sigma points
    propagated by the prediction through `h`, without redrawing them from the predicted `P`,
    so `Q` enters `P` but not the spread of the observed sigma points.
    """
    n = len(x0)
    lam = alpha**2 * (n + kappa) - n
    wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
    wc = wm.copy()
    wm[0] = lam / (n + lam)
    wc[0] = lam / (n + lam) + (1.0 - alpha**2 + beta)
    x, P = x0.copy(), P0.copy()
    xf, Pf = [], []
    for z in zs:
        L = np.linalg.cholesky(P) * np.sqrt(n + lam)
        sigma = np.column_stack([x] + [x + L[:, j] for j in range(n)] + [x - L[:, j] for j in range(n)])
        prop = np.column_stack([f(sigma[:, j]) for j in range(2 * n + 1)])
        x = prop @ wm
        dx = prop - x[:, None]
        P = (dx * wc) @ dx.T + Q
        obs = np.column_stack([h(prop[:, j]) for j in range(2 * n + 1)])
        zhat = obs @ wm
        dz = obs - zhat[:, None]
        S = (dz * wc) @ dz.T + R
        Pxz = (dx * wc) @ dz.T
        K = Pxz @ np.linalg.inv(S)
        x = x + K @ (z - zhat)
        P = P - K @ S @ K.T
        xf.append(x)
        Pf.append(P)
    return np.array(xf), np.array(Pf)


def check_against_filterpy(dt, g_over_l, length, alpha, beta, kappa, Q, R, x0, P0, zs, xf, Pf):
    """Asserts that filterpy's UKF reproduces `ukf` on the pendulum (only if it is installed)."""
    points = MerweScaledSigmaPoints(2, alpha=alpha, beta=beta, kappa=kappa)
    fp = UnscentedKalmanFilter(
        dim_x=2, dim_z=1, dt=dt, points=points,
        fx=lambda s, _dt: pendulum_f(s, dt, g_over_l),
        hx=lambda s: pendulum_h(s, length),
    )
    fp.x, fp.P, fp.Q, fp.R = x0.copy(), P0.copy(), Q.copy(), R.copy()
    for k, z in enumerate(zs):
        fp.predict()
        fp.update(z)
        assert np.allclose(fp.x, xf[k], rtol=0, atol=1e-12), f"filterpy x at step {k}"
        assert np.allclose(fp.P, Pf[k], rtol=0, atol=1e-12), f"filterpy P at step {k}"
    print("filterpy UKF agrees with the reference UKF")


class Emitter:
    """Collects Rust modules of documented `f64` constants."""

    def __init__(self):
        self.parts = []

    def module(self, name, doc):
        self.parts.append(f"\n/// {doc}\npub mod {name} {{\n")

    def end(self):
        self.parts.append("}\n")

    def usize(self, name, value, doc):
        self.parts.append(f"    /// {doc}\n    pub const {name}: usize = {int(value)};\n")

    def usizes(self, name, values, doc):
        body = ", ".join(str(int(v)) for v in values)
        self.parts.append(f"    /// {doc}\n    pub const {name}: [usize; {len(values)}] = [{body}];\n")

    def array(self, name, values, doc):
        flat = np.asarray(values, dtype=float).ravel()
        assert np.all(np.isfinite(flat)), name
        self.parts.append(f"    /// {doc}\n    pub const {name}: [f64; {len(flat)}] = [\n")
        line = "       "
        for v in flat:
            item = f" {rust_float(v)},"
            if len(line) + len(item) > 99:
                self.parts.append(line + "\n")
                line = "       "
            line += item
        self.parts.append(line + "\n    ];\n")

    def scalar(self, name, value, doc):
        self.parts.append(f"    /// {doc}\n    pub const {name}: f64 = {rust_float(float(value))};\n")


def rust_float(v):
    """Shortest round-trip `f64` literal, always with a decimal point or exponent."""
    s = repr(float(v))
    if s == "-0.0":
        return "-0.0"
    if "e" not in s and "." not in s:
        s += ".0"
    return s


def linear_inputs(e, A, Q, H, R, x0, P0):
    """Emits the model matrices of a linear scenario."""
    e.array("A", A, "State transition matrix, row-major.")
    e.array("Q", Q, "Process noise covariance, row-major.")
    e.array("H", H, "Observation matrix, row-major.")
    e.array("R", R, "Measurement noise covariance, row-major.")
    e.array("X0", x0, "Initial state estimate.")
    e.array("P0", P0, "Initial estimate covariance, row-major.")


def emit_cv(e):
    """Constant velocity: KF, RTS, batch smoother, information filter and steady state."""
    dt, steps = 0.1, 50
    A, Q = cv_model(dt, 0.5)
    H = np.array([[1.0, 0.0]])
    R = np.array([[0.25]])
    x0, P0 = np.array([0.0, 1.0]), np.diag([1.0, 1.0])
    _, zs = simulate_linear(np.random.RandomState(1), A, Q, H, R, x0, P0, steps)
    xp, Pp, xf, Pf = kalman(A, Q, H, R, x0, P0, zs)
    xs, Ps = rts(A, xp, Pp, xf, Pf)
    xb, Pb = batch_smoother(A, Q, H, R, x0, P0, zs)
    assert np.allclose(xs, xb, rtol=0, atol=1e-10), "RTS != batch smoother (mean)"
    assert np.allclose(Ps, Pb, rtol=0, atol=1e-10), "RTS != batch smoother (covariance)"

    # Steady state: scipy solves X = A^T X A - A^T X B (R + B^T X B)^-1 B^T X A + Q for
    # (a, b) = (A^T, H^T), which is the prior covariance of the filter.
    P_steady = solve_discrete_are(A.T, H.T, Q, R)
    S = H @ P_steady @ H.T + R
    residual = A @ (P_steady - P_steady @ H.T @ np.linalg.inv(S) @ H @ P_steady) @ A.T + Q - P_steady
    assert np.max(np.abs(residual)) < 1e-12, "Riccati residual"
    K_steady = P_steady @ H.T @ np.linalg.inv(S)
    P_steady_post = (np.eye(2) - K_steady @ H) @ P_steady

    e.module("cv", "Constant velocity `[position, velocity]`, `dt = 0.1`, position measured.")
    linear_inputs(e, A, Q, H, R, x0, P0)
    e.array("Z", zs, "Measurements, one per step.")
    e.array("X_PRIOR", xp, "Predicted states `x_k|k-1`, one row per step.")
    e.array("P_PRIOR", Pp, "Predicted covariances `P_k|k-1`, row-major, one per step.")
    e.array("X_POST", xf, "Filtered states `x_k|k`, one row per step.")
    e.array("P_POST", Pf, "Filtered covariances `P_k|k`, row-major, one per step.")
    e.array("X_SMOOTH", xs, "RTS-smoothed states `x_k|N`, equal to the batch smoother.")
    e.array("P_SMOOTH", Ps, "RTS-smoothed covariances `P_k|N`, equal to the batch smoother.")
    e.array("X_BATCH", xb, "Batch least-squares smoothed states, solved as one linear system.")
    e.array("P_BATCH", Pb, "Batch least-squares smoothed covariances.")
    e.array("P_STEADY_PRIOR", P_steady, "Steady-state prior covariance from `solve_discrete_are`.")
    e.array("P_STEADY_POST", P_steady_post, "Steady-state posterior covariance.")
    e.array("K_STEADY", K_steady, "Steady-state Kalman gain.")
    e.end()


def emit_gravity(e):
    """Free fall `[position, velocity, acceleration]` with position measurements."""
    dt, steps = 1.0, 15
    A = np.array([[1.0, dt, dt * dt / 2.0], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
    Q = np.diag([1e-3, 1e-3, 1e-3])
    H = np.array([[1.0, 0.0, 0.0]])
    R = np.array([[0.5]])
    x0, P0 = np.array([0.0, 0.0, 6.0]), np.diag([1.0, 1.0, 4.0])
    rng = np.random.RandomState(2)
    zs = np.array([[0.5 * 9.81 * (k * dt) ** 2 + rng.normal(0.0, np.sqrt(0.5))] for k in range(1, steps + 1)])
    _, _, xf, Pf = kalman(A, Q, H, R, x0, P0, zs)

    e.module("gravity", "Free fall `[position, velocity, acceleration]`, `dt = 1`, position measured.")
    linear_inputs(e, A, Q, H, R, x0, P0)
    e.array("Z", zs, "Measurements of a body falling at 9.81 m/s², one per step.")
    e.array("X_POST", xf, "Filtered states `x_k|k`, one row per step.")
    e.array("P_POST", Pf, "Filtered covariances `P_k|k`, row-major, one per step.")
    e.end()


def emit_radar(e):
    """Range-bearing radar tracking a constant-velocity target with an EKF."""
    dt, steps = 0.1, 40
    radar = np.array([-10.0, -2.0])
    A1, Q1 = cv_model(dt, 0.1)
    # State [px, py, vx, vy]: interleave the per-axis constant-velocity blocks.
    A = np.kron(A1, np.eye(2))
    Q = np.kron(Q1, np.eye(2))
    R = np.diag([0.05, 0.001])
    x0, P0 = np.array([0.0, 0.0, 1.0, 0.5]), np.diag([0.5, 0.5, 0.2, 0.2])
    rng = np.random.RandomState(3)
    x = gaussian(rng, x0, P0)
    zs = []
    for _ in range(steps):
        x = A @ x + gaussian(rng, 0.0, Q)
        zs.append(radar_h(x, radar) + gaussian(rng, 0.0, R))
    zs = np.array(zs)
    xf, Pf = ekf_radar(A, Q, R, radar, x0, P0, zs)

    e.module("radar", "EKF: constant-velocity target `[px, py, vx, vy]`, radar measures range and bearing.")
    e.array("RADAR", radar, "Radar position `[x, y]`.")
    e.array("A", A, "State transition matrix (and its Jacobian), row-major.")
    e.array("Q", Q, "Process noise covariance, row-major.")
    e.array("R", R, "Measurement noise covariance (range, bearing), row-major.")
    e.array("X0", x0, "Initial state estimate.")
    e.array("P0", P0, "Initial estimate covariance, row-major.")
    e.array("Z", zs, "Measurements `[range, bearing]`, one row per step.")
    e.array("X_POST", xf, "Filtered states `x_k|k`, one row per step.")
    e.array("P_POST", Pf, "Filtered covariances `P_k|k`, row-major, one per step.")
    e.end()


def emit_pendulum(e):
    """Nonlinear pendulum tracked by a UKF from the bob's horizontal position."""
    dt, steps, g_over_l, length = 0.05, 60, 9.81, 1.0
    alpha, beta, kappa = 1.0, 2.0, 1.0
    Q = np.diag([1e-5, 1e-3])
    R = np.array([[0.01]])
    x0, P0 = np.array([1.0, 0.0]), np.diag([0.1, 0.1])
    rng = np.random.RandomState(4)
    x = gaussian(rng, x0, P0)
    zs = []
    for _ in range(steps):
        x = pendulum_f(x, dt, g_over_l) + gaussian(rng, 0.0, Q)
        zs.append(pendulum_h(x, length) + gaussian(rng, 0.0, R))
    zs = np.array(zs)
    xf, Pf = ukf(
        lambda s: pendulum_f(s, dt, g_over_l),
        lambda s: pendulum_h(s, length),
        Q, R, x0, P0, zs, alpha, beta, kappa,
    )

    if UnscentedKalmanFilter is not None:
        check_against_filterpy(dt, g_over_l, length, alpha, beta, kappa, Q, R, x0, P0, zs, xf, Pf)

    e.module("pendulum", "UKF: pendulum `[angle, rate]`, Euler-stepped, bob's horizontal position measured.")
    e.scalar("DT", dt, "Time step in seconds.")
    e.scalar("G_OVER_L", g_over_l, "Gravity over pendulum length, in 1/s².")
    e.scalar("LENGTH", length, "Pendulum length in metres.")
    e.scalar("ALPHA", alpha, "Sigma-point spread `α` (the crate's default).")
    e.scalar("BETA", beta, "Prior-knowledge parameter `β` (the crate's default).")
    e.scalar("KAPPA", kappa, "Secondary scaling `κ` (the crate's default).")
    e.array("Q", Q, "Process noise covariance, row-major.")
    e.array("R", R, "Measurement noise covariance.")
    e.array("X0", x0, "Initial state estimate.")
    e.array("P0", P0, "Initial estimate covariance, row-major.")
    e.array("Z", zs, "Measurements, one per step.")
    e.array("X_POST", xf, "Filtered states `x_k|k`, one row per step.")
    e.array("P_POST", Pf, "Filtered covariances `P_k|k`, row-major, one per step.")
    e.end()


def emit_chi2(e):
    """Chi-square acceptance bounds of the Monte Carlo consistency checks."""
    nees_alpha = MC_ALPHA / len(MC_NEES_STEPS)

    e.module("chi2", "Monte Carlo configuration and two-sided chi-square bounds (`scipy.stats.chi2`).")
    e.usize("RUNS", MC_RUNS, "Independent Monte Carlo runs, each with its own seed.")
    e.usize("STEPS", MC_STEPS, "Predict-correct steps per run.")
    e.scalar("ALPHA", MC_ALPHA, "False-failure probability of each test for a consistent filter.")
    e.usizes("NEES_STEPS", MC_NEES_STEPS, "Steps (1-based) at which the run-averaged NEES is tested.")
    for name, n in (("NEES2", 2), ("NEES4", 4)):
        dof = MC_RUNS * n
        lo = chi2.ppf(nees_alpha / 2.0, dof) / MC_RUNS
        hi = chi2.ppf(1.0 - nees_alpha / 2.0, dof) / MC_RUNS
        e.array(
            name,
            [lo, hi],
            f"Bounds of the NEES of {n} states averaged over the runs at one step: "
            f"`chi2({dof})/{MC_RUNS}` at `α/{len(MC_NEES_STEPS)}` (Bonferroni).",
        )
    for name, m in (("NIS1", 1), ("NIS2", 2)):
        dof = MC_RUNS * MC_STEPS * m
        lo = chi2.ppf(MC_ALPHA / 2.0, dof) / (MC_RUNS * MC_STEPS)
        hi = chi2.ppf(1.0 - MC_ALPHA / 2.0, dof) / (MC_RUNS * MC_STEPS)
        e.array(
            name,
            [lo, hi],
            f"Bounds of the NIS of {m} measurement(s) averaged over all runs and steps: "
            f"`chi2({dof})/{MC_RUNS * MC_STEPS}`.",
        )
    e.end()


def main():
    """Writes `reference.rs`, which `main.rs` excludes from rustfmt."""
    e = Emitter()
    emit_cv(e)
    emit_gravity(e)
    emit_radar(e)
    emit_pendulum(e)
    emit_chi2(e)
    header = (
        "//! Reference vectors of the conformance suite.\n"
        "//!\n"
        "//! Generated by `generate.py` with NumPy and SciPy, independently of minikalman. Do not\n"
        "//! edit by hand: change the script, rerun it and commit both.\n"
        f"//!\n//! Generator: Python {sys.version_info.major}.{sys.version_info.minor}, "
        f"NumPy {np.__version__}, SciPy {scipy.__version__}.\n"
    )
    OUT.write_text(header + "".join(e.parts))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
