## The fading-memory factor `λ`

`λ` (`0 < λ ≤ 1`) is a *fading-memory* or *forgetting* factor. The prediction inflates the
propagated covariance before adding the process noise,

```text
P⁻ = A·P·Aᵀ / λ² + Q
```

With `λ = 1` this is the plain prediction. With `λ < 1`, the filter trusts its own past a little
less on every step, so the gain does not decay towards zero and the estimate keeps following the
measurements.

### What it does

A Kalman filter's covariance shrinks with every correction. If the model is slightly wrong
(unmodeled dynamics, drifting parameters, a `Q` that is too small), `P` becomes overconfident,
the gain `K` goes to zero and the filter stops reacting to new data; it diverges from the truth
while reporting a small uncertainty. Dividing by `λ²` counteracts that: it is equivalent to
weighting a measurement that is `j` steps old by `λ²ʲ` relative to the newest one. Old data
fades out exponentially; the effective memory is about `1 / (1 − λ²) ≈ 1 / (2·(1 − λ))` steps
(for example `λ = 0.99` remembers roughly 50 steps).

### Why it works

Scaling `P` by `1/λ²` is the same as running the filter on a problem in which older
measurements are noisier: the measurement noise of a sample `j` steps back is inflated by
`λ⁻²ʲ`. The filter then computes the optimal estimate for that *exponentially weighted* problem
(see "exponential data weighting" in Anderson & Moore, *Optimal Filtering*, 1979, and the
"fading-memory filter" in Simon, *Optimal State Estimation*, 2006).

### Relation to recursive least squares (RLS)

With `A = I`, `Q = 0` and a time-varying observation row `H = φₖᵀ` (the regressor), the state is a
parameter vector and the filter is exactly the exponentially weighted RLS estimator: running the
tuned prediction with `λ` before each correction minimizes `Σ μᵏ⁻ⁱ·(yᵢ − φᵢᵀ·θ)²` with the classic RLS
forgetting factor

```text
μ = λ²    (equivalently λ = √μ)
```

so the usual RLS range `μ ∈ [0.95, 0.999]` corresponds to `λ ∈ [0.975, 0.9995]`. In other words,
this crate's `λ` is the square root of the RLS forgetting factor: RLS divides its covariance by `μ`
per step, the prediction here by `λ²`.

### `λ` or `Q`?

Both keep the filter alert, in different ways:

- `Q` adds a fixed, absolute amount of uncertainty in the directions you choose; it models a
  known random-walk drift of the state.
- `λ` inflates the uncertainty *relative* to `P`, equally in every direction, without a model of
  the drift; it is the tool when the size or shape of the model error is unknown.

### Caveats

- **Windup.** The inflation is applied on every prediction. Without informative measurements
  (or in state directions the measurements do not excite), `P` grows by `1/λ²` per step without
  bound. Correct regularly, keep `λ` close to 1, or bound `P`.
- **Not more accurate.** Forgetting trades noise suppression for tracking: with a correct model
  and `Q`, the plain prediction (`λ = 1`) is optimal, and `λ < 1` only adds variance.
- **Range.** Values above 1 would shrink `P` and make the filter overconfident; they are outside
  the intended range.
