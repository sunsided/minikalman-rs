// Loads the built module with no imports and checks that the 1D and radar
// filters track targets moving at constant velocity, then runs every lab experiment
// at its page defaults and at the extremes of its knobs, and checks the claims the
// page makes about each.
//
// Usage: node smoke.mjs path/to/minikalman_wasm.wasm

import { readFile } from "node:fs/promises";

const path = process.argv[2];
if (!path) {
  console.error("usage: node smoke.mjs <module.wasm>");
  process.exit(2);
}

const module = await WebAssembly.compile(await readFile(path));
const imports = WebAssembly.Module.imports(module);
if (imports.length > 0) {
  console.error("module expects imports:", imports);
  process.exit(1);
}

const { exports } = await WebAssembly.instantiate(module, {});
const dt = 0.1;
exports.reset(dt, 0.01, 0.25);
let position = 0;
for (let i = 0; i < 200; i++) {
  position = exports.step(2.0 * dt * i);
}

const velocity = exports.velocity();
console.log(`p=${position.toFixed(3)} v=${velocity.toFixed(3)} var=${exports.position_variance().toFixed(4)}`);
if (Math.abs(velocity - 2.0) > 0.05) {
  console.error(`expected v≈2.0, got ${velocity}`);
  process.exit(1);
}

// Radar: a target passing at constant velocity, measured every 0.5 s.
const [vx, vy] = [4.0, 0.5];
exports.radar_reset(-40, 15, 0.5, 0.5, 0.02);
for (let step = 1; step <= 200; step++) {
  exports.radar_predict(dt);
  if (step % 5 === 0) {
    const t = step * dt;
    const [x, y] = [-40 + vx * t, 15 + vy * t];
    exports.radar_correct(0, 0, Math.hypot(x, y), Math.atan2(y, x));
  }
}
for (const [name, variant] of [["EKF", 0], ["UKF", 1]]) {
  const state = [0, 1, 2, 3].map((i) => exports.radar_state(variant, i));
  console.log(`${name}: x=${state[0].toFixed(2)} y=${state[1].toFixed(2)} vx=${state[2].toFixed(2)} vy=${state[3].toFixed(2)}`);
  if (Math.abs(state[2] - vx) > 0.3 || Math.abs(state[3] - vy) > 0.3) {
    console.error(`${name}: expected v≈(${vx}, ${vy})`);
    process.exit(1);
  }
}

// Two radars: two corrections per prediction, with the target circling behind the
// first radar so bearings cross ±π.
const circle = (t) => {
  const phase = (2 * Math.PI * t) / 40;
  return [4 + 28 * Math.cos(phase), 16 * Math.sin(phase)];
};
exports.radar_set_wrap(1);
exports.radar_reset(...circle(0), 1.0, 0.5, 0.12);
let worst = [0, 0];
for (let step = 1; step <= 600; step++) {
  exports.radar_predict(dt);
  const [x, y] = circle(step * dt);
  if (step % 5 === 0) {
    for (const [sx, sy] of [[0, 0], [30, -17]]) {
      exports.radar_correct(sx, sy, Math.hypot(x - sx, y - sy), Math.atan2(y - sy, x - sx));
    }
  }
  if (step > 100) {
    worst = worst.map((w, v) =>
      Math.max(w, Math.hypot(exports.radar_state(v, 0) - x, exports.radar_state(v, 1) - y)));
  }
}
console.log(`two radars, worst error: EKF ${worst[0].toFixed(2)} m, UKF ${worst[1].toFixed(2)} m`);
if (!worst.every((w) => w < 2)) {
  console.error("expected both filters to stay within 2 m of the target");
  process.exit(1);
}

// ── Lab experiments ──

// Reads the summary of the last run; a fresh view, since memory may have grown.
const summary = () => new Float32Array(exports.memory.buffer, exports.summary_ptr(), 64);
const failures = [];
const check = (ok, message) => {
  if (!ok) failures.push(message);
};
const fmt = (values) => Array.from(values, (v) => +v.toPrecision(4)).join(", ");

// Runs an experiment; a trap or a non-finite step count fails the smoke test.
function run(name, ...args) {
  let n;
  try {
    n = exports[`${name}_run`](...args);
  } catch (error) {
    failures.push(`${name}_run(${args.join(", ")}) trapped: ${error.message}`);
    return new Float32Array(64);
  }
  check(n > 0, `${name}_run(${args.join(", ")}) returned ${n} steps`);
  const ptr = exports.trace_ptr();
  check(ptr > 0, "trace_ptr is null");
  return summary().slice();
}

// Outliers: (seed, sigma, rate, size, gate, dof, jump, reacquire).
// Summary: [rmse plain, gated, robust, clean readings, outliers, rejected, caught, w outliers, w clean, re-acquisitions].
{
  const s = run("outliers", 1, 0.5, 0.1, 15, 9, 4, 0, 0);
  console.log(`outliers: rmse plain/gated/robust ${fmt(s.slice(0, 3))}`);
  check(s[1] < 0.5 * s[0] && s[2] < 0.5 * s[0], "outliers: gate and Student-t should halve the plain error");
  const locked = run("outliers", 1, 0.5, 0.1, 15, 9, 4, 3, 0);
  const recovered = run("outliers", 1, 0.5, 0.1, 15, 9, 4, 3, 5);
  console.log(`outliers, 3 m jump: gated ${fmt([locked[1]])} locked out, ${fmt([recovered[1]])} with re-acquire; robust ${fmt([locked[2]])}`);
  check(locked[1] > 5 * locked[2], "outliers: the gate should lock out after a real jump");
  check(recovered[1] < 0.2 * locked[1] && recovered[9] >= 1, "outliers: re-acquiring should recover");
  for (const args of [[0, 0, 0, 0, 0, 0, -99, 0], [4294967295, 99, 1, 999, 999, 999, 99, 999]]) {
    const e = run("outliers", ...args);
    check([0, 1, 2].every((i) => Number.isFinite(e[i])), `outliers(${args}): ${fmt(e.slice(0, 3))}`);
  }
}

// Numerics: (seed, sensors, log10 R, log10 q, steps). Summary: 4 per variant
// [not positive definite steps, first, largest asymmetry, final error], variants
// f32 std, f32 Joseph, I16F16 std, I16F16 Joseph, I32F32 std, I32F32 Joseph.
{
  const s = run("numerics", 1, 10, -4, -5, 600);
  const v = (i) => s.slice(4 * i, 4 * i + 4);
  console.log(`numerics: indefinite steps ${[0, 1, 2, 3, 4, 5].map((i) => v(i)[0]).join(", ")}`);
  check([0, 1, 4, 5].every((i) => v(i)[0] === 0 && v(i)[3] < 0.1), "numerics: f32 and I32F32 should stay positive definite and on track");
  check(v(1)[2] === 0 && v(5)[2] === 0 && v(0)[2] > 0, "numerics: Joseph exactly symmetric, standard not");
  check(v(2)[0] > 0 && v(3)[0] > 0 && v(3)[3] > 1, "numerics: I16F16 should lose definiteness, Joseph should diverge");
  const fine = run("numerics", 1, 10, -2, -5, 600);
  check(fine[8] === 0 && fine[12] === 0, "numerics: I16F16 should be fine with R = 1e-2");
  for (const args of [[0, 0, -99, -99, 0], [4294967295, 99, 99, 99, 99999]]) run("numerics", ...args);
}

// Smoothing: (seed, sigma, q, gap). Summary: [rmse filter, fixed lag, RTS, filter in gap,
// RTS in gap, largest sigma filter, RTS, readings, complete].
{
  const s = run("smoothing", 1, 0.8, 10 ** 0.6, 3);
  console.log(`smoothing: rmse filter/lag/RTS ${fmt(s.slice(0, 3))}, in the gap ${fmt(s.slice(3, 5))}`);
  check(s[2] < s[1] && s[1] < s[0], "smoothing: RTS < fixed lag < filter");
  check(s[4] < 0.25 * s[3] && s[6] < s[5], "smoothing: RTS should bridge the gap");
  for (const args of [[0, 0, 0, 0], [4294967295, 99, 999, 99]]) {
    const e = run("smoothing", ...args);
    check([0, 1, 2].every((i) => Number.isFinite(e[i])), `smoothing(${args}): ${fmt(e.slice(0, 3))}`);
  }
}

// Parameters: (seed, sensor sigma, log10 q, mu). Summary: [speed rmse after (joint, dual),
// a rms settled (joint, dual), b rms settled (joint, dual), re-id steps (joint, dual), ...].
{
  const s = run("parameters", 42, 0.2, -4, 0.98);
  console.log(`parameters: re-identified in ${s[6]} (joint) vs ${s[7]} (dual) steps`);
  // The page's note quotes these two numbers.
  check(s[6] === 25 && s[7] === 82, "parameters: the note quotes 25 (joint) and 82 (dual) steps");
  const frozen = run("parameters", 42, 0.2, -8, 1);
  check(frozen[6] < 0 && frozen[7] < 0, "parameters: q = 1e-8 and mu = 1 should stop both learning");
  for (const args of [[0, 0, -99, 0], [4294967295, 99, 99, 99]]) run("parameters", ...args);
}

// Attitude: (seed, dynamics, gate, hold-off, window, heading). Summary: [rms gated, window,
// largest gated, window, dynamic skipped, static gated, window took dynamic, mean NIS, ...].
{
  const s = run("attitude", 184, 1, 10 ** 1.05, 10, 0.1, 1);
  console.log(`attitude: largest error gated ${fmt([s[2]])}°, window ${fmt([s[3]])}°, dynamic skipped ${fmt([s[4]])}`);
  check(s[2] < 3 && s[3] > 3 * s[2], "attitude: the gate should keep the attitude, the window should not");
  check(s[4] > 0.9 && s[5] < 0.05 && s[6] > 0.9, "attitude: gate and window decisions");
  for (const args of [[0, 0, 0, 0, 0, 0], [4294967295, 99, 99999, 999, 99, 1]]) run("attitude", ...args);
}

if (failures.length > 0) {
  for (const failure of failures) console.error(failure);
  process.exit(1);
}
console.log("all lab experiments ok");
