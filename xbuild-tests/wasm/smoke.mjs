// Loads the built module with no imports and checks that the 1D and radar
// filters track targets moving at constant velocity.
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
