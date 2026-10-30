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
    exports.radar_correct(Math.hypot(x, y), Math.atan2(y, x));
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
