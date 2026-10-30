// Loads the built module with no imports and checks that the filter tracks
// a target moving at constant velocity.
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
