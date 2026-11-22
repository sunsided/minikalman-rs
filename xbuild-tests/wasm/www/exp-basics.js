// The basics: a linear constant-velocity filter on a signal you can draw.
// Uses the step-by-step exports `reset`, `step` and `position_variance`.

import { Scope, instantiate, knobs, num, offset, readout, timeAxis } from "./lab.js";

/** Box-Muller transform for normally distributed noise. */
function gaussian() {
  const u = 1 - Math.random();
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export async function start() {
  const kf = await instantiate();
  const canvas = document.getElementById("hero-plot");
  const scope = new Scope(canvas, { ratio: 0.45, xLabel: "t (s)" });
  canvas.style.cursor = "crosshair";

  const n = 300, dt = 0.1;
  const yRange = [-15, 50]; // fixed, so a drawn signal stays where it was drawn
  const t = timeAxis(n, dt);
  const defaultSignal = () => Float32Array.from({ length: n }, (_, i) => 10 * Math.sin((i * dt) / 4) + i * dt);
  let signal = defaultSignal();
  // Kept between runs, so the knobs compare like with like.
  let noise = Float32Array.from({ length: n }, gaussian);
  let lastDrawn = null;

  // The process noise knob is logarithmic, from 0.01 to 100.
  const q = (v) => 10 ** (-2 + (4 * v) / 100);
  const read = knobs(
    {
      "hero-sigma": (v) => `${v.toFixed(1)} m`,
      "hero-q": (v) => q(v).toPrecision(2),
    },
    () => run(false),
  );

  function run(reveal) {
    const values = read();
    const sigma = values["hero-sigma"];
    kf.reset(dt, q(values["hero-q"]), sigma * sigma);
    const z = signal.map((x, i) => x + sigma * noise[i]);
    const estimate = new Float32Array(n), sd = new Float32Array(n);
    z.forEach((zi, i) => {
      estimate[i] = kf.step(zi);
      sd[i] = Math.sqrt(kf.position_variance());
    });
    scope.draw({
      t,
      y: yRange,
      reveal,
      layers: [
        { kind: "band", lo: offset(estimate, sd, -2), hi: offset(estimate, sd, 2), color: "--c2" },
        { kind: "dots", y: z, color: "--meas", r: 1.5 },
        { kind: "line", y: signal, color: "--truth", dash: [5, 4], width: 1.5 },
        { kind: "line", y: estimate, color: "--c2", width: 2 },
      ],
    });
    const rms = (a) => Math.sqrt(a.reduce((s, v, i) => s + (v - signal[i]) ** 2, 0) / n);
    readout("hero-rms-z", num(rms(z)), "m");
    readout("hero-rms-x", num(rms(estimate)), "m");
  }

  // Drawing: map the pointer to a sample index and value, filling any gap linearly.
  function drawAt(event) {
    const at = scope.toData(event);
    const index = Math.min(Math.max(Math.round(at.x / dt), 0), n - 1);
    const value = Math.min(Math.max(at.y, yRange[0]), yRange[1]);
    if (lastDrawn) {
      const [li, lv] = lastDrawn;
      const steps = Math.abs(index - li);
      for (let k = 1; k < steps; k++) signal[li + Math.sign(index - li) * k] = lv + ((value - lv) * k) / steps;
    }
    signal[index] = value;
    lastDrawn = [index, value];
    run(false);
  }

  canvas.addEventListener("pointerdown", (event) => {
    lastDrawn = null;
    try {
      canvas.setPointerCapture(event.pointerId);
    } catch {
      // Synthetic events have no active pointer to capture.
    }
    canvas.style.touchAction = "none";
    drawAt(event);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (lastDrawn) drawAt(event);
  });
  const end = () => {
    lastDrawn = null;
    canvas.style.touchAction = "";
  };
  canvas.addEventListener("pointerup", end);
  canvas.addEventListener("pointercancel", end);

  document.getElementById("hero-noise").addEventListener("click", () => {
    noise = Float32Array.from({ length: n }, gaussian);
    run(true);
  });
  document.getElementById("hero-reset").addEventListener("click", () => {
    signal = defaultSignal();
    run(true);
  });
  run(true);
}
