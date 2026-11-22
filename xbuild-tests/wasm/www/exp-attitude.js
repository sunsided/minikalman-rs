// Attitude: a quaternion error-state EKF with an NIS-gated accelerometer vs a |a| window.
// Channels and summary: see `src/attitude.rs`.

import { Scope, batch, color, knobs, num, onTheme, readout } from "./lab.js";

const CH = { time: 0, truth: 1, gated: 4, window: 7, error: 10, gate: 12, windowUsed: 13, accel: 14, count: 16 };
const DYNAMIC = 0.3; // m/s², as in the Rust statistics

/** An artificial horizon on a square canvas: roll tilts the horizon, pitch shifts it. */
function dial(canvas) {
  const ctx = canvas.getContext("2d");
  let last = null;
  const paint = () => {
    if (!last) return;
    const [roll, pitch, yaw] = last;
    const size = canvas.clientWidth;
    if (!size) return;
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(size * dpr)) {
      canvas.width = Math.round(size * dpr);
      canvas.height = Math.round(size * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const r = size / 2 - 2;
    ctx.clearRect(0, 0, size, size);
    ctx.save();
    ctx.translate(size / 2, size / 2);
    ctx.beginPath();
    ctx.arc(0, 0, r, 0, 2 * Math.PI);
    ctx.clip();
    ctx.fillStyle = color("--screen");
    ctx.fillRect(-r, -r, 2 * r, 2 * r);

    // The world turns opposite to the body; 1° of pitch moves the horizon by r/45.
    ctx.save();
    ctx.rotate((-roll * Math.PI) / 180);
    const shift = (pitch * r) / 45;
    ctx.globalAlpha = 0.22;
    ctx.fillStyle = color("--c2");
    ctx.fillRect(-2 * r, -2 * r + shift, 4 * r, 2 * r);
    ctx.fillStyle = color("--meas");
    ctx.fillRect(-2 * r, shift, 4 * r, 2 * r);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = color("--ink");
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(-2 * r, shift);
    ctx.lineTo(2 * r, shift);
    ctx.stroke();
    // Pitch ladder every 10°.
    ctx.lineWidth = 1;
    ctx.strokeStyle = color("--ink-soft");
    for (let p = -30; p <= 30; p += 10) {
      if (!p) continue;
      const y = shift - (p * r) / 45;
      ctx.beginPath();
      ctx.moveTo(-r * 0.18, y);
      ctx.lineTo(r * 0.18, y);
      ctx.stroke();
    }
    ctx.restore();

    // The fixed body marker.
    ctx.strokeStyle = color("--ink");
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(-r * 0.55, 0);
    ctx.lineTo(-r * 0.15, 0);
    ctx.lineTo(0, r * 0.1);
    ctx.lineTo(r * 0.15, 0);
    ctx.lineTo(r * 0.55, 0);
    ctx.stroke();
    ctx.restore();

    ctx.strokeStyle = color("--grid-strong");
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(size / 2, size / 2, r, 0, 2 * Math.PI);
    ctx.stroke();
    // Heading tick on the rim.
    const a = ((yaw - 90) * Math.PI) / 180;
    ctx.strokeStyle = color("--ink");
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(size / 2 + Math.cos(a) * (r - 8), size / 2 + Math.sin(a) * (r - 8));
    ctx.lineTo(size / 2 + Math.cos(a) * r, size / 2 + Math.sin(a) * r);
    ctx.stroke();
  };
  new ResizeObserver(paint).observe(canvas);
  onTheme(paint);
  return (angles) => {
    last = angles;
    paint();
  };
}

export async function start() {
  const lab = await batch("attitude", CH.count);
  const opts = { ratio: 0.26, minHeight: 150, maxHeight: 230, xLabel: "t (s)" };
  const error = new Scope(document.getElementById("at-error"), opts);
  const roll = new Scope(document.getElementById("at-roll"), opts);
  const accel = new Scope(document.getElementById("at-accel"), { ...opts, ratio: 0.22 });
  const dials = ["truth", "gated", "window"].map((name) => ({
    name,
    draw: dial(document.getElementById(`at-dial-${name}`)),
    base: CH[name],
    canvas: document.getElementById(`at-dial-${name}`),
  }));
  const timeInput = document.getElementById("at-time");
  const playButton = document.getElementById("at-play");
  const heading = document.getElementById("at-heading");
  const gate = (v) => 10 ** v;

  const read = knobs(
    {
      "at-dyn": (v) => `${v.toFixed(1)}×`,
      "at-gate": (v) => gate(v).toFixed(1),
      "at-hold": (v) => `${v * 20} ms`,
      "at-window": (v) => `${Math.round(v * 100)} %`,
      "at-seed": (v) => String(v),
    },
    () => run(false),
  );

  let data = null;
  let plots = null;

  function showTime(redraw = true) {
    if (!data) return;
    const i = Math.min(Number(timeInput.value), data.n - 1);
    const t = data.channel(CH.time)[i];
    document.getElementById("at-time-value").textContent = `${num(t, 1)} s`;
    for (const d of dials) {
      const angles = [0, 1, 2].map((j) => data.channel(d.base + j)[i]);
      d.draw(angles);
      const text = `roll ${num(angles[0], 1)}°, pitch ${num(angles[1], 1)}°, heading ${num(angles[2], 0)}°`;
      d.canvas.setAttribute("aria-label", `${d.name}: ${text}`);
    }
    if (redraw) for (const [scope, plot] of plots) scope.draw({ ...plot, cursor: i, reveal: false });
  }

  function run(reveal) {
    const k = read();
    const result = lab.run(k["at-seed"], k["at-dyn"], gate(k["at-gate"]), k["at-hold"], k["at-window"], heading.checked ? 1 : 0);
    if (!result) return;
    // The jittered clock may stop a few samples short; those are NaN and dropped here.
    const times = result.channel(CH.time);
    const end = times.findIndex(Number.isNaN);
    const n = end < 0 ? result.n : end;
    const channel = (c) => result.channel(c).subarray(0, n);
    const { summary } = result;
    data = { n, channel };
    const t = channel(CH.time);
    timeInput.max = String(n - 1);
    const decision = channel(CH.gate);
    const acc = channel(CH.accel);
    const windowMoving = channel(CH.windowUsed).map((u, i) => (u && acc[i] > DYNAMIC ? 1 : 0));
    plots = [
      [error, {
        t, y: [0, Math.min(60, Math.max(4, Math.max(...channel(CH.error + 1).filter(Number.isFinite)) * 1.05))],
        layers: [
          { kind: "line", y: channel(CH.error + 1), color: "--c1", width: 1.5 },
          { kind: "line", y: channel(CH.error), color: "--c2", width: 2 },
        ],
      }],
      [roll, {
        t,
        layers: [
          { kind: "line", y: channel(CH.window), color: "--c1", width: 1.5 },
          { kind: "line", y: channel(CH.gated), color: "--c2", width: 1.75 },
          { kind: "line", y: channel(CH.truth), color: "--truth", dash: [6, 3], width: 1.75 },
        ],
      }],
      [accel, {
        t, y: [-1.2, 8],
        layers: [
          { kind: "line", y: acc, color: "--ink-soft", width: 1.5 },
          { kind: "events", mask: decision.map((d) => (d === 2 ? 1 : 0)), color: "--c2", row: 2 },
          { kind: "events", mask: decision.map((d) => (d === 3 ? 1 : 0)), color: "--c4", row: 1 },
          { kind: "events", mask: windowMoving, color: "--c1", row: 0 },
        ],
      }],
    ];
    for (const [scope, plot] of plots) scope.draw({ ...plot, reveal, cursor: Number(timeInput.value) });
    showTime(false);

    readout("at-r0", `${num(summary[0], 1)} / ${num(summary[2], 1)}`, "°");
    readout("at-r1", `${num(summary[1], 1)} / ${num(summary[3], 1)}`, "°");
    const pct = (v) => (Number.isFinite(v) ? `${num(100 * v, 1)} %` : "no motion");
    readout("at-skip", pct(summary[4]));
    readout("at-false", pct(summary[5]));
    readout("at-win", pct(summary[6]));
    readout("at-nis", num(summary[7]));
    document.getElementById("at-error").setAttribute(
      "aria-label",
      `Attitude error over time. Gated: RMS ${num(summary[0], 1)}°, largest ${num(summary[2], 1)}°. ` +
        `Window: RMS ${num(summary[1], 1)}°, largest ${num(summary[3], 1)}°.`,
    );
  }

  // Playback at 4× real time; never starts by itself.
  let playing = false, lastFrame = 0;
  function frame(now) {
    if (!playing) return;
    const steps = Math.max(1, Math.round(((now - lastFrame) / 1000) * 50 * 4));
    lastFrame = now;
    const next = Number(timeInput.value) + steps;
    timeInput.value = String(next >= data.n ? 0 : next);
    showTime();
    requestAnimationFrame(frame);
  }
  playButton.addEventListener("click", () => {
    playing = !playing;
    playButton.textContent = playing ? "Pause" : "Play";
    if (playing) {
      lastFrame = performance.now();
      requestAnimationFrame(frame);
    }
  });
  timeInput.addEventListener("input", () => showTime());
  heading.addEventListener("change", () => run(false));
  run(true);
}
