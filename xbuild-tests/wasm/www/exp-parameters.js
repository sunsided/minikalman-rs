// Parameter estimation: dual RLS + KF vs a joint augmented-state EKF.
// Channels and summary: see `src/parameters.rs`.

import { Scope, batch, knobs, markBest, num, offset, readout, timeAxis } from "./lab.js";

const CH = { speed: 0, z: 1, theta: 2, joint: 4, dual: 9, count: 15 };
const CHANGE_AT = 400;

export async function start(section) {
  const lab = await batch("parameters", CH.count, section);
  const opts = { ratio: 0.62, minHeight: 180, maxHeight: 280, xLabel: "step" };
  const a = new Scope(document.getElementById("pa-a"), opts);
  const b = new Scope(document.getElementById("pa-b"), opts);
  const x = new Scope(document.getElementById("pa-x"), { ratio: 0.22, minHeight: 140, maxHeight: 220, xLabel: "step" });
  const q = (v) => 10 ** v;
  const read = knobs(
    {
      "pa-q": (v) => q(v).toExponential(0),
      "pa-mu": (v) => v.toFixed(3),
      "pa-sigma": (v) => v.toFixed(2),
      "pa-seed": (v) => String(v),
    },
    () => run(false),
  );

  function run(reveal) {
    const k = read();
    const result = lab.run(k["pa-seed"], k["pa-sigma"], k["pa-q"], k["pa-mu"]);
    if (!result) return;
    const { n, channel, summary } = result;
    const t = timeAxis(n, 1);
    const change = { kind: "span", from: CHANGE_AT - 1, to: CHANGE_AT + 1, color: "--ink-soft" };
    const param = (i, range) => ({
      t,
      reveal,
      y: range,
      layers: [
        change,
        ...[["--c2", CH.dual], ["--c3", CH.joint]].map(([color, base]) => {
          const est = channel(base + 1 + i), sd = channel(base + 3 + i);
          return { kind: "band", lo: offset(est, sd, -2), hi: offset(est, sd, 2), color, fit: false };
        }),
        { kind: "line", y: channel(CH.theta + i), color: "--truth", dash: [5, 4], width: 1.5 },
        { kind: "line", y: channel(CH.dual + 1 + i), color: "--c2", width: 1.75 },
        { kind: "line", y: channel(CH.joint + 1 + i), color: "--c3", width: 1.75 },
      ],
    });
    a.draw(param(0, [0.4, 1.1]));
    b.draw(param(1, [0.2, 1.4]));
    const speed = channel(CH.speed);
    const err = (base) => channel(base).map((v, i) => v - speed[i]);
    x.draw({
      t,
      reveal,
      layers: [
        change,
        { kind: "hline", value: 0, color: "--grid-strong", dash: [] },
        { kind: "line", y: err(CH.dual), color: "--c2", width: 1.25 },
        { kind: "line", y: err(CH.joint), color: "--c3", width: 1.25 },
      ],
    });

    const steps = (v) => (v < 0 ? "never" : String(v));
    readout("pa-re0", steps(summary[7]), summary[7] < 0 ? "" : "steps");
    readout("pa-re1", steps(summary[6]), summary[6] < 0 ? "" : "steps");
    const re = [summary[7], summary[6]].map((v) => (v < 0 ? Infinity : v));
    markBest(["pa-re0", "pa-re1"], re);
    readout("pa-a0", `${num(summary[3], 3)}, ${num(summary[5], 3)}`);
    readout("pa-a1", `${num(summary[2], 3)}, ${num(summary[4], 3)}`);
    readout("pa-x0", `${num(summary[9], 3)} / ${num(summary[1], 3)}`);
    readout("pa-x1", `${num(summary[8], 3)} / ${num(summary[0], 3)}`);
    document.getElementById("pa-a").setAttribute(
      "aria-label",
      `Estimates of a over time. After the change at step ${CHANGE_AT}, the joint filter re-identifies the plant ` +
        `in ${steps(summary[6])} steps, the dual estimator in ${steps(summary[7])}.`,
    );
  }

  run(true);
}
