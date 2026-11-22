// Outliers: plain, NIS-gated and Student-t robust corrections on the same readings.
// Channels and summary: see `src/outliers.rs`.

import { Scope, batch, knobs, markBest, num, offset, readout, timeAxis } from "./lab.js";

const CH = { truth: 0, z: 1, outlier: 2, estimate: 3, sigma: 6, rejected: 9, weight: 10, count: 11 };
const COLORS = ["--c1", "--c2", "--c3"];

export async function start() {
  const lab = await batch("outliers", CH.count);
  const scope = new Scope(document.getElementById("out-plot"), { ratio: 0.45, xLabel: "t (s)" });
  const weights = new Scope(document.getElementById("out-weight"), { ratio: 0.16, minHeight: 120, maxHeight: 170, xLabel: "t (s)" });
  const read = knobs(
    {
      "out-rate": (v) => `${Math.round(v * 100)} %`,
      "out-size": (v) => `${v} σ`,
      "out-sigma": (v) => `${v.toFixed(2)} m`,
      "out-gate": (v) => v.toFixed(2),
      "out-dof": (v) => v.toFixed(1),
      "out-jump": (v) => `${v.toFixed(1)} m`,
      "out-seed": (v) => String(v),
    },
    () => run(false),
  );

  function run(reveal) {
    const k = read();
    const result = lab.run(k["out-seed"], k["out-sigma"], k["out-rate"], k["out-size"], k["out-gate"], k["out-dof"], k["out-jump"]);
    if (!result) return;
    const { n, channel, summary } = result;
    const t = timeAxis(n, 0.1);
    const truth = channel(CH.truth);
    const layers = [];
    for (let i = 0; i < 3; i++) {
      const x = channel(CH.estimate + i), sd = channel(CH.sigma + i);
      layers.push({ kind: "band", lo: offset(x, sd, -2), hi: offset(x, sd, 2), color: COLORS[i], fit: false });
    }
    layers.push(
      { kind: "dots", y: channel(CH.z), color: "--meas", r: 1.5, fit: false },
      { kind: "line", y: truth, color: "--truth", dash: [5, 4], width: 1.5 },
    );
    for (let i = 0; i < 3; i++) layers.push({ kind: "line", y: channel(CH.estimate + i), color: COLORS[i], width: i === 0 ? 1.5 : 2 });
    layers.push(
      { kind: "events", mask: channel(CH.outlier), color: "--meas", row: 1 },
      { kind: "events", mask: channel(CH.rejected), color: "--c2", row: 0 },
    );
    scope.draw({ t, layers, reveal });
    weights.draw({
      t,
      y: [0, 1.6],
      reveal,
      layers: [
        { kind: "hline", value: 1, color: "--grid-strong", dash: [2, 3] },
        { kind: "dots", y: channel(CH.weight), color: "--c3", r: 1.8 },
      ],
    });

    const rmse = [summary[0], summary[1], summary[2]];
    ["out-r0", "out-r1", "out-r2"].forEach((id, i) => readout(id, num(rmse[i]), "m"));
    markBest(["out-r0", "out-r1", "out-r2"], rmse);
    readout("out-rz", num(summary[3]), "m");
    readout("out-rej", `${summary[5]} / ${summary[4]}`);
    readout("out-w", `${num(summary[7])} / ${num(summary[8])}`);
    const caught = summary[6], falseAlarms = summary[5] - summary[6];
    document.getElementById("out-plot").setAttribute(
      "aria-label",
      `Position over time. RMS error: plain ${num(rmse[0])} m, gated ${num(rmse[1])} m, robust ${num(rmse[2])} m. ` +
        `The gate rejected ${summary[5]} readings: ${caught} outliers and ${falseAlarms} good ones.`,
    );
  }

  run(true);
}
