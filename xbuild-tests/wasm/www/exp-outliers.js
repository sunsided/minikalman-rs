// Outliers: plain, NIS-gated and Student-t robust corrections on the same readings.
// Channels and summary: see `src/outliers.rs`.

import { Scope, batch, knobs, markBest, num, offset, readout, timeAxis } from "./lab.js";

const CH = { truth: 0, z: 1, outlier: 2, estimate: 3, sigma: 6, rejected: 9, weight: 10, nis: 11, reacquired: 12, count: 13 };
const COLORS = ["--c1", "--c2", "--c3"];
const JUMP_AT = 15; // seconds

export async function start(section) {
  const lab = await batch("outliers", CH.count, section);
  const scope = new Scope(document.getElementById("out-plot"), { ratio: 0.45, xLabel: "t (s)" });
  const nis = new Scope(document.getElementById("out-nis"), { ratio: 0.2, minHeight: 140, maxHeight: 200, xLabel: "t (s)" });
  const weights = new Scope(document.getElementById("out-weight"), { ratio: 0.16, minHeight: 120, maxHeight: 170, xLabel: "t (s)" });
  const read = knobs(
    {
      "out-rate": (v) => `${Math.round(v * 100)} %`,
      "out-size": (v) => `${v} σ`,
      "out-sigma": (v) => `${v.toFixed(2)} m`,
      "out-gate": (v) => v.toFixed(2),
      "out-dof": (v) => v.toFixed(1),
      "out-jump": (v) => `${v.toFixed(1)} m`,
      "out-reacquire": (v) => (v === 0 ? "off" : `${v} in a row`),
      "out-seed": (v) => String(v),
    },
    () => run(false),
  );

  function run(reveal) {
    const k = read();
    const result = lab.run(
      k["out-seed"], k["out-sigma"], k["out-rate"], k["out-size"], k["out-gate"], k["out-dof"], k["out-jump"], k["out-reacquire"],
    );
    if (!result) return;
    const { n, channel, summary } = result;
    const t = timeAxis(n, 0.1);
    const truth = channel(CH.truth);
    const rejected = channel(CH.rejected);
    const jump = k["out-jump"] > 0 ? [{ kind: "span", from: JUMP_AT - 0.05, to: JUMP_AT + 0.05, color: "--ink", label: "" }] : [];

    // The range follows the truth only, so neither outliers nor a filter that lost the
    // target squash the plot; anything outside shows at the edge.
    const layers = [...jump];
    for (let i = 0; i < 3; i++) {
      const x = channel(CH.estimate + i), sd = channel(CH.sigma + i);
      layers.push({ kind: "band", lo: offset(x, sd, -2), hi: offset(x, sd, 2), color: COLORS[i], fit: false });
    }
    layers.push(
      { kind: "dots", y: channel(CH.z), color: "--meas", r: 1.6, fit: false },
      { kind: "line", y: truth, color: "--truth", dash: [6, 3], width: 2 },
    );
    const NAMES = ["plain", "gated", "robust"];
    for (let i = 0; i < 3; i++) {
      layers.push({ kind: "line", y: channel(CH.estimate + i), color: COLORS[i], width: i === 0 ? 1.5 : 2, fit: false, name: NAMES[i] });
    }
    layers.push(
      { kind: "events", mask: channel(CH.outlier), color: "--meas", row: 1 },
      { kind: "events", mask: rejected, color: "--c2", row: 0 },
    );
    scope.draw({ t, layers, reveal });

    const nisValues = channel(CH.nis);
    nis.draw({
      t,
      yLog: true,
      reveal,
      y: [-3, Math.max(3, Math.log10(Math.max(...nisValues.filter(Number.isFinite), 10)) + 0.3)],
      layers: [
        ...jump,
        { kind: "hline", value: k["out-gate"], color: "--c2", label: "gate" },
        { kind: "dots", y: nisValues, mask: rejected.map((r) => 1 - r), color: "--c2", r: 1.6 },
        { kind: "dots", y: nisValues, mask: rejected, color: "--c1", r: 1.8 },
        { kind: "events", mask: channel(CH.reacquired), color: "--c4", row: 0 },
      ],
    });
    weights.draw({
      t,
      y: [0, 1.6],
      reveal,
      layers: [
        ...jump,
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
    readout("out-re", k["out-reacquire"] === 0 ? "off" : String(summary[9]));
    const caught = summary[6], falseAlarms = summary[5] - summary[6];
    document.getElementById("out-plot").setAttribute(
      "aria-label",
      `Position over time. RMS error: plain ${num(rmse[0])} m, gated ${num(rmse[1])} m, robust ${num(rmse[2])} m. ` +
        `The gate rejected ${summary[5]} readings: ${caught} outliers and ${falseAlarms} good ones.`,
    );
  }

  run(true);
}
