// Smoothing: the filter, a fixed-lag smoother and the RTS smoother through a gap.
// Channels and summary: see `src/smoothing.rs`.

import { Scope, batch, knobs, markBest, num, offset, readout, timeAxis } from "./lab.js";

const CH = { truth: 0, z: 1, filter: 2, lag: 4, rts: 6, count: 8 };
const GAP_START = 10; // seconds

export async function start() {
  const lab = await batch("smoothing", CH.count);
  const scope = new Scope(document.getElementById("sm-plot"), { ratio: 0.45, xLabel: "t (s)" });
  const errors = new Scope(document.getElementById("sm-error"), { ratio: 0.2, minHeight: 140, maxHeight: 210, xLabel: "t (s)" });
  // The process noise knob is logarithmic.
  const q = (v) => 10 ** v;
  const read = knobs(
    {
      "sm-gap": (v) => `${v.toFixed(1)} s`,
      "sm-sigma": (v) => `${v.toFixed(1)} m`,
      "sm-q": (v) => q(v).toPrecision(2),
      "sm-seed": (v) => String(v),
    },
    () => run(false),
  );

  function run(reveal) {
    const k = read();
    const result = lab.run(k["sm-seed"], k["sm-sigma"], q(k["sm-q"]), k["sm-gap"]);
    if (!result) return;
    const { n, channel, summary } = result;
    const t = timeAxis(n, 0.1);
    const truth = channel(CH.truth);
    const est = (c) => channel(c), sd = (c) => channel(c + 1);
    const gap = k["sm-gap"] > 0 ? [{ kind: "span", from: GAP_START, to: GAP_START + k["sm-gap"], color: "--ink-soft", label: "no readings" }] : [];
    scope.draw({
      t,
      reveal,
      layers: [
        ...gap,
        { kind: "band", lo: offset(est(CH.filter), sd(CH.filter), -2), hi: offset(est(CH.filter), sd(CH.filter), 2), color: "--c1", fit: false },
        { kind: "band", lo: offset(est(CH.rts), sd(CH.rts), -2), hi: offset(est(CH.rts), sd(CH.rts), 2), color: "--c3", fit: false },
        { kind: "dots", y: channel(CH.z), color: "--meas", r: 1.5, fit: false },
        { kind: "line", y: truth, color: "--truth", dash: [5, 4], width: 1.5 },
        { kind: "line", y: est(CH.filter), color: "--c1", width: 1.75 },
        { kind: "line", y: est(CH.lag), color: "--c4", width: 1.5 },
        { kind: "line", y: est(CH.rts), color: "--c3", width: 2.25 },
      ],
    });
    const diff = (c) => est(c).map((v, i) => v - truth[i]);
    errors.draw({
      t,
      reveal,
      layers: [
        ...gap.map((g) => ({ ...g, label: undefined })),
        { kind: "hline", value: 0, color: "--grid-strong", dash: [] },
        { kind: "line", y: diff(CH.filter), color: "--c1", width: 1.5 },
        { kind: "line", y: diff(CH.lag), color: "--c4", width: 1.5 },
        { kind: "line", y: diff(CH.rts), color: "--c3", width: 2 },
      ],
    });

    const rmse = [summary[0], summary[1], summary[2]];
    ["sm-r0", "sm-r1", "sm-r2"].forEach((id, i) => readout(id, num(rmse[i]), "m"));
    markBest(["sm-r0", "sm-r1", "sm-r2"], rmse);
    const hasGap = k["sm-gap"] > 0;
    readout("sm-g0", hasGap ? num(summary[3]) : "—", hasGap ? "m" : "");
    readout("sm-g1", hasGap ? num(summary[4]) : "—", hasGap ? "m" : "");
    readout("sm-s", hasGap ? `${num(summary[5])} / ${num(summary[6])}` : "—", hasGap ? "m" : "");
    document.getElementById("sm-plot").setAttribute(
      "aria-label",
      `Position over time. RMS error: filter ${num(rmse[0])} m, fixed lag ${num(rmse[1])} m, RTS ${num(rmse[2])} m.`,
    );
  }

  run(true);
}
