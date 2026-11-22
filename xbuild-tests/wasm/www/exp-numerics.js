// Numerics: standard vs Joseph covariance update in f32, I16F16 and I32F32.
// Channels and summary: see `src/numerics.rs`.

import { Scope, batch, knobs, num, timeAxis } from "./lab.js";

const PER_VARIANT = 4;
const CH = { asym: 0, diag: 1, pd: 2, error: 3 };
const VARIANTS = [
  { name: "f32 standard", color: "--c2", dash: [] },
  { name: "f32 Joseph", color: "--c2", dash: [6, 4] },
  { name: "I16F16 standard", color: "--c1", dash: [] },
  { name: "I16F16 Joseph", color: "--c1", dash: [6, 4] },
  { name: "I32F32 standard", color: "--c3", dash: [] },
  { name: "I32F32 Joseph", color: "--c3", dash: [6, 4] },
];
const LSB = 2 ** -16;

const sup = (v) => {
  const digits = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹", ".": "·" };
  return String(v).split("").map((c) => digits[c] ?? c).join("");
};
const power = (v) => (Number.isInteger(v) ? `10${sup(v)}` : (10 ** v).toExponential(1));

export async function start() {
  const lab = await batch("numerics", PER_VARIANT * VARIANTS.length);
  const diag = new Scope(document.getElementById("nu-diag"), { ratio: 0.32, xLabel: "step" });
  const asym = new Scope(document.getElementById("nu-asym"), { ratio: 0.6, minHeight: 170, maxHeight: 260, xLabel: "step" });
  const error = new Scope(document.getElementById("nu-error"), { ratio: 0.6, minHeight: 170, maxHeight: 260, xLabel: "step" });
  const read = knobs(
    {
      "nu-r": power,
      "nu-q": power,
      "nu-sensors": (v) => String(v),
      "nu-steps": (v) => String(v),
      "nu-seed": (v) => String(v),
    },
    () => run(false),
  );

  function run(reveal) {
    const k = read();
    const result = lab.run(k["nu-seed"], k["nu-sensors"], k["nu-r"], k["nu-q"], k["nu-steps"]);
    if (!result) return;
    const { n, channel, summary } = result;
    const t = timeAxis(n, 1);
    const ch = (v, c) => channel(v * PER_VARIANT + c);
    const lines = (c, extra = {}) =>
      VARIANTS.map((v, i) => ({ kind: "line", y: ch(i, c), color: v.color, dash: v.dash, width: i === 3 || i === 2 ? 2 : 1.5, ...extra }));
    const notPd = (i) => ch(i, CH.pd).map((v) => (v ? 0 : 1));

    diag.draw({
      t,
      yLog: true,
      reveal,
      layers: [
        { kind: "hline", value: LSB, color: "--c1", label: "one Q16.16 step, 2⁻¹⁶" },
        ...lines(CH.diag),
        { kind: "events", mask: notPd(2), color: "--c1", row: 1 },
        { kind: "events", mask: notPd(3), color: "--c1", row: 0 },
      ],
    });
    // Exactly symmetric runs have no point on a log scale; they sit on the floor instead.
    const floor = 1e-10;
    asym.draw({
      t,
      yLog: true,
      y: [-10.3, 0.3],
      reveal,
      layers: VARIANTS.map((v, i) => ({
        kind: "line",
        y: ch(i, CH.asym).map((a) => (a > 0 ? Math.max(a, floor) : floor)),
        color: v.color,
        dash: v.dash,
        width: 1.5,
      })),
    });
    error.draw({ t, yLog: true, reveal, layers: lines(CH.error) });

    const rows = VARIANTS.map((v, i) => {
      const [bad, first, worst, final] = summary.slice(4 * i, 4 * i + 4);
      const badText = bad ? `${bad} of ${n} steps, from ${first}` : "never";
      return `<tr><th scope="row" style="--color: var(${v.color})">${v.name}</th>` +
        `<td class="${bad ? "bad" : ""}">${badText}</td>` +
        `<td>${worst === 0 ? "exactly 0" : num(worst)}</td>` +
        `<td class="${final > 1 ? "bad" : ""}">${num(final, 3)} m</td></tr>`;
    });
    document.querySelector("#nu-table tbody").innerHTML = rows.join("");
  }

  run(true);
}
