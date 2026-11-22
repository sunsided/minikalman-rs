// Shared glue of the lab page: loading the module, reading a run, plotting, knobs, theme.
//
// The module exports plain `extern "C"` functions. Each batch experiment writes its run to a
// trace buffer (channel-major `f32`) and a summary buffer, which are read here through
// `trace_ptr()` and `summary_ptr()`. No generated glue, no bundler.

let modulePromise;

/** Compiles the module once; every experiment gets its own instance of it. */
export function compile() {
  // Revalidate, so a rebuilt or redeployed module is never served from a stale cache.
  modulePromise ??= WebAssembly.compileStreaming(fetch("minikalman_wasm.wasm", { cache: "no-cache" }));
  return modulePromise;
}

/** Returns the exports of a fresh instance. */
export async function instantiate() {
  return (await WebAssembly.instantiate(await compile(), {})).exports;
}

/**
 * A batch experiment backed by its own instance. `run(...args)` calls `<name>_run` and
 * returns `{ n, channel(c), summary }`, or `null` if the module trapped, in which case the
 * instance is replaced so the next run starts clean.
 */
export async function batch(name, channels) {
  const module = await compile();
  let exports = new WebAssembly.Instance(module, {}).exports;
  return {
    get exports() {
      return exports;
    },
    run(...args) {
      try {
        const n = exports[`${name}_run`](...args);
        const memory = exports.memory.buffer;
        const trace = new Float32Array(memory, exports.trace_ptr(), channels * n).slice();
        const summary = new Float32Array(memory, exports.summary_ptr(), 64).slice();
        return { n, channel: (c) => trace.subarray(c * n, (c + 1) * n), summary };
      } catch (error) {
        console.warn(`${name}_run trapped; restarting its module`, error);
        exports = new WebAssembly.Instance(module, {}).exports;
        return null;
      }
    },
  };
}

// ── Formatting ──

/** Formats a measured number; `—` for missing values, `diverged` for infinities. */
export function num(value, digits = 2) {
  if (Number.isNaN(value) || value === undefined || value === null) return "—";
  if (!Number.isFinite(value)) return "diverged";
  const abs = Math.abs(value);
  if (abs !== 0 && (abs < 10 ** -digits || abs >= 1e5)) return value.toExponential(1);
  return value.toFixed(digits);
}

/** Sets a readout's value and its unit. */
export function readout(id, value, unit = "") {
  const dd = document.querySelector(`#${id} dd`);
  if (!dd) return;
  dd.textContent = value;
  if (unit) {
    const small = document.createElement("small");
    small.textContent = ` ${unit}`;
    dd.append(small);
  }
}

/** Marks the readout with the smallest finite value among `ids` as the best one. */
export function markBest(ids, values) {
  let best = -1;
  values.forEach((v, i) => {
    if (Number.isFinite(v) && (best < 0 || v < values[best])) best = i;
  });
  ids.forEach((id, i) => document.getElementById(id)?.classList.toggle("best", i === best));
}

// ── Knobs ──

/**
 * Wires range inputs to their `<output>`s and calls `onChange` once per frame while they move.
 * `formats` maps an input id to a function from its numeric value to the shown text.
 * Returns a function that reads all values as `{ id: number }`.
 */
export function knobs(formats, onChange) {
  let pending = false;
  const read = () => Object.fromEntries(Object.keys(formats).map((id) => [id, Number(document.getElementById(id).value)]));
  const update = () => {
    for (const [id, format] of Object.entries(formats)) {
      const input = document.getElementById(id);
      const text = format(Number(input.value));
      const output = document.querySelector(`output[for="${id}"]`);
      if (output) output.textContent = text;
      input.setAttribute("aria-valuetext", text);
    }
  };
  for (const id of Object.keys(formats)) {
    document.getElementById(id).addEventListener("input", () => {
      update();
      if (pending) return;
      pending = true;
      requestAnimationFrame(() => {
        pending = false;
        onChange(read());
      });
    });
  }
  update();
  return read;
}

// ── Theme and motion ──

export const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
const listeners = new Set();

/** Calls `fn` whenever the colours change (theme toggle or system scheme). */
export function onTheme(fn) {
  listeners.add(fn);
}

function themeChanged() {
  for (const fn of listeners) fn();
}

matchMedia("(prefers-color-scheme: dark)").addEventListener("change", themeChanged);

/** Wires the theme toggle button. */
export function themeToggle(button) {
  const effective = () =>
    document.documentElement.dataset.theme ?? (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const label = () => {
    const next = effective() === "dark" ? "light" : "dark";
    button.textContent = `Switch to ${next}`;
  };
  button.addEventListener("click", () => {
    const next = effective() === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem("theme", next);
    } catch {
      // Storage may be blocked; the choice then lasts for this page view.
    }
    label();
    themeChanged();
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", label);
  label();
}

/** Reads a CSS custom property, e.g. `color("--c1")`. */
export function color(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
}

// ── Scope: a plot on a gridded screen ──

/** Returns about `count` round tick values covering `[lo, hi]`. */
function ticks(lo, hi, count) {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const raw = span / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}

const tickText = (v) => {
  const abs = Math.abs(v);
  if (abs !== 0 && (abs < 1e-3 || abs >= 1e5)) return v.toExponential(0);
  return String(+v.toPrecision(4));
};

/**
 * A time plot on a canvas inside a `.scope` element.
 *
 * `draw({ t, layers, y, yLog, cursor, reveal })` plots `layers` over the shared abscissa `t`:
 * - `{ kind: "line", y, color, width, dash }`
 * - `{ kind: "band", lo, hi, color }` (a translucent ±σ band)
 * - `{ kind: "dots", y, color, mask, r }`; dots outside the range show as carets at the edge
 * - `{ kind: "events", mask, color, row }`: ticks along the bottom where `mask` is set
 * - `{ kind: "hline", value, color, dash, label }`
 * Layers with `fit: false` do not widen the automatic range. `y` fixes the range.
 */
export class Scope {
  constructor(canvas, { ratio = 0.42, minHeight = 190, maxHeight = 420, xLabel = "", yLabel = "" } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.options = { ratio, minHeight, maxHeight, xLabel, yLabel };
    this.last = null;
    this.progress = 1;
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    onTheme(() => this.redraw());
    this.resize();
  }

  resize() {
    const width = this.canvas.parentElement.clientWidth;
    if (!width) return;
    const { ratio, minHeight, maxHeight } = this.options;
    const height = Math.round(Math.min(maxHeight, Math.max(minHeight, width * ratio)));
    const dpr = window.devicePixelRatio || 1;
    this.canvas.style.height = `${height}px`;
    this.canvas.width = Math.round(width * dpr);
    this.canvas.height = Math.round(height * dpr);
    this.size = { width, height, dpr };
    this.redraw();
  }

  redraw() {
    if (this.last) this.paint(this.last, this.progress);
  }

  draw(plot) {
    this.last = plot;
    cancelAnimationFrame(this.animation);
    if (!plot.reveal || reducedMotion.matches) {
      this.progress = 1;
      this.paint(plot, 1);
      return;
    }
    // One sweep from left to right, like a scope's trace, when a run is new.
    const start = performance.now();
    const step = (now) => {
      this.progress = Math.min(1, (now - start) / 650);
      this.paint(plot, this.progress);
      if (this.progress < 1) this.animation = requestAnimationFrame(step);
    };
    this.animation = requestAnimationFrame(step);
  }

  /** Maps a pointer event to `{ x, y }` in data units of the last paint. */
  toData(event) {
    const rect = this.canvas.getBoundingClientRect();
    const { margin, w, h, x0, x1, y0, y1 } = this.layout;
    const fx = (event.clientX - rect.left - margin.left) / w;
    const fy = (event.clientY - rect.top - margin.top) / h;
    return { x: x0 + fx * (x1 - x0), y: y1 - fy * (y1 - y0) };
  }

  range(plot) {
    if (plot.y) return plot.y;
    let lo = Infinity, hi = -Infinity;
    const take = (v) => {
      const w = plot.yLog ? Math.log10(v) : v;
      if (Number.isFinite(w)) {
        lo = Math.min(lo, w);
        hi = Math.max(hi, w);
      }
    };
    for (const layer of plot.layers) {
      if (layer.fit === false) continue;
      if (layer.kind === "line" || layer.kind === "dots") layer.y.forEach(take);
      if (layer.kind === "band") {
        layer.lo.forEach(take);
        layer.hi.forEach(take);
      }
      if (layer.kind === "hline") take(layer.value);
    }
    if (!(hi > lo)) [lo, hi] = Number.isFinite(lo) ? [lo - 1, lo + 1] : [0, 1];
    const pad = (hi - lo) * 0.06;
    return [lo - pad, hi + pad];
  }

  paint(plot, progress) {
    if (!this.size) return;
    const { ctx } = this;
    const { width, height, dpr } = this.size;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.font = "11px 'Recursive Variable', ui-sans-serif, system-ui, sans-serif";

    const margin = { left: 46, right: 10, top: 26, bottom: 24 };
    const t = plot.t;
    const n = t.length;
    const [x0, x1] = [t[0], t[n - 1]];
    const [y0, y1] = this.range(plot);
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;
    const px = (x) => margin.left + ((x - x0) / (x1 - x0 || 1)) * w;
    const py = (v) => margin.top + (1 - ((plot.yLog ? Math.log10(v) : v) - y0) / (y1 - y0)) * h;
    const visible = Math.max(1, Math.round(progress * n));
    this.layout = { margin, w, h, x0, x1, y0, y1 };

    // Grid: minor and major lines, labels on the left and bottom.
    ctx.lineWidth = 1;
    ctx.fillStyle = color("--ink-soft");
    ctx.textBaseline = "middle";
    ctx.textAlign = "right";
    for (const v of ticks(y0, y1, Math.max(2, Math.round(h / 45)))) {
      const y = margin.top + (1 - (v - y0) / (y1 - y0)) * h;
      ctx.strokeStyle = color(v === 0 ? "--grid-strong" : "--grid");
      ctx.beginPath();
      ctx.moveTo(margin.left, Math.round(y) + 0.5);
      ctx.lineTo(width - margin.right, Math.round(y) + 0.5);
      ctx.stroke();
      ctx.fillText(plot.yLog ? `1e${v}` : tickText(v), margin.left - 6, y);
    }
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const v of ticks(x0, x1, Math.max(2, Math.round(w / 90)))) {
      const x = px(v);
      ctx.strokeStyle = color("--grid");
      ctx.beginPath();
      ctx.moveTo(Math.round(x) + 0.5, margin.top);
      ctx.lineTo(Math.round(x) + 0.5, margin.top + h);
      ctx.stroke();
      ctx.fillText(tickText(v), x, margin.top + h + 6);
    }
    ctx.textAlign = "right";
    if (this.options.xLabel) ctx.fillText(this.options.xLabel, width - margin.right, margin.top + h + 6);
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    if (this.options.yLabel) ctx.fillText(this.options.yLabel, 4, 14);

    ctx.save();
    ctx.beginPath();
    ctx.rect(margin.left, margin.top - 2, w, h + 4);
    ctx.clip();

    for (const layer of plot.layers) {
      const stroke = color(layer.color);
      ctx.strokeStyle = stroke;
      ctx.fillStyle = stroke;
      if (layer.kind === "band") {
        ctx.globalAlpha = 0.16;
        ctx.beginPath();
        let open = false;
        const back = [];
        for (let i = 0; i < visible; i++) {
          const a = py(layer.hi[i]), b = py(layer.lo[i]);
          if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
          if (open) ctx.lineTo(px(t[i]), a);
          else ctx.moveTo(px(t[i]), a);
          open = true;
          back.push([px(t[i]), b]);
        }
        for (let i = back.length - 1; i >= 0; i--) ctx.lineTo(...back[i]);
        ctx.fill();
        ctx.globalAlpha = 1;
      } else if (layer.kind === "line") {
        ctx.lineWidth = layer.width ?? 1.75;
        ctx.setLineDash(layer.dash ?? []);
        ctx.lineJoin = "round";
        ctx.beginPath();
        let open = false;
        for (let i = 0; i < visible; i++) {
          const y = py(layer.y[i]);
          if (!Number.isFinite(y)) {
            open = false;
            continue;
          }
          const clamped = Math.max(-1e4, Math.min(1e4, y));
          if (open) ctx.lineTo(px(t[i]), clamped);
          else ctx.moveTo(px(t[i]), clamped);
          open = true;
        }
        ctx.stroke();
        ctx.setLineDash([]);
      } else if (layer.kind === "dots") {
        const r = layer.r ?? 1.6;
        for (let i = 0; i < visible; i++) {
          if (layer.mask && !layer.mask[i]) continue;
          const y = py(layer.y[i]);
          if (!Number.isFinite(y)) continue;
          const x = px(t[i]);
          if (y < margin.top) caret(ctx, x, margin.top + 1, 1);
          else if (y > margin.top + h) caret(ctx, x, margin.top + h - 1, -1);
          else {
            ctx.beginPath();
            ctx.arc(x, y, r, 0, 2 * Math.PI);
            ctx.fill();
          }
        }
      } else if (layer.kind === "events") {
        const row = layer.row ?? 0;
        const base = margin.top + h - 2 - row * 9;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        for (let i = 0; i < visible; i++) {
          if (!layer.mask[i]) continue;
          const x = Math.round(px(t[i])) + 0.5;
          ctx.moveTo(x, base);
          ctx.lineTo(x, base - 6);
        }
        ctx.stroke();
      } else if (layer.kind === "hline") {
        const y = Math.round(py(layer.value)) + 0.5;
        ctx.lineWidth = 1;
        ctx.setLineDash(layer.dash ?? [4, 4]);
        ctx.beginPath();
        ctx.moveTo(margin.left, y);
        ctx.lineTo(margin.left + w, y);
        ctx.stroke();
        ctx.setLineDash([]);
        if (layer.label) {
          ctx.textAlign = "right";
          ctx.fillText(layer.label, margin.left + w - 4, y - 4);
          ctx.textAlign = "left";
        }
      }
    }

    if (plot.cursor !== undefined && plot.cursor >= 0 && plot.cursor < n) {
      const x = Math.round(px(t[plot.cursor])) + 0.5;
      ctx.strokeStyle = color("--ink");
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, margin.top);
      ctx.lineTo(x, margin.top + h);
      ctx.stroke();
    }
    ctx.restore();

    ctx.strokeStyle = color("--grid-strong");
    ctx.lineWidth = 1;
    ctx.strokeRect(margin.left + 0.5, margin.top + 0.5, w - 1, h - 1);
  }
}

/** A small caret at the plot edge for a dot outside the range; `dir` 1 points up. */
function caret(ctx, x, y, dir) {
  ctx.beginPath();
  ctx.moveTo(x, y - dir * 0);
  ctx.lineTo(x - 3, y + dir * 5);
  ctx.lineTo(x + 3, y + dir * 5);
  ctx.closePath();
  ctx.fill();
}

/** Returns `[0, dt, 2·dt, …]` with `n` entries. */
export function timeAxis(n, dt) {
  return Float32Array.from({ length: n }, (_, i) => i * dt);
}

/** Returns `a + k·b` for each entry, e.g. `±2σ` bands. */
export function offset(a, b, k) {
  return a.map((v, i) => v + k * b[i]);
}
