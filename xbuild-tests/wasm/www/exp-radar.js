// EKF vs UKF: radars measure range and bearing to a moving target; both filters run on the
// same measurements. Uses the step-by-step exports `radar_reset`, `radar_predict`,
// `radar_correct`, `radar_set_wrap`, `radar_state` and `radar_covariance`.

import { color, instantiate, reducedMotion } from "./lab.js";

/** Box-Muller transform for normally distributed noise. */
function gaussian() {
  const u = 1 - Math.random();
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function polyline(ctx, points, stroke, width = 2) {
  ctx.strokeStyle = stroke;
  ctx.lineWidth = width;
  ctx.beginPath();
  points.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.stroke();
}

/** Sizes a canvas to its CSS width at the device pixel ratio, keeping the 16:9 world window. */
function canvasSize(canvas) {
  const width = canvas.clientWidth || 640;
  const height = Math.round((width * 360) / 640);
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
  }
  return { width, height, dpr };
}

export async function start() {
  const kf = await instantiate();
  const EKF = 0, UKF = 1;
  const variants = [
    { id: EKF, name: "EKF", canvas: document.getElementById("radar-ekf"), color: "--c2" },
    { id: UKF, name: "UKF", canvas: document.getElementById("radar-ukf"), color: "--c3" },
  ];
  const radars = [
    { x: 0, y: 0, color: "--ink", crescent: "--meas" },
    { x: 30, y: -17, color: "--c4", crescent: "--c4" },
  ];

  // World window in metres, centred on the first radar's starting position.
  const view = { xMin: -45, xMax: 45, yMin: -25.3, yMax: 25.3 };
  const dt = 0.1;
  const crescentSamples = 300;
  const trail = 450; // steps of history to draw
  const minRange = 1; // a radar cannot measure a target on top of itself
  const grabRadius = 2.5; // metres
  const drawSpacing = 3; // metres between waypoints of a drawn path

  const $ = (id) => document.getElementById(id);
  const inputs = {
    bearing: $("sigma-bearing"), range: $("sigma-range"), speed: $("speed"),
    processNoise: $("process-noise"),
    measureEvery: $("measure-every"), wrap: $("wrap"), secondRadar: $("second-radar"),
    preset: $("preset"),
  };
  const hints = {
    edit: "Drag a radar or a waypoint of the path (the small circles). The filters keep running.",
    draw: "Press and drag on either plot to draw a new path. End near the start to close the loop.",
    steer: "Move the pointer over a plot; the target heads for it at the set speed.",
  };

  // ── Paths: polylines with constant-speed travel along them ──

  function makePath(points, closed) {
    const cum = [0];
    const n = closed ? points.length : points.length - 1;
    for (let i = 0; i < n; i++) {
      const [ax, ay] = points[i], [bx, by] = points[(i + 1) % points.length];
      cum.push(cum[i] + Math.hypot(bx - ax, by - ay));
    }
    return { points, closed, cum, length: cum[cum.length - 1] };
  }

  function pointAt(path, s) {
    if (path.closed) s = ((s % path.length) + path.length) % path.length;
    else s = Math.min(Math.max(s, 0), path.length);
    let i = 1;
    while (i < path.cum.length - 1 && path.cum[i] < s) i++;
    const [ax, ay] = path.points[i - 1], [bx, by] = path.points[i % path.points.length];
    const seg = path.cum[i] - path.cum[i - 1] || 1;
    const f = (s - path.cum[i - 1]) / seg;
    return { x: ax + f * (bx - ax), y: ay + f * (by - ay) };
  }

  const sampled = (n, f) => Array.from({ length: n }, (_, i) => f((2 * Math.PI * i) / n));
  const presets = {
    circle: () => makePath(sampled(24, (p) => [4 + 28 * Math.cos(p), 16 * Math.sin(p)]), true),
    straight: () => makePath([[-42, -20], [42, 18]], false),
    eight: () => makePath(sampled(32, (p) => {
      const d = 1 + Math.sin(p) ** 2;
      return [36 * Math.cos(p) / d, 6 + 30 * Math.sin(p) * Math.cos(p) / d];
    }), true),
  };

  // ── Simulation state ──

  let path = presets.circle();
  let tool = "edit";
  let paused = false;
  let sim;
  let pointer = null; // world position of the pointer over a plot, for steering
  let drawing = null; // points collected while drawing
  let dragging = null; // { kind: "radar" | "waypoint", index }

  function activeRadars() {
    return inputs.secondRadar.checked ? radars : radars.slice(0, 1);
  }

  function updateLabels() {
    const b = Number(inputs.bearing.value), r = Number(inputs.range.value);
    const every = Number(inputs.measureEvery.value);
    $("sigma-bearing-value").textContent = `${b.toFixed(2)} rad (${(b * 180 / Math.PI).toFixed(1)}°)`;
    $("sigma-range-value").textContent = `${r.toFixed(1)} m`;
    $("speed-value").textContent = `${Number(inputs.speed.value).toFixed(1)} m/s`;
    $("measure-every-value").textContent = `${every} step${every > 1 ? "s" : ""} (${(every * dt).toFixed(1)} s)`;
    $("process-noise-value").textContent = `${Number(inputs.processNoise.value).toFixed(1)} (m/s²)²`;
  }

  function restart(start) {
    updateLabels();
    const sigmaBearing = Number(inputs.bearing.value);
    const sigmaRange = Number(inputs.range.value);
    const at = start ?? pointAt(path, 0);
    kf.radar_set_wrap(inputs.wrap.checked ? 1 : 0);
    kf.radar_reset(at.x, at.y, Number(inputs.processNoise.value), sigmaRange, sigmaBearing);
    sim = {
      step: 0, s: 0, pos: at, sigmaBearing, sigmaRange,
      truth: [at], crescents: [], tracks: variants.map(() => []), measurements: 0,
    };
  }

  function moveTarget() {
    const speed = Number(inputs.speed.value);
    if (tool === "steer") {
      if (!pointer) return;
      const dx = pointer.x - sim.pos.x, dy = pointer.y - sim.pos.y;
      const dist = Math.hypot(dx, dy);
      const stepLength = Math.min(speed * dt, dist);
      if (dist > 1e-6) sim.pos = { x: sim.pos.x + dx / dist * stepLength, y: sim.pos.y + dy / dist * stepLength };
      return;
    }
    sim.s += speed * dt;
    if (!path.closed && sim.s > path.length) return restart();
    sim.pos = pointAt(path, sim.s);
  }

  function advance() {
    sim.step += 1;
    moveTarget();
    const truth = sim.pos;
    sim.truth.push(truth);
    if (sim.truth.length > trail) sim.truth.shift();

    kf.radar_predict(dt);

    const measuring = sim.step % Number(inputs.measureEvery.value) === 0;
    if (measuring) {
      sim.measurements += 1;
      for (const radar of activeRadars()) {
        const dx = truth.x - radar.x, dy = truth.y - radar.y;
        if (Math.hypot(dx, dy) < minRange) continue;
        const range = Math.hypot(dx, dy) + sim.sigmaRange * gaussian();
        const bearing = Math.atan2(dy, dx) + sim.sigmaBearing * gaussian();
        kf.radar_correct(radar.x, radar.y, range, bearing);

        // Sample the measurement likelihood in x/y: the crescent.
        const points = [];
        for (let i = 0; i < crescentSamples; i++) {
          const r = range + sim.sigmaRange * gaussian();
          const b = bearing + sim.sigmaBearing * gaussian();
          points.push([radar.x + r * Math.cos(b), radar.y + r * Math.sin(b)]);
        }
        sim.crescents.push({ radar, points });
      }
      while (sim.crescents.length > 3 * activeRadars().length) sim.crescents.shift();
    }

    variants.forEach((v, k) => {
      sim.tracks[k].push({
        x: kf.radar_state(v.id, 0),
        y: kf.radar_state(v.id, 1),
        pxx: kf.radar_covariance(v.id, 0, 0),
        pxy: kf.radar_covariance(v.id, 0, 1),
        pyy: kf.radar_covariance(v.id, 1, 1),
        corrected: measuring,
      });
      if (sim.tracks[k].length > trail) sim.tracks[k].shift();
    });
    return measuring;
  }

  // ── Rendering ──

  function transform(canvas) {
    const { width, height } = canvasSize(canvas);
    const sx = width / (view.xMax - view.xMin);
    const sy = height / (view.yMax - view.yMin);
    return { sx, sy, px: (x) => (x - view.xMin) * sx, py: (y) => height - (y - view.yMin) * sy };
  }

  function drawRadar(ctx, px, py, radar) {
    ctx.fillStyle = color(radar.color);
    ctx.beginPath();
    ctx.moveTo(px(radar.x), py(radar.y) - 8);
    ctx.lineTo(px(radar.x) - 7, py(radar.y) + 4);
    ctx.lineTo(px(radar.x) + 7, py(radar.y) + 4);
    ctx.fill();
  }

  function draw(variant, track) {
    const { canvas } = variant;
    const ctx = canvas.getContext("2d");
    const { width, height, dpr } = canvasSize(canvas);
    const { sx, sy, px, py } = transform(canvas);

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    // Range rings around the first radar, and the line behind it where its bearing jumps.
    const r0 = radars[0];
    ctx.strokeStyle = color("--grid");
    ctx.lineWidth = 1;
    for (let r = 10; r <= 50; r += 10) {
      ctx.beginPath();
      ctx.ellipse(px(r0.x), py(r0.y), r * sx, r * sy, 0, 0, 2 * Math.PI);
      ctx.stroke();
    }
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = color("--grid-strong");
    ctx.beginPath();
    ctx.moveTo(px(r0.x), py(r0.y));
    ctx.lineTo(px(view.xMin), py(r0.y));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = color("--ink-soft");
    ctx.font = "12px 'Recursive Variable', system-ui, sans-serif";
    ctx.fillText("±180°", px(view.xMin) + 4, py(r0.y) - 4);

    // The planned path, with handles while editing.
    if (tool !== "steer") {
      const pts = path.points.map(([x, y]) => [px(x), py(y)]);
      if (path.closed) pts.push(pts[0]);
      ctx.setLineDash([2, 6]);
      polyline(ctx, pts, color("--grid-strong"), 1.5);
      ctx.setLineDash([]);
      if (tool === "edit") {
        ctx.strokeStyle = color("--ink-soft");
        ctx.lineWidth = 1;
        for (const [x, y] of path.points) {
          ctx.beginPath();
          ctx.arc(px(x), py(y), 4, 0, 2 * Math.PI);
          ctx.stroke();
        }
      }
    }
    if (drawing && drawing.length > 1) {
      polyline(ctx, drawing.map(([x, y]) => [px(x), py(y)]), color("--c1"), 2);
    }

    // Older crescents fade out.
    sim.crescents.forEach(({ radar, points }, i) => {
      ctx.globalAlpha = 0.6 * (i + 1) / sim.crescents.length;
      ctx.fillStyle = color(radar.crescent);
      for (const [x, y] of points) ctx.fillRect(px(x) - 1, py(y) - 1, 2, 2);
    });
    ctx.globalAlpha = 1;

    polyline(ctx, sim.truth.map((p) => [px(p.x), py(p.y)]), color("--truth"));
    const finite = track.filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
    const stroke = color(variant.color);
    polyline(ctx, finite.map((p) => [px(p.x), py(p.y)]), stroke);

    // Corrections as dots on the track; the steps in between are predictions only.
    ctx.fillStyle = stroke;
    for (const p of finite) {
      if (!p.corrected) continue;
      ctx.beginPath();
      ctx.arc(px(p.x), py(p.y), 2.5, 0, 2 * Math.PI);
      ctx.fill();
    }

    const last = track.at(-1);
    if (last && [last.x, last.y, last.pxx, last.pxy, last.pyy].every(Number.isFinite)) {
      // 2σ ellipse from the eigen-decomposition of the position covariance.
      const { pxx: a, pxy: b, pyy: c } = last;
      const mean = (a + c) / 2;
      const spread = Math.sqrt(((a - c) / 2) ** 2 + b * b);
      const major = 2 * Math.sqrt(Math.max(mean + spread, 0));
      const minor = 2 * Math.sqrt(Math.max(mean - spread, 0));
      const angle = 0.5 * Math.atan2(2 * b, a - c);

      ctx.save();
      ctx.translate(px(last.x), py(last.y));
      ctx.scale(sx, -sy); // world units, y up
      ctx.rotate(angle);
      ctx.beginPath();
      ctx.ellipse(0, 0, major, minor, 0, 0, 2 * Math.PI);
      ctx.restore();
      ctx.fillStyle = stroke;
      ctx.globalAlpha = 0.15;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.strokeStyle = stroke;
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    const truth = sim.truth.at(-1);
    ctx.fillStyle = color("--ink");
    ctx.beginPath();
    ctx.arc(px(truth.x), py(truth.y), 4, 0, 2 * Math.PI);
    ctx.fill();

    if (tool === "steer" && pointer) {
      ctx.strokeStyle = color("--c1");
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(px(pointer.x), py(pointer.y), 6, 0, 2 * Math.PI);
      ctx.stroke();
    }

    for (const radar of activeRadars()) drawRadar(ctx, px, py, radar);
  }

  function render() {
    variants.forEach((v, k) => draw(v, sim.tracks[k]));
    const truth = sim.truth.at(-1);
    const parts = [`t = ${(sim.step * dt).toFixed(1)} s`, `${sim.measurements} measurements`];
    for (const [k, v] of variants.entries()) {
      const p = sim.tracks[k].at(-1);
      if (!p) continue;
      const error = Math.hypot(p.x - truth.x, p.y - truth.y);
      parts.push(Number.isFinite(error) ? `${v.name} error ${error.toFixed(2)} m` : `${v.name} diverged, restart`);
    }
    $("radar-status").textContent = parts.join(", ");
  }

  // ── Pointer interaction ──

  function toWorld(canvas, event) {
    const rect = canvas.getBoundingClientRect();
    const fx = (event.clientX - rect.left) / rect.width;
    const fy = (event.clientY - rect.top) / rect.height;
    return { x: view.xMin + fx * (view.xMax - view.xMin), y: view.yMax - fy * (view.yMax - view.yMin) };
  }

  function grab(at) {
    const near = (x, y) => Math.hypot(x - at.x, y - at.y) < grabRadius;
    const r = activeRadars().findIndex((radar) => near(radar.x, radar.y));
    if (r >= 0) return { kind: "radar", index: r };
    const w = path.points.findIndex(([x, y]) => near(x, y));
    if (w >= 0) return { kind: "waypoint", index: w };
    return null;
  }

  for (const { canvas } of variants) {
    canvas.addEventListener("pointerdown", (event) => {
      const at = toWorld(canvas, event);
      if (tool === "draw") drawing = [[at.x, at.y]];
      else if (tool === "edit") dragging = grab(at);
      else pointer = at;
      if (drawing || dragging) {
        try {
          canvas.setPointerCapture(event.pointerId);
        } catch {
          // Synthetic events have no active pointer to capture.
        }
      }
    });
    canvas.addEventListener("pointermove", (event) => {
      const at = toWorld(canvas, event);
      if (tool === "steer") pointer = at;
      if (drawing) {
        const [lx, ly] = drawing.at(-1);
        if (Math.hypot(at.x - lx, at.y - ly) >= drawSpacing) drawing.push([at.x, at.y]);
      } else if (dragging?.kind === "radar") {
        Object.assign(radars[dragging.index], at);
      } else if (dragging?.kind === "waypoint") {
        path.points[dragging.index] = [at.x, at.y];
        path = makePath(path.points, path.closed);
      } else if (tool === "edit") {
        canvas.style.cursor = grab(at) ? "grab" : "default";
      }
    });
    canvas.addEventListener("pointerup", (event) => {
      if (drawing) {
        const at = toWorld(canvas, event);
        drawing.push([at.x, at.y]);
        if (drawing.length >= 3) {
          const [sx0, sy0] = drawing[0], [ex, ey] = drawing.at(-1);
          const closed = Math.hypot(ex - sx0, ey - sy0) < 3 * drawSpacing;
          if (closed) drawing.pop();
          path = makePath(drawing, closed);
          inputs.preset.value = "drawn";
          restart();
        }
        drawing = null;
      }
      dragging = null;
    });
    canvas.addEventListener("pointerleave", () => {
      if (tool === "steer") pointer = null;
    });
  }

  function setTool(next) {
    const leavingSteer = tool === "steer" && next !== "steer";
    tool = next;
    $("tool-hint").textContent = hints[tool];
    for (const { canvas } of variants) canvas.style.cursor = tool === "edit" ? "default" : "crosshair";
    pointer = null;
    // After steering, the target goes back to the start of the path.
    if (leavingSteer) restart();
  }

  // ── Controls ──

  function setPaused(value) {
    paused = value;
    $("pause").textContent = paused ? "Resume" : "Pause";
    $("step").disabled = !paused;
    $("next-measurement").disabled = !paused;
  }

  document.querySelectorAll('input[name="tool"]').forEach((radio) =>
    radio.addEventListener("change", () => setTool(radio.value)));
  inputs.preset.addEventListener("change", () => {
    path = presets[inputs.preset.value]();
    restart();
  });
  for (const input of [inputs.bearing, inputs.range, inputs.processNoise]) {
    input.addEventListener("input", () => restart(tool === "steer" ? sim.pos : undefined));
  }
  for (const input of [inputs.speed, inputs.measureEvery]) input.addEventListener("input", updateLabels);
  inputs.wrap.addEventListener("change", () => kf.radar_set_wrap(inputs.wrap.checked ? 1 : 0));
  $("restart").addEventListener("click", () => restart(tool === "steer" ? sim.pos : undefined));
  $("pause").addEventListener("click", () => setPaused(!paused));
  $("step").addEventListener("click", advance);
  $("next-measurement").addEventListener("click", () => {
    for (let i = 0; i < 1000 && !advance(); i++);
  });

  // Run only while the experiment is on screen.
  let visible = true;
  new IntersectionObserver(([entry]) => (visible = entry.isIntersecting)).observe(variants[0].canvas);

  let last = 0;
  function frame(now) {
    // About 20 filter steps per second, i.e. twice real time.
    if (!paused && visible && now - last > 50) {
      last = now;
      advance();
    }
    render();
    requestAnimationFrame(frame);
  }

  setTool("edit");
  restart();
  // With reduced motion, the simulation waits for Resume or Step.
  if (reducedMotion.matches) setPaused(true);
  requestAnimationFrame(frame);
}
