"""Builds Markdown timing tables from criterion's estimates.json files.

Usage: python3 -I tables.py <criterion_dir> [<criterion_dir> ...] > tables.md
With several dirs (repeat runs), each cell shows the median of the per-run medians and the
spread (max-min) across runs.
"""
import json, os, sys, statistics as st

def load(root):
    out = {}
    for dp, _, fs in os.walk(root):
        if dp.endswith(os.path.join("new")) and "estimates.json" in fs:
            bj = os.path.join(dp, "benchmark.json")
            if not os.path.exists(bj):
                continue
            b = json.load(open(bj))
            e = json.load(open(os.path.join(dp, "estimates.json")))
            out[(b["group_id"], b["function_id"])] = (e["median"]["point_estimate"],
                e["median"]["confidence_interval"]["lower_bound"], e["median"]["confidence_interval"]["upper_bound"])
    return out

runs = [load(d) for d in sys.argv[1:]]
keys = sorted(set().union(*[r.keys() for r in runs]))
groups = {}
for g, f in keys:
    name, mode = f.rsplit("/", 1)
    groups.setdefault(g, {}).setdefault(name, {})[mode] = [r[(g, f)][0] for r in runs if (g, f) in r]

def cell(vals):
    m = st.median(vals)
    if len(vals) > 1:
        return f"{m:.1f} (±{(max(vals)-min(vals))/2:.1f})"
    return f"{m:.1f}"

order = ["kf_2x1/f32","kf_2x1/f64","kf_4x2/f32","kf_4x2/f64","kf_6x3/f32","kf_6x3/f64","ekf_radar_4x2/f32","ekf_radar_4x2/f64"]
for g in order + sorted(set(groups) - set(order)):
    if g not in groups:
        continue
    print(f"\n#### {g} (ns/op, median)\n")
    print("| filter | predict | correct | step |")
    print("|---|---:|---:|---:|")
    for name, modes in groups[g].items():
        print(f"| {name} | " + " | ".join(cell(modes[m]) if m in modes else "-" for m in ["predict","correct","step"]) + " |")
