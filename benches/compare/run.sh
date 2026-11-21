#!/usr/bin/env bash
# One-command reproduction: agreement gate, allocation/memory metrics, timings, tables, code size.
#
# Run from anywhere:  benches/compare/run.sh
#   CPU=<core>            pin the benchmarks to one core with taskset (default: 2; empty disables pinning)
#   BENCH_FILTER=<regex>  select benchmarks, e.g. 'minikalman/' or 'kf_4x2/f32'
#   SKIP_CODESIZE=1       skip the thumbv7em-none-eabihf code size step
#
# Outputs go to benches/compare/results/ (git-ignored):
#   check.json    agreement errors, allocations per step and live heap per filter
#   tables.md     criterion medians per problem (predict, correct, step)
#   codesize.txt  .text/.rodata of each thumbv7em binary (subtract the `baseline` row)
set -euo pipefail
cd "$(dirname "$0")"
here="$(pwd)"
out="$here/results"
mkdir -p "$out"
target="${CARGO_TARGET_DIR:-$here/target}"

# 1. Correctness gate: exits non-zero if any filter disagrees with the f64 reference.
cargo run --locked --release --bin check > "$out/check.json"

# 2. Timings.
pin=()
CPU="${CPU-2}"
if [ -n "$CPU" ] && command -v taskset > /dev/null; then pin=(taskset -c "$CPU"); fi
${pin[@]+"${pin[@]}"} cargo bench --locked --bench compare -- ${BENCH_FILTER:-}
python3 -I tables.py "$target/criterion" > "$out/tables.md"

# 3. Code size on Cortex-M4F (needs `rustup target add thumbv7em-none-eabihf` and llvm-size).
if [ -z "${SKIP_CODESIZE:-}" ]; then
  size_tool="$(command -v llvm-size || command -v rust-size || true)"
  if [ -z "$size_tool" ]; then
    echo "llvm-size not found; skipping the code size step" >&2
  else
    cs_target="${CARGO_TARGET_DIR:-$here/codesize/target}"
    (cd codesize && cargo build --locked --release --bins)
    for b in "$cs_target"/thumbv7em-none-eabihf/release/*; do
      if [ -f "$b" ] && [ -x "$b" ]; then
        "$size_tool" -A "$b" | awk -v n="$(basename "$b")" \
          '$1==".text"{t=$2} $1==".rodata"{r=$2} END{print n, ".text", t, ".rodata", r+0}'
      fi
    done | sort | tee "$out/codesize.txt"
  fi
fi
echo "results in $out" >&2
