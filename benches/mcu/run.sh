#!/usr/bin/env bash
# Builds the firmware, runs it headless in Renode and prints per-operation instruction counts.
#
# Run from anywhere:  benches/mcu/run.sh
#   TARGETS="..."   targets to measure (default: thumbv7em-none-eabihf thumbv6m-none-eabi)
#   PROFILES="..."  Cargo profiles (default: release o3, i.e. opt-level "s" and 3)
#   TYPES="..."     element types, one firmware binary each (default: f32 i16f16 i32f32
#                   nav nav-libm; nav and nav-libm are the f32 attitude and GNSS/INS
#                   workloads with and without minikalman's `unsafe`; nav-libm only on the M4F)
#   BASE=<rev>      also measure minikalman at git revision <rev> (same firmware source,
#                   same Cargo.lock) and print both side by side
#   COUNT_CALLS=1   count calls to the soft-float, 64/128-bit and libm helpers (M4F only:
#                   on the M0+ the hooks on the soft-float helpers stall Renode 1.17)
#
# Outputs go to benches/mcu/results/ (git-ignored): one raw log per binary in head/ (and
# base/), and tables.md. Needs Renode (`renode` on PATH, tested with 1.17), the rustup
# targets, python3, and for COUNT_CALLS=1 llvm-nm (or rust-nm).
set -euo pipefail
cd "$(dirname "$0")"
here="$(pwd)"
out="$here/results"
TARGETS="${TARGETS:-thumbv7em-none-eabihf thumbv6m-none-eabi}"
PROFILES="${PROFILES:-release o3}"
TYPES="${TYPES:-f32 i16f16 i32f32 nav nav-libm}"
COUNT_CALLS="${COUNT_CALLS:-0}"
if [ "$COUNT_CALLS" = 1 ]; then
  nm_tool="$(command -v llvm-nm || command -v rust-nm)" || { echo "COUNT_CALLS=1 needs llvm-nm or rust-nm" >&2; exit 1; }
fi
if [ -n "${CARGO_TARGET_DIR:-}" ] && [ "${CARGO_TARGET_DIR#/}" = "$CARGO_TARGET_DIR" ]; then
  export CARGO_TARGET_DIR="$here/$CARGO_TARGET_DIR"
fi
target_dir="${CARGO_TARGET_DIR:-$here/target}"

# Renode's platform per target. The core decides the instruction stream; the chip around it
# only has to provide flash and RAM at the addresses build.rs links for.
platform() {
  case "$1" in
    thumbv6m-*) echo "$here/renode/stm32l071-large.repl" ;;
    *) echo "platforms/cpus/stm32f412.repl" ;;
  esac
}

# Comma-separated `symbol=address` list of the helpers whose calls bench.py counts.
helpers() {
  [ "$COUNT_CALLS" = 1 ] || return 0
  "$nm_tool" -C --defined-only "$1" \
    | awk '$2 ~ /^[tT]$/ {
        name = $3; sub(/.*::/, "", name)
        if (name ~ /^__aeabi_(f|d|l|ul|i|ui)[a-z0-9]+$/ && name !~ /mem/ \
            || name ~ /^__(u)?(div|mod|mul)ti3$/ || name ~ /^__(u)?divmodti4$/ \
            || name ~ /^(sqrtf|atan2f|sinf|cosf)$/) {
          if (!(name in seen)) { seen[name] = 1; printf "%s%s=0x%s", sep, name, $1; sep = "," }
        }
      }'
}

# measure <source-dir> <log-dir>: builds every target/profile/type in <source-dir> and runs it.
measure() {
  local src="$1" logs="$2" cargo_target="$3"
  mkdir -p "$logs"
  for target in $TARGETS; do
    for profile in $PROFILES; do
      for ty in $TYPES; do
        # Without an FPU the f32 square root is libm::sqrtf either way.
        if [ "$ty" = nav-libm ] && [ "${target#thumbv6m}" != "$target" ]; then continue; fi
        (cd "$src" && CARGO_TARGET_DIR="$cargo_target" cargo build --locked --quiet \
          --profile "$profile" --target "$target" --no-default-features --features "$ty")
        local elf="$cargo_target/$target/$profile/kalman-mcu"
        local log="$logs/$target.$profile.$ty.txt"
        local pid="$logs/.renode.pid"
        rm -f "$log" "$pid"
        timeout 900 renode --disable-gui --plain --hide-log -P -1 --pid-file "$pid" -e "
          mach create; machine LoadPlatformDescription @$(platform "$target");
          sysbus LoadELF @$elf; include @$here/renode/bench.py;
          bench_setup \"$log\" \"$(helpers "$elf")\"; start" > "$logs/.renode.out" 2>&1 || rc=$?
        # The launcher script does not exec, so a timeout would leave the emulator running.
        if [ "${rc:-0}" = 124 ] && [ -f "$pid" ]; then kill "$(cat "$pid")" 2> /dev/null || true; fi
        rc=0
        if ! grep -q '^DONE' "$log" 2> /dev/null; then
          echo "$log: run did not finish; Renode output in $logs/.renode.out" >&2
          exit 1
        fi
        # The navigation binaries repaint the stack and report it before each repaint.
        free="$(awk '$1 == "STACK_FREE" && (m == "" || $2 < m) { m = $2 } END { print m }' "$log")"
        echo "$target $profile $ty: $(grep -vc '^DONE\|^STACK\|^NOTE' "$log") sections, ${free:-?} B of stack never used" >&2
        if [ "${free:-0}" -lt 256 ]; then
          echo "$log: the stack came within 256 B of the statics; the counts are not trustworthy" >&2
          exit 1
        fi
      done
    done
  done
}

rm -rf "$out/head" "$out/base"
measure "$here" "$out/head" "$target_dir"

if [ -n "${BASE:-}" ]; then
  # The same firmware source and lock file against minikalman at $BASE.
  base_src="$target_dir/base-src"
  rm -rf "$base_src"
  mkdir -p "$base_src"
  # From the top level: inside a subdirectory, git archive only packs that subtree.
  git -C "$(git -C "$here" rev-parse --show-toplevel)" archive "$BASE" | tar -x -C "$base_src"
  mkdir -p "$base_src/benches/mcu"
  cp -r "$here/Cargo.toml" "$here/Cargo.lock" "$here/build.rs" "$here/src" "$here/.cargo" \
    "$base_src/benches/mcu/"
  measure "$base_src/benches/mcu" "$out/base" "$target_dir/base-build"
  python3 -I report.py "$out/base" "$out/head" | tee "$out/tables.md"
else
  python3 -I report.py "$out/head" | tee "$out/tables.md"
fi
