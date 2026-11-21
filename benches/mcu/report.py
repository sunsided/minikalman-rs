#!/usr/bin/env python3
"""Turns the raw Renode logs of run.sh into Markdown tables.

    report.py <dir>               one table per core and profile
    report.py <base-dir> <dir>    the same, with the base counts and the change

Each raw file is named `<target>.<profile>.<types>.txt` and holds lines
`<name> <instructions> <ok> [<symbol>=<calls> ...]` (see renode/bench.py). The counts are
reported per call: the mean over the measured iterations minus the marker overhead
(`calib/empty`), with the range when iterations differ.
"""

import collections
import os
import sys

CORES = {'thumbv7em-none-eabihf': 'Cortex-M4F', 'thumbv6m-none-eabi': 'Cortex-M0+'}
PROFILES = {'release': 'opt-level = "s"', 'o3': 'opt-level = 3'}


def load(directory):
    """Returns {(target, profile): {name: (mean, min, max, all_ok, calls)}}."""
    runs = collections.defaultdict(dict)
    for file in sorted(os.listdir(directory)):
        if not file.endswith('.txt'):
            continue
        target, profile = file.split('.')[:2]
        samples = collections.defaultdict(list)
        calls = collections.defaultdict(lambda: collections.Counter())
        done = False
        with open(os.path.join(directory, file)) as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == 'FAIL':
                    sys.exit('%s: %s' % (file, line.strip()))
                if parts[0] in ('DONE', 'STACK_FREE'):
                    done |= parts[0] == 'DONE'
                    continue
                samples[parts[0]].append((int(parts[1]), parts[2] == '1'))
                for kv in parts[3:]:
                    symbol, n = kv.split('=')
                    calls[parts[0]][symbol] += int(n)
        if not done:
            sys.exit('%s: the run did not reach mk_done' % file)
        calib = samples.pop('calib/empty')
        # The first call can carry loop setup (opt-level 3 peels it), so take the minimum.
        overhead = min(n for n, _ in calib)
        for name, values in samples.items():
            counts = [n - overhead for n, _ in values]
            per_call = {s: c / len(values) for s, c in calls[name].items()}
            runs[(target, profile)][name] = (
                sum(counts) / len(counts),
                min(counts),
                max(counts),
                all(ok for _, ok in values),
                per_call,
            )
    return runs


def fmt(entry):
    mean, lo, hi, ok, _ = entry
    text = '%.0f' % mean if lo == hi else '%.0f (%d–%d)' % (mean, lo, hi)
    return text if ok else text + ' **not applied**'


def fmt_calls(calls):
    return ', '.join('%s ×%g' % (s, round(c, 1)) for s, c in sorted(calls.items(), key=lambda kv: -kv[1]))


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    head = load(sys.argv[-1])
    base = load(sys.argv[1]) if len(sys.argv) == 3 else None
    for key in sorted(head, key=lambda k: (list(CORES).index(k[0]), k[1] != 'release')):
        target, profile = key
        print('### %s (`%s`), %s\n' % (CORES.get(target, target), target, PROFILES.get(profile, profile)))
        rows = head[key]
        any_calls = any(e[4] for e in rows.values())
        if base is not None:
            header = ['operation', 'base', 'this tree', 'change']
        else:
            header = ['operation', 'instructions']
        if any_calls:
            header.append('helper calls per op')
        print('| ' + ' | '.join(header) + ' |')
        print('|' + '|'.join(['---'] + ['---:'] * (len(header) - 1 - any_calls) + ['---'] * any_calls) + '|')
        for name in rows:
            cells = [name]
            if base is not None:
                old = base.get(key, {}).get(name)
                cells.append(fmt(old) if old else '–')
                cells.append(fmt(rows[name]))
                cells.append('%+.1f %%' % (100.0 * (rows[name][0] / old[0] - 1)) if old and old[0] else '–')
            else:
                cells.append(fmt(rows[name]))
            if any_calls:
                cells.append(fmt_calls(rows[name][4]))
            print('| ' + ' | '.join(cells) + ' |')
        print()


if __name__ == '__main__':
    main()
