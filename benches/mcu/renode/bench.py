# Renode monitor extension: logs the executed-instruction count of every measured section.
#
# The firmware brackets each operation with calls to `mk_begin()` and
# `mk_end(name_ptr, name_len, ok)`. The hooks read `ExecutedInstructions` of the core at both
# entries and write one line per section:
#
#     <name> <instructions> <ok> [<symbol>=<calls> ...]
#
# `counted` is a comma-separated list of `symbol=0xaddress` pairs (for example the soft-float
# helpers and `sqrtf`); calls to them inside a section are counted and appended to its line.
# `mk_done()` and `mk_fail()` end the run and quit Renode. `mk_stack(free)` writes
# `STACK_FREE <free>`, and `mk_note(name_ptr, name_len, value)` (only in the binaries that have
# it) writes `NOTE <name> <value>`, for sizes and stack use in bytes.


def mc_bench_setup(out_path, counted=''):
    bus = monitor.Machine.SystemBus
    cpu = bus.GetCPUs()[0]
    state = {'start': 0, 'inside': False, 'calls': {}}
    out = open(out_path, 'w')

    def quit_renode():
        # Quitting from the CPU thread would wait for that thread; leave it to the pool.
        import System
        System.Threading.ThreadPool.QueueUserWorkItem(lambda _: monitor.Parse('quit'))

    def begin(cpu, _):
        state['calls'] = {}
        state['inside'] = True
        state['start'] = cpu.ExecutedInstructions

    def end(cpu, _):
        n = cpu.ExecutedInstructions - state['start']
        state['inside'] = False
        ptr = cpu.GetRegister(0).RawValue
        length = cpu.GetRegister(1).RawValue
        ok = cpu.GetRegister(2).RawValue & 0xFF
        name = ''.join(chr(bus.ReadByte(ptr + i)) for i in range(length))
        calls = ' '.join('%s=%d' % kv for kv in sorted(state['calls'].items()))
        out.write(('%s %d %d %s' % (name, n, ok, calls)).rstrip() + '\n')
        out.flush()

    def done(cpu, _):
        # `mk_done` spins on itself, so this hook fires again until Renode has quit.
        if out.closed:
            return
        out.write('DONE %d\n' % cpu.ExecutedInstructions)
        out.close()
        quit_renode()

    def fail(cpu, _):
        if out.closed:
            return
        out.write('FAIL panic at %d instructions\n' % cpu.ExecutedInstructions)
        out.close()
        quit_renode()

    def stack(cpu, _):
        out.write('STACK_FREE %d\n' % cpu.GetRegister(0).RawValue)

    def note(cpu, _):
        ptr = cpu.GetRegister(0).RawValue
        length = cpu.GetRegister(1).RawValue
        name = ''.join(chr(bus.ReadByte(ptr + i)) for i in range(length))
        out.write('NOTE %s %d\n' % (name, cpu.GetRegister(2).RawValue))

    def counter(symbol):
        def hook(cpu, _):
            if state['inside']:
                state['calls'][symbol] = state['calls'].get(symbol, 0) + 1
        return hook

    for sym, fn in (('mk_begin', begin), ('mk_end', end), ('mk_done', done), ('mk_fail', fail),
                    ('mk_stack', stack)):
        cpu.AddHook(bus.GetSymbolAddress(sym), fn)
    # Optional: only the navigation binaries define it.
    try:
        cpu.AddHook(bus.GetSymbolAddress('mk_note'), note)
    except Exception:
        pass
    for pair in [p for p in counted.split(',') if p]:
        symbol, address = pair.split('=')
        cpu.AddHook(int(address, 16) & ~1, counter(symbol))
