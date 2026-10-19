"""WC on ONE card: the one-launch windows + conv + gates (gdn_wc.launch) against the served per-user path, bit for bit, at four and two users, and timed.

Run with QWEN_FAST_TP=4 in the environment (the geometry is the four-card shard's: 2,560 channels, 12 value heads, a at column 4,096 and b at 4,108 of a
4,120-wide projection). One p150a, a 1x1 mesh, no model and no collective: the launch's program is per chip, so one chip proves the two new kernels
compile, run and move the same bytes; whether four chips of a mesh agree is the audited attach's job (every chip, every audited layer).

THE TWO ARMS, over ONE upload of the same host data (neither arm writes an input):
  served  per user the pieces the served split cuts (a tile slice for users 0 and 2, the untilize / slice / tilize round trip for users 1 and 3:
          gdn_device_loop_state.resident_piece), the T2 packed windows (gdn_conv_windows_packed.build_windows_packed, the production window kernel on a 1x1 mesh),
          ONE served gdn_decode_conv_gates call on the user's own piece and windows (batch 16), and the z slice of the piece - the V1 audit's reference path
          (gdn_block_conv_tp.hold_served), not the V1 block chain: WC is held against what V1 is held against.
  wc      gdn_wc.launch(chips=1) on the block: conv, beta, g, z per user and the sixteen advanced windows.
Compared per user: conv, beta, g, z and the four advanced windows (eight tensors), read from the chip and viewed as int16 bit patterns, the in-trace audit's rule
(the runtime's bf16 upload and readback may flush denormals, but they do so for both arms alike).

SECTIONS. compare (regime x seed x users): random finite bf16, edge values (+-0, the smallest normal, denormals, the largest finite, +-1) and gauss (rounded
standard-normal data with small-magnitude history rows, the shape real activations take); replay: the WC launch captured in a trace, replayed after the input
buffers were restaged IN PLACE with a second dataset, against the served arm run eagerly on that dataset (a trace must carry addresses, not data); cache: the
program-cache entries after the first and a repeated launch (the second must add none); control: WC with two users' histories swapped must DIFFER from the served
arm (a compare blind to a wrong history would otherwise pass anything); timing: per users, a captured trace of 24 launches per arm (eager timing would measure the
host's program-descriptor build), 25 rounds of replays in serpentine order, the per-launch median and quartiles of each arm.

Exit: 0 PASS (every compare and the replay exact, no fall-back, the control differs, the cache stable); 1 FAIL; 3 the watchdog; 4 NOT-RUN (a section raised).
The last stdout line is one JSON object (kind wc-card-m) and the line above it 'GDN_WC verdict=...'.
"""

import argparse
import json
import os
import statistics
import sys
import threading
import time

KIND = 'wc-card-m'
USERS = (4, 2)
REGIMES = ('random', 'edge', 'gauss')
LAUNCHES = 24
ROUNDS = 25
WATCHDOG_S = 3000
TRACE_REGION = 96 * 1024 * 1024
EDGE_BITS = (0x0000, 0x8000, 0x0080, 0x8080, 0x0001, 0x8001, 0x007F, 0x7F7F, 0xFF7F, 0x3F80, 0xBF80)
LABELS = ('conv', 'beta', 'g', 'z', 'window0', 'window1', 'window2', 'window3')


def host_data(torch, users, regime, seed, found):
    """The seeded host tensors of one case: the block projection (1, 16 * users, width), per user four histories (1, 1, qkv), four taps (1, 1, qkv), dt_bias and
    neg_exp_A (1, 1, heads), as bfloat16 built from int16 bit patterns (finite only: an infinity or NaN would only test the runtime's own flush)."""
    generator = torch.Generator().manual_seed(seed)

    def finite_bits(shape):
        raw = torch.randint(0, 65536, shape, generator=generator, dtype=torch.int32)
        if regime == 'edge':
            pick = torch.randint(0, len(EDGE_BITS), shape, generator=generator, dtype=torch.int32)
            edge = torch.tensor(EDGE_BITS, dtype=torch.int32)[pick]
            raw = torch.where(torch.rand(shape, generator=generator) < 0.5, edge, raw)
        raw = torch.where((raw & 0x7F80) == 0x7F80, (raw & 0x8000) | 0x3F80, raw)
        return (raw - ((raw & 0x8000) << 1)).to(torch.int16)

    def bf16(shape, scale=1.0):
        if regime == 'gauss':
            return (torch.randn(shape, generator=generator) * scale).to(torch.bfloat16)
        return finite_bits(shape).contiguous().view(torch.bfloat16)

    width = found.gdn_qkvzab
    return dict(users=users, x=bf16((1, 16 * users, width)),
                histories=[[bf16((1, 1, found.gdn_qkv), 0.1) for _ in range(4)] for _ in range(users)],
                taps=[bf16((1, 1, found.gdn_qkv), 0.5) for _ in range(4)], dt_bias=bf16((1, 1, found.gdn_nv)), neg_exp_A=bf16((1, 1, found.gdn_nv)))


def differing(torch, left, right):
    """Elements that differ between two int16 views of the same shape (a shape difference counts every element of the larger)."""
    if tuple(left.shape) != tuple(right.shape):
        return max(left.numel(), right.numel())
    return int((left != right).sum())


def summarize(values):
    ordered = sorted(values)
    quarter = lambda fraction: ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]
    return dict(n=len(ordered), median_us=round(statistics.median(ordered), 2), q1_us=round(quarter(0.25), 2), q3_us=round(quarter(0.75), 2),
                min_us=round(ordered[0], 2))


def verdict(sections, replay, cache, control):
    """(PASS | FAIL | NOT-RUN, exit status). Every compare section exact and none fell back; the replay exact; the cache stable; the control different."""
    everything = list(sections) + [replay, cache, control]
    if not sections or any(section is None or section.get('error') for section in everything):
        return 'NOT-RUN', 4
    if any(section.get('differing', 1) or section.get('fell_back') for section in sections) or replay.get('differing', 1) or replay.get('fell_back'):
        return 'FAIL', 1
    if not cache.get('stable') or not control.get('differing'):
        return 'FAIL', 1
    return 'PASS', 0


class Rig(object):
    """The device side of one case: upload, run each arm, read back. `ttnn` is the module (a fake in the tests)."""

    def __init__(self, ttnn, mesh, torch, found, wc, chain):
        self.ttnn, self.mesh, self.torch, self.found, self.wc, self.chain = ttnn, mesh, torch, found, wc, chain

    def upload(self, host):
        ttnn, mesh = self.ttnn, self.mesh

        def up(tensor):
            return ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        return dict(users=host['users'], x=up(host['x']), histories=[[up(value) for value in user] for user in host['histories']],
                    taps=[up(value) for value in host['taps']], dt_bias=up(host['dt_bias']), neg_exp_A=up(host['neg_exp_A']))

    def restage(self, tensors, host):
        """Overwrite the uploaded buffers IN PLACE with another dataset (the replay section: the trace baked these addresses)."""
        ttnn = self.ttnn
        pairs = [(tensors['x'], host['x']), (tensors['dt_bias'], host['dt_bias']), (tensors['neg_exp_A'], host['neg_exp_A'])]
        pairs += list(zip(tensors['taps'], host['taps']))
        pairs += [(device, source) for devices, sources in zip(tensors['histories'], host['histories']) for device, source in zip(devices, sources)]
        for device, source in pairs:
            staged = ttnn.from_torch(source, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))
            ttnn.copy_host_to_device_tensor(staged, device)
        ttnn.synchronize_device(self.mesh)

    def served(self, tensors):
        """The served per-user path (see the module docstring): {conv, beta, g, z (per user), window (per user, four)} plus everything to free."""
        return self.chain.served_path(self.ttnn, self.mesh, self.found, tensors)

    def wc_launch(self, tensors):
        return self.wc.launch(self.ttnn, self.mesh, tensors['x'], tensors['histories'], tensors['taps'], tensors['dt_bias'], tensors['neg_exp_A'], chips=1)

    def read(self, tensor):
        ttnn, torch = self.ttnn, self.torch
        return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).contiguous().view(torch.int16)


def flatten(made):
    """{conv, beta, g, z, window} -> [(label, tensor)] in LABELS order, per user, as ('conv u0', tensor)."""
    out = []
    for user in range(len(made['conv'])):
        singles = [made['conv'][user], made['beta'][user], made['g'][user], made['z'][user]]
        for label, tensor in zip(LABELS[:4], singles):
            out.append(('%s u%d' % (label, user), tensor))
        for slot in range(4):
            out.append(('window%d u%d' % (slot, user), made['window'][user][slot]))
    return out


def release_all(rig, *groups):
    """Deallocate every device tensor in the (nested) groups once, so a long run does not grow the card's DRAM case over case."""
    seen = set()

    def walk(value):
        if isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif value is not None and id(value) not in seen:
            seen.add(id(value))
            try:
                rig.ttnn.deallocate(value)
            except BaseException:  # noqa: BLE001
                pass

    walk(list(groups))


def compare_made(rig, served, made):
    pairs = list(zip(flatten(served), flatten(made)))
    counts = {left[0]: differing(rig.torch, rig.read(left[1]), rig.read(right[1])) for left, right in pairs}
    return dict(differing=sum(counts.values()), by_tensor={key: value for key, value in counts.items() if value}, compared=len(counts), fell_back=False)


def compare_case(rig, host):
    """One compare section: the served arm and WC over one upload, eight tensors a user, bit for bit."""
    tensors = rig.upload(host)
    served = made = None
    try:
        served = rig.served(tensors)
        made = rig.wc_launch(tensors)
        if made is None:
            return dict(users=host['users'], differing=-1, fell_back=True)
        section = compare_made(rig, served, made)
        section['users'] = host['users']
        return section
    finally:
        release_all(rig, tensors, served, made)


def cache_entries(mesh):
    """The mesh's program-cache entry count, or None where the runtime does not report it."""
    counter = getattr(mesh, 'num_program_cache_entries', None)
    return counter() if callable(counter) else None


def replay_section(rig, host, other):
    """The WC launch captured in a trace over `host`'s buffers, replayed after the buffers were restaged in place with `other`, against the served arm run eagerly
    on `other`. The launch's output tensors are allocated inside the capture and stay alive until the trace is released."""
    ttnn, mesh = rig.ttnn, rig.mesh
    tensors = rig.upload(host)
    served = made = handle = None
    try:
        release_all(rig, rig.wc_launch(tensors))          # the warm call: compile outside the capture
        ttnn.synchronize_device(mesh)
        handle = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            made = rig.wc_launch(tensors)
        finally:
            ttnn.end_trace_capture(mesh, handle, cq_id=0)
        if made is None:
            return dict(users=host['users'], differing=-1, fell_back=True)
        rig.restage(tensors, other)
        served = rig.served(tensors)
        ttnn.execute_trace(mesh, handle, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        section = compare_made(rig, served, made)
        section['users'] = host['users']
        return section
    finally:
        if handle is not None:
            try:
                ttnn.release_trace(mesh, handle)
            except BaseException:  # noqa: BLE001
                pass
        release_all(rig, tensors, served, made)


def cache_section(rig, host):
    """Program-cache entries after a first WC launch and after four more: they must not grow (generic_op caches on the kernels, defines, compile words and
    cores, not on addresses, so a later layer's launch is a hit)."""
    tensors = rig.upload(host)
    try:
        before = cache_entries(rig.mesh)
        release_all(rig, rig.wc_launch(tensors))
        first = cache_entries(rig.mesh)
        for _ in range(4):
            release_all(rig, rig.wc_launch(tensors))
        last = cache_entries(rig.mesh)
        return dict(before=before, after_first=first, after_repeat=last, stable=(first is None) or first == last)
    finally:
        release_all(rig, tensors)


def control_section(rig, host):
    """WC with the histories of users 0 and 1 swapped against the served arm on the right ones: it must differ in at least one element."""
    tensors = rig.upload(host)
    wrong = dict(tensors)
    wrong['histories'] = [tensors['histories'][1], tensors['histories'][0]] + tensors['histories'][2:]
    served = made = None
    try:
        served = rig.served(tensors)
        made = rig.wc_launch(wrong)
        if made is None:
            return dict(differing=0, fell_back=True)
        return compare_made(rig, served, made)
    finally:
        release_all(rig, tensors, served, made)


def capture(rig, run, launches):
    """Capture `launches` back-to-back launches of one arm in one trace (the programs were compiled by a warm call); returns (handle, the outputs, which stay
    alive until the trace is released)."""
    ttnn, mesh = rig.ttnn, rig.mesh
    handle = ttnn.begin_trace_capture(mesh, cq_id=0)
    kept = []
    try:
        try:
            for _ in range(launches):
                kept.append(run())
        finally:
            ttnn.end_trace_capture(mesh, handle, cq_id=0)
    except BaseException:
        ttnn.release_trace(mesh, handle)
        release_all(rig, kept)
        raise
    return handle, kept


def time_case(rig, host, launches=LAUNCHES, rounds=ROUNDS, clock=time.perf_counter):
    """Per-launch microseconds of the served per-user path and the WC launch, each arm under a captured trace of `launches` layers, over serpentine rounds of
    replays. The served arm is the whole per-user path (pieces, windows, conv gates, z), so the difference is what one GDN layer's convolution stage saves."""
    ttnn, mesh = rig.ttnn, rig.mesh
    tensors = rig.upload(host)
    runs = dict(served=lambda: rig.served(tensors), wc=lambda: rig.wc_launch(tensors))
    samples = dict(served=[], wc=[])
    traces, kept = {}, []
    try:
        for name in runs:
            release_all(rig, runs[name]())
            ttnn.synchronize_device(mesh)
            traces[name], outputs = capture(rig, runs[name], launches)
            kept.append(outputs)
            ttnn.synchronize_device(mesh)
        for name in runs:
            ttnn.execute_trace(mesh, traces[name], cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        for index in range(rounds):
            order = ('served', 'wc') if index % 2 == 0 else ('wc', 'served')
            for name in order:
                started = clock()
                ttnn.execute_trace(mesh, traces[name], cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                samples[name].append((clock() - started) / launches * 1e6)
    finally:
        for handle in traces.values():
            try:
                ttnn.release_trace(mesh, handle)
            except BaseException:  # noqa: BLE001
                pass
        release_all(rig, tensors, kept)
    served, wc = summarize(samples['served']), summarize(samples['wc'])
    return dict(users=host['users'], mode='trace', served=served, wc=wc, wc_minus_served_us=round(wc['median_us'] - served['median_us'], 2))


def main(argv=None, torch=None, ttnn=None, wc=None, chain=None, tp_shapes=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', required=True)
    parser.add_argument('--users', default=','.join(str(value) for value in USERS))
    parser.add_argument('--regimes', default=','.join(REGIMES))
    parser.add_argument('--seeds', default='17,23')
    parser.add_argument('--timing', choices=('on', 'off'), default='on')
    options = parser.parse_args(argv)
    users = [int(value) for value in options.users.split(',')]
    regimes = options.regimes.split(',')
    seeds = [int(value) for value in options.seeds.split(',')]
    if any(value not in USERS for value in users) or any(value not in REGIMES for value in regimes):
        print('refusing: users are 4 2, regimes random edge gauss', file=sys.stderr)
        return 2
    timer = threading.Timer(WATCHDOG_S, lambda: (print('GDN_WC watchdog', flush=True), os._exit(3)))
    timer.daemon = True
    timer.start()
    if torch is None:
        import torch
        import ttnn
        import gdn_wc as wc
        import gdn_wc_card_m_chain as chain
        import tp_shapes
    report = dict(kind=KIND, users=users, regimes=regimes, seeds=seeds, environment=dict(QWEN_FAST_TP=os.environ.get('QWEN_FAST_TP')))
    mesh, status, sections = None, 4, []
    try:
        if os.environ.get('QWEN_FAST_TP') != '4':
            raise RuntimeError('QWEN_FAST_TP=4 required (the geometry is the four-card shard\'s)')
        found = tp_shapes.active()
        report['geometry'] = dict(channels=found.gdn_qkv, heads=found.gdn_nv, a_col=found.gdn_a_col, b_col=found.gdn_b_col, width=found.gdn_qkvzab)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=TRACE_REGION)
        grid = mesh.compute_with_storage_grid_size()
        report['grid'] = [grid.x, grid.y]
        report['plans'] = {str(count): {key: value for key, value in wc.plan(grid.x, grid.y, count).items() if key not in ('work', 'shape')} for count in users}
        report['new_kernels_sha256'] = wc.source_sha256()
        rig = Rig(ttnn, mesh, torch, found, wc, chain)
        for count in users:
            for regime in regimes:
                for seed in seeds:
                    host = host_data(torch, count, regime, seed, found)
                    try:
                        section = compare_case(rig, host)
                    except BaseException as error:  # noqa: BLE001
                        section = dict(users=count, error='%s: %s' % (error.__class__.__name__, str(error)[:400]))
                    section.update(regime=regime, seed=seed)
                    sections.append(section)
                    print('GDN_WC compare users=%d regime=%s seed=%d differing=%s fell_back=%s' % (
                        count, regime, seed, section.get('differing', section.get('error')), section.get('fell_back')), flush=True)
        report['compare'] = sections
        first = host_data(torch, users[0], 'random', seeds[0], found)
        second = host_data(torch, users[0], 'random', seeds[-1] + 1, found)
        extras = {}
        for name, call in (('replay', lambda: replay_section(rig, first, second)), ('cache', lambda: cache_section(rig, first)),
                           ('control', lambda: control_section(rig, first))):
            try:
                extras[name] = call()
            except BaseException as error:  # noqa: BLE001
                extras[name] = dict(error='%s: %s' % (error.__class__.__name__, str(error)[:400]))
            print('GDN_WC %s %s' % (name, json.dumps(extras[name], sort_keys=True)), flush=True)
        report.update(extras)
        text, status = verdict(sections, extras['replay'], extras['cache'], extras['control'])
        if options.timing == 'on' and text == 'PASS':
            report['timing'] = []
            for count in users:
                result = time_case(rig, host_data(torch, count, 'random', seeds[0], found))
                report['timing'].append(result)
                print('GDN_WC timing users=%d served_us=%s wc_us=%s delta_us=%s' % (
                    count, result['served']['median_us'], result['wc']['median_us'], result['wc_minus_served_us']), flush=True)
        report['verdict'] = text
        print('GDN_WC verdict=%s sections=%d differing=%d' % (text, len(sections), sum(max(section.get('differing', 0), 0) for section in sections)), flush=True)
    except BaseException as error:  # noqa: BLE001
        report['verdict'] = 'NOT-RUN'
        report['error'] = '%s: %s' % (error.__class__.__name__, str(error)[:500])
        print('GDN_WC verdict=NOT-RUN error=%s' % report['error'], flush=True)
        status = 4
    finally:
        if mesh is not None:
            try:
                ttnn.close_mesh_device(mesh)
            except BaseException:  # noqa: BLE001
                pass
    with open(options.out, 'w') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, sort_keys=True), flush=True)
    timer.cancel()
    return status


if __name__ == '__main__':
    sys.exit(main())
