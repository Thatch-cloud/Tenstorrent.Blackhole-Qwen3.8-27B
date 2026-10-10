"""WPH-3 on ONE card: the verify and collect read-backs read three ways, byte for byte against the served per-shard blocking reads, and timed.

Run with QWEN_FAST_TP=4 in the environment (the vocabulary shard and the candidate chunks are the four-card geometry's). One p150a, a 1x1 mesh, no model and no collective: a
mesh tensor of one shard. What that proves is the part of the lever that does not depend on four chips: the three bindings exist at the pinned build and move the same bytes, and
what one blocking read costs against one non-blocking copy and one fence. What it cannot prove is the four-chip composition (one blocking `from_device` of four shards against four
blocking reads), which is the audited attach's and the timed ABAB's job; the report says so and extrapolates only by the arithmetic it names.

THE THREE READS (batched_reads_tp.py, the very functions the lever calls, with chips=1):
  served    every tensor's shards, one blocking ttnn.to_torch each (the shipped read);
  compose   one ttnn.to_torch(tensor, mesh_composer=ConcatMeshToTensor(mesh, dim=0)) a tensor (QWEN_FAST_BATCHED_READS=1);
  async     ttnn.copy_device_to_host_tensor(tensor, host, blocking=False) into host tensors allocated once (ttnn.allocate_tensor_on_host), ONE synchronize_device, then the same
            composition on the host tensors (QWEN_FAST_BATCHED_READS=async).

THE TENSOR SETS (rows = 64, the quad's block): `verify` is the ids (uint32) and the maxima (bfloat16), (1, 1, 1, 64) row-major, the two tensors shard_predictions reads; `collect` is one
quad's readback, per candidate chunk the top-16 values (bfloat16) and indices (uint32), (1, 1, 64, 16) tile, and the replicated selector features, (1, 1, 64, 256) bfloat16 tile
(2 chunks at the four-card shard, so 5 tensors; the round reads two quads). Seeded data is uploaded once and read every round, so a difference can only be the read.

BYTES. Every read of every tensor is compared with the served one as a bit pattern (dtype, shape and every element's bits). Exit 1 and `BATCHED_READS verdict=FAIL` when a read path that ran
differs; a path that RAISED (a binding that is missing or refuses the shape) is reported as `error` for that path and is not a FAIL of the others.

TIMING. Per set and path the wall time of one read of the whole set (all the tensors), over ROUNDS rounds in serpentine order (the paths alternate their order every round), median, p10,
p90, and the per-tensor figure; for the asynchronous path also its three parts apart (the enqueue loop, the one fence, the host conversion). Those are the numbers that carry to four chips:
`b` the blocking read of one shard (the served read, per tensor), `q` the enqueue of one non-blocking copy, `f` the fence. The report then PROJECTS the four-chip cost of each path by the
arithmetic it names and does not measure: served = chips x the one-shard served read; compose = the one-shard composed read + 3 extra shard enqueues a tensor (3 q n); async = the one-shard
async read + 3 q n (the fence is one fence whatever the shards). A path is a WIN when it is exact and its projection is at most (1 - WIN_FRACTION) of the served projection, a LOSS when it is
slower than the served projection, NEUTRAL between; `BATCHED_READS pick reads=1|async|none` names the exact WIN with the smallest projection (ties to compose, the binding already in use). The
four-chip ABAB decides the rest.

Exit: 0 PASS (every path that ran is exact and at least one ran); 1 FAIL; 3 the watchdog; 4 NOT-RUN (no path ran, or the device would not open). The last stdout line is one JSON object (kind
batched-reads-card-m) and the line above it 'BATCHED_READS verdict=...'.
"""

import argparse
import json
import os
import statistics
import sys
import threading
import time

KIND = 'batched-reads-card-m'
ROWS = 64
ROUNDS = 400
WARMUP = 20
WIN_FRACTION = 0.25
WATCHDOG_S = 1800
PATHS = ('served', 'compose', 'async')
SEED = 29


def summarize(values):
    ordered = sorted(values)
    pick = lambda fraction: ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]
    return dict(n=len(ordered), median_us=round(statistics.median(ordered) * 1e6, 2), p10_us=round(pick(0.10) * 1e6, 2), p90_us=round(pick(0.90) * 1e6, 2),
                min_us=round(ordered[0] * 1e6, 2))


def tensor_sets(chunks):
    """{set name: [(name, shape, dtype name, layout name, torch dtype name)]}: the tensors of each read-back, in the order the lever reads them."""
    verify = [('ids', (1, 1, 1, ROWS), 'uint32', 'ROW_MAJOR_LAYOUT', 'int32'), ('maxima', (1, 1, 1, ROWS), 'bfloat16', 'ROW_MAJOR_LAYOUT', 'bfloat16')]
    collect = []
    for number in range(chunks):
        collect.append(('values%d' % number, (1, 1, ROWS, 16), 'bfloat16', 'TILE_LAYOUT', 'bfloat16'))
        collect.append(('indices%d' % number, (1, 1, ROWS, 16), 'uint32', 'TILE_LAYOUT', 'int32'))
    collect.append(('features', (1, 1, ROWS, 256), 'bfloat16', 'TILE_LAYOUT', 'bfloat16'))
    return dict(verify=verify, collect=collect)


def host_value(torch, shape, torch_dtype, index):
    generator = torch.Generator().manual_seed(SEED + index)
    if torch_dtype == 'int32':
        return torch.randint(0, 62080, shape, generator=generator, dtype=torch.int32)
    return (torch.randn(shape, generator=generator) * 3.0).to(getattr(torch, torch_dtype))


def same_bits(torch, left, right):
    """Dtype, shape and every element's bit pattern (prestage_diff.same_bits)."""
    if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
        return False
    if left.is_floating_point():
        wide = {2: torch.int16, 4: torch.int32, 8: torch.int64}[left.element_size()]
        return bool(torch.equal(left.contiguous().view(wide), right.contiguous().view(wide)))
    return bool(torch.equal(left, right))


class Rig(object):
    """The device side. `ttnn` is the module (a fake in the tests), `reads` batched_reads_tp."""

    def __init__(self, ttnn, mesh, torch, reads):
        self.ttnn, self.mesh, self.torch, self.reads = ttnn, mesh, torch, reads

    def upload(self, specs):
        ttnn, mesh, torch = self.ttnn, self.mesh, self.torch
        tensors, hosts = [], []
        for index, (name, shape, dtype, layout, torch_dtype) in enumerate(specs):
            host = host_value(torch, shape, torch_dtype, index)
            tensors.append(ttnn.from_torch(host, dtype=getattr(ttnn, dtype), layout=getattr(ttnn, layout), device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                           mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)))
            hosts.append(host)
        return tensors, hosts

    def read(self, path, tensors):
        """The per-chip host tensors of `tensors` by `path`: [[chip 0, ...] per tensor]."""
        if path == 'served':
            return self.reads.served(self.ttnn, tensors)
        if path == 'compose':
            return self.reads.composed(self.ttnn, self.mesh, tensors, 1)
        return self.reads.asynchronous(self.ttnn, self.mesh, tensors, 1)


def release_all(rig, tensors):
    for tensor in tensors:
        try:
            rig.ttnn.deallocate(tensor)
        except BaseException:  # noqa: BLE001
            pass


def compare_set(rig, name, specs):
    """For each path, the bytes of one read of the set against the served read of the same tensors. {path: dict(exact, differing, error)}."""
    torch = rig.torch
    tensors, hosts = rig.upload(specs)
    found = {}
    try:
        reference = rig.read('served', tensors)
        uploaded = all(bool(torch.equal(reference[index][0].reshape(hosts[index].shape).to(torch.float64), hosts[index].to(torch.float64))) for index in range(len(hosts)))
        found['served'] = dict(exact=True, upload_unaltered=bool(uploaded))
        for path in ('compose', 'async'):
            try:
                got = rig.read(path, tensors)
                differing = [specs[index][0] for index in range(len(specs))
                             if len(got[index]) != len(reference[index]) or not all(same_bits(torch, a, b) for a, b in zip(got[index], reference[index]))]
                found[path] = dict(exact=not differing, differing=differing)
            except BaseException as error:  # noqa: BLE001
                found[path] = dict(exact=None, error='%s: %s' % (error.__class__.__name__, str(error)[:300]))
    finally:
        release_all(rig, tensors)
    return found


def time_set(rig, specs, paths, rounds=None, warmup=None, clock=None):
    """Per path, the wall time of one read of the whole set over serpentine rounds: {path: summary with per_tensor_us}."""
    rounds = ROUNDS if rounds is None else rounds
    warmup = WARMUP if warmup is None else warmup
    clock = time.perf_counter if clock is None else clock
    tensors, hosts = rig.upload(specs)
    samples = {path: [] for path in paths}
    try:
        for _ in range(warmup):
            for path in paths:
                rig.read(path, tensors)
        for index in range(rounds):
            order = list(paths) if index % 2 == 0 else list(reversed(paths))
            for path in order:
                started = clock()
                rig.read(path, tensors)
                samples[path].append(clock() - started)
    finally:
        release_all(rig, tensors)
        rig.reads.reset()
    summary = {path: summarize(values) for path, values in samples.items()}
    for path in summary:
        summary[path]['per_tensor_us'] = round(summary[path]['median_us'] / len(specs), 2)
    return summary


def time_async_parts(rig, specs, rounds=None, warmup=None, clock=None):
    """The asynchronous read's three parts apart, over `rounds` rounds: the enqueue loop (one non-blocking copy a tensor), the one fence and the host conversion (to_torch on the host tensors
    and the cut). {enqueue, fence, convert: summary}; `enqueue` is q times the tensor count."""
    rounds = ROUNDS if rounds is None else rounds
    warmup = WARMUP if warmup is None else warmup
    clock = time.perf_counter if clock is None else clock
    ttnn, mesh, reads = rig.ttnn, rig.mesh, rig.reads
    tensors, hosts = rig.upload(specs)
    enqueue, fence, convert = [], [], []
    try:
        targets = [reads.host_for(ttnn, mesh, tensor) for tensor in tensors]
        composer = reads.composer_for(ttnn, mesh)
        for index in range(warmup + rounds):
            first = clock()
            for tensor, target in zip(tensors, targets):
                ttnn.copy_device_to_host_tensor(tensor, target, blocking=False)
            second = clock()
            ttnn.synchronize_device(mesh)
            third = clock()
            for target in targets:
                reads.split(ttnn.to_torch(target, mesh_composer=composer), 1)
            fourth = clock()
            if index >= warmup:
                enqueue.append(second - first)
                fence.append(third - second)
                convert.append(fourth - third)
    finally:
        release_all(rig, tensors)
        reads.reset()
    return dict(enqueue=summarize(enqueue), fence=summarize(fence), convert=summarize(convert), tensors=len(specs))


def project(collect, chips):
    """The four-chip projection of each path (microseconds) from the one-shard figures: see the module docstring. {served, compose, async} (a path that did not run is absent)."""
    served = collect['served']['median_us']
    out = dict(served=chips * served)
    parts = collect.get('async_parts')
    # q (the enqueue of one non-blocking copy) is what an extra shard costs; without it (the asynchronous path did not run) an extra shard is priced as a whole blocking read, so the
    # composed path cannot be projected to win and the pick is none.
    extra = (chips - 1) * (parts['enqueue']['median_us'] if parts else served)
    if 'compose' in collect:
        out['compose'] = collect['compose']['median_us'] + extra
    if 'async' in collect and parts:
        out['async'] = collect['async']['median_us'] + extra
    return out


def verdicts(timing, chips=4):
    """({compose, async: WIN | NEUTRAL | LOSS | NOT-RUN | NOT-TIMED}, pick) on the collect set from `project`: a path wins when its projection is at most (1 - WIN_FRACTION) of the served
    projection, loses when it is slower than it; the pick is the winner with the smallest projection (ties to compose), else none."""
    collect = timing.get('collect')
    if collect is None or 'served' not in collect:
        return dict(compose='NOT-TIMED', **{'async': 'NOT-TIMED'}), 'none'
    projected = project(collect, chips)
    out = {}
    for path in ('compose', 'async'):
        if path not in projected:
            out[path] = 'NOT-RUN'
            continue
        out[path] = 'WIN' if projected[path] <= (1 - WIN_FRACTION) * projected['served'] else ('LOSS' if projected[path] > projected['served'] else 'NEUTRAL')
    winners = [(projected[path], 0 if path == 'compose' else 1, path) for path in ('compose', 'async') if out[path] == 'WIN']
    pick = 'none' if not winners else ('1' if min(winners)[2] == 'compose' else 'async')
    return out, pick


def verdict(compares):
    """(PASS | FAIL | NOT-RUN, exit status): FAIL when a path that ran differs; NOT-RUN when no batched path ran at all."""
    ran = [item[path] for item in compares.values() for path in ('compose', 'async') if path in item and item[path].get('exact') is not None]
    if any(not item['exact'] for item in ran):
        return 'FAIL', 1
    if not ran:
        return 'NOT-RUN', 4
    return 'PASS', 0


def main(argv=None, torch=None, ttnn=None, reads=None, tp_shapes=None, draft_shared_head_tp=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', required=True)
    parser.add_argument('--rounds', type=int, default=ROUNDS)
    parser.add_argument('--warmup', type=int, default=WARMUP)
    parser.add_argument('--sets', default='verify,collect')
    parser.add_argument('--timing', choices=('on', 'always', 'off'), default='on')
    options = parser.parse_args(argv)
    sets = options.sets.split(',')
    if any(name not in ('verify', 'collect') for name in sets) or options.rounds < 1:
        print('refusing: sets are verify and collect, rounds at least 1', file=sys.stderr)
        return 2
    timer = threading.Timer(WATCHDOG_S, lambda: (print('BATCHED_READS watchdog', flush=True), os._exit(3)))
    timer.daemon = True
    timer.start()
    if torch is None:
        import torch
        import ttnn
        import batched_reads_tp as reads
        import draft_shared_head_tp
        import tp_shapes
    report = dict(kind=KIND, sets=sets, rows=ROWS, rounds=options.rounds, environment=dict(QWEN_FAST_TP=os.environ.get('QWEN_FAST_TP')))
    mesh, status, compares, rig = None, 4, {}, None
    try:
        if os.environ.get('QWEN_FAST_TP') != '4':
            raise RuntimeError('QWEN_FAST_TP=4 required (the candidate chunks are the four-card shard\'s)')
        chunks = len(draft_shared_head_tp.candidate_chunks())
        report['chunks'] = chunks
        report['bindings'] = {name: bool(callable(getattr(ttnn, name, None))) for name in ('to_torch', 'ConcatMeshToTensor', 'copy_device_to_host_tensor', 'allocate_tensor_on_host',
                                                                                               'synchronize_device')}
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
        rig = Rig(ttnn, mesh, torch, reads)
        specs = tensor_sets(chunks)
        for name in sets:
            compares[name] = compare_set(rig, name, specs[name])
            for path in ('compose', 'async'):
                item = compares[name][path]
                print('BATCHED_READS compare set=%s path=%s exact=%s differing=%s error=%s' % (name, path, item.get('exact'), ','.join(item.get('differing', [])) or '-',
                                                                                        item.get('error', '-')), flush=True)
        report['compare'] = compares
        text, status = verdict(compares)
        if options.timing == 'always' or (options.timing == 'on' and text == 'PASS'):
            report['timing'] = {}
            for name in sets:
                paths = ['served'] + [path for path in ('compose', 'async') if compares[name][path].get('exact')]
                report['timing'][name] = time_set(rig, specs[name], paths, options.rounds, options.warmup)
                report['timing'][name]['tensors'] = len(specs[name])
                if 'async' in paths:
                    report['timing'][name]['async_parts'] = time_async_parts(rig, specs[name], options.rounds, options.warmup)
                print('BATCHED_READS timing set=%s %s' % (name, ' '.join('%s_us=%s' % (path, report['timing'][name][path]['median_us']) for path in paths)), flush=True)
                if 'async_parts' in report['timing'][name]:
                    parts = report['timing'][name]['async_parts']
                    print('BATCHED_READS async_parts set=%s enqueue_us=%s fence_us=%s convert_us=%s tensors=%d' % (
                        name, parts['enqueue']['median_us'], parts['fence']['median_us'], parts['convert']['median_us'], parts['tensors']), flush=True)
            report['timing_verdict'], report['pick'] = verdicts(report['timing'], tp_shapes.chip_count())
            if 'collect' in report['timing']:
                report['projected_four_chip_us'] = project(report['timing']['collect'], tp_shapes.chip_count())
                print('BATCHED_READS projected collect (four chips, microseconds, arithmetic not measurement) %s' % ' '.join(
                    '%s=%.1f' % pair for pair in sorted(report['projected_four_chip_us'].items())), flush=True)
            print('BATCHED_READS timing_verdict compose=%s async=%s' % (report['timing_verdict']['compose'], report['timing_verdict']['async']), flush=True)
            print('BATCHED_READS pick reads=%s' % report['pick'], flush=True)
        report['verdict'] = text
        print('BATCHED_READS verdict=%s sets=%d' % (text, len(sets)), flush=True)
    except BaseException as error:  # noqa: BLE001
        report['verdict'] = 'NOT-RUN'
        report['error'] = '%s: %s' % (error.__class__.__name__, str(error)[:500])
        print('BATCHED_READS verdict=NOT-RUN error=%s' % report['error'], flush=True)
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
