"""F-F2 on ONE card: the drafter permutation launches against the SERVED composition's bytes, on edge data, and timed.

The served K/V assembly (draft_attention_branch.served_key_value), query fold and output unfold (quad_draft_tp, pair_row_exact_tp, octo_draft_tp) run as the served stack runs them, on real
ttnn ops, on one copy of the same host data; draft_permute_tp's launches (QWEN_FAST_DRAFT_PERMUTE) run on another. Every output pair is read back and compared as int16 bit patterns
(-0 and +0 differ), the in-trace audit's rule. Run with QWEN_FAST_TP=4 in the environment (the shard geometry is the four-card one: 8 query and 2 KV heads); one p150a, a 1x1 mesh, no
model and no collective: the launch's program is per chip, so one chip proves the kernels compile, run and move the same bytes. Whether four chips of a mesh agree is the audited
attach's job (every chip, on real data).

THE QUESTION THE CPU CANNOT ANSWER. The CPU tests prove the kernel equals a MODEL of the served composition (a row-axis slice that starts inside a tile and a row-axis concat with a piece that
is not a whole number of tiles canonicalise their output - a zero exponent becomes +0 - and nothing else does). The model is read off the device profile (the quad's concat untilizes the cached
banks too); this probe reads the bytes. It carries the regimes that can tell: 'edge' (random finite patterns, half of them +-0 and denormals of both signs, in the cached banks, the live block and the
queries), 'sweep' (every one of the 65,536 bfloat16 bit patterns, infinities and NaNs included, laid across each tensor) and 'random' (normal values: the rule is invisible on them, and the launch must
still be exact). Beside the verdict arm (cached banks canonical) the K/V sites run the OTHER reading (cached banks raw) as information: if the verdict arm fails and the other matches, the served
concat does not round-trip the banks and `canon_cached` flips; the report says which.

Cases: K/V assembly for the quad, the pair (two 2,048-row histories), the pair at mixed histories (256 and 1,024) and the octo plan; fold and unfold for the pair, the quad and the octo block. Timing: per
case and arm, a captured trace of LAUNCHES launches (eager timing would measure the host's program build), serpentine rounds of replays; the per-launch median and IQR of each arm, and the grid the
launches covered (11 x 10 or 13 x 10, read from the device).

Exit: 0 PASS (every compare exact, no launch fell back); 1 FAIL; 3 the watchdog; 4 NOT-RUN (a section raised). The last stdout line is one JSON object (kind draft-permute-card-m) and the line above
it 'DRAFT_PERMUTE verdict=...'.
"""

import argparse
import json
import os
import statistics
import sys
import threading
import time

KIND = 'draft-permute-card-m'
SITES = ('kv', 'fold', 'unfold')
SHAPES = ('quad', 'pair', 'pair-short', 'octo')
REGIMES = ('random', 'edge', 'sweep')
LAUNCHES = 8
ROUNDS = 15
WATCHDOG_S = 2400
TRACE_REGION = 256 * 1024 * 1024
EDGE_BITS = (0x0000, 0x8000, 0x0080, 0x8080, 0x0001, 0x8001, 0x007F, 0x807F, 0x7F7F, 0xFF7F, 0x3F80, 0xBF80)
GEOMETRY = {'pair': dict(halves=1, users=2, block=16), 'quad': dict(halves=2, users=2, block=16), 'octo': dict(halves=2, users=4, block=8)}
KV_HEADS, QUERY_HEADS, HEAD_DIM = 2, 8, 128
PATTERNS = 65536


class FellBack(Exception):
    """A launch handed the call to the served ops (its marker says why): a verdict FAIL, not a comparison."""


def plan_of(shape):
    """(plan, cache rows per user, live rows) of one K/V case, from the REAL plan builders."""
    if shape == 'quad':
        import quad_draft

        return quad_draft.key_value_plan([2048] * 4, 16)[0], [2048] * 4, 64
    if shape == 'octo':
        import octo_draft_tp

        return octo_draft_tp.key_value_plan([2048] * 8, 16)[0], [2048] * 8, 64
    from dflash_batched_mask import key_value_plan

    if shape == 'pair':
        return key_value_plan([2048, 2048], 16)[0], [2048, 2048], 32
    if shape == 'pair-short':
        return key_value_plan([256, 1024], 16)[0], [256, 1024], 32
    raise ValueError(shape)


def host_bits(torch, shape, regime, generator, offset=0):
    """A seeded int16 tensor of bfloat16 bit patterns: 'random' normal values; 'edge' random finite patterns, half of them +-0 and denormals (and the largest finite, +-1); 'sweep' the flat index plus
    `offset` modulo 65,536, so every pattern appears (and in a different tile from one tensor to the next)."""
    if regime == 'random':
        return torch.randn(shape, generator=generator).to(torch.bfloat16).contiguous().view(torch.int16)
    count = 1
    for extent in shape:
        count *= extent
    if regime == 'sweep':
        raw = (torch.arange(count, dtype=torch.int64) + offset) % PATTERNS
        return (raw - ((raw & 0x8000) << 1)).to(torch.int16).reshape(shape)
    raw = torch.randint(0, PATTERNS, shape, generator=generator, dtype=torch.int32)
    pick = torch.randint(0, len(EDGE_BITS), shape, generator=generator, dtype=torch.int32)
    edge = torch.tensor(EDGE_BITS, dtype=torch.int32)[pick]
    raw = torch.where(torch.rand(shape, generator=generator) < 0.5, edge, raw)
    raw = torch.where((raw & 0x7F80) == 0x7F80, (raw & 0x8000) | 0x3F80, raw)             # no infinity, no NaN
    return (raw - ((raw & 0x8000) << 1)).to(torch.int16)


def summarize(values):
    ordered = sorted(values)
    quarter = lambda fraction: ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]
    return dict(n=len(ordered), median_us=round(statistics.median(ordered), 2), q1_us=round(quarter(0.25), 2), q3_us=round(quarter(0.75), 2),
                min_us=round(ordered[0], 2))


def differing(torch, left, right):
    if tuple(left.shape) != tuple(right.shape):
        return max(left.numel(), right.numel())
    return int((left != right).sum())


def locate(torch, left, right, plan, limit=6):
    """The first `limit` differing (head, row) pairs of two (1, heads, rows, 128) int16 tensors, which plan piece each row belongs to, and the first differing values (unsigned hex)."""
    if tuple(left.shape) != tuple(right.shape):
        return [dict(shape_left=list(left.shape), shape_right=list(right.shape))]
    pieces, offset = [], 0
    for part in plan:
        pieces.append((offset, offset + part['rows'], part['kind'], part['user']))
        offset += part['rows']
    found = []
    for head, row in (left != right).any(dim=3)[0].nonzero().tolist()[:limit]:
        kind, user = next(((kind, user) for low, high, kind, user in pieces if low <= row < high), ('?', -1))
        column = int((left[0, head, row] != right[0, head, row]).nonzero()[0])
        found.append(dict(head=head, row=row, kind=kind, user=user, column=column, mine='0x%04X' % (int(left[0, head, row, column]) & 0xFFFF),
                          served='0x%04X' % (int(right[0, head, row, column]) & 0xFFFF)))
    return found


def verdict(sections):
    """(PASS | FAIL | NOT-RUN, exit status): every compare exact and no launch fell back is a PASS."""
    if not sections or any(section.get('error') for section in sections):
        return 'NOT-RUN', 4
    if any(section.get('differing', 1) or section.get('fell_back') for section in sections):
        return 'FAIL', 1
    return 'PASS', 0


class Rig(object):
    """The device side: upload, run the served arm and the launch, read back. `ttnn` is the module (a fake in the tests); `perm` is draft_permute_tp."""

    def __init__(self, ttnn, mesh, torch, perm, processors):
        self.ttnn, self.mesh, self.torch, self.perm, self.processors = ttnn, mesh, torch, perm, processors
        self.owned = []

    def upload(self, value):
        ttnn, mesh = self.ttnn, self.mesh
        made = ttnn.from_torch(value.contiguous().view(self.torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        self.owned.append(made)
        return made

    def retain(self, tensor):
        self.owned.append(tensor)
        return tensor

    def read(self, tensor):
        ttnn, torch = self.ttnn, self.torch
        return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).contiguous().view(torch.int16)

    def release(self):
        seen = set()
        for tensor in self.owned:
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            try:
                self.ttnn.deallocate(tensor)
            except BaseException:  # noqa: BLE001
                pass
        self.owned = []

    @staticmethod
    def refuse(*arguments):
        raise FellBack()

    # --- K/V ---------------------------------------------------------------------------------------------------------------
    def kv_operands(self, shape, regime, seed):
        torch = self.torch
        plan, rows, live_rows = plan_of(shape)
        generator = torch.Generator().manual_seed(seed)
        host_caches = [{name: host_bits(torch, (1, KV_HEADS, rows[user], HEAD_DIM), regime, generator, offset=7919 * (2 * user + index))
                        for index, name in enumerate('kv')} for user in range(len(rows))]
        host_live = {name: host_bits(torch, (1, KV_HEADS, live_rows, HEAD_DIM), regime, generator, offset=104729 * (index + 1)) for index, name in enumerate('kv')}
        caches = [{name: self.upload(value) for name, value in cache.items()} for cache in host_caches]
        live = {name: self.upload(value) for name, value in host_live.items()}
        return plan, caches, live

    def kv_served(self, plan, caches, live, name):
        import draft_attention_branch

        return draft_attention_branch.served_key_value(self.ttnn, plan, caches, live, self.retain, name)

    def kv_engaged(self, plan, caches, live, shape, canon_cached=True):
        return self.perm.assemble_kv(self.ttnn, plan, caches, live, self.retain, served=self.refuse, site=shape, canon_cached=canon_cached,
                                     processors=self.processors, chips=1)

    # --- fold and unfold ---------------------------------------------------------------------------------------------------
    def fold_operands(self, site, shape, regime, seed):
        torch = self.torch
        geometry = GEOMETRY[shape]
        generator = torch.Generator().manual_seed(seed)
        if site == 'fold':
            host = host_bits(torch, (1, QUERY_HEADS, 32 * geometry['halves'], HEAD_DIM), regime, generator, offset=15485863)
        else:
            heads = KV_HEADS * geometry['halves'] * geometry['users'] * (QUERY_HEADS // KV_HEADS)
            host = host_bits(torch, (1, heads, 32, HEAD_DIM), regime, generator, offset=32452843)
        return self.upload(host)

    def fold_served(self, site, shape, tensor):
        import octo_draft_tp
        import pair_row_exact_tp
        import quad_draft_tp

        function = {('fold', 'pair'): pair_row_exact_tp.fold_query, ('unfold', 'pair'): pair_row_exact_tp.unfold_output,
                    ('fold', 'quad'): quad_draft_tp.quad_fold_query, ('unfold', 'quad'): quad_draft_tp.quad_unfold_output,
                    ('fold', 'octo'): octo_draft_tp.octo_fold_query, ('unfold', 'octo'): octo_draft_tp.octo_unfold_output}[(site, shape)]
        return function(self.ttnn, tensor, self.retain)

    def fold_engaged(self, site, shape, tensor):
        function = self.perm.fold_query if site == 'fold' else self.perm.unfold_output
        return function(self.ttnn, tensor, self.retain, served=self.refuse, site=shape, processors=self.processors, chips=1, **GEOMETRY[shape])


def compare_kv(rig, shape, regime, seed, canon_cached=True):
    """One K/V section: the served assembly and the launch on the same host data, K and V, bit for bit."""
    plan, caches, live = rig.kv_operands(shape, regime, seed)
    served = {name: rig.read(rig.kv_served(plan, caches, live, name)) for name in 'kv'}
    try:
        engaged = rig.kv_engaged(plan, caches, live, shape, canon_cached=canon_cached)
    except FellBack:
        return dict(site='kv', shape=shape, differing=-1, fell_back=True)
    mine = {name: rig.read(engaged[name]) for name in 'kv'}
    counts = {name: differing(rig.torch, mine[name], served[name]) for name in 'kv'}
    section = dict(site='kv', shape=shape, differing=sum(counts.values()), by_tensor=counts, fell_back=False, canon_cached=canon_cached)
    if section['differing']:
        section['first'] = {name: locate(rig.torch, mine[name], served[name], plan) for name in 'kv' if counts[name]}
    return section


def compare_fold(rig, site, shape, regime, seed):
    tensor = rig.fold_operands(site, shape, regime, seed)
    served = rig.read(rig.fold_served(site, shape, tensor))
    try:
        engaged = rig.fold_engaged(site, shape, tensor)
    except FellBack:
        return dict(site=site, shape=shape, differing=-1, fell_back=True)
    mine = rig.read(engaged)
    count = differing(rig.torch, mine, served)
    section = dict(site=site, shape=shape, differing=count, fell_back=False)
    if count:
        mask = (mine != served).any(dim=3)[0] if tuple(mine.shape) == tuple(served.shape) else None
        if mask is not None:
            section['first_heads_rows'] = mask.nonzero().tolist()[:8]
    return section


def capture(rig, run, launches):
    """Capture `launches` back-to-back launches of one arm in one trace (warmed first); returns (handle, the tensors the capture allocated)."""
    ttnn, mesh = rig.ttnn, rig.mesh
    handle = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        try:
            for _ in range(launches):
                run()
        finally:
            ttnn.end_trace_capture(mesh, handle, cq_id=0)
    except BaseException:
        ttnn.release_trace(mesh, handle)
        raise
    return handle


def time_case(rig, site, shape, regime, seed, launches=LAUNCHES, rounds=ROUNDS, clock=time.perf_counter):
    """Per-launch microseconds of the served arm and the launch, each under a captured trace, over serpentine rounds of replays."""
    ttnn, mesh = rig.ttnn, rig.mesh
    if site == 'kv':
        plan, caches, live = rig.kv_operands(shape, regime, seed)
        runs = dict(served=lambda: [rig.kv_served(plan, caches, live, name) for name in 'kv'], launch=lambda: rig.kv_engaged(plan, caches, live, shape))
    else:
        tensor = rig.fold_operands(site, shape, regime, seed)
        runs = dict(served=lambda: rig.fold_served(site, shape, tensor), launch=lambda: rig.fold_engaged(site, shape, tensor))
    samples = dict(served=[], launch=[])
    traces = {}
    try:
        for name in runs:
            runs[name]()                          # the warm call: compile outside the capture
            ttnn.synchronize_device(mesh)
            traces[name] = capture(rig, runs[name], launches)
            ttnn.synchronize_device(mesh)
        for name in runs:
            ttnn.execute_trace(mesh, traces[name], cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        for index in range(rounds):
            for name in (('served', 'launch') if index % 2 == 0 else ('launch', 'served')):
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
    served, launch = summarize(samples['served']), summarize(samples['launch'])
    return dict(site=site, shape=shape, mode='trace', served=served, launch=launch, launch_minus_served_us=round(launch['median_us'] - served['median_us'], 2))


def main(argv=None, torch=None, ttnn=None, perm=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', required=True)
    parser.add_argument('--sites', default=','.join(SITES))
    parser.add_argument('--shapes', default=','.join(SHAPES))
    parser.add_argument('--regimes', default=','.join(REGIMES))
    parser.add_argument('--seeds', default='17')
    parser.add_argument('--timing', choices=('on', 'off'), default='on')
    parser.add_argument('--processors', type=int, choices=(1, 2), default=2)
    options = parser.parse_args(argv)
    sites, shapes, regimes = options.sites.split(','), options.shapes.split(','), options.regimes.split(',')
    seeds = [int(value) for value in options.seeds.split(',')]
    if any(value not in SITES for value in sites) or any(value not in SHAPES for value in shapes) or any(value not in REGIMES for value in regimes):
        print('refusing: sites %s, shapes %s, regimes %s' % (' '.join(SITES), ' '.join(SHAPES), ' '.join(REGIMES)), file=sys.stderr)
        return 2
    timer = threading.Timer(WATCHDOG_S, lambda: (print('DRAFT_PERMUTE watchdog', flush=True), os._exit(3)))
    timer.daemon = True
    timer.start()
    if torch is None:
        import torch
        import ttnn
        import draft_permute_tp as perm
    report = dict(kind=KIND, sites=sites, shapes=shapes, regimes=regimes, seeds=seeds, processors=options.processors,
                  environment=dict(QWEN_FAST_TP=os.environ.get('QWEN_FAST_TP')))
    mesh, rig, status, sections = None, None, 4, []
    try:
        if os.environ.get('QWEN_FAST_TP') != '4':
            raise RuntimeError('QWEN_FAST_TP=4 required (the geometry is the four-card shard\'s)')
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=TRACE_REGION)
        grid = mesh.compute_with_storage_grid_size()
        report['grid'] = [grid.x, grid.y]
        rig = Rig(ttnn, mesh, torch, perm, options.processors)
        for site in sites:
            for shape in shapes:
                if site != 'kv' and shape == 'pair-short':
                    continue
                for regime in regimes:
                    for seed in seeds:
                        try:
                            section = compare_kv(rig, shape, regime, seed) if site == 'kv' else compare_fold(rig, site, shape, regime, seed)
                        except BaseException as error:  # noqa: BLE001
                            section = dict(site=site, shape=shape, error='%s: %s' % (error.__class__.__name__, str(error)[:400]))
                        finally:
                            rig.release()
                        section.update(regime=regime, seed=seed)
                        sections.append(section)
                        print('DRAFT_PERMUTE compare site=%s shape=%s regime=%s seed=%d differing=%s fell_back=%s' % (
                            site, shape, regime, seed, section.get('differing', section.get('error')), section.get('fell_back')), flush=True)
        report['compare'] = sections
        text, status = verdict(sections)
        if 'kv' in sites:
            # information: the OTHER reading of the served concat (cached banks raw), on the data that can tell
            report['cached_raw_reading'] = []
            for shape in shapes:
                for regime in ('edge', 'sweep'):
                    if regime not in regimes:
                        continue
                    try:
                        other = compare_kv(rig, shape, regime, seeds[0], canon_cached=False)
                    except BaseException as error:  # noqa: BLE001
                        other = dict(site='kv', shape=shape, error='%s: %s' % (error.__class__.__name__, str(error)[:300]))
                    finally:
                        rig.release()
                    other.update(regime=regime)
                    report['cached_raw_reading'].append(other)
            canonical = [section for section in sections if section.get('site') == 'kv' and section.get('regime') in ('edge', 'sweep')]
            raw = report['cached_raw_reading']
            report['cached_banks'] = ('canonical (the model holds)' if canonical and all(not section.get('differing') for section in canonical)
                                      else 'raw (flip canon_cached)' if raw and all(section.get('differing', 1) == 0 and not section.get('fell_back') for section in raw)
                                      else 'neither reading matches the served bytes')
            print('DRAFT_PERMUTE cached banks: %s' % report['cached_banks'], flush=True)
        if options.timing == 'on' and text == 'PASS':
            report['timing'] = []
            for site in sites:
                for shape in shapes:
                    if site != 'kv' and shape == 'pair-short':
                        continue
                    try:
                        result = time_case(rig, site, shape, 'random', seeds[0])
                    except BaseException as error:  # noqa: BLE001
                        result = dict(site=site, shape=shape, error='%s: %s' % (error.__class__.__name__, str(error)[:300]))
                    finally:
                        rig.release()
                    report['timing'].append(result)
                    print('DRAFT_PERMUTE timing site=%s shape=%s served_us=%s launch_us=%s delta_us=%s' % (
                        site, shape, result.get('served', {}).get('median_us'), result.get('launch', {}).get('median_us'),
                        result.get('launch_minus_served_us')), flush=True)
        report['verdict'] = text
        print('DRAFT_PERMUTE verdict=%s sections=%d differing=%d' % (text, len(sections), sum(max(section.get('differing', 0), 0) for section in sections)), flush=True)
    except BaseException as error:  # noqa: BLE001
        report['verdict'] = 'NOT-RUN'
        report['error'] = '%s: %s' % (error.__class__.__name__, str(error)[:500])
        print('DRAFT_PERMUTE verdict=NOT-RUN error=%s' % report['error'], flush=True)
        status = 4
    finally:
        if rig is not None:
            rig.release()
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
