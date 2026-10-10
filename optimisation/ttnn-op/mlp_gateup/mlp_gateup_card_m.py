"""WP4 card-M timing harness: the TP4 per-chip MLP decode matmuls at 64 rows, swept over program configs that keep the K loop, then composed.

Run with QWEN_FAST_TP=4 in the environment. One p150a, a 1x1 mesh, no model and no collective: the matmuls are per chip, so one chip settles streaming
efficiency and byte equality; whether four chips agree is the audited attach's job (the lever's own audit, every chip).

WHAT IT SETTLES. The packed verify streams the MLP weights well below the DRAM peak (v676/v678: gate 236, up 284, down 357 GB/s of 512). The plan
(F-D1, F-D2) asks whether that gap is the program CONFIG (a re-partition of the same launches, about 0.35 ms a pass) or needs the FUSED op (gate and
up as one launch with the SwiGLU epilogue, up to 2.1 ms a pass). The sweep times, per shape, the served config and every other config that keeps
in0_block_w, per_core_M, fuse_batch and mcast_in0 (so every output tile is reduced over K in the same blocks and the bytes cannot move): per_core_N
from the least that fits the device grid to the most, on the device's full width and, in a second stage, on narrower widths. The grid is read from
the device (130 cores on 13x10, 110 on 11x10); nothing here knows 11x10. Per candidate it reports microseconds, GB/s (the padded weight bytes over the
time), the fraction of the 512 GB/s peak and of the best row anywhere, and whether the output equals the served config's, bit for bit, on random and
on edge data. Timing is a captured trace of back-to-back launches replayed in serpentine rounds against the served arm, per launch from the slope of a
short and a long trace (host dispatch and the replay's fixed cost drop out).

THEN THE COMPOSITION (--arms includes compose): the served five-launch chain (gate, up, multiply written to DRAM, down; rotating four weight sets, the
way 64 layers do) against the same chain with the multiply written to L1 (F-D2), and against the best exact config of each shape with the L1 multiply
(the QWEN_FAST_MLP_CFG name this run prints). With --arms fused it also times the fused gate|up op (tp4_mlp_fused) at each pairs-per-worker and compares
its product with the served multiply's.

THE RULE (pre-registered). With layers = 64 and ceiling = the best exact GB/s of any shape:
  config gain      = layers x sum over gate and up of (served us - best exact us)
  fused potential  = layers x (best exact gate + best exact up + multiply us - (gate + up bytes at ceiling))
  CONFIG-ENOUGH    config gain >= 0.35 ms and fused potential < 0.35 ms: ship the QWEN_FAST_MLP_CFG name; the fused op is not worth building
  BUILD-FUSED      fused potential >= 0.35 ms: the config leaves at least the plan's lower bound on the table; build and time the fused op
  NEITHER          both under 0.35 ms: the streaming is at the ceiling; F-D2 (the L1 multiply) is all there is
0.35 ms a pass is the plan's own low end for F-D1 (the multiply, the launch and the DRAM round trip).

Exit: 0 PASS (the served arm reproduced itself and every shape produced a result); 1 FAIL (the served control differed from itself or a shape had no
exact result); 3 the watchdog; 4 NOT-RUN. The last stdout line is one JSON object (kind mlp-gateup-card-m); the lines above it start MLP_GATEUP.
"""

import argparse
import json
import math
import os
import statistics
import sys
import threading
import time

KIND = 'mlp-gateup-card-m'
WATCHDOG_S = 3300
TRACE_REGION = 128 * 1024 * 1024
SHORT, LONG = 8, 64
ROUNDS = 9
COMPOSE_SETS = 4
COMPOSE_SHORT, COMPOSE_LONG = 4, 24
DEFAULT_BUDGET_S = 2100
LAYERS = 64
THRESHOLD_MS = 0.35
REGIMES = ('random', 'edge')
DEFAULT_SHAPES = 'mlp_w1,mlp_w3,mlp_w2'
ARMS = ('sweep', 'compose', 'fused', 'probe', 'split')
WIDTH_CHOICES = (12, 11, 10, 8)
STAGE2_TOP = 3


# ---------------------------------------------------------------------------------------------------------------------------
# Pure parts (no ttnn, no torch): candidates, summaries, the rule.
# ---------------------------------------------------------------------------------------------------------------------------

def summarize(values):
    ordered = sorted(values)
    quarter = lambda fraction: ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]
    return dict(n=len(ordered), median_us=round(statistics.median(ordered), 3), q1_us=round(quarter(0.25), 3),
                q3_us=round(quarter(0.75), 3), min_us=round(ordered[0], 3))


def slope_us(short_samples, long_samples, short=SHORT, long=LONG):
    """Per-launch microseconds from the replay wall time of a short and a long trace (microseconds each): the difference of the medians over the
    difference of the launches, so the fixed cost of a replay (the enqueue and the synchronize) drops out."""
    if long <= short:
        raise ValueError('the long trace must hold more launches than the short one')
    return (statistics.median(long_samples) - statistics.median(short_samples)) / float(long - short)


def widths_for(grid_x, choices=WIDTH_CHOICES):
    """The grid widths stage 2 tries: the device's own first, then the narrower of `choices`."""
    return [grid_x] + [width for width in choices if width < grid_x]


def stage_one(lever, shape, grid_x, grid_y):
    """The first stage of one shape: every per_core_N that fits the grid, on the device's full width (the served partition's own width), as
    [{name, pcn, width, config, stage}]. The names are the lever's tokens: g<n> for the gate, u<n> for the up, d<n> for the down; the other
    shapes use their key."""
    token = dict(mlp_w1='g', mlp_w3='u', mlp_w2='d').get(shape.key, shape.key + '_')
    out = []
    for per_core_n in lever.per_core_n_choices(shape, grid_x, grid_y):
        config = lever.named_config(shape, per_core_n, grid_x)
        out.append(dict(name='%s%d' % (token, per_core_n), pcn=per_core_n, width=grid_x, config=config, stage=1))
    return out


def stage_two(lever, shape, grid_x, grid_y, best_pcns, seen):
    """The second stage: the best per_core_N values of stage one on the narrower widths (their rectangles differ, the active cores are the same
    row-major prefix). Names carry `@w<W>`: they are not QWEN_FAST_MLP_CFG names (the lever's w token is global); the report says so."""
    token = dict(mlp_w1='g', mlp_w3='u', mlp_w2='d').get(shape.key, shape.key + '_')
    out = []
    for per_core_n in best_pcns:
        for width in widths_for(grid_x)[1:]:
            config = lever.named_config(shape, per_core_n, width)
            key = (tuple(config.grid), config.per_core_N)
            if key in seen or config.grid[1] > grid_y:
                continue
            seen.add(key)
            out.append(dict(name='%s%d@w%d' % (token, per_core_n, width), pcn=per_core_n, width=width, config=config, stage=2))
    return out


def shape_table(lever):
    """The lever's shapes plus the diagnostic ones: the gate and up with bfloat8_b weights at the same K and N (gate_bf8, up_bf8), to tell what the format does from what the
    shape does (same tile count, twice the bytes)."""
    table = dict(lever.shapes(4))
    table['gate_bf8'] = table['mlp_w1']._replace(key='gate_bf8', dtype='bfp8')
    table['up_bf8'] = table['mlp_w3']._replace(key='up_bf8', dtype='bfp8')
    return table


def served_row(lever, shape, grid_x, t1_gate_cores=None):
    shape_cores = shape if t1_gate_cores is None or shape.key not in ('mlp_w1', 'gate_bf8') else shape._replace(requested_cores=t1_gate_cores)
    config = lever.builder_config(shape_cores, grid_x)
    return dict(name='served', pcn=config.per_core_N, width=grid_x, config=config, stage=0)


def row_of(lever, shape, candidate, timing, exact, differing=None, error=None):
    """One result row (JSON-able): the candidate's identity, the layout it ran, its time and bandwidth, and whether it equalled the served output."""
    config = candidate['config']
    row = dict(shape=shape.key, name=candidate['name'], stage=candidate['stage'], pcn=candidate['pcn'], width=candidate['width'],
               grid=list(config.grid), active_cores=lever.active_cores(config, shape), in0_block_w=config.in0_block_w,
               subblock=[config.out_subblock_h, config.out_subblock_w])
    if error is not None:
        row['error'] = error
        return row
    us = timing['us']
    row.update(us=round(us, 3), gbps=round(lever.gbps(shape, us), 2), pct_peak=round(100.0 * lever.gbps(shape, us) / lever.PEAK_GBPS, 1),
               exact=bool(exact), differing=differing, served_us=round(timing.get('served_us', 0.0), 3),
               delta_us=round(timing.get('delta_us', 0.0), 3))
    return row


def best_exact(rows):
    """The fastest row that is exact (the served row counts), or None."""
    exact = [row for row in rows if 'us' in row and row.get('exact') is True]
    return min(exact, key=lambda row: row['us']) if exact else None


def shape_summary(lever, shape, rows):
    """Served row, best exact row, best row of any exactness, and what the swap is worth, for one shape."""
    served = next((row for row in rows if row['name'] == 'served' and 'us' in row), None)
    best = best_exact(rows)
    timed = [row for row in rows if 'us' in row]
    anyone = min(timed, key=lambda row: row['us']) if timed else None
    inexact = [row['name'] for row in timed if row.get('exact') is not True]
    # the lever's names spell the default-width partitions only: the best exact one of those that beats the served config is what a profile can carry
    default = [row for row in rows if row.get('stage') == 1 and 'us' in row and row.get('exact') is True]
    best_default = min(default, key=lambda row: row['us']) if default else None
    if best_default and served and best_default['us'] >= served['us']:
        best_default = None
    out = dict(shape=shape.key, k=shape.k, n=shape.n, dtype=shape.dtype, bytes=lever.weight_bytes(shape), served=served, best_exact=best,
               best_default_width=best_default, best_any=anyone, inexact=inexact, rows=len(rows), timed=len(timed))
    if served and best:
        out['gain_us'] = round(served['us'] - best['us'], 3)
        out['gain_ms_pass'] = round(LAYERS * out['gain_us'] / 1000.0, 3)
    return out


def ceiling_gbps(summaries):
    """The best exact GB/s of any shape's rows: the streaming rate this device is shown to reach with these ops."""
    rates = [entry['best_exact']['gbps'] for entry in summaries.values() if entry.get('best_exact')]
    return max(rates) if rates else None


def combined_name(summaries):
    """The QWEN_FAST_MLP_CFG name of the best exact default-width row of the gate, up and down; `l1` where none beat the served config."""
    letters = dict(mlp_w1='g', mlp_w3='u', mlp_w2='d')
    tokens = {}
    for key, letter in letters.items():
        entry = summaries.get(key)
        row = entry and entry.get('best_default_width')
        if row:
            tokens[letter] = row['pcn']
    name = ''.join('%s%d' % (letter, tokens[letter]) for letter in 'gud' if letter in tokens)
    return name or 'l1'


def decide(summaries, lever, multiply_us=None, threshold_ms=THRESHOLD_MS, layers=LAYERS):
    """The pre-registered rule (module docstring). Returns the numbers and the verdict word."""
    gate, up = summaries.get('mlp_w1'), summaries.get('mlp_w3')
    ceiling = ceiling_gbps(summaries)
    if not (gate and up and gate.get('served') and up.get('served') and gate.get('best_exact') and up.get('best_exact') and ceiling):
        return dict(verdict='NO-RESULT', reason='the gate or the up has no served or exact row', ceiling_gbps=ceiling)
    config_gain = layers * ((gate['served']['us'] - gate['best_exact']['us']) + (up['served']['us'] - up['best_exact']['us'])) / 1000.0
    floor_us = (gate['bytes'] + up['bytes']) / (ceiling * 1e3)
    multiply = 0.0 if multiply_us is None else multiply_us
    now_us = gate['best_exact']['us'] + up['best_exact']['us'] + multiply
    potential = layers * max(0.0, now_us - floor_us) / 1000.0
    if potential >= threshold_ms:
        word = 'BUILD-FUSED'
    elif config_gain >= threshold_ms:
        word = 'CONFIG-ENOUGH'
    else:
        word = 'NEITHER'
    return dict(verdict=word, config_gain_ms_pass=round(config_gain, 3), fused_potential_ms_pass=round(potential, 3),
                ceiling_gbps=round(ceiling, 1), floor_us=round(floor_us, 2), now_us=round(now_us, 2), multiply_us=multiply_us,
                threshold_ms=threshold_ms, layers=layers, env=dict(QWEN_FAST_MLP_CFG=combined_name(summaries)))


# ---------------------------------------------------------------------------------------------------------------------------
# Host data.
# ---------------------------------------------------------------------------------------------------------------------------

def host_x(torch, rows, k, regime, seed):
    """The activation (1, 1, rows, k) as bfloat16: unit normal, or the edge regime (a third of the elements zero, rows alternating tiny and large
    magnitudes, so the K reduction sees denormal-adjacent and saturating terms)."""
    generator = torch.Generator().manual_seed(seed)
    value = torch.randn(1, 1, rows, k, generator=generator)
    if regime == 'edge':
        scale = torch.ones(rows, 1)
        scale[0::2] = 1e-20
        scale[1::4] = 1e3
        value = value * scale
        value = torch.where(torch.rand(1, 1, rows, k, generator=generator) < 0.33, torch.zeros_like(value), value)
    return value.to(torch.bfloat16)


def host_w(torch, k, n, seed, regime='random'):
    """The weight (1, 1, K padded, N padded) as bfloat16 before the device quantizes it: 0.02 normal; the edge regime zeroes a fifth of the columns."""
    generator = torch.Generator().manual_seed(seed)
    value = torch.randn(1, 1, tile_pad(k), tile_pad(n), generator=generator) * 0.02
    if regime == 'edge':
        value[..., torch.rand(value.shape[-1], generator=generator) < 0.2] = 0.0
    return value.to(torch.bfloat16)


def tile_pad(count):
    return int(math.ceil(count / 32.0)) * 32


# ---------------------------------------------------------------------------------------------------------------------------
# The device side. `ttnn` and `torch` are modules (fakes in the tests).
# ---------------------------------------------------------------------------------------------------------------------------

class Rig(object):
    def __init__(self, ttnn, torch, mesh, lever, grid, t1_gate_cores=None, clock=time.perf_counter):
        self.ttnn, self.torch, self.mesh, self.lever, self.clock = ttnn, torch, mesh, lever, clock
        self.grid_x, self.grid_y = grid
        self.t1_gate_cores = t1_gate_cores
        maker = getattr(ttnn, 'WormholeComputeKernelConfig', None) or getattr(ttnn, 'BlackholeComputeKernelConfig')
        # the model's decode compute config (mlp.py compute_kernel_config_decode): LoFi, fp32 destination, packer L1 accumulation
        self.ckc = maker(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True)

    # -- tensors ----------------------------------------------------------------------------------------------------------
    def dtype(self, key):
        return {'bfp4': self.ttnn.bfloat4_b, 'bfp8': self.ttnn.bfloat8_b, 'bf16': self.ttnn.bfloat16}[key]

    def upload(self, host, dtype, memory):
        ttnn = self.ttnn
        return ttnn.from_torch(host, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=memory,
                               mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))

    def weight(self, shape, seed=0, regime='random'):
        return self.upload(host_w(self.torch, shape.k, shape.n, seed, regime), self.dtype(shape.dtype), self.ttnn.DRAM_MEMORY_CONFIG)

    def activation(self, k, regime='random', seed=1, rows=None):
        return self.upload(host_x(self.torch, rows or self.lever.ROWS, k, regime, seed), self.ttnn.bfloat16, self.ttnn.L1_MEMORY_CONFIG)

    def read(self, tensor):
        torch = self.torch
        return self.ttnn.to_torch(self.ttnn.get_device_tensors(tensor)[0]).contiguous().view(torch.int16)

    def free(self, *tensors):
        for tensor in tensors:
            if tensor is not None:
                try:
                    self.ttnn.deallocate(tensor)
                except BaseException:  # noqa: BLE001
                    pass

    # -- the ops ------------------------------------------------------------------------------------------------------------
    def linear(self, x, weight, shape, config, out_memory=None, dtype=None):
        program = self.lever.program_config(self.ttnn, config, shape.silu)
        extra = {} if dtype is None else dict(dtype=dtype)
        return self.ttnn.linear(x, weight, compute_kernel_config=self.ckc, program_config=program,
                                memory_config=out_memory or self.ttnn.L1_MEMORY_CONFIG, **extra)

    def multiply(self, gate, up, memory):
        return self.ttnn.mul(gate, up, memory_config=memory)

    # -- timing -------------------------------------------------------------------------------------------------------------
    def capture(self, run, launches):
        """Capture `launches` back-to-back runs of one arm in one trace (a warm run beforehand compiled the programs); each output is freed inside
        the capture so the allocator reuses it, as a model's traced forward does."""
        ttnn, mesh = self.ttnn, self.mesh
        handle = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            try:
                for _ in range(launches):
                    self.free(run())
            finally:
                ttnn.end_trace_capture(mesh, handle, cq_id=0)
        except BaseException:
            try:
                ttnn.release_trace(mesh, handle)
            except BaseException:  # noqa: BLE001
                pass
            raise
        return handle

    def release(self, *handles):
        for handle in handles:
            try:
                self.ttnn.release_trace(self.mesh, handle)
            except BaseException:  # noqa: BLE001
                pass

    def arm(self, run, short=SHORT, long=LONG):
        """Warm (compile) and capture one arm's two traces: {run, short, long}."""
        self.free(run())
        self.ttnn.synchronize_device(self.mesh)
        handles = dict(run=run, short=self.capture(run, short), long=self.capture(run, long), n_short=short, n_long=long)
        self.ttnn.synchronize_device(self.mesh)
        return handles

    def drop(self, arm):
        self.release(arm['short'], arm['long'])

    def replay(self, handle, clock=None):
        clock = clock or self.clock
        started = clock()
        self.ttnn.execute_trace(self.mesh, handle, cq_id=0, blocking=False)
        self.ttnn.synchronize_device(self.mesh)
        return (clock() - started) * 1e6

    def race(self, arms, rounds=ROUNDS, clock=None):
        """Serpentine rounds over {name: captured arm}: per arm the per-launch microseconds (slope of its short and long traces) and its samples.
        Each round replays every arm's short and long trace once, in an order that reverses every other round."""
        names = list(arms)
        for name in names:                                  # one untimed replay each
            self.ttnn.execute_trace(self.mesh, arms[name]['long'], cq_id=0, blocking=False)
        self.ttnn.synchronize_device(self.mesh)
        samples = dict((name, dict(short=[], long=[])) for name in names)
        for index in range(rounds):
            for name in (names if index % 2 == 0 else list(reversed(names))):
                for kind in ('short', 'long'):
                    samples[name][kind].append(self.replay(arms[name][kind], clock))
        out = {}
        for name in names:
            arm = arms[name]
            out[name] = dict(us=slope_us(samples[name]['short'], samples[name]['long'], arm['n_short'], arm['n_long']),
                             long_per_launch=summarize([value / arm['n_long'] for value in samples[name]['long']]))
        return out


def differing(torch, left, right):
    if tuple(left.shape) != tuple(right.shape):
        return max(left.numel(), right.numel())
    return int((left != right).sum())


class Sweep(object):
    """One shape's sweep on a Rig: the weight, the activation, the served output and its captured trace."""

    def __init__(self, rig, shape, rounds=ROUNDS):
        self.rig, self.shape, self.rounds = rig, shape, rounds
        lever = rig.lever
        self.weight = rig.weight(shape)
        self.x = rig.activation(shape.k)
        self.edge_x = rig.activation(shape.k, 'edge', seed=2)
        self.edge_weight = rig.weight(shape, seed=3, regime='edge')
        self.served = served_row(lever, shape, rig.grid_x, rig.t1_gate_cores)
        self.reference = {}
        self.served_arm = None

    def output(self, config, regime='random'):
        rig = self.rig
        x, weight = (self.x, self.weight) if regime == 'random' else (self.edge_x, self.edge_weight)
        out = rig.linear(x, weight, self.shape, config)
        try:
            return rig.read(out)
        finally:
            rig.free(out)

    def prepare(self):
        """The served output on both regimes (twice, the second as the control that the served arm reproduces itself) and its captured arm."""
        rig = self.rig
        for regime in REGIMES:
            first = self.output(self.served['config'], regime)
            again = self.output(self.served['config'], regime)
            self.reference[regime] = first
            if differing(rig.torch, first, again):
                raise AssertionError('the served config did not reproduce itself on %s data (%d elements)' % (
                    regime, differing(rig.torch, first, again)))
        self.served_arm = rig.arm(lambda: rig.linear(self.x, self.weight, self.shape, self.served['config']))

    def release(self):
        rig = self.rig
        if self.served_arm is not None:
            rig.drop(self.served_arm)
        rig.free(self.weight, self.x, self.edge_x, self.edge_weight)

    def measure(self, candidate, regimes=('random',)):
        """Compare (every regime in `regimes`) and time one candidate against the served arm; returns the row."""
        rig, shape, lever = self.rig, self.shape, self.rig.lever
        try:
            exact, wrong = True, 0
            for regime in regimes:
                wrong += differing(rig.torch, self.output(candidate['config'], regime), self.reference[regime])
            exact = wrong == 0
            arm = rig.arm(lambda: rig.linear(self.x, self.weight, shape, candidate['config']))
            try:
                raced = rig.race({'served': self.served_arm, 'candidate': arm}, self.rounds)
            finally:
                rig.drop(arm)
        except BaseException as error:  # noqa: BLE001 - a config the program refuses is a row, not the end of the sweep
            return row_of(lever, shape, candidate, None, None, error='%s: %s' % (type(error).__name__, str(error)[:300]))
        timing = dict(us=raced['candidate']['us'], served_us=raced['served']['us'], delta_us=raced['candidate']['us'] - raced['served']['us'])
        return row_of(lever, shape, candidate, timing, exact, differing=wrong)


def sweep_shapes(rig, keys, budget, report, out_line, rounds=ROUNDS):
    """Stage one for every shape, then stage two as the budget allows. Fills report['shapes'][key] = {rows, summary} after every row (a hang keeps
    what was measured)."""
    lever = rig.lever
    table = shape_table(lever)
    sweeps, seen = {}, {}
    for key in keys:
        shape = table[key]
        sweep = Sweep(rig, shape, rounds)
        sweep.prepare()
        sweeps[key] = sweep
        report['shapes'][key] = dict(rows=[], summary=None)
        seen[key] = set()
        served = sweep.served
        row = sweep.measure(served, REGIMES)
        report['shapes'][key]['rows'].append(row)
        seen[key].add((tuple(served['config'].grid), served['config'].per_core_N))
        out_line(row)
    try:
        for key in keys:
            shape, sweep = table[key], sweeps[key]
            for candidate in stage_one(lever, shape, rig.grid_x, rig.grid_y):
                identity = (tuple(candidate['config'].grid), candidate['config'].per_core_N)
                if identity in seen[key]:
                    continue
                if budget.expired():
                    report['truncated'] = 'stage 1 stopped at %s %s' % (key, candidate['name'])
                    raise StopIteration
                seen[key].add(identity)
                row = sweep.measure(candidate)
                report['shapes'][key]['rows'].append(row)
                out_line(row)
        for key in keys:
            shape, sweep = table[key], sweeps[key]
            rows = report['shapes'][key]['rows']
            ranked = sorted([row for row in rows if 'us' in row and row.get('exact') is True and row['name'] != 'served'], key=lambda row: row['us'])
            best = []
            for row in ranked:
                if row['pcn'] not in best:
                    best.append(row['pcn'])
                if len(best) == STAGE2_TOP:
                    break
            for candidate in stage_two(lever, shape, rig.grid_x, rig.grid_y, best, seen[key]):
                if budget.expired():
                    report['truncated'] = 'stage 2 stopped at %s %s' % (key, candidate['name'])
                    raise StopIteration
                row = sweep.measure(candidate)
                rows.append(row)
                out_line(row)
    except StopIteration:
        pass
    # the edge regime for the finalists (the best three exact rows of each shape, served excluded): the byte compare must hold on both regimes
    for key in keys:
        shape, sweep = table[key], sweeps[key]
        rows = report['shapes'][key]['rows']
        finalists = sorted([row for row in rows if 'us' in row and row.get('exact') is True and row['name'] != 'served'],
                           key=lambda row: row['us'])[:3]
        for row in finalists:
            candidate = dict(name=row['name'], pcn=row['pcn'], width=row['width'], stage=row['stage'],
                             config=lever.named_config(shape, row['pcn'], row['width']))
            wrong = differing(rig.torch, sweep.output(candidate['config'], 'edge'), sweep.reference['edge'])
            row['edge_differing'] = wrong
            if wrong:
                row['exact'] = False
                row['differing'] = (row.get('differing') or 0) + wrong
        report['shapes'][key]['summary'] = shape_summary(lever, shape, rows)
    for sweep in sweeps.values():
        sweep.release()
    return report


# ---------------------------------------------------------------------------------------------------------------------------
# The composition: the served chain, the chain with the L1 multiply, the chain at the best configs.
# ---------------------------------------------------------------------------------------------------------------------------

def chain_arm(rig, sets, configs, multiply_memory):
    """A runner of one MLP chain (gate, up, multiply, down) over a rotating weight set; returns the down output (L1)."""
    state = dict(i=0)
    table = rig.lever.shapes(4)

    def run():
        weights = sets[state['i'] % len(sets)]
        state['i'] += 1
        gate = rig.linear(weights['x'], weights['w1'], table['mlp_w1'], configs['mlp_w1'])
        up = rig.linear(weights['x'], weights['w3'], table['mlp_w3'], configs['mlp_w3'])
        product = rig.multiply(gate, up, multiply_memory)
        rig.free(gate, up)
        out = rig.linear(product, weights['w2'], table['mlp_w2'], configs['mlp_w2'])
        rig.free(product)
        return out

    def reset():
        state['i'] = 0

    run.reset = reset
    return run


def compose(rig, summaries, rounds=ROUNDS, clock=time.perf_counter):
    """Time the three chains paired and compare their final outputs with the served chain's. Returns the section."""
    lever, ttnn, torch = rig.lever, rig.ttnn, rig.torch
    table = lever.shapes(4)
    sets = []
    for index in range(COMPOSE_SETS):
        sets.append(dict(x=rig.activation(table['mlp_w1'].k, seed=10 + index), w1=rig.weight(table['mlp_w1'], 20 + index),
                         w3=rig.weight(table['mlp_w3'], 30 + index), w2=rig.weight(table['mlp_w2'], 40 + index)))
    served = dict((key, served_row(lever, table[key], rig.grid_x, rig.t1_gate_cores)['config']) for key in ('mlp_w1', 'mlp_w3', 'mlp_w2'))
    tuned = dict(served)
    names = {}
    for key in served:
        entry = summaries.get(key)
        row = entry and entry.get('best_default_width')
        if row:
            tuned[key] = lever.named_config(table[key], row['pcn'], rig.grid_x)
            names[key] = row['name']
    arms_spec = dict(served=(served, ttnn.DRAM_MEMORY_CONFIG), l1_multiply=(served, ttnn.L1_MEMORY_CONFIG),
                     tuned_l1=(tuned, ttnn.L1_MEMORY_CONFIG))
    runners = dict((name, chain_arm(rig, sets, spec[0], spec[1])) for name, spec in arms_spec.items())
    section = dict(sets=COMPOSE_SETS, tuned_names=names, tuned_cfg=combined_name(summaries))
    captured, outputs = {}, {}
    try:
        for name, run in runners.items():
            run.reset()
            out = run()
            outputs[name] = rig.read(out)
            rig.free(out)
            run.reset()
            captured[name] = rig.arm(run, COMPOSE_SHORT, COMPOSE_LONG)
        raced = rig.race(captured, rounds, clock)
        section['chain_us'] = dict((name, round(value['us'], 3)) for name, value in raced.items())
        section['exact'] = dict((name, differing(torch, outputs[name], outputs['served']) == 0) for name in outputs)
        base = raced['served']['us']
        section['gain_ms_pass'] = dict((name, round(LAYERS * (base - value['us']) / 1000.0, 3)) for name, value in raced.items() if name != 'served')
    finally:
        for arm in captured.values():
            rig.drop(arm)
        for entry in sets:
            rig.free(*entry.values())
    return section


def multiply_micro(rig, rounds=ROUNDS, clock=time.perf_counter):
    """The multiply alone at (64, 4352), output to DRAM and to L1: microseconds per launch, to put the epilogue's share on the table."""
    lever, ttnn = rig.lever, rig.ttnn
    n = lever.shapes(4)['mlp_w1'].n
    a = rig.upload(host_x(rig.torch, lever.ROWS, n, 'random', 5), ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG)
    b = rig.upload(host_x(rig.torch, lever.ROWS, n, 'random', 6), ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG)
    arms = {}
    try:
        for name, memory in (('dram', ttnn.DRAM_MEMORY_CONFIG), ('l1', ttnn.L1_MEMORY_CONFIG)):
            arms[name] = rig.arm(lambda memory=memory: rig.multiply(a, b, memory), SHORT, LONG)
        raced = rig.race(arms, rounds, clock)
    finally:
        for arm in arms.values():
            rig.drop(arm)
        rig.free(a, b)
    return dict((name, round(value['us'], 3)) for name, value in raced.items())


# ---------------------------------------------------------------------------------------------------------------------------
# The fused op (--arms fused): its product against the served multiply's, timed per pairs-per-worker.
# ---------------------------------------------------------------------------------------------------------------------------

def fused_section(rig, fused_module, rounds=ROUNDS, clock=time.perf_counter):
    lever, ttnn, torch = rig.lever, rig.ttnn, rig.torch
    table = lever.shapes(4)
    gate_shape, up_shape = table['mlp_w1'], table['mlp_w3']
    x = rig.activation(gate_shape.k, seed=1)
    w1, w3 = rig.weight(gate_shape, 20), rig.weight(up_shape, 30)
    served_cfg = dict((key, served_row(lever, table[key], rig.grid_x, rig.t1_gate_cores)['config']) for key in ('mlp_w1', 'mlp_w3'))

    def served_product():
        gate = rig.linear(x, w1, gate_shape, served_cfg['mlp_w1'])
        up = rig.linear(x, w3, up_shape, served_cfg['mlp_w3'])
        product = rig.multiply(gate, up, ttnn.DRAM_MEMORY_CONFIG)
        rig.free(gate, up)
        return product

    reference_tensor = served_product()
    reference = rig.read(reference_tensor)
    rig.free(reference_tensor)
    section = dict(rows=[], served_pair_us=None, served_math_approx_mode=bool(getattr(rig.ckc, 'math_approx_mode', True)))
    arms = dict(served=rig.arm(served_product, SHORT, LONG))
    built = {}
    matching = section['served_math_approx_mode']
    # every pairs-per-worker with the served approximation mode; the opposite mode once, at the default pairs, as the diagnostic if the product differs
    variants = [(pairs, matching) for pairs in lever.PAIRS_PER_WORKER] + [(lever.DEFAULT_PAIRS, not matching)]
    try:
        for pairs, approx in variants:
            row = dict(pairs=pairs, math_approx_mode=approx, diagnostic=approx != matching)
            key = 'p%d%s' % (pairs, 'a' if approx else 'x')
            try:
                op = fused_module.FusedGateUp(ttnn, rig.mesh, w1, w3, pairs_per_worker=pairs, grid=(rig.grid_x, rig.grid_y),
                                              math_approx_mode=approx)
                row['workers'] = op.plan['workers']
                product = op(x)
                row['differing'] = differing(torch, rig.read(product), reference)
                row['exact'] = row['differing'] == 0
                rig.free(product)
                arms[key] = rig.arm(lambda op=op: op(x), SHORT, LONG)
                built[key] = op
            except BaseException as error:  # noqa: BLE001
                row['error'] = '%s: %s' % (type(error).__name__, str(error)[:300])
            row['key'] = key
            section['rows'].append(row)
        raced = rig.race(arms, rounds, clock)
        section['served_pair_us'] = round(raced['served']['us'], 3)
        for row in section['rows']:
            if row['key'] in raced:
                row['us'] = round(raced[row['key']]['us'], 3)
                row['delta_us'] = round(raced[row['key']]['us'] - raced['served']['us'], 3)
                row['gain_ms_pass'] = round(LAYERS * (raced['served']['us'] - raced[row['key']]['us']) / 1000.0, 3)
    finally:
        for arm in arms.values():
            rig.drop(arm)
        rig.free(x, w1, w3)
    return section


# ---------------------------------------------------------------------------------------------------------------------------
# What the sweep implies: the SiLU epilogue (pure arithmetic on the rows the sweep already timed).
# ---------------------------------------------------------------------------------------------------------------------------

def epilogue_fit(gate_rows, up_rows, row_tiles=2):
    """The cost of the gate's fused SiLU, from the sweep: at every per_core_N both shapes timed (default width, exact), gate minus up is the epilogue (the bytes, the dtype
    and the reader are the same). Least squares through the origin of that gap on per_core_N: microseconds per output column per core, and per output tile (a column is
    `row_tiles` tiles at 64 rows). Returns None with fewer than three common points."""
    up = dict((row['pcn'], row) for row in up_rows if 'us' in row and row.get('exact') is True and row.get('stage') == 1)
    points = [(row['pcn'], row['us'] - up[row['pcn']]['us']) for row in gate_rows if 'us' in row and row.get('exact') is True and row.get('stage') == 1 and row['pcn'] in up]
    if len(points) < 3:
        return None
    per_column = sum(pcn * gap for pcn, gap in points) / float(sum(pcn * pcn for pcn, gap in points))
    residual = max(abs(gap - per_column * pcn) for pcn, gap in points)
    return dict(points=len(points), us_per_column=round(per_column, 3), us_per_tile=round(per_column / row_tiles, 3), max_residual_us=round(residual, 2),
                gaps=dict((pcn, round(gap, 2)) for pcn, gap in sorted(points)))


# ---------------------------------------------------------------------------------------------------------------------------
# The read probe (--arms probe): request granularity against bandwidth, no compute.
# ---------------------------------------------------------------------------------------------------------------------------

PROBE_SHAPES = (('gate_bf4', 'bfp4', 5120, 4352), ('gate_bf8', 'bfp8', 5120, 4352), ('down_bf8', 'bfp8', 4352, 5120))
WIN_RATIO, MARGINAL_RATIO = 1.25, 1.10


def probe_verdict(rows):
    """READ-GRANULARITY when the best exact bank-contiguous point of the bfloat4_b gate shape reads at least 1.25 times the best stock-pattern point of the same shape;
    MARGINAL from 1.10; NO-EFFECT below; NO-RESULT without both."""
    def best(mode):
        found = [row for row in rows if row['shape'] == 'gate_bf4' and row['mode'] == mode and 'gbps' in row and row.get('exact') is True]
        return max(found, key=lambda row: row['gbps']) if found else None
    stock, bank = best('stock'), best('bank')
    if not stock or not bank:
        return dict(verdict='NO-RESULT', stock=stock and stock['gbps'], bank=bank and bank['gbps'])
    ratio = bank['gbps'] / stock['gbps']
    word = 'READ-GRANULARITY' if ratio >= WIN_RATIO else 'MARGINAL' if ratio >= MARGINAL_RATIO else 'NO-EFFECT'
    return dict(verdict=word, ratio=round(ratio, 3), stock_gbps=stock['gbps'], stock_point=[stock['run'], stock['chunk']], bank_gbps=bank['gbps'],
                bank_point=[bank['run'], bank['chunk']], bank_pct_peak=bank['pct_peak'])


def probe_section(rig, probe_module, rounds=ROUNDS, clock=time.perf_counter, bank_points=None, stock_points=None):
    """Per probe shape: a correctness launch per point (every tile read is copied to the same page of a zeroed twin tensor; the host compares the two), then a read-only timed
    launch per point raced against the stock pattern at three tiles per worker in the same round. Rows: shape, mode, run, chunk, workers, request bytes, us, GB/s, exact."""
    lever, ttnn, torch = rig.lever, rig.ttnn, rig.torch
    banks = int(getattr(getattr(rig.mesh, 'dram_grid_size', None) and rig.mesh.dram_grid_size(), 'x', 8))
    bank_points = tuple(bank_points if bank_points is not None else probe_module.BANK_POINTS)
    stock_points = tuple(stock_points if stock_points is not None else probe_module.STOCK_POINTS)
    points = [('stock', run, chunk) for run, chunk in stock_points] + [('bank', run, chunk) for run, chunk in bank_points]
    section = dict(banks=banks, rows=[])
    for name, dtype, k, n in PROBE_SHAPES:
        tile_rows, tile_columns = lever.tiles(k), lever.tiles(n)
        shape = lever.Shape(name, k, n, dtype, False, 0)
        total_bytes = tile_rows * tile_columns * lever.TILE_BYTES[dtype]
        source = rig.weight(shape, seed=7)
        zeros = rig.upload(torch.zeros(1, 1, tile_pad(k), tile_pad(n), dtype=torch.bfloat16), rig.dtype(dtype), ttnn.DRAM_MEMORY_CONFIG)
        scratch = rig.upload(torch.zeros(1, 1, tile_pad(k), tile_pad(n), dtype=torch.bfloat16), rig.dtype(dtype), ttnn.DRAM_MEMORY_CONFIG)
        reference = rig.read(source)
        probes, arms, rows = {}, {}, {}
        try:
            for mode, run, chunk in points:
                key = '%s-r%d-c%d' % (mode, run, chunk)
                row = dict(shape=name, dtype=dtype, mode=mode, run=run, chunk=chunk)
                rows[key] = row
                try:
                    checker = probe_module.ReadProbe(ttnn, rig.mesh, source, zeros, dtype, tile_rows, tile_columns, banks, mode, run, chunk,
                                                     (rig.grid_x, rig.grid_y), write_back=True)
                    checker()
                    ttnn.synchronize_device(rig.mesh)
                    row['differing'] = differing(torch, rig.read(zeros), reference)
                    row['exact'] = row['differing'] == 0
                    timer = probe_module.ReadProbe(ttnn, rig.mesh, source, scratch, dtype, tile_rows, tile_columns, banks, mode, run, chunk,
                                                   (rig.grid_x, rig.grid_y), write_back=False)
                    row.update(workers=len(timer.plan), request_bytes_max=timer.largest, request_bytes_mean=round(timer.mean, 1),
                               requests_per_row=timer.requests_per_row)
                    probes[key] = timer
                except BaseException as error:  # noqa: BLE001
                    row['error'] = '%s: %s' % (type(error).__name__, str(error)[:300])
            base_key = 'stock-r3-c1'
            if base_key in probes:
                arms[base_key] = rig.arm(lambda p=probes[base_key]: p(), SHORT, LONG)
            for key, timer in probes.items():
                try:
                    if key == base_key:
                        continue
                    arms[key] = rig.arm(lambda p=timer: p(), SHORT, LONG)
                    raced = rig.race({base_key: arms[base_key], key: arms[key]} if base_key in arms else {key: arms[key]}, rounds, clock)
                    us = raced[key]['us']
                    rows[key].update(us=round(us, 3), gbps=round(total_bytes / us / 1e3, 1), pct_peak=round(100.0 * total_bytes / us / 1e3 / lever.PEAK_GBPS, 1))
                    if base_key in raced:
                        rows[key]['delta_vs_stock_r3_us'] = round(us - raced[base_key]['us'], 3)
                    rig.drop(arms.pop(key))
                except BaseException as error:  # noqa: BLE001
                    rows[key]['error'] = '%s: %s' % (type(error).__name__, str(error)[:300])
            if base_key in arms:
                base_row = rows[base_key]
                raced = rig.race({base_key: arms[base_key]}, rounds, clock)
                base_row.update(us=round(raced[base_key]['us'], 3), gbps=round(total_bytes / raced[base_key]['us'] / 1e3, 1),
                                pct_peak=round(100.0 * total_bytes / raced[base_key]['us'] / 1e3 / lever.PEAK_GBPS, 1))
        finally:
            for arm in arms.values():
                rig.drop(arm)
            rig.free(source, zeros, scratch)
        section['rows'].extend(rows[key] for key in ['%s-r%d-c%d' % point for point in points] if key in rows)
    section['decision'] = probe_verdict(section['rows'])
    return section


# ---------------------------------------------------------------------------------------------------------------------------
# Splitting the SiLU out of the gate matmul (--arms split): is it exact, and does it pay?
# ---------------------------------------------------------------------------------------------------------------------------

def split_section(rig, rounds=ROUNDS, clock=time.perf_counter):
    """The served gate (SiLU fused in the matmul), up and multiply, against variants that take the SiLU out of the matmul epilogue (about 1.15 us per output tile, serialized
    after the K loop): A the gate written as float32, silu, typecast to bfloat16, multiply; B the gate written as float32 and the multiply taking the SiLU as its first
    input's activation (one launch fewer, no bfloat16 rounding between the SiLU and the product: expected NOT exact); C the same with the gate written as bfloat16 (rounded
    before the SiLU: the control that must differ). Each variant's product is compared bit for bit with the served one and raced against it."""
    lever, ttnn, torch = rig.lever, rig.ttnn, rig.torch
    table = lever.shapes(4)
    gate_shape, up_shape = table['mlp_w1'], table['mlp_w3']
    x = rig.activation(gate_shape.k, seed=1)
    w1, w3 = rig.weight(gate_shape, 20), rig.weight(up_shape, 30)
    served_cfg = dict((key, served_row(lever, table[key], rig.grid_x, rig.t1_gate_cores)['config']) for key in ('mlp_w1', 'mlp_w3'))
    raw_gate = gate_shape._replace(silu=False)
    l1 = ttnn.L1_MEMORY_CONFIG

    def served():
        gate = rig.linear(x, w1, gate_shape, served_cfg['mlp_w1'])
        up = rig.linear(x, w3, up_shape, served_cfg['mlp_w3'])
        product = rig.multiply(gate, up, l1)
        rig.free(gate, up)
        return product

    def variant_a():
        raw = rig.linear(x, w1, raw_gate, served_cfg['mlp_w1'], dtype=ttnn.float32)
        activated = ttnn.silu(raw, memory_config=l1)
        rig.free(raw)
        rounded = ttnn.typecast(activated, ttnn.bfloat16, memory_config=l1)
        rig.free(activated)
        up = rig.linear(x, w3, up_shape, served_cfg['mlp_w3'])
        product = rig.multiply(rounded, up, l1)
        rig.free(rounded, up)
        return product

    def variant(dtype):
        def run():
            raw = rig.linear(x, w1, raw_gate, served_cfg['mlp_w1'], dtype=dtype)
            up = rig.linear(x, w3, up_shape, served_cfg['mlp_w3'])
            product = ttnn.multiply(raw, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=ttnn.bfloat16, memory_config=l1)
            rig.free(raw, up)
            return product
        return run

    runs = dict(served=served, a_silu_typecast=variant_a, b_fused_in_multiply=variant(ttnn.float32), c_bf16_then_fused=variant(ttnn.bfloat16))
    section = dict(rows=[])
    reference_tensor = served()
    reference = rig.read(reference_tensor)
    rig.free(reference_tensor)
    arms = {}
    try:
        for name, run in runs.items():
            row = dict(variant=name)
            try:
                product = run()
                row['differing'] = differing(torch, rig.read(product), reference)
                row['exact'] = row['differing'] == 0
                rig.free(product)
                arms[name] = rig.arm(run, SHORT, LONG)
            except BaseException as error:  # noqa: BLE001
                row['error'] = '%s: %s' % (type(error).__name__, str(error)[:300])
            section['rows'].append(row)
        raced = rig.race(arms, rounds, clock)
        for row in section['rows']:
            if row['variant'] in raced:
                row['us'] = round(raced[row['variant']]['us'], 3)
                row['delta_us'] = round(raced[row['variant']]['us'] - raced['served']['us'], 3)
                row['gain_ms_pass'] = round(LAYERS * (raced['served']['us'] - raced[row['variant']]['us']) / 1000.0, 3)
    finally:
        for arm in arms.values():
            rig.drop(arm)
        rig.free(x, w1, w3)
    return section


# ---------------------------------------------------------------------------------------------------------------------------
# main.
# ---------------------------------------------------------------------------------------------------------------------------

class Budget(object):
    def __init__(self, seconds, clock=time.time):
        self.clock, self.deadline = clock, clock() + seconds

    def expired(self):
        return self.clock() >= self.deadline


def print_row(row):
    if 'error' in row:
        print('MLP_GATEUP cand shape=%s name=%s grid=%dx%d pcn=%d ERROR %s' % (row['shape'], row['name'], row['grid'][0], row['grid'][1],
                                                                              row['pcn'], row['error'][:120]), flush=True)
        return
    print('MLP_GATEUP cand shape=%s name=%s stage=%d grid=%dx%d cores=%d pcn=%d us=%.2f gbps=%.1f pct_peak=%.1f exact=%d' % (
        row['shape'], row['name'], row['stage'], row['grid'][0], row['grid'][1], row['active_cores'], row['pcn'], row['us'], row['gbps'],
        row['pct_peak'], int(row['exact'])), flush=True)


def print_summary(entry):
    served, best = entry.get('served'), entry.get('best_exact')
    if not served or not best:
        print('MLP_GATEUP shape %s no result (served=%s best_exact=%s)' % (entry['shape'], bool(served), bool(best)), flush=True)
        return
    print('MLP_GATEUP shape %s served %.2f us %.1f GB/s (%d cores) best_exact %s %.2f us %.1f GB/s (%d cores) gain %.2f us = %.3f ms/pass '
          'inexact=%d' % (entry['shape'], served['us'], served['gbps'], served['active_cores'], best['name'], best['us'], best['gbps'],
                          best['active_cores'], entry['gain_us'], entry['gain_ms_pass'], len(entry['inexact'])), flush=True)


def main(argv=None, torch=None, ttnn=None, lever=None, fused_module=None, probe_module=None, clock=time.perf_counter, budget_clock=time.time):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', required=True)
    parser.add_argument('--shapes', default=DEFAULT_SHAPES, help='comma-separated: mlp_w1 mlp_w3 mlp_w2 (gate, up, down), gdn_in attn_in attn_wo gdn_out (R3), gate_bf8 up_bf8 (diagnostic: the same shapes with bfloat8_b weights)')
    parser.add_argument('--arms', default='sweep,compose', help='comma-separated: sweep, compose, fused, probe (read granularity), split (SiLU out of the matmul)')
    parser.add_argument('--rounds', type=int, default=ROUNDS)
    parser.add_argument('--budget-s', type=int, default=DEFAULT_BUDGET_S, help='stop adding candidates after this many seconds')
    parser.add_argument('--t1', choices=('on', 'off'), default='on', help='the gate at 88 requested cores (QWEN_FAST_VERIFY_T1=1, as served) or 44')
    options = parser.parse_args(argv)
    keys = [key for key in options.shapes.split(',') if key]
    arms = [arm for arm in options.arms.split(',') if arm]
    if any(arm not in ARMS for arm in arms) or not keys:
        print('refusing: arms are %s, shapes comma-separated keys' % ', '.join(ARMS), file=sys.stderr)
        return 2
    timer = threading.Timer(WATCHDOG_S, lambda: (print('MLP_GATEUP watchdog', flush=True), os._exit(3)))
    timer.daemon = True
    timer.start()
    if torch is None:
        import torch
        import ttnn
        import tp4_mlp_gateup as lever
    report = dict(kind=KIND, shapes={}, environment=dict(QWEN_FAST_TP=os.environ.get('QWEN_FAST_TP'),
                  TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=os.environ.get('TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE')),
                  arms=arms, t1=options.t1, rounds=options.rounds, budget_s=options.budget_s)
    mesh, status = None, 4
    try:
        if os.environ.get('QWEN_FAST_TP') != '4':
            raise RuntimeError('QWEN_FAST_TP=4 required (the shapes are the four-card shard\'s)')
        table = shape_table(lever)
        unknown = [key for key in keys if key not in table]
        if unknown:
            raise RuntimeError('unknown shapes %s' % ', '.join(unknown))
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=TRACE_REGION)
        grid = mesh.compute_with_storage_grid_size()
        report['grid'] = [int(grid.x), int(grid.y)]
        report['workers'] = int(grid.x) * int(grid.y)
        print('MLP_GATEUP grid %dx%d workers=%d' % (grid.x, grid.y, grid.x * grid.y), flush=True)
        rig = Rig(ttnn, torch, mesh, lever, (int(grid.x), int(grid.y)), t1_gate_cores=88 if options.t1 == 'on' else 44, clock=clock)
        budget = Budget(options.budget_s, budget_clock)
        summaries = {}
        if 'sweep' in arms:
            sweep_shapes(rig, keys, budget, report, print_row, options.rounds)
            summaries = dict((key, report['shapes'][key]['summary']) for key in keys)
            for key in keys:
                print_summary(summaries[key])
            report['summary'] = dict((key, dict((name, value) for name, value in entry.items() if name != 'rows'))
                                     for key, entry in summaries.items())
            report['inexact'] = dict((key, entry['inexact']) for key, entry in summaries.items() if entry['inexact'])
            for key, names in report['inexact'].items():
                print('MLP_GATEUP ALERT %s: configs that keep the K loop but differ from the served output: %s' % (key, ', '.join(names)), flush=True)
            if 'mlp_w1' in report['shapes'] and 'mlp_w3' in report['shapes']:
                report['epilogue'] = epilogue_fit(report['shapes']['mlp_w1']['rows'], report['shapes']['mlp_w3']['rows'])
                if report['epilogue']:
                    print('MLP_GATEUP epilogue %s' % json.dumps(report['epilogue'], sort_keys=True), flush=True)
        if 'compose' in arms:
            report['multiply_us'] = multiply_micro(rig, options.rounds, clock)
            print('MLP_GATEUP multiply dram=%.2f us l1=%.2f us' % (report['multiply_us']['dram'], report['multiply_us']['l1']), flush=True)
            report['compose'] = compose(rig, summaries, options.rounds, clock)
            print('MLP_GATEUP compose chain_us=%s exact=%s gain_ms_pass=%s tuned=%s' % (
                json.dumps(report['compose']['chain_us'], sort_keys=True), json.dumps(report['compose']['exact'], sort_keys=True),
                json.dumps(report['compose']['gain_ms_pass'], sort_keys=True), report['compose']['tuned_cfg']), flush=True)
        if 'fused' in arms:
            if fused_module is None:
                import tp4_mlp_fused as fused_module
            report['fused'] = fused_section(rig, fused_module, options.rounds, clock)
            for row in report['fused']['rows']:
                print('MLP_GATEUP fused %s' % json.dumps(row, sort_keys=True), flush=True)
        if 'probe' in arms:
            if probe_module is None:
                import readprobe as probe_module
            report['probe'] = probe_section(rig, probe_module, options.rounds, clock)
            for row in report['probe']['rows']:
                print('MLP_GATEUP probe %s' % json.dumps(row, sort_keys=True), flush=True)
            print('MLP_GATEUP probe verdict %s' % json.dumps(report['probe']['decision'], sort_keys=True), flush=True)
        if 'split' in arms:
            report['split'] = split_section(rig, options.rounds, clock)
            for row in report['split']['rows']:
                print('MLP_GATEUP split %s' % json.dumps(row, sort_keys=True), flush=True)
        if summaries and all(key in summaries for key in ('mlp_w1', 'mlp_w3')):
            report['decision'] = decide(summaries, lever, (report.get('multiply_us') or {}).get('dram'))
            print('MLP_GATEUP verdict %s %s' % (report['decision']['verdict'], json.dumps(report['decision'], sort_keys=True)), flush=True)
            if report['decision'].get('env'):
                print('MLP_GATEUP env QWEN_FAST_MLP_CFG=%s' % report['decision']['env']['QWEN_FAST_MLP_CFG'], flush=True)
        failed = [key for key in keys if summaries and not (summaries[key].get('served') and summaries[key].get('best_exact'))]
        report['verdict'] = 'FAIL' if failed else 'PASS'
        status = 1 if failed else 0
    except BaseException as error:  # noqa: BLE001
        report['verdict'] = 'NOT-RUN'
        report['error'] = '%s: %s' % (error.__class__.__name__, str(error)[:500])
        print('MLP_GATEUP verdict=NOT-RUN error=%s' % report['error'], flush=True)
        status = 4
    finally:
        if mesh is not None:
            try:
                ttnn.close_mesh_device(mesh)
            except BaseException:  # noqa: BLE001
                pass
    with open(options.out, 'w') as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)
    print(json.dumps(dict((key, value) for key, value in report.items() if key != 'shapes'), sort_keys=True, default=str), flush=True)
    timer.cancel()
    return status


if __name__ == '__main__':
    sys.exit(main())
