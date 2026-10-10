"""CCL sweep probe: the per-call options of the packed verify's reduce-scatter and all-gather, bit-compared and timed in-trace at the stack's own shapes.

WHY (docs: scripts/ci/ccl_options_tp.py, plan F-C1). The verify issues 128 reduce-scatters and 129 all-gathers a pass at 18.2 and 18.4 us median (16.5 and
17.4 us intrinsic, M676). The model calls both with fixed values of five keywords. This probe runs the SAME calls at the SAME shapes with every value
ccl_options_tp offers (the named sets the stack's flag accepts) and answers three questions per value: is it bit-exact against the sequential engine's own
call, how long does one call take inside a captured trace, and does it win, paired, against the served values. It also measures, as PROBE-ONLY arms that no
stack flag can name, what dropping the start barrier costs (barrier_semaphore=None, and persistent buffers, which drop it too).

THE SHAPES (the stack's, from tile_collective_tp and DistributedNorm):
  rs   the unit-major reduce-scatter: a (1, 1, 64, 5120) bfloat16 partial per chip viewed (1, 2, 32, 5120), Ring, dim 3, DRAM output, DRAM intermediate, the
       model's keywords (num_links 2, chunks_per_sync 10, num_workers_per_link 2, num_buffers_per_channel 2). The input is in DRAM or L1 (two scenarios: the
       stack's partials come from a matmul whose output memory depends on the layer).
  ag   the norm's gather: a (1, 1, 64, 1280) bfloat16 tile per chip, dim 3, Ring, 2 links, into the norm's width-sharded L1 config on 32 cores (64 x 160
       shards), or interleaved DRAM for the third scenario.

EXACTNESS. rs: random heavy-tailed bfloat16 partials (integers are exact under any association and cannot see a changed order), the unit-major result of
every config against the model's own tt_all_reduce on each 32-row tile alone (the sequential engine's call; X1's anchor), joined over the four chips, as
int16 bit patterns, for SEEDS seeds; and the first and last call of the captured trace against the eager result. ag: the gathered tensor on every chip
against the host concatenation of the four inputs, with -0 and denormals in the inputs. A config whose bits differ is FAIL-BYTES and is never timed.
The base config's output is fingerprinted (sha256 of the bits) so two runs under different fabric configs or payloads can be compared offline
(compare_reports).

TIMING. Per config two traces of the same call on the same input are captured (N_HIGH and N_LOW calls; the model's cycled semaphores are taken per call,
so each trace has the model's alternation), replayed REPLAYS times per round; per-call time = (median(T_HIGH) - median(T_LOW)) / (N_HIGH - N_LOW), which
removes the replay's fixed host and launch cost. A candidate and the served config alternate A B / B A over ROUNDS rounds and the verdict is read PAIRED:
the per-round differences, their median, how many rounds the candidate won, against an A/A control (the served config twice) that measures the noise. A
config PROMOTES when it is exact on every seed and the median paired gain is at least max(MIN_GAIN_US, 2 x the A/A median |difference|) in at least
ROUNDS - 1 rounds. The stack's own gate is the ABAB at eight users: this probe only decides what is worth that.

SAFETY. A collective that hangs cannot be recovered in-process. One mesh is opened (one fabric config, one payload, one open: a second open wedges the
ethernet cores); the order is the exact-class sweeps first (one-factor-at-a-time around the served values, then the best values combined), the probe-only
arms last; the report is written after every config; a watchdog writes it again and exits 3. Put reset before any rerun.

Stdlib at import (plus ccl_options_tp from scripts/ci); ttnn and torch are handed to run().
"""

from collections import namedtuple
import hashlib
import json
import os
import statistics
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'scripts', 'ci'))
import ccl_options_tp  # noqa: E402

KIND = 'ccl-sweep-quad'
CHIPS = 4
TILE = 32
WIDTH = 5120
SHARD = WIDTH // CHIPS
ROWS = 64
RS_VIEW = (1, ROWS // TILE, TILE, WIDTH)
WS_CORES = (4, 8)                  # a 32-core grid: the norm's activation shard is (rows, WIDTH / 32)
SEEDS = 2
ROUNDS = 5
REPLAYS = 5
N_HIGH, N_LOW = 16, 4
MIN_GAIN_US = 0.3
MAX_CONSECUTIVE_ERRORS = 3       # a wedged fabric fails every config the same way: stop instead of reporting thirty identical errors
CONFIG_DEADLINE_S = 420          # one config: compile of a new variant, two captures, 2 x ROUNDS x 2 x REPLAYS replays
JOB_DEADLINE_S = 2400            # inside the fabric step's 45 minutes

SCENARIOS = (('rs/dram', 'rs', 'dram', None), ('rs/l1', 'rs', 'l1', None), ('ag/l1-ws', 'ag', 'l1', 'ws'), ('ag/dram-ws', 'ag', 'dram', 'ws'),
             ('ag/l1-dram', 'ag', 'l1', 'dram'))
VALUES = {'l': (1,), 'w': (1, 3, 4), 'c': (1, 2, 5, 20, 50), 'b': (1, 3, 4)}
QUICK_VALUES = {'l': (1,), 'w': (1,), 'c': (1, 5), 'b': (1,)}
AG_FLAGS = ('ag-linear', 'ag-bcast')

Config = namedtuple('Config', 'name op overrides probe_only')
Scenario = namedtuple('Scenario', 'name op input output')

SPECIAL_BITS = (0x8000, 0x0000, 0x0001, 0x8001, 0x0080, 0x8080)      # -0, +0, the smallest denormals, the smallest normals


# --- the grids -----------------------------------------------------------------------------------------------------------------------------------


def named(op, text, probe_only=None):
    """The Config of a named set that is all of one op (the stack's flag accepts the same text)."""
    selection = ccl_options_tp.parse_set(text)
    overrides = dict(selection.rs if op == 'rs' else selection.ag)
    if (selection.ag if op == 'rs' else selection.rs):
        raise ValueError('%s is not a %s set' % (text, op))
    return Config(selection.name, op, overrides, probe_only)


def base_config(op):
    return Config('served', op, {}, None)


def sweep_grid(op, quick=False):
    """The one-factor-at-a-time configs around the served values, served first."""
    values = QUICK_VALUES if quick else VALUES
    configs = [base_config(op)]
    for letter in 'lwcb':
        for value in values[letter]:
            configs.append(named(op, '%s-%s%d' % (op, letter, value)))
    if op == 'ag':
        configs += [named(op, token) for token in AG_FLAGS]
    return configs


def probe_only_grid(op):
    """The barrier-removal arms: not in any named set (ccl_options_tp.OPTIONS says why), measured so the owner knows what the barrier costs."""
    return [Config('%s-nobar' % op, op, {'barrier_semaphore': None}, 'nobar'), Config('%s-pbuf' % op, op, {}, 'pbuf')]


def axis_of(config):
    """The one option an OFAT config changes ('' for served or a combination)."""
    return next(iter(config.overrides)) if len(config.overrides) == 1 and not config.probe_only else ''


def combinations(op, results, limit=4):
    """Combined configs from the promoted one-factor results: the best value of each axis together, then the best two axes, then the best three.
    `results` maps a config name to its result row (with 'promote' and 'median_gain_us'). Empty when fewer than two axes won."""
    best = {}
    for row in results:
        if not row.get('promote') or not row.get('axis'):
            continue
        if row['axis'] not in best or row['median_gain_us'] > best[row['axis']]['median_gain_us']:
            best[row['axis']] = row
    ranked = sorted(best.values(), key=lambda row: -row['median_gain_us'])
    configs, seen = [], set()
    for count in sorted(set([len(ranked), 2, 3])):
        if 2 <= count <= len(ranked):
            text = '+'.join(row['name'] for row in ranked[:count])
            config = named(op, text)
            if config.name not in seen:
                seen.add(config.name)
                configs.append(config)
    return configs[:limit]


# --- statistics ----------------------------------------------------------------------------------------------------------------------------------


def per_call_us(high_ns, low_ns):
    """(median(T_HIGH) - median(T_LOW)) / (N_HIGH - N_LOW) in microseconds from two lists of replay times in nanoseconds."""
    return (statistics.median(high_ns) - statistics.median(low_ns)) / (N_HIGH - N_LOW) / 1000.0


def paired_verdict(base_us, cand_us, noise_us):
    """The paired read of one candidate against the served config: per-round gain = base - candidate (positive: the candidate is faster)."""
    gains = [b - c for b, c in zip(base_us, cand_us)]
    median = statistics.median(gains)
    wins = sum(1 for gain in gains if gain > 0)
    needed = max(MIN_GAIN_US, 2.0 * noise_us)
    return dict(median_gain_us=round(median, 3), wins=wins, rounds=len(gains), needed_us=round(needed, 3), gains_us=[round(g, 3) for g in gains],
                base_us=round(statistics.median(base_us), 3), cand_us=round(statistics.median(cand_us), 3),
                faster=median >= needed and wins >= len(gains) - 1)


def bits_equal(torch, left, right):
    """(elements, differing) of two bfloat16 tensors compared as int16 bit patterns."""
    if tuple(left.shape) != tuple(right.shape):
        raise ValueError('shapes %s and %s cannot be compared' % (tuple(left.shape), tuple(right.shape)))
    different = left.contiguous().view(torch.int16) != right.contiguous().view(torch.int16)
    return int(different.numel()), int(different.sum())


def fingerprint(torch, tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.int16).numpy().tobytes()).hexdigest()


def compare_reports(left, right):
    """Which scenarios' served-config fingerprints two reports (two fabric configs, two payloads) share: {scenario: True|False|None}. None: not in both."""
    result = {}
    for name in sorted(set(left.get('fingerprints', {})) | set(right.get('fingerprints', {}))):
        a, b = left.get('fingerprints', {}).get(name), right.get('fingerprints', {}).get(name)
        result[name] = None if a is None or b is None else a == b
    return result


# --- the host data -------------------------------------------------------------------------------------------------------------------------------


def rs_partials(torch, seed):
    """One heavy-tailed full-mantissa bfloat16 partial per chip: (CHIPS, 1, ROWS, WIDTH). The special bit patterns sit in the first row."""
    generator = torch.Generator().manual_seed(7000 + seed)
    base = torch.randn(CHIPS, 1, ROWS, WIDTH, generator=generator)
    scale = torch.exp(torch.randn(CHIPS, 1, ROWS, WIDTH, generator=generator))
    return (base * scale).to(torch.bfloat16)


def ag_inputs(torch, seed):
    """One distinct bfloat16 tile block per chip: (CHIPS, 1, ROWS, SHARD), special bit patterns in the first row of every chip."""
    generator = torch.Generator().manual_seed(9000 + seed)
    data = (torch.randn(CHIPS, 1, ROWS, SHARD, generator=generator) * torch.exp(torch.randn(CHIPS, 1, ROWS, SHARD, generator=generator))).to(torch.bfloat16)
    bits = data.view(torch.int16)
    for index, pattern in enumerate(SPECIAL_BITS):
        bits[:, 0, 0, index] = pattern if pattern < 0x8000 else pattern - 0x10000
    return data


# --- the harness ---------------------------------------------------------------------------------------------------------------------------------


class Harness(object):
    """The probe's hands: uploads, the two calls with a config's keywords, traces. `ttnn`, `torch`, `mesh`, `collective` (the model's TT_CCL) and
    `all_reduce` (the model's tt_all_reduce) are the image's, or fakes in the tests."""

    def __init__(self, ttnn, torch, mesh, collective, all_reduce, clock=time.perf_counter_ns, log=print):
        self.ttnn, self.torch, self.mesh, self.collective, self.all_reduce = ttnn, torch, mesh, collective, all_reduce
        self.clock, self.log = clock, log
        self.errors_in_a_row = 0

    def note_status(self, row):
        """Count configs that ended in an error in a row, across scenarios; the limit stops the sweep (a wedged runtime fails everything the same way)."""
        self.errors_in_a_row = self.errors_in_a_row + 1 if row['status'] == 'ERROR' else 0
        if self.errors_in_a_row >= MAX_CONSECUTIVE_ERRORS:
            raise RuntimeError('%d configs in a row ended in an error (the last: %s); the runtime is not usable, the sweep stops' % (
                self.errors_in_a_row, row.get('error')))

    def memory(self, name):
        ttnn = self.ttnn
        if name == 'dram':
            return ttnn.DRAM_MEMORY_CONFIG
        if name == 'l1':
            return ttnn.L1_MEMORY_CONFIG
        if name == 'ws':
            return ttnn.create_sharded_memory_config(
                shape=(ROWS, WIDTH // (WS_CORES[0] * WS_CORES[1])), core_grid=ttnn.CoreGrid(y=WS_CORES[0], x=WS_CORES[1]),
                strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
        raise ValueError(name)

    def upload(self, host, memory):
        ttnn = self.ttnn
        return ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=memory,
                               mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0))

    def download(self, tensor):
        ttnn, torch = self.ttnn, self.torch
        return [ttnn.to_torch(part).to(torch.bfloat16) for part in ttnn.get_device_tensors(tensor)]

    def topology(self, name):
        return getattr(self.ttnn.Topology, name)

    # the model's calls, keyword for keyword (tt_all_reduce / tile_collective_tp.reduce_unit_major and DistributedNorm.forward)
    def rs_keywords(self, config, out_memory, persistent=None):
        ttnn, collective = self.ttnn, self.collective
        call = dict(persistent_output_buffers=persistent, dim=3, multi_device_global_semaphore=collective.get_and_cycle_rs_semaphore_handles(),
                    barrier_semaphore=collective.get_and_cycle_barrier_semaphore_handle(), num_links=2, memory_config=out_memory,
                    intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG, topology=ttnn.Topology.Ring, chunks_per_sync=10, num_workers_per_link=2,
                    num_buffers_per_channel=2)
        call.update(config.overrides)
        return call

    def ag_keywords(self, config, out_memory, persistent=None):
        ttnn, collective = self.ttnn, self.collective
        call = dict(persistent_output_buffer=persistent, dim=3, multi_device_global_semaphore=collective.get_and_cycle_ag_semaphore_handles(),
                    num_links=2, topology=ttnn.Topology.Ring, memory_config=out_memory,
                    barrier_semaphore=collective.get_and_cycle_barrier_semaphore_handle(), chunks_per_sync=10, num_workers_per_link=2,
                    num_buffers_per_channel=2, subdevice_id=None)
        for name, value in config.overrides.items():
            call[name] = self.topology(value) if name == 'topology' else value
        return call

    def capture(self, issue, count):
        """Capture `count` back-to-back calls of issue(); -> (trace id, the outputs the trace owns)."""
        ttnn = self.ttnn
        trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        outputs = []
        try:
            for _ in range(count):
                outputs.append(issue())
        finally:
            ttnn.end_trace_capture(self.mesh, trace, cq_id=0)
        return trace, outputs

    def replay(self, trace):
        """One replay, host-synchronised, in nanoseconds."""
        ttnn = self.ttnn
        started = self.clock()
        ttnn.execute_trace(self.mesh, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.mesh)
        return self.clock() - started


class Arm(object):
    """One config in one scenario: its eager results, its two captured traces, its persistent buffers (probe-only arms)."""

    def __init__(self, harness, scenario, config):
        self.h, self.scenario, self.config = harness, scenario, config
        self.input = None
        self.view = None
        self.traces = {}
        self.owned = []
        self.persistent = []
        self.issued = 0

    # -- calls
    def out_memory(self):
        return self.h.memory('dram' if self.scenario.op == 'rs' else self.scenario.output)

    def prepare(self, host):
        """Upload the input of the timing traces and, for a persistent-buffer arm, make the rotating buffers (outside any capture)."""
        h = self.h
        memory = h.memory(self.scenario.input)
        self.input = h.upload(host, memory)
        self.owned.append(self.input)
        if self.scenario.op == 'rs':
            self.view = h.ttnn.reshape(self.input, RS_VIEW)
        if self.config.probe_only == 'pbuf':
            self.persistent = [self.make_buffers() for _ in range(2)]

    def make_buffers(self):
        h, ttnn, torch = self.h, self.h.ttnn, self.h.torch
        if self.scenario.op == 'ag':
            zeros = torch.zeros(CHIPS, 1, ROWS, WIDTH, dtype=torch.bfloat16)
            buffer = h.upload(zeros, self.out_memory())
            self.owned.append(buffer)
            return buffer
        zeros = torch.zeros(CHIPS, RS_VIEW[1], RS_VIEW[2], SHARD, dtype=torch.bfloat16)
        output = h.upload(zeros, ttnn.DRAM_MEMORY_CONFIG)
        intermediate, penult = ttnn.experimental.reduce_scatter_minimal_async_create_intermediate_buffer(self.view, dim=3, topology=ttnn.Topology.Ring)
        self.owned += [output, intermediate, penult]
        return [intermediate, output, penult]

    def issue(self, source=None, view=None):
        """One call of the arm's config on its prepared input (or on `source`: a tensor, with `view` for the reduce-scatter)."""
        h, ttnn = self.h, self.h.ttnn
        persistent = None
        if self.persistent:
            persistent = self.persistent[self.issued % len(self.persistent)]
            self.issued += 1
        if self.scenario.op == 'rs':
            return ttnn.experimental.reduce_scatter_minimal_async(view if view is not None else self.view,
                                                                  **h.rs_keywords(self.config, self.out_memory(), persistent))
        return ttnn.experimental.all_gather_async(source if source is not None else self.input,
                                                  **h.ag_keywords(self.config, self.out_memory(), persistent))

    # -- eager exactness
    def eager(self, host):
        """The call on a fresh upload of `host` (CHIPS, ...): -> the per-chip results as torch tensors."""
        h, ttnn = self.h, self.h.ttnn
        tensor = h.upload(host, h.memory(self.scenario.input))
        view = ttnn.reshape(tensor, RS_VIEW) if self.scenario.op == 'rs' else None
        out = self.issue(tensor, view)
        copies = h.download(out)
        if not self.persistent:                                  # a persistent output is the arm's own and is freed with it
            ttnn.deallocate(out)
        ttnn.deallocate(tensor)
        return copies

    # -- traces
    def capture(self, count):
        trace, outputs = self.h.capture(self.issue, count)
        self.traces[count] = (trace, outputs)
        return trace, outputs

    def trace_outputs(self, count):
        """Replay the captured trace of `count` calls once and read its first and last output, per chip (a capture records the calls, it does
        not run them)."""
        trace, outputs = self.traces[count]
        self.h.replay(trace)
        return [self.h.download(outputs[0]), self.h.download(outputs[-1])]

    def release(self):
        h, ttnn = self.h, self.h.ttnn
        for trace, outputs in self.traces.values():
            ttnn.release_trace(h.mesh, trace)
            for output in outputs:
                try:
                    ttnn.deallocate(output)
                except Exception:  # noqa: BLE001 - a persistent output is owned below
                    pass
        self.traces = {}
        for tensor in self.owned:
            try:
                ttnn.deallocate(tensor)
            except Exception:  # noqa: BLE001
                pass
        self.owned = []


class Sweep(object):
    """The sweep of one scenario: the served arm, the candidates, the paired reads."""

    def __init__(self, harness, scenario, options, report, save, deadline):
        self.h, self.scenario, self.options, self.report, self.save = harness, scenario, options, report, save
        self.deadline = deadline
        self.base_arm = None
        self.noise_us = 0.0
        self.torch = harness.torch
        self.rows = []
        self.sources = []
        self.anchors = []
        self.log = harness.log

    # -- host references
    def host_data(self):
        torch = self.torch
        for seed in range(self.options.seeds):
            if self.scenario.op == 'rs':
                source = rs_partials(torch, seed)
                self.sources.append(source)
                self.anchors.append(self.rs_anchor(source))
            else:
                source = ag_inputs(torch, seed)
                self.sources.append(source)
                self.anchors.append(torch.cat([source[chip:chip + 1] for chip in range(CHIPS)], dim=3))

    def rs_anchor(self, source):
        """The sequential engine's call on each 32-row tile alone (the model's tt_all_reduce, Ring, the model's keywords), the four chips' columns joined."""
        h, ttnn, torch = self.h, self.h.ttnn, self.torch
        tiles = []
        for tile in range(ROWS // TILE):
            piece = source[:, :, tile * TILE:(tile + 1) * TILE, :].contiguous()
            out = h.all_reduce(h.upload(piece, ttnn.DRAM_MEMORY_CONFIG), h.mesh, h.collective, cluster_axis=0, dim=3, topology=ttnn.Topology.Ring,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
            tiles.append(torch.cat(h.download(out), dim=3))
            ttnn.deallocate(out)
        return tiles

    def compare(self, copies, seed):
        """(elements, differing) of one eager or traced result against the reference of `seed`, on every chip."""
        torch = self.torch
        if self.scenario.op == 'rs':
            joined = torch.cat(copies, dim=3)                       # (1, 2, 32, 5120): the four chips' columns side by side
            elements = differing = 0
            for tile, anchor in enumerate(self.anchors[seed]):
                count, bad = bits_equal(torch, joined[:, tile:tile + 1], anchor)
                elements, differing = elements + count, differing + bad
            return elements, differing
        elements = differing = 0
        for chip in copies:
            count, bad = bits_equal(torch, chip, self.anchors[seed])
            elements, differing = elements + count, differing + bad
        return elements, differing

    # -- one config
    def exactness(self, arm):
        elements = differing = 0
        for seed in range(self.options.seeds):
            count, bad = self.compare(arm.eager(self.sources[seed]), seed)
            elements, differing = elements + count, differing + bad
        return elements, differing

    def traced_exactness(self, arm, seed):
        """The captured trace's first and last output against the reference."""
        elements = differing = 0
        for copies in arm.trace_outputs(N_LOW):
            count, bad = self.compare(copies, seed)
            elements, differing = elements + count, differing + bad
        return elements, differing

    def measure_row(self, config):
        """Exactness (eager, then in-trace) of one config; the row, and the arm with its traces still captured when it is exact."""
        row = dict(name=config.name, op=config.op, axis=axis_of(config), probe_only=config.probe_only, overrides=dict(
            (key, str(value)) for key, value in config.overrides.items()))
        arm = Arm(self.h, self.scenario, config)
        try:
            arm.prepare(self.sources[0])
            elements, differing = self.exactness(arm)
            row.update(elements=elements, differing=differing)
            if differing:
                row.update(exact=False, status='FAIL-BYTES')
                arm.release()
                return row, None
            arm.capture(N_HIGH)
            arm.capture(N_LOW)
            trace_elements, trace_differing = self.traced_exactness(arm, 0)
            row.update(trace_elements=trace_elements, trace_differing=trace_differing)
            if trace_differing:
                row.update(exact=False, status='FAIL-BYTES-IN-TRACE')
                arm.release()
                return row, None
            row.update(exact=True, status='EXACT')
            return row, arm
        except Exception as error:  # noqa: BLE001 - a config the runtime refuses is a result
            row.update(exact=None, status='ERROR', error='%s: %s' % (type(error).__name__, str(error)[:300]))
            try:
                arm.release()
            except Exception:  # noqa: BLE001
                pass
            return row, None

    def time_pair(self, base, cand):
        """ROUNDS paired rounds of base and cand (A B, then B A, ...): the per-call microseconds of each, per round."""
        h = self.h
        base_us, cand_us = [], []
        for arm in (base, cand):
            for count in (N_HIGH, N_LOW):
                h.replay(arm.traces[count][0])                    # one untimed replay after capture
        for index in range(self.options.rounds):
            order = (base, cand) if index % 2 == 0 else (cand, base)
            seen = {}
            for arm in order:
                high = [h.replay(arm.traces[N_HIGH][0]) for _ in range(self.options.replays)]
                low = [h.replay(arm.traces[N_LOW][0]) for _ in range(self.options.replays)]
                seen[id(arm)] = per_call_us(high, low)
            base_us.append(seen[id(base)])
            cand_us.append(seen[id(cand)])
        return base_us, cand_us

    def run(self, configs):
        """Measure `configs` (served first when the sweep has not started) and extend self.rows."""
        for config in configs:
            started = time.time()
            self.deadline.arm('%s %s' % (self.scenario.name, config.name))
            row, arm = self.measure_row(config)
            if arm is not None:
                try:
                    base_us, cand_us = self.time_pair(self.base_arm, arm)
                    row.update(paired_verdict(base_us, cand_us, self.noise_us))
                    row['promote'] = bool(row['faster']) and not config.probe_only
                except Exception as error:  # noqa: BLE001
                    row.update(status='ERROR', error='timing: %s: %s' % (type(error).__name__, str(error)[:300]), promote=False)
                arm.release()
            else:
                row['promote'] = False
            self.deadline.disarm()
            row['seconds'] = round(time.time() - started, 1)
            self.rows.append(row)
            self.report['scenarios'][self.scenario.name] = dict(rows=self.rows, noise_us=round(self.noise_us, 3))
            self.save(self.report)
            self.log('CCL_SWEEP %s %s status=%s%s' % (self.scenario.name, row['name'], row['status'], ' gain_us=%s wins=%s/%s base_us=%s cand_us=%s promote=%s' % (
                row.get('median_gain_us'), row.get('wins'), row.get('rounds'), row.get('base_us'), row.get('cand_us'), row.get('promote'))
                if 'median_gain_us' in row else (' error=' + row['error'] if row.get('error') else '')))
            self.h.note_status(row)

    def start(self):
        """Host references, the served arm (exactness against the anchor is the control), the A/A noise."""
        self.host_data()
        self.deadline.arm('%s served' % self.scenario.name)
        row, base = self.measure_row(base_config(self.scenario.op))
        self.base_arm = base
        if base is None:
            row['promote'] = False
            self.rows.append(row)
            self.deadline.disarm()
            self.report['scenarios'][self.scenario.name] = dict(rows=self.rows)
            self.save(self.report)
            self.log('CCL_SWEEP %s served status=%s%s' % (self.scenario.name, row['status'], (' error=' + row['error']) if row.get('error') else ''))
            self.h.note_status(row)
            return False
        twin_row, twin = self.measure_row(base_config(self.scenario.op))
        if twin is None:
            self.rows.append(row)
            self.deadline.disarm()
            return False
        base_us, twin_us = self.time_pair(base, twin)
        self.noise_us = statistics.median([abs(a - b) for a, b in zip(base_us, twin_us)])
        twin.release()
        row.update(base_us=round(statistics.median(base_us), 3), noise_us=round(self.noise_us, 3), promote=False)
        self.rows.append(row)
        self.report['scenarios'][self.scenario.name] = dict(rows=self.rows, noise_us=round(self.noise_us, 3))
        self.report.setdefault('fingerprints', {})[self.scenario.name] = self.fingerprint()
        self.deadline.disarm()
        self.save(self.report)
        self.log('CCL_SWEEP %s served status=%s per_call_us=%s noise_us=%s' % (self.scenario.name, row['status'], row['base_us'], row['noise_us']))
        return True

    def fingerprint(self):
        arm = Arm(self.h, self.scenario, base_config(self.scenario.op))
        try:
            arm.prepare(self.sources[0])
            copies = arm.eager(self.sources[0])
        finally:
            arm.release()
        return fingerprint(self.torch, self.torch.cat(copies, dim=3))

    def finish(self):
        """Release the served arm's traces and buffers (the sweeps of all scenarios do not hold the L1 of the sharded gathers at once)."""
        if self.base_arm is not None:
            self.base_arm.release()
            self.base_arm = None

    def rebuild_base(self):
        """A fresh served arm for the probe-only stage; False when it is no longer exact (then nothing is measured against it)."""
        self.deadline.arm('%s served (probe-only stage)' % self.scenario.name)
        row, self.base_arm = self.measure_row(base_config(self.scenario.op))
        self.deadline.disarm()
        return self.base_arm is not None


def summarise(report):
    """The verdict fields of a finished report: per scenario the served per-call time, the best promoted config and its gain, the promote lists."""
    summary = {}
    for name, scenario in sorted(report.get('scenarios', {}).items()):
        rows = scenario.get('rows', [])
        served = next((row for row in rows if row['name'] == 'served'), None)
        best = None
        for row in rows:
            if row.get('promote') and (best is None or row['median_gain_us'] > best['median_gain_us']):
                best = row
        summary[name] = dict(served_us=served.get('base_us') if served else None, noise_us=scenario.get('noise_us'),
                             best=best['name'] if best else None, best_gain_us=best['median_gain_us'] if best else None,
                             exact=sum(1 for row in rows if row.get('exact')), measured=len(rows),
                             failed=[row['name'] for row in rows if row.get('status', '').startswith('FAIL')],
                             errors=[row['name'] for row in rows if row.get('status') == 'ERROR'],
                             promote=[row['name'] for row in rows if row.get('promote')])
    return summary


def verdict(report):
    """(text, exit status): DONE when every served config was exact and measured, BASE-INEXACT when a served config differs from the sequential
    engine's call (nothing else in that scenario means anything), INCOMPLETE when a scenario did not finish, NOT-MEASURED when the mesh did not open."""
    if not report.get('opened'):
        return 'NOT-MEASURED', 2
    scenarios = report.get('scenarios', {})
    if not scenarios:
        return 'INCOMPLETE', 0
    for scenario in scenarios.values():
        served = next((row for row in scenario.get('rows', []) if row['name'] == 'served'), None)
        if served is None or served.get('status') != 'EXACT':
            return 'BASE-INEXACT' if served is not None and served.get('status', '').startswith('FAIL') else 'INCOMPLETE', 1
    if report.get('error') or report.get('unfinished'):
        return 'INCOMPLETE', 0
    return 'DONE', 0


def verdict_line(report, text):
    summary = report.get('summary', {})
    promote = sorted(set(name for scenario in summary.values() for name in scenario.get('promote', [])))
    best = ','.join('%s:%s:%s' % (name, scenario.get('best'), scenario.get('best_gain_us')) for name, scenario in sorted(summary.items()))
    return 'CCL_SWEEP verdict=%s fabric=%s payload=%s scenarios=%d exact=%d/%d promote=%s best=%s' % (
        text, report.get('fabric'), report.get('payload_actual'), len(summary), sum(s['exact'] for s in summary.values()),
        sum(s['measured'] for s in summary.values()), '+'.join(promote) or '-', best or '-')


def run(options, ttnn, torch, mesh, collective, all_reduce, report, save, deadline=None, log=print, clock=time.perf_counter_ns):
    """The sweep on an open mesh; fills `report` (saved after every config). `deadline` (Deadline, or None for no per-config limit)."""
    harness = Harness(ttnn, torch, mesh, collective, all_reduce, clock=clock, log=log)
    deadline = deadline or Deadline(None, None)
    wanted = [scenario for scenario in SCENARIOS if options.only in ('all', scenario[1])]
    sweeps = []
    try:
        for name, op, source, output in wanted:
            sweep = Sweep(harness, Scenario(name, op, source, output), options, report, save, deadline)
            sweeps.append(sweep)
            if not sweep.start():
                report.setdefault('unfinished', []).append(name)
                continue
            sweep.run(sweep_grid(op, options.quick)[1:])
            promoted = combinations(op, sweep.rows)
            if promoted:
                sweep.run(promoted)
            sweep.finish()
        if not options.skip_probe_only:
            for sweep in sweeps:
                if sweep.rows and sweep.rows[0].get('exact') and sweep.rebuild_base():
                    sweep.run(probe_only_grid(sweep.scenario.op))
                    sweep.finish()
    finally:
        for sweep in sweeps:
            sweep.finish()
    return report


class Deadline(object):
    """A per-config limit: arm(label) starts a timer that, when it fires, writes the partial report naming the config and exits 3 (a hung collective
    cannot be recovered in-process); disarm() stops it. With seconds None it does nothing (the tests)."""

    def __init__(self, report, path, seconds=None, exit_function=os._exit, log=print):
        self.report, self.path, self.seconds, self.exit_function, self.log = report, path, seconds, exit_function, log
        self.timer = None

    def arm(self, label):
        self.disarm()
        if self.seconds is None:
            return
        self.timer = threading.Timer(self.seconds, self.fire, args=(label,))
        self.timer.daemon = True
        self.timer.start()

    def disarm(self):
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None

    def fire(self, label):
        self.report['watchdog'] = label
        try:
            with open(self.path, 'w') as handle:
                json.dump(self.report, handle, indent=2, sort_keys=True, default=str)
        except Exception:  # noqa: BLE001 - the process exits either way
            pass
        finally:
            self.log('CCL_SWEEP watchdog: %s took more than %s s; the partial report is written; reset all four cards before the next job'
                     % (label, self.seconds))
            sys.stdout.flush()
            self.exit_function(3)
