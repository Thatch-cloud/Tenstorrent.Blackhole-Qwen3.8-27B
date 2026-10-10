"""H2: do an all_gather on the target sub-device and an all_gather on the drafter sub-device run concurrently on the FOUR-card (1, 4) mesh?

H1 (one card) decides whether two compute traces overlap through two command queues. H2 decides the part H1 cannot: the collectives. Every
verify layer ends in a gather over the four-card ring and so does the quad draft; if the two sub-devices' gathers cannot be in flight together
(the fabric's sender channel is one connection per link), the drafter's gathers queue behind the target's and the overlap is lost whatever the
compute does. Each sub-device has its OWN semaphores (a pool of double-buffered all-gather semaphores and barrier semaphores created on its
own cores), its own command queue (target: queue 0, drafter: queue 1) and its own sub_core_grids, so nothing is shared but the fabric.

Arms (exactness against the host's concatenation on every chip, then timing in interleaved rounds):

  eager     solo target gather, solo drafter gather, then both concurrently (queue 0 and queue 1), a few times: the cheap hang detector
  t_solo, d_solo   a trace of K gathers on one sub-device alone
  shared    BOTH traces concurrently, both on fabric link 0 (num_links 1 on each): (a) sharing a link
  shared2   the same with num_links 2 on each (information only; production gathers use both links)
  separate  (b) the drafter's gathers on link 1, the target's on link 0: needs the link-offset graft (ag_link_offset.patch: the all-gather
            factory adds QWEN_AG_LINK_OFFSET_SD1 to the fabric link of every sub-device-1 program). There is NO Python-visible way to pick a
            link in this tt-metal (num_links counts links from 0; the factory passes its loop index to the fabric), so without the graft this arm
            reads NOT-RUN. The graft is a prepared source, not built.

PASS = shared-link concurrent wall <= 1.1 x max(solo), every compared tensor identical, nothing hung. PASS-SEPARATE-LINKS = only the separate
links overlap (the graft is required). The run holds all four cards: put it LAST in a window and follow it with an all-board reset.

Last stdout lines: 'SUBDEV_H2 verdict=...' then one JSON object (kind subdev-h2). Exit: 0 PASS / PASS-SEPARATE-LINKS / UNTIMED-PASS, 1 FAIL,
2 NOT-MEASURED, 3 the watchdog (a hang; the heartbeat lines 'SUBDEV_H2 alive stage=...' show where).
"""

import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import subdev_plan as plan  # noqa: E402

KIND = 'subdev-h2'
TAG = 'SUBDEV_H2'
LINK_ENV = 'QWEN_AG_LINK_OFFSET_SD1'
CHIPS = 4
# (rows, per-chip width in elements): the verify gather of a 64-row slice (hidden 5120 / 4 chips) and the quad draft's (hidden 2560 / 4).
SHAPES = dict(target=(64, 1280), drafter=(32, 640))
ARMS = ('t_solo', 'd_solo', 'shared', 'shared2', 'separate')


def verdict_line(text, evidence, extra=None):
    def number(key):
        value = evidence.get(key)
        return '-' if value is None else value

    parts = ['%s verdict=%s' % (TAG, text), 'shared_ratio=%s' % number('shared_ratio'), 'separate_ratio=%s' % number('separate_ratio'),
             'shared2_ratio=%s' % number('shared2_ratio'), 't_ms=%s' % number('t_solo_ms'), 'd_ms=%s' % number('d_solo_ms'),
             'shared_ms=%s' % number('shared_ms'), 'separate_ms=%s' % number('separate_ms'),
             'bytes=%s' % ('identical' if not evidence.get('mismatched') and evidence.get('compared') else
                           'DIFFER(%s/%s)' % (evidence.get('mismatched'), evidence.get('compared'))),
             'timing=%d' % int(bool(evidence.get('timing')))]
    for key, value in sorted((extra or {}).items()):
        parts.append('%s=%s' % (key, value))
    return ' '.join(parts)


class Side(object):
    def __init__(self, name, rect, sd, crs, rows, shard):
        self.name, self.rect, self.sd, self.crs, self.rows, self.shard = name, rect, sd, crs, rows, shard
        self.source = self.expected = None
        self.pools = {}


class Pool(object):
    """The double-buffered semaphores of one sub-device's gathers: two sets of (two all-gather semaphores), two barrier semaphores."""

    def __init__(self, ttnn, mesh, crs):
        self.ag = [[ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(2)] for _ in range(2)]
        self.barrier = [ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(2)]
        self.index = 0

    def next(self):
        index, self.index = self.index, (self.index + 1) % 2
        return self.ag[index], self.barrier[index]


class Harness2(object):
    def __init__(self, ttnn, torch, mesh, options, watchdog, heartbeat=None, log=plan.say, clock=time.perf_counter_ns, report=None, persist=None,
                 environ=None):
        self.ttnn, self.torch, self.mesh, self.o, self.wd, self.log, self.clock = ttnn, torch, mesh, options, watchdog, log, clock
        self.heartbeat = heartbeat
        self.report = report if report is not None else {}
        self.persist = persist or (lambda: None)
        self.environ = os.environ if environ is None else environ
        self.keep, self.traces = [], {}
        self.sd = [ttnn.SubDeviceId(0), ttnn.SubDeviceId(1)]
        self.exact = dict(compared=0, mismatched=0, cases=[])
        self.arm_ns = {name: [] for name in ARMS}
        self.manager_id = None
        self.topology = getattr(ttnn.Topology, options.topology)

    def span(self, label, seconds=None):
        return self.wd.span(label, self.o.watchdog_s if seconds is None else seconds)

    def compile_span(self, label):
        return self.wd.span(label, self.o.compile_watchdog_s)

    def stage(self, name):
        self.report['last_stage'] = name
        if self.heartbeat is not None:
            self.heartbeat.set(name)
        self.persist()
        self.log('%s stage=%s' % (TAG, name))

    def crs(self, rect):
        ttnn = self.ttnn
        return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(rect[0], rect[1]), ttnn.CoreCoord(rect[2], rect[3]))})

    def sync_all(self):
        self.ttnn.synchronize_device(self.mesh)

    def sync(self, queue, sub_devices):
        self.ttnn.synchronize_device(self.mesh, cq_id=queue, sub_device_ids=sub_devices)

    def replay(self, name, queue):
        self.ttnn.execute_trace(self.mesh, self.traces[name], cq_id=queue, blocking=False)

    # ------------------------------------------------------------------ build

    def side(self, name, rect, sd, rows, shard):
        side = Side(name, rect, sd, self.crs(rect), rows, shard)
        torch, ttnn = self.torch, self.ttnn
        generator = torch.Generator().manual_seed(self.o.seed + (0 if name == 'target' else 1))
        full = torch.randn(1, 1, rows, CHIPS * shard, generator=generator).to(torch.bfloat16)
        source = torch.cat([full[..., chip * shard:(chip + 1) * shard] for chip in range(CHIPS)], dim=0)
        side.expected = full.contiguous().view(torch.int16)
        side.source = self.upload(source)
        return side

    def upload(self, values):
        ttnn = self.ttnn
        tensor = ttnn.from_torch(values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                 mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0))
        self.keep.append(tensor)
        return tensor

    def pool(self, side, key):
        if key not in side.pools:
            side.pools[key] = Pool(self.ttnn, self.mesh, side.crs)
        return side.pools[key]

    def gather(self, side, tensor, links, pool):
        ttnn = self.ttnn
        semaphores, barrier = pool.next()
        out = ttnn.experimental.all_gather_async(
            tensor, persistent_output_buffer=None, dim=3, multi_device_global_semaphore=semaphores, num_links=links,
            topology=self.topology, memory_config=ttnn.DRAM_MEMORY_CONFIG, barrier_semaphore=barrier, chunks_per_sync=10,
            num_workers_per_link=2, num_buffers_per_channel=2, subdevice_id=side.sd, sub_core_grids=side.crs)
        self.keep.append(out)
        return out

    def run_gathers(self, side, queue, links, pool, count):
        with self.ttnn.command_queue(queue):
            return [self.gather(side, side.source, links, pool) for _ in range(count)]

    def capture(self, label, side, queue, links, pool):
        ttnn = self.ttnn
        with self.compile_span('capture-' + label):
            trace_id = ttnn.begin_trace_capture(self.mesh, cq_id=queue)
            try:
                outputs = self.run_gathers(side, queue, links, pool, self.o.gathers)
            finally:
                ttnn.end_trace_capture(self.mesh, trace_id, cq_id=queue)
            self.sync_all()
        self.traces[label] = trace_id
        return outputs

    # ------------------------------------------------------------------ comparison

    def check(self, case, side, outputs, informational=False):
        """Every chip's part of the first, middle and last gather against the host concatenation, bitwise."""
        ttnn, torch = self.ttnn, self.torch
        picks = sorted({0, len(outputs) // 2, len(outputs) - 1})
        compared = mismatched = 0
        bad = []
        with self.span('readback'):
            for pick in picks:
                for chip, part in enumerate(ttnn.get_device_tensors(outputs[pick])):
                    host = ttnn.to_torch(part).contiguous().view(torch.int16).reshape(side.expected.shape)
                    compared += 1
                    if not torch.equal(host, side.expected):
                        mismatched += 1
                        bad.append('gather %d chip %d' % (pick, chip))
        entry = dict(case=case, compared=compared, mismatched=mismatched, first_bad=bad[:5])
        if informational:
            entry['informational'] = True
        else:
            self.exact['compared'] += compared
            self.exact['mismatched'] += mismatched
        self.exact['cases'].append(entry)
        if bad:
            self.log('%s bytes DIFFER case=%s mismatched=%d/%d first=%s' % (TAG, case, mismatched, compared, bad[:3]))
        return not bad

    # ------------------------------------------------------------------ the experiment

    def eager(self):
        """Solo gathers, then both concurrently on two queues, a few times: bytes against the host and the first sign of a hang."""
        t, d = self.t_side, self.d_side
        self.stage('eager-solo')
        with self.compile_span('eager-target'):
            out_t = self.run_gathers(t, 0, 1, self.pool(t, 'eager'), 2)
            self.sync(0, [self.sd[0]])
        self.check('eager target solo', t, out_t)
        with self.compile_span('eager-drafter'):
            out_d = self.run_gathers(d, 1, 1, self.pool(d, 'eager'), 2)
            self.sync(1, [self.sd[1]])
        self.check('eager drafter solo', d, out_d)
        self.stage('eager-concurrent')
        for index in range(self.o.eager_rounds):
            with self.span('eager-concurrent'):
                out_t = self.run_gathers(t, 0, 1, self.pool(t, 'eager'), 2)
                out_d = self.run_gathers(d, 1, 1, self.pool(d, 'eager'), 2)
                self.sync(1, [self.sd[1]])
                self.sync(0, [self.sd[0]])
            self.sync_all()
            self.check('eager concurrent #%d target' % index, t, out_t)
            self.check('eager concurrent #%d drafter' % index, d, out_d)

    def arm_fns(self):
        sd0, sd1 = self.sd
        both = lambda a, b, flip: (self.replay(b, 1), self.replay(a, 0)) if flip else (self.replay(a, 0), self.replay(b, 1))  # noqa: E731

        def shared(trace_t, trace_d):
            def run(flip):
                both(trace_t, trace_d, flip)
                self.sync(0, [sd0])
                self.sync(1, [sd1])
            return run

        fns = dict(t_solo=lambda flip: (self.replay('t', 0), self.sync(0, [sd0])),
                   d_solo=lambda flip: (self.replay('d', 1), self.sync(1, [sd1])),
                   shared=shared('t', 'd'))
        if 't2' in self.traces:
            fns['shared2'] = shared('t2', 'd2')
        if 'ds' in self.traces:
            fns['separate'] = shared('t', 'ds')
        return fns

    def run_arm(self, fns, name, flip=False, record=False):
        with self.span('arm-' + name):
            started = self.clock()
            fns[name](flip)
            elapsed = self.clock() - started
        self.sync_all()
        if record:
            self.arm_ns[name].append(elapsed)

    def exactness(self, fns):
        t, d = self.t_side, self.d_side
        self.run_arm(fns, 't_solo')
        self.check('trace target solo', t, self.out['t'])
        self.run_arm(fns, 'd_solo')
        self.check('trace drafter solo', d, self.out['d'])
        for name in ('shared', 'shared2', 'separate'):
            if name not in fns:
                continue
            tt, dd, d_side = {'shared': ('t', 'd', d), 'shared2': ('t2', 'd2', d), 'separate': ('t', 'ds', getattr(self, 'd_sep', d))}[name]
            for repeat in range(2):
                self.run_arm(fns, name, flip=bool(repeat))
                self.check('%s #%d target' % (name, repeat), t, self.out[tt])
                self.check('%s #%d drafter' % (name, repeat), d_side, self.out[dd])
        self.stage('exactness-done')

    def timing(self, fns):
        order = [name for name in ARMS if name in fns]
        for index in range(self.o.warmup + self.o.rounds):
            shift = index % len(order)
            arms = order[shift:] + order[:shift]
            if (index // len(order)) % 2:
                arms = list(reversed(arms))
            for name in arms:
                self.run_arm(fns, name, flip=bool((index // 2) % 2), record=index >= self.o.warmup)
        self.stage('timing-done')

    def run(self):
        ttnn, report = self.ttnn, self.report
        # An exported offset would move EVERY sub-device-1 gather onto another link, the shared-link arm's included.
        report['link_env_was_set'] = self.environ.pop(LINK_ENV, None)
        grid = self.mesh.compute_with_storage_grid_size()
        gx, gy = int(grid.x), int(grid.y)
        report['grid'] = [gx, gy]
        report['chips'] = int(self.mesh.get_num_devices())
        self.log('%s grid=%dx%d chips=%d topology=%s' % (TAG, gx, gy, report['chips'], self.o.topology))
        if report['chips'] != CHIPS:
            raise plan.PlanError('H2 needs the four-card mesh, the mesh has %d devices' % report['chips'])
        split = plan.plan_split(gx, gy, self.o.target_cores, self.o.drafter_cores)
        report['plan'] = split
        self.stage('manager')
        started = self.clock()
        with self.span('manager'):
            self.manager_id = self.mesh.create_sub_device_manager(
                [ttnn.SubDevice([self.crs(split['target'])]), ttnn.SubDevice([self.crs(split['drafter'])])], 0)
            self.mesh.load_sub_device_manager(self.manager_id)
            self.sync_all()
        report['manager_ms'] = round((self.clock() - started) / 1e6, 3)
        self.stage('tensors')
        with self.compile_span('upload'):
            self.t_side = self.side('target', split['target'], self.sd[0], *SHAPES['target'])
            self.d_side = self.side('drafter', split['drafter'], self.sd[1], *SHAPES['drafter'])
            self.sync_all()
        self.eager()
        self.stage('capture')
        t, d = self.t_side, self.d_side
        self.out = {}
        self.out['t'] = self.capture('t', t, 0, 1, self.pool(t, 'trace1'))
        self.out['d'] = self.capture('d', d, 1, 1, self.pool(d, 'trace1'))
        if self.o.two_links:
            self.out['t2'] = self.capture('t2', t, 0, 2, self.pool(t, 'trace2'))
            self.out['d2'] = self.capture('d2', d, 1, 2, self.pool(d, 'trace2'))
        offset = self.o.link_offset
        report['link_offset'] = offset
        if offset > 0:
            # (b): the drafter's gathers on link `offset`. Only the graft reads the variable, and it reads it when a program is BUILT, so it is set
            # for this one capture and removed after it; the drafter shape is one the shared arm did not use (rows + 32), so the program is built now
            # (the sub-device is in the program hash, the environment is not).
            self.d_sep = self.side('drafter-separate', self.d_side.rect, self.sd[1], SHAPES['drafter'][0] + 32, SHAPES['drafter'][1])
            self.environ[LINK_ENV] = str(offset)
            try:
                self.out['ds'] = self.capture('ds', self.d_sep, 1, 1, self.pool(self.d_sep, 'trace1'))
            finally:
                self.environ.pop(LINK_ENV, None)
        else:
            report['separate'] = 'NOT-RUN: --link-offset is 0; the link-offset graft (ag_link_offset.patch) is not built into this image'
        report['traces'] = sorted(self.traces)
        fns = self.arm_fns()
        self.stage('exactness')
        self.exactness(fns)
        self.stage('timing')
        self.timing(fns)

    def teardown(self):
        mesh = self.mesh
        try:
            self.sync_all()
        except BaseException as error:  # noqa: BLE001
            self.report.setdefault('teardown_errors', []).append('sync: %s' % ' '.join(str(error).split())[:300])
        for name, trace_id in list(self.traces.items()):
            try:
                self.ttnn.release_trace(mesh, trace_id)
            except BaseException as error:  # noqa: BLE001
                self.report.setdefault('teardown_errors', []).append('release %s: %s' % (name, ' '.join(str(error).split())[:300]))
        self.traces.clear()
        del self.keep[:]
        self.out = {}
        if self.manager_id is not None:
            try:
                mesh.clear_loaded_sub_device_manager()
                mesh.remove_sub_device_manager(self.manager_id)
            except BaseException as error:  # noqa: BLE001
                self.report.setdefault('teardown_errors', []).append('manager: %s' % ' '.join(str(error).split())[:300])


def finish(harness, report, options, error):
    """(verdict text, evidence, printed line) from what the harness measured."""
    arms = {name: plan.summarize(samples) for name, samples in harness.arm_ns.items()} if harness is not None else {}
    report['arms'] = arms
    if harness is not None:
        report['exactness'] = harness.exact
        exactness = dict(compared=harness.exact['compared'], mismatched=harness.exact['mismatched'])
    else:
        exactness = dict(compared=0, mismatched=0)
    separate_run = bool(arms.get('separate', {}).get('n'))
    text, evidence = plan.verdict_h2(arms, exactness, timing=not options.no_timing, error=error, pass_ratio=options.pass_ratio,
                                     separate_run=separate_run)
    for name in ('t_solo', 'd_solo'):
        if arms.get(name, {}).get('median_ms') is not None:
            evidence[name + '_ms'] = arms[name]['median_ms']
    extra = dict(links=1, topology=options.topology, link_offset=report.get('link_offset', 0),
                 separate='RUN' if separate_run else 'NOT-RUN')
    if error:
        extra['error'] = '"%s"' % ' '.join(str(error).split())[:200].replace('"', "'")
    line = verdict_line(text, evidence, extra)
    report.update(verdict=text, evidence=evidence)
    return text, evidence, line


def add_arguments(parser):
    parser.add_argument('--target-cores', type=int, default=plan.TARGET_CORES)
    parser.add_argument('--drafter-cores', type=int, default=plan.DRAFTER_CORES)
    parser.add_argument('--gathers', type=int, default=24, help='all-gathers in each trace')
    parser.add_argument('--eager-rounds', type=int, default=3)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=4)
    parser.add_argument('--topology', choices=('Ring', 'Linear'), default='Ring', help='the model picks Ring on four p150')
    parser.add_argument('--no-two-links', dest='two_links', action='store_false', help='skip the num_links 2 information arm')
    parser.add_argument('--link-offset', type=int, default=0,
                        help='arm (b): the drafter gathers on fabric link N (needs the link-offset graft, which no image has built; 0 skips the arm)')
    parser.add_argument('--no-timing', action='store_true', help='bytes and hangs only (the watcher pass)')
    parser.add_argument('--pass-ratio', type=float, default=plan.PASS_RATIO)
    parser.add_argument('--watchdog-s', type=float, default=180.0)
    parser.add_argument('--compile-watchdog-s', type=float, default=900.0)
    parser.add_argument('--trace-region-bytes', type=int, default=134217728)
    parser.add_argument('--heartbeat-s', type=float, default=30.0)
    parser.add_argument('--seed', type=int, default=0)
    return parser


def problems_of(options):
    problems = []
    if options.link_offset < 0 or options.link_offset > 1:
        problems.append('--link-offset must be 0 or 1 (the fabric trains two links per card pair)')
    if options.gathers < 2:
        problems.append('--gathers must be at least 2')
    if options.rounds < 3 or options.warmup < 0 or options.eager_rounds < 1:
        problems.append('--rounds >= 3, --warmup >= 0, --eager-rounds >= 1')
    if options.pass_ratio <= 1.0:
        problems.append('--pass-ratio must exceed 1.0')
    if options.target_cores < 1 or options.drafter_cores < 1:
        problems.append('both sub-devices need at least one core')
    return problems
