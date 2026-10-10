"""H2: do an all_gather on the target sub-device and an all_gather on the drafter sub-device run concurrently on the FOUR-card (1, 4) mesh?

H1 (one card) decides whether two compute traces overlap through two command queues. H2 decides the part H1 cannot: the collectives. Every
verify layer ends in a gather over the four-card ring and so does the quad draft; if the two sub-devices' gathers cannot be in flight together
(the fabric's sender channel is one connection per link), the drafter's gathers queue behind the target's and the overlap is lost whatever the
compute does. Each sub-device has its OWN semaphores (a pool of double-buffered all-gather semaphores and barrier semaphores created on its
own cores), its own command queue (target: queue 0, drafter: queue 1) and its own sub_core_grids, so nothing is shared but the fabric.

Arms (exactness against the host's concatenation on every chip, then timing in interleaved rounds):

  eager     solo target gather, solo drafter gather, then both concurrently (queue 0 and queue 1), a few times: the cheap hang detector
  t_solo, d_solo   a trace of K gathers on one sub-device alone
  t_solo2   the target's trace at num_links 2 alone: link_cost = t_solo / t_solo2, what the target pays when the drafter owns link 1 (arm group link_cost)
  separate  (b) the drafter's gathers on link 1, the target's on link 0: needs the link-offset graft (ag_link_offset.patch: the all-gather
            factory adds QWEN_AG_LINK_OFFSET_SD1 to the fabric link of every sub-device-1 program). There is NO Python-visible way to pick a
            link in this tt-metal (num_links counts links from 0; the factory passes its loop index to the fabric), so without the graft this arm
            reads NOT-RUN. The decisive arm: it runs right after the solo and link-cost arms, before anything else that could hang.
  chained   target trace, an event, then the drafter trace on queue 1: the serial control (cannot hang)
  one_queue BOTH traces replayed from queue 0 (the drafter's on link 1, the same program as separate), one synchronise on both sub-devices: does the overlap
            need the second command queue? Needs --link-offset 1 and the separate arm (it is only meaningful where separate is proven), runs after it.
            Informational: its bytes are recorded but do not decide the verdict.
  shared    BOTH traces concurrently, both on fabric link 0 (num_links 1 on each): (a) sharing a link (hangs, see below)
  shared2   the same with num_links 2 on each (information only; production gathers use both links; hangs the same way)

MEASURED 2026-10-10 (the first card runs): the mesh opens, the manager loads, eager gathers on both sub-devices (separate semaphores, two queues) run
concurrently and are exact, the four traces capture and replay solo exactly, and the FIRST CONCURRENT REPLAY OF TWO GATHER TRACES ON ONE LINK HANGS. The
fabric does not arbitrate two clients of one router sender channel (worker side: edm_fabric_worker_adapters.hpp open_start / open_finish write the client's
identity into the channel's single location-info block and set the handshake word, with no check that the channel is free; the router has exactly one
local-worker sender channel per link, sender channel 0, fabric.cpp), so the arms that share a link are LAST and may be run alone (probe subdev-shared).

MEASURED 2026-10-11 (the link-offset run): t_solo 0.630 ms (one link), t_solo2 0.438 ms (two links; link_cost 1.44), d_solo 0.319 ms, chained 0.939 ms, and
then the arm that was called one_queue HUNG. It was a shared-link arm: its drafter trace was the link-0 program (captured on queue 0 without the offset), and
the replay of a trace waits only for the sub-devices that trace itself uses (trace/dispatch.cpp issue_trace_commands: one wait per sub-device in the trace's
descriptors, on that sub-device's own stream), so a sub-device-1 trace replayed after a sub-device-0 trace on the SAME queue starts at once: two clients of
router 0's sender channel 0 whatever the queue count. one_queue is therefore now built from the link-1 program and ordered after separate.

PASS = shared-link concurrent wall <= 1.1 x max(solo), every compared tensor identical, nothing hung. PASS-SEPARATE-LINKS = only the separate
links overlap (the graft is required). SAFE-PASS = the arms that cannot hang (solo, chained by an event) are exact and timed; no overlap is judged.
The run holds all four cards: put it LAST in a window and follow it with an all-board reset.

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
# Result keys, in the order the arms RUN. The solo and link-cost arms cannot hang. The decisive arm (separate) comes next, before anything else that could
# hang: its drafter gathers run on another link than the target's, and a hang in it is itself the finding. chained (an event between the queues, serial) cannot
# hang and follows it. Then the arms that put a SECOND client on a router: one_queue (link 1, needs the graft and a proven separate) and, last, the two that
# share a link, which deadlock the mesh (the fabric does not arbitrate two clients of one router sender channel, see the docstring). A hang ends the process:
# nothing that could hang runs before the numbers that matter more.
ARMS = ('t_solo', 'd_solo', 't_solo2', 'separate', 'chained', 'one_queue', 'shared', 'shared2')
ARM_GROUPS = dict(solo=('t_solo', 'd_solo'), link_cost=('t_solo2',), separate=('separate',), chained=('chained',), one_queue=('one_queue',),
                  shared=('shared',), shared2=('shared2',))
GROUP_ORDER = ('solo', 'link_cost', 'separate', 'chained', 'one_queue', 'shared', 'shared2')
SAFE_GROUPS = ('solo', 'link_cost', 'chained')           # cannot hang whatever the fabric does with two clients
HAZARD_GROUPS = ('separate', 'one_queue', 'shared', 'shared2')
DEFAULT_ARMS = 'solo,chained'      # link_cost is the separate-links job's


def parse_groups(text):
    """The arm groups named by --arms, in the order they will run (GROUP_ORDER: the decisive separate arm before every other arm that could hang)."""
    names = [name.strip() for name in str(text).split(',') if name.strip()]
    unknown = [name for name in names if name not in ARM_GROUPS]
    if unknown or not names:
        raise ValueError('--arms names %s; known: %s' % (unknown or 'nothing', ', '.join(ARM_GROUPS)))
    return [group for group in GROUP_ORDER if group in names]


def verdict_line(text, evidence, extra=None):
    def number(key):
        value = evidence.get(key)
        return '-' if value is None else value

    parts = ['%s verdict=%s' % (TAG, text), 'shared_ratio=%s' % number('shared_ratio'), 'separate_ratio=%s' % number('separate_ratio'),
             'shared2_ratio=%s' % number('shared2_ratio'), 't_ms=%s' % number('t_solo_ms'), 'd_ms=%s' % number('d_solo_ms'),
             'shared_ms=%s' % number('shared_ms'), 'separate_ms=%s' % number('separate_ms'), 'chained_ms=%s' % number('chained_ms'),
             'one_queue_ms=%s' % number('one_queue_ms'), 'one_queue_ratio=%s' % number('one_queue_ratio'),
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
        self.completed = []
        self.d_sep = None
        self.manager_id = None
        self.topology = getattr(ttnn.Topology, options.topology)
        self.groups = parse_groups(options.arms)

    def span(self, label, seconds=None):
        return self.wd.span(label, self.o.watchdog_s if seconds is None else seconds)

    def compile_span(self, label):
        return self.wd.span(label, self.o.compile_watchdog_s)

    def stage(self, name):
        self.report['last_stage'] = name
        if self.heartbeat is not None:
            self.heartbeat.set(name)
        self.snapshot()
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

    def warm(self, label, side, queue, links):
        """One eager gather of exactly the program a trace is about to capture: a capture of a program that never ran is a TT_FATAL (its binaries are
        not on the device yet). The programs of the one-link traces are the eager arm's own; the two-link and separate-link ones are new."""
        with self.compile_span('warm-' + label):
            self.run_gathers(side, queue, links, self.pool(side, 'warm'), 1)
            self.sync(queue, [side.sd])

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

    def poison(self, side, outputs):
        """Overwrite the checked gathers with NaN before a replay (see subdev_h1.Harness.poison): stale outputs cannot pass."""
        ttnn, torch = self.ttnn, self.torch
        shape = (1, 1, side.rows, CHIPS * side.shard)
        host = ttnn.from_torch(torch.full(shape, torch.nan, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))
        with self.span('poison'):
            for pick in sorted({0, len(outputs) // 2, len(outputs) - 1}):
                ttnn.copy_host_to_device_tensor(host, outputs[pick])
            self.sync_all()

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
        if 'shared' not in self.groups:
            return
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

        def both(first, second, flip):
            if flip:
                self.replay(second[0], second[1])
                self.replay(first[0], first[1])
            else:
                self.replay(first[0], first[1])
                self.replay(second[0], second[1])

        def concurrent(trace_t, trace_d):
            def run(flip):
                both((trace_t, 0), (trace_d, 1), flip)
                self.sync(0, [sd0])
                self.sync(1, [sd1])
            return run

        def chained(flip):
            self.replay('t', 0)
            event = self.ttnn.record_event(self.mesh, 0, [sd0])
            self.ttnn.wait_for_event(1, event)
            self.replay('d', 1)
            self.sync(0, [sd0])
            self.sync(1, [sd1])

        def one_queue(flip):
            # both traces from queue 0: the drafter's is the link-1 program (ds0), NOT the link-0 one: a trace replay waits only for its own sub-devices,
            # so the two start together and the same link would be the shared arm
            both(('t', 0), ('ds0', 0), flip)
            self.sync(0, [sd0, sd1])

        fns = dict(t_solo=lambda flip: (self.replay('t', 0), self.sync(0, [sd0])),
                   d_solo=lambda flip: (self.replay('d', 1), self.sync(1, [sd1])),
                   chained=chained, shared=concurrent('t', 'd'))
        if 't2' in self.traces:
            fns['t_solo2'] = lambda flip: (self.replay('t2', 0), self.sync(0, [sd0]))
        if 'd2' in self.traces:
            fns['shared2'] = concurrent('t2', 'd2')
        if 'ds' in self.traces:
            fns['separate'] = concurrent('t', 'ds')
        if 'ds0' in self.traces:
            fns['one_queue'] = one_queue
        return {name: fn for name, fn in fns.items() if name in self.selected_arms()}

    def selected_arms(self):
        return [arm for group in self.groups for arm in ARM_GROUPS[group]]

    def snapshot(self):
        """The partial results into the report, so a hang (which ends the process) leaves every number measured before it."""
        self.report['arms'] = {name: plan.summarize(samples) for name, samples in self.arm_ns.items() if samples}
        self.report['exactness'] = self.exact

    def announce(self, name):
        if name in ('shared', 'shared2'):
            self.log('%s next arm=%s hazard="two clients of one fabric router sender channel; a hang here is the finding"' % (TAG, name))
        elif name in ('separate', 'one_queue'):
            self.log('%s next arm=%s hazard="the drafter on link %d, the target on link 0: a hang here means the offset did not separate the routers"'
                     % (TAG, name, self.o.link_offset))
        self.stage('arm-' + name)

    def run_arm(self, fns, name, flip=False, record=False):
        with self.span('arm-' + name):
            started = self.clock()
            fns[name](flip)
            elapsed = self.clock() - started
        self.sync_all()
        if record:
            self.arm_ns[name].append(elapsed)

    def arm_outputs(self, name):
        """(target trace key, drafter trace key, drafter side) of an arm's compared outputs."""
        d = self.d_side
        return {'t_solo': ('t', None, None), 't_solo2': ('t2', None, None), 'd_solo': (None, 'd', d), 'chained': ('t', 'd', d),
                'shared': ('t', 'd', d), 'shared2': ('t2', 'd2', d), 'separate': ('t', 'ds', self.d_sep or d), 'one_queue': ('t', 'ds0', self.d_sep or d)}[name]

    def measure(self, fns):
        """Arm by arm, in the safety order: the bytes (poisoned outputs, two replays), then the timed rounds, then the snapshot. A hang in a later arm cannot
        take an earlier arm's numbers with it, and the arms that could hang run after every arm that cannot."""
        t = self.t_side
        for name in [arm for arm in ARMS if arm in fns]:
            self.announce(name)
            tt, dd, d_side = self.arm_outputs(name)
            for repeat in range(1 if name in ('t_solo', 'd_solo') else 2):
                if tt:
                    self.poison(t, self.out[tt])
                if dd:
                    self.poison(d_side, self.out[dd])
                self.run_arm(fns, name, flip=bool(repeat))
                # one_queue asks about the queue count only: its bytes are recorded (one_queue_bytes) but do not decide the verdict
                informational = name == 'one_queue'
                if tt:
                    self.check('%s #%d target' % (name, repeat), t, self.out[tt], informational=informational)
                if dd:
                    self.check('%s #%d drafter' % (name, repeat), d_side, self.out[dd], informational=informational)
            if not self.o.no_timing:
                for index in range(self.o.warmup + self.o.rounds):
                    self.run_arm(fns, name, flip=bool((index // 2) % 2), record=index >= self.o.warmup)
            self.completed.append(name)
            self.snapshot()
        self.stage('measure-done')

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
        t, d = self.t_side, self.d_side
        offset = self.o.link_offset
        report['link_offset'] = offset
        report['arms_requested'] = list(self.groups)
        # EVERY device allocation and every first run of a program comes BEFORE the first capture: the runtime warns that a buffer allocated while a trace is
        # active may be corrupted when the trace executes (its temporaries are freed and reusable), and a capture of a program that never ran is a TT_FATAL.
        self.stage('warm')
        if 'shared2' in self.groups or 'link_cost' in self.groups:
            self.warm('t2', t, 0, 2)
        if 'shared2' in self.groups:
            self.warm('d2', d, 1, 2)
        on_link = ('separate' in self.groups or 'one_queue' in self.groups) and offset > 0
        if on_link:
            # (b): the drafter's gathers on link `offset`. Only the graft reads the variable, and it reads it when a program is BUILT, so it is set for
            # the warm-up that builds it (and for the captures, which are program-cache hits that must not rebuild on link 0) and removed after; the
            # drafter shape is one the shared arm does not use (rows + 32), so the program is new (the sub-device is in the program hash, the environment
            # is not). The one-queue arm captures the SAME program on queue 0 (ds0).
            self.d_sep = self.side('drafter-separate', self.d_side.rect, self.sd[1], SHAPES['drafter'][0] + 32, SHAPES['drafter'][1])
            self.environ[LINK_ENV] = str(offset)
            try:
                self.warm('ds', self.d_sep, 1, 1)
            finally:
                self.environ.pop(LINK_ENV, None)
        else:
            report['separate'] = 'NOT-RUN: the arm needs --link-offset 1 and the link-offset graft (ag_link_offset.patch)'
        self.sync_all()
        self.stage('capture')
        self.out = {}
        self.out['t'] = self.capture('t', t, 0, 1, self.pool(t, 'trace1'))
        self.out['d'] = self.capture('d', d, 1, 1, self.pool(d, 'trace1'))
        if 'shared2' in self.groups or 'link_cost' in self.groups:
            self.out['t2'] = self.capture('t2', t, 0, 2, self.pool(t, 'trace2'))
        if 'shared2' in self.groups:
            self.out['d2'] = self.capture('d2', d, 1, 2, self.pool(d, 'trace2'))
        if self.d_sep is not None:
            self.environ[LINK_ENV] = str(offset)
            try:
                if 'separate' in self.groups:
                    self.out['ds'] = self.capture('ds', self.d_sep, 1, 1, self.pool(self.d_sep, 'trace1'))
                if 'one_queue' in self.groups:
                    # last of the captures, and an error here drops the informational arm, never the decisive one
                    try:
                        self.out['ds0'] = self.capture('ds0', self.d_sep, 0, 1, self.pool(self.d_sep, 'trace0'))
                    except Exception as caught:  # noqa: BLE001
                        report['one_queue'] = 'NOT-RUN: the capture failed: %s' % ' '.join(str(caught).split())[:300]
                        self.log('%s one_queue=NOT-RUN capture failed: %s' % (TAG, report['one_queue']))
            finally:
                self.environ.pop(LINK_ENV, None)
        report['traces'] = sorted(self.traces)
        fns = self.arm_fns()
        self.stage('measure')
        self.measure(fns)

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


def one_queue_bytes(exact):
    """'identical' or 'DIFFER(m/n)' over the one_queue cases (informational: they are not in the verdict's totals)."""
    cases = [case for case in exact['cases'] if case['case'].startswith('one_queue')]
    compared = sum(case['compared'] for case in cases)
    mismatched = sum(case['mismatched'] for case in cases)
    return 'identical' if compared and not mismatched else 'DIFFER(%d/%d)' % (mismatched, compared)


def partial_findings(harness, options):
    """What a hang leaves behind, from the arms that COMPLETED before it: (line, dict) or (None, None). The decisive separate arm may well have finished when a
    later arm (one_queue, shared) hangs; the watchdog's own line says only HANG, which the port rule would read as NO-GO."""
    arms = {name: plan.summarize(harness.arm_ns[name]) for name in harness.completed if harness.arm_ns.get(name)}
    if not arms.get('separate', {}).get('n') or harness.exact['mismatched']:
        return None, None
    exactness = dict(compared=harness.exact['compared'], mismatched=harness.exact['mismatched'])
    text, evidence = plan.verdict_h2(arms, exactness, timing=not options.no_timing, error=None, pass_ratio=options.pass_ratio, separate_run=True)
    decision, reason = plan.port_rule(evidence, text)
    found = dict(arms=sorted(arms), verdict=text, port=decision, reason=reason, separate_ratio=evidence.get('separate_ratio'),
                 link_cost=evidence.get('link_cost'), completed=list(harness.completed))
    line = '%s before_hang separate=%s separate_ratio=%s link_cost=%s port=%s completed=%s' % (
        TAG, text, evidence.get('separate_ratio', '-'), evidence.get('link_cost', '-'), decision, '+'.join(harness.completed))
    return line, found


def finish(harness, report, options, error):
    """(verdict text, evidence, printed line) from what the harness measured."""
    arms = {name: plan.summarize(samples) for name, samples in harness.arm_ns.items() if samples} if harness is not None else {}
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
    report['arms_requested'] = report.get('arms_requested') or list(parse_groups(options.arms))
    extra = dict(links=1, topology=options.topology, link_offset=report.get('link_offset', 0),
                 separate='RUN' if separate_run else 'NOT-RUN', arms='+'.join(report['arms_requested']))
    if error:
        extra['error'] = '"%s"' % ' '.join(str(error).split())[:200].replace('"', "'")
    if report.get('known_failure'):
        extra['known_failure'] = '"%s"' % ' '.join(str(report['known_failure']).split())[:200].replace('"', "'")
    found = partial_findings(harness, options)[1] if error and harness is not None else None
    if found:
        # an error AFTER the separate arm completed (one_queue is informational): the rule is applied to what the separate arm showed, and says so
        report['before_error'] = found
        extra['port'] = found['port']
        extra['port_basis'] = 'separate-only-after-error'
        report['port_rule'] = dict(decision=found['port'], reason=found['reason'], basis='separate-only-after-error', link_cost_go=plan.LINK_COST_GO,
                                   link_cost_max=plan.LINK_COST_MAX)
    elif separate_run:
        decision, reason = plan.port_rule(evidence, text)
        extra['port'] = decision
        report['port_rule'] = dict(decision=decision, reason=reason, link_cost_go=plan.LINK_COST_GO, link_cost_max=plan.LINK_COST_MAX)
    if evidence.get('link_cost') is not None:
        extra['link_cost'] = evidence['link_cost']
    if harness is not None and 'one_queue' in harness.completed:
        extra['one_queue_bytes'] = one_queue_bytes(harness.exact)
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
    parser.add_argument('--arms', default=DEFAULT_ARMS,
                        help='arm groups to run, comma separated, from %s. The default is the arms that cannot hang; shared and shared2 put two gather '
                             'streams on one fabric link and may deadlock the mesh (separate needs --link-offset 1)' % ', '.join(ARM_GROUPS))
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
    try:
        groups = parse_groups(options.arms)
        if 'separate' in groups and options.link_offset == 0:
            problems.append('--arms separate needs --link-offset 1 (and the graft)')
        if 'one_queue' in groups and options.link_offset == 0:
            problems.append('--arms one_queue needs --link-offset 1 (and the graft): on one link both gathers would share a router sender channel, which is the shared arm')
        if 'one_queue' in groups and 'separate' not in groups:
            problems.append('--arms one_queue needs separate in the same run: it is only meaningful where the separate links are proven, and it runs after them')
        if options.link_offset and 'separate' not in groups:
            problems.append('--link-offset 1 runs the separate arm: add it to --arms')
    except ValueError as error:
        problems.append(str(error))
    if options.gathers < 2:
        problems.append('--gathers must be at least 2')
    if options.rounds < 3 or options.warmup < 0 or options.eager_rounds < 1:
        problems.append('--rounds >= 3, --warmup >= 0, --eager-rounds >= 1')
    if options.pass_ratio <= 1.0:
        problems.append('--pass-ratio must exceed 1.0')
    if options.target_cores < 1 or options.drafter_cores < 1:
        problems.append('both sub-devices need at least one core')
    return problems
