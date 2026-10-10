"""H1: can a drafter-like trace run CONCURRENTLY with a target-like trace on ONE Blackhole chip, through tt-metal sub-devices and two command queues?

One card, ttnn Python only, no model weights, no collectives. The harness

  1. opens a mesh of the devices the container sees (one) with TWO command queues and prints the compute grid;
  2. creates ONE sub-device manager with two disjoint sub-devices, an 80-core TARGET sub-device and a 30-core DRAFTER sub-device (whole columns
     of the reported grid; subdev_plan.plan_split), and times manager create, load and clear (several cycles);
  3. runs a ~400-op target-like recipe (matmuls, silu, products, residual adds shaped like a 64-row verify slice, restricted to the target
     sub-device: matmul core_grid + sub_device_id, eltwise sub_core_grids) and a ~900-op drafter-like recipe (smaller, launch-bound,
     restricted to the drafter sub-device) eagerly for reference, then captures THREE traces: target on queue 0, drafter on queue 1, and the
     drafter again on queue 0 (the one-queue control; a trace replays only on the queue it was captured on);
  4. times, in interleaved rounds, FIVE arms: the target alone, the drafter alone, BOTH CONCURRENTLY (queue 0 and queue 1, non-blocking
     replays, then a per-sub-device synchronise of each queue), CHAINED by an event (target, record_event on queue 0, queue 1 waits, drafter),
     and BOTH ON ONE QUEUE (the control: a single queue dispatching the two traces back to back);
  5. compares the outputs byte for byte against the solo runs: every `--check-every`-th op output and the last, after each arm, after a stress of
     repeated concurrent replays and after the timed loop; also the drafter on queue 0 against the drafter on queue 1, and (information only) the
     traces against the eager runs.

PASS = concurrent wall <= 1.1 x max(solo walls), every compared tensor identical, nothing hung, no error. Every device call is under a
per-call watchdog (subdev_plan.Watchdog: exit 3 on a hang, with a faulthandler backstop for a call that holds the GIL); the watcher pass
(`--no-timing`, TT_METAL_WATCHER set by the harness script) judges bytes and hangs only because the watcher distorts every wall.

The last stdout lines: one 'SUBDEV_H1 verdict=...' line with the numbers, then one JSON object (kind subdev-h1).
Exit: 0 PASS or UNTIMED-PASS, 1 any FAIL / INCOMPLETE, 2 NOT-MEASURED (the device did not open, an API is missing, a TT_FATAL: the error is in the
line), 3 the watchdog (a hang: reset THE TARGET CARD only).

Run (inside the serving image, one card, ttnn from the image): python3 -B subdev_h1.py --out results/subdev-h1-<stamp>.json
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

TILE = 32


class Side(object):
    """One sub-device's share of the experiment: its core rectangle, id, matmul grid, recipe, inputs and queue."""

    def __init__(self, name, rect, grid, sd, crs, recipe, rows):
        self.name, self.rect, self.grid, self.sd, self.crs, self.recipe, self.rows = name, rect, grid, sd, crs, recipe, rows
        self.inputs = {}
        self.core_grid = None


def one_line(text, limit=300):
    return ' '.join(str(text).split())[:limit]


class Harness(object):
    """The experiment. `ttnn` and `torch` are the modules (fakes in the tests); `clock` returns nanoseconds."""

    def __init__(self, ttnn, torch, mesh, options, watchdog, log=plan.say, clock=time.perf_counter_ns, report=None, persist=None):
        self.ttnn, self.torch, self.mesh, self.o, self.wd, self.log, self.clock = ttnn, torch, mesh, options, watchdog, log, clock
        self.report = report if report is not None else {}
        self.persist = persist or (lambda: None)
        self.keep = []                       # every tensor the run allocates: released before the manager is cleared and the mesh closed
        self.traces = {}
        self.sd = [ttnn.SubDeviceId(0), ttnn.SubDeviceId(1)]
        self.manager_id = None
        self.exact = dict(compared=0, mismatched=0, cases=[])
        self.arm_ns = {name: [] for name in plan.ARMS}
        self.ckc = None

    # ------------------------------------------------------------------ plumbing

    def span(self, label, seconds=None):
        return self.wd.span(label, self.o.watchdog_s if seconds is None else seconds)

    def compile_span(self, label):
        return self.wd.span(label, self.o.compile_watchdog_s)

    def stage(self, name):
        self.report['last_stage'] = name
        self.persist()
        self.log('%s stage=%s' % (plan.VERDICT_TAG, name))

    def crs(self, rect):
        ttnn = self.ttnn
        return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(rect[0], rect[1]), ttnn.CoreCoord(rect[2], rect[3]))})

    def sync_all(self):
        self.ttnn.synchronize_device(self.mesh)

    def sync(self, queue, sub_devices):
        self.ttnn.synchronize_device(self.mesh, cq_id=queue, sub_device_ids=sub_devices)

    def replay(self, name, queue):
        self.ttnn.execute_trace(self.mesh, self.traces[name], cq_id=queue, blocking=False)

    # ------------------------------------------------------------------ the sub-device manager

    def build_manager(self, split):
        ttnn, mesh = self.ttnn, self.mesh
        self.target_crs, self.drafter_crs = self.crs(split['target']), self.crs(split['drafter'])
        with self.span('manager-create'):
            started = self.clock()
            self.manager_id = mesh.create_sub_device_manager(
                [ttnn.SubDevice([self.target_crs]), ttnn.SubDevice([self.drafter_crs])], 0)
            create_ms = (self.clock() - started) / 1e6
        manager = dict(create_ms=round(create_ms, 3), load_ms=[], clear_ms=[], load_sync_ms=[], clear_sync_ms=[])
        self.report['manager'] = manager
        for _ in range(self.o.manager_cycles):
            self.timed_manager('load', manager)
            self.timed_manager('clear', manager)
        self.timed_manager('load', manager, keep=True)
        for key in ('load_ms', 'clear_ms'):
            if manager[key]:
                manager[key + '_median'] = round(sorted(manager[key])[len(manager[key]) // 2], 3)
        self.persist()

    def timed_manager(self, which, manager, keep=False):
        mesh = self.mesh
        with self.span('manager-' + which):
            started = self.clock()
            if which == 'load':
                mesh.load_sub_device_manager(self.manager_id)
            else:
                mesh.clear_loaded_sub_device_manager()
            host_ms = (self.clock() - started) / 1e6
            self.sync_all()
            total_ms = (self.clock() - started) / 1e6
        manager[which + '_ms'].append(round(host_ms, 3))
        manager[which + '_sync_ms'].append(round(total_ms, 3))
        if keep:
            self.log('%s manager loaded: create_ms=%s load_ms=%s clear_ms=%s' % (
                plan.VERDICT_TAG, manager['create_ms'], manager['load_ms'], manager['clear_ms']))

    # ------------------------------------------------------------------ tensors and recipes

    def upload(self, values, dtype):
        ttnn = self.ttnn
        tensor = ttnn.from_torch(values, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                 mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))
        self.keep.append(tensor)
        return tensor

    def make_side(self, name, rect, grid, sd, crs, profile, ops, rows, seed):
        torch, ttnn = self.torch, self.ttnn
        recipe = plan.build_recipe(profile, ops)
        side = Side(name, rect, grid, sd, crs, recipe, rows)
        side.core_grid = ttnn.CoreGrid(x=grid[0], y=grid[1])
        generator = torch.Generator().manual_seed(seed)
        hidden = plan.PROFILES[profile]['hidden']
        # Gains that keep 40 to 90 layer pairs of residual silu products finite in bfloat16 (subdev_plan.OUT_GAIN says why); the same seed gives the
        # same bytes every run.
        for weight, (k, n) in plan.weight_shapes(profile).items():
            gain = plan.OUT_GAIN if weight in ('w_down', 'w_o') else plan.WEIGHT_GAIN
            values = torch.randn(1, 1, k, n, generator=generator) * (gain / (k ** 0.5))
            side.inputs[weight] = self.upload(values, ttnn.bfloat8_b)
        side.inputs['h0'] = self.upload(torch.randn(1, 1, rows, hidden, generator=generator).to(torch.bfloat16), ttnn.bfloat16)
        side.inputs['scale'] = self.upload(((torch.rand(1, 1, rows, hidden, generator=generator) - 0.5) * (2 * plan.SCALE_AMPLITUDE)).to(torch.bfloat16),
                                           ttnn.bfloat16)
        return side

    def lower(self, side, kind, first, second):
        ttnn = self.ttnn
        dram = ttnn.DRAM_MEMORY_CONFIG
        if kind == 'mm':
            return ttnn.matmul(first, second, core_grid=side.core_grid, sub_device_id=side.sd, memory_config=dram, dtype=ttnn.bfloat16,
                               compute_kernel_config=self.ckc)
        if kind == 'silu':
            return ttnn.silu(first, memory_config=dram, sub_core_grids=side.crs)
        if kind == 'mul':
            return ttnn.multiply(first, second, memory_config=dram, sub_core_grids=side.crs)
        if kind == 'add':
            return ttnn.add(first, second, memory_config=dram, sub_core_grids=side.crs)
        raise plan.PlanError('unknown op kind %r' % (kind,))

    def run_recipe(self, side, queue):
        """Enqueue the whole recipe on `queue`; returns {op output name: tensor}. Nothing is freed: a trace keeps every buffer it names."""
        env, outputs = dict(side.inputs), {}
        with self.ttnn.command_queue(queue):
            for kind, dest, first, second in side.recipe['ops']:
                tensor = self.lower(side, kind, env[first], env[second] if second else None)
                env[dest] = outputs[dest] = tensor
        self.keep.extend(outputs.values())
        return outputs

    def eager(self, side, queue):
        with self.compile_span('eager-' + side.name):
            outputs = self.run_recipe(side, queue)
            self.sync(queue, [side.sd])
        return outputs

    def capture(self, label, side, queue):
        """Capture the recipe on `queue` as a trace named `label`; returns the output tensors the trace writes."""
        ttnn = self.ttnn
        with self.compile_span('capture-' + label):
            trace_id = ttnn.begin_trace_capture(self.mesh, cq_id=queue)
            try:
                outputs = self.run_recipe(side, queue)
            finally:
                ttnn.end_trace_capture(self.mesh, trace_id, cq_id=queue)
            self.sync_all()
        self.traces[label] = trace_id
        return outputs

    # ------------------------------------------------------------------ byte comparison

    def read(self, outputs, names):
        ttnn, torch = self.ttnn, self.torch
        got = {}
        with self.span('readback'):
            for name in names:
                part = ttnn.get_device_tensors(outputs[name])[0]
                host = ttnn.to_torch(part)
                got[name] = host.contiguous().view(torch.int16).clone()
        return got

    def compare(self, case, reference, got, informational=False):
        """Bitwise comparison of two {name: int16 view} dicts; recorded under `case`. Information-only cases never enter the verdict."""
        torch = self.torch
        bad = [name for name in reference if name not in got or not torch.equal(reference[name], got[name])]
        entry = dict(case=case, compared=len(reference), mismatched=len(bad), first_bad=bad[:5])
        if informational:
            entry['informational'] = True
        else:
            self.exact['compared'] += len(reference)
            self.exact['mismatched'] += len(bad)
        self.exact['cases'].append(entry)
        if bad:
            self.log('%s bytes DIFFER case=%s mismatched=%d/%d first=%s' % (plan.VERDICT_TAG, case, len(bad), len(reference), bad[:3]))
        return not bad

    def poison(self, outputs, names):
        """Overwrite the compared tensors with NaN before a replay, so that a replay that did not run (or did not write) leaves NaN behind and fails the
        comparison instead of passing on the previous run's bytes."""
        ttnn, torch = self.ttnn, self.torch
        cache = {}
        with self.span('poison'):
            for name in names:
                tensor = outputs[name]
                shape = tuple(int(dim) for dim in tensor.shape)
                if shape not in cache:
                    cache[shape] = ttnn.from_torch(torch.full(shape, torch.nan, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                                   mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))
                ttnn.copy_host_to_device_tensor(cache[shape], tensor)
            self.sync_all()

    def poison_for(self, *arms):
        """Poison the outputs the named arms' traces write: 't' (the target's), 'd1' (the drafter's on queue 1), 'd0' (the drafter's on queue 0)."""
        wanted = {'t': (self.t_out, self.names_t), 'd1': (self.d1_out, self.names_d), 'd0': (self.d0_out, self.names_d)}
        for arm in arms:
            self.poison(*wanted[arm])

    def finite(self, tensors):
        torch = self.torch
        return all(bool(torch.isfinite(value.view(torch.bfloat16).float()).all()) for value in tensors.values())

    def live(self, tensors):
        """Whether any compared tensor is not all zeros: a recipe that silently produced zeros would make every byte comparison vacuous."""
        torch = self.torch
        return any(bool(value.view(torch.bfloat16).float().abs().sum() > 0) for value in tensors.values())

    # ------------------------------------------------------------------ the arms

    def arm_t_solo(self, flip=False):
        self.replay('t', 0)
        self.sync(0, [self.sd[0]])

    def arm_d_solo(self, flip=False):
        self.replay('d1', 1)
        self.sync(1, [self.sd[1]])

    def arm_concurrent(self, flip=False):
        if flip:
            self.replay('d1', 1)
            self.replay('t', 0)
        else:
            self.replay('t', 0)
            self.replay('d1', 1)
        if flip:
            self.sync(0, [self.sd[0]])
            self.sync(1, [self.sd[1]])
        else:
            self.sync(1, [self.sd[1]])
            self.sync(0, [self.sd[0]])

    def arm_chained(self, flip=False):
        self.replay('t', 0)
        event = self.ttnn.record_event(self.mesh, 0, [self.sd[0]])
        self.ttnn.wait_for_event(1, event)
        self.replay('d1', 1)
        self.sync(0, [self.sd[0]])
        self.sync(1, [self.sd[1]])

    def arm_one_queue(self, flip=False):
        if flip:
            self.replay('d0', 0)
            self.replay('t', 0)
        else:
            self.replay('t', 0)
            self.replay('d0', 0)
        self.sync(0, list(self.sd))

    def arm(self, name):
        return getattr(self, 'arm_' + name)

    def run_arm(self, name, flip=False, record=False):
        fn = self.arm(name)
        with self.span('arm-' + name):
            started = self.clock()
            fn(flip)
            elapsed = self.clock() - started
        self.sync_all()            # outside the timed span: clears every queue's ownership before the next arm
        if record:
            self.arm_ns[name].append(elapsed)
        return elapsed

    # ------------------------------------------------------------------ the experiment

    def exactness(self):
        """Solo references, then every arm replayed untimed with a byte comparison of both traces' outputs (and the one-queue drafter's)."""
        names_t = plan.checkpoint_names(self.t_side.recipe, self.o.check_every)
        names_d = plan.checkpoint_names(self.d_side.recipe, self.o.check_every)
        self.names_t, self.names_d = names_t, names_d
        self.poison_for('t')
        self.run_arm('t_solo')
        ref_t = self.read(self.t_out, names_t)
        self.poison_for('d1')
        self.run_arm('d_solo')
        ref_d = self.read(self.d1_out, names_d)
        self.ref_t, self.ref_d = ref_t, ref_d
        self.report['finite'] = dict(target=self.finite(ref_t), drafter=self.finite(ref_d))
        self.report['live'] = dict(target=self.live(ref_t), drafter=self.live(ref_d))
        if not all(self.report['live'].values()):
            raise RuntimeError('a trace wrote only zeros (%s): the byte comparisons would be vacuous' % self.report['live'])
        if not all(self.report['finite'].values()):
            raise RuntimeError('a solo reference holds NaN or inf (%s): the trace did not write its outputs (they still hold the poison), or the recipe '
                               'overflows bfloat16; either way the byte comparisons would be vacuous' % self.report['finite'])
        self.compare('target trace vs eager', self.eager_ref_t, ref_t, informational=True)
        self.compare('drafter trace vs eager', self.eager_ref_d, ref_d, informational=True)
        # the drafter captured on queue 0 against the drafter captured on queue 1: same program, other dispatcher
        self.poison_for('d0')
        self.replay('d0', 0)
        self.sync(0, [self.sd[1]])
        self.sync_all()
        self.compare('drafter queue 0 vs queue 1', ref_d, self.read(self.d0_out, names_d))
        for repeat in range(2):
            for name in ('concurrent', 'chained', 'one_queue'):
                self.poison_for('t', 'd0' if name == 'one_queue' else 'd1')
                self.run_arm(name, flip=bool(repeat))
                self.compare('%s #%d target' % (name, repeat), ref_t, self.read(self.t_out, names_t))
                if name == 'one_queue':
                    self.compare('%s #%d drafter(q0)' % (name, repeat), ref_d, self.read(self.d0_out, names_d))
                else:
                    self.compare('%s #%d drafter' % (name, repeat), ref_d, self.read(self.d1_out, names_d))
        self.stage('exactness-done')

    def stress(self):
        self.poison_for('t', 'd1')
        for index in range(self.o.stress):
            self.run_arm('concurrent', flip=bool(index % 2))
        if self.o.stress:
            self.compare('stress x%d target' % self.o.stress, self.ref_t, self.read(self.t_out, self.names_t))
            self.compare('stress x%d drafter' % self.o.stress, self.ref_d, self.read(self.d1_out, self.names_d))
        self.stage('stress-done')

    def timing(self):
        order = list(plan.ARMS)
        rounds = self.o.warmup + self.o.rounds
        for index in range(rounds):
            shift = index % len(order)
            arms = order[shift:] + order[:shift]
            if (index // len(order)) % 2:
                arms = list(reversed(arms))
            for name in arms:
                self.run_arm(name, flip=bool((index // 2) % 2), record=index >= self.o.warmup)
        self.stage('timing-done')
        # the outputs after the timed replays: still the solo references' bytes, written by a replay that ran (the poison is rewritten)
        self.poison_for('t', 'd1')
        self.run_arm('concurrent')
        self.compare('after timing target', self.ref_t, self.read(self.t_out, self.names_t))
        self.compare('after timing drafter', self.ref_d, self.read(self.d1_out, self.names_d))

    def run(self):
        ttnn, report = self.ttnn, self.report
        grid = self.mesh.compute_with_storage_grid_size()
        gx, gy = int(grid.x), int(grid.y)
        report['grid'] = [gx, gy]
        self.log('%s grid=%dx%d cores=%d command_queues=2' % (plan.VERDICT_TAG, gx, gy, gx * gy))
        split = plan.plan_split(gx, gy, self.o.target_cores, self.o.drafter_cores)
        report['plan'] = split
        if not plan.rectangles_disjoint(split['target'], split['drafter']):
            raise plan.PlanError('the two sub-device rectangles intersect: %s %s' % (split['target'], split['drafter']))
        self.log('%s split target=%s (%d cores) drafter=%s (%d cores) exact=%s' % (
            plan.VERDICT_TAG, split['target'], split['target_cores'], split['drafter'], split['drafter_cores'], split['exact']))
        self.ckc = ttnn.init_device_compute_kernel_config(self.mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
                                                          fp32_dest_acc_en=False, packer_l1_acc=True)
        self.stage('manager')
        self.build_manager(split)
        self.stage('tensors')
        with self.compile_span('upload'):
            self.t_side = self.make_side('target', split['target'], split['target_grid'], self.sd[0], self.target_crs, 'target',
                                         self.o.target_ops, self.o.rows, self.o.seed)
            self.d_side = self.make_side('drafter', split['drafter'], split['drafter_grid'], self.sd[1], self.drafter_crs, 'drafter',
                                         self.o.drafter_ops, self.o.drafter_rows, self.o.seed + 1)
            self.sync_all()
        report['recipes'] = dict(target=dict(ops=self.t_side.recipe['count'], matmuls=self.t_side.recipe['matmuls']),
                                 drafter=dict(ops=self.d_side.recipe['count'], matmuls=self.d_side.recipe['matmuls']))
        self.stage('eager')
        names_t = plan.checkpoint_names(self.t_side.recipe, self.o.check_every)
        names_d = plan.checkpoint_names(self.d_side.recipe, self.o.check_every)
        eager_t = self.eager(self.t_side, 0)
        self.eager_ref_t = self.read(eager_t, names_t)
        eager_d = self.eager(self.d_side, 1)
        self.eager_ref_d = self.read(eager_d, names_d)
        self.stage('capture')
        self.t_out = self.capture('t', self.t_side, 0)
        self.d1_out = self.capture('d1', self.d_side, 1)
        self.d0_out = self.capture('d0', self.d_side, 0)
        report['traces'] = sorted(self.traces)
        self.stage('exactness')
        self.exactness()
        self.stage('stress')
        self.stress()
        self.stage('timing')
        self.timing()

    def teardown(self):
        """Release the traces, then the tensors, then clear and remove the manager (traces and local allocations are per manager)."""
        mesh = self.mesh
        manager = self.report.get('manager', {})
        try:
            self.sync_all()
        except BaseException as error:  # noqa: BLE001
            self.report.setdefault('teardown_errors', []).append('sync: %s' % one_line(error))
        for name, trace_id in list(self.traces.items()):
            try:
                self.ttnn.release_trace(mesh, trace_id)
            except BaseException as error:  # noqa: BLE001
                self.report.setdefault('teardown_errors', []).append('release %s: %s' % (name, one_line(error)))
        self.traces.clear()
        del self.keep[:]
        self.t_out = self.d1_out = self.d0_out = self.eager_ref_t = self.eager_ref_d = None
        for side in (getattr(self, 't_side', None), getattr(self, 'd_side', None)):
            if side is not None:
                side.inputs.clear()
        if self.manager_id is not None:
            try:
                with self.span('manager-final-clear'):
                    started = self.clock()
                    mesh.clear_loaded_sub_device_manager()
                    manager['final_clear_ms'] = round((self.clock() - started) / 1e6, 3)
                    mesh.remove_sub_device_manager(self.manager_id)
            except BaseException as error:  # noqa: BLE001
                self.report.setdefault('teardown_errors', []).append('manager: %s' % one_line(error))


def dispatch_config(ttnn, choice):
    if choice == 'worker':
        return ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER)
    if choice == 'ethernet':
        return ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.ETH)
    return ttnn.DispatchCoreConfig()


def result_of(harness, report, options, error):
    """(verdict text, evidence, the printed line) from what the harness measured."""
    arms = {name: plan.summarize(samples) for name, samples in harness.arm_ns.items()} if harness is not None else {}
    report['arms'] = arms
    if harness is not None:
        report['raw_ms'] = {name: [round(value / 1e6, 4) for value in samples] for name, samples in harness.arm_ns.items()}
        report['exactness'] = harness.exact
        exactness = dict(compared=harness.exact['compared'], mismatched=harness.exact['mismatched'])
    else:
        exactness = dict(compared=0, mismatched=0)
    text, evidence = plan.verdict(arms, exactness, timing=not options.no_timing, error=error, pass_ratio=options.pass_ratio)
    evidence = plan.flatten_evidence(arms, evidence)
    extra = {}
    if error:
        extra['error'] = '"%s"' % one_line(error, 200).replace('"', "'")
    line = plan.verdict_line(text, evidence, plan=report.get('plan'), manager=report.get('manager'), extra=extra)
    report.update(verdict=text, evidence=evidence)
    return text, evidence, line


def main(argv=None, ttnn=None, torch=None, log=plan.say, exit_fn=None, clock=time.perf_counter_ns, watchdog=None):
    options, problems = plan.parse_args(argv)
    if problems:
        print('refusing: ' + '; '.join(problems), file=sys.stderr)
        return 2
    report = dict(kind=plan.KIND, options={key: value for key, value in sorted(vars(options).items())}, opened=False, last_stage='start')

    def persist():
        try:
            with open(options.out, 'w') as handle:
                json.dump(report, handle, indent=2, sort_keys=True, default=str)
        except OSError:
            pass

    def on_fire(label):
        report.update(verdict='HANG', error='watchdog: %s exceeded its budget' % label)
        persist()

    if watchdog is None:
        watchdog = plan.Watchdog(on_fire=on_fire, exit_fn=exit_fn).start()
    harness, mesh, error = None, None, None
    try:
        if ttnn is None:
            import torch  # noqa: F811
            import ttnn  # noqa: F811
        for needed in ('SubDevice', 'SubDeviceId', 'command_queue', 'begin_trace_capture', 'record_event', 'wait_for_event'):
            if not hasattr(ttnn, needed):
                raise RuntimeError('this ttnn has no %s' % needed)
        with watchdog.span('open-mesh', options.compile_watchdog_s):
            mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=options.trace_region_bytes,
                                         num_command_queues=2, dispatch_core_config=dispatch_config(ttnn, options.dispatch))
        report['opened'] = True
        mesh.enable_program_cache()      # a trace can only capture programs that already ran: they come from the program cache
        for needed in ('create_sub_device_manager', 'load_sub_device_manager', 'clear_loaded_sub_device_manager'):
            if not hasattr(mesh, needed):
                raise RuntimeError('this MeshDevice has no %s' % needed)
        harness = Harness(ttnn, torch, mesh, options, watchdog, log=log, clock=clock, report=report, persist=persist)
        harness.run()
    except BaseException as caught:  # noqa: BLE001
        error = '%s: %s' % (type(caught).__name__, one_line(caught, 500))
        report['error'] = error
        report['traceback'] = traceback.format_exc()[-3000:]
    finally:
        if harness is not None:
            harness.teardown()
        if mesh is not None:
            try:
                with watchdog.span('close-mesh', options.compile_watchdog_s):
                    ttnn.close_mesh_device(mesh)
                report['closed'] = True
            except BaseException as caught:  # noqa: BLE001
                report['closed'] = '%s: %s' % (type(caught).__name__, one_line(caught))
    text, evidence, line = result_of(harness, report, options, error)
    log(line)
    persist()
    log(json.dumps(report, sort_keys=True, default=str))
    return plan.exit_status(text)


if __name__ == '__main__':
    sys.exit(main())
