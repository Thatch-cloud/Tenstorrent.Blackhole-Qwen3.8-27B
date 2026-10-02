"""The A0 screen's driver, for the GPU host: runs every registered arm over every bundle turn off ONE target forward per prefix group,
resumably, under a memory guard and a deadline, and writes counts to a private results directory (the report reads it:
tf_pair_report.py). Everything here but `real_environment` runs on CPU against fakes in the tests; `real_environment` loads the models
and is verified only by the W0 canary (V0, V1, V3, V4 and the branch self-check).

    python3 scripts/ci/a0_run.py --bundle <bundle dir> --out <results dir> --deadline <epoch> --trip-file <file> [--canary] ...

THE ORDER IS THE PRIORITY. Groups run in a seeded, set-interleaved order, so any prefix of the run is a stratified sample and a stop
at the deadline still leaves a usable screen; within a turn the arms run in the registered order. A turn's failure is recorded (the
exception TYPE only) and the run goes on; an out-of-memory stops the run (it never skips ahead), and a turn whose projected peak
would break the memory floor is not started (counted as deferred, resumable later).

RESULTS (out/): meta.jsonl (copied from the bundle), arm-<name>.jsonl (k, status, compact rounds), v1.jsonl, timing.jsonl, plan.json,
state.json, calibration.json, validity.json. stdout carries counts only; an exception prints its type only.
"""
import argparse
import json
import os
import random
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a0_bundle_io as bundle  # noqa: E402
import tf_pair_report as report  # noqa: E402
import tf_pair_walk as walk  # noqa: E402

GIB = float(1 << 30)
WIDTH = 5 * 5120
TOKEN_BYTES = 65536 + 51200          # target KV (16 full-attention layers) + raw taps, per token
TRANSIENT_GIB = 4.0                  # prefill chunk transients
FLOOR_GIB = 10.0
V1_FLOOR = 0.95
V3_BAND = (0.93, 1.12)
V4_AGREE, V4_MARGIN, V4_ROUNDS = 0.99, 0.25, 0.97
SUBSETS = dict(windows={'swe': 80, 'own': 40}, forced={'swe': 66, 'own': 34}, depth={'swe': 40, 'own': 20})

# name, drafter, proposals, window, mode, subset (None = every turn)
ARM_SPECS = (
    ('dflash2-t16', 'dflash2', 15, 2048, 'free', None),
    ('dspark-t16', 'dspark', 15, None, 'free', None),
    ('dspark-t16-w2048', 'dspark', 15, 2048, 'free', 'windows'),
    ('dspark-t16-w8192', 'dspark', 15, 8192, 'free', 'windows'),
    ('dspark-t16-w32768', 'dspark', 15, 32768, 'free', 'windows'),
    ('dflash2-forced', 'dflash2', 15, 2048, 'forced', 'forced'),
    ('dspark-t8', 'dspark', 7, None, 'free', 'depth'),
    ('dflash2-t8', 'dflash2', 7, 2048, 'free', 'depth'),
)
CORE = ('dflash2-t16', 'dspark-t16')


class RunStop(Exception):
    """The run stops now (an out-of-memory, the watchdog, the deadline); the reason is a plain word."""
    def __init__(self, reason):
        Exception.__init__(self, reason)
        self.reason = reason


def is_oom(error):
    return isinstance(error, MemoryError) or type(error).__name__ in ('OutOfMemoryError', 'AcceleratorError')


# -- the plan ----------------------------------------------------------------------------------------------------------------

def largest_remainder(shares, total):
    """Whole counts per key summing to `total`, proportional to `shares` {key: weight}."""
    weight = float(sum(shares.values())) or 1.0
    base = dict((key, int(total * value / weight)) for key, value in shares.items())
    leftover = total - sum(base.values())
    for key in sorted(shares, key=lambda k: (-(total * shares[k] / weight - base[k]), k))[:leftover]:
        base[key] += 1
    return base


def select_subset(meta, per_set, seed, label, eligible=None):
    """A seeded stratified draw over set x bucket: per set the requested count, spread over that set's buckets in proportion to
    how many eligible turns each holds. -> sorted list of k. A set with fewer eligible turns than asked gives all of them."""
    rng = random.Random('%s:%s' % (seed, label))
    chosen = []
    for name, want in sorted(per_set.items()):
        pool = [row for row in meta if row['set'] == name and (eligible is None or eligible(row))]
        buckets = {}
        for row in pool:
            buckets.setdefault(row['bucket'], []).append(row['k'])
        take = largest_remainder(dict((bucket, len(ks)) for bucket, ks in buckets.items()), min(want, len(pool))) if buckets else {}
        for bucket in sorted(take):
            ks = sorted(buckets[bucket])
            rng.shuffle(ks)
            chosen.extend(ks[:take[bucket]])
    return sorted(chosen)


def make_plan(meta, seed, arms=None):
    """{'arms': [spec dict ...], 'subsets': {label: [k]}}; `arms` restricts the registered list (in its order)."""
    subsets = {}
    for label, per_set in sorted(SUBSETS.items()):
        subsets[label] = select_subset(meta, per_set, seed, label, eligible=(lambda row: row.get('scheduled')) if label == 'forced' else None)
    chosen = [spec for spec in ARM_SPECS if arms is None or spec[0] in arms]
    return dict(seed=seed, subsets=subsets, arms=[dict(name=n, drafter=d, proposals=p, window=w, mode=m, subset=s)
                                                   for n, d, p, w, m, s in chosen])


def group_order(meta, seed):
    """Group ids in the run order: a seeded shuffle within each set, the sets interleaved in proportion to their turns, so any prefix
    of the order is a stratified sample."""
    by_set = {}
    for row in meta:
        by_set.setdefault(row['set'], {}).setdefault(row['group'], 0)
        by_set[row['set']][row['group']] += 1
    rng = random.Random('%s:order' % seed)
    queues = {}
    for name, groups in sorted(by_set.items()):
        ids = sorted(groups)
        rng.shuffle(ids)
        queues[name] = [ids, 0, float(sum(groups.values()))]
    out, given = [], dict((name, 0.0) for name in queues)
    total = float(sum(queue[2] for queue in queues.values()))
    while any(queue[1] < len(queue[0]) for queue in queues.values()):
        open_sets = [name for name, queue in queues.items() if queue[1] < len(queue[0])]
        name = max(open_sets, key=lambda n: (queues[n][2] / total * (len(out) + 1) - given[n], n))
        queue = queues[name]
        group = queue[0][queue[1]]
        queue[1] += 1
        given[name] += by_set[name][group]
        out.append(group)
    return out


# -- the memory guard -----------------------------------------------------------------------------------------------------------

class Guard(object):
    """Stop and admission checks, all injected: the watchdog's trip file, the deadline, the memory the host has now."""

    def __init__(self, trip_file, deadline, clock, read_avail_gib, floor_gib=FLOOR_GIB, transient_gib=TRANSIENT_GIB, peak_gib=lambda: 0.0,
                 cap_gib=None):
        self.trip_file, self.deadline, self.clock, self.read_avail = trip_file, deadline, clock, read_avail_gib
        self.floor, self.transient, self.peak, self.cap = floor_gib, transient_gib, peak_gib, cap_gib

    def stop_reason(self):
        if self.trip_file and os.path.exists(self.trip_file):
            return 'watchdog'
        if self.clock() >= self.deadline:
            return 'deadline'
        return None

    def projected_gib(self, tokens):
        return self.transient + tokens * TOKEN_BYTES / GIB

    def admits(self, tokens):
        return self.read_avail() - self.projected_gib(tokens) >= self.floor

    def peak_ok(self):
        return self.cap is None or self.peak() <= self.cap


# -- the results directory ---------------------------------------------------------------------------------------------------------

class Store(object):
    def __init__(self, directory):
        self.directory = directory
        os.makedirs(directory, mode=0o700, exist_ok=True)

    def path(self, name):
        return os.path.join(self.directory, name)

    def append(self, name, entry):
        with open(self.path(name), 'a', encoding='utf-8', newline='\n') as handle:
            handle.write(json.dumps(entry, sort_keys=True) + '\n')

    def lines(self, name):
        path = self.path(name)
        out = []
        if os.path.isfile(path):
            with open(path, encoding='utf-8') as handle:
                for line in handle:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        continue
        return out

    def write(self, name, value):
        with open(self.path(name), 'w', encoding='utf-8', newline='\n') as handle:
            json.dump(value, handle, sort_keys=True, indent=1)
            handle.write('\n')

    def done(self, arm):
        return set(entry['k'] for entry in self.lines('arm-%s.jsonl' % arm) if entry.get('status') == 'ok')


# -- the run -----------------------------------------------------------------------------------------------------------------------

class Arm(object):
    """What the driver needs of an arm: begin(record, features), propose(sequence, start, count), end()."""


class WalkerArm(object):
    """tf_pair_walk's `propose` over a0_drafters' walkers (begin takes the features only)."""
    def __init__(self, walker):
        self.walker = walker

    def begin(self, record, features):
        self.walker.begin(features)

    def propose(self, sequence, start, count):
        return self.walker.propose(sequence, start, count)

    def end(self):
        self.walker.end()


class Runner(object):
    def __init__(self, bundle_dir, store, group_runner, make_arm, plan, guard, say=print, perf=time.perf_counter, width=WIDTH):
        self.bundle_dir, self.store, self.group_runner, self.make_arm = bundle_dir, store, group_runner, make_arm
        self.plan, self.guard, self.say, self.perf, self.width = plan, guard, say, perf, width
        self.total = 0
        self.v1_done = set(entry['k'] for entry in store.lines('v1.jsonl'))
        self.arms = dict((spec['name'], make_arm(spec)) for spec in plan['arms'])
        self.counts = dict(turns=0, arm_ok=0, arm_failed=0, deferred=0, groups=0)
        self.stopped = None

    def _applies(self, spec, k):
        return spec['subset'] is None or k in self.plan['subsets'][spec['subset']]

    def pending(self, k, done):
        return [spec for spec in self.plan['arms'] if self._applies(spec, k) and k not in done[spec['name']]]

    def run(self, only=None):
        done = dict((spec['name'], self.store.done(spec['name'])) for spec in self.plan['arms'])
        records = list(bundle.read_bundle(self.bundle_dir))
        self.total = len(records)
        by_group = {}
        for record in records:
            by_group.setdefault(record['group'], []).append(record)
        order = group_order([dict(set=r['set'], group=r['group']) for r in records], self.plan['seed'])
        try:
            for group in order:
                members = sorted(by_group[group], key=lambda item: item['order'])
                if only is not None and not any(r['k'] in only for r in members):
                    continue
                wanted = [r for r in members if (only is None or r['k'] in only) and self.pending(r['k'], done)]
                if not wanted:
                    continue
                reason = self.guard.stop_reason()
                if reason:
                    raise RunStop(reason)
                tokens = max(len(r['prompt_ids']) + len(r['output_ids']) for r in members)
                if not self.guard.admits(tokens):
                    self.counts['deferred'] += len(wanted)
                    continue
                self._run_group(members, wanted, done)
        except RunStop as stop:
            self.stopped = stop.reason
        self.store.write('state.json', dict(counts=self.counts, stopped=self.stopped))
        return self.counts

    def _run_group(self, members, wanted, done):
        wanted_k = set(r['k'] for r in wanted)
        try:
            self.group_runner.run_group(members, lambda record, view, v1: self._consume(record, view, v1, done), self.width,
                                        lambda record: record['k'] in wanted_k)
            self.counts['groups'] += 1
        except RunStop:
            raise
        except Exception as error:
            if is_oom(error):
                raise RunStop('oom')
            for record in wanted:                        # a failed group: every pending arm of its turns is a failed turn
                for spec in self.pending(record['k'], done):
                    self.store.append('arm-%s.jsonl' % spec['name'], dict(k=record['k'], status='failed', error=type(error).__name__))
                    self.counts['arm_failed'] += 1

    def _consume(self, record, view, v1, done):
        k = record['k']
        if k not in self.v1_done:
            self.store.append('v1.jsonl', dict(k=k, set=record['set'], **v1.as_dict()))
            self.v1_done.add(k)
        self.counts['turns'] += 1
        for spec in self.pending(k, done):
            reason = self.guard.stop_reason()
            if reason:
                raise RunStop(reason)
            arm = self.arms[spec['name']]
            began = self.perf()
            try:
                arm.begin(record, view)
                if spec['mode'] == 'forced':
                    rounds = walk.walk_forced(record, arm, spec['proposals'], record['schedule'])
                else:
                    rounds = walk.walk_free(record, arm, spec['proposals'])
                entry = dict(k=k, status='ok', rounds=[walk.to_row(item) for item in rounds])
                if spec['mode'] == 'forced':
                    entry['logged'] = [item['logged'] for item in rounds]
                self.store.append('arm-%s.jsonl' % spec['name'], entry)
                self.store.append('timing.jsonl', dict(k=k, arm=spec['name'], rounds=len(rounds), ms=round(1000 * (self.perf() - began), 1)))
                done[spec['name']].add(k)
                self.counts['arm_ok'] += 1
            except Exception as error:
                if is_oom(error):
                    raise RunStop('oom')
                self.store.append('arm-%s.jsonl' % spec['name'], dict(k=k, status='failed', error=type(error).__name__))
                self.counts['arm_failed'] += 1
            finally:
                try:
                    arm.end()
                except Exception:
                    pass
        self.say('turn %d of %d: arms ok %d, failed %d' % (self.counts['turns'], self.total, self.counts['arm_ok'], self.counts['arm_failed']))


# -- the validity gates the driver can compute --------------------------------------------------------------------------------------

def gate_v1(v1_lines):
    """('PASS'|'FAIL'|'NOT_RUN', counts): HF argmax against the logged answer >= 95% of rows overall and per set."""
    rows, agree, by_set = 0, 0, {}
    for entry in v1_lines:
        rows += entry['rows']
        agree += entry['agree']
        pair = by_set.setdefault(entry['set'], [0, 0])
        pair[0] += entry['rows']
        pair[1] += entry['agree']
    if not rows:
        return 'NOT_RUN', dict(rows=0)
    ok = agree >= V1_FLOOR * rows and all(b >= V1_FLOOR * a for a, b in by_set.values())
    return ('PASS' if ok else 'FAIL'), dict(rows=rows, agree=agree)


def served_arm(meta, records):
    """The served schedule as an arm: per scheduled turn the emitted counts of its logged rounds, the terminal round left out."""
    out = {}
    for record in records:
        if record.get('schedule') and len(record['schedule']) > 1:
            out[record['k']] = [dict(committed=value, uncapped=value, cap=16, accepted=value - 1, available=15, offset=1, region='content')
                                for value in record['schedule'][:-1]]
    return out


def gate_v3(meta_rows, free_arm, served, resamples=2000, seed=0):
    """Free-walk DFlash2 tau against the lab's served tau on the same turns: the pooled ratio inside [0.93, 1.12]."""
    meta = dict((row['k'], row) for row in meta_rows)
    keys = sorted(set(free_arm) & set(served))
    if len(keys) < 10:
        return 'NOT_RUN', dict(turns=len(keys))
    units = report.units_by_set(meta, keys, free_arm, served)
    result = report.cluster_bootstrap(units, report.ratio_statistic(), resamples, seed)
    point = result['point']
    ok = point is not None and V3_BAND[0] <= point <= V3_BAND[1]
    return ('PASS' if ok else 'FAIL'), dict(turns=len(keys), ratio=point, low=result['low'], high=result['high'])


def v4_decide(rows, rounds):
    """V4: DSpark on the GPU against the CPU reference. `rows`: [(tokens agree, reference top-2 margin)]; `rounds`: [(gpu accepted,
    reference accepted)]. PASS when the agreement over rows with margin >= 0.25 is >= 99% and the accepted lengths equal in >= 97%."""
    firm = [agree for agree, margin in rows if margin >= V4_MARGIN]
    if not firm or not rounds:
        return 'NOT_RUN'
    return 'PASS' if sum(firm) >= V4_AGREE * len(firm) and sum(1 for a, b in rounds if a == b) >= V4_ROUNDS * len(rounds) else 'FAIL'


def calibration(store):
    """V2's report-only numbers from the forced arm: the share of rounds whose committed equals the served emitted, and the ratio
    of the forced committed total to the served one (counts only)."""
    exact = total = forced = served = 0
    for entry in store.lines('arm-dflash2-forced.jsonl'):
        if entry.get('status') != 'ok':
            continue
        for row, emitted in zip(entry['rounds'], entry['logged']):
            total += 1
            exact += 1 if row[6] == emitted else 0
            forced += row[6]
            served += emitted
    return dict(rounds=total, exact=exact, forced_committed=forced, served_committed=served)


def finalize(store, bundle_dir, guard, gates=None):
    """validity.json from the gates the driver computes (V1, V2, V3, V6) and those the harness passes in (V0, V4)."""
    gates = dict(gates or {})
    records = list(bundle.read_bundle(bundle_dir))
    with open(os.path.join(bundle_dir, bundle.META_NAME), encoding='utf-8') as handle:
        meta_rows = [json.loads(line) for line in handle if line.strip()]
    validity = dict((name, gates.get(name, 'NOT_RUN')) for name in report.VALIDITY_GATES)
    validity['V1'], v1_counts = gate_v1(store.lines('v1.jsonl'))
    free = dict((entry['k'], [walk.from_row(row) for row in entry['rounds']])
                for entry in store.lines('arm-dflash2-t16.jsonl') if entry.get('status') == 'ok')
    validity['V3'], v3_counts = gate_v3(meta_rows, free, served_arm(meta_rows, records))
    forced = calibration(store)
    validity['V2'] = 'PASS' if forced['rounds'] else 'NOT_RUN'
    validity['V6'] = 'PASS' if guard.peak_ok() and not (guard.trip_file and os.path.exists(guard.trip_file)) else 'FAIL'
    store.write('validity.json', validity)
    store.write('calibration.json', dict(forced=forced, v1=v1_counts, v3=v3_counts))
    return validity


def prepare_results(bundle_dir, out):
    """The results directory with the bundle's metadata beside it (the report reads meta.jsonl)."""
    store = Store(out)
    target = store.path('meta.jsonl')
    if not os.path.exists(target):
        shutil.copyfile(os.path.join(bundle_dir, bundle.META_NAME), target)
    return store


# -- the real environment (GPU host only; trusted after the W0 canary) -------------------------------------------------------------

def read_avail_gib():
    with open('/proc/meminfo') as handle:
        for line in handle:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / float(1 << 20)
    return 0.0


def write_phase(path, value):
    if path:
        with open(path, 'w') as handle:
            handle.write(value + '\n')


def drop_file_cache(directory):
    """posix_fadvise(DONTNEED) on every file under `directory`: the page cache of a loaded shard must not stack on the device copy."""
    advise = getattr(os, 'posix_fadvise', None)
    if advise is None:
        return 0
    count = 0
    for base, _, names in os.walk(directory):
        for name in names:
            descriptor = os.open(os.path.join(base, name), os.O_RDONLY)
            try:
                advise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
                count += 1
            finally:
                os.close(descriptor)
    return count


def gate_v0(checks):
    """V0 from booleans the environment builder collected (versions, source and checkpoint pins, target revision, bundle manifest,
    memory caps active, the fast GDN kernels in use): PASS only when every one is True."""
    return 'PASS' if checks and all(checks.values()) else 'FAIL'


def real_environment(options, say):  # pragma: no cover - needs the GPU host, the models and the pins
    """(group_runner, make_arm, guard, gates). Loads the target and the drafters; every step is checked against a pin first."""
    import importlib.util
    import torch
    import a0_drafters as drafters
    import a0_target as target_module
    import a0_upstream as upstream
    import dflash2_torch as d2
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    write_phase(options.phase_file, 'loading')
    checks = {}
    pins = upstream.load_pins(options.pins, ['zlab_model.py', 'dspark.py', 'dflash.py', 'dflash2.safetensors', 'dspark.safetensors'])
    for name in ('dspark.py', 'dflash.py'):                      # the reviewed DSpark sources: checked, kept for the V4 reference
        upstream.read_pinned(os.path.join(options.dspark_src, name), pins[name][1], pins[name][0])
    zlab = upstream.load_pinned_module(os.path.join(options.zlab, 'model.py'), pins['zlab_model.py'][1], 'zlab_model',
                                       expected_bytes=pins['zlab_model.py'][0])
    for directory, name in ((options.dflash2, 'dflash2.safetensors'), (options.dspark, 'dspark.safetensors')):
        upstream.verify_checkpoint(directory, {'model.safetensors': pins[name][1]})
    checks['pins'] = True
    checks['bundle'] = bool(bundle.verify_bundle(options.bundle))
    checks['fast_gdn'] = importlib.util.find_spec('fla') is not None
    checks['memory_caps'] = options.cap_gib is not None
    total = torch.cuda.get_device_properties(0).total_memory / GIB
    torch.cuda.set_per_process_memory_fraction(min(1.0, (options.cap_gib or 96.0) / total))
    model = AutoModelForCausalLM.from_pretrained(options.target, dtype=torch.bfloat16, device_map='cuda').eval()
    drop_file_cache(options.target)
    embed, head = model.get_input_embeddings().weight, model.lm_head.weight
    hf = target_module.HFTarget(model, (5, 19, 33, 47, 61), 'cuda')
    group_runner = target_module.GroupRunner(hf, options.chunk, torch.bfloat16, 'cuda')

    control = zlab.DFlash2DraftModel.from_pretrained(options.dflash2, dtype=torch.bfloat16).to('cuda').eval()
    control_port = d2.Dflash2(d2.real_config()).to(torch.bfloat16).to('cuda').eval()
    drafters.load_port_state(control_port, load_file(os.path.join(options.dflash2, 'model.safetensors'), device='cuda'))
    dflash2_backbone = drafters.UpstreamBackbone(control, embed, lambda: zlab._make_cache(control.config), zlab._crop_to)
    dspark_tensors = load_file(os.path.join(options.dspark, 'model.safetensors'), device='cuda')
    dspark_model = d2.Dflash2(d2.real_dspark_config()).to(torch.bfloat16).to('cuda').eval()
    skipped = tuple(name for name in dspark_tensors if name.startswith(('confidence_head', 'markov_head')))
    drafters.load_port_state(dspark_model, dspark_tensors, ignore=skipped)
    predecessor, successor = dspark_tensors['markov_head.markov_w1.weight'], dspark_tensors['markov_head.markov_w2.weight']
    dspark_backbone = drafters.PortBackbone(dspark_model, embed)
    mask = 248070

    def make_arm(spec):
        if spec['drafter'] == 'dflash2':
            return WalkerArm(drafters.Dflash2Walker(dflash2_backbone, head, control_port, mask, spec['proposals'], spec['window']))
        return WalkerArm(drafters.DSparkWalker(dspark_backbone, head, predecessor, successor, mask, spec['proposals'], spec['window']))

    guard = Guard(options.trip_file, options.deadline, time.time, read_avail_gib, options.floor_gib,
                  peak_gib=lambda: torch.cuda.max_memory_allocated() / GIB, cap_gib=options.cap_gib)
    write_phase(options.phase_file, 'running')
    return group_runner, make_arm, guard, dict(V0=gate_v0(checks))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--deadline', type=float, required=True, help='epoch seconds; the run stops cleanly before this')
    parser.add_argument('--trip-file', required=True, help='the watchdog writes its reason here')
    parser.add_argument('--phase-file')
    parser.add_argument('--target')
    parser.add_argument('--dflash2')
    parser.add_argument('--dspark')
    parser.add_argument('--zlab', help='directory holding the pinned z-lab model.py')
    parser.add_argument('--dspark-src', help='directory holding the pinned DSpark dspark.py and dflash.py')
    parser.add_argument('--pins', help='pins file (JSON)')
    parser.add_argument('--arms', help='comma separated arm names (default: all, in the registered order)')
    parser.add_argument('--canary', action='store_true', help='3 shortest turns and the longest, core arms only, into <out>/canary')
    parser.add_argument('--seed', type=int, default=20261003)
    parser.add_argument('--chunk', type=int, default=4096)
    parser.add_argument('--floor-gib', type=float, default=FLOOR_GIB)
    parser.add_argument('--cap-gib', type=float, default=96.0)
    return parser


def canary_keys(meta_rows):
    """The canary's turns: the three shortest and the single longest (prompt plus answer)."""
    ranked = sorted(meta_rows, key=lambda row: row['prompt_tokens'] + row['answer_tokens'])
    return set(row['k'] for row in ranked[:3] + ranked[-1:])


def main(argv=None, say=print, environment=real_environment):
    options = build_parser().parse_args(argv)
    try:
        bundle.verify_bundle(options.bundle)
        store = prepare_results(options.bundle, os.path.join(options.out, 'canary') if options.canary else options.out)
        with open(store.path('meta.jsonl'), encoding='utf-8') as handle:
            meta_rows = [json.loads(line) for line in handle if line.strip()]
        arms = CORE if options.canary else (options.arms.split(',') if options.arms else None)
        plan = make_plan(meta_rows, options.seed, arms)
        store.write('plan.json', plan)
        group_runner, make_arm, guard, gates = environment(options, say)
        runner = Runner(options.bundle, store, group_runner, make_arm, plan, guard, say, width=getattr(group_runner, 'width', WIDTH))
        runner.run(canary_keys(meta_rows) if options.canary else None)
        validity = finalize(store, options.bundle, guard, gates)
    except Exception as error:               # the type only: a message could carry a path or a value
        say('refused: %s' % type(error).__name__)
        return 2
    say('done: %d turns, %d arms ok, %d failed, %d deferred, stopped: %s' % (
        runner.counts['turns'], runner.counts['arm_ok'], runner.counts['arm_failed'], runner.counts['deferred'], runner.stopped or 'no'))
    say('gates: %s' % ' '.join('%s=%s' % (name, validity[name]) for name in report.VALIDITY_GATES))
    return 0 if runner.stopped in (None, 'deadline') else 3


if __name__ == '__main__':
    sys.exit(main())
