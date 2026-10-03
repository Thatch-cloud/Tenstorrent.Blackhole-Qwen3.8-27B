"""The A0 screen's driver, for the GPU host: runs every registered arm over every bundle turn off ONE target forward per prefix group,
resumably, under a memory guard and a deadline, and writes counts to a private results directory (the report reads it:
tf_pair_report.py). Everything here but `real_environment` runs on CPU against fakes in the tests; `real_environment` loads the models
and is verified only by the W0 canary (V0, V1, V3, V4 and the branch self-check).

    python3 scripts/ci/a0_run.py --bundle <bundle dir> --out <results dir> --deadline <epoch> --trip-file <file> --heartbeat-file <file>
        [--canary | --load-only] ...

THE ORDER IS THE PRIORITY. Groups run in a seeded, set-interleaved order, so any prefix of the run is a stratified sample and a stop
at the deadline still leaves a usable screen; within a turn the arms run in the registered order. A turn's failure is recorded (the
exception TYPE only) and the run goes on; an out-of-memory stops the run (it never skips ahead), and a turn whose projected peak
would break the memory floor is not started (counted as deferred, resumable later).

RESULTS (out/): meta.jsonl (copied from the bundle), arm-<name>.jsonl (k, status, compact rounds), v1.jsonl, timing.jsonl, plan.json,
state.json, calibration.json, validity.json, deferred.jsonl (turns not started for want of memory: the report counts them as lost),
load-memory.jsonl (MemFree / MemAvailable while the models load). stdout carries counts only; an exception prints its type only.
"""
import argparse
import json
import os
import random
import shutil
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a0_bundle_io as bundle  # noqa: E402
import tf_pair_report as report  # noqa: E402
import tf_pair_walk as walk  # noqa: E402
import a0_target as tgt  # noqa: E402

GIB = float(1 << 30)
WIDTH = 5 * 5120
TOKEN_BYTES = 65536 + 51200          # target KV (16 full-attention layers) + raw taps, per token
DSPARK_CONTEXT_BYTES = 20480         # DSpark's context K/V cache per token (5 layers x K and V x 8 heads x 128 x 2 B); full attention keeps all
CAT_FACTOR = 2                       # torch.cat while it grows holds the old and the new copy at once
ADMIT_TOKEN_BYTES = TOKEN_BYTES + CAT_FACTOR * DSPARK_CONTEXT_BYTES
TRANSIENT_GIB = 4.0                  # prefill chunk transients
FLOOR_GIB = 10.0
V1_FLOOR = 0.95
V3_BAND = (0.93, 1.12)
V3_LOW = 0.98                        # the CI lower bound of the control's free-walk / served ratio: a control depressed by 7% must not pass
HEARTBEAT_MAX_AGE = 10.0             # seconds; a watchdog heartbeat older than this stops the harness
V4_AGREE, V4_MARGIN, V4_ROUNDS = 0.99, 0.25, 0.97
V4_TURNS, V4_ROUNDS_PER_TURN, V4_WINDOW, V4_PROPOSALS, V4_MIN_ANSWER = 8, 5, 1024, 7, 24
V4_TURNS_MIN_ROUNDS = 20             # fewer rounds than this cannot show 97% round agreement: NOT_RUN, never PASS
SELFCHECK_PROMPT, SELFCHECK_ANSWER = 512, 16
SELFCHECK_ATOL, SELFCHECK_AGREE = 3e-2, 0.9     # BF16 on the GPU: a different chunking is not bit-identical, a wrong restore is far off
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


OOM_WORDS = ('out of memory', 'alloc_failed', 'failed to allocate', 'cannot allocate memory', 'memory allocation')


def is_oom(error):
    """An allocation failure from ANY layer: the torch allocator's error type, MemoryError, or a RuntimeError out of Triton, cuBLAS
    or a driver call whose text says so (the text is inspected here and never printed or stored)."""
    if isinstance(error, MemoryError) or type(error).__name__ in ('OutOfMemoryError', 'AcceleratorError'):
        return True
    try:
        text = str(error).lower()
    except Exception:
        return False
    return any(word in text for word in OOM_WORDS)


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
    """Stop and admission checks, all injected: the watchdog's trip file and heartbeat, the deadline, the memory the host has now.
    FAIL CLOSED: with a heartbeat file configured, a missing or stale heartbeat stops the run (the watchdog is dead or hung)."""

    def __init__(self, trip_file, deadline, clock, read_avail_gib, floor_gib=FLOOR_GIB, transient_gib=TRANSIENT_GIB, peak_gib=lambda: 0.0,
                 cap_gib=None, heartbeat_file=None, heartbeat_max_age=HEARTBEAT_MAX_AGE, release=None):
        self.trip_file, self.deadline, self.clock, self.read_avail = trip_file, deadline, clock, read_avail_gib
        self.floor, self.transient, self.peak, self.cap = floor_gib, transient_gib, peak_gib, cap_gib
        self.heartbeat_file, self.heartbeat_max_age, self.release = heartbeat_file, heartbeat_max_age, release

    def heartbeat_state(self):
        """'ok', 'missing' or 'stale' (always 'ok' when no heartbeat file is configured)."""
        if not self.heartbeat_file:
            return 'ok'
        try:
            with open(self.heartbeat_file, encoding='utf-8') as handle:
                stamp = float(handle.read().split()[0])
        except (OSError, ValueError, IndexError):
            return 'missing'
        return 'ok' if self.clock() - stamp <= self.heartbeat_max_age else 'stale'

    def stop_reason(self):
        if self.trip_file and os.path.exists(self.trip_file):
            return 'watchdog'
        state = self.heartbeat_state()
        if state != 'ok':
            return 'watchdog_' + state
        if self.clock() >= self.deadline:
            return 'deadline'
        return None

    def make_room(self):
        """Give back what the allocator is holding (the caching allocator's reserved blocks) before MemAvailable is read."""
        if self.release is not None:
            self.release()

    def projected_gib(self, tokens):
        return self.transient + tokens * ADMIT_TOKEN_BYTES / GIB

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
                self.guard.make_room()
                if not self.guard.admits(tokens):
                    self.counts['deferred'] += len(wanted)
                    for record in wanted:                  # kept on disk: the report counts a deferred turn as LOST, not as unrun
                        self.store.append('deferred.jsonl', dict(k=record['k'], tokens=tokens))
                    continue
                self._run_group(members, wanted, done)
                self.guard.make_room()
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
    """Free-walk DFlash2 tau against the lab's served tau on the same turns (paired, so the interval is tight): the pooled ratio inside
    [0.93, 1.12] AND its CI lower bound at least 0.98. A control that walks below the served one by a few percent inflates R by the same
    factor, so the point estimate alone is not enough; the report also gives R times this ratio as a sensitivity."""
    meta = dict((row['k'], row) for row in meta_rows)
    keys = sorted(set(free_arm) & set(served))
    if len(keys) < 10:
        return 'NOT_RUN', dict(turns=len(keys))
    units = report.units_by_set(meta, keys, free_arm, served)
    result = report.cluster_bootstrap(units, report.ratio_statistic(), resamples, seed)
    point = result['point']
    ok = point is not None and V3_BAND[0] <= point <= V3_BAND[1] and result['low'] is not None and result['low'] >= V3_LOW
    return ('PASS' if ok else 'FAIL'), dict(turns=len(keys), ratio=point, low=result['low'], high=result['high'])


def v4_decide(rows, rounds):
    """V4: DSpark on the GPU against the CPU reference. `rows`: [(tokens agree, reference top-2 margin)]; `rounds`: [(gpu accepted,
    reference accepted)]. PASS when the agreement over rows with margin >= 0.25 is >= 99% and the accepted lengths equal in >= 97%."""
    firm = [agree for agree, margin in rows if margin >= V4_MARGIN]
    if not firm or not rounds:
        return 'NOT_RUN'
    return 'PASS' if sum(firm) >= V4_AGREE * len(firm) and sum(1 for a, b in rounds if a == b) >= V4_ROUNDS * len(rounds) else 'FAIL'


def v4_pick(records, turns=V4_TURNS, minimum_answer=V4_MIN_ANSWER):
    """The V4 turns: the shortest of each set in turn (round robin over the sets), answers long enough for the rounds."""
    pools = {}
    for record in sorted(records, key=lambda r: (len(r['prompt_ids']) + len(r['output_ids']), r['k'])):
        if len(record['output_ids']) >= minimum_answer:
            pools.setdefault(record['set'], []).append(record)
    chosen = []
    while len(chosen) < turns and any(pools.values()):
        for name in sorted(pools):
            if pools[name] and len(chosen) < turns:
                chosen.append(pools[name].pop(0))
    return chosen


def v4_positions(answer_len, rounds=V4_ROUNDS_PER_TURN, proposals=V4_PROPOSALS):
    """Answer indices of the V4 anchors: evenly spread over the answer, each with a full block of logged tokens after it."""
    last = answer_len - proposals - 1
    if last < 0:
        return []
    if rounds == 1:
        return [0]
    return sorted(set(int(round(i * last / float(rounds - 1))) for i in range(rounds)))


class V4Env(object):
    """What V4 compares: gpu_block / cpu_block(view, noise_ids, start, window) -> [rows, H] backbone rows of the SAME inputs on the two
    devices; the output head and Markov tables on each device; the mask token."""
    def __init__(self, gpu_block, cpu_block, head, predecessor, successor, cpu_head, cpu_predecessor, cpu_successor, mask):
        self.gpu_block, self.cpu_block, self.head, self.predecessor, self.successor = gpu_block, cpu_block, head, predecessor, successor
        self.cpu_head, self.cpu_predecessor, self.cpu_successor, self.mask = cpu_head, cpu_predecessor, cpu_successor, mask


def v4_collect(records, group_runner, width, env, turns=V4_TURNS, rounds=V4_ROUNDS_PER_TURN, window=V4_WINDOW, proposals=V4_PROPOSALS):
    """Run V4: for the picked turns, one target pass each, then `rounds` anchors per turn through both devices. -> (rows, accepted pairs)
    in v4_decide's form, plus the counts. Needs every picked turn to give at least one round."""
    import a0_drafters as drafters
    rows, pairs, used = [], [], 0
    for record in v4_pick(records, turns):
        sequence = list(record['prompt_ids']) + list(record['output_ids'])
        prompt = len(record['prompt_ids'])
        positions = v4_positions(len(record['output_ids']), rounds, proposals)

        def consume(item, view, v1, sequence=sequence, prompt=prompt, positions=positions):
            for j in positions:
                start = prompt + j
                anchor = sequence[start]
                noise = [anchor] + [env.mask] * (proposals - 1)
                gpu_hidden = env.gpu_block(view, noise, start, window)
                cpu_hidden = env.cpu_block(view, noise, start, window)
                answer_next = sequence[start + 1:start + 1 + proposals]
                new_rows, new_pair = drafters.v4_round(gpu_hidden, cpu_hidden, env.head, env.predecessor, env.successor, anchor,
                                                       answer_next, drafters.matching_prefix, env.cpu_head, env.cpu_predecessor,
                                                       env.cpu_successor)
                rows.extend(new_rows)
                pairs.append(new_pair)
        group_runner.run_group([record], consume, width)
        used += 1
    return rows, pairs, dict(turns=used, rounds=len(pairs), rows=len(rows))


def v4_gate(records, group_runner, width, env):
    """('PASS' | 'FAIL' | 'NOT_RUN', counts): V4 over the picked turns. Too few rounds (a bundle with no answer long enough) is NOT_RUN."""
    rows, pairs, counts = v4_collect(records, group_runner, width, env)
    if counts['rounds'] < V4_TURNS_MIN_ROUNDS:
        return 'NOT_RUN', counts
    status = v4_decide(rows, pairs)
    firm = [agree for agree, margin in rows if margin >= V4_MARGIN]
    counts.update(firm_rows=len(firm), firm_agree=sum(1 for agree in firm if agree),
                  rounds_equal=sum(1 for a, b in pairs if a == b))
    return status, counts


def selfcheck_gate(target, records, atol=SELFCHECK_ATOL, min_agree=SELFCHECK_AGREE, chunk=tgt.CHUNK):
    """('PASS' | 'FAIL' | 'NOT_RUN', stats): snapshot / restore against a fresh prefill on the shortest turn's first tokens (a prompt
    of up to 512 and two answers of 16, from the same logged answer)."""
    usable = [r for r in records if len(r['output_ids']) >= 2 * SELFCHECK_ANSWER and len(r['prompt_ids']) >= 8]
    if not usable:
        return 'NOT_RUN', dict(rows=0)
    record = min(usable, key=lambda r: (len(r['prompt_ids']), r['k']))
    prompt = list(record['prompt_ids'])[:SELFCHECK_PROMPT]
    answer = list(record['output_ids'])
    stats = tgt.branch_selfcheck_stats(target, prompt, answer[:SELFCHECK_ANSWER], answer[SELFCHECK_ANSWER:2 * SELFCHECK_ANSWER], chunk)
    return ('PASS' if tgt.selfcheck_passes(stats, atol, min_agree) else 'FAIL'), stats


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
    details = gates.pop('_details', {})
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
    store.write('calibration.json', dict(forced=forced, v1=v1_counts, v3=v3_counts, checks=details))
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


class CacheDropper(object):
    """While the models load, drop the page cache of their files every `interval` seconds (posix_fadvise DONTNEED reaches the pages the
    loader has already released: shard by shard), and sample MemFree / MemAvailable into `log`. Without it the whole checkpoint sits in
    the page cache on top of its device copy, which is the memory pattern that exhausts a unified-memory host. Used as a context manager;
    `drop(directory)`, `sample()` -> dict and `log(entry)` are injected."""

    def __init__(self, directories, drop, interval=0.5, sample=None, log=None, clock=time.time):
        self.directories, self.drop, self.interval, self.sample, self.log, self.clock = list(directories), drop, interval, sample, log, clock
        self.stop = threading.Event()
        self.thread = None
        self.dropped = 0

    def step(self):
        for directory in self.directories:
            try:
                self.dropped += self.drop(directory)
            except OSError:
                pass
        if self.sample is not None and self.log is not None:
            entry = dict(self.sample())
            entry['t'] = round(self.clock(), 1)
            self.log(entry)

    def _loop(self):
        while not self.stop.is_set():
            self.step()
            self.stop.wait(self.interval)

    def __enter__(self):
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(5.0)
        self.step()                                # the last shard
        return False


def read_meminfo_sample():
    """{'free_gib', 'avail_gib'} from /proc/meminfo (0 where unreadable)."""
    out = dict(free_gib=0.0, avail_gib=0.0)
    try:
        with open('/proc/meminfo') as handle:
            for line in handle:
                if line.startswith('MemFree:'):
                    out['free_gib'] = round(int(line.split()[1]) / float(1 << 20), 2)
                elif line.startswith('MemAvailable:'):
                    out['avail_gib'] = round(int(line.split()[1]) / float(1 << 20), 2)
    except OSError:
        pass
    return out


def gate_v0(checks):
    """V0 from booleans the environment builder collected (versions, source and checkpoint pins, target revision, bundle manifest,
    memory caps active, the fast GDN kernels in use): PASS only when every one is True."""
    return 'PASS' if checks and all(checks.values()) else 'FAIL'


def real_environment(options, say):  # pragma: no cover - needs the GPU host, the models and the pins
    """(group_runner, make_arm, guard, gates). Loads the target and the drafters, every step checked against a pin first, then runs the
    checks that make a verdict possible: V0 (versions, pins, the fast GDN kernel actually bound, neutral drafter scalars and matching
    taps, the mask token, the branch self-check), and V4 (DSpark on the GPU against the CPU reference). Their results come back in
    `gates` (V0, V4) with the numbers under '_details'; a run whose V0 or V4 is not PASS stops before any turn (see main)."""
    import gc
    import importlib.metadata
    import importlib.util
    import torch
    import a0_drafters as drafters
    import a0_target as target_module
    import a0_upstream as upstream
    import dflash2_torch as d2
    import dspark_intake
    from dspark_backbone_reference import CPUBackbone
    from safetensors.torch import load_file
    import transformers
    from transformers import AutoModelForCausalLM

    taps = tuple(dspark_intake.TAPS)
    write_phase(options.phase_file, 'loading')
    checks, details = {}, {}
    memory_log = (lambda entry: write_line(options.memory_log, entry)) if getattr(options, 'memory_log', None) else None

    def loading(directory):
        return CacheDropper([directory], drop_file_cache, sample=read_meminfo_sample, log=memory_log)

    pins = upstream.load_pins(options.pins, ['zlab_model.py', 'dspark.py', 'dflash.py', 'dflash2.safetensors', 'dspark.safetensors'])
    for name in ('dspark.py', 'dflash.py'):                      # the reviewed DSpark sources: checked, kept for the V4 reference
        upstream.read_pinned(os.path.join(options.dspark_src, name), pins[name][1], pins[name][0])
    zlab = upstream.load_pinned_module(os.path.join(options.zlab, 'model.py'), pins['zlab_model.py'][1], 'zlab_model',
                                       expected_bytes=pins['zlab_model.py'][0])
    for directory, name in ((options.dflash2, 'dflash2.safetensors'), (options.dspark, 'dspark.safetensors')):
        with loading(directory):
            upstream.verify_checkpoint(directory, {'model.safetensors': pins[name][1]})
    checks['pins'] = True
    checks['bundle'] = bool(bundle.verify_bundle(options.bundle))
    checks['transformers_pin'] = bool(options.transformers_version) and transformers.__version__ == options.transformers_version
    try:
        fla_version = importlib.metadata.version('flash-linear-attention')
    except importlib.metadata.PackageNotFoundError:
        fla_version = None
    checks['fla_pin'] = bool(options.fla_version) and fla_version == options.fla_version
    checks['memory_caps'] = options.cap_gib is not None
    total = torch.cuda.get_device_properties(0).total_memory / GIB
    torch.cuda.set_per_process_memory_fraction(min(1.0, (options.cap_gib or 96.0) / total))
    with loading(options.target):
        model = AutoModelForCausalLM.from_pretrained(options.target, dtype=torch.bfloat16, device_map='cuda').eval()
    first_linear = next(i for i, kind in enumerate(model.config.layer_types) if kind == 'linear_attention')
    modeling = sys.modules[type(model.model.layers[first_linear].linear_attn).__module__]
    checks['fast_gdn'] = importlib.util.find_spec('fla') is not None and target_module.uses_fla_kernel(modeling.torch_chunk_gated_delta_rule)
    embed, head = model.get_input_embeddings().weight, model.lm_head.weight
    hf = target_module.HFTarget(model, taps, 'cuda')
    group_runner = target_module.GroupRunner(hf, options.chunk, torch.bfloat16, 'cuda')
    records = list(bundle.read_bundle(options.bundle))

    with loading(options.dflash2):
        control = zlab.DFlash2DraftModel.from_pretrained(options.dflash2, dtype=torch.bfloat16).to('cuda').eval()
    control_config = read_json(os.path.join(options.dflash2, 'config.json'))
    for name, ok in upstream.check_draft_config(control_config, taps, ids=list(control.target_layer_ids)).items():
        checks['control_' + name] = ok
    mask = int(control.mask_token_id)
    checks['mask_token'] = mask == 248070
    scale = float(zlab._draft_value(control.config, 'input_embedding_scale', 1.0))
    dflash2_backbone = drafters.UpstreamBackbone(control, embed, lambda: zlab._make_cache(control.config), zlab._crop_to, scale)
    proposer = drafters.UpstreamProposer(control, model.lm_head)

    dspark_config = read_json(os.path.join(options.dspark, 'config.json'))
    for name, ok in upstream.check_draft_config(dspark_config, taps, allow_unnamed=True).items():
        checks['dspark_' + name] = ok
    with loading(options.dspark):
        dspark_tensors = load_file(os.path.join(options.dspark, 'model.safetensors'), device='cuda')
        dspark_cpu = load_file(os.path.join(options.dspark, 'model.safetensors'), device='cpu')
    dspark_model = d2.Dflash2(d2.real_dspark_config()).to(torch.bfloat16).to('cuda').eval()
    skipped = tuple(name for name in dspark_tensors if name.startswith(('confidence_head', 'markov_head')))
    drafters.load_port_state(dspark_model, dspark_tensors, ignore=skipped)
    predecessor, successor = dspark_tensors['markov_head.markov_w1.weight'], dspark_tensors['markov_head.markov_w2.weight']
    dspark_backbone = drafters.PortBackbone(dspark_model, embed)

    # the branch self-check on the real cache (the tiny-model tests vouch for the logic, this for the real model)
    selfcheck, selfcheck_stats = selfcheck_gate(hf, records)
    checks['branch_selfcheck'] = selfcheck == 'PASS'
    details['branch_selfcheck'] = selfcheck_stats

    # V4: the port on the GPU against the repository's CPU reference, same features, same noise rows
    class CpuWeights(object):
        def tensor(self, name):
            return dspark_cpu[name]
    reference_box = [CPUBackbone(CpuWeights(), dspark_config)]       # a box, so it can be emptied after V4

    def gpu_block(view, noise_ids, start, window):
        return dspark_backbone.block_hidden(view, noise_ids, start, window)

    def cpu_block(view, noise_ids, start, window):
        low = max(0, start - window)
        rows = view.rows(low, start).to('cpu', torch.bfloat16)
        width = rows.shape[-1] // len(taps)
        features = dict((layer, rows[:, i * width:(i + 1) * width][None].contiguous()) for i, layer in enumerate(taps))
        noise = embed[torch.as_tensor(noise_ids, device=embed.device)].to('cpu', torch.bfloat16)[None]
        return reference_box[0].forward(features, noise, context_start=low)[0]

    env = V4Env(gpu_block, cpu_block, head, predecessor, successor, head.cpu(), predecessor.cpu(), successor.cpu(), mask)
    v4, v4_counts = v4_gate(records, group_runner, WIDTH, env)
    details['v4'] = v4_counts
    dspark_cpu.clear()                                           # the CPU copies are only for V4
    reference_box.clear()
    env = None
    gc.collect()
    torch.cuda.empty_cache()

    def make_arm(spec):
        if spec['drafter'] == 'dflash2':
            return WalkerArm(drafters.Dflash2Walker(dflash2_backbone, head, proposer, mask, spec['proposals'], spec['window']))
        return WalkerArm(drafters.DSparkWalker(dspark_backbone, head, predecessor, successor, mask, spec['proposals'], spec['window']))

    def release():
        gc.collect()
        torch.cuda.empty_cache()

    guard = Guard(options.trip_file, options.deadline, time.time, read_avail_gib, options.floor_gib,
                  peak_gib=lambda: torch.cuda.max_memory_allocated() / GIB, cap_gib=options.cap_gib,
                  heartbeat_file=options.heartbeat_file, release=release)
    details['checks'] = dict((name, bool(ok)) for name, ok in checks.items())
    write_phase(options.phase_file, 'running')
    return group_runner, make_arm, guard, dict(V0=gate_v0(checks), V4=v4, _details=details)


def read_json(path):  # pragma: no cover - the real environment only
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def write_line(path, entry):
    with open(path, 'a', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(entry, sort_keys=True) + '\n')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--deadline', type=float, required=True, help='epoch seconds; the run stops cleanly before this')
    parser.add_argument('--trip-file', required=True, help='the watchdog writes its reason here')
    parser.add_argument('--heartbeat-file', required=True,
                        help='the watchdog rewrites this every poll; missing or older than %d s stops the run (fail closed)' % HEARTBEAT_MAX_AGE)
    parser.add_argument('--phase-file')
    parser.add_argument('--target')
    parser.add_argument('--dflash2')
    parser.add_argument('--dspark')
    parser.add_argument('--zlab', help='directory holding the pinned z-lab model.py')
    parser.add_argument('--dspark-src', help='directory holding the pinned DSpark dspark.py and dflash.py')
    parser.add_argument('--pins', help='pins file (JSON)')
    parser.add_argument('--transformers-version', help='the pinned transformers version (V0 refuses any other)')
    parser.add_argument('--fla-version', help='the pinned flash-linear-attention version (V0 refuses any other)')
    parser.add_argument('--arms', help='comma separated arm names (default: all, in the registered order)')
    parser.add_argument('--canary', action='store_true', help='3 shortest turns and the longest, core arms only, into <out>/canary')
    parser.add_argument('--load-only', action='store_true',
                        help='load everything, run V0 / V4 and the branch self-check, log MemFree / MemAvailable while loading, run no turn')
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
        options.memory_log = store.path('load-memory.jsonl')
        with open(store.path('meta.jsonl'), encoding='utf-8') as handle:
            meta_rows = [json.loads(line) for line in handle if line.strip()]
        arms = CORE if options.canary else (options.arms.split(',') if options.arms else None)
        plan = make_plan(meta_rows, options.seed, arms)
        store.write('plan.json', plan)
        group_runner, make_arm, guard, gates = environment(options, say)
        if options.load_only:
            say('loaded: %s' % ' '.join('%s=%s' % (name, gates[name]) for name in sorted(gates) if not name.startswith('_')))
            return 0
        if gates.get('V0') != 'PASS' or gates.get('V4') != 'PASS':
            finalize(store, options.bundle, guard, gates)          # the record of why, before any turn is spent
            say('stopped before the run: V0=%s V4=%s' % (gates.get('V0'), gates.get('V4')))
            return 4
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
