"""A closed-loop simulation of the Lever N scheduler on the production shape (stdlib only; docs/tp4-sched-1.md section 6). Every number it prints is an ESTIMATE.

    python3 -B scripts/ci/sched_sim.py                       the calibration check, then every policy of the comparison (markdown tables)
    python3 -B scripts/ci/sched_sim.py --seeds 8 --hours 2   more seeds, longer runs
    python3 -B scripts/ci/sched_sim.py --json                the same, as JSON

WHAT IT MODELS. Eight agents in a closed loop: a tool wait, a returning turn (a SHORT request: a tail of about 3k tokens to prefill at a long context, then an engine rebind),
a decode of a few hundred tokens beside the other agents, and again. Now and then a cold long prompt instead. One device serves one step at a time, as the engine does (it is
synchronous): a prefill step, or a decode round of every live seat. WHAT DECIDES the next step is the shipped code: levern_policy.Alternator (the share, the deadline governor,
the decode-gap floor, the short pacing and its guard, the adaptive few-decoder policy), levern_policy.step_budget (the chunking arithmetic) and the flag parsers (config,
merged_config, adaptive_config, sched_config), driven by a simulated clock. What the sim re-implements is only what vLLM's scheduler and the lifecycle do around it: the
waiting queue, the shortest returning turn first (the short lane), a long parked while a short runs, and a free seat for a request.

THE NUMBERS BEHIND IT (the production evidence of the task, the card logs of docs/tp4-sched-1.md, and the repository's own timing model):
  chunk       a 2,048-token chunk at context p costs 420 ms + 2.5 ms per 1,000 tokens of p (levern_policy.StepTimes' seed, fitted to the 32k and 128k card runs), times a
              scale the calibration sets (the production stack prefills faster than the stack that fit was made on: oneq attention, the 13x10 grid)
  step        an intermediate prefill step costs its chunks plus 5 ms (p90 30 ms) of scheduler, sample and update, measured on 692 steps of a card log; a final step also pays the
              admission: the engine rebind, 541 ms at p50 and 721 ms at p90 in production (lognormal), less what QWEN_FAST_ADMISSION_LEAN removes
  round       a decode round of k live seats costs the measured step plus the host gap: 105, 74, 83, 116, 136, 95, 95, 100 ms for k = 1..8 (the production table's steps 90, 54, 63,
              96, 116, 73-76 and the 15-20 ms host gap of the NP1 note), 72 ms for one seat with the lone-user padded round (np1-fastlane); a round commits 3.7, 6.8, 5.7, 5.7, 6.4,
              5.0, 5.0, 5.0 tokens a seat (the table's rates times its step plus gap)
  switch      the first round after a final step costs the same as any other (measured: 77-80 ms against 74-75 at three and four seats)

WHAT IT CANNOT SAY. It does not know the agents' real tool times, answer lengths or context sizes: those are drawn from distributions chosen so that the BASELINE run
reproduces the production evidence (58% of wall time in prefill steps, 36% in decode rounds, 6% other; the per-seat rates by live users). The comparisons between policies are
then the policies', but a work-conserving system moves delay between a returning turn's wait and the decoders' gaps without making any of it disappear: only the lean
admission removes device time. Read every table that way.
"""

import argparse
import heapq
import json
import math
import random
import statistics
import sys

import levern_policy as policy

CHUNK = policy.CHUNK
SEATS = 8
ROUND_MS = {1: 105.0, 2: 74.0, 3: 83.0, 4: 116.0, 5: 136.0, 6: 95.0, 7: 95.0, 8: 100.0}
TOKENS_PER_ROUND = {1: 3.7, 2: 6.8, 3: 5.7, 4: 5.7, 5: 6.4, 6: 5.0, 7: 5.0, 8: 5.0}
LONE_ROUND_MS_NP1, LONE_TOKENS_NP1 = 72.0, 6.8
SHORT_TOKENS = 16384
PARK_MS = 40.0                  # a long parked to host and brought back around a short: the card log's 22.5 ms out plus the restore in
DECISION_MS = 3.0               # the host time of one scheduling decision outside the steps' own (counted in "other")
PRODUCTION = {'QWEN_FAST_LEVER_N': '1', 'QWEN_PREFIX_REUSE': '1', 'QWEN_FAST_LEVERN_STEP_TOKENS': '2048', 'QWEN_FAST_LEVERN_SOLO_STEP_TOKENS': '16384',
              'QWEN_FAST_LEVERN_PREFILL_SHARE': '0.5', 'QWEN_FAST_LEVERN_TTFT_TARGET_S': '180', 'QWEN_FAST_LEVERN_MAX_DECODE_GAP_S': '8',
              'QWEN_FAST_LEVERN_SHORT_TOKENS': '16384', 'QWEN_FAST_LEVERN_PARK': 'host', 'QWEN_FAST_LEVERN_PARK_SLOTS': '1', 'QWEN_FAST_LEVERN_MAX_PARK_S': '30'}


def lognormal(p50, p90):
    mu, sigma = math.log(p50), (math.log(p90) - math.log(p50)) / 1.2816
    return lambda rng: math.exp(rng.gauss(mu, sigma))


class Workload(object):
    """The agents (see the module docstring): how long a tool takes, how big a returning turn is, how long the answer, how often a cold long prompt, how fast the stack prefills,
    what an admission costs, and whether the lone user takes the padded round (np1)."""

    def __init__(self, agents=8, think_s=6.0, tail=(3000.0, 5500.0), context=(40000, 200000), answer=(350.0, 900.0), cold_probability=0.04, cold_tokens=(32768, 128000),
                 prefill_scale=0.8, rebind_ms=(541.0, 721.0), step_overhead_ms=(5.0, 30.0), np1=False, lean_saves_ms=0.0):
        self.agents, self.think_s, self.tail, self.context, self.answer = agents, think_s, tail, context, answer
        self.cold_probability, self.cold_tokens, self.prefill_scale = cold_probability, cold_tokens, prefill_scale
        self.rebind_ms, self.step_overhead_ms, self.np1, self.lean_saves_ms = rebind_ms, step_overhead_ms, np1, lean_saves_ms

    def replace(self, **changes):
        values = dict(self.__dict__)
        values.update(changes)
        return Workload(**values)


class Turn(object):
    __slots__ = ('agent', 'arrival', 'prompt', 'computed', 'short', 'answer', 'first_token', 'last_token', 'tokens', 'parked', 'last_round', 'fresh')

    def __init__(self, agent, arrival, prompt, start, answer):
        self.agent, self.arrival, self.prompt, self.computed, self.answer = agent, arrival, prompt, start, answer
        self.short = None                   # fixed when the request is admitted: what is left after its hit fits QWEN_FAST_LEVERN_SHORT_TOKENS
        self.first_token = self.last_token = self.last_round = None
        self.tokens, self.parked, self.fresh = 0.0, False, True


class Sim(object):
    def __init__(self, env, workload, seed=1, hours=2.0):
        self.workload, self.rng, self.horizon = workload, random.Random(seed), hours * 3600.0
        env = dict(env)
        self.cfg = policy.config(env)
        self.merged = policy.merged_config(env) if env.get('QWEN_PREFIX_REUSE') == '1' else None
        self.t = 0.0
        self.alt = policy.Alternator(self.cfg, clock=lambda: self.t, merged=self.merged, wall=lambda: self.t, adaptive=policy.adaptive_config(env), sched=policy.sched_config(env))
        self.alt.build_mode = None                  # the governor's fixed BUILD_MS (production's 'learned' is an EWMA of the same figure)
        self.times = policy.StepTimes()
        self.parking = self.merged is not None and self.merged.park == 'host'
        self.rebind = lognormal(*workload.rebind_ms)
        self.overhead = lognormal(*workload.step_overhead_ms)
        self.wake = [(self.rng.uniform(0.0, workload.think_s * 2.0), agent) for agent in range(workload.agents)]
        heapq.heapify(self.wake)
        self.waiting, self.partials, self.decoding, self.done = [], [], [], []
        self.device = dict(prefill=0.0, decode=0.0, other=0.0)
        self.live_time, self.live_tokens, self.gaps = {}, {}, []

    # ------------------------------------------------------------------ the workload
    def new_turn(self, agent, arrival):
        w, rng = self.workload, self.rng
        answer = int(min(3000, max(30, lognormal(*w.answer)(rng))))
        if rng.random() < w.cold_probability:
            prompt = int(rng.uniform(*w.cold_tokens)) // 64 * 64
            return Turn(agent, arrival, prompt, 0, answer)
        tail = int(min(16000, max(1000, lognormal(*w.tail)(rng))))
        context = int(rng.uniform(*w.context)) // CHUNK * CHUNK            # the hit: a 2,048-token boundary
        return Turn(agent, arrival, context + tail, context, answer)

    def wake_due(self):
        while self.wake and self.wake[0][0] <= self.t:
            when, agent = heapq.heappop(self.wake)
            self.waiting.append(self.new_turn(agent, when))

    # ------------------------------------------------------------------ the device
    def live(self):
        return len(self.decoding)

    def seats_free(self):
        return SEATS - len(self.decoding) - len(self.partials)

    def advance(self, ms, kind, tokens=0.0):
        seconds = ms / 1000.0
        k = self.live()
        if k:
            self.live_time[k] = self.live_time.get(k, 0.0) + seconds
            self.live_tokens[k] = self.live_tokens.get(k, 0.0) + tokens
        self.device[kind] += seconds
        self.t += seconds

    def round_cost(self, k):
        if k == 1 and self.workload.np1:
            return LONE_ROUND_MS_NP1, LONE_TOKENS_NP1
        return ROUND_MS[min(k, SEATS)], TOKENS_PER_ROUND[min(k, SEATS)]

    def round(self):
        k = self.live()
        base, mean = self.round_cost(k)
        delivered = 0.0
        for turn in self.decoding:
            got = min(max(1.0, mean * self.rng.uniform(0.75, 1.25)), turn.answer - turn.tokens)
            turn.tokens += got
            delivered += got
            if turn.last_round is not None:
                self.gaps.append(self.t - turn.last_round)
        self.advance(base * (1.0 + self.rng.uniform(-0.04, 0.04)), 'decode', delivered)
        for turn in list(self.decoding):
            turn.last_round = self.t
            if turn.tokens >= turn.answer - 1e-9:
                turn.last_token = self.t
                self.decoding.remove(turn)
                self.done.append(turn)
                heapq.heappush(self.wake, (self.t + self.rng.expovariate(1.0 / self.workload.think_s), turn.agent))

    def chunk_ms(self, start, tokens):
        return sum(self.workload.prefill_scale * policy.StepTimes.seed(start + CHUNK * index) for index in range(max(1, -(-tokens // CHUNK))))

    # ------------------------------------------------------------------ the prefill pass (levern_scheduler.plan_pass, reduced)
    def classify(self, turn):
        turn.short = self.parking and turn.prompt - turn.computed <= (self.merged.short_tokens if self.merged is not None else SHORT_TOKENS)

    def candidate(self):
        """(turn, fresh, parks): the request the pass advances."""
        live_short = [turn for turn in self.partials if turn.short]
        if live_short:
            return min(live_short, key=lambda turn: turn.prompt - turn.computed), False, None
        shorts = []
        if self.parking:
            for turn in self.waiting:
                self.classify(turn)
            shorts = sorted((turn for turn in self.waiting if turn.short), key=lambda turn: (turn.prompt - turn.computed, turn.arrival))
        if shorts and self.seats_free() > 0 and len(self.partials) <= self.merged.park_slots:
            parked = self.partials[0] if self.partials else None
            if parked is None or self.park_age(parked) < self.merged.max_park_s:
                return shorts[0], True, parked
        if self.partials:
            return self.partials[0], False, None
        if self.waiting and self.seats_free() > 0:
            return self.waiting[0], True, None
        return None, False, None

    def park_age(self, turn):
        return getattr(self, 'parked_for', {}).get(id(turn), 0.0)

    def prefill_pass(self):
        """Run one prefill step. -> (ran, short)"""
        turn, fresh, parks = self.candidate()
        if turn is None:
            return False, False
        decodes = self.live()
        extra = 0.0
        if fresh:
            self.waiting.remove(turn)
            self.classify(turn)
            self.partials.append(turn)
            if parks is not None:
                parks.parked = True
                extra += PARK_MS
        elif turn.parked:
            turn.parked = False
            extra += PARK_MS
        budget = policy.step_budget(turn.prompt, turn.computed, decoding=decodes > 0, others_waiting=len(self.waiting), step=self.cfg.step, solo=self.cfg.solo)
        final = turn.computed + budget >= turn.prompt
        ms = self.chunk_ms(turn.computed, budget) + self.overhead(self.rng) + extra
        if final:
            ms += max(80.0, self.rebind(self.rng) - self.workload.lean_saves_ms)
        start = turn.computed
        self.advance(ms, 'prefill')
        turn.computed += budget
        if parks is not None:
            self.parked_for = getattr(self, 'parked_for', {})
            self.parked_for[id(parks)] = self.parked_for.get(id(parks), 0.0) + ms / 1000.0
        if final:
            self.partials.remove(turn)
            turn.first_token, turn.tokens, turn.last_round = self.t, 1.0, self.t
            self.decoding.append(turn)
        else:
            self.times.observe(start, budget, ms)
        return True, bool(turn.short)

    def pending_hint(self):
        hint = []
        for turn in self.partials + self.waiting:
            if turn.short is None:
                self.classify(turn)
            if not turn.short:
                hint.append((turn.arrival, self.times.remaining_ms(turn.prompt, turn.computed, self.cfg.step)))
        return hint

    def short_wait(self):
        waits = [self.t - turn.arrival for turn in self.waiting if turn.short]
        return max(waits) if waits else 0.0

    def run(self):
        alt = self.alt
        while self.t < self.horizon:
            self.wake_due()
            decodes = self.live()
            pending = bool(self.partials) or bool(self.waiting)
            if self.merged is not None:
                alt.pending_hint = self.pending_hint() if pending else []
                if alt.sched is not None and alt.sched.short_wait_s:
                    alt.short_wait_s = self.short_wait() if pending else 0.0
            decision = alt.begin(decodes, pending)
            if decodes or pending:
                self.advance(DECISION_MS, 'other')
            if decision.kind == 'prefill' and pending:
                ran, short = self.prefill_pass()
                if ran:
                    alt.end('prefill', short=short)
                    continue
            if decodes:
                self.round()
                alt.end('decode')
                continue
            if not self.wake:
                break
            idle = max(0.0, self.wake[0][0] - self.t)
            self.device['other'] += idle
            self.t += idle
            alt.end('decode')
        return self.summary()

    # ------------------------------------------------------------------ results
    def summary(self):
        total = sum(self.device.values()) or 1.0
        turns = [turn for turn in self.done if turn.first_token is not None]
        returning = [turn for turn in turns if turn.short]
        ttft = sorted(turn.first_token - turn.arrival for turn in returning)
        cold = sorted(turn.first_token - turn.arrival for turn in turns if not turn.short)
        rates = sorted(turn.answer / max(1e-6, turn.last_token - turn.first_token) for turn in turns if turn.answer >= 100)
        per_seat = {k: self.live_tokens[k] / (self.live_time[k] * k) for k in sorted(self.live_time) if self.live_time[k] > 60.0}
        gaps = sorted(self.gaps)

        def pct(values, q):
            return values[min(len(values) - 1, int(len(values) * q))] if values else float('nan')

        decoded = sum(self.live_tokens.values())
        return dict(prefill_share=self.device['prefill'] / total, decode_share=self.device['decode'] / total, other_share=self.device['other'] / total,
                    turns=len(turns), turns_per_min=len(turns) / (self.t / 60.0), returning=len(returning), ttft_p50=pct(ttft, 0.5), ttft_p90=pct(ttft, 0.9),
                    ttft_mean=statistics.mean(ttft) if ttft else float('nan'), cold=len(cold), cold_ttft_p50=pct(cold, 0.5), cold_ttft_p90=pct(cold, 0.9), seat_rate=statistics.mean(rates) if rates else float('nan'), seat_rate_p10=pct(rates, 0.1),
                    gap_p50=pct(gaps, 0.5), gap_p90=pct(gaps, 0.9), gap_p99=pct(gaps, 0.99), gap_max=gaps[-1] if gaps else float('nan'),
                    agg_tok_s=decoded / self.t, per_seat=per_seat, hours=self.t / 3600.0)


def run(env, workload, seeds=4, hours=2.0):
    """The mean of `seeds` seeded runs of `hours` hours, with the seeds' spread of the figures that matter."""
    rows = [Sim(env, workload, seed=seed, hours=hours).run() for seed in range(1, seeds + 1)]
    out = {key: statistics.mean(row[key] for row in rows) for key in rows[0] if key != 'per_seat'}
    out['spread'] = {key: (min(row[key] for row in rows), max(row[key] for row in rows)) for key in ('prefill_share', 'decode_share', 'seat_rate', 'ttft_p90', 'turns_per_min')}
    ks = sorted({k for row in rows for k in row['per_seat']})
    out['per_seat'] = {k: statistics.mean(row['per_seat'][k] for row in rows if k in row['per_seat']) for k in ks}
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------
# The calibration and the comparison

CALIBRATED = Workload(agents=10, think_s=10.0, prefill_scale=0.75, answer=(350.0, 910.0), cold_probability=0.03)
# What QWEN_FAST_ADMISSION_LEAN removes from a returning turn's stall (ms), from the segments of the audited card run (docs/tp4-sched-1.md section 4): the backstop's reading
# (about 70), the slot adoption's conv-slice readback (30 to 145: the class of rebinds that compiled no programs to the class that compiled three), the 'dram after engine' line's
# reading (about 66) and the memory ledger's op=rebind bracket (about 62 before, 65 after). The class of rebinds that production's 541 ms resembles is the second: 410. The first, 290.
LEAN_SAVES_MS = 410.0
LEAN_SAVES_LOW_MS = 290.0
# QWEN_FAST_ADMISSION_DIAG_TRIM=1, an existing flag the ship-4 profile does not set: the ledger's two prefill points (after the route line, about 100 ms in the audited log) once.
DIAG_TRIM_SAVES_MS = 100.0
# The quiet hours: three agents, a cold long prompt now and then (a third of the turns), almost nobody decoding when it arrives.
QUIET = Workload(agents=3, think_s=25.0, prefill_scale=0.75, answer=(400.0, 1040.0), cold_probability=0.35, cold_tokens=(65536, 253952), tail=(2500.0, 4500.0))
LONE = Workload(agents=2, think_s=40.0, prefill_scale=0.75, answer=(500.0, 1300.0), cold_probability=0.25, cold_tokens=(65536, 200000), tail=(2500.0, 4500.0))


def variants():
    """(section, label, env changes, workload changes): the comparison the note's section 6 prints. The first row of each section is its control."""
    return [
        ('production', 'production (ship-4 pacing)', {}, {}),
        ('production', 'LEAN admission (-%d ms a returning turn)' % LEAN_SAVES_MS, {}, dict(lean_saves_ms=LEAN_SAVES_MS)),
        ('production', 'LEAN admission, the low estimate (-%d ms)' % LEAN_SAVES_LOW_MS, {}, dict(lean_saves_ms=LEAN_SAVES_LOW_MS)),
        ('production', 'DIAG_TRIM (existing flag; -%d ms)' % DIAG_TRIM_SAVES_MS, {}, dict(lean_saves_ms=DIAG_TRIM_SAVES_MS)),
        ('production', 'LEAN + DIAG_TRIM (-%d ms)' % (LEAN_SAVES_MS + DIAG_TRIM_SAVES_MS), {}, dict(lean_saves_ms=LEAN_SAVES_MS + DIAG_TRIM_SAVES_MS)),
        ('production', 'owed cap 1500 ms', {'QWEN_FAST_LEVERN_OWED_CAP_MS': '1500'}, {}),
        ('production', 'short share 0.80', {'QWEN_FAST_LEVERN_SHORT_SHARE': '0.8'}, {}),
        ('production', 'short share 0.65', {'QWEN_FAST_LEVERN_SHORT_SHARE': '0.65'}, {}),
        ('production', 'short share 0.50', {'QWEN_FAST_LEVERN_SHORT_SHARE': '0.5'}, {}),
        ('production', 'short share 0.65 + guard 5 s', {'QWEN_FAST_LEVERN_SHORT_SHARE': '0.65', 'QWEN_FAST_LEVERN_SHORT_WAIT_S': '5'}, {}),
        ('production', 'short share 0.50 + guard 5 s', {'QWEN_FAST_LEVERN_SHORT_SHARE': '0.5', 'QWEN_FAST_LEVERN_SHORT_WAIT_S': '5'}, {}),
        ('production', 'solo step 8192', {'QWEN_FAST_LEVERN_SOLO_STEP_TOKENS': '8192'}, {}),
        ('production', 'adaptive (<=2 decoders)', {'QWEN_FAST_LEVERN_ADAPTIVE': '1'}, {}),
        ('production', 'LEAN + owed cap + short share 0.65 + guard 5 s',
         {'QWEN_FAST_LEVERN_OWED_CAP_MS': '1500', 'QWEN_FAST_LEVERN_SHORT_SHARE': '0.65', 'QWEN_FAST_LEVERN_SHORT_WAIT_S': '5'}, dict(lean_saves_ms=LEAN_SAVES_MS)),
        ('quiet', 'production', {}, {}),
        ('quiet', 'adaptive (<=2 decoders)', {'QWEN_FAST_LEVERN_ADAPTIVE': '1'}, {}),
        ('quiet', 'owed cap 1500 ms', {'QWEN_FAST_LEVERN_OWED_CAP_MS': '1500'}, {}),
        ('quiet', 'solo step 8192', {'QWEN_FAST_LEVERN_SOLO_STEP_TOKENS': '8192'}, {}),
        ('lone', 'production', {}, {}),
        ('lone', 'lone-user padded round (np1)', {}, dict(np1=True)),
        ('lone', 'np1 + owed cap 1500 ms', {'QWEN_FAST_LEVERN_OWED_CAP_MS': '1500'}, dict(np1=True)),
        ('lone', 'np1 + adaptive', {'QWEN_FAST_LEVERN_ADAPTIVE': '1'}, dict(np1=True)),
    ]


BASES = {'production': CALIBRATED, 'quiet': QUIET, 'lone': LONE}


def compare(seeds=4, hours=2.0, only=None):
    """-> {section: [(label, result)]} for variants(), each a seeded mean."""
    out = {}
    for section, label, env_changes, workload_changes in variants():
        if only and section not in only:
            continue
        env = dict(PRODUCTION)
        env.update(env_changes)
        out.setdefault(section, []).append((label, run(env, BASES[section].replace(**workload_changes), seeds=seeds, hours=hours)))
    return out


def tables(results):
    lines = []
    titles = {'production': 'Production shape (10 agents on 8 seats, returning turns of ~3k tokens, 3% cold long prompts)',
              'quiet': 'Quiet hours (3 agents, a third of the turns a cold 64k-254k prompt)', 'lone': 'A lone decoding user (2 agents, 25% cold 64k-200k prompts)'}
    for section, rows in results.items():
        base = rows[0][1]
        lines.append('**%s**' % titles[section])
        lines.append('')
        lines.append('| policy | prefill share | decode share | seat rate (tok/s) | rate at 5-7 live | returning TTFT p50 / p90 (s) | cold TTFT p50 / p90 (s) | gap p90 / max (s) | turns/min |')
        lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
        for label, row in rows:
            high = [row['per_seat'][k] for k in (5, 6, 7) if k in row['per_seat']]
            lines.append('| %s | %.1f%% | %.1f%% | %.1f (%+.0f%%) | %s | %.1f / %.1f | %s | %.2f / %.1f | %.1f (%+.0f%%) |' % (
                label, 100 * row['prefill_share'], 100 * row['decode_share'], row['seat_rate'], 100 * (row['seat_rate'] / base['seat_rate'] - 1),
                '%.1f' % statistics.mean(high) if high else '-', row['ttft_p50'], row['ttft_p90'],
                '%.0f / %.0f' % (row['cold_ttft_p50'], row['cold_ttft_p90']) if row['cold'] else '-', row['gap_p90'], row['gap_max'], row['turns_per_min'],
                100 * (row['turns_per_min'] / base['turns_per_min'] - 1)))
        lines.append('')
    return chr(10).join(lines)


def calibration_table(row):
    target = {1: 17.8, 2: 37.3, 3: 28.8, 4: 21.6, 5: 16.7, 6: 15.0, 7: 15.0}
    lines = ['| live users | production tok/s per user | simulated |', '| ---: | ---: | ---: |']
    for k in range(1, 8):
        lines.append('| %d%s | %.1f | %s |' % (k, ' (6-7: 14-16)' if k == 6 else '', target[k], '%.1f' % row['per_seat'][k] if k in row['per_seat'] else '-'))
    lines.append('')
    lines.append('| share of wall time | production | simulated |')
    lines.append('| --- | ---: | ---: |')
    lines.append('| prefill steps | 58%% | %.1f%% |' % (100 * row['prefill_share']))
    lines.append('| decode rounds | 36%% | %.1f%% |' % (100 * row['decode_share']))
    lines.append('| other | 6%% | %.1f%% |' % (100 * row['other_share']))
    return chr(10).join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    parser.add_argument('--seeds', type=int, default=4)
    parser.add_argument('--hours', type=float, default=2.0)
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--only', nargs='*', default=None, help='sections: production quiet lone')
    arguments = parser.parse_args(argv)
    results = compare(arguments.seeds, arguments.hours, arguments.only)
    if arguments.json:
        sys.stdout.write(json.dumps({section: [(label, row) for label, row in rows] for section, rows in results.items()}, indent=1, default=str) + chr(10))
        return 0
    if 'production' in results:
        sys.stdout.write('**Calibration of the baseline**' + chr(10) + chr(10) + calibration_table(results['production'][0][1]) + chr(10) + chr(10))
    sys.stdout.write(tables(results) + chr(10))
    return 0


if __name__ == '__main__':
    sys.exit(main())
