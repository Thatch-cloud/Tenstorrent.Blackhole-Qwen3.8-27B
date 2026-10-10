#!/usr/bin/env python3
"""Prefill/decode disaggregation model for Qwen3.8-27B on four Blackhole p150a cards (stdlib only).

Compares, on the SAME per-seat turn scripts (paired):
  tp4    today's production: one TP4 pool, Lever N (2048-token steps interleaved with decode rounds, prefill share f=0.5
         while decoders run, short lane <=16384 tokens paced at R=1, solo 16384-token steps when nobody decodes, the 180 s
         TTFT governor and the 8 s decode-gap floor), engine reuse (rebind instead of build), sticky prefix reuse.
  disagg 2P+2D: a TP2 prefill pair (back-to-back solo steps, shorts first, longs FIFO) and a TP2 decode pair (rounds only),
         KV + GDN + conv + drafter state moved between the pairs over a link of bandwidth BW. Prefix KV of a session
         lives on the decode pair (it decoded there); a hit's prefix is pulled to the prefill pair, the tail prefilled, the
         whole context pushed back (policy 'remote'), or a tail <= LOCAL_TAIL is prefilled on the decode pair itself
         under the same R=1 pacing (policy 'local', llm-d's conditional disaggregation).

Every input is marked M (measured, with its source) or E (estimate) in PARAMS_DOC; run with --doc to print it.
Outputs per run: per-seat decode tok/s (client side, from the first token), TTFT p50/p90/p99/max, committed tokens per
elapsed second (all output tokens / makespan), device busy split, hit rate. Deterministic for a given seed.

    python3 docs/tp4-disagg-model.py --mix prod --seeds 5
    python3 docs/tp4-disagg-model.py --calibrate          # the two measured TP4 shapes (parked_turns, concurrent8_skew)
    python3 docs/tp4-disagg-model.py --sweep --seeds 4    # every traffic mix x every arm (the table in docs/tp4-disagg-step1.md)
    python3 docs/tp4-disagg-model.py --doc                # every input with its source
"""
import argparse
import heapq
import math
import random
import statistics

CH = 2048

PARAMS_DOC = """
Every input, M = measured on cards (source), M-d = derived from measurements, E = estimate. TP4 = production profile
c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-traffic. v547 = run 37993848597 (the cutover G1 gate on that profile; per-step
medians parsed from its container log). Repo paths are relative to the Qwen repo.
 TP4 2048-step       0.31 s + 3.7 ms per 1k of start context, +0.05 s per step       M v547 step medians 359/467/624/748/868/992/1112/1197 ms
                                                                                   (start 0-32k ... 224-256k); 252k cold = 96 s model vs 95.6-97.8 M
 TP4 admission       +0.9 s on the final step (rebind 0.52 s + finalize)            M-d v547: governor admission cost median 536 ms; final step
                                                                                   median 1.66 s under 16k (cold build would be 2.57 s, v611/v613)
 hit restore         0.08 s GDN checkpoint host->device                            M G1 prefix gate run 36274183527: restore p50 58-83 ms for 154 MB
 TP4 decode round    live 1..8: 85/70/86/149/156/159/175/177 ms                    M v547 decode-step medians by live (1-7), X2 v544 (8-live 177 ms at 33k;
                     tokens/seat/round 2.5 (lone 4-row engine), 3.6 packed            157-161 at 4k); lone 85 ms x 2.5 = 29.4 tok/s M; packed 3.3-4.8 M;
                                                                                   v547 steady 8 users 168 tok/s agg = 21/seat -> 3.6 at 175 ms
 Lever N             f=0.5, KMAX=8, short<=16384 at R=1, solo 16384, T*=180 s, floor 8 s   M profile env; semantics docs/lever-n-prefix-merged-route.md
 TP2 prefill factor  r = 1.6 x TP4 step time (range 1.5-2.3)                        M G1 same image: TTFT 1.5/11.2/20.8 s (TP2 J3r) vs 1.0/6.8/14.7 s
                                                                                   (TP4 J3) -> 1.51-1.65; pair-era S2 131k 55-62 s vs TP4 35 s -> 1.6-1.8
 TP2 decode factor   k = 1.8 x TP4 round time (range 1.6-2.0)                       M-d G1 1.51-1.63 (J3r/J3); TP2 S2 4x131k rounds 162-168 ms M (v233-238)
                                                                                   vs TP4 one block ~81-100 ms; weights+per-user work per chip double
 KV bytes            34,816 B/token (16 layers x 4 KV heads x 256 x K,V, bf8)       config.json; bf8 = 1,088 B per 1,024-element tile
 fixed state         0.26 GB/request: GDN fp32 + conv 154 MB, drafter taps 105 MB   qwen_prefix_registry.py:89-91 (153,944,064 B); DFlash2 config
                                                                                   (5 taps x 2048 x 5120 bf16); taps can be 0 if D re-prefills 2048
 link                fabric 40 GB/s per direction (range 10-160)                    E; M: pair all-gather 84-90 GB/s on 2 links (fabric-bandwidth doc),
                                                                                   2 chip pairs in parallel; NO point-to-point op in our code today
                     host-staged 2 GB/s (range 0.3-5)                                M-d: checkpoint restore 1.9-2.7 GB/s H2D, capture 3.6-7.7 GB/s D2H;
                                                                                   cards A and B are on PCIe x4; no KV write op exists
 KV capacity         TP4 1,277,952 tokens (19,968 x 64)                             M profile; TP2 8-seat decode pair 400k E, range 150-450k: pair-era
                                                                                   c2-packed held 525k at 4 seats (9.1 GB/chip, M) with no parked engines;
                                                                                   8 parked engines cost 3.96 GB/chip at TP4 (M) and about twice that at TP2
 traffic mixes       synthetic coding-agent sessions (make_scripts): 'prod' = short sessions with sticky prefix reuse (prompt mean ~28k,
                     ~70% prefix hits), answers median 150 tokens (p90 ~540), tool gaps median 2 s (E: never measured); 'longagent',
                     'cold', 'decode' are the stress mixes. Calibrated against the two measured TP4 shapes (--calibrate).
"""

STEP_A, STEP_B = 0.31, 3.7e-6
EPS = 0.05
ADMIT = 0.90
P_FINAL = 0.40
RESTORE = 0.08
PARK = 0.10
F_SHARE = 0.5
KMAX = 8
SHORT = 16384
SOLO = 16384
TTFT_TARGET = 180.0
FLOOR = 8.0
KV_B = 34816
FIXED_B = 0.26e9
TP4_KV = 19968 * 64
JOIN = 0.0
SHORT_R1 = True

# TP4 decode: (round seconds, tokens per seat per round) by live count (v547, X2)
ROUNDS = {1: 0.085, 2: 0.070, 3: 0.086, 4: 0.149, 5: 0.156, 6: 0.159, 7: 0.175, 8: 0.177}


def tp4_round(live):
    live = max(1, min(8, live))
    return ROUNDS[live], (2.5 if live == 1 else 3.6)


def chunk_time(start, n, r=1.0):
    full = STEP_A + STEP_B * start
    t = full if n == CH else full * max(0.4, n / CH)
    return t * r


def span_time(start, n, r=1.0):
    """Time to prefill tokens [start, start+n) in 2048 chunks (the model's own boundaries)."""
    t, pos, end = 0.0, start, start + n
    while pos < end:
        boundary = (pos // CH + 1) * CH
        step = min(boundary, end) - pos
        t += chunk_time(pos, step, r)
        pos += step
    return t


# --------------------------------------------------------------------------------------------- workloads
def lognormal(rng, median, sigma, lo, hi):
    return int(min(hi, max(lo, rng.lognormvariate(math.log(median), sigma))))


def make_scripts(mix, seats, turns, seed, gap_scale=1.0):
    """Per seat a list of turns: dict(session, P, O, gap). P is the whole prompt; reuse is decided at run time."""
    rng = random.Random(seed)
    scripts = []
    for seat in range(seats):
        rows, session, P = [], 0, None
        left = 0
        for t in range(turns):
            if mix == 'prod':
                if left <= 0 or P is None or P > 230000:
                    session += 1
                    left = 1 + int(rng.expovariate(1 / 3.2))
                    P = lognormal(rng, 11000, 0.8, 2500, 120000)
                else:
                    P = P + rows[-1]['O'] + lognormal(rng, 2600, 1.1, 60, 60000)
                O = lognormal(rng, 150, 1.0, 16, 8192)
                gap = lognormal(rng, 2.0, 0.8, 0.2, 60.0) if left > 1 else lognormal(rng, 8.0, 0.8, 1.0, 120.0)
            elif mix == 'cold':        # no reuse: every turn a new long prompt (E-SIM before prefix reuse; skew-like)
                session += 1
                P = lognormal(rng, 40000, 1.0, 4000, 253920)
                O = lognormal(rng, 300, 1.0, 16, 8192)
                gap = lognormal(rng, 2.0, 0.8, 0.2, 60.0)
                left = 1
            elif mix == 'decode':      # long generations on short, reused prompts
                if left <= 0 or P is None or P > 200000:
                    session += 1
                    left = 1 + int(rng.expovariate(1 / 6.0))
                    P = lognormal(rng, 9000, 0.5, 2500, 60000)
                else:
                    P = P + rows[-1]['O'] + lognormal(rng, 800, 0.8, 60, 20000)
                O = lognormal(rng, 2000, 0.6, 200, 8192)
                gap = lognormal(rng, 2.0, 0.8, 0.2, 60.0)
            elif mix == 'longagent':   # long-context agent sessions (sessions grown to 100-250k, prefix reuse on)
                if left <= 0 or P is None or P > 240000:
                    session += 1
                    left = 1 + int(rng.expovariate(1 / 12.0))
                    P = lognormal(rng, 60000, 0.7, 8000, 200000)
                else:
                    P = P + rows[-1]['O'] + lognormal(rng, 3000, 1.1, 60, 60000)
                O = lognormal(rng, 300, 1.0, 16, 8192)
                gap = lognormal(rng, 2.0, 0.8, 0.2, 60.0)
            else:
                raise ValueError(mix)
            left -= 1
            rows.append(dict(session=(seat, session), P=min(P, 253920), O=O, gap=gap * gap_scale))
        scripts.append(rows)
    return scripts


# parked_turns as served (v612 run 37921182714, production-profile image): the server's prompt_tokens per turn, seat-major
PARKED_TURNS_PROMPTS = (4241, 8048, 10525, 16102, 19357, 6773, 11388, 13653, 24104, 3647, 10405, 26222, 41094, 7045, 11790, 23465,
                        21906, 4470, 7081, 11134, 23965, 3502, 7302, 10701, 14747, 3639, 7173, 10802, 16525, 21402, 6693, 10569,
                        14503, 21236, 3963, 11693, 13145, 21358, 3691, 7540)


def parked_turns_scripts():
    out = []
    for seat in range(8):
        out.append([dict(session=(seat, t), P=PARKED_TURNS_PROMPTS[seat * 5 + t], O=256, gap=2.0) for t in range(5)])
    return out, [0.5 * seat for seat in range(8)]


def skew_scripts():
    P = (253920, 4096, 4096, 4096, 253920, 4096, 4096, 4096)
    return [[dict(session=(s, 0), P=P[s], O=800, gap=0.0)] for s in range(8)], [0.0] * 8


# --------------------------------------------------------------------------------------------- simulation core
class Req:
    __slots__ = ('seat', 'idx', 'session', 'P', 'O', 'start', 'pos', 'arrive', 'first', 'end', 'made', 'hit', 'long',
                 'parked', 'where', 'local')

    def __init__(self, seat, idx, row, arrive):
        self.seat, self.idx, self.session, self.P, self.O = seat, idx, row['session'], row['P'], row['O']
        self.arrive, self.first, self.end, self.made = arrive, None, None, 0
        self.start = 0          # first token position still to prefill
        self.pos = 0
        self.hit = False
        self.long = False
        self.parked = False
        self.where = None
        self.local = False


class Sim:
    def __init__(self, mode, scripts, offsets=None, r=1.6, k=1.8, bw=40e9, lat=0.005, dcap=400000, policy='local',
                 local_tail=SHORT, gpu_cap=None, reuse=True):
        self.mode, self.scripts = mode, scripts
        self.offsets = offsets or [0.0] * len(scripts)
        self.r, self.k, self.bw, self.lat, self.policy, self.local_tail = r, k, bw, lat, policy, local_tail
        self.reuse = reuse
        self.cap = TP4_KV if mode == 'tp4' else dcap
        self.colo = mode in ('tp4', 'tp2rep')
        self.pr = r if mode == 'tp2rep' else 1.0
        self.now = 0.0
        self.events = []
        self.seq = 0
        self.done = []
        self.next_idx = [0] * len(scripts)
        self.resident = {}          # session -> retained prefix tokens (published, floor2048 of its last prompt), with last use
        self.last_use = {}
        self.busy = dict(prefill=0.0, decode=0.0, rebind=0.0, idle=0.0, xfer=0.0, dprefill=0.0)
        # tp4 device or the two pairs
        self.waiting = []           # requests waiting for prefill (on the prefill device)
        self.dwaiting = []          # disagg: requests prefilling locally on the decode pair (short tails)
        self.live = []              # decoding requests
        self.importq = []           # disagg: transferred requests waiting to join the decode pair
        self.inflight_long = None
        self.dev_free = {'P': True, 'D': True}
        self.link_free_at = 0.0
        self.owed = 0.0
        self.last_round_at = 0.0
        self.ideal = False

    # -- event plumbing
    def push(self, t, kind, payload=None):
        self.seq += 1
        heapq.heappush(self.events, (t, self.seq, kind, payload))

    def run(self, horizon=1e9):
        for seat in range(len(self.scripts)):
            self.push(self.offsets[seat], 'arrive', seat)
        while self.events:
            t, _, kind, payload = heapq.heappop(self.events)
            if t > horizon:
                break
            self.now = t
            getattr(self, 'on_' + kind)(payload)
            self.kick()
        return self

    def on_arrive(self, seat):
        idx = self.next_idx[seat]
        if idx >= len(self.scripts[seat]):
            return
        self.next_idx[seat] += 1
        req = Req(seat, idx, self.scripts[seat][idx], self.now)
        kept = self.resident.get(req.session, 0) if self.reuse else 0
        c0 = min(kept, ((req.P - CH) // CH) * CH) if kept else 0
        if c0 >= CH:
            req.hit, req.start = True, c0
        req.pos = req.start
        req.long = (req.P - req.start) > SHORT
        if self.mode == 'disagg' and self.policy == 'local' and req.hit and (req.P - req.start) <= self.local_tail:
            req.local = True
            self.dwaiting.append(req)
        else:
            self.waiting.append(req)

    def on_dev_done(self, payload):
        dev, fn = payload
        fn()
        self.dev_free[dev] = True

    def on_xfer_done(self, req):
        self.importq.append(req)

    def on_finish(self, req):
        pass

    # -- capacity: evict idle sessions LRU until `need` more tokens fit
    def admit_capacity(self, req):
        used = sum(self.resident.values()) + sum(r.P + r.O for r in self.live) - sum(
            self.resident.get(r.session, 0) for r in self.live)
        need = req.P + req.O - (self.resident.get(req.session, 0))
        if used + need <= self.cap:
            return True
        live_sessions = set(r.session for r in self.live) | {req.session}
        for s in sorted(self.resident, key=lambda s: self.last_use.get(s, 0)):
            if s in live_sessions:
                continue
            used -= self.resident.pop(s)
            if used + need <= self.cap:
                return True
        return used + need <= self.cap

    def finish_req(self, req):
        req.end = self.now
        self.done.append(req)
        self.live.remove(req)
        if self.reuse:
            self.resident[req.session] = (req.P // CH) * CH
            self.last_use[req.session] = self.now
        row = self.scripts[req.seat][req.idx]
        self.push(self.now + row['gap'], 'arrive', req.seat)

    # -- decode round on a device: everyone live advances
    def round_cost(self, live):
        rt, g = tp4_round(live)
        return (rt * self.k, g) if self.mode in ('disagg', 'tp2rep') else (rt, g)

    def do_round(self, dev):
        rt, g = self.round_cost(len(self.live))
        members = list(self.live)

        def done():
            for r in members:
                if r not in self.live:
                    continue
                r.made += g
                if r.made >= r.O:
                    self.finish_req(r)
            self.last_round_at = self.now
        self.busy['decode'] += rt
        self.push(self.now + rt, 'dev_done', (dev, done))
        self.dev_free[dev] = False
        return rt

    # -- one prefill step for req on dev; returns its duration
    def do_step(self, dev, req, tokens, r, then):
        n = min(tokens, req.P - req.pos)
        final = req.pos + n >= req.P
        t = span_time(req.pos, n, r) + EPS
        if req.hit and req.pos == req.start:
            t += RESTORE
        if req.parked:
            t += PARK
            req.parked = False
        if final and (self.colo or req.local):
            t += ADMIT
            self.busy['rebind'] += ADMIT
        elif final and not self.ideal:
            t += P_FINAL
        self.busy['prefill' if dev == 'P' or self.colo else 'dprefill'] += t

        def done():
            req.pos += n
            if req.pos >= req.P:
                then(req)
        self.push(self.now + t, 'dev_done', (dev, done))
        self.dev_free[dev] = False
        return t

    def pick(self, queue):
        shorts = [q for q in queue if (q.P - q.pos) <= SHORT]
        if shorts:
            return min(shorts, key=lambda q: (q.P - q.pos, q.arrive))
        return min(queue, key=lambda q: q.arrive)

    # -- the schedulers
    def kick(self):
        if self.colo:
            self.kick_tp4()
        else:
            self.kick_disagg()

    def tp4_join(self, req):
        req.first = self.now
        req.made = 1
        self.waiting.remove(req)
        if req is self.inflight_long:
            self.inflight_long = None
        self.live.append(req)
        if req.made >= req.O:
            self.finish_req(req)

    def kick_tp4(self):
        if not self.dev_free['P']:
            return
        pending = [q for q in self.waiting if self.admit_capacity(q)] if self.waiting else []
        if not pending and not self.live:
            return
        if not pending:
            self.do_round('P')
            return
        req = self.pick(pending)
        if self.inflight_long is not None and req is not self.inflight_long and self.inflight_long.pos > self.inflight_long.start:
            self.inflight_long.parked = True
        if req.long:
            self.inflight_long = req
        if not self.live:
            self.owed = 0.0
            self.do_step('P', req, SOLO, self.pr, self.tp4_join)
            return
        # decoders live: interleave
        if self.owed > 0:
            self.owed -= self.do_round('P')
            return
        overdue = self.now - self.last_round_at >= FLOOR
        if overdue:
            self.do_round('P')
            return
        t = self.do_step('P', req, CH, self.pr, self.tp4_join)
        # every step (chunk, short, final with its admission) is repaid at share f with decode rounds, at least one (RMIN = 1),
        # at most KMAX; the deadline governor lifts f to 1.0 for a long whose projected TTFT passes T* (the floor still runs)
        projected = (self.now - req.arrive) + (1.0 / F_SHARE) * span_time(req.pos, req.P - req.pos, self.pr)
        f = 1.0 if (req.long and projected > TTFT_TARGET) else F_SHARE
        rt = self.round_cost(len(self.live))[0]
        if not req.long and req.pos < req.P and SHORT_R1:
            self.owed = 1e-9            # a short's chunk steps alternate at R = 1 (merged route sec 9); its final step is repaid at f
        else:
            self.owed = 0.0 if f >= 1.0 else max(1e-9, min(t * (1 - f) / f, KMAX * rt))

    # disagg: P = prefill pair, D = decode pair, a link between them
    def xfer(self, req, tokens, then_kind):
        size = tokens * KV_B + FIXED_B
        start = max(self.now, self.link_free_at)
        dur = self.lat + size / self.bw
        self.link_free_at = start + dur
        self.busy['xfer'] += dur
        self.push(start + dur, then_kind, req)

    def p_done(self, req):
        self.waiting.remove(req)
        if req is self.inflight_long:
            self.inflight_long = None
        self.xfer(req, req.P, 'xfer_done')

    def d_local_done(self, req):
        self.dwaiting.remove(req)
        req.first = self.now
        req.made = 1
        self.live.append(req)
        if req.made >= req.O:
            self.finish_req(req)

    def kick_disagg(self):
        if self.dev_free['P'] and self.waiting:
            req = self.pick(self.waiting)
            if self.inflight_long is not None and req is not self.inflight_long and self.inflight_long.pos > self.inflight_long.start:
                self.inflight_long.parked = True
            if req.long:
                self.inflight_long = req
            if req.hit and req.pos == req.start and req.where != 'pulled':
                # pull the prefix from the decode pair first (it lives there); the P pair waits on the link
                req.where = 'pulled'
                size = req.start * KV_B + FIXED_B
                start = max(self.now, self.link_free_at)
                dur = self.lat + size / self.bw
                self.link_free_at = start + dur
                self.busy['xfer'] += dur
                self.dev_free['P'] = False
                self.push(start + dur, 'dev_done', ('P', lambda: None))
                return
            self.do_step('P', req, SOLO, self.r, self.p_done)
        if self.dev_free['D']:
            # imports first (a rebind each, device-exclusive), then local short prefills at R=1, then rounds
            ready = [q for q in self.importq if self.admit_capacity(q)]
            if ready:
                req = ready[0]
                self.importq.remove(req)
                admit = 0.0 if self.ideal else ADMIT
                self.busy['rebind'] += admit

                def joined(req=req):
                    req.first = self.now
                    req.made = 1
                    self.live.append(req)
                    if req.made >= req.O:
                        self.finish_req(req)
                self.push(self.now + admit, 'dev_done', ('D', joined))
                self.dev_free['D'] = False
                return
            local = [q for q in self.dwaiting if self.admit_capacity(q)]
            if local and (not self.live or self.owed <= 0):
                req = min(local, key=lambda q: (q.P - q.pos, q.arrive))
                self.do_step('D', req, CH if self.live else SOLO, self.r, self.d_local_done)
                self.owed = 1e-9 if self.live else 0.0
                return
            if self.live:
                self.owed -= self.do_round('D')


# --------------------------------------------------------------------------------------------- reporting
def pct(values, p):
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))]


def summarise(sim):
    done = [r for r in sim.done if r.first is not None]
    if not done:
        return {}
    makespan = max(r.end for r in done) - min(r.arrive for r in done)
    tokens = sum(r.O for r in done)
    ttft = [r.first - r.arrive for r in done]
    rates = [(r.O - 1) / (r.end - r.first) for r in done if r.O > 8 and r.end > r.first]
    dec_tok = sum(r.O - 1 for r in done if r.end > r.first)
    dec_time = sum(r.end - r.first for r in done if r.end > r.first)
    turn = [r.end - r.arrive for r in done]
    seats = len(sim.scripts)
    hits = sum(1 for r in done if r.hit)
    prompt = sum(r.P for r in done)
    cached = sum(r.start for r in done)
    return dict(turns=len(done), makespan_s=round(makespan, 1), committed_tok_s=round(tokens / makespan, 2),
                committed_tok_s_seat=round(tokens / makespan / seats, 2),
                decode_tok_s_seat_mean=round(statistics.mean(rates), 2) if rates else None,
                decode_tok_s_seat_tokw=round(dec_tok / dec_time, 2) if dec_time else None,
                ttft_p50=round(pct(ttft, 50), 2), ttft_p90=round(pct(ttft, 90), 2), ttft_p99=round(pct(ttft, 99), 2),
                ttft_max=round(max(ttft), 2), turn_mean_s=round(statistics.mean(turn), 2),
                hit_rate=round(hits / len(done), 3), cached_share=round(cached / prompt, 3) if prompt else None,
                mean_prompt=int(prompt / len(done)),
                busy={k: round(v, 1) for k, v in sim.busy.items() if v})


def run_replicas(scripts, r=1.6, k=1.8, cap=500000):
    """Two colocated TP2 pools side by side (cards M+A and B+C), each serving half the seats with Lever N at TP2 speeds."""
    half = len(scripts) // 2 or 1
    sims = [Sim('tp2rep', scripts[:half], r=r, k=k, dcap=cap).run()]
    if len(scripts) > 1:
        sims.append(Sim('tp2rep', scripts[half:], r=r, k=k, dcap=cap).run())
    merged = Sim('tp2rep', scripts, r=r, k=k, dcap=cap)
    merged.done = [q for sim in sims for q in sim.done]
    for sim in sims:
        for key, v in sim.busy.items():
            merged.busy[key] += v
    return merged


def run_pair(scripts, offsets=None, **kw):
    a = summarise(Sim('tp4', scripts, offsets).run())
    b = summarise(Sim('disagg', scripts, offsets, **kw).run())
    return a, b


def fmt(label, s):
    return ('%-34s turns %4d makespan %7.1f s | committed %6.1f tok/s (%5.2f/seat) | decode/seat mean %5.1f tokw %5.1f | '
            'TTFT p50 %6.2f p90 %6.2f p99 %6.2f max %6.2f | turn %6.2f s | hit %.2f cached %.2f P %6d' % (
                label, s['turns'], s['makespan_s'], s['committed_tok_s'], s['committed_tok_s_seat'],
                s['decode_tok_s_seat_mean'] or 0, s['decode_tok_s_seat_tokw'] or 0, s['ttft_p50'], s['ttft_p90'],
                s['ttft_p99'], s['ttft_max'], s['turn_mean_s'], s['hit_rate'], s['cached_share'] or 0, s['mean_prompt']))


def mean_of(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return statistics.mean(vals) if vals else float('nan')


SHORTS = (('commit', 'committed_tok_s'), ('dec/seat', 'decode_tok_s_seat_mean'), ('dec/seat-tokw', 'decode_tok_s_seat_tokw'),
          ('ttft50', 'ttft_p50'), ('ttft90', 'ttft_p90'), ('ttft99', 'ttft_p99'), ('ttftmax', 'ttft_max'), ('turn', 'turn_mean_s'))
KEYS = ('committed_tok_s', 'decode_tok_s_seat_mean', 'decode_tok_s_seat_tokw', 'ttft_p50', 'ttft_p90', 'ttft_p99', 'ttft_max', 'turn_mean_s')


def sweep(seeds=4, turns=60):
    """The scenario table: every mix, the central disagg assumptions and the sensitivity arms; means over seeds."""
    arms = [('2P+2D central (r1.6 k1.8 fabric40 remote cap400k)', dict(policy='remote')),
            ('2P+2D conditional: tails<=16k prefilled on D', dict(policy='local')),
            ('2P+2D optimistic (r1.5 k1.6 fabric160 cap600k)', dict(policy='remote', r=1.5, k=1.6, bw=160e9, dcap=600000)),
            ('2P+2D pessimistic (r2.0 k2.0 fabric10 cap300k)', dict(policy='remote', r=2.0, k=2.0, bw=10e9, dcap=300000)),
            ('2P+2D host-staged 2 GB/s', dict(policy='remote', bw=2e9, lat=0.05)),
            ('2P+2D ideal (r1.5 k1.6, free xfer, no cap, no admit)', dict(policy='remote', r=1.5, k=1.6, bw=1e15, lat=0.0, dcap=10 ** 9, ideal=True))]
    mixes = [('prod: 8 busy agents (gap median 2 s)', 'prod', 1.0, 8), ('prod: 8 agents, gaps x10 (median 20 s)', 'prod', 10.0, 8),
             ('prod: 1 agent alone', 'prod', 1.0, 1),
             ('longagent: 60k-240k sessions with reuse', 'longagent', 1.0, 8), ('cold: no reuse, 40k median prompts', 'cold', 1.0, 8),
             ('decode: ~2k-token answers on short prompts', 'decode', 1.0, 8)]
    global F_SHARE, ADMIT
    out = []
    for mname, mix, gs, seats in mixes:
        base, dis, alt, adm = [], {name: [] for name, _ in arms}, {0.3: [], 0.7: []}, []
        rep = []
        for seed in range(seeds):
            scripts = make_scripts(mix, seats, turns, 1000 + seed, gs)
            base.append(summarise(Sim('tp4', scripts).run()))
            for f in alt:
                F_SHARE = f
                alt[f].append(summarise(Sim('tp4', scripts).run()))
                F_SHARE = 0.5
            ADMIT = 0.3
            adm.append(summarise(Sim('tp4', scripts).run()))
            ADMIT = 0.9
            rep.append(summarise(run_replicas(scripts)))
            for name, kw in arms:
                kw = dict(kw)
                ideal = kw.pop('ideal', False)
                sim = Sim('disagg', scripts, **kw)
                sim.ideal = ideal
                dis[name].append(summarise(sim.run()))
        out.append((mname, base, dis, alt))
        line = lambda label, rows: '  %-55s ' % label + ' '.join('%s %6.1f' % (short, mean_of(rows, k)) for short, k in SHORTS)
        busy = base[0]['busy']
        total = sum(busy.values())
        print('%s   [TP4 device split seed0: %s]' % (mname, ', '.join('%s %.0f%%' % (k, 100 * v / total) for k, v in busy.items())))
        print(line('TP4 pool today (Lever N f=0.5, ER, prefix)', base))
        for f in alt:
            print(line('TP4 pool, Lever N f=%.1f' % f, alt[f]))
        print(line('TP4 pool, admission 0.9 -> 0.3 s (no disagg)', adm))
        print(line('2 x TP2 colocated pools, 4 seats each (cap 500k)', rep))
        for name, _ in arms:
            print(line(name, dis[name]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--mix', default='prod', choices=('prod', 'cold', 'decode', 'longagent'))
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--turns', type=int, default=60)
    ap.add_argument('--r', type=float, default=1.6)
    ap.add_argument('--k', type=float, default=1.8)
    ap.add_argument('--bw', type=float, default=40e9)
    ap.add_argument('--dcap', type=int, default=400000)
    ap.add_argument('--policy', default='remote', choices=('local', 'remote'))
    ap.add_argument('--local-tail', type=int, default=SHORT)
    ap.add_argument('--f', type=float, default=0.5, help='Lever N prefill share on the TP4 pool')
    ap.add_argument('--calibrate', action='store_true')
    ap.add_argument('--doc', action='store_true')
    ap.add_argument('--sweep', action='store_true')
    a = ap.parse_args(argv)
    global F_SHARE
    F_SHARE = a.f
    if a.doc:
        print(PARAMS_DOC)
        return
    if a.sweep:
        sweep(a.seeds, a.turns)
        return
    if a.calibrate:
        for name, (scripts, offsets) in (('parked_turns (M v612/v614 ER: makespan 244.5/248.7 s, TTFT p50 6.0-6.1 max 73.8-84.3, decode/turn 12.5-14.2)', parked_turns_scripts()),
                                         ('concurrent8_skew (M: 2x252k first tokens 113.7/215.2 s v547; 126/231 v567; 232-237 v542/v592)', skew_scripts())):
            t4, dg = run_pair(scripts, offsets, r=a.r, k=a.k, bw=a.bw, dcap=a.dcap, policy=a.policy, local_tail=a.local_tail)
            print(name)
            print(fmt('  tp4', t4))
            print(fmt('  2P+2D', dg))
        return
    for seed in range(a.seeds):
        scripts = make_scripts(a.mix, 8, a.turns, seed)
        t4, dg = run_pair(scripts, r=a.r, k=a.k, bw=a.bw, dcap=a.dcap, policy=a.policy, local_tail=a.local_tail)
        print('seed %d' % seed)
        print(fmt('  tp4', t4))
        print(fmt('  2P+2D', dg))


if __name__ == '__main__':
    main()
