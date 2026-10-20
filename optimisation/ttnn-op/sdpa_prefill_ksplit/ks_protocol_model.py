"""CPU model of the [QWEN-SDPA-PF] K-split protocol: S chains per q chunk (one per slice) plus the slice-to-slice partial hand-off, as a discrete-event model checker.
No ttnn, no torch. It extends ../sdpa_prefill_chain/pf_protocol_model.py (same actor / waiter machinery, same chain state machine, same constants) by:

  THE SLICES      a q chunk of B = C + q + 1 K blocks is streamed by E = effective_slices(B, S) slices (ksplit_planner), slice s taking [s * B / E, (s + 1) * B / E). Each slice is a
                  G6 chain of M members (the M heads that share a KV head: 6 at TP4) running the chain protocol over ITS blocks; a slice s >= E is idle (no actor does anything).
  THE HAND-OFF    slice s > 0 is a RECEIVER: its reader reserves cb_prev_out and cb_lse_in (immediately: they are empty), credits the sender (core of slice s - 1, same head) with an atomic,
                  waits for an atomic barrier (the BH read-barrier hazard of R7), runs its rounds, then waits for the VALID flag, pushes both CBs and resets the flag. Slice s < E - 1 is a
                  SENDER: its writer waits for the compute kernel's output, then for the credit, writes output and lse into the receiver's CBs (a bit copy: the payload is the coverage
                  [begin, end) of the K blocks it holds), waits for both acks, relays VALID, pops. The last active slice writes the output to DRAM. The compute actor of a receiver folds
                  the previous coverage [0, begin) with its own [begin, end) at the end of its rounds (the ring merge; the model checks the coverages are adjacent) and, after an EVEN
                  number of rounds, pops the one dummy K/V pair the reader pushed (the pinned ring loop's epilogue).

  variant 'ok'     (a) every actor terminates, (b) every K / V slot holds what its consumer expects and every hand-off payload is the coverage the receiver's merge expects, the final
                   payload of every head is [0, B) exactly, (c) every semaphore ends at its initial value with nothing in flight and no message lands on a core that exited or that
                   has not reserved its CBs, and every CB drains. Also reported: the K/V blocks each slice's injector read (each block of the q chunk exactly once over the slices),
                   the chain slot writes, and the hand-off writes (E - 1 per head, 64 KB + 16 KB each, always slice s to s + 1 of the same head).
  variant 'hang'   the test flag 0x400: every receiver withholds its credit. Must deadlock with exactly the blocked set hang_blocked(E, M) says and no content error.
  mutations        (the checker's own teeth, each a protocol bug it must catch)
                   'valid_before_ack'   the sender raises VALID without waiting for the data writes' acks
                   'no_credit_wait'     the sender writes without waiting for the receiver's credit
                   'dummy_missing'      the reader does not push the dummy K/V pair after an even number of rounds
                   'eff_mismatch'       the compute kernel of slice 1 takes its range with another E than the reader (the range arithmetic of the two kernels differing)
"""

import heapq
import itertools
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'sdpa_prefill_chain'))

import ksplit_planner as planner  # noqa: E402
import pf_protocol_model as chain  # noqa: E402

KV_SLOTS = chain.KV_SLOTS
Q_CB_TILES = chain.Q_CB_TILES
Q_TILES_PER_UNIT = chain.Q_TILES_PER_UNIT
Q_SUBBLOCK_TILES = chain.Q_SUBBLOCK_TILES
VARIANTS = ('ok', 'hang')
MUTATIONS = (None, 'valid_before_ack', 'no_credit_wait', 'dummy_missing', 'eff_mismatch')
(KFREE, VFREE, KFULL, VFULL, QFREE, QFULL, RCV, SND, ABAR, WBAR, HVALID, HCREDIT, PREVFULL, OUTFULL) = range(14)
WAIT_NAMES = ('k_free', 'v_free', 'k_full', 'v_full', 'q_free', 'q_full', 'valid', 'credit', 'atomic_barrier', 'write_barrier', 'handoff_valid', 'handoff_credit',
              'prev_full', 'out_full')
SLOT = 64          # waiter key stride: more cores than any model run has


class Result:
    def __init__(self):
        self.errors = []
        self.blocked = {}
        self.terminated = False
        self.final = {}
        self.events = 0
        self.reads = {}            # slice -> [K blocks the injector read]
        self.chain_writes = {}     # slice -> chain slot writes issued (K + V)
        self.handoffs = []         # (sender core, receiver core, bytes)
        self.final_payloads = {}   # head -> coverage written to DRAM

    def as_dict(self):
        return dict(errors=self.errors[:10], error_count=len(self.errors), blocked=self.blocked, terminated=self.terminated, final=self.final, events=self.events)


def hang_blocked(eff, members):
    """The blocked set of the planted hang (variant 'hang'): every receiver slice s >= 1 waits for VALID in its reader and for the previous output in its compute, its writer waits for
    an output that never comes; the writers of slice 0 wait for a credit that never comes. (Core c is slice c // members, head c % members.)"""
    blocked = {}
    for head in range(members):
        blocked['writer %d' % head] = 'handoff_credit'
        for s in range(1, eff):
            core = s * members + head
            blocked['reader %d' % core] = 'handoff_valid'
            blocked['consumer %d' % core] = 'prev_full'
            blocked['writer %d' % core] = 'out_full'
    return blocked


def simulate(C, q, seed, slices=2, members=6, variant='ok', mutation=None, stall=0.02):
    """One q chunk (per-head chunk index q) of a call at chunk_start C * 128 with `slices` slices of `members` heads each, under random timing."""
    if variant not in VARIANTS:
        raise ValueError('unknown variant %r' % (variant,))
    if mutation not in MUTATIONS:
        raise ValueError('unknown mutation %r' % (mutation,))
    rng = random.Random((((seed * 1000003) ^ (C * 7919) ^ (q * 104729) ^ (slices * 15485863)) * 4 + VARIANTS.index(variant)) * 8 + MUTATIONS.index(mutation))
    blocks = planner.blocks_of(C * 128, q)
    eff = planner.effective_slices(blocks, slices)
    M = members
    n = slices * M
    spans = [planner.slice_range(blocks, s, slices) for s in range(slices)]
    if mutation == 'eff_mismatch' and eff >= 2:
        # the compute kernel of slice 1 reads its range with E + 1 slices of the same blocks: a different range from the reader's
        wrong = planner.slice_begin(blocks, 1, eff + 1), planner.slice_begin(blocks, 2, eff + 1)
    else:
        wrong = None
    result = Result()
    errors = result.errors
    out_bytes, lse_bytes = planner.handoff_bytes()

    def active(core):
        return core // M < eff

    def rounds_of(core, compute=False):
        s = core // M
        span = spans[s] if s < eff else None
        if compute and wrong is not None and s == 1:
            span = wrong
        return [] if span is None else list(range(span[0], span[1]))

    kf, vf = [KV_SLOTS] * n, [KV_SLOTS] * n
    kq, vq = [0] * n, [0] * n
    kw, vw, kr, vr = [0] * n, [0] * n, [0] * n, [0] * n
    kslot = [[None, None] for _ in range(n)]
    vslot = [[None, None] for _ in range(n)]
    qf, qq = [Q_CB_TILES] * n, [0] * n
    snd, rcv = [0] * n, [0] * n                       # the chain's semaphores (INVALID initial)
    hcredit, hvalid = [0] * n, [0] * n                # the hand-off's semaphores (INVALID initial)
    prev_reserved, prev_full, prev_payload, prev_lse = [False] * n, [False] * n, [None] * n, [None] * n
    writer_exited = [False] * n
    out_full, out_payload = [False] * n, [None] * n
    ia, iw = [0] * n, [0] * n
    exited = [False] * n
    speed = [rng.uniform(0.3, 3.0) for _ in range(n)]
    heap = []
    counter = itertools.count()
    waiters = {}
    clock = [0.0]
    for s in range(slices):
        result.reads[s] = []
        result.chain_writes[s] = 0

    def delay(low, high):
        extra = 50.0 if rng.random() < stall else 0.0
        return rng.uniform(low, high) + extra

    def schedule(at, what):
        heapq.heappush(heap, (at, next(counter), what))

    def notify(kind, j):
        key = kind * SLOT + j
        pending = waiters.get(key)
        if not pending:
            return
        keep = []
        for actor, predicate in pending:
            if predicate():
                schedule(clock[0], actor)
            else:
                keep.append((actor, predicate))
        if keep:
            waiters[key] = keep
        else:
            del waiters[key]

    # ---- chain messages (the pf model's, within one slice) --------------------------------------------------------------------------
    def chain_credit(src, dst):
        def land():
            if exited[dst]:
                errors.append('credit from %d landed on %d after it exited' % (src, dst))
            snd[dst] += 1
            if snd[dst] > 1:
                errors.append('double credit on %d (%d)' % (dst, snd[dst]))
            ia[src] -= 1
            notify(SND, dst)
            notify(ABAR, src)
        schedule(clock[0] + delay(0.2, 3.0), land)

    def chain_write(src, slots, slot, issued):
        dst = src + 1

        def land():
            data = slots[src][slot]
            if data != issued:
                errors.append('slot %d of %d changed while being forwarded (%r -> %r)' % (slot, src, issued, data))
            if exited[dst]:
                errors.append('write from %d landed on %d after it exited' % (src, dst))
            slots[dst][slot] = data
            iw[src] -= 1
            notify(WBAR, src)
        schedule(clock[0] + delay(0.5, 6.0), land)

    def chain_relay(src):
        dst = src + 1

        def land():
            if exited[dst]:
                errors.append('VALID from %d landed on %d after it exited' % (src, dst))
            rcv[dst] = 1
            iw[src] -= 1
            notify(RCV, dst)
            notify(WBAR, src)
        schedule(clock[0] + delay(0.2, 3.0), land)

    # ---- hand-off messages ------------------------------------------------------------------------------------------------------------
    def handoff_credit(src, dst):
        def land():
            if writer_exited[dst]:
                errors.append('hand-off credit from %d landed on %d after its writer exited' % (src, dst))
            hcredit[dst] += 1
            if hcredit[dst] > 1:
                errors.append('double hand-off credit on %d' % dst)
            ia[src] -= 1
            notify(HCREDIT, dst)
            notify(ABAR, src)
        schedule(clock[0] + delay(0.2, 3.0), land)

    def handoff_write(src, dst, payload, lse):
        def land():
            if not prev_reserved[dst]:
                errors.append('hand-off from %d landed on %d before it reserved cb_prev_out / cb_lse_in' % (src, dst))
            if exited[dst]:
                errors.append('hand-off from %d landed on %d after it exited' % (src, dst))
            (prev_lse if lse else prev_payload)[dst] = payload
            iw[src] -= 1
            notify(WBAR, src)
        schedule(clock[0] + delay(0.5, 6.0), land)

    def handoff_relay(src, dst):
        def land():
            if exited[dst]:
                errors.append('hand-off VALID from %d landed on %d after it exited' % (src, dst))
            hvalid[dst] = 1
            iw[src] -= 1
            notify(HVALID, dst)
            notify(WBAR, src)
        schedule(clock[0] + delay(0.2, 3.0), land)

    def chain_credit_prev(j, withhold=False):
        rcv[j] = 0
        if not withhold:
            ia[j] += 1
            chain_credit(j, j - 1)

    # ---- actors -----------------------------------------------------------------------------------------------------------------------
    def reader(core):
        s, head = divmod(core, M)
        if not active(core):
            exited[core] = True
            return
        mine = rounds_of(core)
        receive, forward = head > 0, head < M - 1
        receiver = s > 0
        yield delay(0.0, 30.0)                                    # cores start at different times
        if receiver:
            prev_reserved[core] = True                            # cb_prev_out / cb_lse_in are empty: reserve_back returns at once
            if variant != 'hang':
                ia[core] += 1
                handoff_credit(core, core - M)
            yield (ABAR, lambda: ia[core] == 0)
        for position, k in enumerate(mine):
            first = position == 0
            if receive:
                yield (KFREE, lambda: kf[core] > 0)
                kf[core] -= 1
                ks = kw[core]
                yield (VFREE, lambda: vf[core] > 0)
                vf[core] -= 1
                vs = vw[core]
                chain_credit_prev(core)
                yield (RCV, lambda: rcv[core] == 1)
                kq[core] += 1
                kw[core] ^= 1
                notify(KFULL, core)
            else:
                yield (KFREE, lambda: kf[core] > 0)
                kf[core] -= 1
                ks = kw[core]
                yield delay(2.0, 12.0)
                kslot[core][ks] = ('K', k)
                result.reads[s].append(k)
                kq[core] += 1
                kw[core] ^= 1
                notify(KFULL, core)
            if first:
                yield (ABAR, lambda: ia[core] == 0)
                for _sub in range(Q_TILES_PER_UNIT // Q_SUBBLOCK_TILES):
                    yield (QFREE, lambda: qf[core] >= Q_SUBBLOCK_TILES)
                    qf[core] -= Q_SUBBLOCK_TILES
                    yield delay(0.5, 2.0)
                    qq[core] += Q_SUBBLOCK_TILES
                    notify(QFULL, core)
            if receive:
                vq[core] += 1
                vw[core] ^= 1
                notify(VFULL, core)
            else:
                yield (VFREE, lambda: vf[core] > 0)
                vf[core] -= 1
                vs = vw[core]
                yield delay(2.0, 12.0)
                vslot[core][vs] = ('V', k)
                vq[core] += 1
                vw[core] ^= 1
                notify(VFULL, core)
            if forward:
                yield (SND, lambda: snd[core] == 1)
                snd[core] = 0
                iw[core] += 2
                result.chain_writes[s] += 2
                chain_write(core, kslot, ks, kslot[core][ks])
                chain_write(core, vslot, vs, vslot[core][vs])
                yield (WBAR, lambda: iw[core] == 0)
                iw[core] += 1
                chain_relay(core)
        if receiver:
            yield (HVALID, lambda: hvalid[core] == 1)
            prev_full[core] = True
            notify(PREVFULL, core)
            hvalid[core] = 0
        if len(mine) % 2 == 0 and mutation != 'dummy_missing':
            yield (KFREE, lambda: kf[core] > 0)
            kf[core] -= 1
            kslot[core][kw[core]] = ('DUMMY',)
            kq[core] += 1
            kw[core] ^= 1
            notify(KFULL, core)
            yield (VFREE, lambda: vf[core] > 0)
            vf[core] -= 1
            vslot[core][vw[core]] = ('DUMMY',)
            vq[core] += 1
            vw[core] ^= 1
            notify(VFULL, core)
        yield (WBAR, lambda: iw[core] == 0)
        yield (ABAR, lambda: ia[core] == 0)
        rcv[core] = 0
        snd[core] = 0
        exited[core] = True

    def consumer(core):
        s, _head = divmod(core, M)
        if not active(core):
            return
        mine = rounds_of(core, compute=True)
        span = spans[s]
        for position, k in enumerate(mine):
            first = position == 0
            unit_end = position + 1 == len(mine)
            yield (KFULL, lambda: kq[core] > 0)
            if first:
                yield (QFULL, lambda: qq[core] >= Q_SUBBLOCK_TILES)
                yield (QFULL, lambda: qq[core] >= Q_TILES_PER_UNIT)
            want = ('K', k)
            slot = kr[core]
            if kslot[core][slot] != want:
                errors.append('core %d round %d: K slot holds %r, expected %r' % (core, position, kslot[core][slot], want))
            yield speed[core] * rng.uniform(4.0, 8.0)
            if kslot[core][slot] != want:
                errors.append('core %d round %d: K slot overwritten before its pop (%r)' % (core, position, kslot[core][slot]))
            kq[core] -= 1
            kf[core] += 1
            kr[core] ^= 1
            notify(KFREE, core)
            yield (VFULL, lambda: vq[core] > 0)
            want = ('V', k)
            slot = vr[core]
            if vslot[core][slot] != want:
                errors.append('core %d round %d: V slot holds %r, expected %r' % (core, position, vslot[core][slot], want))
            yield speed[core] * rng.uniform(2.0, 5.0)
            if vslot[core][slot] != want:
                errors.append('core %d round %d: V slot overwritten before its pop (%r)' % (core, position, vslot[core][slot]))
            vq[core] -= 1
            vf[core] += 1
            vr[core] ^= 1
            notify(VFREE, core)
            if unit_end:
                qq[core] -= Q_TILES_PER_UNIT
                qf[core] += Q_TILES_PER_UNIT
                notify(QFREE, core)
        # the ring merge: the previous slices' coverage [0, begin) and this slice's [begin, end) -> [0, end)
        coverage = (span[0], span[1])
        if s > 0:
            yield (PREVFULL, lambda: prev_full[core])
            for name, previous in (('output', prev_payload[core]), ('lse', prev_lse[core])):
                if previous != (0, span[0]):
                    errors.append('core %d merges the previous %s of coverage %r, expected %r' % (core, name, previous, (0, span[0])))
            prev_full[core] = False
            prev_payload[core] = None
            prev_lse[core] = None
            coverage = (0, span[1])
        yield speed[core] * rng.uniform(1.0, 3.0)
        out_payload[core] = coverage
        out_full[core] = True
        notify(OUTFULL, core)
        if len(mine) % 2 == 0:
            yield (KFULL, lambda: kq[core] > 0)
            yield (VFULL, lambda: vq[core] > 0)
            if kslot[core][kr[core]] != ('DUMMY',) or vslot[core][vr[core]] != ('DUMMY',):
                errors.append('core %d: the epilogue pop found %r / %r, not the dummy pair' % (core, kslot[core][kr[core]], vslot[core][vr[core]]))
            kq[core] -= 1
            kf[core] += 1
            kr[core] ^= 1
            notify(KFREE, core)
            vq[core] -= 1
            vf[core] += 1
            vr[core] ^= 1
            notify(VFREE, core)

    def writer(core):
        s, head = divmod(core, M)
        if not active(core):
            return
        yield (OUTFULL, lambda: out_full[core])
        if s == eff - 1:
            result.final_payloads[head] = out_payload[core]                  # the DRAM write
            out_full[core] = False
            writer_exited[core] = True
            return
        if mutation != 'no_credit_wait':
            yield (HCREDIT, lambda: hcredit[core] == 1)
            hcredit[core] = 0
        receiver = core + M
        payload = out_payload[core]
        iw[core] += 2
        result.handoffs.append((core, receiver, out_bytes + lse_bytes))
        handoff_write(core, receiver, payload, False)      # the 32 output tiles
        handoff_write(core, receiver, payload, True)       # the 4 lse tiles
        if mutation != 'valid_before_ack':
            yield (WBAR, lambda: iw[core] == 0)
        iw[core] += 1
        handoff_relay(core, receiver)
        yield (WBAR, lambda: iw[core] == 0)
        out_full[core] = False
        writer_exited[core] = True

    actors = {}
    blocked_on = {}
    done = set()

    def make(name, generator, j):
        actors[name] = (generator, j)
        schedule(0.0, name)

    for core in range(n):
        make(('reader', core), reader(core), core)
        make(('consumer', core), consumer(core), core)
        make(('writer', core), writer(core), core)

    def step(name):
        generator, j = actors[name]
        while True:
            try:
                request = next(generator)
            except StopIteration:
                done.add(name)
                blocked_on.pop(name, None)
                return
            if request.__class__ is float:
                schedule(clock[0] + request, name)
                return
            kind, predicate = request
            if predicate():
                continue
            waiters.setdefault(kind * SLOT + j, []).append((name, predicate))
            blocked_on[name] = kind
            return

    events = 0
    while heap:
        at, _seq, what = heapq.heappop(heap)
        clock[0] = at
        events += 1
        if what.__class__ is tuple:
            blocked_on.pop(what, None)
            step(what)
        else:
            what()
    result.events = events
    result.terminated = len(done) == len(actors)
    result.blocked = {'%s %d' % name: WAIT_NAMES[kind] for name, kind in sorted(blocked_on.items()) if name not in done}
    result.final = dict(sender=list(snd), receiver=list(rcv), hcredit=list(hcredit), hvalid=list(hvalid), atomics_in_flight=list(ia), writes_in_flight=list(iw),
                        k_free=list(kf), v_free=list(vf), q_free=list(qf), prev_full=list(prev_full), out_full=list(out_full), blocks=blocks, eff=eff, members=M)
    return result


def check_ok(result):
    """[problems] of an 'ok' run."""
    problems = []
    if not result.terminated:
        problems.append('(a) did not terminate; blocked %r' % (result.blocked,))
    if result.errors:
        problems.append('(b) %d content/ordering errors, first %r' % (len(result.errors), result.errors[0]))
    final = result.final
    if result.terminated:
        blocks, eff, members = final['blocks'], final['eff'], final['members']
        if any(final[name][i] for name in ('sender', 'receiver', 'hcredit', 'hvalid', 'atomics_in_flight', 'writes_in_flight') for i in range(len(final[name]))):
            problems.append('(c) semaphores or in-flight counts not back to initial: %r' % (final,))
        busy = eff * members
        if any(final['k_free'][i] != KV_SLOTS or final['v_free'][i] != KV_SLOTS or final['q_free'][i] != Q_CB_TILES for i in range(busy)):
            problems.append('(c) CBs not drained: %r' % (final,))
        if any(final['prev_full']) or any(final['out_full']):
            problems.append('(c) a hand-off CB was left full: %r' % (final,))
        wrong = {head: payload for head, payload in result.final_payloads.items() if payload != (0, blocks)}
        if len(result.final_payloads) != members or wrong:
            problems.append('(b) the final payloads are not [0, %d) for every head: %r' % (blocks, result.final_payloads))
        covered = sorted(k for reads in result.reads.values() for k in reads)
        if covered != list(range(blocks)):
            problems.append('(b) the injectors read blocks %r, not [0, %d) once each' % (covered, blocks))
        if len(result.handoffs) != (eff - 1) * members or any(dst != src + members for src, dst, _b in result.handoffs):
            problems.append('(b) the hand-offs are %r, not %d writes from slice s to s + 1 of the same head' % (result.handoffs, (eff - 1) * members))
    return problems


def check_hang(result):
    """[problems] of a 'hang' run: it deadlocks with exactly the blocked set of the planted hang and no content error."""
    problems = []
    if result.final['eff'] < 2:                       # one active slice: nobody hands anything over, the flag changes nothing
        return check_ok(result)
    if result.terminated:
        problems.append('(d) the planted hang terminated')
    expected = hang_blocked(result.final['eff'], result.final['members'])
    if result.blocked != expected:
        problems.append('(d) blocked %r, expected exactly %r' % (result.blocked, expected))
    if result.errors:
        problems.append('(d) content errors before the hang: %r' % (result.errors[0],))
    return problems


def sweep(seeds, starts=(0, 3, 4, 8, 16, 33), qs=(0, 3, 7), slices=(2, 3, 4), members=6, variants=('ok',), mutation=None):
    """Run the grid; returns ({variant: [(C, q, S, seed, problems)] of the failing runs}, run count). Engaged shapes only matter, but the idle and one-slice shapes are in the grid on purpose."""
    failures = {variant: [] for variant in variants}
    runs = 0
    for variant in variants:
        for C in starts:
            for q in qs:
                for count in slices:
                    for seed in seeds:
                        result = simulate(C, q, seed, count, members, variant, mutation)
                        runs += 1
                        problems = check_ok(result) if variant == 'ok' else check_hang(result)
                        if problems:
                            failures[variant].append((C, q, count, seed, problems))
    return failures, runs


def main(argv=None):
    import argparse
    import time

    parser = argparse.ArgumentParser(description='run the K-split protocol model checker grid')
    parser.add_argument('--seeds', type=int, default=40)
    parser.add_argument('--members', type=int, default=6)
    options = parser.parse_args(argv)
    began = time.time()
    failures, runs = sweep(range(options.seeds), members=options.members, variants=('ok', 'hang'))
    print('ks model checker: %d runs in %.0f s' % (runs, time.time() - began))
    for variant, items in failures.items():
        print('  %-5s failures %d%s' % (variant, len(items), '' if not items else ', first %r' % (items[0],)))
    return 0 if not any(failures.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
