"""CPU model of the [QWEN-SDPA-PF] G6 K/V chain: the host topology (factory F4), the per-core round
lists, and a discrete-event model checker of the reader sync protocol (sdpa-prefill-share-spec.md
5.3 items 2-4). No ttnn, no torch.

HOST PORTS (served shape: B=1, 12 Q heads, 2 KV heads, 2048 rows, q/k chunk 128, 110 cores)
  decompose()        q_chunk_remapping.hpp decompose_global_q_index (zigzag when causal)
  core_ranges()      the global-Q range per core (factory fd8c0676 F:389-410 and the reader RT loop)
  g6_topology()      factory F4: group cores by their whole (batch, kv_head, q_chunk) unit list, then
                     raster or NoC-cost order, roles and the 14 reader RT words (F5 order)
  round_list()       the reader's k loop per unit (R:361-402) and compute's (compute_common.hpp
                     1617-1654), compact [(batch, kv_head, q, n_k)] per core

THE MODEL CHECKER (simulate)
  One G6 group: 6 readers running the R7 state machine (reserve K and V, credit_prev - reset the
  receiver flag, then credit upstream unless withheld, exactly the R1 helper -, wait VALID, push K;
  the Q subblocks at a unit's first chunk after an atomic
  barrier; push V; wait the downstream credit, reset it, write the K and V slots, write barrier,
  relay VALID; at exit barriers and the R8 reset), 6 compute consumers (wait K, at a unit's first
  chunk Q, compute, pop K, wait V, compute, pop V, pop Q at the unit's end) with per-core random
  speeds, 2-slot K and V CBs and a 64-tile Q CB per core, semaphores with atomic increment, and
  remote writes / atomics / relays landing after random delays (a forwarded slot's bytes are
  sampled when the write lands, so a source overwritten mid-flight is caught). Every slot carries
  a content tag (K or V, kv_head, k_chunk); a consumer checks the tag when it starts on a slot and
  again when it pops it.

  variant 'ok'       asserts (a) every reader and consumer terminates, (b) every consumer check
                     sees the content its own round list expects, (c) every semaphore ends at its
                     initial value with nothing in flight and no message landing after its target
                     exited.
  variant 'hang'     the test_hang flag: the sink withholds its credit on its last round (credit_prev
                     with withhold: the flag is still reset). Must deadlock with exactly the sink's
                     VALID wait, P4's credit wait and the sink's consumer (waiting for that K) stuck,
                     and with no content error. The hang mutation 'withhold_skips_reset' (the first
                     K64g reader: the whole credit_prev call was skipped, reset included) is caught
                     here: the sink sees its previous round's VALID, pushes a stale slot and exits.
  variant 'wrong_c'  member 3 runs with C + 1 (two more rounds): must deadlock, i.e. the call
                     never completes, so no output - right or wrong - is ever returned.
  variant 'wrong_m'  member 3 runs another m (the SAME number of rounds, different chunks): the
                     protocol completes and the consumers see foreign chunks. This is the case the
                     spec's 4.2 'a hang, not corruption' does not cover: equal-length divergence is
                     silent corruption, which is why the host grouping key must be the whole unit
                     list (checked by g6_topology's tests and by mutation M-A on hardware).
"""

import heapq
import itertools
import random

# --- the served shape --------------------------------------------------------------------------

SERVED = dict(B=1, NQH=12, NKH=2, rows=2048, q_chunk=128, k_chunk=128, head_dim=256, block=64, grid=(11, 10),
              tile=32)
NOC_GRID = (17, 12)
Q_TILES_PER_UNIT = 32       # Sq_chunk_t (4) x DHt (8)
Q_CB_TILES = 64             # q_buffer_factor 2
Q_SUBBLOCK_TILES = 16       # qk_subblock_h (2) x DHt (8)
KV_SLOTS = 2


def decompose(index, q_num_chunks, nqh, zigzag):
    """(nb, nq, q_chunk) of a global Q index: q_chunk_remapping.hpp decompose_global_q_index."""
    if zigzag:
        head, pos = divmod(index, q_num_chunks)
        q = pos // 2 if pos % 2 == 0 else q_num_chunks - 1 - pos // 2
        index = head * q_num_chunks + q
    return index // (nqh * q_num_chunks), (index // q_num_chunks) % nqh, index % q_num_chunks


def core_ranges(num_cores, total_q_chunks, pair_distribute):
    """[(global_q_start, global_q_count)] per linear core (F:389-410, clamp of the reader RT loop)."""
    if pair_distribute:
        pairs = total_q_chunks // 2
        base, extra_cores, extra = (pairs // num_cores) * 2, pairs % num_cores, 2
    else:
        base, extra_cores, extra = total_q_chunks // num_cores, total_q_chunks % num_cores, 1
    out = []
    for i in range(num_cores):
        start = i * base + min(i, extra_cores) * extra
        count = base + (extra if i < extra_cores else 0)
        if start >= total_q_chunks:
            start, count = total_q_chunks, 0
        elif start + count > total_q_chunks:
            count = total_q_chunks - start
        out.append((start, count))
    return out


def geometry(B=1, NQH=12, NKH=2, rows=2048, q_chunk=128, grid=(11, 10), causal=True):
    q_num_chunks = rows // q_chunk
    num_cores = grid[0] * grid[1]
    total = B * NQH * q_num_chunks
    return dict(B=B, NQH=NQH, NKH=NKH, q_num_chunks=q_num_chunks, num_cores=num_cores, total=total, grid=grid,
                causal=causal, pair=causal and q_num_chunks % 2 == 0)


def unit_lists(geo):
    """Per core: [(nb, nq, q)] in the order the reader visits them."""
    out = []
    for start, count in core_ranges(geo['num_cores'], geo['total'], geo['pair']):
        out.append([decompose(start + u, geo['q_num_chunks'], geo['NQH'], geo['causal']) for u in range(count)])
    return out


def order_cost(phys, order, noc_grid=NOC_GRID):
    """F4's cost: sum over consecutive members of the direction-agnostic torus Manhattan distance,
    in uint32 arithmetic (kQwenNocX - dx wraps if dx > kQwenNocX; std::min then picks dx)."""
    total = 0
    for a, b in zip(order, order[1:]):
        (ax, ay), (bx, by) = phys[a], phys[b]
        dx, dy = abs(ax - bx), abs(ay - by)
        total += min(dx, (noc_grid[0] - dx) % (1 << 32)) + min(dy, (noc_grid[1] - dy) % (1 << 32))
    return total


def g6_topology(geo=None, coords=None, noc_order=False, noc_grid=NOC_GRID):
    """Factory F4 in Python. coords: linear core -> (x, y) NoC coordinate (worker_core_from_logical_core);
    identity-free placeholders are used when None (only the order cost reads them).
    Returns dict(groups={key: [cores in chain order]}, info=[per-core 14-word dict], chains, members)."""
    geo = geo or geometry()
    q_per_kv = geo['NQH'] // geo['NKH']
    groups = {}
    for core, units in enumerate(unit_lists(geo)):
        if not units:
            continue
        key = tuple(value for nb, nq, q in units for value in (nb, nq // q_per_kv, q))
        groups.setdefault(key, []).append(core)
    if coords is None:
        coords = {i: (i % geo['grid'][0], i // geo['grid'][0]) for i in range(geo['num_cores'])}
    blank = dict(participates=0, is_injector=0, is_sink=0, batch=0, head=0, q_chunk_start=0, q_chunk_count=0,
                 prev_x=0, prev_y=0, next_x=0, next_y=0, next_core_q_chunks=0, mcast_num_dests=0, mcast_sender_wait=0)
    info = [dict(blank) for _ in range(geo['num_cores'])]
    ordered, chains, members_total = {}, 0, 0
    for key in sorted(groups):                 # std::map iteration order (only the log counters see it)
        members = groups[key]
        if len(members) < 2:
            continue
        phys = [tuple(coords[i]) for i in members]
        order = list(range(len(members)))
        if noc_order:
            best, best_cost = order, order_cost(phys, order, noc_grid)
            for probe in itertools.permutations(range(len(members))):   # lexicographic, as next_permutation
                cost = order_cost(phys, probe, noc_grid)
                if cost < best_cost:
                    best, best_cost = list(probe), cost
            order = best
        units = len(key) // 3
        for p, index in enumerate(order):
            core = members[index]
            sink = p + 1 == len(order)
            prev = phys[order[p - 1]] if p > 0 else (0, 0)
            nxt = (0, 0) if sink else phys[order[p + 1]]
            info[core] = dict(participates=1, is_injector=int(p == 0), is_sink=int(sink), batch=key[0], head=key[1],
                              q_chunk_start=key[2], q_chunk_count=units, prev_x=prev[0], prev_y=prev[1],
                              next_x=nxt[0], next_y=nxt[1], next_core_q_chunks=0 if sink else units,
                              mcast_num_dests=0, mcast_sender_wait=0)
        ordered[key] = [members[index] for index in order]
        chains += 1
        members_total += len(members)
    return dict(groups=ordered, info=info, chains=chains, members=members_total, all_groups=groups)


RT_ORDER = ('participates', 'is_injector', 'is_sink', 'batch', 'head', 'q_chunk_start', 'q_chunk_count', 'prev_x',
            'prev_y', 'next_x', 'next_y', 'next_core_q_chunks', 'mcast_num_dests', 'mcast_sender_wait')


def rt_words(entry):
    return [entry[name] for name in RT_ORDER]


# --- round lists ---------------------------------------------------------------------------------

def page_blocks(C, rows=2048, q_chunk=128, block=64):
    """The model's page-table width at chunk_start C * q_chunk (forward_prefill_paged: needed blocks
    padded to a multiple of 32)."""
    needed = -(-(C * q_chunk + rows) // block)
    return -(-needed // 32) * 32


def extents(C, rows=2048, q_chunk=128, k_chunk=128, block=64, blocks=None, tile=32):
    """(Skt, valid_Skt, valid_Sqt, Sq_chunk_t, Sk_chunk_t) of the flexible path (F:251-290)."""
    blocks = page_blocks(C, rows, q_chunk, block) if blocks is None else blocks
    Sk = rows + blocks * block
    padded_Sk = -(-Sk // k_chunk) * k_chunk
    return padded_Sk // tile, -(-Sk // tile), -(-rows // tile), q_chunk // tile, k_chunk // tile


def reader_k_chunks(C, q, ext):
    """The reader's k loop count for one unit (R:361-402): k while k * Sk_chunk_t < q_high."""
    Skt, _valid_Skt, _valid_Sqt, Sq, Sk = ext
    q_low = (C + q) * Sq
    q_high = min(q_low + Sq, Skt)
    count = 0
    while count * Sk < q_high:
        count += 1
    return count


def compute_k_chunks(C, q, ext):
    """compute_common.hpp 1617-1654 (STANDARD, causal, chunked): k_chunk_end = ceil(q_high / Sk)."""
    Skt, _valid_Skt, _valid_Sqt, Sq, Sk = ext
    q_start = (C + q) * Sq
    q_high = min(q_start + Sq, Skt)
    return (q_high + Sk - 1) // Sk


def kv_row_tiles(C, k, ext):
    """kv_row_tile_count of one K/V chunk (R:403-405, flexible bound R:275-278)."""
    _Skt, valid_Skt, valid_Sqt, Sq, Sk = ext
    bound = min(C * Sq + valid_Sqt, valid_Skt)
    start = min(k * Sk, bound)
    return min(start + Sk, bound) - start


def reader_k_count(C, q, ext):
    """reader_k_chunks in closed form (the smallest k with k * Sk >= q_high); the CPU tests check
    the two agree, and the round sweep over C = 0..1100 uses this one."""
    Skt, _valid_Skt, _valid_Sqt, Sq, Sk = ext
    q_high = min((C + q) * Sq + Sq, Skt)
    return -(-q_high // Sk)


def round_list(C, units, q_per_kv, ext):
    """Compact round list of one core: [(nb, kv_head, q, n_k)] per unit (the rounds are the k
    chunks 0..n_k-1 of each unit, in order)."""
    return [(nb, nq // q_per_kv, q, reader_k_count(C, q, ext)) for nb, nq, q in units]


def expand(compact):
    """[(unit, k, first_of_unit, content)] with content = (nb, kv_head, k)."""
    out = []
    for unit, (nb, kvh, _q, count) in enumerate(compact):
        for k in range(count):
            out.append((unit, k, k == 0, (nb, kvh, k)))
    return out


def served_group_rounds(C, m, g=0, geo=None):
    """The expanded round list every member of G6 group (g, m) runs at chunk_start C * 128."""
    geo = geo or geometry()
    ext = extents(C)
    units = [(0, 6 * g, m), (0, 6 * g, geo['q_num_chunks'] - 1 - m)]
    return expand(round_list(C, units, geo['NQH'] // geo['NKH'], ext))


# --- the discrete-event model checker -------------------------------------------------------------

VARIANTS = ('ok', 'hang', 'wrong_c', 'wrong_m')
MUTATIONS = (None, 'relay_before_ack', 'credit_before_reserve', 'no_reset')
HANG_MUTATIONS = ('withhold_skips_reset',)      # only meaningful with variant 'hang'
ALL_MUTATIONS = MUTATIONS + HANG_MUTATIONS
KFREE, VFREE, KFULL, VFULL, QFREE, QFULL, RCV, SND, ABAR, WBAR = range(10)
WAIT_NAMES = ('k_free', 'v_free', 'k_full', 'v_full', 'q_free', 'q_full', 'valid', 'credit', 'atomic_barrier',
              'write_barrier')


class Result:
    def __init__(self):
        self.errors = []
        self.blocked = {}
        self.terminated = False
        self.final = {}
        self.events = 0

    def as_dict(self):
        return dict(errors=self.errors[:10], error_count=len(self.errors), blocked=self.blocked,
                    terminated=self.terminated, final=self.final, events=self.events)


def simulate(C, m, seed, variant='ok', size=6, stall=0.02, mutation=None):
    """One G6 group (KV head 0, light chunk m) at chunk_start C * 128 under random timing.

    mutation (the checker's own teeth; each is a protocol bug the model must catch):
      'relay_before_ack'       VALID relayed without waiting for the data writes' acks
      'credit_before_reserve'  the receiver credits upstream before reserving its K/V slots
      'no_reset'               the receiver does not reset its flag to INVALID before crediting
      'withhold_skips_reset'   (hang variant) the withheld credit also skips the flag reset
    """
    if variant not in VARIANTS:
        raise ValueError('unknown variant %r' % (variant,))
    if mutation not in ALL_MUTATIONS:
        raise ValueError('unknown mutation %r' % (mutation,))
    # A stable seed (str hashes are salted per process): the variant enters by its index.
    rng = random.Random((((seed * 1000003) ^ (C * 7919) ^ (m * 104729)) * 4 + VARIANTS.index(variant)) * 4
                        + ALL_MUTATIONS.index(mutation))
    rounds = [served_group_rounds(C, m) for _ in range(size)]
    odd = size // 2
    if variant == 'wrong_c':
        rounds[odd] = served_group_rounds(C + 1, m)
    elif variant == 'wrong_m':
        rounds[odd] = served_group_rounds(C, (m + 3) % 8)
    last = size - 1
    result = Result()
    errors = result.errors

    # per-core state
    kf, vf = [KV_SLOTS] * size, [KV_SLOTS] * size        # free slots
    kq, vq = [0] * size, [0] * size                       # full slots
    kw, vw, kr, vr = [0] * size, [0] * size, [0] * size, [0] * size
    kslot = [[None, None] for _ in range(size)]
    vslot = [[None, None] for _ in range(size)]
    qf, qq = [Q_CB_TILES] * size, [0] * size
    snd, rcv = [0] * size, [0] * size                     # INVALID initial values
    ia, iw = [0] * size, [0] * size                       # atomics / writes in flight
    exited = [False] * size
    speed = [rng.uniform(0.3, 3.0) for _ in range(size)]  # per-core compute speed

    heap = []
    counter = itertools.count()
    waiters = {}
    clock = [0.0]

    def delay(low, high):
        extra = 50.0 if rng.random() < stall else 0.0
        return rng.uniform(low, high) + extra

    def schedule(at, what):
        heapq.heappush(heap, (at, next(counter), what))

    def notify(kind, j):
        key = kind * 16 + j
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

    # messages ---------------------------------------------------------------------------------
    def credit(src, dst):
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

    def write(src, slots, slot, issued):
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

    def relay(src):
        dst = src + 1

        def land():
            if exited[dst]:
                errors.append('VALID from %d landed on %d after it exited' % (src, dst))
            rcv[dst] = 1
            iw[src] -= 1
            notify(RCV, dst)
            notify(WBAR, src)
        schedule(clock[0] + delay(0.2, 3.0), land)

    def credit_prev(j, withhold):
        """The R1 helper: reset this core's receiver flag, then (unless withheld) credit upstream."""
        if mutation == 'withhold_skips_reset' and withhold:
            return                                    # the first K64g reader: the whole call skipped
        if mutation != 'no_reset':
            rcv[j] = 0
        if not withhold:
            ia[j] += 1
            credit(j, j - 1)

    # actors: generators yielding a delay (float) or (kind, predicate) -------------------------------
    def reader(j):
        mine = rounds[j]
        receive, forward = j > 0, j < last
        count = len(mine)
        for n, (_unit, _k, first, content) in enumerate(mine):
            if receive:
                early = mutation == 'credit_before_reserve'
                withhold = variant == 'hang' and j == last and n + 1 == count   # pf_test_hang && is_sink && last_round
                if early:
                    credit_prev(j, withhold)
                yield (KFREE, lambda: kf[j] > 0)
                kf[j] -= 1
                ks = kw[j]
                yield (VFREE, lambda: vf[j] > 0)
                vf[j] -= 1
                vs = vw[j]
                if not early:
                    credit_prev(j, withhold)
                yield (RCV, lambda: rcv[j] == 1)
                kq[j] += 1
                kw[j] ^= 1
                notify(KFULL, j)
            else:
                yield (KFREE, lambda: kf[j] > 0)
                kf[j] -= 1
                ks = kw[j]
                yield delay(2.0, 12.0)
                kslot[j][ks] = ('K',) + content
                kq[j] += 1
                kw[j] ^= 1
                notify(KFULL, j)
            if first:
                yield (ABAR, lambda: ia[j] == 0)
                for _sub in range(Q_TILES_PER_UNIT // Q_SUBBLOCK_TILES):
                    yield (QFREE, lambda: qf[j] >= Q_SUBBLOCK_TILES)
                    qf[j] -= Q_SUBBLOCK_TILES
                    yield delay(0.5, 2.0)
                    qq[j] += Q_SUBBLOCK_TILES
                    notify(QFULL, j)
            if receive:
                vq[j] += 1
                vw[j] ^= 1
                notify(VFULL, j)
            else:
                yield (VFREE, lambda: vf[j] > 0)
                vf[j] -= 1
                vs = vw[j]
                yield delay(2.0, 12.0)
                vslot[j][vs] = ('V',) + content
                vq[j] += 1
                vw[j] ^= 1
                notify(VFULL, j)
            if forward:
                yield (SND, lambda: snd[j] == 1)
                snd[j] = 0
                iw[j] += 2
                write(j, kslot, ks, kslot[j][ks])
                write(j, vslot, vs, vslot[j][vs])
                if mutation != 'relay_before_ack':
                    yield (WBAR, lambda: iw[j] == 0)
                iw[j] += 1
                relay(j)
        yield (WBAR, lambda: iw[j] == 0)
        yield (ABAR, lambda: ia[j] == 0)
        rcv[j] = 0
        snd[j] = 0
        exited[j] = True

    def consumer(j):
        mine = rounds[j]
        count = len(mine)
        for n, (_unit, _k, first, content) in enumerate(mine):
            unit_end = n + 1 == count or mine[n + 1][2]
            yield (KFULL, lambda: kq[j] > 0)
            if first:
                yield (QFULL, lambda: qq[j] >= Q_SUBBLOCK_TILES)
                yield (QFULL, lambda: qq[j] >= Q_TILES_PER_UNIT)
            want = ('K',) + content
            slot = kr[j]
            if kslot[j][slot] != want:
                errors.append('core %d round %d: K slot holds %r, expected %r' % (j, n, kslot[j][slot], want))
            yield speed[j] * rng.uniform(4.0, 8.0)
            if kslot[j][slot] != want:
                errors.append('core %d round %d: K slot overwritten before its pop (%r)' % (j, n, kslot[j][slot]))
            kq[j] -= 1
            kf[j] += 1
            kr[j] ^= 1
            notify(KFREE, j)
            yield (VFULL, lambda: vq[j] > 0)
            want = ('V',) + content
            slot = vr[j]
            if vslot[j][slot] != want:
                errors.append('core %d round %d: V slot holds %r, expected %r' % (j, n, vslot[j][slot], want))
            yield speed[j] * rng.uniform(2.0, 5.0)
            if vslot[j][slot] != want:
                errors.append('core %d round %d: V slot overwritten before its pop (%r)' % (j, n, vslot[j][slot]))
            vq[j] -= 1
            vf[j] += 1
            vr[j] ^= 1
            notify(VFREE, j)
            if unit_end:
                qq[j] -= Q_TILES_PER_UNIT
                qf[j] += Q_TILES_PER_UNIT
                notify(QFREE, j)

    actors = {}
    blocked_on = {}

    def make(name, generator, j):
        actors[name] = (generator, j)
        schedule(0.0, name)

    for j in range(size):
        make(('reader', j), reader(j), j)
        make(('consumer', j), consumer(j), j)
    done = set()

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
            waiters.setdefault(kind * 16 + j, []).append((name, predicate))
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
    result.blocked = {'%s %d' % name: WAIT_NAMES[kind] for name, kind in sorted(blocked_on.items())
                      if name not in done}
    result.final = dict(sender=list(snd), receiver=list(rcv), atomics_in_flight=list(ia), writes_in_flight=list(iw),
                        k_free=list(kf), v_free=list(vf), q_free=list(qf))
    return result


def check_ok(result):
    """[problems] of an 'ok' run: (a) termination, (b) content, (c) semaphores and nothing in flight."""
    problems = []
    if not result.terminated:
        problems.append('(a) did not terminate; blocked %r' % (result.blocked,))
    if result.errors:
        problems.append('(b) %d content/ordering errors, first %r' % (len(result.errors), result.errors[0]))
    final = result.final
    if result.terminated and (any(final['sender']) or any(final['receiver']) or any(final['atomics_in_flight'])
                              or any(final['writes_in_flight'])):
        problems.append('(c) semaphores or in-flight counts not back to initial: %r' % (final,))
    if result.terminated and (any(v != KV_SLOTS for v in final['k_free'] + final['v_free'])
                              or any(v != Q_CB_TILES for v in final['q_free'])):
        problems.append('(c) CBs not drained: %r' % (final,))
    return problems


HANG_BLOCKED = {'reader 5': 'valid', 'reader 4': 'credit', 'consumer 5': 'k_full'}


def check_hang(result):
    problems = []
    if result.terminated:
        problems.append('(d) the planted hang terminated')
    if result.blocked != HANG_BLOCKED:
        problems.append('(d) blocked %r, expected exactly %r' % (result.blocked, HANG_BLOCKED))
    if result.errors:
        problems.append('(d) content errors before the hang: %r' % (result.errors[0],))
    return problems


def sweep(seeds, starts=(0, 1, 15, 16, 33, 64), ms=(0, 3, 7), variants=('ok',)):
    """Run the grid; returns {variant: [(C, m, seed, problems)]} of the failing runs and a count."""
    failures = {variant: [] for variant in variants}
    runs = 0
    for variant in variants:
        for C in starts:
            for m in ms:
                for seed in seeds:
                    result = simulate(C, m, seed, variant)
                    runs += 1
                    if variant == 'ok':
                        problems = check_ok(result)
                    elif variant == 'hang':
                        problems = check_hang(result)
                    elif variant == 'wrong_c':
                        problems = [] if not result.terminated else ['(e) members with different C terminated']
                    else:
                        problems = [] if (result.terminated and result.errors) else [
                            'wrong_m: expected completion with foreign chunks, got terminated=%s errors=%d'
                            % (result.terminated, len(result.errors))]
                    if problems:
                        failures[variant].append((C, m, seed, problems))
    return failures, runs


def _worker(arguments):
    seeds, starts, ms, variants = arguments
    return sweep(seeds, starts, ms, variants)


def main(argv=None):
    import argparse
    import multiprocessing
    import time

    parser = argparse.ArgumentParser(description='run the G6 protocol model checker grid')
    parser.add_argument('--seeds', type=int, default=2000)
    parser.add_argument('--jobs', type=int, default=multiprocessing.cpu_count())
    parser.add_argument('--variants', default='ok,hang,wrong_c,wrong_m')
    options = parser.parse_args(argv)
    variants = tuple(options.variants.split(','))
    began = time.time()
    chunks = [(list(range(s, options.seeds, options.jobs)), (0, 1, 15, 16, 33, 64), (0, 3, 7), variants)
              for s in range(options.jobs)]
    with multiprocessing.Pool(options.jobs) as pool:
        parts = pool.map(_worker, chunks)
    failures = {variant: [] for variant in variants}
    runs = 0
    for part, count in parts:
        runs += count
        for variant, items in part.items():
            failures[variant].extend(items)
    print('model checker: %d runs (%d seeds x C {0,1,15,16,33,64} x m {0,3,7} x %s) in %.0f s'
          % (runs, options.seeds, ','.join(variants), time.time() - began))
    for variant in variants:
        print('  %-8s failures %d%s' % (variant, len(failures[variant]),
                                        '' if not failures[variant] else ', first %r' % (failures[variant][0],)))
    return 0 if not any(failures.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
