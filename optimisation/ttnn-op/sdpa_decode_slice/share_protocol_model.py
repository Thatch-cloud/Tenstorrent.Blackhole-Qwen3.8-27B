"""An exhaustive model of the K/V share protocol with K1b's read-ahead leader (R9), against the stage-3 leader.

One bundle position p: the leader (entry 0) and its B-1 twins, each a reader and a compute kernel, with the
factory's CBs (K and V two-slot rings, a one-slot mask CB), the READY semaphore on the leader and a VALID
semaphore on every twin. Every NoC transaction the protocol issues is an in-flight item delivered at an
arbitrary later point (the model explores every order): the leader's K and V multicasts (their bytes are
the leader slot's contents WHEN DELIVERED, so a source overwritten too early shows), the VALID multicast,
and each twin's READY increment. A DRAM read is one step (read_k / read_v end in a read barrier). A write
barrier waits for the leader's in-flight writes, an atomic barrier for the twin's increments. The waits are
the kernels' equality waits (Semaphore::wait), and the wait and the reset of READY are separate steps.

Programs (n = 0 .. N-1 on this core):
  stage-3 leader  read K(n), read V(n); wait READY == B-1; READY = 0; multicast K(n), V(n); write barrier;
                  VALID(n); mask(n)
  R9 leader       prologue: read K(0), V(0). Per n: wait READY == B-1; READY = 0; multicast K(n), V(n);
                  mask(n); read K(n+1), V(n+1) unless n+1 == N; write barrier; VALID(n). Exit barrier.
  twin            reserve K, reserve V; VALID = 0; READY += 1 on the leader; wait VALID == 1; push K, V;
                  mask(n). Exit: atomic barrier, VALID = 0.
  compute         wait K(n) (its slot must hold chunk n); pop K; [mask(n): wait, check, pop]; wait V(n) (check);
                  pop V. ('mask_first' pops K after the mask, the other plausible order.)
Masks: 'none', 'tail' (the root core's last chunk only, the served tail mode) or 'every' (every chunk).

Checked in every reachable state: no deadlock (some step is enabled until every program has finished and
nothing is in flight); a multicast lands only in a twin slot that twin has reserved for that chunk; a DRAM
read never lands in a leader slot a multicast is still reading; every compute sees chunk n in the slot it
pops for chunk n (K, V and mask), in order. Broken variants the checks must catch: 'no_barrier' (the R9
leader sends VALID without the write barrier) and 'ready_first' (a twin raises READY before it reserves).

    py -3.11 share_protocol_model.py                  # the default sweep (B = 2, 3; N = 1..6 / 1..4)
"""

import argparse
import sys

K, V, M = 0, 1, 2          # the rings: K and V (two slots), the mask (one slot)
SLOTS = (2, 2, 1)


class Violation(Exception):
    pass


def leader_program(chunks, mode, masks):
    ops = []
    if mode == 'stage3':
        for n in range(chunks):
            ops += [('read', K, n), ('read', V, n), ('wait_ready',), ('reset_ready',), ('mcast', n), ('wbarrier',),
                    ('valid',)]
            if masks(n):
                ops.append(('mask', n))
        ops.append(('wbarrier',))
        return tuple(ops)
    ops += [('read', K, 0), ('read', V, 0)]
    for n in range(chunks):
        ops += [('wait_ready',), ('reset_ready',), ('mcast', n)]
        if masks(n):
            ops.append(('mask', n))
        if n + 1 < chunks:
            ops += [('read', K, n + 1), ('read', V, n + 1)]
        if mode != 'no_barrier':
            ops.append(('wbarrier',))
        ops.append(('valid',))
    ops.append(('wbarrier',))
    return tuple(ops)


def twin_program(chunks, masks, ready_first=False):
    ops = []
    for n in range(chunks):
        if ready_first:
            ops += [('reset_valid',), ('ready',), ('reserve', K), ('reserve', V)]
        else:
            ops += [('reserve', K), ('reserve', V), ('reset_valid',), ('ready',)]
        ops += [('wait_valid',), ('push', K, n), ('push', V, n)]
        if masks(n):
            ops.append(('mask', n))
    ops += [('abarrier',), ('reset_valid',)]
    return tuple(ops)


def compute_program(chunks, masks, order='k_first'):
    ops = []
    for n in range(chunks):
        ops.append(('wait', K, n))
        mask = [('wait', M, n), ('pop', M)] if masks(n) else []
        ops += ([('pop', K)] + mask) if order == 'k_first' else (mask + [('pop', K)])
        ops += [('wait', V, n), ('pop', V)]
    return tuple(ops)


def masks_for(kind, chunks):
    if kind == 'none':
        return lambda n: False
    if kind == 'tail':
        return lambda n: n == chunks - 1
    if kind == 'every':
        return lambda n: True
    raise ValueError(kind)


class Model:
    """State: (pcs, rings, ready, valids, flight). pcs: one program counter per actor (leader reader, twin
    readers, one compute per core). rings[core][ring] = (wr, rd, reserved, slots). flight: sorted in-flight
    items ('data', twin, ring, slot), ('valid', twin), ('ready', twin)."""

    def __init__(self, batches, chunks, mode='readahead', masks='tail', order='k_first', ready_first=False):
        self.batches, self.chunks, self.mode = batches, chunks, mode
        mask = masks_for(masks, chunks)
        self.twins = batches - 1
        self.programs = ((leader_program(chunks, mode, mask),)
                         + tuple(twin_program(chunks, mask, ready_first) for _ in range(self.twins))
                         + tuple(compute_program(chunks, mask, order) for _ in range(batches)))

    def initial(self):
        rings = tuple(tuple((0, 0, False, (None,) * SLOTS[ring]) for ring in (K, V, M)) for _ in range(self.batches))
        return (tuple(0 for _ in self.programs), rings, 0, tuple(0 for _ in range(self.twins)), ())

    # --- helpers --------------------------------------------------------------------------------------
    @staticmethod
    def set_ring(rings, core, ring, value):
        core_rings = list(rings[core])
        core_rings[ring] = value
        out = list(rings)
        out[core] = tuple(core_rings)
        return tuple(out)

    def reader_core(self, actor):
        return 0 if actor == 0 else actor           # actor 0 = leader reader, 1..twins = twin readers

    def compute_core(self, actor):
        return actor - 1 - self.twins

    # --- one step of one actor, or None if blocked ---------------------------------------------------
    def step(self, state, actor):
        pcs, rings, ready, valids, flight = state
        program = self.programs[actor]
        if pcs[actor] >= len(program):
            return None
        op = program[pcs[actor]]
        advance = pcs[:actor] + (pcs[actor] + 1,) + pcs[actor + 1:]
        kind = op[0]
        if actor <= self.twins:
            core = self.reader_core(actor)
            twin = actor - 1
            if kind == 'read':                                  # leader: reserve, DRAM read, push (read_k/read_v)
                ring, n = op[1], op[2]
                wr, rd, reserved, slots = rings[core][ring]
                if wr - rd >= SLOTS[ring]:
                    return None
                slot = wr % SLOTS[ring]
                if any(item[0] == 'data' and item[2] == ring and item[3] == slot for item in flight):
                    raise Violation('leader DRAM read of chunk %d into %s slot %d while its multicast is in flight'
                                    % (n, 'KV'[ring], slot))
                slots = slots[:slot] + (n,) + slots[slot + 1:]
                return (advance, self.set_ring(rings, core, ring, (wr + 1, rd, False, slots)), ready, valids, flight)
            if kind == 'wait_ready':
                return (advance, rings, ready, valids, flight) if ready == self.twins else None
            if kind == 'reset_ready':
                return (advance, rings, 0, valids, flight)
            if kind == 'mcast':                                 # both slots of chunk n, to every twin
                n = op[1]
                items = []
                for ring in (K, V):
                    wr, rd, reserved, slots = rings[0][ring]
                    matches = [slot for slot in range(SLOTS[ring]) if slots[slot] == n]
                    if len(matches) != 1:
                        raise Violation('leader multicasts chunk %d but its %s ring holds %r' % (n, 'KV'[ring], slots))
                    items += [('data', t, ring, matches[0]) for t in range(self.twins)]
                return (advance, rings, ready, valids, tuple(sorted(flight + tuple(items))))
            if kind == 'wbarrier':
                pending = [item for item in flight if item[0] in ('data', 'valid')]
                return None if pending else (advance, rings, ready, valids, flight)
            if kind == 'valid':
                items = tuple(('valid', t) for t in range(self.twins))
                return (advance, rings, ready, valids, tuple(sorted(flight + items)))
            if kind == 'mask':                                  # reserve, read, push (read_mask_chunk)
                n = op[1]
                wr, rd, reserved, slots = rings[core][M]
                if wr - rd >= SLOTS[M]:
                    return None
                return (advance, self.set_ring(rings, core, M, (wr + 1, rd, False, (n,))), ready, valids, flight)
            if kind == 'reserve':                               # twin
                ring = op[1]
                wr, rd, reserved, slots = rings[core][ring]
                if wr - rd >= SLOTS[ring]:
                    return None
                return (advance, self.set_ring(rings, core, ring, (wr, rd, True, slots)), ready, valids, flight)
            if kind == 'reset_valid':
                return (advance, rings, ready, valids[:twin] + (0,) + valids[twin + 1:], flight)
            if kind == 'ready':
                return (advance, rings, ready, valids, tuple(sorted(flight + (('ready', twin),))))
            if kind == 'wait_valid':
                return (advance, rings, ready, valids, flight) if valids[twin] == 1 else None
            if kind == 'push':
                ring, n = op[1], op[2]
                wr, rd, reserved, slots = rings[core][ring]
                if not reserved:
                    raise Violation('twin %d pushes %s without a reservation' % (twin, 'KV'[ring]))
                return (advance, self.set_ring(rings, core, ring, (wr + 1, rd, False, slots)), ready, valids, flight)
            if kind == 'abarrier':
                pending = [item for item in flight if item == ('ready', twin)]
                return None if pending else (advance, rings, ready, valids, flight)
            raise ValueError(op)
        core = self.compute_core(actor)
        if kind == 'wait':
            ring, n = op[1], op[2]
            wr, rd, reserved, slots = rings[core][ring]
            if rd >= wr:
                return None
            got = slots[rd % SLOTS[ring]]
            if got != n:
                raise Violation('core %d compute expected chunk %d in its %s slot %d, found %r'
                                % (core, n, 'KVM'[ring], rd % SLOTS[ring], got))
            return (advance, rings, ready, valids, flight)
        if kind == 'pop':
            ring = op[1]
            wr, rd, reserved, slots = rings[core][ring]
            return (advance, self.set_ring(rings, core, ring, (wr, rd + 1, reserved, slots)), ready, valids, flight)
        raise ValueError(op)

    def deliver(self, state, index):
        pcs, rings, ready, valids, flight = state
        item = flight[index]
        rest = flight[:index] + flight[index + 1:]
        if item[0] == 'ready':
            return (pcs, rings, ready + 1, valids, rest)
        if item[0] == 'valid':
            twin = item[1]
            return (pcs, rings, ready, valids[:twin] + (1,) + valids[twin + 1:], rest)
        _, twin, ring, slot = item
        core = twin + 1
        wr, rd, reserved, slots = rings[core][ring]
        if not reserved or wr % SLOTS[ring] != slot:
            raise Violation('a multicast lands in twin %d %s slot %d, which it has not reserved (wr=%d rd=%d)'
                            % (twin, 'KV'[ring], slot, wr, rd))
        content = rings[0][ring][3][slot]
        slots = slots[:slot] + (content,) + slots[slot + 1:]
        return (pcs, self.set_ring(rings, core, ring, (wr, rd, reserved, slots)), ready, valids, rest)

    def successors(self, state):
        out = []
        for actor in range(len(self.programs)):
            nxt = self.step(state, actor)
            if nxt is not None:
                out.append(nxt)
        flight = state[4]
        for index in range(len(flight)):
            if index and flight[index] == flight[index - 1]:
                continue                                        # identical items: one order suffices
            out.append(self.deliver(state, index))
        return out

    def finished(self, state):
        pcs, _rings, _ready, _valids, flight = state
        return not flight and all(pc == len(program) for pc, program in zip(pcs, self.programs))

    def explore(self, limit=2_000_000):
        """Every reachable state: dict(states, finished, deadlocks, violation). Stops at the first violation or
        deadlock (with a short trace of the program counters) or at `limit` states (then complete=False)."""
        start = self.initial()
        seen = {start}
        stack = [start]
        result = dict(states=0, finished=0, deadlock=None, violation=None, complete=True)
        while stack:
            state = stack.pop()
            result['states'] += 1
            if result['states'] > limit:
                result['complete'] = False
                break
            if self.finished(state):
                result['finished'] += 1
                continue
            try:
                nexts = self.successors(state)
            except Violation as error:
                result['violation'] = str(error)
                return result
            if not nexts:
                result['deadlock'] = self.describe(state)
                return result
            for nxt in nexts:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return result

    def describe(self, state):
        pcs, rings, ready, valids, flight = state
        names = ['leader'] + ['twin%d' % t for t in range(self.twins)] + ['compute%d' % c for c in range(self.batches)]
        where = ', '.join('%s@%s' % (name, self.programs[i][pc] if pc < len(self.programs[i]) else 'done')
                          for i, (name, pc) in enumerate(zip(names, pcs)))
        return '%s; READY=%d VALID=%r in flight=%r' % (where, ready, valids, flight)


SWEEP = [(2, chunks) for chunks in range(1, 7)] + [(3, chunks) for chunks in range(1, 5)]


def sweep(modes=('stage3', 'readahead'), masks=('none', 'tail', 'every'), orders=('k_first', 'mask_first'),
          shapes=SWEEP):
    """[(label, result)] over the default grid: both leaders, every mask kind and compute order."""
    out = []
    for mode in modes:
        for mask in masks:
            for order in orders:
                for batches, chunks in shapes:
                    model = Model(batches, chunks, mode, mask, order)
                    out.append(('%s B=%d N=%d mask=%s compute=%s' % (mode, batches, chunks, mask, order),
                                model.explore()))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    parser.add_argument('--broken', action='store_true', help='also run the two broken variants')
    args = parser.parse_args(argv)
    status = 0
    for label, result in sweep():
        ok = result['complete'] and not result['violation'] and not result['deadlock'] and result['finished']
        print('%-60s %s states=%d' % (label, 'ok' if ok else 'FAIL %s' % (result['violation'] or result['deadlock']),
                                      result['states']))
        status |= not ok
    if args.broken:
        for label, model in (('no_barrier B=2 N=3', Model(2, 3, 'no_barrier', 'tail')),
                             ('ready_first B=2 N=3', Model(2, 3, 'readahead', 'tail', ready_first=True))):
            result = model.explore()
            print('%-60s %s' % ('broken ' + label, result['violation'] or result['deadlock'] or 'NOT CAUGHT'))
            status |= not (result['violation'] or result['deadlock'])
    return status


if __name__ == '__main__':
    sys.exit(main())
