"""Stage E, E3: serving_parked_engines.ParkedEngineSet and its attach (design sections 2.2, 2.4, 6.3, 6.4).

On the census world (test_parked_census), with the real engine, device, proposal and pool:
  - the attach order: the set after the block and before the lifecycle, closed after the lifecycle and
    before the block; no engine resident afterwards; the flag strictly '0' or '1', off building nothing;
  - the refusals before anything is built: blocks that do not read their carries in place (or no block),
    captures that are not the sequential widths, per-request collectives or eager proposals, the deferred
    QWEN_FAST_PARKED_DRAFTS, a lent slot, a resident engine;
  - the build: every slot parked on a synthetic request at P_cap = 1 over an all-page-0 table, native slot 0
    restored from the zeroed carry first, slot 0's drafter warmed at 2048 with the publish prewarm (its
    marker counting what it warmed), the built marker and the P7p ledger point;
  - the DRAM stop at k < 4, whose missing slots serve today's per-request builds;
  - the slot rule is acquire's: the same arrivals get the same slots with the flag on or off;
  - park, unpark (the counter, the marker, the slot back to the pool, today's build on it) and re-park at
    idle;
  - close: every engine and device closed, every slot and weight returned;
  - the parked cycles under the census: four users churning at mixed lengths and budgets, twenty rebinds
    per slot and more, packed rounds between, with no read of anything a replay may have overwritten,
    nothing allocated that outlives a request, and every live tensor owned.
"""

from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import serving_parked_engines as parked  # noqa: E402
from test_parked_census import PAGE_WIDTH, CensusCapture, World  # noqa: E402


def block(carries=True):
    return SimpleNamespace(carries_in_place=carries)


def make_set(world, *, blocks=None, capture_rows=4, environ=None, **extra):
    from serving_request_factory import device_components
    import os

    environment = dict(os.environ)
    environment.update(environ or {})
    engines = parked.ParkedEngineSet(operations=world.ops, model=world.model, sampler=world.sampler,
        helpers=world.helpers, pool=world.pool, weights=world.weights, fixtures=world.fixtures,
        collectives=world.collectives, blocks=(block(),) if blocks is None else blocks, capture_rows=capture_rows,
        components=device_components(), environ=environment, **extra)
    world.stack.callback(engines.close)
    return engines


def pages_for(blocks):
    pages = torch.full((1, PAGE_WIDTH), blocks[0], dtype=torch.int32)
    pages[0, :len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
    return pages


class ParkedRequest:
    """E4's bridge, as far as this step needs one: a taken slot rebound from a prefill of `prompt` tokens
    (from_prefill's host part is E4's), driven by FastRequest's step, and parked at its end."""

    def __init__(self, world, engines, entry, request_id, prompt, budget, first_block=1):
        from greedy_session import GreedySession
        from serving_fast_request import FastRequest

        self.world, self.engines, self.entry, self.request_id = world, engines, entry, request_id
        blocks = tuple(range(first_block, first_block + -(-(prompt + budget) // 64) + 1))
        capture = CensusCapture(world.ops, world.mesh, prompt)
        bound = engines.rebind_slot(entry, capture.outputs(), pages_for(blocks), position=prompt, request_id=request_id,
            make_session=lambda runtime: GreedySession(request_id, [1] * prompt, 7, vocab_size=248320,
                max_new_tokens=budget, eos_ids=(), neural={'dflash2': runtime}, verifier_rows=16, lookup_enabled=False))
        capture.close()
        self.session, self.engine = bound.session, bound.engine
        self.request = FastRequest(bound.session, bound.engine, bound.runtime, release_drafter=lambda: None)

    def step(self):
        if self.session.finished:
            return False
        self.request.step(self.request_id, cancelled=lambda: False)
        return not self.session.finished

    def finish(self, reason=None):
        result = self.engines.park(self.entry, reason=reason)
        self.session.close(self.request_id)
        return result


class BuildTests(unittest.TestCase):
    def test_every_slot_is_parked_on_a_synthetic_request_and_no_engine_is_resident_after(self):
        with World() as world:
            engines = make_set(world, environ={'QWEN_FAST_PUBLISH_PREWARM': '1'})
            with patch('memory_ledger.record') as ledger:
                self.assertEqual(engines.build(), 4)
            ledger.assert_called_once()
            self.assertEqual(ledger.call_args.args, ('P7p',))
            self.assertIs(ledger.call_args.kwargs['parked_engines'], engines)
            self.assertIsNone(world.engine_module._resident)
            for entry in engines.slots:
                self.assertEqual(entry.state, 'parked')
                self.assertEqual(entry.engine.phase, 'parked')
                self.assertIs(entry.device.pool_slot, entry.slot)
                self.assertTrue(entry.slot.lent)
                self.assertEqual(entry.engine.buckets[1]['capture_position'], 1)
                self.assertTrue(torch.equal(entry.engine.pages, torch.zeros((1, PAGE_WIDTH), dtype=torch.int32)))
                self.assertEqual(sorted(entry.engine.buckets), [1, 2, 4])
                self.assertEqual(tuple(entry.device.proposal_capture.buckets), (2048,))
            # slot 0's drafter was warmed at its steady state; the others were left at P_cap
            self.assertEqual([entry.device.position for entry in engines.slots], [2048, 1, 1, 1])
            warm = [line for line in world.lines if line.startswith(parked.WARM_MARKER)]
            self.assertEqual(len(warm), 1)
            self.assertRegex(warm[0], r'P=2048 ms=[0-9.]+ window=256 prewarm_pairs=7$')
            prewarm = [line for line in world.lines if line.startswith('[PINDIAG] publish prewarm')]
            self.assertEqual(len(prewarm), 1)
            self.assertIn('count=7', prewarm[0])
            built = [line for line in world.lines if line.startswith(parked.BUILT_MARKER)]
            self.assertEqual(len(built), 1)
            self.assertRegex(built[0], r'k=4 of 4 attach_ms=[0-9.]+ trace_used=[0-9]+ free=[0-9]+ largest_free=[0-9]+$')
            self.assertEqual(world.ops.violations, [])

    def test_native_slot_zero_is_restored_from_the_zeroed_carry_before_each_synthetic_build(self):
        with World() as world:
            engines = make_set(world)
            ops = world.ops
            ops.tracking = True
            engines.build()
            ops.tracking = False
            for entry in engines.slots:
                carry = [value.serial for snapshot in entry.slot.verifier.carry for value in snapshot]
                native = [value.serial for helper in world.helpers for value in helper.live]
                zeroed = next(index for index, event in enumerate(ops.trail)
                              if event[0] == 'write' and event[1] == carry[0] and event[2] == 'full_like')
                reads = [index for index, event in enumerate(ops.trail[zeroed:], zeroed)
                         if event[0] == 'read' and event[1] == carry[0]]
                writes = [index for index, event in enumerate(ops.trail[reads[0]:], reads[0])
                          if event[0] == 'write' and event[1] == native[0]]
                self.assertTrue(reads and writes, 'carry read, native slot 0 written, after the zeroing')
                captured = next(index for index, event in enumerate(ops.trail[zeroed:], zeroed) if event[0] == 'begin')
                self.assertLess(writes[0], captured, 'before the synthetic capture')

    def test_without_the_prewarm_flag_the_drafter_is_still_warmed_and_nothing_published(self):
        with World(environment={'QWEN_FAST_PUBLISH_PREWARM': '0'}) as world:
            engines = make_set(world, environ={'QWEN_FAST_PUBLISH_PREWARM': '0'})
            engines.build()
            warm = [line for line in world.lines if line.startswith(parked.WARM_MARKER)]
            self.assertRegex(warm[0], r'prewarm_pairs=0$')
            self.assertFalse([line for line in world.lines if line.startswith('[PINDIAG] publish prewarm')])


class RefusalTests(unittest.TestCase):
    def assert_nothing_built(self, world):
        self.assertFalse(any(slot.lent for slot in world.pool.slots))
        self.assertEqual(world.ops.captures, 1, 'the block\'s trace only')

    def test_the_set_is_refused_before_anything_is_built(self):
        cases = (
            (dict(blocks=(block(False),)), 'carries_in_place'),
            (dict(blocks=(block(True), block(False))), 'carries_in_place'),
            (dict(blocks=(SimpleNamespace(),)), 'carries_in_place'),
            (dict(blocks=()), 'carries_in_place'),
            (dict(capture_rows=None), 'sequential captures'),
            (dict(capture_rows=16), 'sequential captures'),
            (dict(environ={'QWEN_FAST_SHARED_CCL': '0'}), 'shared collectives'),
            (dict(environ={'QWEN_FAST_EAGER_PROPOSAL': '1'}), 'captured proposals'),
            (dict(environ={parked.DEFERRED_DRAFTS_FLAG: '0'}), 'deferred and not implemented'),
            (dict(environ={parked.PROJECT_ROWS_FLAG: '100'}), parked.PROJECT_ROWS_FLAG),
            (dict(environ={parked.AUDIT_FLAG: 'yes'}), parked.AUDIT_FLAG),
        )
        for options, message in cases:
            with self.subTest(options=options), World() as world:
                with self.assertRaisesRegex(ValueError, message):
                    make_set(world, **options)
                self.assert_nothing_built(world)

    def test_a_lent_slot_or_a_resident_engine_is_refused(self):
        with World() as world:
            slot = world.pool.acquire(owner='request')
            with self.assertRaisesRegex(ValueError, 'before any request'):
                make_set(world)
            world.pool.release(slot)
            world.engine_module._resident = object()
            try:
                with self.assertRaisesRegex(ValueError, 'one is resident'):
                    make_set(world)
            finally:
                world.engine_module._resident = None


class DramStopTests(unittest.TestCase):
    def readings(self, frees):
        """serving_prefill_admission.dram_reading answering each call with the next free figure (bytes)."""
        values = iter(frees)

        def reading(pool):
            free = next(values)
            return dict(free=free, largest_free=4 * 10 ** 9, trace_largest_free=None, trace_unread='n/a'), None
        return patch('serving_prefill_admission.dram_reading', side_effect=reading)

    def test_the_build_stops_where_the_split_is_short_and_the_rest_serve_per_request(self):
        import serving_prefill_admission as admission

        with World() as world:
            engines = make_set(world)
            need = admission.engine_build_peak() + parked.parked_arrival_need(engines.reserve(), 256)
            enough = need + admission.STRANDED_BYTES
            with self.readings([enough, enough, enough - 1, enough]):
                self.assertEqual(engines.build(), 2)
            self.assertEqual([entry.state for entry in engines.slots], ['parked', 'parked', 'unparked', 'unparked'])
            self.assertEqual([slot.lent for slot in world.pool.slots], [True, True, False, False])
            stopped = [line for line in world.lines if line.startswith('[PINDIAG] parked engines stopped')]
            self.assertEqual(len(stopped), 1)
            self.assertIn('k=2 of 4: short of free', stopped[0])
            # the parked slots first, then today's build on the first unlent one
            first, second = engines.take(), engines.take()
            self.assertEqual((first.index, second.index), (0, 1))
            self.assertIsNone(engines.take())
            request = world.admit('request', 300, 16)
            self.assertIs(request.runtime.drafter.pool_slot, world.pool.slots[2])
            request.close('request')
            for entry in (first, second):
                entry.state = 'parked'

    def test_the_need_is_an_engine_build_beside_a_parked_arrival(self):
        import serving_prefill_admission as admission

        reserve = 256 * 1024 * 1024
        self.assertEqual(parked.parked_arrival_need(reserve, 256),
                         admission.PREFILL_TRANSIENT_BYTES + 100 * 10 ** 6 + reserve)
        self.assertEqual(parked.parked_arrival_need(reserve, 0), admission.PREFILL_TRANSIENT_BYTES + 350 * 10 ** 6 + reserve)
        self.assertEqual(parked.parked_arrival_need(reserve, 256, single_released=True),
                         admission.PREFILL_TRANSIENT_BYTES + 327 * 10 ** 6 + reserve)


class SlotRuleTests(unittest.TestCase):
    def test_the_same_arrivals_get_the_same_slots_with_the_flag_on_or_off(self):
        rng = random.Random(7)
        script = []
        live = []
        for _ in range(60):
            if live and (len(live) == 4 or rng.random() < 0.45):
                script.append(('leave', live.pop(rng.randrange(len(live)))))
            else:
                name = 'r%d' % len(script)
                script.append(('arrive', name))
                live.append(name)
        with World() as world:
            # flag off: the pool's first-free rule, through acquire
            off, slots = {}, {}
            for kind, name in script:
                if kind == 'arrive':
                    slots[name] = world.pool.acquire(owner=name)
                    off[name] = slots[name].index
                else:
                    world.pool.release(slots.pop(name))
            for slot in slots.values():
                world.pool.release(slot)
            # flag on: the set's take, with slot 3 unparked to mix in today's builds
            engines = make_set(world)
            engines.build()
            engines.unpark(engines.slots[3], 'test')
            on, held = {}, {}
            for kind, name in script:
                if kind == 'arrive':
                    entry = engines.take()
                    if entry is None:
                        held[name] = world.pool.acquire(owner=name)
                        on[name] = held[name].index
                    else:
                        held[name] = entry
                        on[name] = entry.index
                else:
                    taken = held.pop(name)
                    if isinstance(taken, parked.ParkedSlot):
                        taken.state = 'parked'
                    else:
                        world.pool.release(taken)
            self.assertEqual(on, off)
            for taken in held.values():
                if isinstance(taken, parked.ParkedSlot):
                    taken.state = 'parked'
                else:
                    world.pool.release(taken)


class ParkCycleTests(unittest.TestCase):
    def test_a_request_parks_its_slot_again_and_a_failed_one_unparks_it_until_an_idle_repark(self):
        with World() as world:
            engines = make_set(world)
            engines.build()
            entry = engines.take()
            request = ParkedRequest(world, engines, entry, 'first', 300, 24)
            while request.step():
                pass
            self.assertIsNone(request.finish())
            self.assertEqual((entry.state, entry.rebinds, entry.parks), ('parked', 1, 1))
            rebinds = [line for line in world.lines if line.startswith(parked.REBIND_MARKER)]
            self.assertRegex(rebinds[-1], r'req=first slot=0 gen=2 P=300 budget=24 ms=[0-9.]+ single_rebuilt=0 window=256$')
            # a request whose engine failed: the slot is unparked, its engine and device closed, the slot back
            entry = engines.take()
            request = ParkedRequest(world, engines, entry, 'second', 300, 24)
            request.step()
            request.engine.phase = 'failed'
            self.assertEqual(request.finish(), 'engine phase failed')
            self.assertEqual((entry.state, entry.engine, entry.device, engines.unparks), ('unparked', None, None, 1))
            self.assertFalse(entry.slot.lent)
            self.assertIn('[PINDIAG] parked slot 0 unparked: engine phase failed', world.lines)
            # slot 0 now serves today's build, then re-parks at an idle moment
            self.assertIsNone(engines.take())
            today = world.admit('third', 300, 16)
            self.assertIs(today.runtime.drafter.pool_slot, world.pool.slots[0])
            self.assertEqual(engines.repark_idle(), [], 'not while the slot is lent')
            today.close('third')
            self.assertEqual(engines.repark_idle(), [0])
            self.assertEqual((entry.state, engines.reparks), ('parked', 1))
            self.assertEqual(entry.engine.buckets[1]['capture_position'], 1, 'a synthetic build, never a real one')
            self.assertIsNone(world.engine_module._resident)
            self.assertTrue(any(line.startswith('[PINDIAG] parked slot 0 re-parked ms=') for line in world.lines))
            self.assertEqual(world.ops.violations, [])

    def test_a_rebind_that_fails_unparks_the_slot_and_propagates(self):
        with World() as world:
            engines = make_set(world)
            engines.build()
            entry = engines.take()
            with patch.object(parked, 'rebind_device', side_effect=RuntimeError('device fault')):
                with self.assertRaisesRegex(RuntimeError, 'device fault'):
                    ParkedRequest(world, engines, entry, 'request', 300, 16)
            self.assertEqual((entry.state, engines.unparks), ('unparked', 1))
            self.assertFalse(entry.slot.lent)


class CloseTests(unittest.TestCase):
    def test_close_returns_every_slot_and_weight_even_with_a_request_still_serving(self):
        with World() as world:
            engines = make_set(world)
            engines.build()
            entry = engines.take()
            request = ParkedRequest(world, engines, entry, 'request', 300, 16)
            request.step()
            engines.close()
            self.assertTrue(all(slot.state == 'closed' for slot in engines.slots))
            self.assertFalse(any(slot.lent for slot in world.pool.slots))
            self.assertEqual(world.weights.borrowers, [])
            engines.close()
            with self.assertRaisesRegex(ValueError, 'closed'):
                engines.take()

    def test_the_audit_installs_the_replay_ledger_until_close(self):
        import verifier_engine

        with World() as world:
            engines = make_set(world, environ={parked.AUDIT_FLAG: '1'})
            self.assertTrue(engines.ledger.installed)
            engines.build()
            self.assertGreater(engines.ledger.count, 0)
            engines.close()
            self.assertIsNone(verifier_engine._replay_count)


class AttachTests(unittest.TestCase):
    def test_the_set_is_built_after_the_block_and_closed_before_it(self):
        import test_serving_runtime

        case = test_serving_runtime.RuntimeAttachmentTests()
        for four_as_two in (False, True):
            with self.subTest(four_as_two=four_as_two):
                case.exercise(packed=True, users=4, four_as_two=four_as_two, parked={})
                (options,) = case.parked_calls
                self.assertEqual(options['capture_rows'], 4)
                self.assertEqual(len(options['blocks']), 2 if four_as_two else 1)
                self.assertEqual(sorted(options), ['blocks', 'capture_rows', 'collectives', 'fixtures', 'helpers',
                                                   'model', 'operations', 'pool', 'sampler', 'weights'])
        with self.assertRaisesRegex(RuntimeError, 'request failed'):
            case.exercise(packed=True, users=4, four_as_two=False, parked={}, fail=True)
        with self.assertRaisesRegex(RuntimeError, 'attach failed'):
            case.exercise(packed=True, users=4, four_as_two=False, parked={}, attach_fail=True)

    def test_off_nothing_is_imported_or_built_and_a_bad_value_is_refused(self):
        import test_serving_runtime

        case = test_serving_runtime.RuntimeAttachmentTests()
        for environ in ({}, {'QWEN_FAST_PARKED_ENGINES': '0'}):
            with self.subTest(environ=environ), \
                    patch('serving_parked_engines.ParkedEngineSet', side_effect=AssertionError('built')):
                case.exercise(packed=True, users=4, four_as_two=False, extra_env=environ)
        with self.assertRaisesRegex(ValueError, 'QWEN_FAST_PARKED_ENGINES must be 0 or 1'):
            case.exercise(packed=True, users=4, four_as_two=False, parked={},
                          extra_env={'QWEN_FAST_PARKED_ENGINES': 'yes'},
                          refused_after_blocks="QWEN_FAST_PARKED_ENGINES must be 0 or 1, got 'yes'")


class ParkedCensusTests(unittest.TestCase):
    """The parked cycles under the census: what E0's census holds today's churn to."""

    def test_four_users_churning_on_parked_engines_read_nothing_a_replay_may_have_overwritten(self):
        rng = random.Random(11)
        with World() as world:
            engines = make_set(world)
            engines.build()
            attached = set(world.ops.live)
            live, serial = [], 0
            lengths = (1, 2, 15, 16, 17, 63, 64, 65, 127, 128, 129, 300, 2047, 2048, 2049, 3000)
            budgets = (2, 3, 4, 5, 16, 40)
            while min(entry.rebinds for entry in engines.slots) < 20:
                while len(live) < 4:
                    entry = engines.take()
                    self.assertIsNotNone(entry)
                    prompt = rng.choice(lengths)
                    budget = min(rng.choice(budgets), PAGE_WIDTH * 64 - prompt - 1)
                    serial += 1
                    live.append(ParkedRequest(world, engines, entry, 'r%d' % serial, prompt, budget,
                                              first_block=1 + 70 * entry.index))
                world.replay_block()
                for request in list(live):
                    if not request.step():
                        self.assertIsNone(request.finish())
                        live.remove(request)
                if rng.random() < 0.2 and live:
                    request = live.pop(rng.randrange(len(live)))
                    self.assertIsNone(request.finish(), 'a cancelled request parks too')
                self.assertEqual(world.unowned([], [('parked', engines)] + [('live%d' % index, request.request)
                                                                          for index, request in enumerate(live)]), [])
            for request in live:
                self.assertIsNone(request.finish())
            self.assertEqual(world.ops.violations, [])
            self.assertEqual(set(world.ops.live), attached, 'nothing a request did outlives it')
            self.assertEqual(engines.unparks, 0)
            self.assertGreaterEqual(min(entry.rebinds for entry in engines.slots), 20)


if __name__ == '__main__':
    unittest.main()
