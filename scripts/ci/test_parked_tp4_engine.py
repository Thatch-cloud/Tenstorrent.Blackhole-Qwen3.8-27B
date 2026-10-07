"""Engine reuse, E1': verifier_engine_tp.VerifierEngine.park, rebind_refusal and rebind, and the page-table write they share with
VerifierPageBinding.refresh (serving_parked_engines.write_page_tables; design 2.2, 3.4), on the four-chip census world.

On the census world (test_parked_census): the real VerifierEngine captured over a pool slot's storage.
Held here:
  - a rebound engine's host state is a freshly built engine's for the same request, apart from the
    whitelisted counters and the capture position its traces were taken at;
  - a rebind allocates nothing on the device, rewrites every fixture page table in full with the
    request's table, puts every bucket's first verify back on the unreplayed path, resets every retained
    block's flags and drops a stale packed adoption;
  - budgets under four get the widths a fresh engine would;
  - the rebind refuses what the constructor refuses, with its messages, before writing anything, and a
    refused engine stays parked;
  - park fences, refuses a block in flight as close() does, returns why an engine cannot park, and
    clears the residency a parked engine held;
  - the page-table write issues refresh's operations, call for call;
  - a rebound engine serves verifies and commits census-clean.
"""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import serving_parked_engines as parked  # noqa: E402
from test_parked_tp4_census import PAGE_WIDTH, World  # noqa: E402

TARGET_TAPS = (5, 19, 33, 47, 61)
# Host attributes a rebound engine may differ from a fresh one in: timings, the ledger's mark, and the
# retained block's replay counter.
WHITELIST = {'setup_ms', 'rebind_ms', 'replay_mark', 'request_widths'}


def zero_drafts(request_id, history, count):
    return tuple(0 for _ in range(count))


def session(request_id, prompt, budget, *, drafts=False):
    from greedy_session import GreedySession

    return GreedySession(request_id, [1] * prompt, 7, vocab_size=248320, max_new_tokens=budget, eos_ids=(),
                         neural={'dflash2': zero_drafts} if drafts else None, verifier_rows=16, lookup_enabled=False)


def request_blocks(prompt, budget, first=1):
    return tuple(range(first, first + -(-(prompt + budget) // 64) + 1))


def request_pages(prompt, budget, first=1):
    blocks = request_blocks(prompt, budget, first)
    pages = torch.full((1, PAGE_WIDTH), blocks[0], dtype=torch.int32)
    pages[0, :len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
    return pages


def build_engine(world, request_session, pages, slot):
    engine_module = world.engine_module
    return engine_module.VerifierEngine(world.model, request_session, pages, world.helpers, sampler=world.sampler,
        norm_batch=True, attention_replay=False, replay_group_rows=4, max_verify_rows=16, native_sampling_rows=True,
        retain_feature_taps=TARGET_TAPS, commit_only_gdn=True, target_attention_t16=False, storage=slot.verifier,
        capture_rows=4)


def parked_engine(world, index=0):
    """A synthetic engine over pool slot `index` (P_cap = 1, budget 16, an all-page-0 table), parked."""
    slot = world.pool.slots[index]
    if slot.lent or any(not other.lent for other in world.pool.slots[:index]):
        raise AssertionError('the pool lends the first free slot')
    world.pool.acquire(owner='parked-%d' % index)
    synthetic = session('parked-%d' % index, 1, 16)
    engine = build_engine(world, synthetic, torch.zeros((1, PAGE_WIDTH), dtype=torch.int32), slot)
    if engine.park() is not None:
        raise AssertionError('a fresh synthetic engine must park')
    synthetic.close('parked-%d' % index)

    def retire():
        if engine.phase in ('verifying', 'verified', 'committing'):
            engine.phase = 'failed'
        engine.close()
        if slot.lent:
            world.pool.release(slot)
    world.stack.callback(retire)
    return engine, slot


def retained_flags(retained):
    return {name: value for name, value in vars(retained).items()
            if name not in ('records', 'operations', 'fixture', 'replay_epoch')}


def host_state(engine):
    """Everything a verify, a publish or the packed step reads from the engine's host side, with device
    tensors by their addresses."""
    from gdn_multitoken_conv import addresses

    operations = engine.operations

    def place(values):
        return [tuple(addresses(operations, value)) for value in values]

    state = {name: value for name, value in vars(engine).items()
             if name not in WHITELIST | {'buckets', 'session', 'model', 'helpers', 'operations', 'mesh', 'sampler',
                                         'pages', 'initial', 'carry', 'borrowed', 'storage'}}
    state['session'] = engine.session.request_id
    state['pages'] = engine.pages.tolist()
    state['initial'] = [place(snapshot) for snapshot in engine.initial]
    state['carry'] = [place(snapshot) for snapshot in engine.carry]
    state['borrowed'] = sorted(place(engine.borrowed))
    state['storage'] = id(engine.storage)
    buckets = {}
    for key, bucket in engine.buckets.items():
        fixture = bucket['fixture']
        buckets[key] = dict(rows=bucket['rows'], first=bucket['first'], commits=sorted(bucket['commits']),
            checkpoints=[place(snapshot) for snapshot in bucket['checkpoints']],
            target_features=place(bucket['target_features']),
            tables=place([fixture.pages, fixture.singleton_pages]),
            inputs=place(fixture.inputs()),
            retained=None if fixture.retained is None else retained_flags(fixture.retained))
    state['buckets'] = buckets
    return state


class RebindStateTests(unittest.TestCase):
    def fresh_state(self, world, prompt, budget):
        slot = world.pool.acquire(owner='fresh')
        request = session('request', prompt, budget)
        engine = build_engine(world, request, request_pages(prompt, budget), slot)
        state = host_state(engine)
        engine.close()
        request.close('request')
        world.pool.release(slot)
        return state

    def test_a_rebound_engine_is_a_fresh_engine_for_the_same_request(self):
        # R4: the widths a rebound engine asks and serves are a cold engine's for the request, whatever the budget, so the host state is the
        # fresh one's for budgets under four too (a fresh engine captures fewer widths than the parked one keeps).
        for prompt, budget in ((4096, 48), (1, 16), (2049, 5), (300, 64), (300, 3), (300, 2)):
            with self.subTest(prompt=prompt, budget=budget), World() as world:
                fresh = self.fresh_state(world, prompt, budget)
                engine, slot = parked_engine(world)
                # a request served first, so first, the flags and the pages are not the construction's
                served = session('served', 300, 40)
                engine.rebind(served, request_pages(300, 40, first=9))
                for _ in range(3):
                    ticket = served.propose('served', max_rows=1)
                    predictions, _ = engine.verify(ticket)
                    served.commit('served', ticket, predictions, engine.publish)
                self.assertIsNone(engine.park())
                request = session('request', prompt, budget)
                engine.rebind(request, request_pages(prompt, budget))
                rebound = host_state(engine)
                if budget < 4:
                    # A fresh engine captures only the widths its budget holds; the parked one keeps (1, 2, 4) and asks the fresh one's. What
                    # may differ is the captures, never what the request sees.
                    for state in (rebound, fresh):
                        state.pop('buckets'), state.pop('captured_widths'), state.pop('borrowed')
                    self.assertEqual(rebound['widths'], fresh['widths'])
                self.assertEqual(rebound, fresh)
                self.assertEqual(world.ops.violations, [])
                self.assertIs(world.base_engine_module._resident, engine)

    def test_the_capture_position_is_the_one_difference_in_the_buckets(self):
        with World() as world:
            engine, slot = parked_engine(world)
            engine.rebind(session('request', 4096, 48), request_pages(4096, 48))
            self.assertEqual({bucket['capture_position'] for bucket in engine.buckets.values()}, {1})
            self.assertEqual(engine.position, 4096)

    def test_the_engine_is_the_four_card_subclass_and_leaves_the_pair_untouched(self):
        import verifier_engine

        with World() as world:
            engine, slot = parked_engine(world)
            self.assertIsInstance(engine, verifier_engine.VerifierEngine)
            self.assertIs(type(engine), world.engine_module.VerifierEngine)
            for name in ('park', 'park_refusal', 'rebind', 'rebind_refusal'):
                self.assertFalse(hasattr(verifier_engine.VerifierEngine, name), 'the pair\'s engine file is held byte for byte: %s' % name)


class RebindDeviceTests(unittest.TestCase):
    def test_a_rebind_allocates_nothing_and_rewrites_every_page_table_in_full(self):
        from test_parked_tp4_census import payload_digest

        with World() as world:
            engine, slot = parked_engine(world)
            ops = world.ops
            pages = request_pages(4096, 48, first=3)
            allocated, live = ops.allocated_count, set(ops.live)
            ops.tracking = True
            engine.rebind(session('request', 4096, 48), pages)
            ops.tracking = False
            self.assertEqual(ops.allocated_count, allocated, 'no device allocation during a rebind')
            self.assertEqual(set(ops.live), live)
            tables = {}
            for bucket in engine.buckets.values():
                fixture = bucket['fixture']
                for tensor in (fixture.pages, fixture.singleton_pages):
                    tables[tensor.serial] = payload_digest(pages[:, :tensor.shape[1]].repeat(tensor.shape[0], 1).contiguous())
            uploads = [event for event in ops.trail if event[0] == 'upload']
            self.assertEqual(len(uploads), len(tables), 'each table once')
            self.assertEqual({event[1]: event[3] for event in uploads}, tables)
            self.assertTrue(torch.equal(engine.pages, pages))
            # the saves: initial from native slot 0, then the carry
            writes = [event[1] for event in ops.trail if event[0] == 'write' and event[2] == 'copy']
            initial = [value.serial for snapshot in engine.initial for value in snapshot]
            carry = [value.serial for snapshot in engine.carry for value in snapshot]
            self.assertEqual(writes, initial + carry)

    def test_first_and_the_retained_flags_are_reset_and_a_stale_packed_key_dropped(self):
        from gdn_records import RetainedGDNBlock

        with World() as world:
            engine, slot = parked_engine(world)
            engine.phase = 'idle'
            for bucket in engine.buckets.values():
                bucket['first'] = False
                retained = bucket['fixture'].retained
                if retained is not None:
                    retained.selected_prefix, retained.decisions = 2, {0: 1}
                    retained.replay_ready = retained.fence_owed = True
                    retained.commit_serial, retained.replay_epoch = 5, 7
                    retained.replay_fence, retained.replay_fence_ms = 'f9', 3.0
            engine.buckets[('packed', 1)] = dict(rows=16, capture_position=0, checkpoints=[], fixture=None, trace=None,
                                                 output=None, commits={}, first=False, target_features=[])
            engine.phase = 'parked'
            engine.rebind(session('request', 4096, 48), request_pages(4096, 48))
            self.assertNotIn(('packed', 1), engine.buckets)
            self.assertEqual(sorted(engine.buckets), [1, 2, 4])
            reference = retained_flags(RetainedGDNBlock(2, world.ops))
            for bucket in engine.buckets.values():
                self.assertIs(bucket['first'], True)
                retained = bucket['fixture'].retained
                if retained is not None:
                    self.assertEqual(retained_flags(retained), dict(reference, rows=retained.rows))
                    self.assertEqual(retained.replay_epoch, 7, 'a counter, kept')


class RebindWidthTests(unittest.TestCase):
    def test_budgets_under_four_get_the_tickets_a_fresh_engine_would(self):
        with World() as world:
            engine, slot = parked_engine(world)
            for budget in (2, 3, 4, 5, 16):
                request = session('request', 300, budget, drafts=True)
                engine.rebind(request, request_pages(300, budget))
                fresh_widths = world.engine_module.capture_widths(300, PAGE_WIDTH * 64, 16, budget - 1, 4)
                while not request.finished:
                    rows = engine.proposal_rows()
                    self.assertEqual(rows, max(width for width in fresh_widths
                                               if width <= request.max_new_tokens - len(request.emitted)))
                    ticket = request.propose('request', max_rows=rows, selected='dflash2')
                    self.assertEqual(len(ticket.tokens), rows)
                    self.assertTrue(engine.serves(ticket))
                    predictions, _ = engine.verify(ticket)
                    request.commit('request', ticket, predictions, engine.publish)
                self.assertIsNone(engine.park())
                request.close('request')
            self.assertEqual(world.ops.violations, [])

    def test_a_wider_ticket_than_a_cold_engines_is_refused_at_serves_bucket_key_and_verify(self):
        from types import SimpleNamespace as Ticket

        with World() as world:
            engine, slot = parked_engine(world)
            request = session('request', 300, 3, drafts=True)
            engine.rebind(request, request_pages(300, 3))
            self.assertEqual(engine.widths, (1, 2))
            self.assertEqual(engine.captured_widths, (1, 2, 4))
            wide = Ticket(tokens=[1, 2, 3, 4], position=engine.position)
            self.assertFalse(engine.serves(wide))
            with self.assertRaisesRegex(ValueError, 'serves the widths'):
                engine.bucket_key(wide)
            with self.assertRaises(ValueError):
                engine.verify(wide)
            self.assertEqual(engine.phase, 'idle', 'a refused ticket leaves the engine as it was')

    def test_the_widths_negative_control_keeps_the_captured_widths(self):
        with World() as world:
            engine, slot = parked_engine(world)
            engine.keep_captured_widths = True
            engine.rebind(session('request', 300, 3), request_pages(300, 3))
            self.assertEqual(engine.widths, (1, 2, 4))
            self.assertIsNone(engine.request_widths)


class RebindRefusalTests(unittest.TestCase):
    """The rebind's host checks are the constructor's, in its order and with its messages, and run before
    anything is written: a refused engine stays parked and its device untouched."""

    def cases(self):
        finished = session('finished', 300, 1)
        pending = session('pending', 300, 16)
        pending.propose('pending', max_rows=1)
        return (
            ('finished', finished, request_pages(300, 16)),
            ('pending', pending, request_pages(300, 16)),
            ('two tables', session('two', 300, 16), request_pages(300, 16).repeat(2, 1)),
            ('over capacity', session('over', 4300, 100), request_pages(300, 16)),
        )

    def test_the_rebind_refuses_what_the_constructor_refuses_with_its_messages(self):
        with World() as world:
            fresh_slot = world.pool.slots[1]
            fresh_slot.lent = True
            expected = {}
            for name, request, pages in self.cases():
                with self.assertRaises(ValueError) as caught:
                    build_engine(world, request, pages, fresh_slot)
                expected[name] = str(caught.exception)
            fresh_slot.lent = False
            engine, slot = parked_engine(world)
            ops = world.ops
            for name, request, pages in self.cases():
                with self.subTest(case=name):
                    trail = len(ops.trail)
                    ops.tracking = True
                    with self.assertRaises(ValueError) as caught:
                        engine.rebind(request, pages)
                    ops.tracking = False
                    self.assertEqual(str(caught.exception), expected[name])
                    self.assertEqual(engine.phase, 'parked')
                    self.assertEqual(ops.trail[trail:], [], 'nothing touched the device')

    def test_the_host_only_refusal_names_what_the_rebind_would_and_touches_nothing(self):
        with World() as world:
            engine, slot = parked_engine(world)
            ops = world.ops
            trail = len(ops.trail)
            ops.tracking = True
            self.assertIsNone(engine.rebind_refusal(300, 16, (1, PAGE_WIDTH)))
            self.assertIn('captured over a', engine.rebind_refusal(300, 16, (1, PAGE_WIDTH + 4)))
            self.assertIn('one request page table', engine.rebind_refusal(300, 16, (2, PAGE_WIDTH)))
            self.assertIn('fit the request page capacity', engine.rebind_refusal(PAGE_WIDTH * 64 - 4, 100, (1, PAGE_WIDTH)))
            ops.tracking = False
            self.assertEqual(ops.trail[trail:], [], 'nothing was read or written')
            engine.rebind(session('request', 300, 16), request_pages(300, 16))
            self.assertIn('not parked', engine.rebind_refusal(300, 16, (1, PAGE_WIDTH)))

    def test_a_table_of_another_width_and_an_engine_that_is_not_parked_are_refused(self):
        with World() as world:
            engine, slot = parked_engine(world)
            with self.assertRaisesRegex(ValueError, 'captured over a'):
                engine.rebind(session('request', 300, 16), torch.zeros((1, PAGE_WIDTH + 4), dtype=torch.int32))
            engine.rebind(session('request', 300, 16), request_pages(300, 16))
            with self.assertRaisesRegex(ValueError, 'Only a parked engine can be rebound'):
                engine.rebind(session('again', 300, 16), request_pages(300, 16))


class ParkTests(unittest.TestCase):
    def test_park_fences_first_and_clears_the_residency_it_held(self):
        with World() as world:
            engine, slot = parked_engine(world)
            engine.rebind(session('request', 300, 16), request_pages(300, 16))
            self.assertIs(world.base_engine_module._resident, engine)
            fences = world.ops.syncs
            self.assertIsNone(engine.park())
            self.assertEqual(world.ops.syncs, fences + 1)
            self.assertEqual(engine.phase, 'parked')
            self.assertIsNone(engine.session)
            self.assertIsNone(world.base_engine_module._resident)

    def test_a_block_in_flight_is_refused_as_close_refuses_it(self):
        with World() as world:
            engine, slot = parked_engine(world)
            request = session('request', 300, 16)
            engine.rebind(request, request_pages(300, 16))
            ticket = request.propose('request', max_rows=1)
            engine.verify(ticket)
            with self.assertRaisesRegex(ValueError, 'Finish or abort the pending verifier block before closing'):
                engine.park()
            request.abort('request', ticket, engine.publish)
            self.assertIsNone(engine.park())
            with self.assertRaisesRegex(ValueError, 'Only a live engine can park'):
                engine.park()

    def test_an_engine_that_cannot_park_says_why_and_is_left_for_its_owner_to_close(self):
        with World() as world:
            engine, slot = parked_engine(world)
            engine.rebind(session('request', 300, 16), request_pages(300, 16))
            engine.phase = 'failed'
            fences = world.ops.syncs
            self.assertEqual(engine.park(), 'engine phase failed')
            self.assertEqual(world.ops.syncs, fences + 1, 'fenced before deciding')
            self.assertEqual(engine.phase, 'failed')
            engine.phase = 'idle'
            engine.buckets[4]['fixture'].retained.poisoned = True
            self.assertIn('poisoned', engine.park())
            engine.buckets[4]['fixture'].retained.poisoned = False
            self.assertIsNone(engine.park())
            engine.close()
            self.assertEqual(engine.phase, 'closed')


class CarryTraceTests(unittest.TestCase):
    """D2: a parked engine closed at unpark or shutdown releases the carry traces the inherited close would have left (it takes them only in the
    phases idle, preparing and failed)."""

    def test_a_parked_engine_closes_its_carry_traces(self):
        with World() as world:
            engine, slot = parked_engine(world)
            traces = {'save': SimpleNamespace(released=False, label='save'), 'restore': SimpleNamespace(released=False, label='restore')}
            engine.carry_traces = dict(traces)
            engine.close()
            self.assertEqual(engine.phase, 'closed')
            self.assertEqual(engine.carry_traces, False)
            self.assertTrue(all(trace.released for trace in traces.values()), 'every carry trace was released with the parked engine')

    def test_the_inherited_close_still_refuses_a_block_in_flight(self):
        with World() as world:
            engine, slot = parked_engine(world)
            engine.rebind(session('request', 300, 16), request_pages(300, 16))
            engine.phase = 'verified'
            with self.assertRaisesRegex(ValueError, 'Finish or abort'):
                engine.close()
            engine.phase = 'idle'


class PageTableWriteTests(unittest.TestCase):
    def test_the_write_is_refresh_call_for_call(self):
        from serving_page_binding import VerifierPageBinding

        with World() as world:
            engine, slot = parked_engine(world)
            initial = request_pages(300, 16, first=1)
            engine.rebind(session('request', 300, 16), initial)
            blocks = request_blocks(300, 16, first=1)
            binding = VerifierPageBinding(engine, blocks, physical_pages=512)
            grown = blocks + (40, 41)
            ops = world.ops
            ops.tracking, start = True, len(ops.trail)
            self.assertTrue(binding.refresh(grown, position=300, rows=4))
            refreshed = ops.trail[start:]
            host = torch.full((1, PAGE_WIDTH), grown[0], dtype=torch.int32)
            host[0, :len(grown)] = torch.tensor(grown, dtype=torch.int32)
            start = len(ops.trail)
            parked.write_page_tables(ops, world.mesh, parked.page_table_bindings(engine), host)
            written = ops.trail[start:]
            ops.tracking = False
            self.assertGreater(len(refreshed), 6)
            self.assertEqual(written, refreshed)

    def test_the_pages_negative_control_swaps_the_writer_for_this_call_only(self):
        with World() as world:
            engine, slot = parked_engine(world)
            ops = world.ops
            engine.page_table_writer = lambda *arguments: None
            ops.tracking, start = True, len(ops.trail)
            engine.rebind(session('request', 300, 16), request_pages(300, 16, first=5))
            ops.tracking = False
            self.assertEqual([event for event in ops.trail[start:] if event[0] == 'upload'], [], 'no table was written')

    def test_the_bindings_name_one_address_per_chip_and_refuse_a_moved_table_and_other_page_owners(self):
        with World() as world:
            engine, slot = parked_engine(world)
            bindings = parked.page_table_bindings(engine)
            identity, (tensor, shape) = next(iter(bindings.items()))
            self.assertEqual(len(identity), 4, 'one address per chip')
            moved = dict(bindings)
            moved[(1, 2)] = moved.pop(identity)
            with self.assertRaisesRegex(ValueError, 'Captured page metadata addresses changed'):
                parked.write_page_tables(world.ops, world.mesh, moved, engine.pages)
            engine.buckets[1]['fixture'].replay_reader = SimpleNamespace(audit=None)
            with self.assertRaisesRegex(ValueError, 'sequential captures only'):
                parked.page_table_bindings(engine)


if __name__ == '__main__':
    unittest.main()
