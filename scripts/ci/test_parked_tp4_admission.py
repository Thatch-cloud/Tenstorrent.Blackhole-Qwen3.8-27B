"""Engine reuse, E5': the parked admission at four chips (design 3.3, 5.5, 5.7; the build table's E5').

  - THE PARKED TERMS (serving_prefill_admission): the free term before the prefill is the prefill transient, the
    rebind's peak R and a released single's rebuild S, with the reserve - nothing is built at admission - and the
    post-prefill backstop's is R + S + the reserve; the contiguous term is today's; the trace term asks the single's
    trace only with S. The measured capture bytes (single, pair, quad) are the ones the coordinator's headroom asks
    under the flag (test_parked_wiring holds the coordinator to them);
  - the term table at the design's section 5.3 readings: at the v70 point (three decoding and a pair) today's terms
    hold a 123,136-token arrival and the parked ones admit it, with or without S; at the worst corner's low reading
    the parked terms admit it without S and hold it with S;
  - flag off: split_short, the predicate, the backstop and the registration are the base commit's, call for call and
    line for line;
  - the parked set's arrival_terms follow its slots (the lowest free slot's terms, today's for an unparked one), and
    the predicate asks them at every call;
  - QWEN_FAST_GATE_DRAM_BALLAST (gate only): parsed strictly before anything is built, held from the end of the
    parked build (after P7p) to its close, in whole tiles and buffers of at most 32 MiB, replicated, never read.
"""

from pathlib import Path
import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import serving_parked_engines as parked  # noqa: E402
import serving_prefill_admission as admission  # noqa: E402
from test_parked_tp4_census import World, base_module  # noqa: E402
from test_parked_tp4_set import make_set  # noqa: E402

MB = 10 ** 6
RESERVE = 256 * 1024 * 1024
LONG = 123136
CPU_WORKFLOW = ROOT / '.github' / 'workflows' / 'qwen-integration-cpu.yml'


class FourCards(unittest.TestCase):
    """The terms are keyed by chip count (the measured capture bytes are the four-card mesh's): this process serves at four chips."""

    def setUp(self):
        patcher = patch.dict(os.environ, {'QWEN_FAST_TP': '4'})
        patcher.start()
        self.addCleanup(patcher.stop)


class TermTests(FourCards):
    def test_the_parked_need_is_the_prefill_the_rebind_a_released_single_and_the_reserve(self):
        rebind, single = admission.PARKED_REBIND_BYTES, admission.measured_single_capture_bytes()
        self.assertEqual(admission.parked_need(LONG, RESERVE, rebind=rebind, single=0),
                         admission.PREFILL_TRANSIENT_BYTES + rebind + RESERVE)
        self.assertEqual(admission.parked_need(LONG, RESERVE, rebind=rebind, single=single),
                         admission.PREFILL_TRANSIENT_BYTES + rebind + single + RESERVE)
        self.assertEqual(admission.parked_need(300, RESERVE, rebind=rebind, single=0), rebind + RESERVE,
                         'a short prompt has no prefill transient, as today')
        self.assertEqual(admission.parked_backstop_need(RESERVE, rebind=rebind, single=single), rebind + single + RESERVE)
        self.assertEqual(admission.parked_trace_need(0), 0)
        self.assertEqual(admission.parked_trace_need(single), admission.single_trace_bytes())
        # design 3.3: 668 MB (+206 with S at four cards) at a 268 MB reserve; 900 MB below today's at the same prompt
        self.assertEqual(round(admission.parked_need(LONG, RESERVE, rebind=rebind, single=0) / MB, 1), 668.4)
        self.assertEqual(round(admission.parked_need(LONG, RESERVE, rebind=rebind, single=single) / MB, 1), 874.7)
        self.assertEqual(admission.dram_need(LONG, RESERVE) - admission.parked_need(LONG, RESERVE, rebind=rebind,
                                                                                    single=0), 900 * MB)
        self.assertEqual(round(admission.parked_backstop_need(RESERVE, rebind=rebind, single=0) / MB, 1), 368.4)

    def test_the_longest_arrival_is_priced_at_the_long_prefill_tier_when_one_is_set(self):
        flat = parked.parked_arrival_need(RESERVE, 256)
        self.assertEqual(flat, admission.parked_need(None, RESERVE, rebind=parked.rebind_peak_bytes(256), single=0))
        with patch.dict(os.environ, {admission.LONG_FROM_FLAG: '65536', admission.LONG_MB_FLAG: '1500'}):
            tiered = parked.parked_arrival_need(RESERVE, 256)
        self.assertEqual(tiered - flat, 1500 * MB - admission.prefill_transient_bytes())
        self.assertGreater(tiered, flat)

    def test_the_constants_are_the_designs(self):
        self.assertEqual(admission.PARKED_REBIND_BYTES, 100 * MB)
        self.assertEqual(admission.PARKED_REBIND_WHOLE_BYTES, 350 * MB)
        # the four-card mesh's measured captures (run v579: single 206.3, pair 187.6-188.2, quad 374.2-375.3 MB per chip; an engine 0.468-0.495 GB)
        self.assertEqual((admission.measured_single_capture_bytes(), admission.measured_pair_capture_bytes(),
                          admission.measured_quad_capture_bytes()), (206_300_000, 188_200_000, 375_300_000))
        self.assertEqual((admission.parked_engine_bytes(), admission.single_trace_bytes()), (490_000_000, 2_500_000))
        # the pair's, keyed by chip count: the TP2 line's constants are what they were
        with patch.dict(os.environ, {'QWEN_FAST_TP': '2'}):
            self.assertEqual((admission.measured_single_capture_bytes(), admission.measured_pair_capture_bytes(),
                              admission.measured_quad_capture_bytes(), admission.single_trace_bytes()),
                             (227 * MB, 212_800_000, 428_300_000, 4 * MB))
        # the set reads them: R by its window, S for a released single
        self.assertEqual(parked.rebind_peak_bytes(256), admission.PARKED_REBIND_BYTES)
        self.assertEqual(parked.rebind_peak_bytes(0), admission.PARKED_REBIND_WHOLE_BYTES)
        self.assertEqual(parked.single_capture_bytes(), admission.measured_single_capture_bytes())
        self.assertEqual(parked.parked_arrival_need(RESERVE, 256, single_released=True),
                         admission.parked_need(LONG, RESERVE, rebind=admission.PARKED_REBIND_BYTES,
                                               single=admission.measured_single_capture_bytes()))

    def test_bad_terms_are_refused(self):
        for rebind, single in ((-1, 0), (0, -1), (1.5, 0), (0, None)):
            with self.subTest(rebind=rebind, single=single), self.assertRaises(ValueError):
                admission.parked_need(LONG, RESERVE, rebind=rebind, single=single)
        with self.assertRaises(ValueError):
            admission.parked_backstop_need(-1, rebind=0, single=0)
        with self.assertRaisesRegex(ValueError, 'trace-region need'):
            admission.split_short(10 ** 10, 10 ** 10, 0, RESERVE, 0, trace_need=-4)
        with self.assertRaisesRegex(ValueError, 'ladder credit'):
            admission.split_short(10 ** 10, 10 ** 10, 0, RESERVE, 0, credit=-1)

    def test_the_ladders_credit_is_the_free_terms_alone(self):
        need, stranded = 900 * MB, admission.STRANDED_BYTES
        short = (need + stranded - 1, 10 ** 10, need, RESERVE, None)
        self.assertEqual(admission.split_short(*short), ('free',))
        self.assertEqual(admission.split_short(*short, credit=1), ())
        low = (10 ** 10, 100 * MB, need, RESERVE, None)
        self.assertEqual(admission.split_short(*low, credit=10 ** 12), ('contiguous',), 'freed buffers are not a larger free block')


class TableTests(FourCards):
    """Design section 5.3's readings, the smallest chip, a 123,136-token arrival, the 256 MiB reserve."""

    def admits(self, free, largest, parked_terms, trace=None):
        pool = SimpleNamespace(dram_statistics=lambda: [dict(free=int(free), largest_free=int(largest))],
                               trace_statistics=(lambda: [dict(largest_free=int(trace))]) if trace is not None else None)
        if trace is None:
            del pool.trace_statistics
        predicate = admission.dram_predicate(pool, RESERVE, **({} if parked_terms is False else
                                                               dict(parked=lambda: parked_terms)))
        return predicate(LONG)

    def test_at_the_v70_point_today_holds_and_the_parked_terms_admit(self):
        free, largest = 1.514e9, 1320.7 * MB
        ok, detail = self.admits(free, largest, False)
        self.assertEqual((ok, detail['short']), (False, ('free',)), 'today: 1568 MB of 1214 MB')
        for single in (0, admission.measured_single_capture_bytes()):
            with self.subTest(single=single):
                ok, detail = self.admits(free, largest, dict(rebind=admission.PARKED_REBIND_BYTES, single=single))
                self.assertTrue(ok, detail)
                self.assertEqual(detail['parked'], dict(rebind=admission.PARKED_REBIND_BYTES, single=single))

    def test_at_the_worst_corners_low_reading_s_decides(self):
        free, largest = 1.06e9, 913.2 * MB
        self.assertFalse(self.admits(free, largest, False)[0])
        ok, _ = self.admits(free, largest, dict(rebind=admission.PARKED_REBIND_BYTES, single=0))
        self.assertTrue(ok, 'about +0.09 GB (section 5.3)')
        ok, detail = self.admits(free, largest, dict(rebind=admission.PARKED_REBIND_BYTES,
                                                     single=admission.measured_single_capture_bytes()))
        self.assertEqual((ok, detail['short']), (False, ('free',)))
        ok, _ = self.admits(free, largest, dict(rebind=admission.PARKED_REBIND_BYTES, single=admission.measured_single_capture_bytes(),
                                                credit=300 * MB))
        self.assertTrue(ok, 'the release ladder\'s credit stands for what it could free')
        ok, _ = self.admits(free, largest, dict(rebind=admission.PARKED_REBIND_WHOLE_BYTES, single=0))
        self.assertFalse(ok, 'the whole projection costs about 0.25 GB more (R = 350 MB)')

    def test_idle_with_every_single_held_admits(self):
        ok, _ = self.admits(1.30e9, 1300 * MB, dict(rebind=admission.PARKED_REBIND_BYTES, single=0))
        self.assertTrue(ok)
        self.assertFalse(self.admits(1.30e9, 1300 * MB, False)[0])

    def test_the_trace_term_asks_the_singles_trace_only_with_s(self):
        free, largest = 3e9, 2000 * MB
        ok, detail = self.admits(free, largest, dict(rebind=admission.PARKED_REBIND_BYTES, single=0), trace=0)
        self.assertTrue(ok, detail)
        ok, detail = self.admits(free, largest, dict(rebind=admission.PARKED_REBIND_BYTES,
                                                     single=admission.measured_single_capture_bytes()), trace=2 * MB)
        self.assertEqual((ok, detail['short']), (False, ('trace',)))
        ok, _ = self.admits(free, largest, dict(rebind=admission.PARKED_REBIND_BYTES,
                                                single=admission.measured_single_capture_bytes()), trace=3 * MB)
        self.assertTrue(ok)
        ok, detail = self.admits(free, largest, False, trace=40 * MB)
        self.assertEqual((ok, detail['short']), (False, ('trace',)), "today's 44 MB")

    def test_the_contiguous_term_is_todays(self):
        ok, detail = self.admits(3e9, 690 * MB, dict(rebind=admission.PARKED_REBIND_BYTES, single=0))
        self.assertEqual((ok, detail['short']), (False, ('contiguous',)))
        self.assertTrue(self.admits(3e9, 697 * MB, dict(rebind=admission.PARKED_REBIND_BYTES, single=0))[0])


class FlagOffTests(FourCards):
    GRID = [(free, largest, trace) for free in (0.9e9, 1.2e9, 1.6e9, 2.4e9, 5e9)
            for largest in (300 * MB, 500 * MB, 700 * MB, 1500 * MB) for trace in (None, 10 * MB, 60 * MB)]

    @classmethod
    def setUpClass(cls):
        cls.base = base_module('serving_prefill_admission')

    def test_split_short_is_the_base_commits(self):
        for free, largest, trace in self.GRID:
            for need in (400 * MB, 1568 * MB):
                arguments = (int(free), int(largest), int(need), RESERVE, None if trace is None else int(trace))
                self.assertEqual(admission.split_short(*arguments), self.base.split_short(*arguments))
                self.assertEqual(admission.split_short(*arguments, contiguous=696 * MB),
                                 self.base.split_short(*arguments, contiguous=696 * MB))

    def test_the_predicate_without_the_set_is_the_base_commits(self):
        for free, largest, trace in self.GRID:
            pool = SimpleNamespace(dram_statistics=lambda free=free, largest=largest: [
                dict(free=int(free), largest_free=int(largest))])
            if trace is not None:
                pool.trace_statistics = lambda trace=trace: [dict(largest_free=int(trace))]
            for prompt in (300, 2048, LONG, None):
                with self.subTest(free=free, largest=largest, trace=trace, prompt=prompt):
                    today = admission.dram_predicate(pool, RESERVE)(prompt)
                    self.assertEqual(today, self.base.dram_predicate(pool, RESERVE)(prompt))
                    self.assertEqual(admission.dram_predicate(pool, RESERVE, parked=lambda: None)(prompt), today,
                                     'an unparked next slot: today\'s terms')
        unread = SimpleNamespace()
        self.assertEqual(admission.dram_predicate(unread, RESERVE)(LONG), self.base.dram_predicate(unread, RESERVE)(LONG))
        with self.assertRaisesRegex(ValueError, 'zero-argument callable'):
            admission.dram_predicate(unread, RESERVE, parked=dict(rebind=1, single=0))

    def backstop(self, module, pool, parked_terms=None, admission_module=admission):
        lines = []
        options = {} if parked_terms is None else dict(parked=parked_terms)
        with patch.dict(sys.modules, {'serving_prefill_admission': admission_module}):
            try:
                result = module.dram_backstop(pool, request_id='r', reserve=RESERVE,
                                              log=lambda template, *values: lines.append(template.format(*values)),
                                              **options)
            except Exception as failure:
                result = (type(failure).__name__, str(failure))
        return result, lines

    def test_the_backstop_without_the_terms_is_the_base_commits(self):
        import serving_request_factory

        fast = base_module('serving_fast_request')
        with patch.dict(sys.modules, {'serving_fast_request': fast, 'serving_prefill_admission': self.base}):
            base = base_module('serving_request_factory')
        for free, largest, trace in self.GRID:
            pool = SimpleNamespace(dram_statistics=lambda free=free, largest=largest: [
                dict(free=int(free), largest_free=int(largest))])
            if trace is not None:
                pool.trace_statistics = lambda trace=trace: [dict(largest_free=int(trace))]
            with self.subTest(free=free, largest=largest, trace=trace):
                self.assertEqual(self.backstop(serving_request_factory, pool),
                                 self.backstop(base, pool, admission_module=self.base))

    def test_the_backstop_asks_the_parked_terms(self):
        import serving_request_factory

        terms = dict(rebind=admission.PARKED_REBIND_BYTES, single=admission.measured_single_capture_bytes())
        need = admission.parked_backstop_need(RESERVE, **terms)
        enough = need + admission.STRANDED_BYTES
        pool = SimpleNamespace(dram_statistics=lambda: [dict(free=enough, largest_free=500 * MB)],
                               trace_statistics=lambda: [dict(largest_free=5 * MB)])
        self.assertEqual(self.backstop(serving_request_factory, pool, terms), (500 * MB, []))
        short = SimpleNamespace(dram_statistics=lambda: [dict(free=enough - 1, largest_free=500 * MB)],
                                trace_statistics=lambda: [dict(largest_free=1 * MB)])
        (kind, message), lines = self.backstop(serving_request_factory, short, terms)
        self.assertEqual(kind, 'RequestRefused')
        self.assertIn('short of free+trace', message)
        self.assertIn('for the parked rebind peak plus the reserve (%d bytes)' % need, message)
        self.assertIn('need=%d short=free+trace' % need, lines[0])
        without = dict(terms, single=0)
        self.assertEqual(self.backstop(serving_request_factory, SimpleNamespace(
            dram_statistics=lambda: [dict(free=enough - 1, largest_free=500 * MB)],
            trace_statistics=lambda: [dict(largest_free=0)]), without)[0], 500 * MB, 'no single to rebuild: no trace term')


class RegistrationTests(FourCards):
    def test_the_predicate_asks_the_sets_terms_at_every_call_and_says_so_once(self):
        import serving_request_factory

        answers = [None, dict(rebind=admission.PARKED_REBIND_BYTES, single=0)]
        engines = SimpleNamespace(arrival_terms=Mock(side_effect=lambda: answers[0]),
                                  arrival_rebind_bytes=Mock(return_value=admission.PARKED_REBIND_BYTES))
        pool = SimpleNamespace(dram_statistics=lambda: [dict(free=int(1.514e9), largest_free=int(1320.7 * MB))])
        lines, modules = [], {}
        with patch('serving_prefill_admission.register_dram_predicate',
                   side_effect=lambda admits: modules.update(admits=admits) or (lambda: None)):
            serving_request_factory.register_dram_admission(
                pool, log=lambda template, *values: lines.append(template.format(*values)), parked=engines)
        admits = modules['admits']
        self.assertFalse(admits(LONG)[0], "today's terms for an unparked next slot")
        answers.pop(0)
        ok, detail = admits(LONG)
        self.assertTrue(ok)
        self.assertEqual(detail['parked'], dict(rebind=admission.PARKED_REBIND_BYTES, single=0))
        self.assertEqual(engines.arrival_terms.call_count, 2)
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[1].startswith(serving_request_factory.DRAM_PARKED_REGISTERED +
                                            'an arrival onto a parked engine needs prefill 300000000 at >= 2048 '
                                            'prompt tokens + rebind 100000000 + a released single\'s rebuild '
                                            '206300000 + reserve 268435456 bytes per chip'), lines[1])

    def test_without_the_set_the_registration_is_the_base_commits_line_for_line(self):
        import serving_request_factory

        base_admission = base_module('serving_prefill_admission')
        fast = base_module('serving_fast_request')
        with patch.dict(sys.modules, {'serving_fast_request': fast, 'serving_prefill_admission': base_admission}):
            base = base_module('serving_request_factory')
        pool = SimpleNamespace(dram_statistics=lambda: [dict(free=int(1.514e9), largest_free=int(1320.7 * MB))])

        def run(module, admission_module):
            lines, registered = [], []
            with patch.dict(sys.modules, {'serving_prefill_admission': admission_module}), \
                    patch.object(admission_module, 'register_dram_predicate',
                                 side_effect=lambda admits: registered.append(admits) or (lambda: None)):
                module.register_dram_admission(pool, log=lambda template, *values: lines.append(template.format(*values)))
            return lines, [admits(prompt) for admits in registered for prompt in (300, LONG)]

        self.assertEqual(run(serving_request_factory, admission), run(base, base_admission))


class ArrivalTermsTests(FourCards):
    def test_the_terms_are_the_lowest_free_slots(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        rebind, single = admission.PARKED_REBIND_BYTES, admission.measured_single_capture_bytes()

        def terms(engines):
            found = engines.arrival_terms()
            return None if found is None else {name: value for name, value in found.items() if name != 'credit'}
        with World(environment={'QWEN_FAST_PARKED_ENGINES': '1'}) as world:
            engines = make_set(world)
            engines.build()
            self.assertEqual(terms(engines), dict(rebind=rebind, single=0))
            self.assertEqual(engines.arrival_terms()['credit'], engines.ladder_credit(engines.slots[0]))
            PackedProposalCoordinator()._release_single_user(engines.slots[0].device)
            self.assertEqual(terms(engines), dict(rebind=rebind, single=single))
            taken = engines.take()
            self.assertIs(taken, engines.slots[0])
            self.assertIs(engines.peek(), engines.slots[1])
            self.assertEqual(terms(engines), dict(rebind=rebind, single=0))
            engines.unpark(engines.slots[1], 'test')
            self.assertIsNone(engines.peek())
            self.assertIsNone(engines.arrival_terms(), "today's build takes slot 1: today's terms")
            taken.state = 'parked'
            self.assertEqual(terms(engines), dict(rebind=rebind, single=single))
            engines.close()
            self.assertIsNone(engines.arrival_terms())

    def test_the_whole_projection_asks_its_own_r(self):
        with World(environment={'QWEN_FAST_PARKED_ENGINES': '1'}) as world:
            engines = make_set(world, environ={parked.PROJECT_ROWS_FLAG: '0'})
            engines.build()
            self.assertEqual({name: value for name, value in engines.arrival_terms().items() if name != 'credit'},
                             dict(rebind=admission.PARKED_REBIND_WHOLE_BYTES, single=0))
            self.assertEqual(engines.arrival_rebind_bytes(), admission.PARKED_REBIND_WHOLE_BYTES)


class RecordingOps:
    bfloat16, TILE_LAYOUT, DRAM_MEMORY_CONFIG = 'bf16', 'tile', 'dram'

    def __init__(self, fail_at=None):
        self.uploads, self.freed, self.fail_at = [], [], fail_at

    def ReplicateTensorToMesh(self, mesh):
        return ('replicate', mesh)

    def from_torch(self, value, **options):
        if self.fail_at is not None and len(self.uploads) == self.fail_at:
            raise RuntimeError('out of DRAM')
        tensor = SimpleNamespace(shape=tuple(value.shape), options=options, nbytes=value.numel() * 2)
        self.uploads.append(tensor)
        return tensor

    def deallocate(self, tensor):
        self.freed.append(tensor)


class BallastTests(unittest.TestCase):
    def test_the_flag_is_a_strict_byte_count(self):
        self.assertEqual(parked.ballast_bytes({}), 0)
        self.assertEqual(parked.ballast_bytes({parked.BALLAST_FLAG: '0'}), 0)
        self.assertEqual(parked.ballast_bytes({parked.BALLAST_FLAG: '1048576'}), 1048576)
        for value in ('1e6', '-1', '', ' 5', '0x10', '01', 'yes'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, parked.BALLAST_FLAG):
                parked.ballast_bytes({parked.BALLAST_FLAG: value})

    def test_it_is_whole_tiles_in_replicated_buffers_of_at_most_32_mib(self):
        ops, mesh = RecordingOps(), object()
        ballast = parked.DramBallast(ops, mesh, 100 * 2 ** 20 + 1)
        self.assertEqual(ballast.size, 100 * 2 ** 20 + 64 * 1024)
        self.assertEqual([tensor.nbytes for tensor in ops.uploads],
                         [32 * 2 ** 20, 32 * 2 ** 20, 32 * 2 ** 20, 4 * 2 ** 20 + 64 * 1024])
        for tensor in ops.uploads:
            self.assertEqual(tensor.shape[:2] + tensor.shape[3:], (1, 1, 1024))
            self.assertEqual(tensor.shape[2] % 32, 0)
            self.assertEqual(tensor.options['mesh_mapper'], ('replicate', mesh))
            self.assertEqual((tensor.options['dtype'], tensor.options['memory_config']), ('bf16', 'dram'))
        ballast.close()
        self.assertEqual(ops.freed, ops.uploads)
        ballast.close()
        self.assertEqual(len(ops.freed), 4, 'once')
        with self.assertRaisesRegex(ValueError, 'positive ballast'):
            parked.DramBallast(ops, mesh, 0)

    def test_a_failed_allocation_frees_what_it_took(self):
        ops = RecordingOps(fail_at=2)
        with self.assertRaisesRegex(RuntimeError, 'out of DRAM'):
            parked.DramBallast(ops, object(), 100 * 2 ** 20)
        self.assertEqual(ops.freed, ops.uploads)

    def test_the_set_holds_it_from_after_p7p_to_its_close_and_refuses_a_bad_value_before_building(self):
        with World(environment={'QWEN_FAST_PARKED_ENGINES': '1'}) as world:
            with self.assertRaisesRegex(ValueError, parked.BALLAST_FLAG):
                make_set(world, environ={parked.BALLAST_FLAG: 'lots'})
            self.assertFalse(any(slot.lent for slot in world.pool.slots))
            engines = make_set(world, environ={parked.BALLAST_FLAG: str(8 * 2 ** 20)})
            order = []
            with patch('memory_ledger.record', side_effect=lambda phase, **kw: order.append((phase, engines.ballast))):
                engines.build()
            self.assertEqual(order, [('P7p', None)], 'the P7p reading is the parked set without the ballast')
            self.assertEqual(engines.ballast.size, 8 * 2 ** 20)
            live = [tensor for tensor in world.ops.live.values() if tensor.shape == (1, 1, 4096, 1024)]
            self.assertEqual(len(live), 1)
            line = [line for line in world.lines if line.startswith(parked.BALLAST_MARKER)]
            self.assertEqual(len(line), 1)
            self.assertRegex(line[0], r'bytes=8388608 buffers=1 \(gate only; unread\) trace_used=')
            self.assertEqual(world.unowned([], [('parked', engines)]), [])
            engines.close()
            self.assertIsNone(engines.ballast)
            self.assertFalse([tensor for tensor in world.ops.live.values() if tensor.shape == (1, 1, 4096, 1024)])
            self.assertEqual(world.ops.violations, [])

    def test_unset_nothing_is_allocated(self):
        with World(environment={'QWEN_FAST_PARKED_ENGINES': '1'}) as world:
            engines = make_set(world)
            with patch.object(parked, 'DramBallast', side_effect=AssertionError('allocated')):
                engines.build()
            self.assertIsNone(engines.ballast)
            self.assertFalse([line for line in world.lines if line.startswith(parked.BALLAST_MARKER)])


class ShippingTests(unittest.TestCase):
    # The image build runs this suite without the repository's .github tree: the allowlist check is a checkout test.
    @unittest.skipUnless(CPU_WORKFLOW.exists(), 'no .github tree (inside the image)')
    def test_the_suite_runs_in_the_cpu_workflow(self):
        self.assertRegex(CPU_WORKFLOW.read_text(encoding='utf-8'), r'python -B -m unittest [^\n]*\btest_parked_tp4_admission\b')


if __name__ == '__main__':
    unittest.main()
