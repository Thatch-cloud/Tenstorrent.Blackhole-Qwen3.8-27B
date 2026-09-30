"""C2-any (QWEN_FAST_ANY_REQUEST, plan stage S1 items 1-3 and D4's ignore_eos): the request
factory's Phase 0 switch, the per-request budget, the host-refusal classification, and the
attach-time source check that replaces the per-request T16 qualification.

Every test pairs the flag ON with the same call OFF, because the exact profile (v235) must
stay byte-identical: off, from_prefill must hand the engine, the session, the drafter and the
proposal exactly the arguments it always did."""

from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'speculative-decoding/harness'))

import serving_fast_policy
import serving_request_factory
import serving_runtime
from serving_request_factory import RequestRefused, from_prefill
import test_serving_request_factory
import test_serving_runtime

ON = {'QWEN_FAST_ANY_REQUEST': '1'}
OFF = {'QWEN_FAST_ANY_REQUEST': '0'}


class FlagTests(unittest.TestCase):
    def test_unset_and_zero_are_off_one_is_on_anything_else_is_refused(self):
        self.assertFalse(serving_fast_policy.any_request_enabled({}))
        self.assertFalse(serving_fast_policy.any_request_enabled({'QWEN_FAST_ANY_REQUEST': '0'}))
        self.assertTrue(serving_fast_policy.any_request_enabled({'QWEN_FAST_ANY_REQUEST': '1'}))
        for value in ('true', '2', '', ' 1'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'QWEN_FAST_ANY_REQUEST'):
                serving_fast_policy.any_request_enabled({'QWEN_FAST_ANY_REQUEST': value})

    def test_only_captures_narrower_than_a_replayed_block_count_as_sequential(self):
        self.assertEqual(serving_request_factory.REPLAY_MIN_ROWS, 8)
        for rows, expected in ((None, False), (32, False), (16, False), (8, False), (4, True), (2, True), (1, True)):
            with self.subTest(rows=rows):
                self.assertIs(serving_request_factory.sequential_captures(rows), expected)


class RequestBudgetTests(unittest.TestCase):
    def budget(self, max_tokens, prompt, capacity, ceiling=16384):
        return serving_request_factory.request_budget(SimpleNamespace(max_tokens=max_tokens), prompt_tokens=prompt,
                                                      capacity=capacity, ceiling=ceiling)

    def test_the_request_budget_is_its_max_tokens_within_the_ceiling_and_the_page_room(self):
        self.assertEqual(self.budget(100, 60, 131328), 100)
        self.assertEqual(self.budget(1, 60, 131328), 1)
        self.assertEqual(self.budget(20000, 60, 131328), 16384, 'the server ceiling')
        self.assertEqual(self.budget(16384, 123136, 131328), 8192, 'the page room after the prompt')
        self.assertEqual(self.budget(256, 131072, 131328, ceiling=256), 256, 'the exact shape')

    def test_a_budget_needs_integers_a_positive_max_tokens_and_a_prompt_inside_its_pages(self):
        for max_tokens, prompt, capacity in ((None, 60, 4352), (0, 60, 4352), (True, 60, 4352), (10, 0, 4352),
                                             (10, 4352, 4352), (10, 5000, 4352), (10, 60, 4352.0)):
            with self.subTest(max_tokens=max_tokens, prompt=prompt, capacity=capacity), self.assertRaises(ValueError):
                self.budget(max_tokens, prompt, capacity)


class FactoryTests(unittest.TestCase):
    """from_prefill through the existing factory fixture (fake engine, device and proposal, the
    real GreedySession and DFlashRequestRuntime)."""

    def setUp(self):
        # The attach-time source check has run in this process, as serving_runtime makes sure.
        serving_request_factory._ATTACH_QUALIFICATION.clear()
        serving_request_factory._ATTACH_QUALIFICATION['/attached'] = dict(report_sha256='r')
        self.addCleanup(serving_request_factory._ATTACH_QUALIFICATION.clear)

    def test_on_phase_zero_is_refused_for_the_engine_until_the_attach_check_has_run(self):
        """An image whose serving_runtime predates attach_source_check would otherwise serve
        with no source check at all. Configuration, so a plain ValueError (fatal), before the
        drafter or any other device work - and before the request's own checks, so on such an
        image a request those checks would refuse (the platform's max_tokens=1 warmup reaching
        the page check with a budget of 1, on a lifecycle without D4) still meets THIS error."""
        serving_request_factory._ATTACH_QUALIFICATION.clear()
        for refusal in (dict(), dict(max_tokens=1), dict(blocks=64), dict(max_tokens=0)):
            with self.subTest(refusal=refusal):
                with self.assertRaisesRegex(ValueError, 'attach-time check never ran') as caught:
                    self.build(ON, capture_rows=4, **refusal)
                self.assertNotIsInstance(caught.exception, RequestRefused)
                self.assert_no_device_work()
        # the gate still guards engines that keep it, and off nothing changes
        request, _, _, _ = self.build(ON, capture_rows=None)
        request.close('request')
        request, _, _, _ = self.build(OFF, capture_rows=4)
        request.close('request')

    def build(self, environ, *, capture_rows=4, max_tokens=256, ignore_eos=False, blocks=65, prompt=4096,
              pages=None, seed=None, helpers=48, block_groups=None, frontier=None):
        case = test_serving_request_factory.RequestFactoryTests()
        components, device, engines, arguments = case.fixture()
        state = arguments['state']
        state.sampling_params.max_tokens = max_tokens
        state.sampling_params.ignore_eos = ignore_eos
        state.prompt_token_ids = [1] * prompt
        state.block_ids = (list(range(blocks)),) if block_groups is None else block_groups
        if seed is not None:
            state.output_token_ids = [seed]
        if frontier is not None:
            state.num_computed_tokens = frontier
        device.position = prompt   # the fake drafter sits at the prefilled frontier
        if pages is None:
            pages = torch.tensor([list(range(blocks)) + [0] * (68 - blocks)], dtype=torch.int32)
        extra = {} if capture_rows is None else dict(capture_rows=capture_rows)
        # Recorded before the call, so a test whose build raises can still ask what it touched.
        self.components, self.adopt = components, Mock(wraps=serving_request_factory.adopt_prefill_slot)
        with patch.dict('os.environ', environ), \
                patch('serving_request_factory.device_components',
                      return_value=components) as self.device_components, \
                patch('serving_request_factory.adopt_prefill_slot', self.adopt):
            request = from_prefill(object(), SimpleNamespace(args=SimpleNamespace(vocab_size=100),
                mesh_device=object()), object(), pages, [object()] * helpers, **arguments, **extra)
        return request, components, device, engines

    def assert_no_device_work(self):
        """Nothing past the host checks ran: no slot adoption (the first device write), no
        drafter, no proposal, no engine, no collectives."""
        self.adopt.assert_not_called()
        self.device_components.assert_not_called()
        for name in ('device', 'proposal', 'engine', 'collectives'):
            getattr(self.components, name).assert_not_called()

    def test_off_every_engine_session_and_drafter_argument_is_the_one_it_always_was(self):
        for environ in ({}, OFF):
            for capture_rows in (None, 4):
                with self.subTest(environ=environ, capture_rows=capture_rows):
                    request, components, device, _ = self.build(environ, capture_rows=capture_rows, max_tokens=100,
                                                                ignore_eos=True)
                    options = components.engine.call_args.kwargs
                    self.assertIs(options['attention_replay'], True)
                    self.assertIs(options['target_attention_t16'], True)
                    self.assertEqual(options['replay_group_rows'], 4)
                    self.assertEqual(options['max_verify_rows'], 16)
                    # the server's OUTPUT_BUDGET, not the request's 100
                    self.assertEqual(request.session.max_new_tokens, serving_fast_policy.OUTPUT_BUDGET)
                    self.assertEqual(components.device.call_args.kwargs['max_new_tokens'],
                                     serving_fast_policy.OUTPUT_BUDGET)
                    components.proposal.assert_called_once_with(device, max_new_tokens=serving_fast_policy.OUTPUT_BUDGET)
                    # the snapshot EOS even under ignore_eos, as the gate relies on today
                    self.assertEqual(request.session.eos_ids, (99,))
                    request.close('request')

    def test_on_beside_the_four_user_block_the_engine_drops_replay_and_the_t16_gate(self):
        request, components, _, engines = self.build(ON, capture_rows=4)
        options = components.engine.call_args.kwargs
        self.assertIs(options['attention_replay'], False)
        self.assertIs(options['target_attention_t16'], False)
        self.assertEqual(options['capture_rows'], 4)
        # everything else the engine is given is what it always was
        self.assertEqual((options['replay_group_rows'], options['max_verify_rows'], options['norm_batch'],
                          options['native_sampling_rows'], options['commit_only_gdn']), (4, 16, True, True, True))
        self.assertIs(request.runtime.engine, engines[0])
        request.close('request')

    def test_on_without_the_capture_cap_phase_zero_does_not_apply(self):
        """The runtime refuses this attach (sequential_captures), but the factory on its own
        keeps replay and the gate wherever an engine may capture a replayed width."""
        for capture_rows in (None, 16, 8):
            with self.subTest(capture_rows=capture_rows):
                request, components, _, _ = self.build(ON, capture_rows=capture_rows)
                options = components.engine.call_args.kwargs
                self.assertIs(options['attention_replay'], True)
                self.assertIs(options['target_attention_t16'], True)
                request.close('request')

    def test_on_the_session_drafter_and_proposal_take_the_requests_own_max_tokens(self):
        request, components, device, _ = self.build(ON, max_tokens=100)
        self.assertEqual(request.session.max_new_tokens, 100)
        self.assertEqual(components.device.call_args.kwargs['max_new_tokens'], 100)
        components.proposal.assert_called_once_with(device, max_new_tokens=100)
        request.close('request')

    def test_on_the_budget_stops_at_the_page_room_after_the_prompt(self):
        # 68 pages hold 4352 positions; a 4200-token prompt leaves 152 of the 200 asked for.
        request, components, device, _ = self.build(ON, max_tokens=200, prompt=4200, blocks=68)
        self.assertEqual(request.session.max_new_tokens, 152)
        components.proposal.assert_called_once_with(device, max_new_tokens=152)
        request.close('request')

    def test_on_ignore_eos_leaves_the_session_no_eos_to_stop_at(self):
        request, _, _, _ = self.build(ON, ignore_eos=True)
        self.assertEqual(request.session.eos_ids, ())
        request.close('request')
        request, _, _, _ = self.build(ON, ignore_eos=False)
        self.assertEqual(request.session.eos_ids, (99,))
        request.close('request')

    def test_on_a_refusal_of_the_requests_own_terms_is_its_alone_and_touches_no_device(self):
        """The three host-side checks of the request itself: its sampling contract, its budget
        and its page table."""
        refusals = (
            (dict(max_tokens=0), 'greedy single-sequence'),                   # sampling contract
            (dict(blocks=64), 'Scheduler must own capture warmup pages'),     # page table
        )
        for refusal, message in refusals:
            with self.subTest(refusal=refusal):
                with self.assertRaisesRegex(RequestRefused, message) as caught:
                    self.build(ON, **refusal)
                self.assert_no_device_work()
                self.assertIsInstance(caught.exception, ValueError, 'still a ValueError to every old caller')
                with self.assertRaises(ValueError) as plain:
                    self.build(OFF, **refusal)
                self.assert_no_device_work()
                self.assertNotIsInstance(plain.exception, RequestRefused, 'off, the exception type is unchanged')
                self.assertEqual(str(plain.exception), str(caught.exception), 'the same message either way')

    def test_on_a_budget_the_request_cannot_have_is_its_alone(self):
        """ON only: off, every request's budget is the server's OUTPUT_BUDGET. request_budget
        refuses a prompt that fills its page table; a budget of 1 (max_tokens=1, which D4 ends at
        the lifecycle before it gets here) is the page check's to refuse."""
        for refusal, message in ((dict(prompt=4352, blocks=68), 'prompt inside its page capacity'),
                                 (dict(max_tokens=1), 'T16 request bounds')):
            with self.subTest(refusal=refusal):
                with self.assertRaisesRegex(RequestRefused, message):
                    self.build(ON, **refusal)
                self.assert_no_device_work()

    def test_on_engine_and_runner_invariants_stay_fatal(self):
        """Not the request's terms but evidence the engine is wrong: a bad sampled seed means a
        sampler or device fault, an EOS seed here a lifecycle bug. Plain ValueErrors under the
        flag too - the lifecycle fails the engine rather than let the others decode on it - and
        still before any device work."""
        invariants = (
            (dict(helpers=47), 'all GDN helpers'),
            (dict(frontier=4096), 'frontier inside the prompt'),
            (dict(seed=100), 'Valid target-selected prefill seed'),
            (dict(seed=-1), 'Valid target-selected prefill seed'),
            (dict(seed=99), 'Terminal prefill must finish without allocating a verifier'),
            (dict(block_groups=(list(range(65)), list(range(65)))), 'One scheduler-owned KV group'),
        )
        for invariant, message in invariants:
            for environ in (ON, OFF):
                with self.subTest(invariant=invariant, environ=environ):
                    with self.assertRaisesRegex(ValueError, message) as caught:
                        self.build(environ, **invariant)
                    self.assertIs(type(caught.exception), ValueError)
                    self.assert_no_device_work()

    def test_on_a_device_side_failure_is_not_a_request_refusal(self):
        case = test_serving_request_factory.RequestFactoryTests()
        components, device, _, arguments = case.fixture()
        components.engine.side_effect = ValueError('capture failed')
        with patch.dict('os.environ', ON), \
                patch('serving_request_factory.device_components', return_value=components), \
                self.assertRaises(ValueError) as caught:
            from_prefill(object(), SimpleNamespace(args=SimpleNamespace(vocab_size=100), mesh_device=object()),
                         object(), torch.tensor([list(range(65)) + [0] * 3], dtype=torch.int32), [object()] * 48,
                         capture_rows=4, **arguments)
        self.assertNotIsInstance(caught.exception, RequestRefused)
        device.close.assert_called_once()

    def test_host_refusals_passes_everything_through_when_disabled(self):
        with self.assertRaises(ValueError) as caught:
            with serving_request_factory.host_refusals(False):
                raise ValueError('x')
        self.assertIs(type(caught.exception), ValueError)
        with self.assertRaises(RuntimeError):
            with serving_request_factory.host_refusals(True):
                raise RuntimeError('not a refusal')


class PhaseZeroEngineTests(unittest.TestCase):
    """The real VerifierEngine, built both ways beside the four-user block (captures capped at
    4 rows): replay attention with the T16 gate, as today, and Phase 0 without either. Every
    capture (width and position), every pooled bucket it borrows, and every fixture it builds
    must be the same - only the bucket keys (a width vs a (width, family) pair) and the gate
    differ."""

    class Stop(Exception):
        pass

    def construct(self, **options):
        from greedy_session import GreedySession
        from verifier_engine import VerifierEngine

        session = GreedySession('request', [0] * 4096, 0, vocab_size=100, max_new_tokens=256)
        helpers = [SimpleNamespace(live=[object()] * 5, allocate=Mock(return_value=[object()] * 5), save=Mock(),
                                   restore=Mock()) for _ in range(48)]
        taken, fixtures, seen = [], [], {}

        def take(rows):
            taken.append(rows)
            return SimpleNamespace(target_features=[], mtp_hidden=None, batch=object(),
                                   checkpoints=[[object()] * 5 for _ in range(48)])

        storage = SimpleNamespace(take=take, initial=[[object()] * 5 for _ in range(48)],
                                  carry=[[object()] * 5 for _ in range(48)])

        def fixture(engine, rows, checkpoints, *, retain, position=None, pack=None, storage=None):
            fixtures.append((rows, position, retain, storage is not None))
            return SimpleNamespace(close=Mock())

        def before_capture(engine):
            seen['buckets'] = [(bucket['rows'], bucket['capture_position']) for bucket in engine.buckets.values()]
            seen['keys'] = list(engine.buckets)
            raise self.Stop()

        gate = ModuleType('target_t16_attention_gate')
        gate.validate_request_option, gate.qualify = Mock(), Mock()
        with patch.dict(sys.modules, {'ttnn': SimpleNamespace(synchronize_device=Mock()),
                                      'target_t16_attention_gate': gate}), \
                patch('verifier_engine.addresses', return_value=(1, 2)), patch('verifier_engine.release_owned'), \
                patch.object(VerifierEngine, 'fixture', fixture), self.assertRaises(self.Stop):
            VerifierEngine(SimpleNamespace(mesh_device=object(), layers=[object()] * 64), session,
                           SimpleNamespace(shape=(1, 2052)), helpers, sampler=object(), norm_batch=True,
                           replay_group_rows=4, max_verify_rows=16, native_sampling_rows=True, commit_only_gdn=True,
                           capture_rows=4, storage=storage, before_capture=before_capture, **options)
        return seen, taken, fixtures, gate

    def test_phase_zero_builds_the_same_captures_from_the_same_pooled_widths_without_the_gate(self):
        replayed, replayed_taken, replayed_fixtures, gate = self.construct(attention_replay=True,
                                                                             target_attention_t16=True)
        gate.qualify.assert_called_once()
        self.assertIs(gate.validate_request_option.call_args.args[0], True)
        sequential, sequential_taken, sequential_fixtures, gate = self.construct(attention_replay=False,
                                                                                  target_attention_t16=False)
        gate.qualify.assert_not_called()
        self.assertIs(gate.validate_request_option.call_args.args[0], False)
        self.assertEqual(sequential['buckets'], [(1, 4096), (2, 4096), (4, 4096)])
        self.assertEqual(sequential['buckets'], replayed['buckets'])
        self.assertEqual(sequential_taken, replayed_taken)
        self.assertEqual(sequential_taken, [1, 2, 4])
        self.assertEqual(sequential_fixtures, replayed_fixtures)
        self.assertEqual(replayed['keys'], [(1, None), (2, None), (4, None)])
        self.assertEqual(sequential['keys'], [1, 2, 4])

    def test_phase_zero_proposes_and_serves_the_same_widths_at_every_step_of_the_budget(self):
        """One difference, and it is unreachable: the replay plan also refuses a ticket wider
        than the remaining budget, the plain width lookup does not. The session never drafts
        one (GreedySession.propose caps its ticket at min(max_rows, remaining), and
        proposal_rows is the same either way), so every ticket that can exist is served alike."""
        from attention_request_plan import capture_plan
        from verifier_engine import VerifierEngine

        def engine(plan):
            built = VerifierEngine.__new__(VerifierEngine)
            built.widths, built.replay_plan = (1, 2, 4), plan
            built.buckets = dict.fromkeys([(1, None), (2, None), (4, None)] if plan else [1, 2, 4])
            return built

        plain, replayed = engine(None), engine(capture_plan(4096, 131328, 16, 255, max_verify_rows=4))
        for advanced in range(255):
            session = SimpleNamespace(max_new_tokens=256, emitted=[0] * (1 + advanced), verifier_rows=16)
            plain.session = replayed.session = session
            plain.position = replayed.position = 4096 + advanced
            with self.subTest(advanced=advanced):
                rows = plain.proposal_rows()
                self.assertEqual(rows, replayed.proposal_rows())
                self.assertLessEqual(rows, 255 - advanced)
                for width in (1, 2, 4):
                    ticket = SimpleNamespace(position=4096 + advanced, tokens=(0,) * width)
                    self.assertTrue(plain.serves(ticket))
                    self.assertEqual(replayed.serves(ticket), width <= 255 - advanced)
                    if width <= 255 - advanced:
                        self.assertEqual(plain.bucket_key(ticket), replayed.bucket_key(ticket)[0])

    def test_the_session_never_drafts_a_ticket_wider_than_its_remaining_budget(self):
        from greedy_session import GreedySession

        for budget in (2, 3, 4, 5):
            session = GreedySession('request', [0] * 8, 0, vocab_size=100, max_new_tokens=budget)
            ticket = session.propose('request', max_rows=4)
            self.assertLessEqual(len(ticket.tokens), budget - 1)


class AttachSourceCheckTests(unittest.TestCase):
    def setUp(self):
        serving_request_factory._ATTACH_QUALIFICATION.clear()
        self.addCleanup(serving_request_factory._ATTACH_QUALIFICATION.clear)

    def test_the_same_qualification_runs_once_per_directory_and_its_evidence_is_kept(self):
        evidence = dict(report_sha256='abc', runtime_component_sources={'a.py': '1', 'b.py': '2'})
        qualify, log = Mock(return_value=evidence), Mock()
        self.assertIs(serving_request_factory.attach_source_check('/tree', qualify=qualify, log=log), evidence)
        self.assertIs(serving_request_factory.attach_source_check('/tree', qualify=qualify, log=log), evidence)
        qualify.assert_called_once_with(Path('/tree'))
        log.assert_called_once()
        self.assertIn('attach source check', log.call_args.args[0])
        self.assertEqual(log.call_args.args[2:], ('2 pinned component sources hashed',
                                                  'report_sha256,runtime_component_sources', 'abc'))

    def test_the_marker_reports_what_the_staged_qualify_target_really_returns(self):
        """Staged at the served geometry the gate's qualify is frozen_combined_runtime.qualify_target:
        it runs qualify (which hashes every pinned source) and returns only qualify's 'target'
        evidence plus the target-replay.json pin - no source list. Driven through the real
        qualify_target, so the marker is held to the shape it will print on the rig."""
        import frozen_combined_runtime

        inner = dict(context=32768, target=dict(target_replay_qualified=True, rows=16),
                     runtime_component_sources={'attention_replay.py': '1', 'attention_batch.py': '2'},
                     report_sha256='draft-numerical pin')
        log = Mock()
        with patch.object(frozen_combined_runtime, 'qualify', return_value=inner) as hashed, \
                patch.dict('os.environ', {'QWEN_FROZEN_COMBINED_CONTEXT': '32768'}):
            evidence = serving_request_factory.attach_source_check(
                '/tree', qualify=frozen_combined_runtime.qualify_target, log=log)
        hashed.assert_called_once_with(Path('/tree'))
        target_pin = frozen_combined_runtime.CONTEXT_REPORTS[32768]['target-replay.json']
        self.assertEqual(evidence, dict(target_replay_qualified=True, rows=16, report_sha256=target_pin))
        self.assertNotIn('runtime_component_sources', evidence)
        self.assertEqual(log.call_args.args[2:], ('sources hashed inside qualify, count not returned',
                                                  'report_sha256,rows,target_replay_qualified', target_pin))

    def test_a_mismatch_raises_and_is_not_cached(self):
        qualify = Mock(side_effect=ValueError('Combined runtime component source differs: attention_replay.py'))
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, 'source differs'):
                serving_request_factory.attach_source_check('/tree', qualify=qualify, log=Mock())
        self.assertEqual(qualify.call_count, 2)

    def test_a_qualification_that_returns_no_evidence_is_refused(self):
        for evidence in (None, {}, 'ok'):
            with self.subTest(evidence=evidence), self.assertRaisesRegex(ValueError, 'must return its evidence'):
                serving_request_factory.attach_source_check('/tree', qualify=Mock(return_value=evidence), log=Mock())

    def test_by_default_it_is_the_engines_own_gate_on_the_engines_own_directory(self):
        """The call VerifierEngine made per request: target_t16_attention_gate.qualify(
        Path(verifier_engine.__file__).parent)."""
        gate = ModuleType('target_t16_attention_gate')
        gate.qualify = Mock(return_value=dict(report_sha256='r'))
        engine = ModuleType('verifier_engine')
        engine.__file__ = str(Path('/staged/scripts/ci/verifier_engine.py'))
        with patch.dict(sys.modules, {'target_t16_attention_gate': gate, 'verifier_engine': engine}):
            serving_request_factory.attach_source_check(log=Mock())
        gate.qualify.assert_called_once_with(Path('/staged/scripts/ci'))


class AttachWiringTests(unittest.TestCase):
    def test_on_beside_the_four_user_block_the_attach_runs_the_source_check_once(self):
        for four_as_two in (False, True):
            with self.subTest(four_as_two=four_as_two), \
                    patch.object(serving_runtime, 'attach_source_check') as check:
                test_serving_runtime.RuntimeAttachmentTests().exercise(packed=True, users=4, four_as_two=four_as_two, extra_env=ON)
                check.assert_called_once_with()

    def test_off_the_attach_never_runs_it(self):
        for environ in ({}, OFF):
            with self.subTest(environ=environ), patch.object(serving_runtime, 'attach_source_check') as check:
                test_serving_runtime.RuntimeAttachmentTests().exercise(packed=True, users=4, four_as_two=False, extra_env=environ)
                check.assert_not_called()

    def test_on_with_no_packed_block_the_engines_capture_the_sequential_widths(self):
        """The c2 profile (QWEN_FAST_PACKED_STEP=0): no block is built, the engines are capped at the
        sequential widths (1, 2, 4) instead of refused, and the source check still runs once."""
        for users in (1, 4):
            with self.subTest(users=users), patch.object(serving_runtime, 'attach_source_check') as check:
                test_serving_runtime.RuntimeAttachmentTests().exercise(packed=False, users=users, extra_env=ON)
                check.assert_called_once_with()

    def test_on_an_attach_whose_engines_capture_replayed_widths_is_refused_before_the_pool(self):
        """Beside the two-user 32-row block, which keeps the full T16 captures: the gate would
        still refuse every request that is not the frozen shape, so the attach is refused."""
        for packed, users in ((True, 2),):
            with self.subTest(users=users), patch.object(serving_runtime, 'attach_source_check') as check, \
                    patch.object(serving_runtime, 'ServingBufferPool') as pool, \
                    self.assertRaisesRegex(ValueError, 'QWEN_FAST_ANY_REQUEST=1 needs per-request captures'):
                self.attach(packed, users)
            check.assert_not_called()
            pool.assert_not_called()

    def attach(self, packed, users):
        from contextlib import nullcontext
        from test_serving_fast_policy import FastPolicyTests

        model = SimpleNamespace(args=object(), mesh_device=object(),
            layers=[SimpleNamespace(is_full_attention=False, attention=object()) for _ in range(48)])
        config = FastPolicyTests().fixture()
        config.scheduler_config.max_num_seqs = users
        config.scheduler_config.scheduler_cls = 'named'
        worker = SimpleNamespace(vllm_config=config, model_runner=SimpleNamespace(model=SimpleNamespace(model=[model])))
        env = dict(ON, QWEN_FAST_PACKED_STEP='1' if packed else '0')
        with patch.dict('os.environ', env), patch.dict(sys.modules, {
                'models.common.sampling.generator': SimpleNamespace(SamplingGenerator=Mock()),
                'models.tt_transformers.tt.ccl': SimpleNamespace(TT_CCL=Mock()),
                'gdn_snapshot': SimpleNamespace(ActiveSnapshot=Mock())}), \
                patch.object(serving_runtime, 'pindiag'), \
                patch('sampling_link_policy.sampler_links', side_effect=lambda *args: nullcontext()):
            with serving_runtime.attach_combined_runtime(worker, Mock(), directory='.', runtime_root='.',
                    fixtures=('m', ['l'] * 5, {}, {}), native_attention_evidence='native',
                    block_stream={'streams': 'serial', 'evidence': 'stream'}, kv_publication_evidence='dma',
                    eos_ids=(99,), cancelled=lambda: False):
                self.fail('the attach must be refused')


if __name__ == '__main__':
    unittest.main()
