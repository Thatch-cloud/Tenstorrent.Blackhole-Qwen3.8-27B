"""S2 D1 (design section 4, W5): survivor narrowing (W5a) and the abort of a refused packed round
(W5b), with the real GreedySession and, where vLLM is installed, the real scheduler.

A round drafted at the packed block's width that the block then does not serve - a partner aborted
after the drafts, a padded check failed at the step - holds tickets the survivors' own engines never
captured. serving_packed_step.narrow_round cuts each to the widest width its engine serves
(GreedySession.narrow) and the sequential step serves them; refuse_round is left for a ticket nothing
narrower serves, and under the D2 request quarantine it now ends those requests as FINISHED_ABORTED
in the same engine step (serving_packed_step.abort_refused) instead of failing the engine one step
later.

The step-level cases over test_serving_packed_step's fakes live in that module; here:
- NarrowTests: GreedySession.narrow itself.
- RealSessionStepTests: the narrowing inside packed_device_step with real GreedySessions, committed
  through the sequential step the way FastRequest.step commits.
- InstalledVllmNarrowingTests: vLLM's own scheduler takes a narrowed commit below the width it
  scheduled (rolled back like rejected drafts, frontier agreeing with the worker's), and a RUNNING
  request that returns ZERO tokens and is registered with the quarantine ends FINISHED_ABORTED with
  its blocks freed while its partner decodes on. Skipped where vLLM is not installed (the CPU suite);
  qwen-fast-vllm-cpu.yml runs it on vLLM 0.25.1 with the scheduler class the engine builds,
  TTScheduler(AsyncScheduler)."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'speculative-decoding/harness'))

from greedy_session import BlockTicket, GreedySession  # noqa: E402
from serving_fast_request import CommittedOutput  # noqa: E402
import serving_request_quarantine as quarantine  # noqa: E402

PROMPT = (0, 1, 2) * 12


def neural(request_id, history, count):
    """A feature drafter standing in for DFlash: `count` proposals continuing the history."""
    return tuple((history[-1] + 1 + index) % 100 for index in range(count))


def session(request_id='request', budget=64, lookup=False):
    """A session drafting full block-width tickets through `neural` (selected 'dflash2', as
    FastRequest.prepare selects it); with `lookup`, test_greedy_session's lookup-first session over the
    repeating prompt instead, whose tickets carry a match length."""
    return GreedySession(request_id, PROMPT, 0, vocab_size=100, max_new_tokens=budget,
                         neural=None if lookup else {'dflash2': neural}, lookup_enabled=lookup)


def draft(live, rows=16):
    return live.propose(live.request_id, max_rows=rows, selected='dflash2')


class NarrowTests(unittest.TestCase):
    def test_the_narrowed_ticket_keeps_the_frontier_seed_and_leading_proposals_under_a_new_epoch(self):
        live = session()
        drafted = draft(live)
        self.assertEqual((len(drafted.tokens), drafted.source), (16, 'dflash2'))
        for rows in (8, 4, 2):
            with self.subTest(rows=rows):
                before = live.pending
                narrowed = live.narrow('request', before, rows)
                self.assertIsInstance(narrowed, BlockTicket)
                self.assertEqual(narrowed, BlockTicket('request', before.epoch + 1, before.position, before.tokens[:rows],
                                                       'dflash2', 0))
                self.assertIs(live.pending, narrowed)
                self.assertEqual((live.phase, live.epoch), ('pending', before.epoch + 1))
                self.assertEqual(narrowed.tokens[0], live.seed)
        # a single row is the target's own step, as propose makes it
        target = live.narrow('request', live.pending, 1)
        self.assertEqual((target.tokens, target.source, target.match_length), ((live.seed,), 'target', 0))
        # the same width re-issues the ticket (a new epoch, same tokens)
        again = live.narrow('request', target, 1)
        self.assertEqual((again.tokens, again.epoch), (target.tokens, target.epoch + 1))

    def test_a_lookup_tickets_match_length_is_clamped_to_the_proposals_it_keeps(self):
        live = session(lookup=True)
        drafted = live.propose('request', max_rows=16)
        self.assertEqual(drafted.source, 'lookup')
        self.assertGreater(drafted.match_length, 1)
        narrowed = live.narrow('request', drafted, 2)
        self.assertEqual((narrowed.source, narrowed.match_length, narrowed.tokens), ('lookup', 1, drafted.tokens[:2]))

    def test_refusals_change_nothing(self):
        live = session()
        drafted = draft(live, 8)
        stale = BlockTicket(*[getattr(drafted, name) for name in ('request_id', 'epoch', 'position', 'tokens', 'source')])
        cases = {'a wider width': ('request', drafted, 16), 'a width that is no bucket': ('request', drafted, 3),
                 'zero rows': ('request', drafted, 0), 'a bool': ('request', drafted, True),
                 'another request': ('other', drafted, 4), 'a ticket that is not the live one': ('request', stale, 4)}
        for name, arguments in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                live.narrow(*arguments)
            self.assertEqual((live.pending, live.phase, live.epoch), (drafted, 'pending', drafted.epoch))
        # an idle, committing, failed or closed session has no live ticket to narrow
        live.commit('request', drafted, tuple(drafted.tokens[1:]) + (5,), lambda prefix: None)
        with self.assertRaises(ValueError):
            live.narrow('request', drafted, 4)
        ticket = draft(live, 4)
        live.fail_verification('request', ticket)
        with self.assertRaises(ValueError):
            live.narrow('request', ticket, 2)

    def test_the_narrowed_ticket_commits_at_most_its_rows_and_the_old_one_is_dead(self):
        for rows in (1, 2, 4, 8):
            with self.subTest(rows=rows):
                live = session()
                drafted = draft(live)
                narrowed = live.narrow('request', drafted, rows)
                with self.assertRaises(ValueError):
                    live.commit('request', drafted, drafted.tokens[1:] + (5,), lambda prefix: None)
                position, emitted = live.position, list(live.emitted)
                published = []
                # the target agrees with every proposal the cut kept, then corrects
                decision = live.commit('request', narrowed, tuple(narrowed.tokens[1:]) + (5,),
                                       lambda prefix: published.append(prefix))
                self.assertEqual(published, [rows])
                self.assertEqual(decision.emitted, tuple(narrowed.tokens[1:]) + (5,))
                self.assertEqual((live.position, live.emitted), (position + rows, emitted + list(decision.emitted)))
                self.assertEqual((live.phase, live.pending, live.committed_blocks), ('idle', None, 1))
                self.assertEqual(live.committed_block_proposals, rows - 1)
                # and the next proposal is an ordinary one at the new frontier
                self.assertEqual(draft(live, 4).position, position + rows)

    def test_a_narrowed_ticket_can_still_be_aborted_or_failed(self):
        live = session()
        narrowed = live.narrow('request', draft(live), 4)
        live.abort('request', narrowed, lambda prefix: None)
        self.assertEqual((live.phase, live.pending, live.aborted_blocks), ('idle', None, 1))
        narrowed = live.narrow('request', draft(live), 2)
        live.fail_verification('request', narrowed)
        self.assertEqual(live.phase, 'failed')
        live.close('request')


class RealSessionStepTests(unittest.TestCase):
    """packed_device_step over test_serving_packed_step's fake block and engines, with real
    GreedySessions: a four-user round with one partner gone is narrowed to the engines' widest
    capture and each survivor commits through its own sequential step."""

    def requests(self, count, widths=(1, 2, 4)):
        from test_serving_packed_step import FakeBlock, FakeEngine, FakeRuntime

        block = FakeBlock(users=4)
        made, stepped = [], []
        for segment in range(count):
            live = session('user%d' % segment)
            engine = FakeEngine(live, widths=widths)
            block.bind(engine, segment)
            request = SimpleNamespace(session=live, engine=engine, runtime=FakeRuntime(engine), closed=False,
                                      busy=False, cancelled=False, collect_timings=False, rows=[])

            def step(request_id, *, cancelled, request=request):
                ticket = request.session.pending
                stepped.append(request_id)
                request.rows.append(len(ticket.tokens))
                if not request.engine.serves(ticket):
                    raise ValueError('a ticket its own engine never captured')
                decision = request.session.commit(request_id, ticket, tuple(ticket.tokens[1:]) + (5,),
                                                  lambda prefix: None)
                return CommittedOutput(request_id, tuple(decision.emitted), request.session.position,
                                       request.session.finished)

            request.step = step
            draft(request.session)
            made.append(request)
        return block, made, stepped

    def test_survivors_of_a_finished_or_aborted_partner_are_narrowed_and_each_commits(self):
        import io
        from unittest.mock import patch

        from serving_packed_step import packed_device_step

        for count in (3, 2, 1):
            with self.subTest(live=count):
                block, requests, stepped = self.requests(count)
                drafted = [request.session.pending for request in requests]
                self.assertEqual({len(ticket.tokens) for ticket in drafted}, {16})
                entries = [dict(request_id=request.session.request_id, request=request, ticket=request.session.pending)
                           for request in requests]
                with patch.dict(sys.modules, loguru=None), patch('sys.stdout', new_callable=io.StringIO) as output:
                    outputs = packed_device_step(entries, cancelled=lambda: False, block=block)
                self.assertEqual(block.calls, [])
                self.assertEqual(stepped, [request.session.request_id for request in requests])
                for request, ticket, committed in zip(requests, drafted, outputs):
                    self.assertEqual(request.rows, [4])
                    self.assertEqual(committed.token_ids, tuple(ticket.tokens[1:4]) + (5,))
                    self.assertEqual((request.session.phase, request.session.position), ('idle', len(PROMPT) + 4))
                    self.assertEqual(request.session.epoch, ticket.epoch + 1)
                self.assertEqual(output.getvalue().count('[PINDIAG] packed survivor narrowed'), count)

    def test_the_widest_capture_wins_and_a_servable_ticket_is_left_alone(self):
        import io
        from unittest.mock import patch

        from serving_packed_step import packed_device_step

        block, requests, stepped = self.requests(2, widths=(1, 2, 4, 8))
        entries = [dict(request_id=request.session.request_id, request=request, ticket=request.session.pending)
                   for request in requests]
        with patch.dict(sys.modules, loguru=None), patch('sys.stdout', new_callable=io.StringIO):
            packed_device_step(entries, cancelled=lambda: False, block=block)
        self.assertEqual([request.rows for request in requests], [[8], [8]])
        # an engine serving the block width itself is never narrowed (the round is only ineligible)
        block, requests, stepped = self.requests(2, widths=(1, 2, 4, 8, 16))
        epochs = [request.session.epoch for request in requests]
        entries = [dict(request_id=request.session.request_id, request=request, ticket=request.session.pending)
                   for request in requests]
        packed_device_step(entries, cancelled=lambda: False, block=block)
        self.assertEqual([request.rows for request in requests], [[16], [16]])
        self.assertEqual([request.session.epoch for request in requests], epochs)


def vllm_installed():
    try:
        import vllm  # noqa: F401
    except ImportError:
        return False
    return True


def build_scheduler(scheduler_type, max_num_seqs=1):
    """test_serving_scheduler.RealSchedulerTests.scheduler's configuration (4352 positions, 15 draft
    tokens, 64-token blocks, no prefix caching), with `max_num_seqs` requests allowed at once."""
    from tempfile import TemporaryDirectory

    import torch
    from transformers import GPT2Config
    from vllm.config import (CacheConfig, DeviceConfig, ModelConfig, ParallelConfig, SchedulerConfig,
                             SpeculativeConfig, VllmConfig)
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.structured_output import StructuredOutputManager

    temporary = TemporaryDirectory()
    directory = temporary.name
    GPT2Config(n_positions=8192, n_embd=256, n_layer=1, n_head=4).save_pretrained(directory)
    model = ModelConfig(model=directory, dtype='float32', max_model_len=4352, skip_tokenizer_init=True, seed=0)
    speculative = SpeculativeConfig(model='ngram', num_speculative_tokens=15)
    speculative.method = 'dflash'
    config = VllmConfig(model_config=model, device_config=DeviceConfig(device='cpu'),
        scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs, max_num_batched_tokens=4352,
            max_model_len=4352, is_encoder_decoder=False, enable_chunked_prefill=False,
            async_scheduling=False, watermark=0.0),
        cache_config=CacheConfig(block_size=64, enable_prefix_caching=False),
        parallel_config=ParallelConfig(), speculative_config=speculative)
    config.cache_config.num_gpu_blocks = 128
    cache = KVCacheConfig(num_blocks=128, kv_cache_tensors=[], kv_cache_groups=[
        KVCacheGroupSpec(['layer'], FullAttentionSpec(block_size=64, num_kv_heads=2, head_size=256,
                                                      dtype=torch.bfloat16))])
    register_all_kvcache_specs(config)
    scheduler = scheduler_type(config, cache, StructuredOutputManager(config), block_size=64)
    scheduler.use_v2_model_runner = False
    scheduler._qwen_temporary = temporary
    return scheduler


def runner_view(prompt, outputs):
    """The TT runner's view of one resident request as serving_vllm_state reads it, at the frontier
    vLLM holds after `outputs` (the last output token not yet computed)."""
    import numpy as np

    state = SimpleNamespace(prompt_token_ids=list(prompt), output_token_ids=list(outputs),
                            num_computed_tokens=len(prompt) + len(outputs) - 1)
    tokens = np.full((1, 4352 + 16), -1, dtype=np.int64)
    tokens[0, :len(prompt) + len(outputs)] = [*prompt, *outputs]
    batch = SimpleNamespace(num_reqs=1, req_id_to_index={'request': 0}, req_output_token_ids=[state.output_token_ids],
                            token_ids_cpu=tokens, num_tokens=np.array([len(prompt) + len(outputs)]),
                            num_computed_tokens_cpu=np.array([state.num_computed_tokens]))
    runner = SimpleNamespace(requests={'request': state}, input_batch=batch,
                             model_config=SimpleNamespace(max_model_len=4352, get_vocab_size=lambda: 1000))
    return runner, state


class InstalledVllmNarrowingTests(unittest.TestCase):
    """vLLM 0.25.1's own scheduler: a narrowed commit and a refused round's zero-token abort."""

    def setUp(self):
        if not vllm_installed():
            self.skipTest('vLLM is not installed')
        sys.modules.pop(quarantine.HOLDER_KEY, None)
        self.addCleanup(sys.modules.pop, quarantine.HOLDER_KEY, None)

    def scheduler_types(self):
        """vLLM's Scheduler, the fixture TTScheduler(AsyncScheduler) the engine builds on the TT
        platform, and the installed plugin's own where this environment has it - each a fresh
        subclass, so wrapping one for the quarantine never leaks."""
        from vllm.v1.core.sched.scheduler import Scheduler

        from test_serving_request_quarantine import plugin_scheduler_class

        classes = [('vLLM Scheduler', type('Scheduler', (Scheduler,), {})), ('fixture TTScheduler', plugin_scheduler_class())]
        try:
            from vllm_tt_plugin.scheduler import TTScheduler
        except ImportError:
            pass
        else:
            classes.append(('installed plugin TTScheduler', type('TTScheduler', (TTScheduler,), {})))
        return classes

    @staticmethod
    def decode_step(scheduler, request_ids, drafts):
        """Hand every request its 15 drafts and schedule its 16-token verify."""
        from vllm.v1.outputs import DraftTokenIds

        scheduler.update_draft_token_ids(DraftTokenIds(list(request_ids), [list(drafts)] * len(request_ids)))
        return scheduler.schedule()

    def test_the_scheduler_takes_a_narrowed_commit_below_the_scheduled_width(self):
        """The round was scheduled at 16 tokens (the seed and 15 drafts) for the packed block; the
        survivor was narrowed to `width` rows and committed `committed` <= width tokens. vLLM rolls
        the rest back as rejected drafts: its frontier and output tokens are the worker's
        (serving_vllm_state.apply_committed_output), and the next 16-token round is admitted at
        that frontier by the fast path's own contract."""
        from vllm.sampling_params import SamplingParams
        from vllm.v1.request import Request, RequestStatus

        from serving_vllm_contract import admit_scheduler_output
        from serving_vllm_packed import packed_model_runner_output
        from serving_vllm_state import apply_committed_output

        prompt, drafts = [42] * 4096, list(range(101, 116))
        for source, scheduler_type in self.scheduler_types():
            for width in (1, 2, 4):
                for committed in range(1, width + 1):
                    with self.subTest(scheduler=source, width=width, committed=committed):
                        scheduler = build_scheduler(scheduler_type)
                        request = Request('request', prompt, SamplingParams(temperature=0, max_tokens=256), None)
                        scheduler.add_request(request)
                        scheduler.update_from_output(scheduler.schedule(), packed_model_runner_output(
                            [CommittedOutput('request', (100,), 4096, False)]))
                        scheduled = self.decode_step(scheduler, ['request'], drafts)
                        self.assertEqual(scheduled.num_scheduled_tokens, {'request': 16})
                        self.assertEqual(scheduled.scheduled_spec_decode_tokens, {'request': drafts})
                        # the narrowed ticket (100, 101, ..) kept width - 1 drafts; the target accepted
                        # committed - 1 of them and corrected the next
                        tokens = (*drafts[:committed - 1], 55)
                        output = CommittedOutput('request', tokens, 4096 + committed, False)
                        runner, state = runner_view(prompt, [100])
                        apply_committed_output(runner, state, output)
                        scheduler.update_from_output(scheduled, packed_model_runner_output([output]))
                        self.assertEqual(request.status, RequestStatus.RUNNING)
                        self.assertEqual(list(request.output_token_ids), [100, *tokens])
                        self.assertEqual(request.num_computed_tokens, 4096 + committed)
                        self.assertEqual((state.output_token_ids, state.num_computed_tokens),
                                         (list(request.output_token_ids), request.num_computed_tokens))
                        self.assertEqual(getattr(request, 'num_output_placeholders', 0), 0)
                        following = self.decode_step(scheduler, ['request'], drafts)
                        self.assertEqual(list(following.scheduled_cached_reqs.num_computed_tokens), [4096 + committed])
                        ticket = SimpleNamespace(request_id='request', position=4096 + committed, tokens=(55, *drafts))
                        prepared = SimpleNamespace(closed=False, cancelled=False, busy=False,
                                                   engine=SimpleNamespace(phase='idle'), session=SimpleNamespace(
                                                       phase='pending', pending=ticket, request_id='request',
                                                       position=4096 + committed))
                        self.assertIs(admit_scheduler_output(prepared, following), ticket)

    def test_a_zero_token_running_request_registered_with_the_quarantine_ends_finished_aborted(self):
        """W5b: refuse_round returns ZERO tokens for its requests - RUNNING decodes, not prefills -
        and registers them with the quarantine. vLLM builds no EngineCoreOutput for a request with
        no tokens (no rollback, no stop check); the wrapper finishes it as FINISHED_ABORTED
        anyway, adds its finishing output (the client's finish_reason 'abort') and frees its
        blocks, in the same update_from_output, while a partner in the same step decodes on; the
        next step names it in finished_req_ids and schedules only the partner; its seat takes a
        new prompt."""
        from vllm.sampling_params import SamplingParams
        from vllm.v1.engine import FinishReason
        from vllm.v1.request import Request, RequestStatus

        from serving_vllm_packed import packed_model_runner_output

        drafts = list(range(101, 116))
        for source, scheduler_type in self.scheduler_types():
            for partner in (False, True):
                with self.subTest(scheduler=source, partner=partner):
                    sys.modules.pop(quarantine.HOLDER_KEY, None)
                    scheduler = build_scheduler(scheduler_type, max_num_seqs=2)
                    pool = scheduler.kv_cache_manager.block_pool
                    free = pool.get_num_free_blocks()
                    names = ['A', 'B'] if partner else ['A']
                    requests = {name: Request(name, [42 + index] * 1024, SamplingParams(temperature=0, max_tokens=256),
                                              None) for index, name in enumerate(names)}
                    for request in requests.values():
                        scheduler.add_request(request)
                    prefill = scheduler.schedule()
                    self.assertEqual(set(prefill.num_scheduled_tokens), set(names), 'both prompts in one prefill step')
                    scheduler.update_from_output(prefill, packed_model_runner_output(
                        [CommittedOutput(name, (100,), 1024, False) for name in prefill.num_scheduled_tokens]))
                    scheduled = self.decode_step(scheduler, names, drafts)
                    self.assertEqual(scheduled.num_scheduled_tokens, {name: 16 for name in names})
                    log = Mock()
                    quarantine.install(SimpleNamespace(scheduler_config=SimpleNamespace(scheduler_cls=scheduler_type)),
                                       log=log)
                    quarantine.register('A', 'packed round refused: test')
                    committed = [CommittedOutput('A', (), 1024, True, True)]
                    if partner:
                        committed.append(CommittedOutput('B', (101, 102, 55), 1027, False))
                    outputs = scheduler.update_from_output(scheduled, packed_model_runner_output(committed))
                    finishing = [output for batch in outputs.values() for output in batch.outputs if output.request_id == 'A']
                    self.assertEqual(len(finishing), 1, 'the wrapper added the zero-token request its output')
                    self.assertEqual((finishing[0].new_token_ids, finishing[0].finish_reason), ([], FinishReason.ABORT))
                    self.assertEqual(requests['A'].status, RequestStatus.FINISHED_ABORTED)
                    self.assertNotIn('A', scheduler.requests)
                    self.assertEqual(quarantine.holder().pending, {})
                    held = len(scheduler.kv_cache_manager.get_block_ids('B')[0]) if partner else 0
                    self.assertEqual(pool.get_num_free_blocks(), free - held, "A's blocks are free again")
                    if partner:
                        progressed = [output for batch in outputs.values() for output in batch.outputs
                                      if output.request_id == 'B']
                        self.assertEqual((progressed[0].new_token_ids, progressed[0].finish_reason), ([101, 102, 55], None))
                        self.assertEqual((requests['B'].status, requests['B'].num_computed_tokens),
                                         (RequestStatus.RUNNING, 1027))
                    following = self.decode_step(scheduler, ['B'] if partner else [], drafts) if partner else \
                        scheduler.schedule()
                    self.assertEqual(following.finished_req_ids, {'A'})
                    self.assertEqual(following.num_scheduled_tokens, {'B': 16} if partner else {})
                    if partner:
                        self.assertEqual(list(following.scheduled_cached_reqs.num_computed_tokens), [1027])
                    else:
                        replacement = Request('C', [43] * 1024, SamplingParams(temperature=0, max_tokens=256), None)
                        scheduler.add_request(replacement)
                        self.assertEqual(scheduler.schedule().num_scheduled_tokens, {'C': 1024})

    def test_the_control_without_the_registration_leaves_the_zero_token_request_running(self):
        """VR4 Q3, the reason W5b needs the quarantine: the same zero-token step, unregistered, gets
        no output, no rollback and no stop check - the request stays RUNNING with its frontier
        advanced by the 16 scheduled tokens. TTScheduler (an AsyncScheduler) schedules it again
        past its real frontier (where refuse_round's failed session then kills the engine at the
        draft); vLLM's plain Scheduler never schedules it again at all."""
        from vllm.sampling_params import SamplingParams
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler
        from vllm.v1.request import Request, RequestStatus

        from serving_vllm_packed import packed_model_runner_output

        drafts = list(range(101, 116))
        for source, scheduler_type in self.scheduler_types():
            with self.subTest(scheduler=source):
                scheduler = build_scheduler(scheduler_type)
                request = Request('A', [42] * 1024, SamplingParams(temperature=0, max_tokens=256), None)
                scheduler.add_request(request)
                scheduler.update_from_output(scheduler.schedule(), packed_model_runner_output(
                    [CommittedOutput('A', (100,), 1024, False)]))
                scheduled = self.decode_step(scheduler, ['A'], drafts)
                outputs = scheduler.update_from_output(scheduled, packed_model_runner_output(
                    [CommittedOutput('A', (), 1024, True, True)]))
                self.assertEqual([output for batch in outputs.values() for output in batch.outputs
                                  if output.request_id == 'A'], [])
                self.assertEqual((request.status, request.num_computed_tokens), (RequestStatus.RUNNING, 1024 + 16))
                self.assertIn('A', scheduler.requests)
                following = self.decode_step(scheduler, ['A'], drafts)
                self.assertEqual(following.finished_req_ids, set(), 'never finished')
                if issubclass(scheduler_type, AsyncScheduler):
                    self.assertEqual(following.num_scheduled_tokens, {'A': 16}, 'scheduled again past its real frontier')
                else:
                    self.assertEqual(following.num_scheduled_tokens, {}, 'wedged: its frontier is past its tokens')


if __name__ == '__main__':
    unittest.main()
