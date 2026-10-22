"""Sampled speculative text equals plain sampled text THROUGH THE PACKED READBACK PATH (QWEN_FAST_SAMPLED, docs/tp4-sampled-verify.md), pure CPU.

The real GreedySession runs several requests at once in a 64-row block, each with its own spec and seed. Every round the fake mesh holds the toy model's logits for the rows of the
tickets (the 248,320-token vocabulary over four 62,080-column shards), the block's trace outputs are built by the real sampled_verify.device_slice (stock-topk semantics on the
fake device), read back through the batched-reads layer, merged, drawn (packed_predictions) and committed by the real session. The committed text of every request must equal one-
token-at-a-time sampling with its seed (the plain sampler, which calls the reference path's own functions on the full-vocabulary logits), whatever the drafter, the draft widths,
which segments are idle, who finishes first, the order of the entries, a greedy request beside them.

Also here: flag-off byte-identity of every hook (the gate against an unchanged copy of the old function, the packed block's operation, the request factory, the lifecycle) and the
greedy request in a sampled block keeping the argmax path.
"""
import contextlib
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / 'speculative-decoding' / 'harness'))

from greedy_session import BlockTicket, GreedySession  # noqa: E402
import packed_verifier  # noqa: E402
from packed_shapes import m3_shape  # noqa: E402
import sampled_accept as sa  # noqa: E402
import sampled_test_support as support  # noqa: E402
import sampled_verify as sv  # noqa: E402
from tp_test_support import four_cards  # noqa: E402

SHAPE = m3_shape(1024)
ROWS = 16
SEGMENTS = 4


def no_drafter(request_id, history, count):
    return ()


class Request(object):
    """One request of the block: the real session (token ids are vocabulary ids), its prompt, its spec and key, its drafter."""

    def __init__(self, name, prompt, count, spec, seed, drafter):
        self.name, self.prompt, self.count, self.spec, self.seed, self.drafter = name, list(prompt), count, spec, seed, drafter
        self.sampling = None if spec.greedy else sv.Sampling(spec, seed)
        first = support.toy_sample(self.prompt, spec, seed)
        self.session = GreedySession(name, tuple(self.prompt), first, vocab_size=support.VOCAB, max_new_tokens=count, eos_ids=(support.EOS,),
                                     neural={'dflash2': no_drafter}, lookup_enabled=False)
        self.session.sampling = self.sampling
        self.ticket = None

    def propose(self, width):
        history = self.prompt + self.session.emitted
        drafts = list(self.drafter(history, width - 1))
        if len(drafts) < width - 1:
            # the neural drafter always fills the block's width (a lookup only replaces rows it finds): a function of the committed prefix and the drafts so far
            drafts += support.peaked_drafter(history + drafts, width - 1 - len(drafts))
        tokens = (self.session.seed,) + tuple(drafts)
        session = self.session
        session.epoch += 1
        self.ticket = BlockTicket(session.request_id, session.epoch, session.position, tokens[:width], 'dflash2', 0)
        session.pending, session.phase = self.ticket, 'pending'
        return self.ticket

    def rows(self):
        """The model's logits for each row of the ticket: after the committed prefix and the ticket's tokens up to the row."""
        history = self.prompt + self.session.emitted
        return [support.full_row(history + list(self.ticket.tokens[1:i + 1])) for i in range(len(self.ticket.tokens))]


def greedy_text(prompt, count):
    context, out = list(prompt), []
    while len(out) < count:
        token = int(np.argmax(support.full_row(context)))
        out.append(token)
        context.append(token)
        if token == support.EOS:
            break
    return out


class PackedRoundRunner(object):
    """Drives rounds through the packed readback path: stages the logits of every live entry's rows in its segment, builds the trace outputs, reads, merges, draws, commits."""

    def __init__(self, test, requests, rng):
        self.test, self.requests, self.rng = test, requests, rng
        self.rounds = self.sampled_rounds = self.idle_rounds = 0

    def round(self):
        live = [request for request in self.requests if not request.session.finished]
        segments = self.rng.sample(range(SEGMENTS), len(live))
        array = np.full((SEGMENTS * ROWS, support.VOCAB), -9.0, dtype=np.float32)
        array += np.random.default_rng(self.rounds).normal(size=array.shape).astype(np.float32) * 0.01      # the idle segments' rows hold finite junk
        entries = []
        for request, segment in zip(live, segments):
            ticket = request.propose(ROWS)
            for index, row in enumerate(request.rows()):
                array[segment * ROWS + index] = row
            entries.append(dict(request=SimpleNamespace(session=request.session), ticket=ticket, request_id=request.name))
        ops = support.Ops()
        logits = support.logits_tensor(array)
        outputs = sv.device_slice(ops, logits, SEGMENTS * ROWS)
        block = SimpleNamespace(output=(logits, None, None, None, *outputs), operations=ops, mesh='mesh', block_rows=SEGMENTS * ROWS, rows_per_user=ROWS, shape=SHAPE,
                                users=SEGMENTS, shard_audit=False)
        host = [int(token) for token in torch.from_numpy(array).bfloat16().float().numpy().argmax(axis=1)]
        predictions = [host[segment * ROWS:(segment + 1) * ROWS] for segment in segments]
        out, metrics = sv.packed_predictions(block, entries, segments, host, predictions, None)
        self.test.assertEqual(len(out), len(entries))
        for request, rows in zip(live, out):
            decision = request.session.commit(request.name, request.ticket, rows, lambda state_rows: None)
            self.test.assertGreaterEqual(len(decision.emitted), 1)
        self.rounds += 1
        self.sampled_rounds += metrics.get('entries', 0) > 0
        self.idle_rounds += len(live) < SEGMENTS
        return len(live)

    def run(self):
        while any(not request.session.finished for request in self.requests):
            self.round()
        return {request.name: list(request.session.emitted) for request in self.requests}


class Clean(unittest.TestCase):
    def setUp(self):
        sv.reset()
        self.addCleanup(sv.reset)
        patcher = mock.patch.dict(os.environ, support.clean_environment(), clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        logger = mock.patch.object(sv, 'log_line')
        logger.start()
        self.addCleanup(logger.stop)
        context = four_cards()
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)


class TextTests(Clean):
    PROMPTS = ([phi for phi in map(support.phi, (5, 17, 3, 3, 40))], [support.phi(token) for token in (60, 2, 9, 9, 9, 9, 12, 60, 2, 9)], [support.phi(1)])

    def test_every_drafter_and_spec_commits_the_plain_sampled_text_in_a_shared_block(self):
        names = ('approx', 'peaked', 'hash', 'copy')
        for seed_base in (3, 4):
            requests = []
            for index, name in enumerate(names):
                spec = support.SPECS[(index + seed_base) % len(support.SPECS)]
                requests.append(Request('r%d' % index, self.PROMPTS[index % len(self.PROMPTS)], 40 + 7 * index, spec, (seed_base << 40) + 977 * index + 1,
                                        support.DRAFTERS[name]))
            runner = PackedRoundRunner(self, requests, random.Random(seed_base))
            texts = runner.run()
            for request in requests:
                expected = (support.plain_run(request.prompt, request.count, request.seed, request.spec) if not request.spec.greedy
                            else greedy_text(request.prompt, request.count))
                self.assertEqual(texts[request.name], expected, (seed_base, request.name, request.spec))
            self.assertGreater(runner.sampled_rounds, 3)

    def test_the_coupling_is_exercised_drafts_are_both_accepted_and_corrected(self):
        request = Request('r', self.PROMPTS[0], 60, support.SPECS[1], 99, support.approx_drafter)
        runner = PackedRoundRunner(self, [request], random.Random(1))
        runner.run()
        session = request.session
        self.assertGreater(session.accepted_proposals, 0)
        self.assertLess(session.accepted_proposals, session.committed_block_proposals)
        self.assertEqual(list(session.emitted), support.plain_run(request.prompt, 60, 99, support.SPECS[1]))

    def test_idle_segments_and_early_finishers_change_nothing(self):
        requests = [Request('short', self.PROMPTS[0], 6, support.SPECS[0], 11, support.approx_drafter),
                    Request('long', self.PROMPTS[1], 90, support.SPECS[1], 12, support.peaked_drafter),
                    Request('mid', self.PROMPTS[2], 30, support.SPECS[2], 13, support.approx_drafter)]
        runner = PackedRoundRunner(self, requests, random.Random(5))
        texts = runner.run()
        self.assertGreater(runner.idle_rounds, 0)
        for request in requests:
            self.assertEqual(texts[request.name], support.plain_run(request.prompt, request.count, request.seed, request.spec), request.name)

    def test_the_text_does_not_depend_on_the_segment_a_request_lands_in_or_who_else_is_live(self):
        texts = []
        for seed, crowd in ((1, 0), (2, 0), (3, 2)):
            requests = [Request('r', self.PROMPTS[1], 50, support.SPECS[1], 321, support.approx_drafter)]
            requests += [Request('x%d' % index, self.PROMPTS[index], 20 + 9 * index, support.SPECS[index], 7 + index, support.approx_drafter) for index in range(crowd)]
            texts.append(PackedRoundRunner(self, requests, random.Random(seed)).run()['r'])
        self.assertEqual(texts[0], texts[1])
        self.assertEqual(texts[0], texts[2])
        self.assertEqual(texts[0], support.plain_run(self.PROMPTS[1], 50, 321, support.SPECS[1]))

    def test_a_different_seed_gives_a_different_text(self):
        first = Request('a', self.PROMPTS[0], 40, support.SPECS[2], 1, support.approx_drafter)
        second = Request('b', self.PROMPTS[0], 40, support.SPECS[2], 2, support.approx_drafter)
        texts = PackedRoundRunner(self, [first, second], random.Random(0)).run()
        self.assertNotEqual(texts['a'], texts['b'])

    def test_a_greedy_request_in_a_sampled_block_keeps_the_argmax_path(self):
        greedy = Request('g', self.PROMPTS[0], 40, support.SPECS[3], 5, support.peaked_drafter)
        sampled = Request('s', self.PROMPTS[1], 40, support.SPECS[1], 6, support.peaked_drafter)
        self.assertIsNone(greedy.sampling)
        runner = PackedRoundRunner(self, [greedy, sampled], random.Random(2))
        texts = runner.run()
        self.assertEqual(texts['g'], greedy_text(self.PROMPTS[0], 40))
        self.assertEqual(texts['s'], support.plain_run(self.PROMPTS[1], 40, 6, support.SPECS[1]))

    def test_a_block_of_greedy_requests_reads_nothing_back_and_commits_the_greedy_text(self):
        requests = [Request('g%d' % index, self.PROMPTS[index], 30, support.SPECS[3], index, support.peaked_drafter) for index in range(3)]
        runner = PackedRoundRunner(self, requests, random.Random(3))
        texts = runner.run()
        self.assertEqual(runner.sampled_rounds, 0)
        for request in requests:
            self.assertEqual(texts[request.name], greedy_text(request.prompt, 30))


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# flag off: every hook is the code that ran before
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def old_validate_request_sampling(parameters, *, prompt_tokens, eos_ids=()):
    """serving_fast_policy.validate_request_sampling as it stood at the base of the sampled-verify branch (the condition, verbatim)."""
    import serving_fast_policy as policy

    primary_eos = getattr(parameters, '_eos_token_id', None)
    if (any(type(token) is not int or token < 0 for token in eos_ids)
            or any(type(token) is not int or token not in eos_ids for token in (parameters.stop_token_ids or ()))
            or (primary_eos is not None and primary_eos not in eos_ids)):
        raise ValueError('Only target-snapshot EOS stop tokens are supported')
    if (type(prompt_tokens) is not int or prompt_tokens < 1
            or prompt_tokens > policy.MAXIMUM_PROMPT_TOKENS
            or parameters.temperature != 0 or parameters.n != 1
            or type(parameters.max_tokens) is not int
            or not 1 <= parameters.max_tokens <= policy.OUTPUT_BUDGET
            or parameters.min_tokens != 0
            or not isinstance(parameters.ignore_eos, bool)
            or parameters.logprobs is not None or parameters.prompt_logprobs is not None
            or parameters.presence_penalty != 0 or parameters.frequency_penalty != 0
            or parameters.repetition_penalty != 1
            or parameters.stop
            or getattr(parameters, 'structured_outputs', None) is not None
            or getattr(parameters, 'logit_bias', None)
            or getattr(parameters, 'allowed_token_ids', None)
            or getattr(parameters, 'bad_words', None)):
        raise ValueError('Fast serving requires a greedy single-sequence request with an '
                         'unpenalised sampler and at most %d output tokens' % policy.OUTPUT_BUDGET)


def request_params(**values):
    base = dict(temperature=0, n=1, max_tokens=256, min_tokens=0, ignore_eos=False, logprobs=None, prompt_logprobs=None, presence_penalty=0, frequency_penalty=0,
                repetition_penalty=1, stop=None, stop_token_ids=None, structured_outputs=None, logit_bias=None, allowed_token_ids=None, bad_words=None, top_p=1.0, top_k=0,
                min_p=0.0, seed=None, extra_args=None)
    base.update(values)
    return SimpleNamespace(**base)


def outcome(function, params, **options):
    try:
        function(params, **options)
    except ValueError as error:
        return 'refused', str(error)
    return 'ok', None


class FlagOffTests(Clean):
    def test_the_engine_side_gate_is_the_old_function_call_for_call(self):
        import serving_fast_policy as policy

        rng = random.Random(7)
        for switch in (None, '0'):
            environ = support.clean_environment({} if switch is None else {sv.FLAG: switch})
            with mock.patch.dict(os.environ, environ, clear=True):
                for _ in range(600):
                    values = dict(temperature=rng.choice((0, 0.0, 0.3, 1.0)), n=rng.choice((1, 1, 2)), max_tokens=rng.choice((1, 256, 10 ** 6, None)),
                                  min_tokens=rng.choice((0, 0, 2)), ignore_eos=rng.choice((False, True, 0)), logprobs=rng.choice((None, None, 1)),
                                  presence_penalty=rng.choice((0, 0, 0.5)), frequency_penalty=rng.choice((0, 0, -1)), repetition_penalty=rng.choice((1, 1, 1.2)),
                                  stop=rng.choice((None, None, ['x'])), stop_token_ids=rng.choice((None, [], [248046], [7])), bad_words=rng.choice((None, None, ['x'])),
                                  extra_args=rng.choice((None, {sv.SEED_KEY: 5})), seed=rng.choice((None, 3)))
                    options = dict(prompt_tokens=rng.choice((None, 0, 1, 100, 10 ** 7)), eos_ids=rng.choice(((), (248046, 248044))))
                    params = request_params(**values)
                    self.assertEqual(outcome(policy.validate_request_sampling, params, **options), outcome(old_validate_request_sampling, params, **options), (switch, values, options))

    def test_the_gate_imports_nothing_with_the_switch_off(self):
        import serving_fast_policy as policy

        with mock.patch.dict(sys.modules, {'sampled_verify': None}):
            self.assertFalse(policy.sampled_request(request_params(temperature=1.0)))
            policy.validate_request_sampling(request_params(), prompt_tokens=100)
            with self.assertRaises(ValueError):
                policy.validate_request_sampling(request_params(temperature=1.0), prompt_tokens=100)

    def test_the_refusal_message_is_the_old_one_with_the_switch_off_and_says_why_with_it_on(self):
        import serving_fast_policy as policy

        sampled = request_params(temperature=1.0, extra_args={sv.SEED_KEY: 3}, top_k=8)
        with self.assertRaises(ValueError) as off:
            policy.validate_request_sampling(sampled, prompt_tokens=100)
        self.assertNotIn('QWEN_FAST_SAMPLED', str(off.exception))
        with mock.patch.dict(os.environ, {sv.FLAG: '1'}):
            with self.assertRaisesRegex(ValueError, 'QWEN_FAST_SAMPLED=1: a sampled request also needs the attach'):
                policy.validate_request_sampling(sampled, prompt_tokens=100)

    def test_the_gate_admits_a_well_formed_sampled_request_only_after_the_attach(self):
        import serving_fast_policy as policy

        sampled = request_params(temperature=0.7, top_p=0.9, top_k=8, extra_args={sv.SEED_KEY: 3})
        with mock.patch.dict(os.environ, {sv.FLAG: '1'}):
            with self.assertRaises(ValueError):
                policy.validate_request_sampling(sampled, prompt_tokens=100)
            sv.attach([SimpleNamespace(sampled=True)])
            policy.validate_request_sampling(sampled, prompt_tokens=100)
            for broken in (dict(extra_args=None), dict(seed=4), dict(presence_penalty=0.1), dict(logprobs=1), dict(n=2), dict(stop=['a']), dict(top_k=0), dict(min_p=0.2)):
                with self.subTest(broken=broken), self.assertRaises(ValueError):
                    policy.validate_request_sampling(request_params(**dict(vars(sampled), **broken)), prompt_tokens=100)
            policy.validate_request_sampling(request_params(), prompt_tokens=100)        # greedy: unchanged

    def test_the_request_factory_attaches_nothing_with_the_switch_off(self):
        import serving_request_factory as factory

        request = SimpleNamespace(session=SimpleNamespace())
        with mock.patch.dict(sys.modules, {'sampled_verify': None}):
            factory.attach_sampling(request, request_params(temperature=0.9))
        self.assertFalse(hasattr(request.session, 'sampling'))
        with mock.patch.dict(os.environ, {sv.FLAG: '1'}):
            factory.attach_sampling(request, request_params(temperature=0.9, top_k=8, extra_args={sv.SEED_KEY: 3}))
            self.assertEqual(request.session.sampling, sv.Sampling(sa.SamplingSpec(0.9, 1.0, 8), 3))
            factory.attach_sampling(request, request_params())
            self.assertIsNone(request.session.sampling)
            with self.assertRaises(ValueError):
                factory.attach_sampling(request, request_params(temperature=0.9, top_k=8))

    def engine(self, sampled, with_slices=False):
        events = []
        ids, values = support.Tensor([torch.zeros(1, 1, 64, 1)] * 4), support.Tensor([torch.zeros(1, 1, 64, 1)] * 4)
        logits = support.logits_tensor(np.zeros((64, support.VOCAB), dtype=np.float32))

        class Capture(object):
            @contextlib.contextmanager
            def capture(self):
                events.append('capture')
                yield

        engine = SimpleNamespace(feature_capture=Capture(), shard_argmax=True, sampled=sampled, block_rows=64, operations=support.Ops(), sampler=None,
                                 sample_shards=lambda logits, warm=False: (ids, values))
        engine.sampled_outputs = lambda logits, shards: packed_verifier.PackedVerifierEngine.sampled_outputs(engine, logits, shards)
        fixture = SimpleNamespace(run=lambda sharded_logits=False: logits)
        return engine, fixture, (logits, ids, values), events

    def test_the_blocks_operation_is_unchanged_without_the_sampled_readback(self):
        engine, fixture, expected, events = self.engine(sampled=False)
        with mock.patch.dict(sys.modules, {'sampled_verify': None}):
            result = packed_verifier.PackedVerifierEngine.operation(engine, fixture)
        self.assertEqual(result, expected)
        self.assertEqual(events, ['capture'])
        self.assertEqual(engine.operations.topks, [])

    def test_with_the_readback_the_trace_outputs_are_the_old_ones_plus_the_slice_at_the_end(self):
        engine, fixture, expected, _ = self.engine(sampled=True)
        result = packed_verifier.PackedVerifierEngine.operation(engine, fixture)
        self.assertEqual(result[:3], expected)
        self.assertIsNone(result[3], 'the pinned sampler\'s reference slot stays where the audit reads it')
        self.assertEqual(len(result), 4 + 8)
        self.assertEqual(sv.tensors_of(SimpleNamespace(output=result, block_rows=64)), result[4:])
        self.assertEqual(engine.operations.topks, [((1, 1, 32, 32768), 64)] * 4)

    def test_a_refusing_build_releases_the_shard_outputs_and_raises(self):
        engine, fixture, expected, _ = self.engine(sampled=True)
        engine.operations = support.Ops(k_limit=32)
        with mock.patch.object(packed_verifier, 'release_owned') as release, self.assertRaisesRegex(sv.SampledRefused, 'refused k=64'):
            packed_verifier.PackedVerifierEngine.operation(engine, fixture)
        self.assertEqual(release.call_args_list[0][0][1], list(expected[1:]), 'the shard argmax outputs')
        self.assertEqual(engine.operations.freed[-1], expected[0], 'and the logits, as any failing operation frees them')

    def test_describe_names_the_readback_only_when_it_is_on(self):
        source = (HERE / 'packed_verifier.py').read_text(encoding='utf-8')
        self.assertIn("**({} if not getattr(self, 'sampled', False) else dict(sampled=", source)

    def test_the_block_refuses_what_cannot_take_the_readback(self):
        base = dict(shard_argmax=True, shard_problem=None, block_rows=64, operations=support.Ops())
        self.assertIsNone(sv.block_problem(SimpleNamespace(**base)))
        self.assertIn('shard argmax', sv.block_problem(SimpleNamespace(**dict(base, shard_argmax=False, shard_problem='a request seed is active'))))
        self.assertIn('a request seed is active', sv.block_problem(SimpleNamespace(**dict(base, shard_argmax=False, shard_problem='a request seed is active'))))
        self.assertIn('rows', sv.block_problem(SimpleNamespace(**dict(base, block_rows=48))))
        self.assertIn('no topk', sv.block_problem(SimpleNamespace(**dict(base, operations=SimpleNamespace(slice=lambda: 0, deallocate=lambda t: None)))))
        with mock.patch.dict(os.environ, {'QWEN_FAST_TP': '2'}):
            self.assertIn('four-card', sv.block_problem(SimpleNamespace(**base)))


if __name__ == '__main__':
    unittest.main()
