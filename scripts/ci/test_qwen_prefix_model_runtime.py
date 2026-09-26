"""The G1 model graft, executed: exact reuse, the F1/F2/F3/S7 guards, and nothing changed when off.

The REAL model.py and qwen36_vllm.py (the IMG bytes qwen_prefix_model_patch pins), stock and
staged, run on a recording fake ttnn with a toy hybrid model whose chunk program has section
2.0.4's dependency structure (qwen_prefix_model_fixture). The registry is the P0a prototype of
qwen_prefix_registry, driven as the scheduler graft drives it. What is held:

  exact      a hit - chained over 7 turns, tail-only, at previous-prompt lengths 2047/2048/2049,
             after an early divergence (an older checkpoint), from a gap capture another
             conversation took, and four rows in one step - equals a cold run of the same prompt on
             fresh blocks: logits, KV [0, L) through the page table, and the GDN slot, byte for
             byte, on the traced and the eager loop;
  sensitive  the harness sees what the design says breaks exactness: no conv_carry, no restore;
             and the unguarded eager tail (Lever N's loop alone) raises ttnn.deallocate(None);
  guards     start_pos > 0 without a registry, a request id, a committed grant, a current grant,
             the row's grant or matching tokens is an assertion before any chunk runs (F2); a
             resumed row on the unbatched path asserts; a restore before warmup asserts (F3);
  warmup     the restore path is chosen on the plugin's compile-only call and compiles nothing on a
             hit (copy_host_to_device_tensor when it round-trips, else ttnn.copy warmed), once;
  capture    a MemoryError while capturing skips the checkpoint, counts it, and the request is exact
             (S7); a registry refusal is reported; unreachable plan positions are dropped;
  audit      QWEN_PREFIX_AUDIT=1 digests match between hit and cold, and the audit only reads;
  off        with QWEN_PREFIX_REUSE unset the staged files make exactly the stock files' ttnn calls
             and results (prefill and warmup, traced and eager), and the capability is False.
"""

import contextlib
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import prefix_scheduler_graft as graft  # noqa: E402
import qwen_prefix_model_fixture as F  # noqa: E402
import qwen_prefix_model_patch as patcher  # noqa: E402

ON = {'QWEN_PREFIX_REUSE': '1'}


@contextlib.contextmanager
def prefix_env(env):
    """Exactly these QWEN_PREFIX_* settings, whatever the shell running the tests has set."""
    with mock.patch.dict(os.environ, env):
        for key in [key for key in os.environ if key.startswith('QWEN_PREFIX_') and key not in env]:
            del os.environ[key]
        yield


@contextlib.contextmanager
def registry_scope(registry):
    with mock.patch.dict(sys.modules):
        sys.modules.pop(patcher.REGISTRY_KEY, None)
        if registry is None:
            yield None
        else:
            with F.registry_installed(registry):
                yield registry


class Engine(object):
    """One engine process: a model (stock or staged), its vLLM wrapper, a registry and a block pool."""

    def __init__(self, traced=False, environ=ON, sources=None, registry=True, h2d_refuse=False):
        self.fake = F.FakeTTNN()
        self.fake.h2d_refuse = h2d_refuse
        self.log = F.FakeLogger()
        model_source, vllm_source = sources or F.staged_sources()
        self.module = F.load_source(model_source, 'qwen36_model_under_test', F.model_stubs(self.fake, self.log))
        self.vllm = F.load_source(vllm_source, 'qwen36_vllm_under_test', F.vllm_stubs(self.fake, self.log), environ)
        self.toy = F.Toy(self.module, self.fake, traced=traced)
        self.model = self.toy.model
        self.wrapper = F.build_wrapper(self.vllm, self.toy)
        self.registry = graft.PrefixRegistry(budget_bytes=1 << 30) if registry else None
        self.pool = F.Pool()
        self.environ = dict(environ)

    def warm(self, enable_trace=False):
        with prefix_env(self.environ):
            self.wrapper.warmup_model_prefill(self.model._paged_kv_caches, enable_trace)

    def prefill(self, rows, req_ids=True, environ=None):
        """rows: (req_id, tokens, page_row, start_pos, slot). The runner's kwargs, as submit_prefill
        builds them plus the runner patch's request ids."""
        width = max(len(tokens) for _, tokens, _, _, _ in rows)
        batch = torch.zeros(len(rows), width, dtype=torch.int64)
        for index, (_, tokens, _, _, _) in enumerate(rows):
            batch[index, :len(tokens)] = tokens
        kwargs = dict(empty_slots=[row[4] for row in rows], start_pos=[row[3] for row in rows])
        if req_ids:
            kwargs[patcher.REQ_IDS_KWARG] = [row[0] for row in rows]
        env = dict(self.environ)
        env.update(environ or {})
        with registry_scope(self.registry), prefix_env(env):
            logits, _ = self.wrapper.prefill_forward(
                batch, torch.cat([row[2] for row in rows], dim=0), None, [len(row[1]) for row in rows], **kwargs)
        return logits

    def turn(self, conv, tokens, q, plan=(), slot=0, h=None, shared_row=None):
        """One admitted turn: the grant the scheduler commits, a page row sharing the first Q/64
        blocks vLLM's cache returned, the prefill. Returns (logits, row, slot)."""
        source = shared_row if shared_row is not None else conv.get('row')
        shared = source[0, :q // F.BLOCK].tolist() if q else []
        row = self.pool.row(len(tokens), shared)
        F.admit(self.registry, conv['id'], tokens, q, plan, h=h)
        logits = self.prefill([(conv['id'], tokens, row, q, slot)])
        conv['row'] = row
        return logits, row, slot

    def cold(self, tokens, slot=3, req='cold'):
        """The salted cold arm: same engine, no grant, fresh blocks."""
        row = self.pool.row(len(tokens))
        if self.registry is not None:
            self.registry.begin_step()
        return self.prefill([(req, tokens, row, 0, slot)]), row, slot

    def rows_prefix(self):
        return self.log.lines(patcher.MARKER_ROW)


class ExactTestBase(unittest.TestCase):
    traced = False

    def engine(self, **kwargs):
        kwargs.setdefault('traced', self.traced)
        engine = Engine(**kwargs)
        engine.warm()
        return engine

    def run_turn_and_cold(self, engine, conv, tokens, q, plan=(), **kwargs):
        hit = engine.turn(conv, tokens, q, plan, **kwargs)
        hit_slot = [(rec.clone(), [c.clone() for c in convs]) for rec, convs in engine.toy.slot(hit[2])]
        cold = engine.cold(tokens, req='cold-%s-%d' % (conv['id'], len(tokens)))
        self.assertTrue(torch.equal(hit[0], cold[0]), 'logits differ at L=%d Q=%d' % (len(tokens), q))
        for a, b in zip(engine.toy.kv(hit[1], len(tokens)), engine.toy.kv(cold[1], len(tokens))):
            self.assertTrue(torch.equal(a, b), 'KV differs at L=%d Q=%d' % (len(tokens), q))
        for a, b in zip(hit_slot, engine.toy.slot(cold[2])):
            self.assertTrue(torch.equal(a[0], b[0]) and all(torch.equal(x, y) for x, y in zip(a[1], b[1])),
                            'GDN slot differs at L=%d Q=%d' % (len(tokens), q))
        return hit, cold


class EagerExactness(ExactTestBase):
    """trace_mode decode_only: the eager loop C1 reuses (design 2.2, the eager exactness gate)."""

    traced = False

    def test_a_seven_turn_chain_equals_a_cold_run_at_every_turn(self):
        engine = self.engine()
        base = F.prompt(12000, seed=1)
        conv, q, resumed = {'id': 'conv'}, 0, []
        for length in (3000, 5500, 7300, 8500, 9000, 11000, 11500):
            boundary = length // F.CHUNK * F.CHUNK
            plan = [boundary] if boundary > q else []
            engine.toy.segments.clear()
            self.run_turn_and_cold(engine, conv, base[:length], q, plan)
            # The hit ran only from Q: its first segment starts there (a tail-only hit: the tail).
            resumed.append(engine.toy.segments[0][1])
            if plan:
                q = plan[0]
        self.assertEqual(resumed, [0, 2048, 4096, 6144, 8192, 8192, 10240])
        markers = engine.rows_prefix()
        self.assertTrue(any('Q=8192 L=9000' in line and 'captured=[]' in line for line in markers))
        self.assertTrue(any('Q=10240 L=11500' in line for line in markers))
        self.assertEqual(engine.registry.stats['grants'], 6)

    def test_previous_prompt_lengths_2047_2048_2049(self):
        for first, second, expect_q in ((2047, 3000, 0), (2048, 2700, 2048), (2049, 4200, 2048)):
            with self.subTest(first=first):
                engine = self.engine()
                tokens = F.prompt(5000, seed=first)
                conv = {'id': 'c%d' % first}
                plan = [first // F.CHUNK * F.CHUNK] if first >= F.CHUNK else []
                self.run_turn_and_cold(engine, conv, tokens[:first], 0, plan)
                q = expect_q
                if q:
                    self.assertIsNotNone(engine.registry.get(F.key_at(tokens[:second].tolist(), q)))
                self.run_turn_and_cold(engine, conv, tokens[:second], q,
                                       [second // F.CHUNK * F.CHUNK] if second // F.CHUNK * F.CHUNK > q else [])

    def test_an_early_divergence_falls_back_to_an_older_checkpoint(self):
        engine = self.engine()
        base = F.prompt(9000, seed=7)
        conv, q = {'id': 'div'}, 0
        for length in (2500, 4500, 6500):
            boundary = length // F.CHUNK * F.CHUNK
            self.run_turn_and_cold(engine, conv, base[:length], q, [boundary])
            q = boundary
        edited = base[:8000].clone()
        edited[5000:] = F.prompt(3000, seed=8)       # the template rewrote history after 5000
        self.assertIsNone(engine.registry.get(F.key_at(edited.tolist(), 6144)))
        self.run_turn_and_cold(engine, conv, edited, 4096, [6144])

    def test_a_gap_capture_serves_a_third_conversation_on_the_first_conversations_blocks(self):
        engine = self.engine()
        system = F.prompt(5000, seed=11)
        a = torch.cat([system, F.prompt(3000, seed=12)])
        b = torch.cat([system, F.prompt(3500, seed=13)])
        c = torch.cat([system, F.prompt(2500, seed=14)])
        conv_a, conv_b, conv_c = {'id': 'A'}, {'id': 'B'}, {'id': 'C'}
        self.run_turn_and_cold(engine, conv_a, a, 0, [6144])
        # B shares 4992 cached tokens with A but no checkpoint below them: Q = 0, and the gap
        # boundary floor2048(h) = 4096 is planned alongside B's own last boundary.
        self.run_turn_and_cold(engine, conv_b, b, 0, [4096, 8192], h=4992)
        self.assertIn('captured=[4096:stored', engine.rows_prefix()[-2])
        # C: Q = 4096 from B's gap checkpoint; KV [0, 4096) are the blocks A cached first.
        self.run_turn_and_cold(engine, conv_c, c, 4096, [6144], h=4992, shared_row=conv_a['row'])

    def test_four_mixed_rows_in_one_step(self):
        engine = self.engine()
        base = F.prompt(12000, seed=21)
        other = F.prompt(12000, seed=22)
        conv1, conv2 = {'id': 'one'}, {'id': 'two'}
        engine.turn(conv1, base[:4500], 0, [4096])
        engine.turn(conv2, other[:6200], 0, [6144])
        # One step: a chunked hit, a tail-only hit, a cold row with a plan, a cold row without.
        engine.registry.begin_step()
        rows = []
        specs = ((conv1, base[:9000], 4096, [8192], 0), (conv2, other[:7000], 6144, [], 1),
                 ({'id': 'three'}, F.prompt(5000, seed=23), 0, [4096], 2), ({'id': 'four'}, F.prompt(1500, seed=24), 0, [], 3))
        for conv, tokens, q, plan, slot in specs:
            shared = conv['row'][0, :q // F.BLOCK].tolist() if q else []
            row = engine.pool.row(len(tokens), shared)
            F.admit(engine.registry, conv['id'], tokens, q, plan, step=False)
            rows.append((conv['id'], tokens, row, q, slot))
        logits = engine.prefill(rows)
        slots = [[(r.clone(), [c.clone() for c in cs]) for r, cs in engine.toy.slot(s)] for s in range(4)]
        for index, (req, tokens, row, q, slot) in enumerate(rows):
            cold = engine.cold(tokens, slot=3, req='cold-' + req)
            self.assertTrue(torch.equal(logits[index:index + 1], cold[0]), req)
            for a, b in zip(engine.toy.kv(row, len(tokens)), engine.toy.kv(cold[1], len(tokens))):
                self.assertTrue(torch.equal(a, b), req)
            for a, b in zip(slots[slot], engine.toy.slot(3)):
                self.assertTrue(torch.equal(a[0], b[0]), req)
        self.assertIsNotNone(engine.registry.get(F.key_at(rows[2][1].tolist(), 4096)))

    def test_rope_is_staged_for_the_whole_prompt_on_a_hit(self):
        engine = self.engine()
        base = F.prompt(7000, seed=31)
        conv = {'id': 'rope'}
        engine.turn(conv, base[:4500], 0, [4096])
        engine.toy.ropes.clear()
        engine.turn(conv, base[:7000], 4096, [6144])
        self.assertEqual(engine.toy.ropes, [(1, 7000)])


class TracedExactness(EagerExactness):
    """trace_mode all (general's default): the captured chunk trace, replayed from Q/2048."""

    traced = True

    def test_a_gap_boundary_inside_the_traced_loop_costs_one_synchronize(self):
        engine = self.engine()
        tokens = F.prompt(9000, seed=41)
        engine.registry.begin_step()
        F.admit(engine.registry, 'gap', tokens, 0, [4096, 8192], h=4992)
        row = engine.pool.row(len(tokens))
        engine.fake.log.clear()
        engine.prefill([('gap', tokens, row, 0, 0)])
        log = engine.fake.log
        traces = [i for i, op in enumerate(log) if op[0] == 'execute_trace']
        self.assertEqual(len(traces), 4)
        # After the second chunk (ending at 4096): a synchronize, then the capture's reads, all
        # before the third chunk's trace.
        between = log[traces[1] + 1:traces[2]]
        self.assertEqual(between[0], ('synchronize',))
        self.assertTrue(any(op[0] == 'to_torch' for op in between))


class Sensitivity(ExactTestBase):
    """The harness sees each failure section 2.0.4 names; otherwise its equalities prove nothing."""

    def hit_after(self, engine, mutate=None):
        base = F.prompt(8000, seed=51)
        conv = {'id': 'sens'}
        engine.turn(conv, base[:4500], 0, [4096])
        if mutate:
            mutate(engine)
        hit = engine.turn(conv, base[:8000], 4096, [])
        hit_slot = [r.clone() for r, _ in engine.toy.slot(0)]
        cold = engine.cold(base[:8000])
        same_slot = all(torch.equal(a, b) for a, (b, _) in zip(hit_slot, engine.toy.slot(3)))
        return torch.equal(hit[0], cold[0]) and same_slot

    def test_an_exact_hit_is_exact(self):
        self.assertTrue(self.hit_after(self.engine()))

    def test_a_restore_without_conv_carry_is_caught(self):
        def drop_carry(engine):
            checkpoint = next(iter(engine.registry.entries.values()))
            checkpoint.carry = [torch.zeros_like(c) for c in checkpoint.carry]
        self.assertFalse(self.hit_after(self.engine(), drop_carry))

    def test_a_resume_without_restore_is_caught(self):
        def no_restore(engine):
            engine.model._qwen_prefix_restore = lambda *args, **kwargs: None
        self.assertFalse(self.hit_after(self.engine(), no_restore))

    def test_the_unguarded_eager_tail_deallocates_none_on_a_tail_only_hit(self):
        model, vllm = F.staged_sources()
        guard = ('            if last_hidden is not None:\n'
                 '                ttnn.deallocate(last_hidden)\n')
        self.assertEqual(model.count(patcher.EAGER_TAIL_NEW), 1)
        self.assertEqual(patcher.EAGER_TAIL_NEW.count(guard), 1)
        unguarded = model.replace(patcher.EAGER_TAIL_NEW, patcher.EAGER_TAIL_NEW.replace(
            guard, '            ttnn.deallocate(last_hidden)\n'))
        engine = self.engine(sources=(unguarded, vllm))
        base = F.prompt(5000, seed=52)
        conv = {'id': 'tail'}
        engine.turn(conv, base[:4200], 0, [4096])
        with self.assertRaisesRegex(TypeError, r'deallocate\(None\)'):
            engine.turn(conv, base[:5000], 4096, [])
        # The graft as staged serves the same turn.
        engine = self.engine()
        engine.turn(conv, base[:4200], 0, [4096])
        engine.turn(conv, base[:5000], 4096, [])


class GrantGuards(ExactTestBase):
    """F2: a resumed row the scheduler did not grant exactly as asked is an assertion, before any
    chunk of the step runs and with the batched GDN buffers rebound."""

    def setUp(self):
        self.engine_ = self.engine()
        self.tokens = F.prompt(6000, seed=61)
        self.conv = {'id': 'g'}
        self.engine_.turn(self.conv, self.tokens[:4500], 0, [4096])
        self.engine_.toy.segments.clear()

    def row(self, start, req='g'):
        return (req, self.tokens, self.engine_.pool.row(len(self.tokens), self.conv['row'][0, :64].tolist()), start, 0)

    def assert_refused(self, pattern, rows, **kwargs):
        with self.assertRaisesRegex(AssertionError, pattern):
            self.engine_.prefill(rows, **kwargs)
        self.assertEqual(self.engine_.toy.segments, [])
        for layer in self.engine_.model.layers:
            if not layer.is_full_attention:
                self.assertEqual(layer.attention.B, 4)

    def test_no_registry(self):
        self.engine_.registry = None
        self.assert_refused('no prefix registry is installed', [self.row(4096)])

    def test_no_request_ids_from_the_runner(self):
        F.admit(self.engine_.registry, 'g', self.tokens, 4096)
        self.assert_refused('the runner passed no request id', [self.row(4096)], req_ids=False)

    def test_no_committed_grant(self):
        self.engine_.registry.begin_step()
        self.assert_refused('no committed grant', [self.row(4096)])

    def test_a_stale_grant(self):
        F.admit(self.engine_.registry, 'g', self.tokens, 0, [4096])
        self.assert_refused('stale grant Q=0', [self.row(4096)])

    def test_tokens_that_differ_from_the_checkpoint(self):
        F.admit(self.engine_.registry, 'g', self.tokens, 4096)
        checkpoint = self.engine_.registry.grant_for('g').checkpoint
        checkpoint.token_ids[100] += 1
        self.assert_refused('token ids differ', [self.row(4096)])

    def test_a_grant_that_is_not_the_rows(self):
        grant = self.engine_.registry.entries
        self.engine_.registry = SimpleNamespace(grant_for=lambda req: SimpleNamespace(req_id='other', q=4096, plan=()),
                                                entries=grant)
        self.assert_refused('belongs to other', [self.row(4096)])

    def test_a_boundary_that_is_not_a_chunk(self):
        self.engine_.registry = SimpleNamespace(grant_for=lambda req: SimpleNamespace(req_id=req, q=1000, plan=()))
        self.assert_refused('not a 2048-token chunk boundary', [self.row(1000)])

    def test_a_bad_row_stops_the_step_before_a_good_row_runs(self):
        self.engine_.registry.begin_step()
        good = ('fresh', F.prompt(3000, seed=62), self.engine_.pool.row(3000), 0, 1)
        self.assert_refused('no committed grant', [good, self.row(4096)])

    def test_the_unbatched_path_refuses_a_resumed_row(self):
        F.admit(self.engine_.registry, 'g', self.tokens, 4096)
        self.engine_.model.args.max_batch_size = 1
        with self.assertRaisesRegex(AssertionError, 'reached the unbatched path'):
            self.engine_.prefill([self.row(4096)])

    def test_a_restore_before_the_warmup_chose_a_path(self):
        engine = Engine(traced=self.traced)
        conv = {'id': 'cold-engine'}
        engine.turn(conv, self.tokens[:4500], 0, [4096])
        with self.assertRaisesRegex(AssertionError, 'before _qwen_prefix_warm_restore chose a path'):
            engine.turn(conv, self.tokens, 4096, [])


class Warmup(ExactTestBase):
    """F3: the restore path is chosen and compiled before any trace is parked; a hit compiles nothing."""

    def hit(self, engine):
        base = F.prompt(7000, seed=71)
        conv = {'id': 'w'}
        engine.turn(conv, base[:4500], 0, [4096])
        engine.fake.log.clear()
        engine.turn(conv, base[:7000], 4096, [6144])
        return engine.rows_prefix()[-1]

    def test_h2d_is_chosen_when_it_round_trips_and_a_hit_compiles_nothing(self):
        engine = self.engine()
        warm = engine.log.lines(patcher.MARKER_WARM)
        self.assertEqual(len(warm), 1)
        self.assertIn("restore_mode=h2d results={'copy': 'exact', 'h2d': 'exact'}", warm[0])
        self.assertIn('gdn_layers=3', warm[0])
        marker = self.hit(engine)
        self.assertNotIn('compile', [op[0] for op in engine.fake.log])
        self.assertEqual([op for op in engine.fake.log if op[0] == 'copy' and op[1] == (2, 3, 4)], [])
        before, after = marker.split('programs_across_restore=(')[1].rstrip(')').split(', ')
        self.assertEqual(before, after)

    def test_copy_is_the_fallback_and_its_programs_were_compiled_at_warmup(self):
        engine = self.engine(h2d_refuse=True)
        warm = engine.log.lines(patcher.MARKER_WARM)[0]
        self.assertIn('restore_mode=copy', warm)
        self.assertIn("'h2d': 'refused (RuntimeError", warm)
        self.hit(engine)
        restore_copies = [op for op in engine.fake.log if op[0] == 'copy' and op[1] in ((2, 2, 2, 2), (2, 3, 4))]
        self.assertTrue(restore_copies)
        self.assertNotIn('compile', [op[0] for op in engine.fake.log])

    def test_the_operator_can_force_copy(self):
        engine = self.engine(environ=dict(ON, QWEN_PREFIX_RESTORE='copy'))
        self.assertIn('restore_mode=copy', engine.log.lines(patcher.MARKER_WARM)[0])

    def test_no_exact_path_refuses_to_start(self):
        engine = Engine(h2d_refuse=True)
        engine.fake.copy_corrupt = True
        with self.assertRaisesRegex(RuntimeError, 'no GDN restore path round-trips'):
            engine.warm()

    def test_it_runs_once_on_the_compile_only_call_before_the_trace_capture(self):
        engine = Engine(traced=True)
        engine.warm(enable_trace=False)
        self.assertEqual(engine.toy.segments, [])
        engine.warm(enable_trace=True)
        self.assertEqual(engine.toy.segments, [('capture-trace',)])
        self.assertEqual(len(engine.log.lines(patcher.MARKER_WARM)), 1)
        for layer in engine.model.layers:
            if not layer.is_full_attention:
                self.assertEqual(layer.attention.B, 4)
                self.assertTrue(torch.equal(layer.attention.rec_state.data, torch.zeros_like(layer.attention.rec_state.data)))

    def test_a_model_off_the_batched_path_skips_it_and_says_so(self):
        engine = Engine()
        engine.model.args.max_batch_size = 1
        engine.warm()
        self.assertEqual(engine.log.lines(patcher.MARKER_WARM), [])
        self.assertTrue(engine.log.lines('model warm skipped'))


class Capture(ExactTestBase):
    """S7: a capture never fails a request."""

    def test_a_memory_error_skips_the_checkpoint_and_the_request_is_exact(self):
        engine = self.engine()

        def refuse_carry(tensor):
            if tensor.shape == (2, 3, 4):
                raise MemoryError('host checkpoint')
        engine.fake.to_torch_hook = refuse_carry
        tokens = F.prompt(5000, seed=81)
        hit, cold = self.run_turn_and_cold(engine, {'id': 'm'}, tokens, 0, [4096])
        self.assertIn('captured=[4096:skipped]', engine.rows_prefix()[-2])
        self.assertEqual(engine.registry.stats['capture_failures'], 1)
        self.assertEqual(len(engine.registry.entries), 0)

    def test_a_registry_refusal_is_reported(self):
        engine = self.engine()
        engine.registry.budget_bytes = 10
        engine.turn({'id': 'b'}, F.prompt(5000, seed=82), 0, [4096])
        self.assertIn('captured=[4096:refused', engine.rows_prefix()[-1])
        self.assertEqual(engine.registry.stats['capture_skipped_budget'], 1)

    def test_a_registry_that_raises_is_contained(self):
        engine = self.engine()
        engine.registry = SimpleNamespace(
            grant_for=lambda req: SimpleNamespace(req_id=req, q=0, plan=[(4096, 'k')]),
            capture=mock.Mock(side_effect=RuntimeError('registry down')), stats={'capture_failures': 0, 'capture_ms': 0})
        engine.prefill([('r', F.prompt(5000, seed=83), engine.pool.row(5000), 0, 0)])
        self.assertEqual(engine.registry.stats['capture_failures'], 1)
        self.assertTrue(engine.log.lines('capture not stored'))

    def test_the_checkpoint_is_the_models_own_state_at_the_boundary(self):
        engine = self.engine()
        tokens = F.prompt(6000, seed=84)
        engine.turn({'id': 'own'}, tokens, 0, [4096])
        checkpoint = engine.registry.get(F.key_at(tokens.tolist(), 4096))
        self.assertEqual((checkpoint.pos, len(checkpoint.rec), len(checkpoint.carry)), (4096, 3, 3))
        self.assertEqual(tuple(checkpoint.rec[0].shape), (2, 2, 2, 2))
        self.assertEqual(checkpoint.rec[0].dtype, torch.float32)
        self.assertEqual(tuple(checkpoint.carry[0].shape), (2, 3, 4))
        self.assertEqual(checkpoint.nbytes, 3 * (2 * 8 * 4 + 2 * 12 * 2))
        # Captured after chunk 2, before chunk 3 and the tail: a fresh cold run stopped at 4096.
        reference = self.engine()
        reference.cold(tokens[:4096])
        rec = [layer.attention.slots[3][0] for layer in reference.model.layers if not layer.is_full_attention]
        for a, b in zip(checkpoint.rec, rec):
            self.assertTrue(torch.equal(a, b))

    def test_unreachable_plan_positions_are_dropped_and_reported(self):
        engine = self.engine()
        engine.turn({'id': 'd'}, F.prompt(5000, seed=85), 0, [4096, 6144])
        self.assertIn('dropped=[6144]', engine.rows_prefix()[-1])


class Audit(ExactTestBase):
    """QWEN_PREFIX_AUDIT=1: program-free digests that line up between a hit and a cold run."""

    def digests(self, engine):
        out = {}
        for line in engine.log.lines(patcher.MARKER_AUDIT):
            fields = dict(part.split('=', 1) for part in line.split() if '=' in part)
            key = ('window', fields['window']) if 'window' in fields else ('state',)
            out.setdefault(fields['req'], {})[key] = fields.get('kv') or (fields['gdn_slot'], fields['logits'])
        return out

    def test_hit_and_cold_digests_match_and_the_audit_only_reads(self):
        engine = self.engine(environ=dict(ON, QWEN_PREFIX_AUDIT='1'))
        base = F.prompt(7000, seed=91)
        conv = {'id': 'audit'}
        engine.turn(conv, base[:4500], 0, [4096])
        spans = []
        original = engine.model._qwen_prefix_audit

        def spy(*args, **kwargs):
            start = len(engine.fake.log)
            original(*args, **kwargs)
            spans.append(engine.fake.log[start:])
        engine.model._qwen_prefix_audit = spy
        engine.turn(conv, base[:7000], 4096, [6144])
        engine.cold(base[:7000], req='audit-cold')
        digests = self.digests(engine)
        hit, cold = digests['audit'], digests['audit-cold']
        self.assertEqual(sorted(hit), sorted(cold))
        self.assertEqual(len([k for k in hit if k[0] == 'window']), 4)
        self.assertEqual(hit, cold)
        for span in spans:
            self.assertEqual({op[0] for op in span}, {'to_torch'})
            self.assertEqual(len(span), 2 * len(engine.model._paged_kv_caches))
        self.assertTrue(any('new=1' in line and 'window=2' in line and 'Q=4096' in line
                            for line in engine.log.lines(patcher.MARKER_AUDIT)))


class OffIsStock(unittest.TestCase):
    """QWEN_PREFIX_REUSE unset: the staged files make the stock files' ttnn calls and results."""

    CASES = ([4097, 1000], [6144], [2048], [300])

    def pair(self, traced, environ=None, registry=False):
        stock = Engine(traced=traced, environ=environ or {}, sources=F.stock_sources(), registry=registry)
        staged = Engine(traced=traced, environ=environ or {}, registry=registry)
        return stock, staged

    def run_cases(self, engine):
        results = []
        for case in self.CASES:
            rows = []
            for slot, length in enumerate(case):
                tokens = F.prompt(length, seed=length + slot)
                rows.append(('r%d-%d' % (length, slot), tokens, engine.pool.row(length), 0, slot))
            logits = engine.prefill(rows)
            results.append((logits, [engine.toy.kv(row[2], len(row[1])) for row in rows],
                            [engine.toy.slot(slot) for slot in range(len(case))]))
        return results

    def assert_same(self, stock, staged, ignore_compiles=False):
        a, b = self.run_cases(stock), self.run_cases(staged)
        keep = (lambda log: [op for op in log if op[0] != 'compile']) if ignore_compiles else list
        self.assertEqual(keep(stock.fake.log), keep(staged.fake.log))
        self.assertEqual(stock.toy.segments, staged.toy.segments)
        for (la, kva, sa), (lb, kvb, sb) in zip(a, b):
            self.assertTrue(torch.equal(la, lb))
            for x, y in zip(kva, kvb):
                for p, q in zip(x, y):
                    self.assertTrue(torch.equal(p, q))
            for x, y in zip(sa, sb):
                for (rp, cp), (rq, cq) in zip(x, y):
                    self.assertTrue(torch.equal(rp, rq) and all(torch.equal(m, n) for m, n in zip(cp, cq)))

    def test_eager_prefill_is_stock(self):
        self.assert_same(*self.pair(traced=False))

    def test_traced_prefill_is_stock(self):
        self.assert_same(*self.pair(traced=True))

    def test_warmup_is_stock(self):
        for traced in (False, True):
            stock, staged = self.pair(traced)
            for engine in (stock, staged):
                engine.warm(enable_trace=False)
                engine.warm(enable_trace=True)
            self.assertEqual(stock.fake.log, staged.fake.log)
            self.assertEqual(stock.toy.segments, staged.toy.segments)
            self.assertEqual(staged.log.lines('prefix'), [])

    def test_the_capability_is_off_unless_the_flag_is_exactly_one(self):
        for environ, expected in (({}, False), ({'QWEN_PREFIX_REUSE': '0'}, False),
                                  ({'QWEN_PREFIX_REUSE': 'true'}, False), (ON, True)):
            engine = Engine(environ=environ)
            self.assertIs(engine.vllm.Qwen36ForCausalLM.model_capabilities['supports_prefix_caching'], expected)
        stock = Engine(environ=ON, sources=F.stock_sources())
        self.assertIs(stock.vllm.Qwen36ForCausalLM.model_capabilities['supports_prefix_caching'], False)

    def test_on_with_no_grants_computes_what_stock_computes(self):
        stock = Engine(traced=True, environ={}, sources=F.stock_sources(), registry=False)
        staged = Engine(traced=True, environ=ON, registry=False)
        staged.warm()
        staged.fake.log.clear()
        # The warmup compiled the GDN reset's copies already; everything else is the same call.
        self.assert_same(stock, staged, ignore_compiles=True)
        self.assertEqual(len(staged.rows_prefix()), 5)


if __name__ == '__main__':
    unittest.main()
