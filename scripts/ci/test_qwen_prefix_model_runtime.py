"""The G1 model graft, executed: exact reuse, the F1/F2/F3/S7 guards, and nothing changed when off.

The REAL model.py and qwen36_vllm.py (the IMG bytes qwen_prefix_model_patch pins), stock and
staged, run on a recording fake ttnn with a toy hybrid model whose chunk program has section
2.0.4's dependency structure (qwen_prefix_model_fixture). The registry is qwen_prefix_registry,
driven as the scheduler graft (qwen_prefix_scheduler_patch) drives it. What is held:

  exact      a hit - chained over 7 turns, tail-only, at previous-prompt lengths 2047/2048/2049,
             after an early divergence (an older checkpoint), from a gap capture another
             conversation took, and four rows in one step - equals a cold run of the same prompt on
             fresh blocks: logits, KV [0, L) through the page table, and the GDN slot, byte for
             byte, on the traced and the eager loop;
  sensitive  the harness sees what the design says breaks exactness: no conv_carry, no restore;
             and the unguarded eager tail (Lever N's loop alone) raises ttnn.deallocate(None);
  guards     start_pos > 0 without a registry, a request id, a committed grant, a current grant,
             the row's grant or matching tokens, or with a checkpoint whose layer count, shapes or
             dtypes are not the scratch's, is an assertion before any row runs (F2); a resumed row
             on the unbatched path asserts; a restore before warmup asserts (F3);
  warmup     the restore path is chosen on the plugin's compile-only call and compiles nothing on a
             hit (copy_host_to_device_tensor when it round-trips, else ttnn.copy warmed), once per
             model; each path is judged from a zeroed scratch with its own pattern, chip by chip,
             so an h2d that writes nothing, one chip, or chip 0 to both is not chosen;
  capture    a MemoryError while capturing skips the checkpoint, counts it, and the request is exact
             (S7); a registry refusal is reported; unreachable plan positions are dropped;
  fast path  the C2 fast path's prefill capture binds the staged model on every profile (its
             prefill_paged_slots* entries are the stock ones), and the prefix route refuses to run
             under any fast-path capture;
  markers    every row names the registry's presence, the grant and its plan, and the program cache
             before and after it; growth, a missing registry and an unknown count are warned;
  audit      QWEN_PREFIX_AUDIT=1 digests match between hit and cold, and the audit only reads;
  dram       G2's reading: "[PINDIAG] dram after registry" once, when the route first runs with the
             registry present, and "dram after first capture" once, after the first row that stored a
             checkpoint; each is serving_buffer_pool's reading and text, reads only, reports a refused
             view instead of raising, never appears without the registry or the switch, and
             prefix_markers reads it;
  contract   the registry's model contract: the warmup declares mid-loop captures (on the holder,
             before the scheduler exists), every capture is filed with the loop's own token count
             (loop_pos), restores and captures are counted through the registry; and the harness
             (prefix_markers, prefix_judge) reads the rows, digests and audit rows the model prints;
  off        with QWEN_PREFIX_REUSE unset the staged files make exactly the stock files' ttnn calls
             and results (prefill and warmup, traced and eager), add only private _qwen_prefix*
             names to the classes, and the capability is False.

Grants carry the scheduler graft's own capture plan (SchedulerGraft.plan) unless a test forces one.
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

import qwen_prefix_model_fixture as F  # noqa: E402
import qwen_prefix_model_patch as patcher  # noqa: E402
import qwen_prefix_registry as prefix_registry  # noqa: E402

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

    def __init__(self, traced=False, environ=ON, sources=None, registry=True, h2d_refuse=False, h2d_mode='exact'):
        self.fake = F.FakeTTNN()
        self.fake.h2d_refuse = h2d_refuse
        self.fake.h2d_mode = h2d_mode
        self.log = F.FakeLogger()
        model_source, vllm_source = sources or F.staged_sources()
        self.module = F.load_source(model_source, 'qwen36_model_under_test', F.model_stubs(self.fake, self.log))
        self.vllm = F.load_source(vllm_source, 'qwen36_vllm_under_test', F.vllm_stubs(self.fake, self.log), environ)
        self.toy = F.Toy(self.module, self.fake, traced=traced)
        self.model = self.toy.model
        self.wrapper = F.build_wrapper(self.vllm, self.toy)
        self.registry = prefix_registry.PrefixRegistry(budget_bytes=1 << 30) if registry else None
        self.pool = F.Pool()
        self.environ = dict(environ)

    def warm(self, enable_trace=False):
        # Inside the engine's registry scope: the warmup's mid-loop declaration reaches this registry
        # (in serving it lands on the holder first; shared_registry carries it over), and any holder
        # the warmup creates is gone afterwards.
        with registry_scope(self.registry), prefix_env(self.environ):
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

    def turn(self, conv, tokens, q, plan=None, slot=0, h=None, shared_row=None, force_plan=False):
        """One admitted turn: the grant the scheduler commits (its plan is SchedulerGraft.plan's
        unless force_plan), a page row sharing the first Q/64 blocks vLLM's cache returned, the
        prefill. Returns (logits, row, slot)."""
        source = shared_row if shared_row is not None else conv.get('row')
        shared = source[0, :q // F.BLOCK].tolist() if q else []
        row = self.pool.row(len(tokens), shared)
        F.admit(self.registry, conv['id'], tokens, q, plan, h=h, force_plan=force_plan)
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

    def run_turn_and_cold(self, engine, conv, tokens, q, plan=None, **kwargs):
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
        hit = engine.turn(conv, base[:8000], 4096)
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

    def test_a_restore_that_loses_chip_1s_carry_is_caught(self):
        """Review finding 10: the toy reads both chips' carries, so chip 1 alone matters."""
        def drop_chip1_carry(engine):
            checkpoint = next(iter(engine.registry.entries.values()))
            carry = [c.clone() for c in checkpoint.carry]
            for c in carry:
                c[1:] = 0
            checkpoint.carry = carry
        self.assertFalse(self.hit_after(self.engine(), drop_chip1_carry))

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

    def test_a_checkpoint_with_the_wrong_layer_count_stops_the_step_before_any_row_runs(self):
        """Review finding 5: the layer count is checked with the grant, not inside the row loop."""
        F.admit(self.engine_.registry, 'g', self.tokens, 4096)
        checkpoint = self.engine_.registry.grant_for('g').checkpoint
        checkpoint.rec = checkpoint.rec[:2]
        good = ('fresh', F.prompt(3000, seed=63), self.engine_.pool.row(3000), 0, 1)
        self.assert_refused('holds 2/3 GDN states, the model 3', [good, self.row(4096)])

    def test_a_checkpoint_whose_shape_is_not_the_scratchs_is_refused(self):
        F.admit(self.engine_.registry, 'g', self.tokens, 4096)
        checkpoint = self.engine_.registry.grant_for('g').checkpoint
        checkpoint.carry = [c[:, :2] for c in checkpoint.carry]
        self.assert_refused(r"GDN layer 0 checkpoint .* is not the scratch's", [self.row(4096)])

    def test_a_checkpoint_whose_dtype_is_not_the_scratchs_is_refused(self):
        F.admit(self.engine_.registry, 'g', self.tokens, 4096)
        checkpoint = self.engine_.registry.grant_for('g').checkpoint
        checkpoint.rec = [r.to(torch.bfloat16) for r in checkpoint.rec]
        self.assert_refused(r"GDN layer 0 checkpoint .*bfloat16.* is not the scratch's", [self.row(4096)])

    def test_the_unbatched_path_refuses_a_resumed_row(self):
        F.admit(self.engine_.registry, 'g', self.tokens, 4096)
        self.engine_.model.args.max_batch_size = 1
        with self.assertRaisesRegex(AssertionError, 'reached the unbatched path'):
            self.engine_.prefill([self.row(4096)])

    def test_a_restore_before_the_warmup_chose_a_path(self):
        engine = Engine(traced=self.traced)
        conv = {'id': 'cold-engine'}
        engine.turn(conv, self.tokens[:4500], 0, [4096])
        engine.toy.segments.clear()
        with self.assertRaisesRegex(AssertionError, 'before _qwen_prefix_warm_restore chose a path'):
            engine.turn(conv, self.tokens, 4096, [])
        # Refused with the grant, before the scratch was bound or any chunk ran.
        self.assertEqual(engine.toy.segments, [])


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
        with self.assertRaisesRegex(RuntimeError, r"no allowed GDN restore path \['h2d', 'copy'\] round-trips"):
            engine.warm()

    def test_an_h2d_that_accepts_the_spec_but_writes_wrongly_is_not_chosen(self):
        """Review finding 2: each path is judged from a zeroed scratch with its own pattern and
        chip by chip, so copy's pattern cannot vouch for an h2d that writes nothing, writes chip 0
        only, or writes chip 0's shard to both chips. The hit then restores through copy, exactly."""
        for mode, chips in (('noop', '[0, 1]'), ('chip0', '[1]'), ('chip0_to_all', '[1]')):
            with self.subTest(h2d_mode=mode):
                engine = self.engine(h2d_mode=mode)
                warm = engine.log.lines(patcher.MARKER_WARM)[0]
                self.assertIn("restore_mode=copy results={'copy': 'exact', 'h2d': 'differs on chip(s) %s'}" % chips,
                              warm)
                base = F.prompt(8000, seed=72)
                conv = {'id': 'wrong-h2d-' + mode}
                self.run_turn_and_cold(engine, conv, base[:4500], 0)
                self.run_turn_and_cold(engine, conv, base[:8000], 4096)
                self.assertEqual([op for op in engine.fake.log if op[0] == 'h2d' and op[1] == (2, 3, 4)],
                                 [('h2d', (2, 3, 4), 'bfloat16')] * 3, 'h2d ran only at warmup')

    def test_a_forced_h2d_that_writes_wrongly_refuses_to_start(self):
        engine = Engine(environ=dict(ON, QWEN_PREFIX_RESTORE='h2d'), h2d_mode='chip0')
        with self.assertRaisesRegex(RuntimeError, r"no allowed GDN restore path \['h2d'\]"):
            engine.warm()

    def test_each_path_is_written_into_a_zeroed_scratch(self):
        engine = Engine()
        seen = []
        restore = engine.model._qwen_prefix_restore

        def spy(rec, carry, mode=None):
            states = [(layer.attention.rec_state.data.clone(), layer.attention.conv_carry.data.clone())
                      for layer in engine.model.layers if not layer.is_full_attention]
            seen.append((mode, all(not r.any() and not c.any() for r, c in states),
                         sorted({float(t.min()) for t in rec + carry})))
            return restore(rec, carry, mode=mode)
        engine.model._qwen_prefix_restore = spy
        engine.warm()
        self.assertEqual([(mode, zeroed) for mode, zeroed, _ in seen], [('copy', True), ('h2d', True)])
        # Each path writes its own pattern, and no pattern holds a zero a no-op would leave.
        self.assertNotEqual(seen[0][2], seen[1][2])
        self.assertGreater(min(seen[0][2] + seen[1][2]), 0.0)

    def test_the_warmup_is_keyed_on_the_model_not_the_wrapper(self):
        """Review finding 6: a model that has not chosen a restore path is warmed whatever the
        wrapper did before; a model that has is not warmed again through another wrapper."""
        engine = self.engine()
        again = F.build_wrapper(engine.vllm, engine.toy)
        with prefix_env(engine.environ):
            again.warmup_model_prefill(engine.model._paged_kv_caches, False)
        self.assertEqual(len(engine.log.lines(patcher.MARKER_WARM)), 1)
        fresh = F.Toy(engine.module, engine.fake).model
        engine.wrapper.model = [fresh]
        engine.warm()
        self.assertEqual(len(engine.log.lines(patcher.MARKER_WARM)), 2)
        self.assertEqual(fresh._qwen_prefix_restore_mode, 'h2d')

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
        # A plan the scheduler graft would not make (6144 lies past a 5000-token prompt).
        engine.turn({'id': 'd'}, F.prompt(5000, seed=85), 0, [4096, 6144], force_plan=True)
        self.assertIn('plan=[4096] ', engine.rows_prefix()[-1])
        self.assertIn('dropped=[6144]', engine.rows_prefix()[-1])


class FastPathCoexistence(ExactTestBase):
    """Review finding 1: the C2 fast path's prefill capture enumerates prefill_paged_slots* on the
    model and refuses any entry it does not know, on every profile of the image. The staged model
    must stay bindable, and the prefix route must refuse to run under a capture."""

    def test_the_fast_path_capture_binds_the_staged_model_on_every_profile(self):
        import dflash_prefill_window as window
        stock = Engine(environ={}, sources=F.stock_sources())
        entries = lambda model: sorted(n for n in dir(model) if n.startswith('prefill_paged_slots'))  # noqa: E731
        for environ in ({}, ON):
            with self.subTest(environ=environ):
                engine = Engine(environ=environ)
                self.assertEqual(entries(engine.model), entries(stock.model))
                bindings = window.PrefillWindowCapture(None, engine.model, 4000, ()).bindings()
                self.assertEqual(sorted(name for _, name, _ in bindings if name.startswith('prefill_paged_slots')),
                                 ['prefill_paged_slots'])
                # The capture's own marker is one the prefix route refuses under.
                markers = [name for _, name, value in bindings if not callable(value)]
                self.assertEqual(markers, ['_qwen_dflash_prefill_capture'])
                self.assertLessEqual(set(markers), set(engine.module._QWEN_PREFIX_FAST_PATH_MARKERS))

    def test_the_prefix_route_refuses_to_run_under_a_fast_path_capture(self):
        engine = self.engine()
        for marker in engine.module._QWEN_PREFIX_FAST_PATH_MARKERS:
            with self.subTest(marker=marker):
                setattr(engine.model, marker, object())
                engine.toy.segments.clear()
                engine.registry.begin_step()
                with self.assertRaisesRegex(AssertionError, 'does not run under the C2 fast path'):
                    engine.prefill([('f', F.prompt(3000, seed=5), engine.pool.row(3000), 0, 0)])
                self.assertEqual(engine.toy.segments, [])
                delattr(engine.model, marker)
        self.assertEqual(set(engine.module._QWEN_PREFIX_FAST_PATH_MARKERS),
                         {'_qwen_dflash_prefill_capture', '_qwen_dspark_prefill_capture', '_qwen_target_feature_capture'})


class RowMarker(ExactTestBase):
    """Review findings 3 and 4: every row says whether the registry is there, what was granted,
    and how the program cache moved; a missing registry and an unknown program count are warned."""

    def test_the_marker_names_the_registry_the_grant_and_the_plan(self):
        engine = self.engine()
        base = F.prompt(7000, seed=76)
        conv = {'id': 'm'}
        engine.turn(conv, base[:4500], 0)
        self.assertIn('registry=present grant=committed Q=0 L=4500 plan=[4096] ', engine.rows_prefix()[-1])
        engine.turn(conv, base[:7000], 4096)
        self.assertIn('registry=present grant=committed Q=4096 L=7000 plan=[6144] ', engine.rows_prefix()[-1])
        engine.cold(base[:7000])
        self.assertIn('registry=present grant=none Q=0 L=7000 plan=[] ', engine.rows_prefix()[-1])

    def test_a_missing_registry_is_named_on_every_row_and_warned_once(self):
        engine = self.engine(registry=False)
        for seed in (1, 2):
            engine.prefill([('n%d' % seed, F.prompt(3000, seed=seed), engine.pool.row(3000), 0, 0)])
        rows = engine.rows_prefix()
        self.assertEqual(len(rows), 2)
        self.assertTrue(all('registry=absent grant=none' in row for row in rows), rows)
        warned = [(level, message) for level, message in engine.log.records
                  if 'no prefix registry is installed in this process' in message]
        self.assertEqual([level for level, _ in warned], ['warning'])

    def test_every_row_logs_the_program_cache_before_and_after_it(self):
        engine = self.engine()
        engine.turn({'id': 'p'}, F.prompt(5000, seed=73), 0)
        self.assertRegex(engine.rows_prefix()[-1], r' programs=(\d+)->\1 programs_across_restore=None$')
        self.assertEqual(engine.log.lines('program growth'), [])

    def test_a_compile_during_any_row_is_warned(self):
        engine = self.engine()
        engine.fake.programs.clear()   # as if the warmup had compiled nothing: the row's reset compiles
        engine.turn({'id': 'p'}, F.prompt(5000, seed=74), 0)
        self.assertRegex(engine.rows_prefix()[-1], r' programs=0->[1-9]\d* ')
        warned = [(level, message) for level, message in engine.log.records if 'program growth' in message]
        self.assertEqual(len(warned), 1)
        self.assertEqual(warned[0][0], 'warning')
        self.assertIn('after warmup (F3)', warned[0][1])

    def test_an_unknown_program_count_is_warned_at_warmup(self):
        engine = Engine()
        engine.fake.program_count_known = False
        engine.warm()
        warned = [(level, message) for level, message in engine.log.records
                  if 'the per-row F3 no-compile check is blind' in message]
        self.assertEqual([level for level, _ in warned], ['warning'])
        engine.turn({'id': 'u'}, F.prompt(5000, seed=75), 0)
        self.assertIn(' programs=None->None ', engine.rows_prefix()[-1])


class DramReading(ExactTestBase):
    """G2's DRAM reading (the bring-up gate's NOT_EXERCISED without it): each chip's DRAM allocator
    figures, logged by the prefix route once the registry exists and once after the first stored
    checkpoint, in the fast path's own text."""

    def dram(self, engine):
        return engine.log.lines(patcher.MARKER_DRAM)

    def probe(self, engine):
        return engine.model._qwen_prefix_gdn_layers()[0].rec_state

    def test_the_registry_reading_comes_once_when_the_route_first_runs_with_the_registry(self):
        from serving_buffer_pool import dram_statistics, format_dram

        engine = self.engine()
        engine.cold(F.prompt(3000, seed=91), req='first')
        engine.cold(F.prompt(3100, seed=92), req='second')
        lines = [line for line in self.dram(engine) if patcher.DRAM_REGISTRY + ':' in line]
        want = patcher.MARKER_DRAM + patcher.DRAM_REGISTRY + ': ' + format_dram(
            dram_statistics(engine.fake, self.probe(engine)))
        self.assertEqual(lines, [want])
        self.assertIn('chip0 allocated=25.90GB free=7.21GB largest_free=7040.0MB of 33.10GB; chip1 ', want)
        self.assertEqual(sorted(set(device for device, _ in engine.fake.memory_views)), ['chip0', 'chip1'])
        self.assertEqual(set(kind for _, kind in engine.fake.memory_views), {'DRAM'})

    def test_the_first_capture_reading_follows_the_first_stored_checkpoint_once(self):
        engine = self.engine()
        engine.cold(F.prompt(5000, seed=93), req='plain')
        self.assertEqual([line for line in self.dram(engine) if patcher.DRAM_FIRST_CAPTURE in line], [],
                         'a row that stored nothing takes no capture reading')
        refusing = self.engine()
        refusing.registry.budget_bytes = 10
        refusing.turn({'id': 'refused'}, F.prompt(5000, seed=94), 0, [4096])
        self.assertIn('captured=[4096:refused', refusing.rows_prefix()[-1])
        self.assertEqual([line for line in self.dram(refusing) if patcher.DRAM_FIRST_CAPTURE in line], [],
                         'a refused capture stored nothing')
        base = F.prompt(9000, seed=95)
        conv = {'id': 'captures'}
        engine.turn(conv, base[:4500], 0)
        engine.turn(conv, base[:9000], 4096)
        self.assertEqual(len(engine.registry.entries), 2)
        captured = [line for line in self.dram(engine) if patcher.DRAM_FIRST_CAPTURE in line]
        self.assertEqual(len(captured), 1, 'once per process, not once per capture')
        self.assertTrue(captured[0].startswith(patcher.MARKER_DRAM + patcher.DRAM_FIRST_CAPTURE + ': chip0 allocated='),
                        captured[0])
        order = [message for _, message in engine.log.records
                 if patcher.MARKER_DRAM in message or patcher.MARKER_ROW in message]
        first_stored = [i for i, message in enumerate(order) if ':stored:' in message][0]
        self.assertIn(patcher.DRAM_FIRST_CAPTURE, order[first_stored + 1], 'right after the row that stored')

    def test_no_reading_without_the_registry_or_the_switch(self):
        engine = self.engine(registry=False)
        engine.prefill([('n', F.prompt(5000, seed=96), engine.pool.row(5000), 0, 0)])
        self.assertEqual((self.dram(engine), engine.fake.memory_views), ([], []))
        stock = self.engine(environ={})
        stock.prefill([('s', F.prompt(5000, seed=97), stock.pool.row(5000), 0, 0)], req_ids=False)
        self.assertEqual((self.dram(stock), stock.fake.memory_views), ([], []))
        direct = self.engine()
        before = (list(self.dram(direct)), list(direct.fake.memory_views))
        with prefix_env({}):
            direct.module._qwen_prefix_dram(patcher.DRAM_REGISTRY, self.probe(direct))
        self.assertEqual((self.dram(direct), direct.fake.memory_views), before, 'QWEN_PREFIX_REUSE unset')

    def test_a_refused_view_is_reported_and_the_request_is_exact(self):
        engine = self.engine()
        engine.fake.memory_view_refuse = True
        tokens = F.prompt(5000, seed=98)
        self.run_turn_and_cold(engine, {'id': 'r'}, tokens, 0, [4096])
        self.assertEqual(self.dram(engine), [
            patcher.MARKER_DRAM + point + ': unavailable (RuntimeError: no allocator on this device)'
            for point in (patcher.DRAM_REGISTRY, patcher.DRAM_FIRST_CAPTURE)])

    def test_the_reading_is_serving_buffer_pools_and_only_reads(self):
        from serving_buffer_pool import dram_statistics, format_dram

        engine = self.engine()
        module, tensor = engine.module, self.probe(engine)
        engine.fake.dram_views['chip1'] = dict(num_banks=8, total_bytes_per_bank=4138123648,
                                               total_bytes_allocated_per_bank=4111316352,
                                               total_bytes_free_per_bank=26807296,
                                               largest_contiguous_bytes_free_per_bank=712896)
        dead = F.FakeTensor(torch.zeros(1), engine.fake.float32, 'TILE', True)
        dead.deallocated = True
        for case, target in (('two chips', tensor), ('no tensor', None), ('a deallocated tensor', dead)):
            with self.subTest(case=case):
                self.assertEqual(module._qwen_prefix_dram_statistics(target), dram_statistics(engine.fake, target))
                self.assertEqual(module._qwen_prefix_format_dram(module._qwen_prefix_dram_statistics(target)),
                                 format_dram(dram_statistics(engine.fake, target)))
        engine.fake.memory_view_refuse = True
        self.assertEqual(module._qwen_prefix_dram_statistics(tensor), dram_statistics(engine.fake, tensor))
        engine.fake.memory_view_refuse = False
        log, programs = list(engine.fake.log), set(engine.fake.programs)
        with prefix_env(ON):
            module._qwen_prefix_dram('probe', tensor)
        self.assertEqual((engine.fake.log, engine.fake.programs), (log, programs), 'allocator views only')
        self.assertEqual(self.dram(engine)[-1].split(': ', 1)[1], format_dram(dram_statistics(engine.fake, tensor)))
        self.assertIn('chip1 allocated=32.89GB free=0.21GB largest_free=5.7MB of 33.10GB', self.dram(engine)[-1])

    def test_the_gate_harness_reads_both_readings(self):
        import prefix_markers

        engine = self.engine()
        base = F.prompt(7000, seed=99)
        engine.turn({'id': 'h'}, base[:4500], 0)
        readings = prefix_markers.scan(self.dram(engine))['dram_readings']
        self.assertEqual([reading['point'] for reading in readings],
                         [prefix_markers.DRAM_REGISTRY, prefix_markers.DRAM_FIRST_CAPTURE])
        for reading in readings:
            self.assertIsNone(reading['unavailable'])
            self.assertEqual([chip['chip'] for chip in reading['chips']], [0, 1])
            self.assertEqual((reading['chips'][0]['free_gb'], reading['chips'][0]['largest_free_mb']), (7.21, 7040.0))


class Audit(ExactTestBase):
    """QWEN_PREFIX_AUDIT=1: program-free digests that line up between a hit and a cold run."""

    def digests(self, engine):
        out = {}
        for line in engine.log.lines(patcher.MARKER_AUDIT):
            fields = dict(part.split('=', 1) for part in line.split() if '=' in part)
            key = ('window', fields['window']) if 'window' in fields else ('state',)
            out.setdefault(fields['req'], {})[key] = fields.get('kv') or (
                fields['kv_range'], fields['kv_sha'], fields['slot_sha'], fields['logits_sha'])
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


class RegistryContract(ExactTestBase):
    """G1 integration: the model graft against qwen_prefix_registry's model contract, and its markers
    against the harness that judges them on hardware (prefix_markers, prefix_judge)."""

    REQ = 'chatcmpl-pfx-m-0001-%s-1a2b3c4d'

    def test_the_warmup_declares_mid_loop_captures_before_the_scheduler_exists(self):
        engine = Engine(traced=self.traced)
        with mock.patch.dict(sys.modules), prefix_env(engine.environ):
            sys.modules.pop(patcher.REGISTRY_KEY, None)
            engine.wrapper.warmup_model_prefill(engine.model._paged_kv_caches, False)
            holder = sys.modules[patcher.REGISTRY_KEY]
            self.assertIs(holder.mid_loop_capture, True)
            self.assertIsNone(getattr(holder, 'registry', None), 'no scheduler, no registry yet')
            registry = prefix_registry.shared_registry(environ={})
            self.assertTrue(registry.mid_loop_capture, 'the scheduler\'s registry inherits the declaration')
            self.assertIs(prefix_registry.shared_registry(), registry)
            self.assertIs(prefix_registry.current_registry(), registry)

    def test_a_registry_that_missed_the_warmup_learns_it_at_the_first_prefill(self):
        engine = Engine(traced=self.traced)
        with mock.patch.dict(sys.modules), prefix_env(engine.environ):
            sys.modules.pop(patcher.REGISTRY_KEY, None)
            engine.wrapper.warmup_model_prefill(engine.model._paged_kv_caches, False)
        self.assertFalse(engine.registry.mid_loop_capture)
        engine.turn({'id': 'late'}, F.prompt(3000, seed=31), 0, [2048])
        self.assertTrue(engine.registry.mid_loop_capture)

    def test_the_stock_warmup_declares_nothing(self):
        engine = Engine(traced=self.traced, environ={})
        with mock.patch.dict(sys.modules), prefix_env({}):
            sys.modules.pop(patcher.REGISTRY_KEY, None)
            engine.wrapper.warmup_model_prefill(engine.model._paged_kv_caches, False)
            self.assertNotIn(patcher.REGISTRY_KEY, sys.modules)

    def test_captures_mid_loop_and_at_the_drain_are_filed_at_their_own_positions(self):
        """The gap boundary (below the loop's drain) and the drain boundary are both stored; the
        registry refuses none as taken at the wrong point, and each is the cold state there."""
        engine = self.engine()
        system = F.prompt(5000, seed=32)
        a = torch.cat([system, F.prompt(3000, seed=33)])
        b = torch.cat([system, F.prompt(3500, seed=34)])
        self.run_turn_and_cold(engine, {'id': 'A'}, a, 0, [6144])
        self.run_turn_and_cold(engine, {'id': 'B'}, b, 0, [4096, 8192], h=4992)
        stats = engine.registry.stats
        self.assertEqual((stats['captures'], stats['capture_wrong_position'], stats['capture_failures']), (3, 0, 0))
        self.assertGreater(stats['capture_ms'], 0)
        for tokens, pos in ((a, 6144), (b, 4096), (b, 8192)):
            self.assertEqual(engine.registry.get(F.key_at(tokens.tolist(), pos)).pos, pos)

    def test_a_restore_is_counted_through_the_registry(self):
        engine = self.engine()
        base = F.prompt(7000, seed=35)
        conv = {'id': 'r'}
        engine.turn(conv, base[:4500], 0)
        engine.turn(conv, base[:7000], 4096)
        self.assertEqual(engine.registry.stats['restores'], 1)
        self.assertGreaterEqual(engine.registry.stats['restore_ms'], 0.0)

    def rows(self, engine):
        import prefix_markers as pm

        return pm.scan(engine.log.lines('[PREFIX'))

    def test_the_harness_reads_the_models_rows_and_digests(self):
        import prefix_judge as judge

        engine = self.engine(environ=dict(ON, QWEN_PREFIX_DIGESTS='1'))
        base = F.prompt(7000, seed=36)
        conv = {'id': self.REQ % 'turn'}
        engine.turn(conv, base[:4500], 0)
        engine.turn({'id': self.REQ % 'hit', 'row': conv['row']}, base[:7000], 4096)
        engine.cold(base[:7000], req=self.REQ % 'cold')
        scanned = self.rows(engine)
        first, hit, cold = scanned['rows']
        self.assertEqual((first['tag'], first['q'], first['l'], first['captured'], first['capture_failed']),
                         ('pfx-m-0001-turn', 0, 4500, [4096], []))
        self.assertEqual((hit['tag'], hit['q'], hit['l'], hit['captured'], cold['q'], cold['captured']),
                         ('pfx-m-0001-hit', 4096, 7000, [6144], 0, []))
        self.assertEqual(hit['path'], 'traced' if self.traced else 'eager')
        self.assertIsInstance(hit['restored_ms'], float)
        self.assertIsNone(first['restored_ms'])
        self.assertIsInstance(first['capture_ms'], float)
        for row in (first, hit, cold):
            self.assertIsInstance(row['programs_before'], int)
            self.assertIsInstance(row['programs'], int)
            self.assertRegex(row['slot_sha'], '^[0-9a-f]{32}$')
            self.assertRegex(row['logits_sha'], '^[0-9a-f]{32}$')
        self.assertEqual(judge.row_growth(hit, None), (0, 'the row itself'))
        self.assertEqual((hit['slot_sha'], hit['logits_sha']), (cold['slot_sha'], cold['logits_sha']))
        self.assertNotEqual(first['slot_sha'], hit['slot_sha'])
        self.assertEqual(judge.digest_problems(dict(tag='cold', markers=dict(row=cold)),
                                               dict(tag='hit', markers=dict(row=hit))), [])

    def test_without_the_digest_switch_the_rows_carry_none(self):
        engine = self.engine()
        engine.turn({'id': self.REQ % 'plain'}, F.prompt(3000, seed=37), 0)
        row, = self.rows(engine)['rows']
        self.assertIsNone(row['slot_sha'])
        self.assertIsNone(row['logits_sha'])

    def test_the_harness_reads_the_audit_summary_and_the_judge_compares_it(self):
        import prefix_judge as judge

        engine = self.engine(environ=dict(ON, QWEN_PREFIX_AUDIT='1'))
        base = F.prompt(7000, seed=38)
        conv = {'id': self.REQ % 'turn'}
        engine.turn(conv, base[:4500], 0)
        engine.turn({'id': self.REQ % 'hit', 'row': conv['row']}, base[:7000], 4096)
        engine.cold(base[:7000], req=self.REQ % 'cold')
        scanned = self.rows(engine)
        self.assertEqual(len(scanned['audits']), 3, 'the judge takes the first audit row per request')
        audits = dict((entry['tag'], entry) for entry in scanned['audits'])
        self.assertEqual(sorted(audits), ['pfx-m-0001-cold', 'pfx-m-0001-hit', 'pfx-m-0001-turn'],
                         'one summary row per request; the per-window lines are not audit rows')
        hit, cold = audits['pfx-m-0001-hit'], audits['pfx-m-0001-cold']
        self.assertEqual((hit['kv_range'], cold['kv_range']), ('0:7000', '0:7000'))
        self.assertEqual(judge.audit_problems(dict(tag='cold', markers=dict(audit=cold)),
                                              dict(tag='hit', markers=dict(audit=hit))), [])
        wrong = dict(hit, kv_sha='0' * 32)
        self.assertEqual([severity for severity, _ in judge.audit_problems(
            dict(tag='cold', markers=dict(audit=cold)), dict(tag='hit', markers=dict(audit=wrong)))], ['FAIL'])
        rows = dict((row['tag'], row) for row in scanned['rows'])
        self.assertEqual(rows['pfx-m-0001-hit']['slot_sha'], hit['slot_sha'], 'audit implies the row digests')


class TracedRegistryContract(RegistryContract):
    traced = True


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

    def test_the_staged_classes_add_only_private_prefix_names(self):
        """The call logs above cannot see a class attribute, and the fast path's capture reads the
        model's attributes (review finding 1): whatever the flag, the staged classes add only
        _qwen_prefix* names, none of them public."""
        for environ in ({}, ON):
            stock, staged = Engine(environ=environ, sources=F.stock_sources()), Engine(environ=environ)
            for cls in ('Qwen36Model', 'Qwen36ForCausalLM'):
                module = 'module' if cls == 'Qwen36Model' else 'vllm'
                before = set(dir(getattr(getattr(stock, module), cls)))
                after = set(dir(getattr(getattr(staged, module), cls)))
                self.assertLessEqual(before, after, cls)
                self.assertTrue(all(name.startswith('_qwen_prefix') for name in after - before),
                                sorted(after - before))

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
