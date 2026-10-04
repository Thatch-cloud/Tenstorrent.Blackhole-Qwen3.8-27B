"""Lever N at TP4: the model route (levern_route), executed on the REAL qwen36 model.py after the G1 stage, on the recording fake ttnn and the toy
hybrid model of qwen_prefix_model_fixture (whose chunk program has the dependency structure the exactness argument rests on: a chunk's KV and GDN state
depend on its tokens, positions, every earlier position's KV read through the page table, and the recurrent state AND conv carry it starts from).

What is held:
  exact      a prompt that arrives in the steps levern_policy.plan names, through the vLLM wrapper's prefill_forward exactly as the plugin runner
             calls it (tokens[:end], prompt_lens=[end], start_pos=[start], one slot), equals the whole prompt on the stock path: logits, KV [0, P) through
             the page table, and the GDN slot, byte for byte, at every boundary length, with a decode-time interference that does not touch the scratch
             between the steps (and the harness SEES a continuation from the wrong state);
  skips      a step that is not the last writes no decode slot and snapshots no scratch; the last writes the slot once;
  owner      a continuation with no owner, the wrong start, another request's id, a replayed step, a skipped chunk, or after a whole prompt ran in between
             is an AssertionError before any device work; the owner is cleared on release and on an error; the foreign fault refuses once;
  announce   a chunked step with no announcement, or one that disagrees with the runner's start_pos/prompt_lens, is refused;
  off        with neither switch set nothing is installed and the wrapper is the vLLM wrapper's own;
  audit      the digests of a split prefill equal the whole prompt's, the control arm (audit alone) logs the same, and a corrupted step changes them;
  warm       the route's warm runs the plan's three steps, leaves no owner, and reports the program cache;
  install    a model without the G1 stage's resumable loops, a second install, and a wrapper without prefill_forward are refused by name."""

import contextlib
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import levern_policy as policy  # noqa: E402
import levern_route as route  # noqa: E402
import qwen_prefix_model_fixture as F  # noqa: E402

CHUNK = F.CHUNK
ON = {'QWEN_FAST_LEVER_N': '1'}
AUDIT = {'QWEN_FAST_LEVER_N': '1', 'QWEN_FAST_LEVERN_AUDIT': '1'}
CONTROL = {'QWEN_FAST_LEVERN_AUDIT': '1'}


class Rig(object):
    """One engine: the staged model (the G1 stage's output, which every C2 build applies), its vLLM wrapper and a block pool, with the route
    installed under `environ`."""

    def __init__(self, environ=ON, sources=None, install=True, traced=False):
        self.fake = F.FakeTTNN()
        self.logger = F.FakeLogger()
        model_source, vllm_source = sources or F.staged_sources()
        self.module = F.load_source(model_source, 'qwen36_model_under_test', F.model_stubs(self.fake, self.logger))
        self.vllm = F.load_source(vllm_source, 'qwen36_vllm_under_test', F.vllm_stubs(self.fake, self.logger))
        self.toy = F.Toy(self.module, self.fake, traced=traced)
        self.model = self.toy.model
        self.wrapper = F.build_wrapper(self.vllm, self.toy)
        self.pool = F.Pool()
        self.environ = dict(environ)
        self.lines = []
        self.runner = mock.Mock()
        self.runner.model = self.wrapper
        self.handle = None
        self.stock_forward = self.wrapper.prefill_forward
        if install:
            with self.scope():
                self.handle = route.install(self.runner, self.model, environ=self.environ, log=self.log)

    def log(self, message, *values):
        self.lines.append(message.format(*values) if values else message)

    def marker(self, text):
        return [line for line in self.lines if text in line]

    @contextlib.contextmanager
    def scope(self):
        with mock.patch.dict(sys.modules, {'ttnn': self.fake.module()}):
            yield

    def call(self, req_id, tokens, row, start, end, total, slot, announce=True, start_pos=None, lengths=None):
        """One prefill step as the plugin runner submits it: the wrapper's prefill_forward with the runner's kwargs, after the lifecycle's
        announcement."""
        if announce:
            route.announce(self.model, route.Step(req_id, start, end, total))
        batch = tokens[:end].reshape(1, -1)
        try:
            with self.scope():
                logits, _ = self.wrapper.prefill_forward(
                    batch, row, None, [end] if lengths is None else lengths, empty_slots=[slot],
                    start_pos=[start] if start_pos is None else start_pos)
        finally:
            route.withdraw(self.model)
        return logits

    def split(self, req_id, tokens, row, slot, steps=None, between=None, decoding=True):
        total = len(tokens)
        steps = policy.plan(total, decoding=decoding) if steps is None else steps
        logits = None
        for index, (start, end) in enumerate(steps):
            logits = self.call(req_id, tokens, row, start, end, total, slot)
            if between is not None and index < len(steps) - 1:
                between(index)
        return logits

    def whole(self, req_id, tokens, row, slot):
        return self.call(req_id, tokens, row, 0, len(tokens), len(tokens), slot)

    def state(self, row, length, slot):
        return (self.toy.kv(row, length), self.toy.slot(slot))


def assert_same_state(case, a, b, label):
    kv_a, slot_a = a
    kv_b, slot_b = b
    for x, y in zip(kv_a, kv_b):
        case.assertTrue(torch.equal(x, y), 'KV differs: %s' % label)
    case.assertEqual(len(slot_a), len(slot_b))
    for (rec_a, convs_a), (rec_b, convs_b) in zip(slot_a, slot_b):
        case.assertTrue(torch.equal(rec_a, rec_b), 'GDN recurrent state differs: %s' % label)
        case.assertTrue(all(torch.equal(x, y) for x, y in zip(convs_a, convs_b)), 'GDN conv state differs: %s' % label)


class ExactnessTests(unittest.TestCase):
    PROMPTS = (2047, 2048, 2049, 4095, 4096, 4097, 6143, 6144, 6145, 8192, 12289, 20001)

    def test_a_split_prompt_equals_the_whole_prompt_at_every_boundary_length(self):
        rig = Rig()
        for total in self.PROMPTS:
            with self.subTest(total=total):
                tokens = F.prompt(total, seed=total)
                row_whole, row_split = rig.pool.row(total), rig.pool.row(total)
                whole = rig.whole('whole-%d' % total, tokens, row_whole, 3)
                split = rig.split('split-%d' % total, tokens, row_split, 2)
                self.assertTrue(torch.equal(whole, split), 'logits differ at P=%d' % total)
                assert_same_state(self, rig.state(row_whole, total, 3), rig.state(row_split, total, 2), 'P=%d' % total)

    def test_the_solo_step_plan_is_exact_too(self):
        rig = Rig()
        tokens = F.prompt(40000, seed=7)
        row_whole, row_split = rig.pool.row(40000), rig.pool.row(40000)
        whole = rig.whole('whole', tokens, row_whole, 3)
        split = rig.split('split', tokens, row_split, 2, decoding=False)
        self.assertTrue(torch.equal(whole, split))
        assert_same_state(self, rig.state(row_whole, 40000, 3), rig.state(row_split, 40000, 2), 'solo plan')

    def test_another_step_size_is_exact(self):
        rig = Rig()
        tokens = F.prompt(30000, seed=9)
        row_whole, row_split = rig.pool.row(30000), rig.pool.row(30000)
        whole = rig.whole('whole', tokens, row_whole, 3)
        split = rig.split('split', tokens, row_split, 2, steps=policy.plan(30000, step=4 * CHUNK))
        self.assertTrue(torch.equal(whole, split))
        assert_same_state(self, rig.state(row_whole, 30000, 3), rig.state(row_split, 30000, 2), 'step 8192')

    def test_work_that_does_not_touch_the_scratch_between_the_steps_changes_nothing(self):
        rig = Rig()
        total = 14000
        tokens = F.prompt(total, seed=3)
        row_whole, row_split = rig.pool.row(total), rig.pool.row(total)
        whole = rig.whole('whole', tokens, row_whole, 3)

        def decode_rounds(index):
            # decode rounds read and write the batched buffers and the decoders' own blocks: here, a write to another request's slot and blocks
            for dn in (layer.attention for layer in rig.model.layers if not layer.is_full_attention):
                dn.slots[7] = (torch.full((1,), float(index)), [])

        split = rig.split('split', tokens, row_split, 2, between=decode_rounds)
        self.assertTrue(torch.equal(whole, split))
        assert_same_state(self, rig.state(row_whole, total, 3), rig.state(row_split, total, 2), 'interleaved')

    def test_the_harness_sees_a_continuation_from_the_wrong_state(self):
        """Sensitivity: skipping the GDN carry the steps rely on (the scratch re-zeroed between steps) changes the bytes."""
        rig = Rig()
        total = 9000
        tokens = F.prompt(total, seed=5)
        row_whole, row_split = rig.pool.row(total), rig.pool.row(total)
        whole = rig.whole('whole', tokens, row_whole, 3)

        def clobber(index):
            with rig.scope():
                previous = rig.model._bind_gdn_prefill_scratch()
                rig.model._reset_gdn_state_for_new_sequence()
                rig.model._unbind_gdn_prefill_scratch(previous)

        split = rig.split('split', tokens, row_split, 2, between=clobber)
        self.assertFalse(torch.equal(whole, split) and all(
            torch.equal(a, b) for a, b in zip(rig.toy.kv(row_whole, total), rig.toy.kv(row_split, total))
            ) and all(torch.equal(a[0], b[0]) for a, b in zip(rig.toy.slot(3), rig.toy.slot(2))),
            'the harness cannot see a continuation from a clobbered scratch')

    def test_a_short_prompt_is_the_stock_path_byte_for_byte(self):
        rig = Rig()
        tokens = F.prompt(3000, seed=2)
        row = rig.pool.row(3000)
        rig.whole('short', tokens, row, 1)
        self.assertEqual(rig.marker('lever N route req'), [], 'a whole prompt never enters the route')
        stock = Rig(environ={}, install=False)
        stock_row = stock.pool.row(3000)
        stock.call('short', tokens, stock_row, 0, 3000, 3000, 1, announce=False)
        assert_same_state(self, rig.state(row, 3000, 1), stock.state(stock_row, 3000, 1), 'short')


class SkipTests(unittest.TestCase):
    def test_only_the_last_step_writes_the_slot_and_snapshots_the_scratch(self):
        rig = Rig()
        total = 9000
        tokens = F.prompt(total, seed=4)
        row = rig.pool.row(total)
        steps = policy.plan(total)
        fake_before = []
        for index, (start, end) in enumerate(steps):
            rig.call('req', tokens, row, start, end, total, 2)
            slot_written = any(entry is not None for entry in rig.toy.slot(2))
            last = index == len(steps) - 1
            self.assertEqual(slot_written, last, 'slot written after step %d of %d' % (index + 1, len(steps)))
            self.assertIs(vars(rig.model)[route.WROTE_ATTR], last)
            # a to_torch of the recurrent state is the snapshot: 2 shards x (1 + K conv) per layer happen on the last step only
            snapshots = sum(1 for entry in rig.fake.log if entry[0] == 'to_torch' and len(entry[1]) == 4)
            fake_before.append(snapshots)
        self.assertEqual(fake_before[:-1], [0] * (len(steps) - 1))
        self.assertGreater(fake_before[-1], 0)

    def test_the_route_lines_name_each_step_and_the_final_one(self):
        rig = Rig()
        total = 6145
        tokens = F.prompt(total, seed=6)
        row = rig.pool.row(total)
        rig.split('req', tokens, row, 1)
        lines = rig.marker('lever N route req=')
        self.assertEqual(len(lines), 3)
        for line, (start, end, final) in zip(lines, ((0, 2048, 0), (2048, 4096, 0), (4096, 6145, 1))):
            self.assertIn('req=req start=%d end=%d prompt=6145 final=%d wrote_slot=%d ms=' % (start, end, final, final), line)
            self.assertIn('programs=', line)

    def test_a_step_reports_the_program_cache_and_it_does_not_grow(self):
        rig = Rig()
        total = 10000
        tokens = F.prompt(total, seed=8)
        row = rig.pool.row(total)
        with rig.scope():
            route.warm(mock.Mock(max_num_blocks_per_req=4096), rig.model, environ=ON, log=rig.log)
        rig.lines.clear()
        rig.split('req', tokens, row, 1)
        for line in rig.marker('lever N route req='):
            before, after = line.rsplit('programs=', 1)[1].split(' window=')[0].split('->')
            self.assertEqual(before, after)
            self.assertIn(' window=0', line)


class OwnerTests(unittest.TestCase):
    def started(self, total=9000):
        rig = Rig()
        tokens = F.prompt(total, seed=11)
        row = rig.pool.row(total)
        rig.call('a', tokens, row, 0, 2048, total, 1)
        return rig, tokens, row, total

    def refused(self, rig, call):
        before = list(rig.fake.log)
        with self.assertRaises(AssertionError) as caught:
            call()
        self.assertEqual(rig.fake.log, before, 'refused before any device work')
        return str(caught.exception)

    def test_the_owner_continues(self):
        rig, tokens, row, total = self.started()
        self.assertEqual(route.owner(rig.model), ('a', 2048))
        rig.call('a', tokens, row, 2048, 4096, total, 1)
        self.assertEqual(route.owner(rig.model), ('a', 4096))

    def test_a_replayed_step_is_refused(self):
        rig, tokens, row, total = self.started()
        rig.call('a', tokens, row, 2048, 4096, total, 1)
        message = self.refused(rig, lambda: rig.call('a', tokens, row, 2048, 4096, total, 1))
        self.assertIn('owned by', message)

    def test_a_skipped_chunk_is_refused(self):
        rig, tokens, row, total = self.started()
        message = self.refused(rig, lambda: rig.call('a', tokens, row, 4096, 6144, total, 1))
        self.assertIn('owned by', message)

    def test_another_requests_continuation_is_refused(self):
        rig, tokens, row, total = self.started()
        message = self.refused(rig, lambda: rig.call('b', tokens, row, 2048, 4096, total, 1))
        self.assertIn("owned by ('a', 2048)", message)

    def test_a_continuation_with_no_owner_is_refused(self):
        rig = Rig()
        tokens = F.prompt(9000, seed=12)
        row = rig.pool.row(9000)
        self.refused(rig, lambda: rig.call('a', tokens, row, 2048, 4096, 9000, 1))

    def test_a_whole_prompt_between_the_steps_takes_the_owner_away(self):
        """The stock path resets the scratch at start 0: the suspended prompt's continuation must be refused, not run on another prompt's state."""
        rig, tokens, row, total = self.started()
        other = F.prompt(3000, seed=13)
        rig.whole('b', other, rig.pool.row(3000), 3)
        self.assertIsNone(route.owner(rig.model))
        self.refused(rig, lambda: rig.call('a', tokens, row, 2048, 4096, total, 1))

    def test_an_unannounced_prompt_takes_the_owner_away_too(self):
        rig, tokens, row, total = self.started()
        rig.call('warm', F.prompt(2100, seed=1), rig.pool.row(2100), 0, 2100, 2100, 3, announce=False)
        self.assertIsNone(route.owner(rig.model))

    def test_a_new_start_zero_prompt_takes_a_new_owner(self):
        rig, tokens, row, total = self.started()
        rig.call('b', tokens, rig.pool.row(total), 0, 2048, total, 2)
        self.assertEqual(route.owner(rig.model), ('b', 2048))
        self.refused(rig, lambda: rig.call('a', tokens, row, 2048, 4096, total, 1))

    def test_release_clears_only_its_own_request(self):
        rig, tokens, row, total = self.started()
        self.assertFalse(route.release(rig.model, 'b'))
        self.assertEqual(route.owner(rig.model), ('a', 2048))
        self.assertTrue(route.release(rig.model, 'a'))
        self.assertIsNone(route.owner(rig.model))
        self.refused(rig, lambda: rig.call('a', tokens, row, 2048, 4096, total, 1))

    def test_the_final_step_clears_the_owner(self):
        rig = Rig()
        tokens = F.prompt(5000, seed=14)
        rig.split('a', tokens, rig.pool.row(5000), 1)
        self.assertIsNone(route.owner(rig.model))

    def test_an_error_inside_a_step_clears_the_owner_and_rebinds_the_decode_buffers(self):
        rig, tokens, row, total = self.started()
        original = rig.toy.chunk_masked

        def boom(*args, **kwargs):
            raise RuntimeError('device went away')

        rig.model._forward_prefill_chunk_masked_tp = boom
        with self.assertRaises(RuntimeError):
            rig.call('a', tokens, row, 2048, 4096, total, 1)
        rig.model._forward_prefill_chunk_masked_tp = original
        self.assertIsNone(route.owner(rig.model))
        self.assertTrue(all(layer.attention.B == 4 for layer in rig.model.layers if not layer.is_full_attention),
                        'the batched decode buffers are bound again')
        self.refused(rig, lambda: rig.call('a', tokens, row, 2048, 4096, total, 1))

    def test_the_foreign_fault_refuses_the_first_continuation_once(self):
        rig = Rig(environ=dict(ON, QWEN_FAST_LEVERN_FAULT='foreign'))
        tokens = F.prompt(9000, seed=15)
        row = rig.pool.row(9000)
        rig.call('a', tokens, row, 0, 2048, 9000, 1)
        message = self.refused(rig, lambda: rig.call('a', tokens, row, 2048, 4096, 9000, 1))
        self.assertIn('__foreign__', message)
        self.assertEqual(rig.handle.owner_refusals, 1)
        # the fault fires once: a restarted prompt completes
        rig.split('a', tokens, rig.pool.row(9000), 1)


class AnnouncementTests(unittest.TestCase):
    def test_a_chunked_step_with_no_announcement_is_refused(self):
        rig = Rig()
        tokens = F.prompt(9000, seed=16)
        row = rig.pool.row(9000)
        with self.assertRaises(AssertionError) as caught:
            rig.call('a', tokens, row, 2048, 4096, 9000, 1, announce=False)
        self.assertIn('without a lifecycle announcement', str(caught.exception))

    def test_the_runners_step_must_be_the_announced_one(self):
        rig = Rig()
        tokens = F.prompt(9000, seed=17)
        row = rig.pool.row(9000)
        rig.call('a', tokens, row, 0, 2048, 9000, 1)
        for kwargs in (dict(start_pos=[4096]), dict(lengths=[6144])):
            with self.subTest(**kwargs), self.assertRaises(AssertionError) as caught:
                rig.call('a', tokens, row, 2048, 4096, 9000, 1, **kwargs)
            self.assertIn('is not the lifecycle', str(caught.exception))
        self.assertEqual(route.owner(rig.model), ('a', 2048), 'a refused step leaves the scratch as it was')

    def test_the_announcement_is_consumed_by_one_call_and_withdrawn_when_unused(self):
        rig = Rig()
        route.announce(rig.model, route.Step('a', 0, 100, 100))
        route.withdraw(rig.model)
        self.assertNotIn(route.STEP_ATTR, vars(rig.model))
        with self.assertRaises(TypeError):
            route.announce(rig.model, ('a', 0, 100, 100))

    def test_a_split_step_in_a_process_without_the_route_is_refused(self):
        rig = Rig(environ=CONTROL)
        tokens = F.prompt(9000, seed=18)
        with self.assertRaises(AssertionError) as caught:
            rig.call('a', tokens, rig.pool.row(9000), 0, 2048, 9000, 1)
        self.assertIn('without QWEN_FAST_LEVER_N=1', str(caught.exception))

    def test_the_route_refuses_a_misaligned_boundary_before_any_device_work(self):
        rig = Rig()
        tokens = F.prompt(9000, seed=19)
        row = rig.pool.row(9000)
        before = list(rig.fake.log)
        for start, end in ((0, 1024), (0, 3000), (1024, 4096)):
            with self.subTest(start=start, end=end), self.assertRaises(AssertionError) as caught:
                rig.call('a', tokens, row, start, end, 9000, 1)
            self.assertIn('boundary', str(caught.exception))
        self.assertEqual(rig.fake.log, before)


class OffTests(unittest.TestCase):
    def test_with_neither_switch_nothing_is_installed(self):
        rig = Rig(environ={}, install=False)
        with rig.scope():
            self.assertIsNone(route.install(rig.runner, rig.model, environ={}, log=rig.log))
        self.assertNotIn('prefill_forward', vars(rig.wrapper))
        self.assertNotIn(route.ENTRY, vars(rig.model))
        self.assertEqual(rig.lines, [])

    def test_a_malformed_switch_is_refused(self):
        rig = Rig(environ={}, install=False)
        with self.assertRaises(ValueError):
            route.install(rig.runner, rig.model, environ={'QWEN_FAST_LEVER_N': 'yes'}, log=rig.log)

    def test_the_audit_alone_installs_the_wrapper_but_no_route(self):
        rig = Rig(environ=CONTROL)
        self.assertIn('prefill_forward', vars(rig.wrapper))
        self.assertNotIn(route.ENTRY, vars(rig.model))
        self.assertTrue(rig.marker('lever N route installed: route=0 audit=1'))

    def test_uninstall_removes_what_install_set(self):
        rig = Rig()
        rig.handle.uninstall()
        self.assertNotIn('prefill_forward', vars(rig.wrapper))
        self.assertNotIn(route.ENTRY, vars(rig.model))
        self.assertNotIn(route.HANDLE_ATTR, vars(rig.model))

    def test_an_unannounced_whole_prompt_with_the_route_installed_is_the_stock_result(self):
        rig = Rig()
        stock = Rig(environ={}, install=False)
        tokens = F.prompt(5000, seed=20)
        row_a, row_b = rig.pool.row(5000), stock.pool.row(5000)
        a = rig.call('x', tokens, row_a, 0, 5000, 5000, 1, announce=False)
        b = stock.call('x', tokens, row_b, 0, 5000, 5000, 1, announce=False)
        self.assertTrue(torch.equal(a, b))
        assert_same_state(self, rig.state(row_a, 5000, 1), stock.state(row_b, 5000, 1), 'unannounced')


class InstallTests(unittest.TestCase):
    def test_a_model_without_the_resumable_loops_is_refused_by_name(self):
        rig = Rig(environ={}, install=False, sources=F.stock_sources())
        with self.assertRaises(ValueError) as caught:
            route.install(rig.runner, rig.model, environ=ON, log=rig.log)
        message = str(caught.exception)
        self.assertIn('prefill_traced_chunked lacks start', message)
        self.assertIn('G1 stage', message)

    def test_a_second_install_is_refused(self):
        rig = Rig()
        with self.assertRaises(ValueError) as caught:
            route.install(rig.runner, rig.model, environ=ON, log=rig.log)
        self.assertIn('already installed', str(caught.exception))

    def test_a_wrapper_without_prefill_forward_is_refused(self):
        rig = Rig(environ={}, install=False)
        runner = mock.Mock()
        runner.model = object()
        with self.assertRaises(ValueError):
            route.install(runner, rig.model, environ=ON, log=rig.log)

    def test_the_audit_alone_needs_no_resumable_loops(self):
        rig = Rig(environ={}, install=False, sources=F.stock_sources())
        self.assertIsNotNone(route.install(rig.runner, rig.model, environ=CONTROL, log=rig.log))


class AuditTests(unittest.TestCase):
    def digests(self, rig):
        return [line.split('lever N digest ')[1] for line in rig.marker('lever N digest')]

    def test_the_control_and_the_split_arm_log_the_same_digests(self):
        for total in (4097, 9000, 12288):
            with self.subTest(total=total):
                control, split = Rig(environ=CONTROL), Rig(environ=AUDIT)
                tokens = F.prompt(total, seed=total + 1)
                control.whole('c', tokens, control.pool.row(total), 3)
                split.split('c', tokens, split.pool.row(total), 3)
                a, b = self.digests(control), self.digests(split)
                self.assertEqual(len(a), 1)
                self.assertEqual(a, b)
                fields = dict(item.split('=') for item in a[0].split()[0:0] + a[0].split()[1:])
                self.assertEqual(sorted(fields), ['kv_sha', 'logits_sha', 'prompt', 'slot_sha', 'tokens_sha'])
                self.assertTrue(all(len(fields[name]) == 32 for name in ('kv_sha', 'logits_sha', 'slot_sha', 'tokens_sha')))

    def test_a_different_prompt_has_different_digests(self):
        one, two = Rig(environ=CONTROL), Rig(environ=CONTROL)
        one.whole('c', F.prompt(5000, seed=1), one.pool.row(5000), 3)
        two.whole('c', F.prompt(5000, seed=2), two.pool.row(5000), 3)
        self.assertNotEqual(self.digests(one), self.digests(two))

    def test_a_corrupted_step_changes_the_digests(self):
        control, split = Rig(environ=CONTROL), Rig(environ=AUDIT)
        total = 9000
        tokens = F.prompt(total, seed=21)
        control.whole('c', tokens, control.pool.row(total), 3)
        row = split.pool.row(total)

        def clobber(index):
            with split.scope():
                previous = split.model._bind_gdn_prefill_scratch()
                split.model._reset_gdn_state_for_new_sequence()
                split.model._unbind_gdn_prefill_scratch(previous)

        split.split('c', tokens, row, 3, between=clobber)
        self.assertNotEqual(self.digests(control), self.digests(split))

    def test_no_digest_without_the_audit_switch(self):
        rig = Rig(environ=ON)
        rig.split('c', F.prompt(9000, seed=22), rig.pool.row(9000), 3)
        self.assertEqual(self.digests(rig), [])

    def test_the_audit_only_reads(self):
        control, plain = Rig(environ=CONTROL), Rig(environ=ON)
        total = 5000
        tokens = F.prompt(total, seed=23)
        row_a, row_b = control.pool.row(total), plain.pool.row(total)
        control.whole('c', tokens, row_a, 3)
        plain.whole('c', tokens, row_b, 3)
        assert_same_state(self, control.state(row_a, total, 3), plain.state(row_b, total, 3), 'audit reads only')
        writes = [entry for entry in control.fake.log if entry[0] in ('copy', 'h2d', 'deallocate')]
        baseline = [entry for entry in plain.fake.log if entry[0] in ('copy', 'h2d', 'deallocate')]
        self.assertEqual(writes, baseline)


class WarmTests(unittest.TestCase):
    def test_the_warm_runs_the_plans_three_steps_and_leaves_no_owner(self):
        rig = Rig()
        runner = mock.Mock(max_num_blocks_per_req=4096)
        with rig.scope():
            self.assertTrue(route.warm(runner, rig.model, environ=ON, log=rig.log))
        self.assertEqual(policy.plan(route.WARM_PROMPT), [(0, 2048), (2048, 4096), (4096, 6208)])
        lines = rig.marker('lever N route req=__levern_warm__')
        self.assertEqual(len(lines), 3)
        self.assertIsNone(route.owner(rig.model))
        self.assertNotIn(route.WROTE_ATTR, vars(rig.model))
        warmed = rig.marker('lever N route warmed before the packed traces: steps=3')
        self.assertEqual(len(warmed), 1)
        self.assertIn('programs=', warmed[0])

    def test_the_warm_is_left_alone_without_the_route(self):
        rig = Rig(environ={}, install=False)
        runner = mock.Mock(max_num_blocks_per_req=4096)
        self.assertFalse(route.warm(runner, rig.model, environ={}, log=rig.log))
        self.assertFalse(route.warm(runner, rig.model, environ=ON, log=rig.log), 'switch on but nothing installed')

    def test_the_warm_needs_the_page_width(self):
        rig = Rig()
        with rig.scope(), self.assertRaises(ValueError):
            route.warm(mock.Mock(max_num_blocks_per_req=None), rig.model, environ=ON, log=rig.log)

    def test_a_request_after_the_warm_is_exact(self):
        rig = Rig()
        runner = mock.Mock(max_num_blocks_per_req=4096)
        with rig.scope():
            route.warm(runner, rig.model, environ=ON, log=rig.log)
        total = 9000
        tokens = F.prompt(total, seed=24)
        row_whole, row_split = rig.pool.row(total), rig.pool.row(total)
        whole = rig.whole('w', tokens, row_whole, 3)
        split = rig.split('s', tokens, row_split, 2)
        self.assertTrue(torch.equal(whole, split))
        assert_same_state(self, rig.state(row_whole, total, 3), rig.state(row_split, total, 2), 'after the warm')


class MeasuredLineTests(unittest.TestCase):
    """The route line reports what the step did (a counted slot write, the window snapshot's programs), and the digest line is keyed on the prompt."""

    def test_wrote_slot_counts_the_real_slot_writes_and_the_shadow_comes_off(self):
        rig = Rig()
        total = 6145
        tokens = F.prompt(total, seed=31)
        real = rig.model._write_gdn_slot
        calls = []
        spy = lambda *a, **k: (calls.append(a[0]), real(*a, **k))[1]  # noqa: E731
        vars(rig.model)['_write_gdn_slot'] = spy
        rig.split('req', tokens, rig.pool.row(total), 1)
        self.assertEqual(calls, [1])
        lines = rig.marker('lever N route req=')
        self.assertEqual([line.split('wrote_slot=')[1].split(' ')[0] for line in lines], ['0', '0', '1'])
        self.assertIs(vars(rig.model)['_write_gdn_slot'], spy, 'the instance attribute the model had is restored')
        del vars(rig.model)['_write_gdn_slot']
        rig.split('again', tokens, rig.pool.row(total), 1)
        self.assertNotIn('_write_gdn_slot', vars(rig.model), 'no shadow is left behind')

    def test_the_step_reports_the_window_programs_it_saw(self):
        rig = Rig()
        total = 6145
        tokens = F.prompt(total, seed=32)
        counter = iter(range(0, 100, 3))
        with mock.patch.object(route, 'window_programs', lambda: next(counter)):
            rig.split('req', tokens, rig.pool.row(total), 1)
        windows = [line.rsplit('window=', 1)[1] for line in rig.marker('lever N route req=')]
        self.assertEqual(windows, ['3', '3', '3'])

    def test_a_digest_names_its_prompt_and_a_long_prompt_skips_the_kv_read(self):
        rig = Rig(AUDIT)
        total = 6145
        tokens = F.prompt(total, seed=33)
        rig.split('req', tokens, rig.pool.row(total), 1)
        line = rig.marker('lever N digest')[0]
        self.assertIn('tokens_sha=' + route.tokens_sha(tokens.reshape(1, -1), total), line)
        self.assertNotIn('kv_sha=' + policy.KV_SKIPPED, line)
        with mock.patch.object(policy, 'KV_DIGEST_MAX_PROMPT', 4096), mock.patch.object(route, 'kv_digest', side_effect=AssertionError('read')):
            rig.split('again', tokens, rig.pool.row(total), 1)
        self.assertIn('kv_sha=' + policy.KV_SKIPPED, rig.marker('lever N digest')[-1])


class TracedLoopTests(unittest.TestCase):
    def test_the_traced_chunk_loop_is_exact_too(self):
        rig = Rig(traced=True)
        total = 9000
        tokens = F.prompt(total, seed=25)
        row_whole, row_split = rig.pool.row(total), rig.pool.row(total)
        whole = rig.whole('whole', tokens, row_whole, 3)
        split = rig.split('split', tokens, row_split, 2)
        self.assertTrue(torch.equal(whole, split))
        assert_same_state(self, rig.state(row_whole, total, 3), rig.state(row_split, total, 2), 'traced')


if __name__ == '__main__':
    unittest.main()
