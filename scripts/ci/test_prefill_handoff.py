"""PF2-H, the device-side hand-off of the final step's GDN state (prefill_handoff, levern_route), executed on the REAL staged model.py with the recording fake ttnn and the toy
hybrid model of qwen_prefix_model_fixture, on both routes (the stage-1 route and the merged route).

What is held:
  settings   the flags are strict: unset is off, '0' is off with the timing line, '1' is the device hand-off, anything else raises; the audit needs '1', the flag needs Lever N;
  exact      the decode slot a device-side hand-off writes equals the host path's, byte for byte, with the logits and the KV, at every boundary length, on both routes;
  skips      on the device path the final step reads nothing back to the host and uploads nothing (the 240 reads and 240 uploads of the profile, 15 and 15 on the toy), makes one
             clone per scratch tensor, writes the slot once and counts it (wrote_slot), and leaves the persistent scratch intact for the next prompt (the harness SEES a hand-off
             that gave the scratch itself to the consuming write);
  off        with neither flag set the route is the route without the lever (no handoff, no timing line); '0' runs the same device calls as unset and only adds the timing line;
  audit      the audited step reads the scratch and the decode state, finds the slot equal and the other rows unchanged ('exact=True'), and a corrupting clone is caught and raises
             (and without the audit the same corruption goes unnoticed: the audit has teeth); it runs for the first 8 final steps and then every 24th;
  fallback   a clone that fails writes nothing, logs 'fell back' once, serves the request on the host path (the slot equals it) and latches the device path off;
  warm       the warm takes the device path, so no program is compiled by any later final step;
  census     the module makes no ttnn or model call the audit has not seen."""

import ast
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import levern_policy as policy  # noqa: E402
import levern_route as route  # noqa: E402
import prefill_handoff as handoff_module  # noqa: E402
import qwen_prefix_model_fixture as F  # noqa: E402
from test_levern_route import Rig, assert_same_state  # noqa: E402

DEVICE = {'QWEN_FAST_LEVER_N': '1', 'QWEN_FAST_PREFILL_HANDOFF': '1'}
HOST_TIMED = {'QWEN_FAST_LEVER_N': '1', 'QWEN_FAST_PREFILL_HANDOFF': '0'}
ON = {'QWEN_FAST_LEVER_N': '1'}
AUDITED = dict(DEVICE, QWEN_FAST_PREFILL_HANDOFF_AUDIT='1')
LENGTHS = (4096, 4097, 6143, 6208, 8193, 12289)      # a prompt of one step is the stock path, which this lever leaves alone (WholePromptTests)


def final_calls(fake, start_mark):
    """The fake's calls since `start_mark`, by kind."""
    kinds = {}
    for entry in fake.log[start_mark:]:
        kinds[entry[0]] = kinds.get(entry[0], 0) + 1
    return kinds


def device_uploads(fake, start_mark):
    return sum(1 for entry in fake.log[start_mark:] if entry[0] == 'from_torch' and entry[1])


def run_prompt(rig, total, seed=3, slot=2, req='req'):
    tokens = F.prompt(total, seed=seed)
    row = rig.pool.row(total)
    mark = len(rig.fake.log)
    logits = rig.split(req, tokens, row, slot)
    return logits, rig.state(row, total, slot), mark


class SettingsTests(unittest.TestCase):
    def test_unset_is_off(self):
        self.assertEqual(handoff_module.settings({}), (False, False, False))
        self.assertFalse(handoff_module.requested({}))

    def test_zero_is_off_with_the_timing_line_and_one_is_the_device_path(self):
        self.assertEqual(handoff_module.settings(HOST_TIMED), (True, False, False))
        self.assertEqual(handoff_module.settings(DEVICE), (True, True, False))
        self.assertEqual(handoff_module.settings(AUDITED), (True, True, True))
        self.assertTrue(handoff_module.requested(HOST_TIMED))

    def test_anything_else_raises(self):
        for value in ('', '2', 'true', 'on', ' 1'):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    handoff_module.settings(dict(ON, QWEN_FAST_PREFILL_HANDOFF=value))
        with self.assertRaises(ValueError):
            handoff_module.settings(dict(DEVICE, QWEN_FAST_PREFILL_HANDOFF_AUDIT='yes'))

    def test_the_audit_needs_the_device_path_and_the_flag_needs_lever_n(self):
        with self.assertRaises(ValueError):
            handoff_module.settings(dict(ON, QWEN_FAST_PREFILL_HANDOFF_AUDIT='1'))
        with self.assertRaises(ValueError):
            handoff_module.settings(dict(HOST_TIMED, QWEN_FAST_PREFILL_HANDOFF_AUDIT='1'))
        with self.assertRaises(ValueError):
            handoff_module.settings({'QWEN_FAST_PREFILL_HANDOFF': '1'})

    def test_install_refuses_the_flag_without_the_route_and_a_malformed_value(self):
        for environ in ({'QWEN_FAST_LEVERN_AUDIT': '1', 'QWEN_FAST_PREFILL_HANDOFF': '1'}, dict(ON, QWEN_FAST_PREFILL_HANDOFF='maybe')):
            with self.subTest(environ=environ):
                with self.assertRaises(ValueError):
                    Rig(environ)

    def test_audit_due_is_the_first_eight_then_every_twenty_fourth(self):
        handoff = handoff_module.Handoff(True, True, True, environ=AUDITED)
        due = []
        for index in range(60):
            handoff.calls = index
            if handoff.audit_due():
                due.append(index)
        self.assertEqual(due, list(range(8)) + [31, 55])
        handoff.latched = 'x'
        handoff.calls = 0
        self.assertFalse(handoff.audit_due())


class StageOneRouteTests(unittest.TestCase):
    def test_the_device_slot_equals_the_host_slot_at_every_boundary_length(self):
        host, device = Rig(ON), Rig(DEVICE)
        for total in LENGTHS:
            with self.subTest(total=total):
                logits_h, state_h, _ = run_prompt(host, total, seed=total)
                logits_d, state_d, _ = run_prompt(device, total, seed=total)
                self.assertTrue(torch.equal(logits_h, logits_d))
                assert_same_state(self, state_h, state_d, 'P=%d' % total)

    def test_the_final_step_reads_nothing_back_and_uploads_nothing(self):
        host, device = Rig(ON), Rig(DEVICE)
        _, _, mark_h = run_prompt(host, 4097)
        _, _, mark_d = run_prompt(device, 4097)
        scratch_tensors = 3 * 5                      # three GDN layers of the toy: a recurrent state and four conv taps each
        uploads_h = device_uploads(host.fake, mark_h)
        uploads_d = device_uploads(device.fake, mark_d)
        self.assertEqual(uploads_h - uploads_d, scratch_tensors)
        clones = [entry for entry in device.fake.log[mark_d:] if entry[0] == 'clone']
        self.assertEqual(len(clones), scratch_tensors)
        reads_h = final_calls(host.fake, mark_h).get('to_torch', 0)
        reads_d = final_calls(device.fake, mark_d).get('to_torch', 0)
        self.assertEqual(reads_h - reads_d, scratch_tensors)
        self.assertEqual(device.toy.model.layers[0].attention.write_order, [2])

    def test_a_non_final_step_does_nothing_new(self):
        rig = Rig(DEVICE)
        tokens = F.prompt(6208, seed=5)
        row = rig.pool.row(6208)
        mark = len(rig.fake.log)
        rig.call('p', tokens, row, 0, 2048, 6208, 2)
        self.assertEqual([entry for entry in rig.fake.log[mark:] if entry[0] == 'clone'], [])
        self.assertEqual(rig.handle.handoff.calls, 0)
        self.assertFalse(vars(rig.model).get(route.WROTE_ATTR))

    def test_the_slot_write_is_counted_for_the_lifecycle(self):
        rig = Rig(DEVICE)
        tokens = F.prompt(4097, seed=5)
        row = rig.pool.row(4097)
        steps = policy.plan(4097, decoding=True)
        for start, end in steps[:-1]:
            rig.call('p', tokens, row, start, end, 4097, 2)
            self.assertFalse(vars(rig.model).get(route.WROTE_ATTR))
        rig.call('p', tokens, row, steps[-1][0], steps[-1][1], 4097, 2)
        self.assertTrue(vars(rig.model).get(route.WROTE_ATTR))

    def test_the_scratch_survives_the_hand_off_for_the_next_prompt(self):
        rig = Rig(DEVICE)
        for total, slot in ((4097, 1), (6208, 3), (4096, 0)):
            _, state, _ = run_prompt(rig, total, seed=total, slot=slot, req='r%d' % total)
        whole = Rig(ON)
        tokens = F.prompt(4096, seed=4096)
        row = whole.pool.row(4096)
        whole.whole('w', tokens, row, 0)
        assert_same_state(self, state, whole.state(row, 4096, 0), 'third prompt on a reused scratch')

    def test_a_hand_off_that_gave_the_scratch_to_the_consuming_write_is_seen(self):
        rig = Rig(DEVICE)

        def no_clone(self, ttnn, model):
            model._ensure_gdn_prefill_scratch()
            return [(dn, rec, convs) for dn, rec, convs, _carry, _zero in model._gdn_prefill_scratch]

        with mock.patch.object(handoff_module.Handoff, 'clones', no_clone):
            run_prompt(rig, 4097, seed=1, slot=1, req='a')
            with self.assertRaises(RuntimeError):
                run_prompt(rig, 4097, seed=2, slot=1, req='b')

    def test_unset_is_the_route_without_the_lever(self):
        rig = Rig(ON)
        self.assertIsNone(rig.handle.handoff)
        run_prompt(rig, 4097)
        self.assertEqual(rig.marker('[PF2]'), [])
        self.assertEqual(rig.marker('handoff'), [])

    def test_with_both_flags_unset_the_module_is_never_imported(self):
        import builtins

        imported = []
        real = builtins.__import__

        def spy(name, *args, **kwargs):
            if name == 'prefill_handoff':
                imported.append(name)
            return real(name, *args, **kwargs)

        with mock.patch.object(builtins, '__import__', spy):
            rig = Rig(ON)
            run_prompt(rig, 4097)
            self.assertEqual(imported, [])
            self.assertIsNone(rig.handle.handoff)
            Rig(AUDITED)
        self.assertEqual(imported, ['prefill_handoff'])

    def test_zero_runs_the_same_device_calls_and_only_adds_the_timing_line(self):
        plain, timed = Rig(ON), Rig(HOST_TIMED)
        _, state_p, mark_p = run_prompt(plain, 4097)
        _, state_t, mark_t = run_prompt(timed, 4097)
        assert_same_state(self, state_p, state_t, 'timed control')
        self.assertEqual(plain.fake.log[mark_p:], timed.fake.log[mark_t:])
        lines = timed.marker(handoff_module.TIMING_MARKER)
        self.assertEqual(len(lines), 1)
        self.assertIn('mode=host', lines[0])
        self.assertIn('read_ms=', lines[0])
        self.assertEqual(timed.marker(handoff_module.ENGAGED_MARKER), [])

    def test_the_device_path_logs_engaged_once_and_a_timing_line_per_final_step(self):
        rig = Rig(DEVICE)
        for total in (4097, 6208, 8193):
            run_prompt(rig, total, seed=total, req='r%d' % total)
        self.assertEqual(len(rig.marker(handoff_module.ENGAGED_MARKER)), 1)
        lines = rig.marker(handoff_module.TIMING_MARKER)
        self.assertEqual(len(lines), 3)
        self.assertTrue(all('mode=device' in line and 'clone_ms=' in line and 'layers=3' in line for line in lines))
        self.assertIn('handoff=device', rig.marker('route installed')[0])

    def test_the_install_line_names_the_mode(self):
        self.assertIn('handoff=host', Rig(HOST_TIMED).marker('route installed')[0])
        self.assertIn('handoff=device audit', Rig(AUDITED).marker('route installed')[0])
        self.assertNotIn('handoff', Rig(ON).marker('route installed')[0])


class WholePromptTests(unittest.TestCase):
    def test_a_prompt_of_one_step_is_the_stock_path_and_the_lever_leaves_it_alone(self):
        rig = Rig(DEVICE)
        for total in (2047, 2048, 2049):
            tokens = F.prompt(total, seed=total)
            row = rig.pool.row(total)
            mark = len(rig.fake.log)
            rig.split('w%d' % total, tokens, row, 1)
            self.assertEqual([entry for entry in rig.fake.log[mark:] if entry[0] == 'clone'], [])
        self.assertEqual(rig.marker(handoff_module.TIMING_MARKER), [])
        self.assertEqual(rig.handle.handoff.calls, 0)


class AuditTests(unittest.TestCase):
    def test_the_audited_final_steps_are_exact(self):
        rig = Rig(AUDITED)
        for index, total in enumerate((4097, 6208, 4096, 8193)):
            run_prompt(rig, total, seed=total, slot=index % 4, req='r%d' % index)
        lines = rig.marker(handoff_module.AUDIT_MARKER)
        self.assertEqual(len(lines), 4)
        self.assertTrue(all('exact=True' in line for line in lines))
        self.assertEqual(rig.marker(handoff_module.AUDIT_MISMATCH_MARKER), [])
        self.assertEqual(rig.handle.handoff.audited, 4)

    def test_the_audit_runs_on_the_first_eight_final_steps_then_every_twenty_fourth(self):
        rig = Rig(AUDITED)
        for index in range(33):
            run_prompt(rig, 4097, seed=index, slot=index % 4, req='r%d' % index)
        self.assertEqual(rig.handle.handoff.audited, 8 + 1)       # the first eight, then the 32nd call (index 31)

    def test_the_audit_asks_for_the_host_copy_and_the_plain_hand_off_does_not(self):
        self.assertTrue(Rig(AUDITED).handle.handoff.wants_host(False))
        plain = Rig(DEVICE).handle.handoff
        self.assertFalse(plain.wants_host(False))
        self.assertTrue(plain.wants_host(True))              # the Lever N digest audit still gets its bytes
        with mock.patch.dict(plain.environ, {'QWEN_PREFIX_DIGESTS': '1'}):
            self.assertTrue(plain.wants_host(False))

    def test_a_corrupting_clone_is_caught_by_the_audit_and_not_without_it(self):
        rig = Rig(AUDITED)
        rig.fake.clone_corrupt = True
        with self.assertRaises(AssertionError) as caught:
            run_prompt(rig, 4097, seed=1, slot=1)
        self.assertIn('audit mismatch', str(caught.exception))
        self.assertEqual(len(rig.marker(handoff_module.AUDIT_MISMATCH_MARKER)), 1)
        blind = Rig(DEVICE)
        blind.fake.clone_corrupt = True
        _, bad, _ = run_prompt(blind, 4097, seed=1, slot=1)
        good = Rig(ON)
        _, fine, _ = run_prompt(good, 4097, seed=1, slot=1)
        with self.assertRaises(AssertionError):
            assert_same_state(self, bad, fine, 'corrupt clone')

    def test_a_write_that_touches_another_slot_is_caught(self):
        rig = Rig(AUDITED)
        rig.fake.clone_corrupt = False
        original = rig.model.layers[0].attention.write_slot

        def leaky(slot, rec, convs):
            original(slot, rec, convs)
            rig.model.layers[0].attention.rec_state.data[0].add_(1.0)       # chip 0's row 0 of the batched state: slot 0 of a request that wrote slot 2

        rig.model.layers[0].attention.write_slot = leaky
        with self.assertRaises(AssertionError):
            run_prompt(rig, 4097, seed=1, slot=2)


class FallbackTests(unittest.TestCase):
    def test_a_refused_clone_falls_back_to_the_host_path_and_stays_there(self):
        rig = Rig(DEVICE)
        rig.fake.clone_refuse = True
        host = Rig(ON)
        for total, slot in ((4097, 1), (6208, 2)):
            _, state, _ = run_prompt(rig, total, seed=total, slot=slot, req='r%d' % total)
            _, expected, _ = run_prompt(host, total, seed=total, slot=slot, req='r%d' % total)
            assert_same_state(self, state, expected, 'fallback P=%d' % total)
        self.assertEqual(len(rig.marker(handoff_module.FELL_BACK_MARKER)), 1)
        self.assertIn('reason=RuntimeError', rig.marker(handoff_module.FELL_BACK_MARKER)[0])
        self.assertIsNotNone(rig.handle.handoff.latched)
        self.assertEqual(rig.marker(handoff_module.ENGAGED_MARKER), [])
        self.assertEqual(len([line for line in rig.marker(handoff_module.TIMING_MARKER) if 'mode=host' in line]), 2)       # a latched hand-off times the host path it fell back to

    def test_a_clone_that_fails_midway_leaves_the_slot_untouched(self):
        rig = Rig(DEVICE)
        original = rig.fake.clone
        state = {'count': 0}

        def flaky(tensor, **kwargs):
            state['count'] += 1
            if state['count'] == 7:
                raise RuntimeError('out of memory')
            return original(tensor, **kwargs)

        rig.fake.clone = flaky
        tokens = F.prompt(4097, seed=1)
        row = rig.pool.row(4097)
        with mock.patch.dict(sys.modules, {'ttnn': rig.fake.module()}):
            rig.split('r', tokens, row, 1)
        self.assertEqual(rig.marker(handoff_module.ENGAGED_MARKER), [])
        wrote = [layer.attention.write_order for layer in rig.model.layers if not layer.is_full_attention]
        self.assertEqual(wrote, [[1], [1], [1]])             # each layer written once, by the host path, after the clones were freed
        freed = [entry for entry in rig.fake.log if entry[0] == 'deallocate']
        self.assertTrue(freed)


class WarmTests(unittest.TestCase):
    def test_the_warm_compiles_what_the_final_steps_need(self):
        rig = Rig(DEVICE)
        with rig.scope():
            self.assertTrue(route.warm(mock.Mock(max_num_blocks_per_req=4096), rig.model, environ=DEVICE, log=rig.log))
        programs = set(rig.fake.programs)
        self.assertTrue(any(entry[0] == 'clone' for entry in programs))
        for total in (4097, 6208, 8193, 12289):
            run_prompt(rig, total, seed=total, req='r%d' % total)
        self.assertEqual(set(rig.fake.programs), programs)


class MergedRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global MergedEngine, MERGED
        from test_levern_prefix_route import MERGED, MergedEngine  # noqa: F401

    def prompt_run(self, environ, total=6209, slot=1, seed=11):
        engine = MergedEngine(environ=environ)
        tokens = F.prompt(total, seed=seed)
        row = engine.pool.row(total)
        mark = len(engine.fake.log)
        logits = engine.run('m', tokens, row, 0, slot)
        return engine, logits, engine.state(row, total, slot), mark

    def test_the_device_slot_equals_the_host_slot_on_the_merged_route(self):
        host, logits_h, state_h, _ = self.prompt_run(dict(MERGED))
        device, logits_d, state_d, mark = self.prompt_run(dict(MERGED, QWEN_FAST_PREFILL_HANDOFF='1'))
        self.assertTrue(torch.equal(logits_h, logits_d))
        assert_same_state(self, state_h, state_d, 'merged')
        self.assertTrue(device.handle.handoff.engaged >= 1)
        self.assertEqual(device.handle.handoff.fallbacks, 0)

    def test_the_merged_final_step_is_counted_and_the_audit_is_exact(self):
        engine, _, _, _ = self.prompt_run(dict(MERGED, QWEN_FAST_PREFILL_HANDOFF='1', QWEN_FAST_PREFILL_HANDOFF_AUDIT='1'))
        self.assertTrue(vars(engine.model).get(route.WROTE_ATTR))
        self.assertTrue(engine.marker('audit n=1 exact=True'))

    def test_the_merged_warm_takes_the_device_path_and_compiles_nothing_later(self):
        engine = MergedEngine(environ=dict(MERGED, QWEN_FAST_PREFILL_HANDOFF='1'))
        with engine.scope():
            route.warm(engine.runner, engine.model, environ=engine.merged_env, log=engine.log_line)
        programs = set(engine.fake.programs)
        self.assertTrue(any(entry[0] == 'clone' for entry in programs))
        total = 6209
        tokens = F.prompt(total, seed=2)
        row = engine.pool.row(total)
        engine.run('after', tokens, row, 0, 2)
        self.assertEqual(set(engine.fake.programs), programs)

    def test_a_merged_fallback_serves_the_request_from_the_host(self):
        engine = MergedEngine(environ=dict(MERGED, QWEN_FAST_PREFILL_HANDOFF='1'))
        engine.fake.clone_refuse = True
        total = 6209
        tokens = F.prompt(total, seed=3)
        row = engine.pool.row(total)
        engine.run('x', tokens, row, 0, 1)
        host = MergedEngine(environ=dict(MERGED))
        row_h = host.pool.row(total)
        host.run('x', tokens, row_h, 0, 1)
        assert_same_state(self, engine.state(row, total, 1), host.state(row_h, total, 1), 'merged fallback')
        self.assertEqual(len(engine.marker(handoff_module.FELL_BACK_MARKER)), 1)


class CensusTests(unittest.TestCase):
    SOURCE = HERE / 'prefill_handoff.py'
    ALLOWED_TTNN = {'clone', 'deallocate', 'to_torch', 'ConcatMeshToTensor'}
    ALLOWED_MODEL = {'_ensure_gdn_prefill_scratch'}

    def calls(self):
        tree = ast.parse(self.SOURCE.read_text(encoding='utf-8'))
        found = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                found.setdefault(node.func.value.id, set()).add(node.func.attr)
        return found

    def test_the_module_calls_only_audited_ttnn_and_model_methods(self):
        calls = self.calls()
        self.assertLessEqual(calls.get('ttnn', set()), self.ALLOWED_TTNN)
        self.assertLessEqual(calls.get('model', set()), self.ALLOWED_MODEL)

    def test_the_only_device_write_is_the_graft_s_own_write_slot(self):
        calls = self.calls()
        self.assertEqual(calls.get('dn', set()) - {'write_slot'}, set())
        source = self.SOURCE.read_text(encoding='utf-8')
        self.assertEqual(source.count('.write_slot('), 1 + 0)

    def test_the_module_imports_nothing_but_the_standard_library_at_the_top(self):
        tree = ast.parse(self.SOURCE.read_text(encoding='utf-8'))
        top = sorted(alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names)
        self.assertEqual(top, ['os', 'time'])

    def test_the_route_imports_the_module_only_when_a_flag_names_it(self):
        text = (HERE / 'levern_route.py').read_text(encoding='utf-8')
        self.assertEqual(text.count('import prefill_handoff'), 1)
        self.assertIn('HANDOFF_FLAGS', text)
        self.assertEqual(route.HANDOFF_FLAGS, (handoff_module.FLAG, handoff_module.AUDIT_FLAG))


if __name__ == '__main__':
    unittest.main()
