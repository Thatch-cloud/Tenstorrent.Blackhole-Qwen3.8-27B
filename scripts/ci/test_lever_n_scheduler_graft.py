"""The scheduler graft: M2 item 2, applied to the plugin source as captured.

fixtures/plugin_scheduler.py is TTScheduler as cpu-probe 35665853903 dumped it from
/opt/qwen-fast-plugin/src/vllm_tt_plugin/scheduler.py (sha256 a1bd6257d3a14c90, 207
lines). Testing against the real text matters more than usual here: the file carries
_schedule_decode_only immediately below the target, with a nearly identical
hide-restore-in-finally shape, so a text-scoped edit could plausibly land on the wrong
method and nothing downstream would say so.

Run 35690327326 is why this graft exists. M2's one-in-flight rule was delivered as an
overlay - serving_one_in_flight.install() setting scheduler_config.scheduler_cls - and
the plugin's own platform.check_and_update_config overwrites that attribute at lines
1079 and 1085. Last writer wins, so the policy never ran once. A marker firing from
inside the method proved it: zero occurrences, while install()'s marker fired three
times.

docs/lever-n-plugin-contract-2026-09-19.md had said where it belongs: two changes
"both in the vLLM TT plugin, not our overlay code".
"""

import ast
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent))

from lever_n_model_patch import MARKER_SCHEDULER, patch_scheduler

FIXTURE = Path(__file__).parent / 'fixtures' / 'plugin_scheduler.py'
NEWLINE = chr(10)


def source():
    return FIXTURE.read_text(encoding='utf-8')


def method(text, name):
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ''.join(lines[node.lineno - 1:node.end_lineno])
    raise AssertionError('no method named %s' % name)


class SchedulerGraftTests(unittest.TestCase):
    def test_the_fixture_is_the_shape_the_graft_anchors_to(self):
        text = source()
        self.assertIn('def _schedule_prefill_only', text)
        self.assertIn('def _schedule_decode_only', text)
        self.assertIn('self.max_num_running_reqs = max(0, saved_max - len(pure_decodes))', text)
        self.assertIn('create_request_queue(self.policy)', text,
                      'the hide idiom the graft copies must already exist in the file')

    def test_it_applies_and_parses(self):
        ast.parse(patch_scheduler(source()))

    def test_decode_only_is_untouched(self):
        """The real risk here. _schedule_decode_only sits directly below the target
        with the same hide-restore-in-finally shape, and it already hides its queues
        correctly - editing it would be both wrong and silent."""
        before = method(source(), '_schedule_decode_only')
        after = method(patch_scheduler(source()), '_schedule_decode_only')
        self.assertEqual(before, after)

    def test_a_partial_hides_both_queues(self):
        body = method(patch_scheduler(source()), '_schedule_prefill_only')
        self.assertIn('if partial_prefills:', body)
        self.assertIn('self.waiting = create_request_queue(self.policy)', body)
        self.assertIn('self.skipped_waiting = create_request_queue(self.policy)', body)

    def test_the_queues_are_merged_back_not_dropped(self):
        """prepend_requests, exactly as _schedule_decode_only does it: whatever the
        base scheduler put on the blank queue has to survive the restore."""
        body = method(patch_scheduler(source()), '_schedule_prefill_only')
        self.assertIn('_qwen_saved_waiting.prepend_requests(self.waiting)', body)
        self.assertIn('_qwen_saved_skipped.prepend_requests(self.skipped_waiting)', body)
        self.assertIn('self.waiting = _qwen_saved_waiting', body)

    def test_no_partial_caps_capacity_to_one(self):
        """Run 35690327326 died on two FRESH prompts in one step. The shipped cap is
        saved_max minus the decodes, which is four when nothing is running.

        Driven rather than string-matched. This used to name the generated line
        verbatim and broke when the cap grew a third branch for the prefill gate,
        with nothing wrong - what matters is the cap it produces.
        """
        self.assertEqual(run_prefill_only(partials=0, decodes=0)['max_running'], 1)
        self.assertEqual(run_prefill_only(partials=0, decodes=2)['max_running'], 1)

    def test_the_marker_is_emitted_from_inside_the_method(self):
        """install()'s marker proved only that a config string had been assigned. This
        one cannot appear unless the grafted code actually executed."""
        body = method(patch_scheduler(source()), '_schedule_prefill_only')
        self.assertIn(MARKER_SCHEDULER, body)
        self.assertIn('logger.info(', body)

    def test_the_original_restore_still_happens_in_finally(self):
        body = method(patch_scheduler(source()), '_schedule_prefill_only')
        tail = body[body.index('finally:'):]
        for statement in ('self.running.extend(pure_decodes)',
                          'self.max_num_running_reqs = saved_max',
                          'self.waiting = _qwen_saved_waiting'):
            with self.subTest(statement=statement):
                self.assertIn(statement, tail)

    def test_patching_twice_raises(self):
        once = patch_scheduler(source())
        with self.assertRaisesRegex(ValueError, 'already carries'):
            patch_scheduler(once)

    def test_a_source_without_the_anchor_raises(self):
        stub = ('class X:' + NEWLINE +
                '    def _schedule_prefill_only(self):' + NEWLINE +
                '        return 1' + NEWLINE)
        with self.assertRaises(ValueError):
            patch_scheduler(stub)


if __name__ == '__main__':
    unittest.main()


def run_prefill_only(partials=0, decodes=0, gate=None, saved_max=4):
    """Build the patched TTScheduler and actually run _schedule_prefill_only.

    Returns what the base scheduler saw when it was called - the capacity cap and
    whether the waiting queue had been blanked - plus whether it was restored after.
    String-matching the generated source cannot tell any of that apart.
    """
    import sys as _sys
    import types as _types

    module = _types.ModuleType('patched_scheduler_graft')
    seen = {}
    state = {}

    class Queue(list):
        def prepend_requests(self, other):
            self[:0] = list(other)

    class Base(object):
        def schedule(self):
            seen['max_running'] = self.max_num_running_reqs
            seen['waiting_is_blank'] = self.waiting is not state['waiting']
            return _types.SimpleNamespace(total_num_scheduled_tokens=1)

    class Mode(object):
        DEFAULT = 'DEFAULT'
        PREFILL_ONLY = 'PREFILL_ONLY'
        DECODE_ONLY = 'DECODE_ONLY'

    module.__dict__.update(
        AsyncScheduler=Base, SchedulerOutput=object, Request=object,
        TTSchedulingMode=Mode, RequestQueue=Queue,
        create_request_queue=lambda policy: Queue(),
        cast=lambda kind, value: value,
        logger=_types.SimpleNamespace(info=lambda *a, **k: None))
    exec(compile(patch_scheduler(source()), 'patched_scheduler_graft', 'exec'),
         module.__dict__)

    def request(is_chunk):
        return _types.SimpleNamespace(is_prefill_chunk=is_chunk)

    built = module.TTScheduler.__new__(module.TTScheduler)
    built.running = ([request(True)] * partials) + ([request(False)] * decodes)
    state['waiting'] = Queue([request(False)])
    built.waiting = state['waiting']
    built.skipped_waiting = Queue()
    built.policy = None
    built.max_num_running_reqs = saved_max

    saved_gate = _sys.modules.get('_qwen_prefill_gate')
    _sys.modules.pop('_qwen_prefill_gate', None)
    if gate is not None:
        holder = _types.ModuleType('_qwen_prefill_gate')
        holder.held = gate
        _sys.modules['_qwen_prefill_gate'] = holder
    try:
        built._schedule_prefill_only()
    finally:
        _sys.modules.pop('_qwen_prefill_gate', None)
        if saved_gate is not None:
            _sys.modules['_qwen_prefill_gate'] = saved_gate
    seen['restored'] = built.waiting is state['waiting']
    seen['max_restored'] = built.max_num_running_reqs == saved_max
    return seen


class PrefillGateTests(unittest.TestCase):
    """The scheduler must ask the lifecycle, not infer from is_prefill_chunk.

    Run 35714211185 (v69) died with a fresh request admitted while the lifecycle held
    its prefill slot. partial_prefills is false at BOTH ends of a prefill - before the
    first chunk exists and after the last is consumed - so it cannot answer "is a
    prefill in flight". The lifecycle publishes request_id under a fixed sys.modules
    key; this reads it.
    """

    def test_a_held_gate_admits_nobody_and_hides_the_queue(self):
        seen = run_prefill_only(partials=0, decodes=1, gate='cmpl-someone')
        self.assertEqual(seen['max_running'], 0)
        self.assertTrue(seen['waiting_is_blank'])

    def test_without_the_gate_a_fresh_prompt_is_still_admitted(self):
        """The negative control. If this also capped at zero, the gate would be doing
        nothing visible and the engine would simply never admit anyone."""
        seen = run_prefill_only(partials=0, decodes=1, gate=None)
        self.assertEqual(seen['max_running'], 1)
        self.assertFalse(seen['waiting_is_blank'])

    def test_a_partial_still_hides_regardless_of_the_gate(self):
        for gate in (None, 'cmpl-someone'):
            seen = run_prefill_only(partials=1, decodes=1, gate=gate)
            self.assertEqual(seen['max_running'], 1, gate)
            self.assertTrue(seen['waiting_is_blank'], gate)

    def test_the_queue_is_restored_when_the_gate_hid_it(self):
        """Hiding without restoring loses the queue, so the hide site and the finally
        block must share one condition - which is why both use _qwen_hide."""
        seen = run_prefill_only(partials=0, decodes=1, gate='cmpl-someone')
        # Assert it was hidden FIRST. Without this the test passes unpatched, where
        # nothing hides and so 'restored' is trivially true - a green light for the
        # wrong reason.
        self.assertTrue(seen['waiting_is_blank'])
        self.assertTrue(seen['restored'])
        self.assertTrue(seen['max_restored'])

    def test_an_unreadable_gate_reads_as_not_held(self):
        """The plugin must never hard-depend on the baked tree being importable: a run
        where the lifecycle never loaded has to schedule exactly as it did before."""
        import sys as _sys
        import types as _types

        class Broken(object):
            # sys.modules accepts any object, and a module type is immutable, so the
            # raising attribute lives on an ordinary class. getattr with a default
            # swallows only AttributeError - a RuntimeError propagates, which is
            # exactly why the graft wraps the read in try/except BaseException.
            @property
            def held(self):
                raise RuntimeError('gate unreadable')

        broken = Broken()
        saved = _sys.modules.get('_qwen_prefill_gate')
        _sys.modules['_qwen_prefill_gate'] = broken
        try:
            seen = run_prefill_only(partials=0, decodes=1, gate=None)
        finally:
            _sys.modules.pop('_qwen_prefill_gate', None)
            if saved is not None:
                _sys.modules['_qwen_prefill_gate'] = saved
        self.assertEqual(seen['max_running'], 1)


class LifecyclePublishesTheGateTests(unittest.TestCase):
    """The other half: the lifecycle actually sets what the scheduler reads."""

    def test_setting_request_id_publishes_it_and_clearing_it_clears(self):
        import serving_lifecycle
        holder = serving_lifecycle.FastServingLifecycle.__new__(
            serving_lifecycle.FastServingLifecycle)
        holder.request_id = 'cmpl-abc'
        self.assertEqual(serving_lifecycle.prefill_gate().held, 'cmpl-abc')
        holder.request_id = None
        self.assertIsNone(serving_lifecycle.prefill_gate().held)

    def test_the_property_reads_back_what_was_set(self):
        import serving_lifecycle
        holder = serving_lifecycle.FastServingLifecycle.__new__(
            serving_lifecycle.FastServingLifecycle)
        holder.request_id = 'cmpl-xyz'
        self.assertEqual(holder.request_id, 'cmpl-xyz')
