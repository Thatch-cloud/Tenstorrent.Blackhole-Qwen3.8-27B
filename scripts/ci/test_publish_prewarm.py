"""QWEN_FAST_PUBLISH_PREWARM (publish_prewarm.py) and QWEN_FAST_SEQ_PUBLISH_LOG (serving_sequential_step):
the k5dbg verdict's fix (section 2) and its confirming experiment's logging (section 3), with their wiring.

What is held here, all on the host:
  - warm(): every (rows, prefix) of the engine's captured buckets published once per process, in bucket
    then prefix order, as prepare_publication(features, prefix, position=device.position) and then
    discard_publication(publication); a second request warms nothing; nothing pending and the frontier
    unmoved afterwards (refused otherwise); history_rows below 2048 skipped; the marker line;
  - that call is DFlashRequestRuntime.publish's own - the sequential commit's - features included, and
    under QWEN_FAST_ROUND_B1 it takes the non-fused branch;
  - exactness on a torch fake running the REAL DFlashDevice.prepare_publication (both branches) and
    DraftKVHistory.prepare: a prewarm writes only the spares, and after the next real publications the
    committed history, the banks and the frontiers are bit for bit a never-warmed device's;
  - from_prefill: flag off (unset, 0, anything but 1) its parent's (6a49230d) calls exactly, publish_prewarm
    never imported; flag on, warm(device, engine) once, after the engine and before the request binds it,
    and a raise releases everything;
  - the sequential step: flag off its parent's calls exactly; on, the [SEQ-PUBLISH] lines within the log
    budget, the stage timer and the split sink restored whatever the step does;
  - the gate: a prewarm line that warmed something required, one [SEQ-PUBLISH] step line per sequential
    step the phase log ended, the prewarm lines reported; flag off, its parent's results exactly;
  - the arm: the refusals, the passthroughs, the host sampler and the io gate (bash, stubbed), and flag off
    its parent's text plus insertions that do nothing; both image copy lists; the CPU allowlist; LF.
"""

from contextlib import ExitStack, contextmanager
import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import torch

import lever_n_m3native_gate as gate
import publish_prewarm
import serving_sequential_step as sequential


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
# This change's parent: flag off, every module it touches must be this commit's, call for call.
PARENT = '6a49230d'
ARM = HERE / 'lever_n_m3native_run_arm.sh'
CPU_WORKFLOW = ROOT / '.github' / 'workflows' / 'qwen-integration-cpu.yml'
POSITION = 131072


def git(*arguments):
    try:
        result = subprocess.run(['git', *arguments], capture_output=True, cwd=str(ROOT), timeout=60)
    except (OSError, subprocess.SubprocessError):
        raise unittest.SkipTest('no git')
    if result.returncode != 0:
        raise unittest.SkipTest('no git history for %s' % PARENT)
    return result.stdout.decode('utf-8')


def parent_text(relative):
    return git('show', '%s:%s' % (PARENT, relative))


def change_commit():
    """The commit that added publish_prewarm.py - this change - or None while it is uncommitted. The text
    and diff proofs compare what THIS change did against PARENT: its own commit once there is one, so a
    later edit of the arm, the copy lists or the gate workflow (the experiment's tags) is not read as this
    change's (test_early_draft pins its range the same way); the working tree before that."""
    added = git('log', '--diff-filter=A', '--format=%H', '--', 'scripts/ci/publish_prewarm.py').split()
    return added[-1] if added else None


def change_text(relative):
    """`relative` as this change left it: at its commit, or in the working tree while uncommitted."""
    commit = change_commit()
    if commit is None:
        return (ROOT / relative).read_text(encoding='utf-8')
    return git('show', '%s:%s' % (commit, relative))


def parent_module(relative, name):
    """scripts/ci/<relative> at PARENT, loaded under `name` beside today's modules."""
    module = ModuleType(name)
    module.__file__ = str(HERE / relative)
    exec(compile(parent_text('scripts/ci/' + relative), '%s@%s' % (relative, PARENT), 'exec'), module.__dict__)
    return module


def clean_environment(**flags):
    """No QWEN_FAST_* flag but the ones given."""
    environment = {name: value for name, value in os.environ.items() if not name.startswith('QWEN_FAST_')}
    environment.update(flags)
    return patch.dict(os.environ, environment, clear=True)


def fresh_warmed():
    return patch.object(publish_prewarm, '_WARMED', set())


# -- warm() on a recording fake ---------------------------------------------------------------------------

class FakeDevice:
    """The DFlashDevice surface warm() touches, with its pending/frontier semantics: prepare refuses a
    second live publication, discard clears it (both the device's and its cache's)."""

    def __init__(self, calls, *, history_rows=2048, position=POSITION, counts=None, leave_pending=False,
                 move=False, fail=None):
        self.calls, self.history_rows, self.position = calls, history_rows, position
        self.pending = None
        self.kv_history = SimpleNamespace(pending=None)
        self.mesh = None if counts is None else SimpleNamespace(num_program_cache_entries=Mock(side_effect=counts))
        self.leave_pending, self.move, self.fail = leave_pending, move, fail

    def prepare_publication(self, features, prefix, **keywords):
        self.calls.append(('prepare', features, prefix, keywords))
        if self.fail is not None and self.fail == (features[0].rows, prefix):
            raise RuntimeError('prepare failed')
        if self.pending is not None:
            raise ValueError('One live target-feature publication at the committed frontier required')
        self.kv_history.pending = SimpleNamespace(status='prepared', prefix=prefix)
        self.pending = SimpleNamespace(status='prepared', prefix=prefix, kv=self.kv_history.pending)
        return self.pending

    def discard_publication(self, publication):
        self.calls.append(('discard', publication))
        if publication is not self.pending:
            raise ValueError('Only the current prepared feature publication may be discarded')
        publication.status = 'discarded'
        if not self.leave_pending:
            self.pending = self.kv_history.pending = None
        if self.move:
            self.position += 1


def taps(rows, label='tap'):
    return tuple(SimpleNamespace(rows=rows, name='%s%d.%d' % (label, rows, index)) for index in range(5))


def fake_engine(widths=(1, 2, 4), **extra):
    buckets = {}
    for rows in widths:
        buckets[rows] = dict(rows=rows, feature_capture=SimpleNamespace(outputs=Mock(return_value=taps(rows))))
    buckets.update(extra)
    return SimpleNamespace(buckets=buckets)


def recorder():
    lines = []
    return lines, lambda template, *values: lines.append(template.format(*values))


SEVEN = ((1, 1), (2, 1), (2, 2), (4, 1), (4, 2), (4, 3), (4, 4))


class WarmTests(unittest.TestCase):
    def setUp(self):
        patcher = fresh_warmed()
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_each_rows_prefix_is_published_once_in_bucket_then_prefix_order(self):
        calls, (lines, log) = [], recorder()
        device, engine = FakeDevice(calls, counts=[100, 107]), fake_engine()
        self.assertEqual(publish_prewarm.warm(device, engine, log=log), SEVEN)
        expected = []
        for rows, prefix in SEVEN:
            features = engine.buckets[rows]['feature_capture'].outputs.return_value
            expected.append(('prepare', features, prefix, dict(position=POSITION)))
            expected.append('discard')
        self.assertEqual([entry if entry[0] == 'prepare' else 'discard' for entry in calls], expected)
        for index in range(0, len(calls), 2):
            # Each prepare's own publication is discarded before the next prepare.
            self.assertEqual(calls[index + 1][1].prefix, calls[index][2])
            self.assertEqual(calls[index + 1][1].status, 'discarded')
        for rows in (1, 2, 4):
            engine.buckets[rows]['feature_capture'].outputs.assert_called_once_with()
        self.assertEqual(publish_prewarm._WARMED, set(SEVEN))
        self.assertEqual(len(lines), 1)
        self.assertRegex(lines[0], r'^\[PINDIAG\] publish prewarm pairs=1:1,2:1-2,4:1-4 count=7 ms=[0-9]+\.[0-9]{2} '
                                   r'program_cache=100->107$')

    def test_a_second_request_warms_nothing_and_reads_no_device(self):
        calls, (lines, log) = [], recorder()
        publish_prewarm.warm(FakeDevice(calls, counts=[1, 2]), fake_engine(), log=log)
        calls.clear()
        second, engine = FakeDevice(calls, counts=[]), fake_engine()
        self.assertEqual(publish_prewarm.warm(second, engine, log=log), ())
        self.assertEqual(calls, [])
        second.mesh.num_program_cache_entries.assert_not_called()
        for bucket in engine.buckets.values():
            bucket['feature_capture'].outputs.assert_not_called()
        self.assertEqual(lines[1], '[PINDIAG] publish prewarm pairs=none count=0 ms=0.00 program_cache=n/a->n/a')

    def test_a_new_width_is_warmed_for_its_prefixes_only(self):
        calls, (lines, log) = [], recorder()
        publish_prewarm.warm(FakeDevice(calls), fake_engine((4,)), log=log)
        calls.clear()
        warmed = publish_prewarm.warm(FakeDevice(calls), fake_engine((1, 2, 4, 8)), log=log)
        self.assertEqual(warmed, ((1, 1), (2, 1), (2, 2)) + tuple((8, prefix) for prefix in range(1, 9)))
        self.assertIn(' pairs=1:1,2:1-2,8:1-8 count=11 ', lines[1])

    def test_nothing_is_pending_and_the_frontier_is_unmoved_afterwards(self):
        calls = []
        device = FakeDevice(calls)
        publish_prewarm.warm(device, fake_engine(), log=recorder()[1])
        self.assertIsNone(device.pending)
        self.assertIsNone(device.kv_history.pending)
        self.assertEqual((device.position, device.history_rows), (POSITION, 2048))

    def test_below_2048_history_rows_nothing_is_warmed(self):
        for rows in (1, 2047, None):
            with self.subTest(history_rows=rows):
                calls, (lines, log) = [], recorder()
                device, engine = FakeDevice(calls, history_rows=rows, counts=[]), fake_engine()
                self.assertEqual(publish_prewarm.warm(device, engine, log=log), ())
                self.assertEqual(calls, [])
                self.assertEqual(publish_prewarm._WARMED, set())
                for bucket in engine.buckets.values():
                    bucket['feature_capture'].outputs.assert_not_called()
                self.assertEqual(lines, ['[PINDIAG] publish prewarm skipped history_rows=%s '
                                         '(the publication shapes settle at 2048)' % rows])

    def test_a_publication_left_pending_or_a_moved_frontier_is_refused(self):
        for keywords in (dict(leave_pending=True), dict(move=True)):
            with self.subTest(**keywords), fresh_warmed():
                with self.assertRaisesRegex(AssertionError, 'leave nothing pending'):
                    publish_prewarm.warm(FakeDevice([], **keywords), fake_engine(), log=recorder()[1])
                self.assertEqual(publish_prewarm._WARMED, set(), 'a pair is recorded only once it passed the check')
        device = FakeDevice([])
        device.pending = object()
        with self.assertRaisesRegex(AssertionError, 'leave nothing pending'):
            publish_prewarm.warm(device, fake_engine(), log=recorder()[1])

    def test_a_failed_prepare_propagates_and_records_only_the_pairs_before_it(self):
        calls = []
        with self.assertRaisesRegex(RuntimeError, 'prepare failed'):
            publish_prewarm.warm(FakeDevice(calls, fail=(4, 3)), fake_engine(), log=recorder()[1])
        self.assertEqual(publish_prewarm._WARMED, {(1, 1), (2, 1), (2, 2), (4, 1), (4, 2)})

    def test_packed_and_featureless_buckets_are_not_published(self):
        calls = []
        engine = fake_engine((4,), packed=dict(rows=16, feature_capture=SimpleNamespace(outputs=Mock()), packed=('b', 0)),
                             bare=dict(rows=2, feature_capture=None))
        self.assertEqual(publish_prewarm.warm(FakeDevice(calls), engine, log=recorder()[1]),
                         tuple((4, prefix) for prefix in range(1, 5)))
        engine.buckets['packed']['feature_capture'].outputs.assert_not_called()

    def test_program_cache_reads_fall_back_to_n_a(self):
        for mesh in (None, SimpleNamespace(), SimpleNamespace(num_program_cache_entries=Mock(side_effect=RuntimeError))):
            with self.subTest(mesh=mesh):
                self.assertEqual(publish_prewarm.program_cache(SimpleNamespace(mesh=mesh)), 'n/a')
        self.assertEqual(publish_prewarm.program_cache(SimpleNamespace(mesh=SimpleNamespace(
            num_program_cache_entries=lambda: 12))), 12)

    def test_describe_pairs(self):
        self.assertEqual(publish_prewarm.describe_pairs(()), 'none')
        self.assertEqual(publish_prewarm.describe_pairs(SEVEN), '1:1,2:1-2,4:1-4')
        self.assertEqual(publish_prewarm.describe_pairs(((4, 2), (4, 4), (16, 1))), '4:2,4:4,16:1')

    def test_the_default_log_is_loguru_or_print(self):
        with patch.dict(sys.modules, {'loguru': None}), patch('builtins.print') as printed:
            publish_prewarm.warm(FakeDevice([], history_rows=7), fake_engine())
        printed.assert_called_once_with('[PINDIAG] publish prewarm skipped history_rows=7 '
                                        '(the publication shapes settle at 2048)', flush=True)


# -- the call is the sequential commit's -------------------------------------------------------------------

class SequentialCallTests(unittest.TestCase):
    """DFlashRequestRuntime.publish - the sequential commit - and warm() make the same drafter call."""

    def engine(self, ticket, session):
        from verifier_engine import VerifierEngine

        engine = SimpleNamespace(phase='verified', pending=ticket, session=session, retain_feature_taps=(5, 19),
                                 buckets=fake_engine().buckets, pending_key=4, publish=Mock())
        engine.verified_features_for_publication = lambda given: VerifierEngine.verified_features_for_publication(
            engine, given)
        return engine

    def test_the_runtime_and_the_prewarm_call_prepare_publication_identically(self):
        from dflash_request_runtime import DFlashRequestRuntime

        drafter = SimpleNamespace(position=POSITION, max_drafts=15, propose=Mock(), prepare_publication=Mock(),
                                  commit_publication=Mock(), discard_publication=Mock())
        drafter.prepare_publication.return_value = publication = SimpleNamespace(status='prepared')

        def commit(given):
            drafter.position += 3
        drafter.commit_publication.side_effect = commit
        runtime = DFlashRequestRuntime(drafter, position=POSITION)
        ticket = SimpleNamespace(position=POSITION, tokens=(1, 2, 3, 4))
        session = SimpleNamespace(pending=ticket, phase='committing')
        engine = self.engine(ticket, session)
        runtime.session, runtime.engine, runtime.phase = session, engine, 'idle'
        runtime.publish(3)
        served = drafter.prepare_publication.call_args
        drafter.commit_publication.assert_called_once_with(publication)

        calls = []
        device = FakeDevice(calls)
        with fresh_warmed():
            publish_prewarm.warm(device, SimpleNamespace(buckets=engine.buckets), log=recorder()[1])
        warmed = [entry for entry in calls if entry[0] == 'prepare' and entry[2] == 3]
        self.assertEqual(len(warmed), 1)
        self.assertEqual(served, call(warmed[0][1], 3, position=POSITION), 'same features, prefix and keywords')
        self.assertIs(served.args[0], warmed[0][1], "the bucket's own taps, the object itself")

    def test_under_round_b1_the_call_takes_the_non_fused_branch(self):
        from dflash_device import DFlashDevice

        features, device = taps(4), SimpleNamespace()
        with clean_environment(QWEN_FAST_ROUND_B1='1'), \
                patch.object(DFlashDevice, '_prepare_publication_round_b1', return_value='pending') as b1:
            self.assertEqual(DFlashDevice.prepare_publication(device, features, 3, position=POSITION), 'pending')
        b1.assert_called_once_with(device, features, 3, position=POSITION, merge_release=False, fused_steady_state=False)

    def test_the_sequential_step_installs_no_publish_option(self):
        """serving_packed_step.commit_entry wraps the PACKED commits in install_publish_options /
        install_fused_commit; the sequential step's module names neither, so its commits call the
        drafter's own prepare_publication, the one warm() calls."""
        text = (HERE / 'serving_sequential_step.py').read_text(encoding='utf-8')
        for name in ('install_publish_options', 'install_fused_commit', 'install_merge_release',
                     'prepare_publication =', 'prepare_publication=', 'PUBLISH_OPTIONS'):
            self.assertNotIn(name, text)


# -- exactness on the real publication code --------------------------------------------------------------

def torch_operations():
    operations = SimpleNamespace(bfloat16=torch.bfloat16, float32=torch.float32, TILE_LAYOUT='tile',
                                 DRAM_MEMORY_CONFIG='dram')
    operations.MatmulMultiCoreReuseMultiCast1DProgramConfig = lambda **keywords: 'program'
    operations.from_torch = lambda value, **keywords: value.clone()
    operations.ReplicateTensorToMesh = lambda mesh: mesh
    operations.slice = lambda value, start, end: value[tuple(slice(first, last)
                                                             for first, last in zip(start, end, strict=True))]
    operations.pad = lambda value, padding, fill: torch.nn.functional.pad(
        value, tuple(item for pair in reversed(padding) for item in pair), value=fill)
    operations.concat = lambda values, dim, **keywords: torch.cat(values, dim=dim)
    operations.zeros_like = torch.zeros_like
    operations.copy = lambda source, destination: destination.copy_(source)
    # Cheap stand-ins, deterministic in their inputs: what is held is where the rows go, not the math.
    operations.matmul = lambda value, weight, **keywords: value[..., :5120].float() * weight
    operations.typecast = lambda value, dtype: value.to(dtype)
    operations.rms_norm = lambda value, epsilon, weight, **keywords: value * weight
    operations.synchronize_device = Mock()
    operations.deallocate = Mock()
    return operations


def storage_address(operations, value):
    pointer = value.untyped_storage().data_ptr()
    return pointer, pointer + 1


def project_key_value(operations, inputs, query, tables, retain, *, parameters):
    from draft_head_preparation import rope_reference

    rows = inputs.shape[2]
    value = (inputs[..., :512] * (parameters['layer'] + 1)).reshape(1, rows, 4, 128).transpose(1, 2).contiguous()
    return dict(k=retain(rope_reference(value, *tables)), v=retain(value))


@contextmanager
def real_publication_code():
    with ExitStack() as stack:
        for target, replacement in (
                ('dflash_device.addresses', storage_address), ('draft_kv_history.addresses', storage_address),
                ('dflash_device.release_owned', lambda operations, owned: None),
                ('draft_kv_history.release_owned', lambda operations, owned: None),
                ('dflash_device.concatenate_local_features', lambda operations, parts: torch.cat(parts, dim=3)),
                ('dflash_device.gather_add_projection',
                 lambda operations, mesh, collectives, partial, **keywords: partial * 2),
                ('draft_kv_history.project_key_value', project_key_value)):
            stack.enter_context(patch(target, side_effect=replacement))
        yield


def feature_taps(rows, seed):
    generator = torch.Generator().manual_seed(seed)
    return tuple(torch.randn((1, 1, rows, 2560), generator=generator).bfloat16() for _ in range(5))


def build_device(operations, seed):
    """A DFlashDevice at the 131k steady state - history_rows 2048 - with pool-lent K/V banks, built from
    the same seed twice gives two devices identical bit for bit."""
    from dflash_device import DFlashDevice
    from draft_kv_history import KV_SHAPE, DraftKVHistory

    generator = torch.Generator().manual_seed(seed)
    history = torch.randn((1, 1, 2048, 5120), generator=generator).bfloat16()
    storage = [{side: {head: torch.zeros(KV_SHAPE, dtype=torch.bfloat16) for head in ('k', 'v')}
                for side in ('active', 'spare')} for layer in range(2)]
    device = object.__new__(DFlashDevice)
    device.__dict__.update(operations=operations, mesh='mesh', collectives='collectives', kernel='kernel',
                           projection=torch.tensor(0.5), feature_norm=torch.tensor(1.0, dtype=torch.bfloat16),
                           closed=False, pending=None, position=4096, history_rows=2048, history=history,
                           spare_history=torch.zeros_like(history), owned=[], borrowed=[], progress=None,
                           proposal_capture=None, published_rows=0)
    device.kv_history = DraftKVHistory(operations, 'mesh', [dict(layer=layer) for layer in range(2)], history,
                                       position=4096, history_rows=2048, storage=storage)
    return device


def bits(value):
    return value.contiguous().view(torch.int16).clone()


def state(device):
    """Everything a publication can write, as bit patterns, and the frontiers."""
    banks = {'%s%d%s' % (side, layer, head): bits(bank[head])
             for side, pairs in (('active', device.kv_history.active), ('spare', device.kv_history.spare))
             for layer, bank in enumerate(pairs) for head in ('k', 'v')}
    return dict(history=bits(device.history), spare_history=bits(device.spare_history), banks=banks,
                frontier=(device.position, device.history_rows, device.published_rows, device.kv_history.position,
                          device.kv_history.history_rows))


class ExactnessTests(unittest.TestCase):
    """The real DFlashDevice.prepare_publication (flag-off branch and B1's non-fused branch) and the real
    DraftKVHistory.prepare, on torch. A prewarm with garbage features writes the two spares and nothing
    else; after the next real publication every tensor and frontier equals a device that never warmed."""

    def assert_same(self, left, right, label):
        self.assertEqual(left['frontier'], right['frontier'], label)
        for name in ('history', 'spare_history'):
            self.assertTrue(torch.equal(left[name], right[name]), '%s: %s' % (label, name))
        self.assertEqual(left['banks'].keys(), right['banks'].keys())
        for name in left['banks']:
            self.assertTrue(torch.equal(left['banks'][name], right['banks'][name]), '%s: bank %s' % (label, name))

    def test_a_prewarm_leaves_no_state_behind(self):
        for b1 in ('0', '1'):
            with self.subTest(round_b1=b1), clean_environment(QWEN_FAST_ROUND_B1=b1), real_publication_code(), \
                    fresh_warmed():
                operations = torch_operations()
                warmed, plain = build_device(operations, 3), build_device(operations, 3)
                before = state(warmed)
                self.assert_same(before, state(plain), 'built alike')
                engine = SimpleNamespace(buckets={rows: dict(rows=rows, feature_capture=SimpleNamespace(
                    outputs=lambda rows=rows: feature_taps(rows, 90 + rows))) for rows in (1, 2, 4)})
                self.assertEqual(publish_prewarm.warm(warmed, engine, log=recorder()[1]), SEVEN)
                after = state(warmed)
                # Only the spares moved: the committed history, the active banks and every frontier did not.
                self.assertEqual(after['frontier'], before['frontier'])
                self.assertTrue(torch.equal(after['history'], before['history']))
                for name in before['banks']:
                    moved = not torch.equal(after['banks'][name], before['banks'][name])
                    self.assertEqual(moved, name.startswith('spare'), name)
                self.assertFalse(torch.equal(after['spare_history'], before['spare_history']), 'the prewarm wrote the spare')
                self.assertIsNone(warmed.pending)
                self.assertIsNone(warmed.kv_history.pending)
                # The next real publications overwrite both spares in full before each swap.
                for number, (prefix, seed) in enumerate(((4, 5), (2, 6), (3, 7), (1, 8))):
                    features = feature_taps(4, seed)
                    for device in (warmed, plain):
                        publication = device.prepare_publication(features, prefix, position=device.position)
                        device.commit_publication(publication)
                    self.assert_same(state(warmed), state(plain), 'after real publication %d' % number)
                self.assertEqual(plain.position, 4096 + 10)
                self.assertFalse(torch.equal(state(plain)['history'], before['history']),
                                 'the publications moved the history')


# -- from_prefill ---------------------------------------------------------------------------------------------

def factory_fixture():
    from test_serving_request_factory import RequestFactoryTests

    return RequestFactoryTests('test_prefilled_seed_not_emitted_or_prefilled_twice').fixture()


def build_request(module, components, arguments):
    with patch.object(module, 'device_components', return_value=components):
        return module.from_prefill(object(), SimpleNamespace(args=SimpleNamespace(vocab_size=100), mesh_device=object()),
                                   object(), torch.tensor([list(range(65)) + [0] * 3], dtype=torch.int32),
                                   [object()] * 48, **arguments)


def factory_calls(module):
    """Every call from_prefill made on its components, the device, the capture and the engine, with
    callables by name - comparable between two modules."""
    components, device, engines, arguments = factory_fixture()
    request = build_request(module, components, arguments)

    def plain(value):
        if isinstance(value, (bool, int, float, str, type(None))):
            return value
        if isinstance(value, (tuple, list)):
            return type(value)(plain(item) for item in value)
        if callable(value):
            return getattr(value, '__name__', type(value).__name__)
        return type(value).__name__

    def calls(mock):
        return [(plain(tuple(entry.args)), {key: plain(value) for key, value in entry.kwargs.items()})
                for entry in mock.call_args_list]

    record = dict(
        device=calls(components.device), engine=calls(components.engine), proposal=calls(components.proposal),
        collectives=calls(components.collectives),
        drafter={name: calls(getattr(device, name)) for name in
                 ('propose', 'prepare_publication', 'commit_publication', 'discard_publication', 'close')},
        capture=(arguments['capture'].outputs.mock_calls, arguments['capture'].close.mock_calls),
        engine_close=[engine.close.mock_calls for engine in engines],
        request=(type(request).__name__, request.session.position, request.runtime.phase, request.collect_timings,
                 request.runtime.engine is engines[0]))
    request.close('request')
    return record


class FactoryTests(unittest.TestCase):
    def test_flag_off_from_prefill_is_the_parents_call_for_call_and_imports_nothing(self):
        import serving_request_factory

        parent = parent_module('serving_request_factory.py', 'serving_request_factory_prewarm_parent')
        for environ in ({}, {publish_prewarm.FLAG: '0'}, {publish_prewarm.FLAG: 'yes'}, {publish_prewarm.FLAG: ''}):
            with self.subTest(environ=environ), clean_environment(**environ), \
                    patch.dict(sys.modules, {'publish_prewarm': None}):
                self.assertEqual(factory_calls(serving_request_factory), factory_calls(parent))

    def test_flag_on_warms_once_after_the_engine_before_the_request_binds_it(self):
        import serving_request_factory

        components, device, engines, arguments = factory_fixture()
        seen, events = [], []
        bound = serving_request_factory.FastRequest

        def warm(given_device, given_engine):
            events.append('warm')
            seen.append((given_device, given_engine, given_engine.close.call_count,
                         components.engine.call_count, device.close.call_count))

        def fast_request(*args, **keywords):
            events.append('request')
            return bound(*args, **keywords)
        with clean_environment(**{publish_prewarm.FLAG: '1'}), patch('publish_prewarm.warm', side_effect=warm) as warmed, \
                patch.object(serving_request_factory, 'FastRequest', side_effect=fast_request):
            request = build_request(serving_request_factory, components, arguments)
        warmed.assert_called_once()
        self.assertEqual(events, ['warm', 'request'])
        self.assertEqual(seen, [(device, engines[0], 0, 1, 0)])
        self.assertIs(request.runtime.engine, engines[0])
        self.assertEqual(request.runtime.phase, 'idle', 'bound after the warm')
        request.close('request')

    def test_a_failed_warm_releases_the_engine_the_drafter_and_the_capture(self):
        import serving_request_factory

        components, device, engines, arguments = factory_fixture()
        with clean_environment(**{publish_prewarm.FLAG: '1'}), \
                patch('publish_prewarm.warm', side_effect=RuntimeError('warm failed')):
            with self.assertRaisesRegex(RuntimeError, 'warm failed'):
                build_request(serving_request_factory, components, arguments)
        engines[0].close.assert_called_once()
        device.close.assert_called_once()
        arguments['capture'].close.assert_called_once()

    def test_the_factory_names_the_modules_flag(self):
        text = (HERE / 'serving_request_factory.py').read_text(encoding='utf-8')
        self.assertIn("os.environ.get('%s') == '1'" % publish_prewarm.FLAG, text)
        self.assertEqual((publish_prewarm.FLAG, publish_prewarm.MARKER, publish_prewarm.HISTORY_ROWS),
                         ('QWEN_FAST_PUBLISH_PREWARM', '[PINDIAG] publish prewarm', 2048))


# -- the sequential step's log -------------------------------------------------------------------------------

class SequentialRequest:
    """FastRequest.step's publication, faked: publish() through the runtime's publication_stage seam and
    the B1 split sink, the frontier and the program cache moved."""

    def __init__(self, request_id, prefix, *, counts=(40, 43), fail=False):
        self.calls, self.prefix, self.fail = [], prefix, fail
        self.runtime = SimpleNamespace(position=POSITION, drafter=SimpleNamespace(
            mesh=SimpleNamespace(num_program_cache_entries=Mock(side_effect=list(counts)))))
        self.seen = {}

    def step(self, request_id, *, cancelled):
        from dflash_traced_publish import PUBLICATION_SPLITS, add_split

        self.calls.append((request_id, cancelled()))
        self.seen['timer'] = 'publication_stage' in self.runtime.__dict__
        self.seen['splits'] = PUBLICATION_SPLITS.get()
        if self.fail:
            raise RuntimeError('step failed')
        stage = getattr(self.runtime, 'publication_stage', None)
        for name in ('features', 'prepare_history', 'publish_target', 'commit_history'):
            if stage is not None:
                with stage(name, self.prefix):
                    pass
        splits = PUBLICATION_SPLITS.get()
        if splits is not None:
            for name, seconds in (('proj', 1.5), ('hist', 0.25), ('kv', 0.5), ('sync', 0.125), ('rel', 0.0625)):
                add_split(splits, name, seconds)
        self.runtime.position += self.prefix
        return SimpleNamespace(request_id=request_id, token_ids=[1] * self.prefix)


def sequential_entry(request_id, request, rows=4):
    return dict(request_id=request_id, request=request,
                ticket=SimpleNamespace(request_id=request_id, tokens=tuple(range(rows))))


class SeqPublishLogTests(unittest.TestCase):
    def logged(self, entries, *, b1='0'):
        lines = []
        with patch.object(sequential, 'SEQ_PUBLISH_LOG', True), clean_environment(QWEN_FAST_ROUND_B1=b1), \
                patch('serving_packed_step.audit_log', side_effect=lambda message, **values: lines.append(
                    message.format(**values))):
            outputs = sequential.sequential_packed_step(entries, cancelled=lambda: False)
        return outputs, lines

    def test_flag_off_the_step_is_the_parents_call_for_call(self):
        parent = parent_module('serving_sequential_step.py', 'serving_sequential_step_prewarm_parent')
        self.assertFalse(sequential.SEQ_PUBLISH_LOG)

        def run(module):
            requests = [Mock(), Mock()]
            for name, request in zip('BA', requests):
                request.step.return_value = SimpleNamespace(request_id=name, token_ids=[7])
            entries = [dict(request_id=name, request=request, ticket=SimpleNamespace(request_id=name, tokens=(1, 2)))
                       for name, request in zip('BA', requests)]

            def cancelled():
                return False
            with patch('serving_packed_step.audit_log') as logged:
                outputs = module.sequential_packed_step(entries, cancelled=cancelled)
            # Every call on each request, the cancellation callback by identity.
            calls = [[(entry[0], entry[1], {key: 'cancelled' if value is cancelled else value
                                            for key, value in entry[2].items()}) for entry in request.mock_calls]
                     for request in requests]
            return calls, outputs, logged.mock_calls

        mine, theirs = run(sequential), run(parent)
        self.assertEqual(mine[0], [[('step', ('B',), dict(cancelled='cancelled'))],
                                   [('step', ('A',), dict(cancelled='cancelled'))]])
        self.assertEqual(mine[0], theirs[0])
        self.assertEqual(mine[1], theirs[1])
        self.assertEqual((mine[2], theirs[2]), ([], []))

    def test_flag_on_three_lines_per_step_under_b1(self):
        request = SequentialRequest('A', 3)
        outputs, lines = self.logged([sequential_entry('A', request)], b1='1')
        self.assertEqual([output.request_id for output in outputs], ['A'])
        self.assertEqual(request.calls, [('A', False)])
        self.assertTrue(request.seen['timer'])
        self.assertEqual(request.seen['splits'], {'proj': 1500.0, 'hist': 250.0, 'kv': 500.0, 'sync': 125.0, 'rel': 62.5})
        self.assertEqual(len(lines), 3)
        self.assertRegex(lines[0], r'^\[SEQ-PUBLISH\] request=A rows=4 prefix=3 step_ms=[0-9]+\.[0-9]{2} cache=40->43$')
        self.assertRegex(lines[1], r'^\[SEQ-PUBLISH\] request=A stages features=[0-9.]+ prepare_history=[0-9.]+ '
                                   r'publish_target=[0-9.]+ commit_history=[0-9.]+$')
        self.assertEqual(lines[2], '[SEQ-PUBLISH] request=A splits proj=1500.00 hist=250.00 kv=500.00 sync=125.00 rel=62.50')
        self.assertEqual([bool(gate.SEQ_PUBLISH_STEP.search(line)) for line in lines], [True, False, False],
                         'the gate counts one step line per step')
        self.assertNotIn('publication_stage', request.runtime.__dict__, 'the stage timer is restored')
        from dflash_traced_publish import PUBLICATION_SPLITS
        self.assertIsNone(PUBLICATION_SPLITS.get(), 'the split sink is reset')

    def test_without_b1_no_split_sink_and_two_lines(self):
        request = SequentialRequest('A', 0)
        outputs, lines = self.logged([sequential_entry('A', request, rows=2)])
        self.assertIsNone(request.seen['splits'])
        self.assertEqual(len(lines), 2)
        self.assertIn(' rows=2 prefix=0 ', lines[0])

    def test_every_step_in_order_and_a_raising_step_restores_and_logs_nothing(self):
        first, second = SequentialRequest('B', 4), SequentialRequest('A', 1)
        outputs, lines = self.logged([sequential_entry('B', first), sequential_entry('A', second)])
        self.assertEqual([line.split()[1] for line in lines], ['request=B'] * 2 + ['request=A'] * 2)
        broken = SequentialRequest('A', 2, fail=True)
        with self.assertRaisesRegex(RuntimeError, 'step failed'):
            self.logged([sequential_entry('A', broken)], b1='1')
        self.assertNotIn('publication_stage', broken.runtime.__dict__)
        from dflash_traced_publish import PUBLICATION_SPLITS
        self.assertIsNone(PUBLICATION_SPLITS.get())

    def test_a_refused_stage_timer_installs_no_split_sink_and_steps_nothing(self):
        request = SequentialRequest('A', 2)
        request.runtime.publication_stage = lambda name, prefix: None
        with self.assertRaisesRegex(ValueError, 'already overridden'):
            self.logged([sequential_entry('A', request)], b1='1')
        self.assertEqual(request.calls, [])
        from dflash_traced_publish import PUBLICATION_SPLITS
        self.assertIsNone(PUBLICATION_SPLITS.get(), 'nothing left installed')

    def test_a_request_without_a_runtime_still_steps_and_logs(self):
        request = Mock(spec=['step'])
        request.step.return_value = SimpleNamespace(request_id='A', token_ids=[1])
        outputs, lines = self.logged([dict(request_id='A', request=request, ticket=SimpleNamespace(request_id='A'))])
        request.step.assert_called_once()
        self.assertIn('rows=n/a prefix=n/a', lines[0])
        self.assertIn('cache=n/a->n/a', lines[0])

    def test_every_line_fits_the_log_budget(self):
        from dflash_device import AUDIT_LINE_BUDGET

        request_id = 'x' * 64
        worst = [sequential.SEQ_PUBLISH_LINE.format(request=request_id[:48], rows=16, prefix=16, step_ms=99999.99,
                                                    before=999999, after=999999),
                 sequential.SEQ_PUBLISH_STAGES_LINE.format(request=request_id[:48], stages=' '.join(
                     '%s=%.2f' % (name, 99999.99) for name in ('features', 'prepare_history', 'publish_target',
                                                                'commit_history'))),
                 sequential.SEQ_PUBLISH_SPLIT_LINE.format(request=request_id[:48], splits=' '.join(
                     '%s=%.2f' % (name, 99999.99) for name in sequential.SEQ_PUBLISH_SPLITS))]
        for line in worst:
            with self.subTest(line=line):
                self.assertLessEqual(len(line), AUDIT_LINE_BUDGET)
                self.assertTrue(line.startswith(gate.SEQ_PUBLISH_MARKER))

    def test_the_flag_is_read_once_at_import_and_refuses_other_values(self):
        def load(**environ):
            spec = importlib.util.spec_from_file_location('seq_prewarm_copy', HERE / 'serving_sequential_step.py')
            module = importlib.util.module_from_spec(spec)
            with clean_environment(**environ):
                spec.loader.exec_module(module)
            return module
        self.assertFalse(load().SEQ_PUBLISH_LOG)
        self.assertFalse(load(QWEN_FAST_SEQ_PUBLISH_LOG='0').SEQ_PUBLISH_LOG)
        self.assertTrue(load(QWEN_FAST_SEQ_PUBLISH_LOG='1').SEQ_PUBLISH_LOG)
        with self.assertRaisesRegex(ValueError, 'QWEN_FAST_SEQ_PUBLISH_LOG must be unset, 0 or 1'):
            load(QWEN_FAST_SEQ_PUBLISH_LOG='yes')
        self.assertEqual(sequential.SEQ_PUBLISH_LOG_FLAG, gate.SEQ_PUBLISH_LOG_FLAG)


# -- the gate ------------------------------------------------------------------------------------------------

PREWARM_FIRST = '[PINDIAG] publish prewarm pairs=1:1,2:1-2,4:1-4 count=7 ms=84.12 program_cache=812->871'
PREWARM_NONE = '[PINDIAG] publish prewarm pairs=none count=0 ms=0.00 program_cache=n/a->n/a'
SEQ_LINE = '[SEQ-PUBLISH] request=cmpl-1 rows=4 prefix=4 step_ms=120.50 cache=871->871'
# The v228 arm's QWEN_FAST_* flags (the B arm the verdict read).
V228 = dict(QWEN_FAST_ROUND_B1='1', QWEN_FAST_VERIFY_T1='1', QWEN_FAST_GDN_USER_BATCH='1', QWEN_FAST_GDN_SEQ_BLOCK='1',
            QWEN_FAST_MEMORY_LEDGER='1', QWEN_FAST_SKIP_BLOCK_STREAM='1', QWEN_FAST_SINGLE_GATEUP='1',
            QWEN_FAST_DRAFT_BF8='1', QWEN_FAST_SDPA_MODES='tail,share,slice', QWEN_FAST_VERIFY_T2='1',
            QWEN_FAST_PAIR_MASK_REFRESH='1', QWEN_FAST_PADDED_BLOCK='1', QWEN_FAST_C1_EXACT='1', QWEN_FAST_PRESTAGE='1',
            QWEN_FAST_ROUND_FENCES='1', QWEN_FAST_PIPELINED_PUBLISH='1', QWEN_FAST_FUSED_COMMIT='1',
            QWEN_FAST_FUSED_COMMIT_INPLACE='1', QWEN_FAST_FUSED_COMMIT_LIVE_BANKS='1', QWEN_FAST_EARLY_DRAFT='1',
            QWEN_FAST_GDN_AFTER_PAIRS='1', QWEN_FAST_PAIR_ROW_EXACT='1', QWEN_FAST_GDN_PREFILL_CONV='1',
            QWEN_FAST_SDPA_PF='1', QWEN_FAST_PACKED_AUDIT='1')


def log_of(*lines):
    return '\n'.join('2026-09-25 | INFO | ' + line for line in lines)


class GateTests(unittest.TestCase):
    def test_the_gate_parses_the_modules_own_line(self):
        lines, log = recorder()
        with fresh_warmed():
            publish_prewarm.warm(FakeDevice([], counts=[5, 9]), fake_engine(), log=log)
        match = gate.PUBLISH_PREWARM_LINE.search(log_of(lines[0]))
        self.assertEqual(match.group(1, 2, 4, 5), ('1:1,2:1-2,4:1-4', '7', '5', '9'))
        self.assertTrue(lines[0].startswith(gate.PUBLISH_PREWARM_MARKER))
        self.assertTrue(gate.PUBLISH_PREWARM_MARKER.startswith(publish_prewarm.MARKER))
        self.assertTrue(gate.PUBLISH_PREWARM_SKIPPED.startswith(publish_prewarm.MARKER))
        self.assertEqual(gate.PUBLISH_PREWARM_FLAG, publish_prewarm.FLAG)

    def test_the_flag_requires_a_line_that_warmed_something(self):
        on = {publish_prewarm.FLAG: '1'}
        self.assertEqual(gate.required_flag_markers(on, 4), {publish_prewarm.FLAG: [gate.PUBLISH_PREWARM_MARKER]})
        self.assertEqual(gate.required_flag_markers(on, 1), {publish_prewarm.FLAG: [gate.PUBLISH_PREWARM_MARKER]})
        report = gate.flag_marker_report(on, 4, log_of(PREWARM_FIRST, PREWARM_NONE, PREWARM_NONE, PREWARM_NONE))
        self.assertEqual(report['missing'], [])
        self.assertTrue(report['found'][publish_prewarm.FLAG][gate.PUBLISH_PREWARM_MARKER])
        self.assertEqual(report['publish_prewarm']['prewarm'][0],
                         dict(pairs='1:1,2:1-2,4:1-4', count=7, ms=84.12, program_cache_before='812',
                              program_cache_after='871'))
        self.assertEqual([line['count'] for line in report['publish_prewarm']['prewarm']], [7, 0, 0, 0])
        self.assertEqual(gate.flag_marker_report(on, 4, '')['missing'],
                         ['%s: %s' % (publish_prewarm.FLAG, gate.PUBLISH_PREWARM_MARKER)])
        skipped = gate.flag_marker_report(on, 4, log_of('[PINDIAG] publish prewarm skipped history_rows=512 (...)'))
        self.assertEqual(skipped['missing'], ['%s: %s' % (publish_prewarm.FLAG, gate.PUBLISH_PREWARM_MARKER)])
        self.assertEqual(skipped['publish_prewarm']['skipped'], 1)
        nothing = gate.flag_marker_report(on, 4, log_of(PREWARM_NONE))['missing']
        self.assertEqual(len(nothing), 1)
        self.assertIn('a prewarm that warmed something', nothing[0])
        # A marker the line pattern cannot read is not a pass: a changed format must fail, not go unread.
        garbled = gate.flag_marker_report(on, 4, log_of('[PINDIAG] publish prewarm pairs=1:1 count=seven'))
        self.assertTrue(garbled['found'][publish_prewarm.FLAG][gate.PUBLISH_PREWARM_MARKER])
        self.assertEqual(len(garbled['missing']), 1)
        self.assertIn('a prewarm that warmed something (0 parsed line(s)', garbled['missing'][0])
        self.assertIn('0 or 1', gate.flag_marker_report({publish_prewarm.FLAG: 'yes'}, 4, '')['missing'][0])

    def test_every_sequential_step_the_phase_log_ended_needs_its_line(self):
        on = {sequential.SEQ_PUBLISH_LOG_FLAG: '1'}
        ended = '[PHASE] step cmpl-1 end 120.5 ms'
        stages = ('[SEQ-PUBLISH] request=cmpl-1 stages features=0.10 prepare_history=2.00 publish_target=9.00 '
                  'commit_history=0.01')
        splits = '[SEQ-PUBLISH] request=cmpl-1 splits proj=1.00 hist=0.50 kv=0.40 sync=0.10 rel=0.01'
        self.assertEqual(gate.required_flag_markers(on, 4), {})
        step = ['[PHASE] step cmpl-1 begin', SEQ_LINE, stages, splits, ended]
        report = gate.flag_marker_report(on, 4, log_of(*(step * 2)))
        self.assertEqual(report['missing'], [])
        counts = report['publish_prewarm']
        self.assertEqual((counts['seq_publish_steps'], counts['phase_step_ends']), (2, 2))
        missing = gate.flag_marker_report(on, 4, log_of(ended, ended, SEQ_LINE, stages))['missing']
        self.assertEqual(missing, [sequential.SEQ_PUBLISH_LOG_FLAG + ': a [SEQ-PUBLISH] line for every sequential step '
                                   '(2 ended, 1 logged)'])
        self.assertEqual(len(gate.flag_marker_report(on, 1, log_of(ended))['missing']), 1, 'at any user count')
        # Without the phase log (or with no sequential step) there is nothing to hold them to.
        self.assertEqual(gate.flag_marker_report(on, 4, '')['missing'], [])
        self.assertEqual(gate.flag_marker_report(on, 4, log_of(SEQ_LINE))['missing'], [])
        # The flag off, the same log asks nothing.
        self.assertEqual(gate.flag_marker_report({}, 4, log_of(ended))['missing'], [])
        self.assertEqual(gate.select_diagnostic(['x ' + SEQ_LINE, 'chatter']), ['x ' + SEQ_LINE])

    def test_flag_off_the_gate_is_the_parents(self):
        parent = parent_module('lever_n_m3native_gate.py', 'lever_n_m3native_gate_prewarm_parent')
        logs = ('', log_of(PREWARM_FIRST, PREWARM_NONE, SEQ_LINE, '[PINDIAG] round b1 engaged site=publication',
                           '[PINDIAG] publish prewarm skipped history_rows=5'))
        environs = ({}, {publish_prewarm.FLAG: '0', sequential.SEQ_PUBLISH_LOG_FLAG: '0'}, V228,
                    dict(V228, **{publish_prewarm.FLAG: '', sequential.SEQ_PUBLISH_LOG_FLAG: '0'}))
        for environ in environs:
            for users in (1, 4):
                for log in logs:
                    with self.subTest(environ=sorted(environ), users=users, log=bool(log)):
                        self.assertEqual(gate.required_flag_markers(environ, users, 131072),
                                         parent.required_flag_markers(environ, users, 131072))
                        self.assertEqual(gate.flag_marker_report(environ, users, log, 131072),
                                         parent.flag_marker_report(environ, users, log, 131072))
        lines = ['a [PINDIAG] x', 'b [PACKED] y', 'c chatter', 'd ERROR z', 'e [PHASE] w', 'f [GDN-SEQ-BLOCK-AUDIT] v',
                 PREWARM_FIRST] * 3
        self.assertEqual(gate.select_diagnostic(lines), parent.select_diagnostic(lines))
        self.assertEqual(gate.select_diagnostic(lines, cap=4), parent.select_diagnostic(lines, cap=4))
        with_seq = lines + [SEQ_LINE]
        self.assertEqual(gate.select_diagnostic(with_seq), parent.select_diagnostic(with_seq) + [SEQ_LINE])


# -- the arm --------------------------------------------------------------------------------------------------

def arm_text():
    return ARM.read_text(encoding='utf-8')


def between(text, start, end, *, include_end=True):
    first = text.index(start)
    last = text.index(end, first)
    return text[first:last + (len(end) if include_end else 0)]


def validation_block(text):
    return between(text, '# Publish prewarm (publish_prewarm.py; default off).',
                   '  *) echo "M3NATIVE_HOST_SAMPLER must be 1 or unset, got \'$M3NATIVE_HOST_SAMPLER\'" >&2; '
                   'exit 1 ;;\nesac\n')


def start_block(text):
    return between(text, '# Host pressure, validated above:', '# The tt-metal watcher', include_end=False)


def stop_block(text):
    return between(text, 'if [ -n "$host_sampler_pid" ]; then\n', '\nfi\n')


PASSTHROUGH = ('${M3NATIVE_PUBLISH_PREWARM:+-e QWEN_FAST_PUBLISH_PREWARM=1}',
               '${M3NATIVE_SEQ_PUBLISH_LOG:+-e QWEN_FAST_SEQ_PUBLISH_LOG=1}')


class Bash:
    def bash(self, script, *, cwd=None, **environ):
        found = shutil.which('bash')
        if found is None:
            self.skipTest('no bash')
        try:
            return subprocess.run([found, '-c', script], capture_output=True, text=True, timeout=120, cwd=cwd,
                                  env=dict(PATH=environ.pop('PATH', os.environ.get('PATH', '')), **environ))
        except OSError as error:
            self.skipTest('bash unusable: %s' % error)


class ArmTests(Bash, unittest.TestCase):
    def validate(self, **environ):
        script = 'set -euo pipefail' + chr(10) + validation_block(arm_text()) + 'echo VALID' + chr(10)
        return self.bash(script, **environ)

    def test_the_arm_refuses_what_it_cannot_pass(self):
        for environ in ({}, dict(M3NATIVE_PUBLISH_PREWARM='1'), dict(M3NATIVE_SEQ_PUBLISH_LOG='1'),
                        dict(M3NATIVE_IO_GATE='1'), dict(M3NATIVE_IO_GATE='required'), dict(M3NATIVE_HOST_SAMPLER='1'),
                        dict(M3NATIVE_PUBLISH_PREWARM='1', M3NATIVE_SEQ_PUBLISH_LOG='1', M3NATIVE_IO_GATE='1',
                             M3NATIVE_HOST_SAMPLER='1')):
            with self.subTest(accepted=environ):
                result = self.validate(**environ)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(result.stdout.endswith('VALID\n'))
        self.assertEqual(self.validate().stdout, 'VALID\n', 'unset, nothing is said')
        self.assertIn('publish prewarm 1 sequential publish log 0', self.validate(M3NATIVE_PUBLISH_PREWARM='1').stdout)
        for environ, message in ((dict(M3NATIVE_PUBLISH_PREWARM='0'), 'M3NATIVE_PUBLISH_PREWARM must be 1 or unset'),
                                 (dict(M3NATIVE_PUBLISH_PREWARM='yes'), 'M3NATIVE_PUBLISH_PREWARM must be 1 or unset'),
                                 (dict(M3NATIVE_SEQ_PUBLISH_LOG='2'), 'M3NATIVE_SEQ_PUBLISH_LOG must be 1 or unset'),
                                 (dict(M3NATIVE_IO_GATE='0'), 'M3NATIVE_IO_GATE must be 1, required or unset'),
                                 (dict(M3NATIVE_IO_GATE='strict'), 'M3NATIVE_IO_GATE must be 1, required or unset'),
                                 (dict(M3NATIVE_HOST_SAMPLER='yes'), 'M3NATIVE_HOST_SAMPLER must be 1 or unset')):
            with self.subTest(refused=environ):
                result = self.validate(**environ)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertNotIn('VALID', result.stdout)

    def test_the_two_switches_cross_after_the_k5a_lines_before_the_entrypoint(self):
        text = arm_text()
        lines = text.split(chr(10))
        start = next(number for number, line in enumerate(lines) if PASSTHROUGH[0] in line)
        self.assertEqual(lines[start - 1].strip(), '${M3NATIVE_GDN_SEQ_BLOCK_AUDIT:+-e QWEN_FAST_GDN_SEQ_BLOCK_AUDIT='
                                                   '$M3NATIVE_GDN_SEQ_BLOCK_AUDIT} ' + chr(92))
        for offset, expected in enumerate(PASSTHROUGH):
            with self.subTest(line=expected):
                self.assertEqual(text.count(expected), 1)
                self.assertEqual(lines[start + offset].strip(), expected + ' ' + chr(92), 'nothing else on the line')
                self.assertLess(text.index(expected), text.index('--entrypoint python3'))
        self.assertLess(text.index(validation_block(text)), text.index('docker run --rm --name "$name"'))

    def test_unset_nothing_crosses_and_set_each_crosses_as_1(self):
        script = 'printf "%s|" ' + ' '.join(PASSTHROUGH) + chr(10)
        self.assertEqual(self.bash(script).stdout.strip('|'), '')
        both = self.bash(script, M3NATIVE_PUBLISH_PREWARM='1', M3NATIVE_SEQ_PUBLISH_LOG='1')
        self.assertEqual(both.stdout, '-e|QWEN_FAST_PUBLISH_PREWARM=1|-e|QWEN_FAST_SEQ_PUBLISH_LOG=1|')

    def test_the_container_side_reads_the_names_that_cross(self):
        through = re.findall(r'-e (QWEN_FAST_(?:PUBLISH_PREWARM|SEQ_PUBLISH_LOG))=', arm_text())
        self.assertEqual(sorted(through), [publish_prewarm.FLAG, sequential.SEQ_PUBLISH_LOG_FLAG])
        factory = (HERE / 'serving_request_factory.py').read_text(encoding='utf-8')
        self.assertIn("'%s'" % publish_prewarm.FLAG, factory)
        self.assertEqual(sequential.SEQ_PUBLISH_LOG_FLAG, 'QWEN_FAST_SEQ_PUBLISH_LOG')

    def test_flag_off_the_arm_is_its_parents_text_plus_insertions_that_do_nothing(self):
        # The text this change left (change_text), so a later arm edit is not read as this change's.
        text = change_text('scripts/ci/lever_n_m3native_run_arm.sh')
        parent = parent_text('scripts/ci/lever_n_m3native_run_arm.sh')
        passthrough = ''.join('  %s %s\n' % (line, chr(92)) for line in PASSTHROUGH)
        stripped = text
        for block in (validation_block(text), start_block(text), passthrough, stop_block(text)):
            self.assertEqual(stripped.count(block), 1)
            stripped = stripped.replace(block, '', 1)
        self.assertEqual(stripped, parent)
        # Unset, the inserted code (as it stands today) says nothing, writes nothing and runs nothing.
        text = arm_text()
        with tempfile.TemporaryDirectory() as directory:
            stub = self.stub(directory)
            os.makedirs(os.path.join(directory, 'experiment-results'))
            script = 'set -euo pipefail\n' + validation_block(text) + start_block(text) + stop_block(text) + 'echo DONE\n'
            result = self.bash(script, cwd=directory, PATH=stub + os.pathsep + os.environ.get('PATH', ''),
                               STUB_LOG=os.path.join(directory, 'stub.log'))
            self.assertEqual((result.returncode, result.stdout, result.stderr), (0, 'DONE\n', ''))
            self.assertEqual(os.listdir(os.path.join(directory, 'experiment-results')), [])
            self.assertFalse(os.path.exists(os.path.join(directory, 'stub.log')))

    @staticmethod
    def stub(directory):
        """A python3 that records its argv and exits with STUB_EXIT: runner_io_gate, never run for real."""
        folder = os.path.join(directory, 'stub')
        os.makedirs(folder)
        path = os.path.join(folder, 'python3')
        with open(path, 'w', encoding='utf-8', newline='\n') as handle:
            handle.write('#!/usr/bin/env bash\nprintf \'%s\\n\' "$*" >> "$STUB_LOG"\nexit "${STUB_EXIT:-0}"\n')
        os.chmod(path, 0o755)
        return folder

    def io_gate(self, directory, **environ):
        stub = self.stub(directory)
        os.makedirs(os.path.join(directory, 'experiment-results'), exist_ok=True)
        script = 'set -euo pipefail\n' + start_block(arm_text()) + 'echo AFTER\n'
        return self.bash(script, cwd=directory, PATH=stub + os.pathsep + os.environ.get('PATH', ''),
                         STUB_LOG=os.path.join(directory, 'stub.log'), **environ)

    def test_the_io_gate_reports_and_fails_the_arm_only_when_required(self):
        for value, code, fails in (('1', '75', False), ('1', '0', False), ('required', '0', False),
                                   ('required', '75', True)):
            with self.subTest(gate=value, exit=code), tempfile.TemporaryDirectory() as directory:
                result = self.io_gate(directory, M3NATIVE_IO_GATE=value, STUB_EXIT=code)
                with open(os.path.join(directory, 'stub.log'), encoding='utf-8') as handle:
                    self.assertEqual(handle.read(), '-B scripts/ci/runner_io_gate.py --attempts 4 '
                                                    '--output experiment-results/runner-io-admission.json\n')
                self.assertIn('runner io gate: exit %s ' % code, result.stdout)
                if fails:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('M3NATIVE_IO_GATE=required: the host was not quiet', result.stderr)
                    self.assertNotIn('AFTER', result.stdout)
                else:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('AFTER', result.stdout)

    def test_an_existing_report_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            os.makedirs(os.path.join(directory, 'experiment-results'))
            with open(os.path.join(directory, 'experiment-results', 'runner-io-admission.json'), 'w') as handle:
                handle.write('{}')
            result = self.io_gate(directory, M3NATIVE_IO_GATE='1', STUB_EXIT='0')
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(os.path.join(directory, 'stub.log'), encoding='utf-8') as handle:
                self.assertRegex(handle.read(), r'--output experiment-results/runner-io-admission-[0-9]+\.json\n$')

    def fake_proc(self, directory, pressure=True):
        root = os.path.join(directory, 'fakeproc')
        os.makedirs(os.path.join(root, 'pressure'))
        if pressure:
            with open(os.path.join(root, 'pressure', 'io'), 'w', newline='\n') as handle:
                handle.write('some avg10=0.00 avg60=0.10 avg300=0.20 total=100\n'
                             'full avg10=0.00 avg60=0.05 avg300=0.10 total=50\n')
            with open(os.path.join(root, 'pressure', 'cpu'), 'w', newline='\n') as handle:
                handle.write('some avg10=1.00 avg60=2.00 avg300=3.00 total=900\n')
        with open(os.path.join(root, 'loadavg'), 'w', newline='\n') as handle:
            handle.write('0.50 0.40 0.30 1/200 12345\n')
        return start_block(arm_text()).replace('/proc/', 'fakeproc/')

    def test_the_sampler_logs_every_second_until_the_container_is_gone(self):
        with tempfile.TemporaryDirectory() as directory:
            block = self.fake_proc(directory)
            os.makedirs(os.path.join(directory, 'experiment-results'))
            script = 'set -euo pipefail\n' + block + 'sleep 2.6\n' + stop_block(arm_text()) + 'echo AFTER\n'
            result = self.bash(script, cwd=directory, M3NATIVE_HOST_SAMPLER='1')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('host sampler: pid ', result.stdout)
            self.assertRegex(result.stdout, r'host sampler: stopped after the container, [0-9]+ lines')
            self.assertTrue(result.stdout.endswith('AFTER\n'))
            with open(os.path.join(directory, 'experiment-results', 'host-pressure.log'), encoding='utf-8') as handle:
                lines = handle.read().splitlines()
            kinds = {}
            for line in lines:
                stamp, source, rest = line.split(' ', 2)
                self.assertRegex(stamp, r'^[0-9]+\.[0-9]+$')
                kinds.setdefault((source, rest), []).append(float(stamp))
            self.assertEqual(set(kinds), {('io', 'some avg10=0.00 avg60=0.10 avg300=0.20 total=100'),
                                          ('io', 'full avg10=0.00 avg60=0.05 avg300=0.10 total=50'),
                                          ('cpu', 'some avg10=1.00 avg60=2.00 avg300=3.00 total=900'),
                                          ('load', '0.50 0.40 0.30 1/200 12345')})
            stamps = kinds[('load', '0.50 0.40 0.30 1/200 12345')]
            self.assertGreaterEqual(len(stamps), 2)
            self.assertTrue(all(0.9 <= later - earlier <= 2.0 for earlier, later in zip(stamps, stamps[1:])), stamps)

    def test_a_missing_pressure_file_is_skipped_and_the_sampler_stops_with_the_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            block = self.fake_proc(directory, pressure=False)
            os.makedirs(os.path.join(directory, 'experiment-results'))
            result = self.bash('set -euo pipefail\n' + block + 'sleep 1.2\necho EXIT\n', cwd=directory,
                               M3NATIVE_HOST_SAMPLER='1')
            self.assertEqual(result.returncode, 0, result.stderr)
            log = os.path.join(directory, 'experiment-results', 'host-pressure.log')
            time.sleep(2.5)
            size = os.path.getsize(log)
            time.sleep(1.5)
            self.assertEqual(os.path.getsize(log), size, 'no sample after the arm exited')
            with open(log, encoding='utf-8') as handle:
                sources = {line.split(' ', 2)[1] for line in handle.read().splitlines()}
            self.assertEqual(sources, {'load'})

    def test_the_host_blocks_read_proc_only_and_write_into_the_results_dir(self):
        text = arm_text()
        for block in (start_block(text), stop_block(text), validation_block(text)):
            self.assertNotIn('/dev/tenstorrent', block)
            self.assertNotIn('tt-smi', block)
        for block in (start_block(text), stop_block(text)):
            self.assertNotIn('docker', block)
        self.assertEqual(sorted(set(re.findall(r'/proc/[a-z/{},$]+', start_block(text)))),
                         ['/proc/loadavg', '/proc/pressure/$source', '/proc/pressure/{io,cpu}'])
        for name in ('experiment-results/host-pressure.log', 'experiment-results/runner-io-admission.json'):
            self.assertIn(name, start_block(text))
        workflow = (ROOT / '.github' / 'workflows' / 'qwen-lever-n-m3native-gate.yml').read_text(encoding='utf-8')
        self.assertIn('path: m3native/experiment-results/', workflow, 'the gate workflow uploads the results dir')
        self.assertLess(text.index(start_block(text)), text.index('docker run --rm --name "$name"'))
        self.assertLess(text.index('docker run --rm --name "$name"'), text.index(stop_block(text)))


# -- shipping ----------------------------------------------------------------------------------------------

class ShippingTests(unittest.TestCase):
    def test_the_module_reaches_the_image_through_both_copy_lists(self):
        from test_serving_image_copy_closure import context_modules, dockerfile_modules, dockerfile_text

        docker = dockerfile_text()
        for name in ('publish_prewarm.py', 'serving_request_factory.py', 'serving_sequential_step.py',
                     'dflash_device.py', 'draft_kv_history.py', 'dflash_request_runtime.py', 'serving_packed_step.py'):
            with self.subTest(module=name):
                self.assertIn(name, dockerfile_modules(docker))
                self.assertIn(name, context_modules())

    def test_the_image_lists_only_gained_the_module(self):
        for relative, inserted in (
                ('docker/qwen-fast-serving.Dockerfile',
                 '# Publish prewarm (QWEN_FAST_PUBLISH_PREWARM, default off): serving_request_factory imports '
                 'publish_prewarm when the\n# flag is set, so it must reach the image beside it.\n'
                 'COPY scripts/ci/publish_prewarm.py /experiment-scripts/ci/\n'),
                ('.github/workflows/qwen-fast-serving-image.yml',
                 '          for name in publish_prewarm.py; do\n'
                 '            cp "serving-build-orchestrator/scripts/ci/$name" "$context/scripts/ci/"\n'
                 '          done\n')):
            with self.subTest(file=relative):
                text = change_text(relative)
                self.assertEqual(text.count(inserted), 1)
                self.assertEqual(text.replace(inserted, '', 1), parent_text(relative))

    def test_the_cpu_suite_runs_this_file(self):
        self.assertRegex(CPU_WORKFLOW.read_text(encoding='utf-8'), r'python -B -m unittest [^\n]*\btest_publish_prewarm\b')

    def test_the_pinned_and_workflow_sources_are_untouched(self):
        """By this change: PARENT against its own commit (the working tree while uncommitted). The gate
        workflow gets the experiment's tags next, which is not this change."""
        pinned = ['scripts/ci/dflash_device.py', 'scripts/ci/draft_kv_history.py', 'scripts/ci/dflash_request_runtime.py',
                  'scripts/ci/verifier_engine.py', 'scripts/ci/serving_packed_step.py', 'scripts/ci/runner_io_gate.py',
                  '.github/workflows/qwen-lever-n-m3native-gate.yml']
        commit = change_commit()
        self.assertEqual(git('diff', '--name-only', PARENT, *([commit] if commit else []), '--', *pinned).strip(), '')

    def test_every_touched_file_is_lf(self):
        for relative in ('scripts/ci/publish_prewarm.py', 'scripts/ci/serving_request_factory.py',
                         'scripts/ci/serving_sequential_step.py', 'scripts/ci/lever_n_m3native_gate.py',
                         'scripts/ci/lever_n_m3native_run_arm.sh', 'scripts/ci/test_publish_prewarm.py',
                         'scripts/ci/test_dflash_round_b1.py',
                         'docker/qwen-fast-serving.Dockerfile', '.github/workflows/qwen-fast-serving-image.yml',
                         '.github/workflows/qwen-integration-cpu.yml'):
            with self.subTest(file=relative):
                self.assertNotIn(b'\r', (ROOT / relative).read_bytes())


if __name__ == '__main__':
    unittest.main()
