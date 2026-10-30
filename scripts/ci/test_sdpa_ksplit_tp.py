"""CPU tests of the K-split lever at the model (sdpa_pf_ksplit_tp.py), its error statistic (ksplit_numerics.py) at the audit, its smoke rule (ksplit_smoke.py) and its binding by tp_addresses: the
strict flags, the off path, the wrapper stacked on the oneq lever's own wrapper over the graft's _qwen_pf_program_config (taken from the committed graft source and run against a fake ttnn), the
eligibility (the planner's reading of the factory), the one-time binary check, the audit's sampling and judgement, the markers and the smoke rule.

Run from scripts/ci:  python -B -m unittest test_sdpa_ksplit_tp
"""

import ast
import itertools
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

try:
    import torch
except ImportError:
    torch = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GRAFT = ROOT / 'docker' / 'qwen-c2-graft' / 'graft' / 'attention' / 'tp.py'
KSPLIT = ROOT / 'optimisation' / 'ttnn-op' / 'sdpa_prefill_ksplit'
for path in (HERE, KSPLIT, ROOT / 'optimisation' / 'ttnn-op' / 'sdpa_prefill_chain', ROOT / 'optimisation' / 'ttnn-op' / 'sdpa_prefill_oneq'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import apply_factory_pf as pf  # noqa: E402
import apply_factory_pk as pk  # noqa: E402
import ksplit_numerics as numerics  # noqa: E402
import ksplit_planner as planner  # noqa: E402
import ksplit_smoke as smoke  # noqa: E402
import sdpa_pf_ksplit_tp as lever  # noqa: E402
import sdpa_pf_oneq_tp as oneq  # noqa: E402

ENV_ON = {'QWEN_FAST_SDPA_PF': '1', 'QWEN_FAST_SDPA_PF_ONEQ': '1', 'QWEN_FAST_SDPA_PF_KSPLIT': '1', 'QWEN_FAST_TP': '4'}


class Config:
    """ttnn.SDPAProgramConfig as the graft builds it: keyword fields, readable as attributes."""

    def __init__(self, **fields):
        self.fields = fields

    def __getattr__(self, name):
        try:
            return self.__dict__['fields'][name]
        except KeyError:
            raise AttributeError(name)

    def __repr__(self):
        return 'Config(%r)' % (self.fields,)


class Grid:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __eq__(self, other):
        return (self.x, self.y) == (other.x, other.y)

    def __hash__(self):
        return hash((self.x, self.y))


def graft_function(fake_ttnn):
    """The graft's _qwen_pf_program_config and the constants it reads, compiled from the committed source with `ttnn` = the fake."""
    tree = ast.parse(GRAFT.read_text(encoding='utf-8'))
    wanted = {'_QWEN_PF_ROWS', '_QWEN_PF_CHUNK', '_QWEN_PF_FLAGS', '_QWEN_PF_TAG', '_QWEN_PF_DEFAULT_FLAGS'}
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in wanted for t in node.targets):
            body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == '_qwen_pf_program_config':
            body.append(node)
    namespace = {'ttnn': fake_ttnn}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(GRAFT), 'exec'), namespace)    # noqa: S102 - the committed graft source
    return namespace['_qwen_pf_program_config'], namespace


def make_layer(word=0x5EFA0003, bf8=True, nh=6, nkv=1, grid=(13, 10)):
    mesh = types.SimpleNamespace(compute_with_storage_grid_size=lambda: Grid(*grid))
    return types.SimpleNamespace(_sdpa_pf_word=word, _sdpa_bf8=bf8, NH=nh, NKV=nkv, mesh=mesh)


class Fake:
    def __init__(self):
        self.calls = []
        self.deallocated = []
        self.transformer = types.SimpleNamespace(chunked_scaled_dot_product_attention=self.op)
        self.SDPAProgramConfig = Config

    def op(self, **kwargs):
        self.calls.append(kwargs)
        return None

    def deallocate(self, tensor):
        self.deallocated.append(tensor)


class SettingsTests(unittest.TestCase):
    def test_off_is_off_and_imports_nothing(self):
        for env in ({}, {'QWEN_FAST_SDPA_PF_KSPLIT': '0'}, {'QWEN_FAST_SDPA_PF_KSPLIT': '0', 'QWEN_FAST_SDPA_PF_KSPLIT_AUDIT': '0'}):
            self.assertEqual(lever.settings(env), (False, False))
            self.assertFalse(lever.enabled(env))
            with mock.patch.object(lever.importlib, 'import_module') as imported:
                self.assertEqual(lever.install(env), [])
                imported.assert_not_called()

    def test_strict_values_and_pairings(self):
        for bad in ('2', 'true', '', 'yes', '01'):
            with self.assertRaises(ValueError, msg=bad):
                lever.settings(dict(ENV_ON, QWEN_FAST_SDPA_PF_KSPLIT=bad))
        with self.assertRaisesRegex(ValueError, 'AUDIT needs QWEN_FAST_SDPA_PF_KSPLIT=1'):
            lever.settings({'QWEN_FAST_SDPA_PF_KSPLIT_AUDIT': '1'})
        with self.assertRaisesRegex(ValueError, 'must be 0 or 1'):
            lever.settings(dict(ENV_ON, QWEN_FAST_SDPA_PF_KSPLIT_AUDIT='2'))
        self.assertEqual(lever.settings(ENV_ON), (True, False))
        self.assertEqual(lever.settings(dict(ENV_ON, QWEN_FAST_SDPA_PF_KSPLIT_AUDIT='1')), (True, True))

    def test_the_lever_needs_the_oneq_call_and_what_that_needs(self):
        with self.assertRaisesRegex(ValueError, 'needs QWEN_FAST_SDPA_PF_ONEQ=1'):
            lever.settings({'QWEN_FAST_SDPA_PF_KSPLIT': '1', 'QWEN_FAST_SDPA_PF': '1', 'QWEN_FAST_TP': '4'})
        with self.assertRaisesRegex(ValueError, 'needs QWEN_FAST_SDPA_PF_ONEQ=1'):
            lever.settings({'QWEN_FAST_SDPA_PF_KSPLIT': '1', 'QWEN_FAST_SDPA_PF_ONEQ': '0', 'QWEN_FAST_SDPA_PF': '1', 'QWEN_FAST_TP': '4'})
        with self.assertRaisesRegex(ValueError, 'needs QWEN_FAST_SDPA_PF=1'):
            lever.settings({'QWEN_FAST_SDPA_PF_KSPLIT': '1', 'QWEN_FAST_SDPA_PF_ONEQ': '1', 'QWEN_FAST_TP': '4'})
        with self.assertRaisesRegex(ValueError, 'four-card lever'):
            lever.settings(dict(ENV_ON, QWEN_FAST_TP='2'))
        for flags in ('0x1', '0x3', '0x5', '0x7', '3', ''):
            self.assertEqual(lever.settings(dict(ENV_ON, QWEN_FAST_SDPA_PF_FLAGS=flags))[0], True, flags)
        with self.assertRaisesRegex(ValueError, 'production chain flag set'):
            lever.settings(dict(ENV_ON, QWEN_FAST_SDPA_PF_FLAGS='0xB'))

    def test_constants_equal_the_factory_patch_the_planner_the_smoke_rule_and_the_contract(self):
        self.assertEqual(lever.KSPLIT_BIT, pk.FLAG_KSPLIT)
        self.assertEqual(lever.ONEQ_BIT, oneq.ONEQ_BIT)
        self.assertEqual(lever.PF_TAG, pk.PF_TAG)
        self.assertEqual(lever.SLICE_CAP, pk.SLICE_CAP)
        self.assertEqual(lever.SLICE_CAP, planner.SLICE_CAP)
        self.assertEqual(lever.ROWS, planner.QUALIFIED_ROWS)
        self.assertTrue(set(lever.ROWS) <= set(oneq.ROWS) and set(lever.ROWS) <= set(pf.QUALIFIED_ROWS), 'the rows the split takes are chain-qualified rows')
        self.assertEqual(lever.Q_CHUNK, planner.Q_CHUNK)
        self.assertEqual(lever.BINARY_MARKER.decode(), pk.FATAL_SLICES.split(':')[0])
        self.assertIn(lever.BINARY_MARKER.decode(), pk.NEW_QWEN_STRINGS[0])
        self.assertEqual(smoke.ROWS, lever.ROWS)
        self.assertEqual((smoke.FLAG, smoke.AUDIT_FLAG), (lever.FLAG, lever.AUDIT_FLAG))
        for name in ('ENGAGED', 'SKIPPED', 'FELL_BACK', 'AUDIT', 'MISMATCH'):
            self.assertEqual(getattr(smoke, name), getattr(lever, {'MISMATCH': 'AUDIT_MISMATCH_MARKER'}.get(name, name + '_MARKER')))

    def test_the_markers_are_prefixes_of_one_smoke_marker(self):
        base = 'sdpa prefill ksplit'
        self.assertEqual(lever.ENGAGED_MARKER, '[PINDIAG] %s engaged' % base)
        self.assertEqual(lever.FELL_BACK_MARKER, '[PINDIAG] %s fell back' % base)
        self.assertEqual(lever.AUDIT_MARKER, '[PINDIAG] %s audit' % base)
        self.assertTrue(lever.AUDIT_MISMATCH_MARKER.startswith(lever.AUDIT_MARKER))
        self.assertNotIn('fell back', lever.SKIPPED_MARKER, 'a call the split does not take by design must never look like a fall-back to the generic fusion rule')


class EligibilityTests(unittest.TestCase):
    def test_the_slice_count_is_the_planners(self):
        for rows, nqh, cores in itertools.product((128, 256, 512, 1024, 1536, 2048), (4, 6, 8, 12), (48, 64, 96, 110, 130)):
            self.assertEqual(lever.split_k_slices(rows, nqh, cores), planner.split_k_slices(rows, nqh, cores), (rows, nqh, cores))

    def test_tp4_rows_on_both_grids(self):
        for grid in ((13, 10), (11, 10)):
            for rows in (512, 1024):
                self.assertIsNone(lever.eligibility(6, 1, rows, grid), (grid, rows))
            self.assertIn('no second slice', lever.eligibility(6, 1, 2048, grid))
        self.assertIn('no second slice', lever.eligibility(6, 1, 256, (13, 10)), '256 rows would split but is not chain-qualified')

    def test_what_the_split_does_not_take(self):
        self.assertIn('one slice', lever.eligibility(6, 1, 1024, (8, 8)))                       # 48 chunks on 64 cores
        self.assertIsNone(lever.eligibility(6, 1, 512, (8, 8)))                                  # 24 chunks: two slices
        self.assertIn('one slice', lever.eligibility(12, 2, 1024, (13, 10)))                    # TP2 heads: 96 chunks on 130 cores
        self.assertIn('odd q chunk count', lever.eligibility(6, 1, 1920, (13, 10)))
        self.assertIn('exceed the 130 cores', lever.eligibility(12, 2, 2048, (13, 10)))

    def test_eligibility_is_the_planners_reading_of_the_factory(self):
        for nqh, nkh, rows, grid in itertools.product((4, 6, 8, 12), (1, 2), (512, 1024, 2048), ((8, 8), (11, 10), (13, 10), (8, 12))):
            if nqh % nkh:
                continue
            mine = planner.plan(nqh, nkh, rows, 128, grid)
            if rows in lever.ROWS:
                self.assertEqual(lever.eligibility(nqh, nkh, rows, grid) is None, mine['refusal'] is None, (nqh, nkh, rows, grid))
            else:
                self.assertIsNotNone(lever.eligibility(nqh, nkh, rows, grid), 'the 2,048-row pass is the oneq call even where a second slice would fit (4 heads: 64 chunks)')

    def test_topology_is_what_the_factory_logs(self):
        for rows in (512, 1024):
            mine = planner.plan(rows=rows)
            self.assertEqual(lever.topology(6, 1, rows, (13, 10)), (mine['slices'], mine['total'], mine['busy'], mine['chain_count']))


@unittest.skipIf(torch is None, 'torch not installed')
class WrapperTests(unittest.TestCase):
    def setUp(self):
        lever.reset_state()
        oneq.reset_state()
        self.fake = Fake()
        self.logged = []
        self.saved_ttnn = sys.modules.get('ttnn')
        sys.modules['ttnn'] = self.fake
        for module in (lever, oneq):
            patch = mock.patch.object(module, '_log', self.logged.append)
            patch.start()
            self.addCleanup(patch.stop)
        self.graft, _ = graft_function(self.fake)
        self.served = Config(served=True)
        oneq._STATE['binary_ok'] = True
        oneq._STATE['original'] = self.graft

    def oneq_wrapper(self, layer, served, qk, flexible, S):
        return oneq.program_config(layer, served, qk, flexible, S, original=self.graft, binary_check=lambda: True)

    def tearDown(self):
        lever.reset_state()
        oneq.reset_state()
        if self.saved_ttnn is None:
            sys.modules.pop('ttnn', None)
        else:
            sys.modules['ttnn'] = self.saved_ttnn

    def twin(self, layer, S, flexible=True, qk=128, binary=True):
        return lever.program_config(layer, self.served, qk, flexible, S, original=self.oneq_wrapper, binary_check=lambda: binary)

    def test_a_tail_call_gets_the_ksplit_bit_on_top_of_the_oneq_word_and_nothing_else_changes(self):
        layer = make_layer()
        for rows in lever.ROWS:
            oneq_config = self.oneq_wrapper(layer, self.served, 128, True, rows)
            mine = self.twin(layer, rows)
            self.assertEqual(mine.fields.pop('max_cores_per_head_batch'), 0x5EFA001B)
            self.assertEqual(oneq_config.fields.pop('max_cores_per_head_batch'), 0x5EFA000B)
            self.assertEqual(mine.fields, oneq_config.fields, 'only the word differs from the oneq lever\'s own config')

    def test_every_flag_set_keeps_its_bits(self):
        for flags in oneq.PRODUCTION_FLAGS:
            layer = make_layer(word=lever.PF_TAG | flags)
            self.assertEqual(self.twin(layer, 1024).fields['max_cores_per_head_batch'], lever.PF_TAG | flags | 0x8 | 0x10)

    def test_the_2048_row_chunk_pass_is_the_oneq_call_bitwise(self):
        layer = make_layer()
        mine = self.twin(layer, 2048)
        self.assertEqual(mine.fields['max_cores_per_head_batch'], 0x5EFA000B)
        mine = [line for line in self.logged if 'ksplit' in line]
        self.assertEqual(mine, ['%s rows=2048 reason=%s' % (lever.SKIPPED_MARKER, lever.eligibility(6, 1, 2048, (13, 10)))])
        self.twin(layer, 2048)
        self.assertEqual(len([line for line in self.logged if 'ksplit' in line]), 1, 'once per rows and reason')
        self.assertEqual(lever._STATE['skipped'], 2)
        self.assertNotIn('fell back', mine[0])
        self.assertTrue(any('oneq engaged flags=0xb rows=2048' in line for line in self.logged), 'the oneq lever engaged its own 2,048-row call')

    def test_calls_the_graft_keeps_served_stay_served_and_log_nothing(self):
        layer = make_layer()
        self.assertIs(self.twin(layer, 4096), self.served)
        self.assertIs(self.twin(layer, 1920), self.served)
        self.assertIs(self.twin(layer, 1024, flexible=False), self.served)
        self.assertIs(self.twin(layer, 1024, qk=256), self.served)
        self.assertIs(self.twin(make_layer(bf8=False), 1024), self.served)
        self.assertIs(self.twin(make_layer(word=None), 1024), self.served)
        self.assertEqual(self.logged, [])

    def test_a_call_the_oneq_lever_did_not_take_is_a_fall_back_not_a_split(self):
        layer = make_layer(nh=12, nkv=2)                                       # TP2 heads at 2048 rows: 192 chunks: oneq falls back to the plain chain
        config = self.twin(layer, 2048)
        self.assertEqual(config.fields['max_cores_per_head_batch'], 0x5EFA0003)
        self.assertTrue(any(line.startswith('%s rows=2048 reason=the call is not a oneq call' % lever.FELL_BACK_MARKER) for line in self.logged), self.logged)
        self.assertEqual(lever._STATE['fallbacks'], 1)

    def test_a_small_grid(self):
        layer = make_layer(grid=(8, 8))
        self.assertEqual(self.twin(layer, 512).fields['max_cores_per_head_batch'], 0x5EFA001B)       # 24 chunks: two slices of 48 cores
        self.assertEqual(self.twin(layer, 1024).fields['max_cores_per_head_batch'], 0x5EFA000B)      # 48 chunks on 64 cores: one slice, the oneq call

    def test_engaged_is_logged_once_per_rows_with_the_topology(self):
        layer = make_layer()
        self.twin(layer, 1024)
        self.twin(layer, 1024)
        self.twin(layer, 512)
        engaged = [line for line in self.logged if lever.ENGAGED_MARKER in line]
        self.assertEqual(engaged, ['[PINDIAG] sdpa prefill ksplit engaged flags=0x1b rows=1024 heads=6/1 q_chunks=48 slices=2 cores=130 busy=96 chains=16',
                                   '[PINDIAG] sdpa prefill ksplit engaged flags=0x1b rows=512 heads=6/1 q_chunks=24 slices=4 cores=130 busy=96 chains=16'])
        self.assertEqual(lever._STATE['engaged'], 3)
        for line in engaged:
            self.assertRegex(line, smoke.ENGAGED_ROWS)

    def test_the_binary_check_runs_once_and_refuses_a_binary_without_the_edits(self):
        layer = make_layer()
        calls = []
        lever.program_config(layer, self.served, 128, True, 1024, original=self.oneq_wrapper, binary_check=lambda: calls.append(1) or True)
        lever.program_config(layer, self.served, 128, True, 512, original=self.oneq_wrapper, binary_check=lambda: calls.append(1) or True)
        self.assertEqual(calls, [1])
        lever.reset_state()
        with self.assertRaisesRegex(RuntimeError, 'lacks the K-split edits.*K64j-OQ-KS'):
            self.twin(layer, 1024, binary=False)

    def test_a_pass_the_split_skips_never_asks_for_the_binary(self):
        layer = make_layer()
        lever.program_config(layer, self.served, 128, True, 2048, original=self.oneq_wrapper, binary_check=lambda: self.fail('asked'))

    def test_binary_has_ksplit_reads_the_one_mapped_library(self):
        with tempfile.TemporaryDirectory() as directory:
            so = Path(directory) / '_ttnncpp.so'
            so.write_bytes(b'\x7fELF\x00[QWEN-SDPA-PF] oneq needs one q chunk per core\x00')
            maps = Path(directory) / 'maps'
            maps.write_text('7f00 r-xp 0 00:00 0 %s\n' % so)
            self.assertFalse(lever.binary_has_ksplit(str(maps)))
            so.write_bytes(so.read_bytes() + lever.BINARY_MARKER + b': {} q chunks on {} cores\x00')
            self.assertTrue(lever.binary_has_ksplit(str(maps)))
            other = Path(directory) / 'lib' / '_ttnncpp.so'
            other.parent.mkdir()
            other.write_bytes(b'x')
            maps.write_text('7f00 r-xp 0 00:00 0 %s\n7f01 r-xp 0 00:00 0 %s\n' % (so, other))
            with self.assertRaises(RuntimeError):
                lever.binary_has_ksplit(str(maps))

    def test_without_install_the_wrapper_refuses_to_guess_what_it_wraps(self):
        with self.assertRaisesRegex(RuntimeError, 'was not captured'):
            lever.program_config(make_layer(), self.served, 128, True, 1024)


@unittest.skipIf(torch is None, 'torch not installed')
class InstallTests(unittest.TestCase):
    def setUp(self):
        lever.reset_state()
        oneq.reset_state()
        self.fake = Fake()
        self.saved = {name: sys.modules.get(name) for name in ('ttnn', lever.GRAFT_MODULE)}
        sys.modules['ttnn'] = self.fake
        self.graft, _ = graft_function(self.fake)
        self.module = types.ModuleType(lever.GRAFT_MODULE)
        self.module._qwen_pf_program_config = self.graft
        self.holder = types.ModuleType('some_other_holder')
        self.holder._qwen_pf_program_config = self.graft
        for module in (self.module, self.holder):
            sys.modules[module.__name__] = module

    def tearDown(self):
        lever.reset_state()
        oneq.reset_state()
        sys.modules.pop('some_other_holder', None)
        for name, module in self.saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_it_refuses_a_graft_function_that_is_not_the_oneq_wrapper(self):
        with self.assertRaisesRegex(RuntimeError, 'not the oneq lever\'s wrapper'):
            lever.install(ENV_ON)
        self.assertIs(self.module._qwen_pf_program_config, self.graft)

    def test_it_stacks_on_the_oneq_wrapper_and_rebinds_every_holder(self):
        oneq_rebound = oneq.install(ENV_ON)
        self.assertEqual(len(oneq_rebound), 2)
        oneq_wrapper = self.module._qwen_pf_program_config
        rebound = lever.install(ENV_ON)
        self.assertEqual({id(namespace) for namespace, _name, _old in rebound}, {id(self.module.__dict__), id(self.holder.__dict__)})
        self.assertTrue(all(old is oneq_wrapper for _n, _name, old in rebound))
        self.assertTrue(self.module._qwen_pf_program_config._ksplit_wrapper)
        self.assertIs(self.holder._qwen_pf_program_config, self.module._qwen_pf_program_config)
        self.assertIs(lever._STATE['original'], oneq_wrapper)
        self.assertEqual(lever.install(ENV_ON), [], 'installing twice binds nothing new')
        for namespace, name, old in reversed(rebound):
            namespace[name] = old
        self.assertIs(self.module._qwen_pf_program_config, oneq_wrapper)

    def test_the_bound_pair_gives_the_ksplit_word_and_serves_through_the_graft(self):
        oneq.install(ENV_ON)
        lever.install(ENV_ON)
        oneq._STATE['binary_ok'] = True
        lever._STATE['binary_ok'] = True
        layer = make_layer()
        bound = self.module._qwen_pf_program_config
        self.assertEqual(bound(layer, Config(served=True), 128, True, 1024).fields['max_cores_per_head_batch'], 0x5EFA001B)
        self.assertEqual(bound(layer, Config(served=True), 128, True, 2048).fields['max_cores_per_head_batch'], 0x5EFA000B)
        self.assertEqual(bound(make_layer(word=None), Config(served=True), 128, True, 1024).fields, {'served': True})

    def test_a_malformed_setting_fails_at_attach(self):
        with self.assertRaises(ValueError):
            lever.install(dict(ENV_ON, QWEN_FAST_SDPA_PF_KSPLIT='maybe'))
        self.assertIs(self.module._qwen_pf_program_config, self.graft)


class TpAddressesHookTests(unittest.TestCase):
    """tp_addresses.install() binds the lever AFTER the oneq one, and only when one of its two flags is SET (any value, so a malformed one reaches the strict parser)."""

    FOUR = {'QWEN_FAST_TP': '4', 'QWEN_FAST_SDPA_PF': '1', 'QWEN_FAST_SDPA_PF_ONEQ': '1'}

    @classmethod
    def setUpClass(cls):
        if torch is None:
            raise unittest.SkipTest('torch not installed')

    def setUp(self):
        InstallTests.setUp(self)
        self.addCleanup(InstallTests.tearDown, self)
        import tp_addresses
        self.tp_addresses = tp_addresses
        self.addCleanup(tp_addresses.uninstall)

    def test_flag_unset_or_zero_leaves_the_oneq_wrapper_and_never_asks_the_lever(self):
        for extra in ({}, {'QWEN_FAST_SDPA_PF_KSPLIT': '0'}):
            with self.subTest(extra=extra):
                self.tp_addresses.uninstall()
                lever.reset_state()
                oneq.reset_state()
                self.tp_addresses.install(dict(self.FOUR, **extra))
                self.assertTrue(self.module._qwen_pf_program_config._oneq_wrapper)
                self.assertFalse(getattr(self.module._qwen_pf_program_config, '_ksplit_wrapper', False))
        self.tp_addresses.uninstall()
        with mock.patch.object(lever, 'install', side_effect=AssertionError('asked')) as asked:
            self.tp_addresses.install(dict(self.FOUR))
        asked.assert_not_called()

    def test_flag_on_binds_the_pair_in_order_and_uninstall_puts_the_graft_function_back(self):
        self.tp_addresses.install(dict(self.FOUR, QWEN_FAST_SDPA_PF_KSPLIT='1'))
        self.assertTrue(self.module._qwen_pf_program_config._ksplit_wrapper)
        self.assertIs(self.holder._qwen_pf_program_config, self.module._qwen_pf_program_config)
        self.assertTrue(lever._STATE['original']._oneq_wrapper, 'the lever captured the oneq wrapper, bound just before')
        self.tp_addresses.uninstall()
        self.assertIs(self.module._qwen_pf_program_config, self.graft)
        self.assertIs(self.holder._qwen_pf_program_config, self.graft)

    def test_the_lever_gets_the_environment_install_was_given(self):
        given = dict(self.FOUR, QWEN_FAST_SDPA_PF_KSPLIT='1', QWEN_FAST_SDPA_PF_KSPLIT_AUDIT='1')
        with mock.patch.object(lever, 'install', return_value=[]) as called:
            self.tp_addresses.install(dict(given))
        called.assert_called_once_with(given)

    def test_a_malformed_or_orphan_flag_reaches_the_strict_parser_and_fails_at_attach(self):
        for extra in ({'QWEN_FAST_SDPA_PF_KSPLIT': 'yes'}, {'QWEN_FAST_SDPA_PF_KSPLIT_AUDIT': '1'}):
            with self.subTest(extra=extra):
                with self.assertRaises(ValueError):
                    self.tp_addresses.install(dict(self.FOUR, **extra))
                self.tp_addresses.uninstall()

    def test_the_ksplit_flag_without_the_oneq_flag_fails_at_attach(self):
        with self.assertRaisesRegex(ValueError, 'needs QWEN_FAST_SDPA_PF_ONEQ=1'):
            self.tp_addresses.install({'QWEN_FAST_TP': '4', 'QWEN_FAST_SDPA_PF': '1', 'QWEN_FAST_SDPA_PF_KSPLIT': '1'})

    def test_the_binding_comes_after_the_oneq_one_in_the_source(self):
        source = (HERE / 'tp_addresses.py').read_text(encoding='utf-8')
        self.assertLess(source.index('import sdpa_pf_oneq_tp'), source.index('import sdpa_pf_ksplit_tp'))
        self.assertLess(source.index('import sdpa_pf_ksplit_tp'), source.index('    return rebound'))
        self.assertEqual(source.count('import sdpa_pf_ksplit_tp'), 1)


@unittest.skipIf(torch is None, 'torch not installed')
class AuditTests(unittest.TestCase):
    def setUp(self):
        lever.reset_state()
        self.fake = Fake()
        self.logged = []
        self.saved_ttnn = sys.modules.get('ttnn')
        sys.modules['ttnn'] = self.fake
        patch = mock.patch.object(lever, '_log', self.logged.append)
        patch.start()
        self.addCleanup(patch.stop)
        self.served = Config(served=True)
        self.original = lambda layer, served, qk, flexible, S: Config(max_cores_per_head_batch=0x5EFA000B)
        lever._STATE.update(original=self.original, audit=True, binary_ok=True)
        self.fake.ConcatMeshToTensor = lambda mesh, dim: ('compose', mesh, dim)
        self.fake.to_torch = lambda tensor, mesh_composer=None: tensor

    def tearDown(self):
        lever.reset_state()
        if self.saved_ttnn is None:
            sys.modules.pop('ttnn', None)
        else:
            sys.modules['ttnn'] = self.saved_ttnn

    def engage(self, rows=1024):
        layer = make_layer()
        return lever.program_config(layer, self.served, 128, True, rows, original=self.original)

    def make_audited(self, mine, reference):
        config = self.engage()
        outputs = {0x5EFA001B: mine, 0x5EFA000B: reference}
        calls = []

        def op(*args, **kwargs):
            calls.append(kwargs['program_config'].fields['max_cores_per_head_batch'])
            return outputs[kwargs['program_config'].fields['max_cores_per_head_batch']]

        self.fake.transformer.chunked_scaled_dot_product_attention = op
        lever._STATE['op'] = None
        lever._install_audit(self.fake)
        return config, self.fake.transformer.chunked_scaled_dot_product_attention, calls

    def call(self, audited, config, rows=1024):
        return audited(input_tensor_q=types.SimpleNamespace(shape=(1, 6, rows, 256)), program_config=config)

    def test_outputs_inside_the_bound_pass_and_both_programs_ran(self):
        reference = torch.randn(1, 6, 1024, 256).to(torch.bfloat16)
        mine = (reference.float() * (1 + 2e-4)).to(torch.bfloat16)
        config, audited, calls = self.make_audited(mine, reference)
        self.call(audited, config)
        self.assertEqual(calls, [0x5EFA001B, 0x5EFA000B])
        line = [text for text in self.logged if lever.AUDIT_MARKER in text][0]
        self.assertIn('n=1 call=0 rows=1024', line)
        self.assertIn('within_bound=True', line)
        self.assertNotIn('exact=True', line, 'a statistic, not a bitwise audit')
        self.assertEqual(smoke.problems({lever.FLAG: '1', lever.AUDIT_FLAG: '1'}, '\n'.join(self.logged + [lever.ENGAGED_MARKER + ' flags=0x1b rows=1024'])), [])

    def test_an_output_outside_the_bound_raises_and_logs_the_mismatch(self):
        reference = torch.randn(1, 6, 1024, 256).to(torch.bfloat16)
        mine = (reference.float() * 1.5).to(torch.bfloat16)
        config, audited, _calls = self.make_audited(mine, reference)
        with self.assertRaisesRegex(RuntimeError, 'outside the waiver bound'):
            self.call(audited, config)
        self.assertTrue(any(lever.AUDIT_MISMATCH_MARKER in text and 'within_bound=False' in text for text in self.logged), self.logged)
        self.assertTrue(smoke.problems({lever.FLAG: '1', lever.AUDIT_FLAG: '1'}, '\n'.join(self.logged)))

    def test_a_non_finite_output_is_a_mismatch(self):
        reference = torch.randn(1, 6, 1024, 256).to(torch.bfloat16)
        mine = reference.clone()
        mine.view(-1)[7] = float('nan')
        config, audited, _calls = self.make_audited(mine, reference)
        with self.assertRaises(RuntimeError):
            self.call(audited, config)

    def test_a_shape_difference_is_a_mismatch(self):
        config, audited, _calls = self.make_audited(torch.zeros(1, 6, 512, 256), torch.zeros(1, 6, 1024, 256))
        with self.assertRaisesRegex(RuntimeError, 'shapes'):
            self.call(audited, config)

    def test_a_call_that_is_not_the_ksplit_one_is_not_audited(self):
        reference = torch.randn(1, 6, 1024, 256).to(torch.bfloat16)
        config, audited, calls = self.make_audited(reference, reference)
        audited(input_tensor_q=None, program_config=Config(max_cores_per_head_batch=0x5EFA000B))
        self.assertEqual(calls, [0x5EFA000B])
        self.assertEqual(lever._STATE['audit_calls'], 0)

    def test_sampling_is_the_first_eight_then_every_24th(self):
        reference = torch.randn(1, 6, 128, 256).to(torch.bfloat16)
        config, audited, calls = self.make_audited(reference, reference)
        for _ in range(8 + 24 + 1):
            self.call(audited, config, rows=128)
        audited_calls = calls.count(0x5EFA000B)
        self.assertEqual(audited_calls, 8 + 2, 'call indexes 0..7, then 8 and 32')
        self.assertEqual(lever._STATE['audited'], audited_calls)
        self.assertEqual(lever._STATE['audit_calls'], 33)

    def drifted(self, factor=1.04):
        """An output further from the oneq call than the base bound allows (rms ~2.6 ulps, max ~8 against 1.5 / 8) but inside what a long context's noise allows."""
        reference = torch.randn(1, 6, 1024, 256, generator=torch.Generator().manual_seed(3)).to(torch.bfloat16)
        return (reference.float() * factor).to(torch.bfloat16), reference

    def test_the_bound_grows_with_the_context_the_audit_reads_from_the_call(self):
        mine, reference = self.drifted()
        config, audited, _calls = self.make_audited(mine, reference)
        with self.assertRaisesRegex(RuntimeError, 'outside the waiver bound'):
            audited(input_tensor_q=types.SimpleNamespace(shape=(1, 6, 1024, 256)), program_config=config, chunk_start_idx=0)
        self.assertTrue(any('context=1024 within_bound=False' in text for text in self.logged), self.logged)
        self.logged.clear()
        lever._STATE['audit_calls'] = 0
        audited(input_tensor_q=types.SimpleNamespace(shape=(1, 6, 1024, 256)), program_config=config, chunk_start_idx=131072 - 1024)
        line = [text for text in self.logged if lever.AUDIT_MARKER in text][0]
        self.assertIn('context=131072 ', line)
        self.assertIn('within_bound=True', line)

    def test_the_flexible_paths_device_tensor_is_read_back_for_the_context(self):
        mine, reference = self.drifted()
        config, audited, _calls = self.make_audited(mine, reference)
        audited(input_tensor_q=types.SimpleNamespace(shape=(1, 6, 1024, 256)), program_config=config, chunk_start_idx_tensor=torch.tensor([[130048]], dtype=torch.int32))
        self.assertIn('context=131072 ', [text for text in self.logged if lever.AUDIT_MARKER in text][0])

    def test_an_unreadable_context_takes_the_lenient_end_and_says_so(self):
        mine, reference = self.drifted()
        config, audited, _calls = self.make_audited(mine, reference)
        self.call(audited, config)
        self.assertIn('context=None ', [text for text in self.logged if lever.AUDIT_MARKER in text][0])
        unreadable = object()

        def to_torch(tensor, mesh_composer=None):
            if tensor is unreadable:
                raise RuntimeError('device read failed')
            return tensor
        self.fake.to_torch = to_torch
        lever._STATE['audit_calls'] = 0
        self.logged.clear()
        audited(input_tensor_q=types.SimpleNamespace(shape=(1, 6, 1024, 256)), program_config=config, chunk_start_idx_tensor=unreadable)
        self.assertIn('context=None ', [text for text in self.logged if lever.AUDIT_MARKER in text][0])

    def test_a_wrong_merge_is_outside_the_bound_at_every_context(self):
        reference = torch.randn(1, 6, 1024, 256).to(torch.bfloat16)
        mine = (reference.float() * 1.5).to(torch.bfloat16)
        config, audited, _calls = self.make_audited(mine, reference)
        with self.assertRaises(RuntimeError):
            audited(input_tensor_q=types.SimpleNamespace(shape=(1, 6, 1024, 256)), program_config=config, chunk_start_idx=253952)

    def test_the_audit_is_off_without_its_flag(self):
        lever._STATE['audit'] = False
        config = self.engage()
        self.assertIsNone(lever._STATE['pair'])
        self.assertIsNone(lever._STATE['op'])
        self.assertEqual(config.fields['max_cores_per_head_batch'], 0x5EFA001B)


class BoundTests(unittest.TestCase):
    """ksplit_numerics.bound_for: the provisional bound grows with the context (docs/tp4-prefill3-ksplit.md, the waiver card); pure arithmetic."""

    def test_the_base_bound_holds_up_to_two_thousand_tokens_and_grows_with_the_square_root_of_the_blocks(self):
        base = numerics.BOUND
        for context in (128, 1024, 2048):
            self.assertEqual(numerics.bound_for(context), base, context)
        at_8k = numerics.bound_for(8192)
        self.assertAlmostEqual(at_8k['max_ulps'], base['max_ulps'] * 2.0)
        self.assertAlmostEqual(at_8k['rms_ulps'], base['rms_ulps'] * 2.0)
        self.assertAlmostEqual(at_8k['frac_gt2'], base['frac_gt2'] * 4.0)
        at_131k = numerics.bound_for(131072)
        self.assertAlmostEqual(at_131k['max_ulps'], base['max_ulps'] * 8.0)
        self.assertAlmostEqual(at_131k['rms_ulps'], base['rms_ulps'] * 8.0)
        self.assertEqual(at_131k['frac_gt2'], numerics.FRAC_CAP, 'capped')

    def test_monotone_and_unknown_means_the_lenient_end(self):
        previous = numerics.bound_for(128)
        for context in (2048, 4096, 16384, 65536, 131072, 262144):
            current = numerics.bound_for(context)
            for key in current:
                self.assertGreaterEqual(current[key], previous[key], (context, key))
            previous = current
        self.assertEqual(numerics.bound_for(None), numerics.bound_for(numerics.MAX_CONTEXT))

    def test_the_emulated_noise_stays_inside_it_and_a_wrong_merge_does_not(self):
        # the CPU emulation's flat-softmax figures (ksplit_emulation.py, S = 2, worst of 2 seeds; ESTIMATES): (context, max ulps, rms ulps, share above 2 ulps)
        for context, max_ulps, rms_ulps, share in ((1024, 3.0, 0.33, 0.0), (8192, 8.0, 0.78, 0.015), (32768, 12.0, 1.48, 0.116), (131072, 30.0, 2.84, 0.341)):
            stats = dict(nonfinite=0, max_ulps=max_ulps, rms_ulps=rms_ulps, frac_gt2=share)
            self.assertEqual(numerics.within_bound(stats, numerics.bound_for(context)), [], context)
        wrong = dict(nonfinite=0, max_ulps=120.0, rms_ulps=30.0, frac_gt2=0.9)
        for context in (1024, 32768, 131072, None):
            self.assertTrue(numerics.within_bound(wrong, numerics.bound_for(context)), context)

    def test_a_replaced_base_bound_is_scaled_not_ignored(self):
        scaled = numerics.bound_for(32768, bound={'max_ulps': 2.0, 'rms_ulps': 0.5, 'frac_gt2': 0.01})
        self.assertAlmostEqual(scaled['max_ulps'], 2.0 * 4.0)
        self.assertAlmostEqual(scaled['rms_ulps'], 0.5 * 4.0)


class SmokeRuleTests(unittest.TestCase):
    ENGAGED = '[PINDIAG] sdpa prefill ksplit engaged flags=0x1b rows=1024 heads=6/1 q_chunks=48 slices=2 cores=130 busy=96 chains=16'

    def test_a_lever_that_is_on_needs_an_engaged_line_for_the_rows_it_takes(self):
        env = {lever.FLAG: '1'}
        self.assertEqual(smoke.problems(env, self.ENGAGED), [])
        self.assertTrue(any('no engaged line' in problem for problem in smoke.problems(env, 'nothing')))
        self.assertTrue(any('rows the split does not take' in problem for problem in smoke.problems(env, self.ENGAGED.replace('rows=1024', 'rows=2048'))))

    def test_skipped_lines_are_information_and_a_fall_back_fails(self):
        env = {lever.FLAG: '1'}
        skipped = '%s rows=2048 reason=no second slice' % lever.SKIPPED_MARKER
        self.assertEqual(smoke.problems(env, self.ENGAGED + '\n' + skipped), [])
        fell = '%s rows=2048 reason=the call is not a oneq call (word 0x5efa0003)' % lever.FELL_BACK_MARKER
        self.assertTrue(any('fell back' in problem for problem in smoke.problems(env, self.ENGAGED + '\n' + fell)))

    def test_the_audit_needs_a_passing_line_and_a_mismatch_fails(self):
        env = {lever.FLAG: '1', lever.AUDIT_FLAG: '1'}
        passing = '[PINDIAG] sdpa prefill ksplit audit n=1 call=0 rows=1024 max_ulps=1.5 rms_ulps=0.1 frac_gt2=0.0 within_bound=True'
        self.assertEqual(smoke.problems(env, self.ENGAGED + '\n' + passing), [])
        self.assertTrue(any('never audited' in problem for problem in smoke.problems(env, self.ENGAGED)))
        bad = '[PINDIAG] sdpa prefill ksplit audit mismatch n=1 call=0 rows=1024 within_bound=False max_ulps 9'
        self.assertTrue(any('outside the waiver bound' in problem for problem in smoke.problems(env, self.ENGAGED + '\n' + passing + '\n' + bad)))
        self.assertTrue(any('never audited' in problem for problem in smoke.problems(env, self.ENGAGED + '\n[PINDIAG] sdpa prefill ksplit audit n=1 exact=True')),
                        'the generic exact=True line of the other levers is not this audit')

    def test_a_lever_that_is_off_logs_nothing(self):
        self.assertEqual(smoke.problems({}, 'unrelated'), [])
        self.assertTrue(smoke.problems({lever.FLAG: '0'}, self.ENGAGED))
        self.assertTrue(smoke.problems({}, '[PINDIAG] sdpa prefill ksplit audit n=1 within_bound=True'))

    def test_the_generic_fusion_rules_hold_the_same_markers(self):
        import c2_smoke_check
        levers = [row for row in c2_smoke_check.FUSION_LEVERS if row[0] == lever.FLAG]
        self.assertEqual(len(levers), 1)
        self.assertEqual(levers[0][1:4], ('1', lever.ENGAGED_MARKER, lever.FELL_BACK_MARKER))
        self.assertNotIn(lever.AUDIT_FLAG, [row[0] for row in c2_smoke_check.FUSION_AUDITS], 'the audit is judged by ksplit_smoke (within_bound), not by the exact=True table')
        self.assertIn('ksplit_smoke', c2_smoke_check.FUSION_RULES)
        env = {lever.FLAG: '1'}
        self.assertEqual(c2_smoke_check.fusion_problems(self.ENGAGED, env), [])
        self.assertTrue(c2_smoke_check.fusion_problems('nothing', env))
        self.assertEqual(c2_smoke_check.fusion_problems(self.ENGAGED + '\n%s rows=2048 reason=x' % lever.SKIPPED_MARKER, env), [])
        self.assertTrue(c2_smoke_check.fusion_problems(self.ENGAGED + '\n%s rows=2048 reason=x' % lever.FELL_BACK_MARKER, env))


if __name__ == '__main__':
    unittest.main()
