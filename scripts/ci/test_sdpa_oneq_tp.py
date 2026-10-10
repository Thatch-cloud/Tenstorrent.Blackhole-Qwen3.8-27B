"""CPU tests of the oneq lever at the model (sdpa_pf_oneq_tp.py): the strict flags, the off path, the wrapper over the graft's own
_qwen_pf_program_config (taken from the committed graft source and run against a fake ttnn), the eligibility, the one-time binary check,
the audit's sampling and comparison, and the markers.

Run from scripts/ci:  python -B -m unittest test_sdpa_oneq_tp
"""

import ast
import contextlib
import io
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
CHAIN = ROOT / 'optimisation' / 'ttnn-op' / 'sdpa_prefill_chain'
ONEQ = ROOT / 'optimisation' / 'ttnn-op' / 'sdpa_prefill_oneq'
for path in (HERE, CHAIN, ONEQ):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import apply_factory_pf as pf  # noqa: E402
import apply_factory_ps as ps  # noqa: E402
import sdpa_pf_oneq_tp as lever  # noqa: E402

ENV_ON = {'QWEN_FAST_SDPA_PF': '1', 'QWEN_FAST_SDPA_PF_ONEQ': '1', 'QWEN_FAST_TP': '4'}


class Config:
    """ttnn.SDPAProgramConfig as the graft builds it: keyword fields."""

    def __init__(self, **fields):
        self.fields = fields

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
        for env in ({}, {'QWEN_FAST_SDPA_PF_ONEQ': '0'}, {'QWEN_FAST_SDPA_PF_ONEQ': '0', 'QWEN_FAST_SDPA_PF_ONEQ_AUDIT': '0'}):
            self.assertEqual(lever.settings(env), (False, False))
            self.assertFalse(lever.enabled(env))
            with mock.patch.object(lever.importlib, 'import_module') as imported:
                self.assertEqual(lever.install(env), [])
                imported.assert_not_called()

    def test_strict_values_and_pairings(self):
        for bad in ('2', 'true', '', 'yes', '01'):
            with self.assertRaises(ValueError, msg=bad):
                lever.settings(dict(ENV_ON, QWEN_FAST_SDPA_PF_ONEQ=bad))
        with self.assertRaisesRegex(ValueError, 'AUDIT needs QWEN_FAST_SDPA_PF_ONEQ=1'):
            lever.settings({'QWEN_FAST_SDPA_PF_ONEQ_AUDIT': '1'})
        with self.assertRaisesRegex(ValueError, 'must be 0 or 1'):
            lever.settings(dict(ENV_ON, QWEN_FAST_SDPA_PF_ONEQ_AUDIT='2'))
        self.assertEqual(lever.settings(ENV_ON), (True, False))
        self.assertEqual(lever.settings(dict(ENV_ON, QWEN_FAST_SDPA_PF_ONEQ_AUDIT='1')), (True, True))

    def test_the_lever_needs_the_chain_four_cards_and_a_production_flag_set(self):
        with self.assertRaisesRegex(ValueError, 'needs QWEN_FAST_SDPA_PF=1'):
            lever.settings({'QWEN_FAST_SDPA_PF_ONEQ': '1', 'QWEN_FAST_TP': '4'})
        with self.assertRaisesRegex(ValueError, 'four-card lever'):
            lever.settings({'QWEN_FAST_SDPA_PF_ONEQ': '1', 'QWEN_FAST_SDPA_PF': '1', 'QWEN_FAST_TP': '2'})
        for flags in ('0x1', '0x3', '0x5', '0x7', '3', ''):
            self.assertEqual(lever.settings(dict(ENV_ON, QWEN_FAST_SDPA_PF_FLAGS=flags))[0], True, flags)
        for flags in ('0x2', '0xB', '0x103', 'bogus'):
            with self.assertRaisesRegex(ValueError, 'production chain flag set'):
                lever.settings(dict(ENV_ON, QWEN_FAST_SDPA_PF_FLAGS=flags))

    def test_constants_equal_the_graft_and_the_factory_patch(self):
        fake = Fake()
        _fn, graft = graft_function(fake)
        self.assertEqual(tuple(graft['_QWEN_PF_ROWS']), lever.ROWS)
        self.assertEqual(tuple(graft['_QWEN_PF_ROWS']), tuple(pf.QUALIFIED_ROWS))
        self.assertEqual(graft['_QWEN_PF_CHUNK'], lever.Q_CHUNK)
        self.assertEqual(tuple(graft['_QWEN_PF_FLAGS']), lever.PRODUCTION_FLAGS)
        self.assertEqual(graft['_QWEN_PF_TAG'], lever.PF_TAG)
        self.assertEqual(graft['_QWEN_PF_DEFAULT_FLAGS'], lever.DEFAULT_FLAGS)
        self.assertEqual(lever.ONEQ_BIT, ps.FLAG_ONEQ)
        self.assertEqual(lever.PF_TAG, ps.PF_TAG)
        self.assertEqual(lever.BINARY_MARKER.decode(), ps.FATAL_TEXT.split(':')[0])
        self.assertIn(lever.BINARY_MARKER.decode(), ps.FATAL_TEXT)

    def test_the_markers_are_the_three_prefixes_of_one_smoke_marker(self):
        base = 'sdpa prefill oneq'
        self.assertEqual(lever.ENGAGED_MARKER, '[PINDIAG] %s engaged' % base)
        self.assertEqual(lever.FELL_BACK_MARKER, '[PINDIAG] %s fell back' % base)
        self.assertEqual(lever.AUDIT_MARKER, '[PINDIAG] %s audit' % base)
        self.assertTrue(lever.AUDIT_MISMATCH_MARKER.startswith(lever.AUDIT_MARKER))


class EligibilityTests(unittest.TestCase):
    def test_tp4_shapes_on_both_grids(self):
        for grid in ((13, 10), (11, 10)):
            for rows in lever.ROWS:
                self.assertIsNone(lever.eligibility(6, 1, rows, grid), (grid, rows))

    def test_what_the_factory_refuses_the_lever_does_not_ask_for(self):
        self.assertIn('exceed the 130 cores', lever.eligibility(12, 2, 2048, (13, 10)))      # TP2: 192 q chunks
        self.assertIn('exceed the 110 cores', lever.eligibility(12, 2, 2048, (11, 10)))
        self.assertIsNone(lever.eligibility(12, 2, 1024, (11, 10)))                           # 96 chunks fit
        self.assertIn('odd q chunk count', lever.eligibility(6, 1, 1920, (13, 10)))
        self.assertIn('not a whole number', lever.eligibility(6, 1, 2000, (13, 10)))
        self.assertIn('do not split', lever.eligibility(6, 4, 2048, (13, 10)))
        self.assertIn('exceed the 64 cores', lever.eligibility(6, 1, 2048, (8, 8)))

    def test_eligibility_is_the_planners_reading_of_the_factory(self):
        import oneq_planner as planner

        for nqh, nkh, rows, grid in itertools.product((4, 6, 8, 12), (1, 2), lever.ROWS, ((8, 8), (11, 10), (13, 10), (8, 12))):
            if nqh % nkh:
                continue
            the_plan = planner.plan(nqh, nkh, rows, 128, grid, oneq=True)
            self.assertEqual(lever.eligibility(nqh, nkh, rows, grid) is None, the_plan['refusal'] is None, (nqh, nkh, rows, grid))


@unittest.skipIf(torch is None, 'torch not installed')
class WrapperTests(unittest.TestCase):
    def setUp(self):
        lever.reset_state()
        self.fake = Fake()
        self.logged = []
        self.saved_ttnn = sys.modules.get('ttnn')
        sys.modules['ttnn'] = self.fake
        patch = mock.patch.object(lever, '_log', self.logged.append)
        patch.start()
        self.addCleanup(patch.stop)
        self.original, _ = graft_function(self.fake)
        self.served = Config(served=True)

    def tearDown(self):
        lever.reset_state()
        if self.saved_ttnn is None:
            sys.modules.pop('ttnn', None)
        else:
            sys.modules['ttnn'] = self.saved_ttnn

    def twin(self, layer, S, flexible=True, qk=128, binary=True):
        return lever.program_config(layer, self.served, qk, flexible, S, original=self.original, binary_check=lambda: binary)

    def test_a_chain_call_gets_the_oneq_bit_and_nothing_else_changes(self):
        layer = make_layer()
        for rows in lever.ROWS:
            chain = self.original(layer, self.served, 128, True, rows)
            mine = self.twin(layer, rows)
            self.assertIsNot(chain, self.served)
            self.assertEqual(mine.fields.pop('max_cores_per_head_batch'), 0x5EFA000B)
            self.assertEqual(chain.fields.pop('max_cores_per_head_batch'), 0x5EFA0003)
            self.assertEqual(mine.fields, chain.fields, 'only the word differs from the graft\'s own config')

    def test_every_flag_set_keeps_its_bits(self):
        for flags in lever.PRODUCTION_FLAGS:
            layer = make_layer(word=lever.PF_TAG | flags)
            self.assertEqual(self.twin(layer, 2048).fields['max_cores_per_head_batch'], lever.PF_TAG | flags | 0x8)

    def test_calls_the_graft_keeps_served_stay_served(self):
        layer = make_layer()
        self.assertIs(self.twin(layer, 4096), self.served)                    # not a qualified S
        self.assertIs(self.twin(layer, 1920), self.served)
        self.assertIs(self.twin(layer, 2048, flexible=False), self.served)    # legacy start
        self.assertIs(self.twin(layer, 2048, qk=256), self.served)            # q/k chunk 256
        self.assertIs(self.twin(make_layer(bf8=False), 2048), self.served)    # not bf8
        self.assertIs(self.twin(make_layer(word=None), 2048), self.served)    # QWEN_FAST_SDPA_PF off
        self.assertEqual(self.logged, [])

    def test_a_chain_call_the_factory_would_refuse_is_served_by_the_plain_chain(self):
        layer = make_layer(nh=12, nkv=2, grid=(13, 10))                       # TP2 head counts at 2048 rows: 192 q chunks on 130 cores
        chain = self.original(layer, self.served, 128, True, 2048)
        mine = self.twin(layer, 2048)
        self.assertEqual(mine.fields, chain.fields)
        self.assertEqual(mine.fields['max_cores_per_head_batch'], 0x5EFA0003)
        self.assertEqual(len(self.logged), 1)
        self.assertIn('%s rows=2048 reason=192 q chunks (12 heads x 16) exceed the 130 cores' % lever.FELL_BACK_MARKER, self.logged[0])
        self.twin(layer, 2048)
        self.assertEqual(len(self.logged), 1, 'once per rows and reason')
        # the same layer at 1024 rows fits the grid and engages
        self.assertEqual(self.twin(layer, 1024).fields['max_cores_per_head_batch'], 0x5EFA000B)

    def test_a_small_grid_falls_back(self):
        layer = make_layer(grid=(8, 8))
        self.assertEqual(self.twin(layer, 2048).fields['max_cores_per_head_batch'], 0x5EFA0003)
        self.assertEqual(self.twin(layer, 512).fields['max_cores_per_head_batch'], 0x5EFA000B)     # 24 chunks fit 64 cores
        self.assertEqual(self.twin(layer, 1024).fields['max_cores_per_head_batch'], 0x5EFA000B)    # 48 fit

    def test_engaged_is_logged_once_per_rows_with_the_topology(self):
        layer = make_layer()
        self.twin(layer, 2048)
        self.twin(layer, 2048)
        self.twin(layer, 1024)
        self.assertEqual(len(self.logged), 2)
        self.assertEqual(self.logged[0], '[PINDIAG] sdpa prefill oneq engaged flags=0xb rows=2048 heads=6/1 q_chunks=96 cores=130 chunks_per_core=1 chains=16')
        self.assertIn('rows=1024 heads=6/1 q_chunks=48 cores=130 chunks_per_core=1 chains=8', self.logged[1])
        self.assertEqual(lever._STATE['engaged'], 3)

    def test_the_binary_check_runs_once_and_refuses_a_binary_without_the_edits(self):
        layer = make_layer()
        calls = []
        lever.program_config(layer, self.served, 128, True, 2048, original=self.original, binary_check=lambda: calls.append(1) or True)
        lever.program_config(layer, self.served, 128, True, 1024, original=self.original, binary_check=lambda: calls.append(1) or True)
        self.assertEqual(calls, [1])
        lever.reset_state()
        with self.assertRaisesRegex(RuntimeError, 'lacks the oneq edits.*K64j-OQ'):
            self.twin(layer, 2048, binary=False)

    def test_binary_has_oneq_reads_the_one_mapped_library(self):
        with tempfile.TemporaryDirectory() as directory:
            so = Path(directory) / '_ttnncpp.so'
            so.write_bytes(b'\x7fELF\x00[QWEN-SDPA-PF] flags=\x00')
            maps = Path(directory) / 'maps'
            maps.write_text('7f00 r-xp 0 00:00 0 %s\n' % so)
            self.assertFalse(lever.binary_has_oneq(str(maps)))
            so.write_bytes(so.read_bytes() + lever.BINARY_MARKER + b' {} q chunks on {} cores\x00')
            self.assertTrue(lever.binary_has_oneq(str(maps)))
            other = Path(directory) / 'lib' / '_ttnncpp.so'
            other.parent.mkdir()
            other.write_bytes(b'x')
            maps.write_text('7f00 r-xp 0 00:00 0 %s\n7f01 r-xp 0 00:00 0 %s\n' % (so, other))
            with self.assertRaises(RuntimeError):
                lever.binary_has_oneq(str(maps))

    def test_without_install_the_wrapper_refuses_to_guess_the_graft_function(self):
        with self.assertRaisesRegex(RuntimeError, 'was not captured'):
            lever.program_config(make_layer(), self.served, 128, True, 2048)


@unittest.skipIf(torch is None, 'torch not installed')
class InstallTests(unittest.TestCase):
    def setUp(self):
        lever.reset_state()
        self.fake = Fake()
        self.saved = {name: sys.modules.get(name) for name in ('ttnn', lever.GRAFT_MODULE)}
        sys.modules['ttnn'] = self.fake
        self.original, _ = graft_function(self.fake)
        self.module = types.ModuleType(lever.GRAFT_MODULE)
        self.module._qwen_pf_program_config = self.original
        self.holder = types.ModuleType('some_other_holder')
        self.holder._qwen_pf_program_config = self.original
        self.bystander = types.ModuleType('bystander')
        self.bystander._qwen_pf_program_config = lambda *a: None
        for module in (self.module, self.holder, self.bystander):
            sys.modules[module.__name__] = module

    def tearDown(self):
        lever.reset_state()
        for name in ('some_other_holder', 'bystander'):
            sys.modules.pop(name, None)
        for name, module in self.saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_install_rebinds_every_holder_of_the_graft_function_and_only_those(self):
        rebound = lever.install(ENV_ON)
        self.assertEqual({id(namespace) for namespace, _name, _old in rebound}, {id(self.module.__dict__), id(self.holder.__dict__)})
        self.assertTrue(all(old is self.original for _n, _name, old in rebound))
        self.assertTrue(self.module._qwen_pf_program_config._oneq_wrapper)
        self.assertIs(self.holder._qwen_pf_program_config, self.module._qwen_pf_program_config)
        self.assertIsNot(self.bystander._qwen_pf_program_config, self.module._qwen_pf_program_config)
        self.assertIs(lever._STATE['original'], self.original)
        # undo as tp_addresses.uninstall does
        for namespace, name, old in rebound:
            namespace[name] = old
        self.assertIs(self.module._qwen_pf_program_config, self.original)

    def test_install_twice_binds_nothing_new(self):
        self.assertEqual(len(lever.install(ENV_ON)), 2)
        self.assertEqual(lever.install(ENV_ON), [])

    def test_the_bound_wrapper_has_the_graft_functions_signature_and_serves_through_it(self):
        lever.install(ENV_ON)
        layer = make_layer()
        with mock.patch.object(lever, '_log', lambda text: None), mock.patch.object(lever, 'binary_has_oneq', lambda *a: True):
            served = Config(served=True)
            self.assertIs(self.module._qwen_pf_program_config(layer, served, 128, True, 4096), served)
            self.assertEqual(self.module._qwen_pf_program_config(layer, served, 128, True, 2048).fields['max_cores_per_head_batch'], 0x5EFA000B)

    def test_install_refuses_a_graft_without_the_prefill_opt_in(self):
        del self.module._qwen_pf_program_config
        with self.assertRaisesRegex(RuntimeError, 'has no _qwen_pf_program_config'):
            lever.install(ENV_ON)

    def test_a_malformed_setting_fails_at_attach(self):
        with self.assertRaises(ValueError):
            lever.install(dict(ENV_ON, QWEN_FAST_SDPA_PF_ONEQ='maybe'))
        self.assertIs(self.module._qwen_pf_program_config, self.original)


class TpAddressesHookTests(unittest.TestCase):
    """tp_addresses.install() is where the lever is bound at four cards: it imports sdpa_pf_oneq_tp and calls its install(environ) only when one of the two
    flags is SET (any value, so a malformed one reaches the strict parser); the entries it rebinds are put back by tp_addresses.uninstall()."""

    FOUR = {'QWEN_FAST_TP': '4', 'QWEN_FAST_SDPA_PF': '1'}

    def setUp(self):
        InstallTests.setUp(self)                      # the fake ttnn and the fake graft module, its holder and a bystander
        self.addCleanup(InstallTests.tearDown, self)
        import tp_addresses
        self.tp_addresses = tp_addresses
        self.addCleanup(tp_addresses.uninstall)

    def test_flag_unset_or_zero_leaves_the_graft_function_and_never_asks_the_lever(self):
        for extra in ({}, {'QWEN_FAST_SDPA_PF_ONEQ': '0'}):
            with self.subTest(extra=extra):
                self.tp_addresses.uninstall()
                lever.reset_state()
                self.tp_addresses.install(dict(self.FOUR, **extra))
                self.assertIs(self.module._qwen_pf_program_config, self.original)
                self.assertIs(self.holder._qwen_pf_program_config, self.original)
        self.tp_addresses.uninstall()
        with mock.patch.object(lever, 'install', side_effect=AssertionError('asked')) as asked:
            self.tp_addresses.install(dict(self.FOUR))
        asked.assert_not_called()

    def test_flag_on_binds_the_wrapper_and_uninstall_puts_the_graft_function_back(self):
        self.tp_addresses.install(dict(self.FOUR, QWEN_FAST_SDPA_PF_ONEQ='1'))
        self.assertTrue(self.module._qwen_pf_program_config._oneq_wrapper)
        self.assertIs(self.holder._qwen_pf_program_config, self.module._qwen_pf_program_config)
        self.assertIsNot(self.bystander._qwen_pf_program_config, self.module._qwen_pf_program_config)
        self.tp_addresses.uninstall()
        self.assertIs(self.module._qwen_pf_program_config, self.original)
        self.assertIs(self.holder._qwen_pf_program_config, self.original)

    def test_the_lever_gets_the_environment_install_was_given(self):
        given = dict(self.FOUR, QWEN_FAST_SDPA_PF_ONEQ='1', QWEN_FAST_SDPA_PF_ONEQ_AUDIT='1')
        with mock.patch.object(lever, 'install', return_value=[]) as called:
            self.tp_addresses.install(dict(given))
        called.assert_called_once_with(given)

    def test_a_malformed_or_orphan_flag_reaches_the_strict_parser_and_fails_at_attach(self):
        for extra in ({'QWEN_FAST_SDPA_PF_ONEQ': 'yes'}, {'QWEN_FAST_SDPA_PF_ONEQ_AUDIT': '1'}):
            with self.subTest(extra=extra):
                with self.assertRaises(ValueError):
                    self.tp_addresses.install(dict(self.FOUR, **extra))
                self.assertIs(self.module._qwen_pf_program_config, self.original)
                self.tp_addresses.uninstall()

    def test_the_pair_never_reaches_it(self):
        with mock.patch.object(lever, 'install', side_effect=AssertionError('asked')):
            with self.assertRaises(ValueError):
                self.tp_addresses.install({'QWEN_FAST_TP': '2', 'QWEN_FAST_SDPA_PF_ONEQ': '1', 'QWEN_FAST_SDPA_PF': '1'})
        self.assertIs(self.module._qwen_pf_program_config, self.original)


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
        self.original, _ = graft_function(self.fake)
        self.served = Config(served=True)
        lever._STATE.update(original=self.original, audit=True, binary_ok=True)
        self.mesh = object()
        self.fake.ConcatMeshToTensor = lambda mesh, dim: ('compose', mesh, dim)
        self.fake.to_torch = lambda tensor, mesh_composer=None: tensor

    def tearDown(self):
        lever.reset_state()
        if self.saved_ttnn is None:
            sys.modules.pop('ttnn', None)
        else:
            sys.modules['ttnn'] = self.saved_ttnn

    def engage(self, rows=2048):
        layer = make_layer()
        layer.mesh = types.SimpleNamespace(compute_with_storage_grid_size=lambda: Grid(13, 10))
        return lever.program_config(layer, self.served, 128, True, rows, original=self.original)

    def make_audited(self, mine, reference):
        config = self.engage()
        outputs = {0x5EFA000B: mine, 0x5EFA0003: reference}
        calls = []

        def op(**kwargs):
            calls.append(kwargs['program_config'].fields['max_cores_per_head_batch'])
            return outputs[kwargs['program_config'].fields['max_cores_per_head_batch']]

        self.fake.transformer.chunked_scaled_dot_product_attention = op
        lever._STATE['op'] = None
        lever._install_audit(self.fake)
        return config, self.fake.transformer.chunked_scaled_dot_product_attention, calls

    def test_equal_outputs_pass_and_both_programs_ran(self):
        a = torch.randn(1, 2, 4, 4).to(torch.bfloat16)
        config, audited, calls = self.make_audited(a, a.clone())
        out = audited(input_tensor_q=None, program_config=config)
        self.assertIs(out, a)
        self.assertEqual(calls, [0x5EFA000B, 0x5EFA0003])
        self.assertEqual(len(self.fake.deallocated), 1)

    def test_a_single_flipped_bit_is_a_mismatch_that_raises(self):
        a = torch.randn(1, 2, 4, 4).to(torch.bfloat16)
        b = a.clone()
        b.view(torch.int16)[0, 0, 0, 0] ^= 1
        config, audited, _calls = self.make_audited(a, b)
        with self.assertRaisesRegex(RuntimeError, 'oneq SDPA output differs'):
            audited(input_tensor_q=None, program_config=config)
        self.assertTrue(any(line.startswith(lever.AUDIT_MISMATCH_MARKER) and '1 of 32 elements differ' in line for line in self.logged), self.logged)

    def test_negative_zero_and_zero_differ_as_bit_patterns(self):
        a = torch.zeros(1, 1, 2, 2, dtype=torch.bfloat16)
        b = a.clone()
        b[0, 0, 0, 0] = -0.0
        self.assertTrue(torch.equal(a, b))                       # value-equal, bit-different: the audit must not use ==
        config, audited, _calls = self.make_audited(a, b)
        with self.assertRaises(RuntimeError):
            audited(input_tensor_q=None, program_config=config)

    def test_a_call_that_is_not_the_oneq_one_is_not_audited(self):
        a = torch.zeros(1, 1, 2, 2, dtype=torch.bfloat16)
        config, audited, calls = self.make_audited(a, a.clone())
        other = Config(max_cores_per_head_batch=0x5EFA0003)
        audited(input_tensor_q=None, program_config=other)
        self.assertEqual(calls, [0x5EFA0003])
        self.assertEqual(lever._STATE['audited'], 0)

    def test_sampling_is_the_first_eight_then_every_24th(self):
        a = torch.zeros(1, 1, 2, 2, dtype=torch.bfloat16)
        config, audited, calls = self.make_audited(a, a.clone())
        for _ in range(8 + 24 * 3 + 5):
            audited(input_tensor_q=None, program_config=config)
        audited_calls = [i for i in range(8 + 24 * 3 + 5) if i < 8 or (i - 8) % 24 == 0]
        self.assertEqual(lever._STATE['audited'], len(audited_calls))
        self.assertEqual(len(audited_calls), 8 + 4)
        self.assertEqual(calls.count(0x5EFA0003), 12)
        lines = [line for line in self.logged if line.startswith(lever.AUDIT_MARKER)]
        self.assertEqual(len(lines), 8)             # audits 1-8 are logged, 9-12 are not (the next logged one is the 16th)

    def test_the_audit_is_off_without_its_flag(self):
        lever._STATE['audit'] = False
        self.engage()
        self.assertIsNone(lever._STATE['pair'])
        self.assertIsNone(lever._STATE['op'])


if __name__ == '__main__':
    unittest.main()
