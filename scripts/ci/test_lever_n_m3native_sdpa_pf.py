"""Prefill lever #1 wired into the model gate: QWEN_FAST_SDPA_PF (lever_n_m3native_patch section I).

The G6 K/V chain of the chunked prefill SDPA (optimisation/ttnn-op/sdpa_prefill_chain, graft K64g)
reaches the model through three pieces, each tested here on CPU:

  graft   PATCHES['attention/tp.py'] = patch_attention_tp_full: the decode-side patch_attention_tp,
          then patch_attention_tp_sdpa_pf. Applied to the image's real attention/tp.py
          (fixtures/qwen36_attention_tp.py, sha256 e0c685a4: what every graft through v143 staged,
          and v144 on image A5 126b30df too - run 35917834242's graft artifact, tp.py.orig),
          the two compose (either order gives the same file, and removing the opt-in gives the
          decode graft back byte for byte), and the grafted forward_prefill_paged - executed against
          a recording fake ttnn - makes exactly the decode graft's ttnn calls with the flag off, and
          with it on differs only in the chain word, only on the calls card M qualified.
  arm     lever_n_m3native_run_arm.sh: M3NATIVE_SDPA_PF=1 -> -e QWEN_FAST_SDPA_PF=1 (+ the flags) and
          the graft's sdpa/ op directory mounted, the JIT cache keyed on that whole tree; refusals
          before the run, including a graft whose sdpa/ is not the run image's plus the chain reader
          alone (docker create / cp / diff -rq) or whose MANIFEST.sha256 does not verify, and the
          flags without the flag. Driven with bash, both the block alone and the whole script
          against a stub docker (the launched argv).
  gate    lever_n_m3native_gate.required_flag_markers: the graft's [PINDIAG] line and the factory's
          '[QWEN-SDPA-PF] flags=0x<flags> kv_chain=1 chains=16 members=96' line (spec 6.4; the
          topology tail whenever a prompt holds a 2048-token chunk) under the flag, every factory
          line with the requested flags and a card-M-qualified topology; no factory line without it.
"""

import ast
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import lever_n_m3native_gate as gate  # noqa: E402
import lever_n_m3native_patch as patcher  # noqa: E402

FIXTURE = HERE / 'fixtures' / 'qwen36_attention_tp.py'
FIXTURE_SHA256 = 'e0c685a43796f6f8a0ba42fd70a9533b502461b50fdda15e51c8753340f3dc3a'
CHAIN_DIR = ROOT / 'optimisation' / 'ttnn-op' / 'sdpa_prefill_chain'
ARM = HERE / 'lever_n_m3native_run_arm.sh'
SDPA_DIR = '/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa'
NL = chr(10)
BS = chr(92)


def fixture_text():
    return FIXTURE.read_text(encoding='utf-8')


def arm_text():
    return ARM.read_text(encoding='utf-8').replace(chr(13) + chr(10), NL)


# ---------------------------------------------------------------------------------------------
# The graft: composition on the real attention/tp.py
# ---------------------------------------------------------------------------------------------

class GraftCompositionTests(unittest.TestCase):
    def test_the_fixture_is_the_image_file_every_graft_staged(self):
        self.assertEqual(hashlib.sha256(FIXTURE.read_bytes()).hexdigest(), FIXTURE_SHA256)
        self.assertNotIn(chr(13), fixture_text())

    def test_the_table_maps_tp_py_to_the_composed_patch(self):
        self.assertIs(patcher.PATCHES['attention/tp.py'], patcher.patch_attention_tp_full)
        self.assertIs(patcher.SOURCES['attention/tp.py'][1], patcher.patch_attention_tp_full)
        self.assertEqual(patcher.SOURCES['attention/tp.py'][0], patcher.MODEL_ROOT)
        # tp_common.py is sha-pinned on the serving path: never grafted, by any table.
        self.assertNotIn('tp_common.py', patcher.with_lever_n())

    def test_it_composes_with_the_decode_graft_and_replaces_nothing(self):
        source = fixture_text()
        decode = patcher.patch_attention_tp(source)
        full = patcher.patch_attention_tp_full(source)
        self.assertEqual(patcher.unpatch_attention_tp_sdpa_pf(full), decode)          # the decode graft, intact
        self.assertEqual(full, patcher.patch_attention_tp(patcher.patch_attention_tp_sdpa_pf(source)))   # order-free
        self.assertEqual(patcher.unpatch_attention_tp_sdpa_pf(patcher.patch_attention_tp_sdpa_pf(source)), source)
        self.assertEqual(full.count('Lever N M3native'), decode.count('Lever N M3native'))
        self.assertEqual(full.count(patcher.SDPA_PF_CALL_OLD), 1)                      # the served statement kept
        ast.parse(full)

    def test_only_init_and_forward_prefill_paged_change_plus_the_helpers(self):
        decode = patcher.patch_attention_tp(fixture_text())
        full = patcher.patch_attention_tp_full(fixture_text())
        body = full[:-len(patcher.SDPA_PF_HELPERS)]
        self.assertTrue(full.endswith(patcher.SDPA_PF_HELPERS))
        changed = set()
        for name in ('__init__', '_qkv', '_wo_proj', 'forward_decode', 'forward_prefill', 'forward_prefill_paged',
                     '_decode_from_prep', 'set_paged_kv_cache', '_make_heads'):
            before = decode.splitlines(True)
            after = body.splitlines(True)
            b = ''.join(before[slice(*patcher.function_span(decode, name))])
            a = ''.join(after[slice(*patcher.function_span(body, name))])
            if a != b:
                changed.add(name)
        self.assertEqual(changed, {'__init__', 'forward_prefill_paged'})
        prefill = ''.join(body.splitlines(True)[slice(*patcher.function_span(body, 'forward_prefill_paged'))])
        self.assertIn(patcher.SDPA_PF_CALL_NEW, prefill)
        self.assertIn(patcher.SDPA_PF_INIT_NEW, body)

    def test_a_second_application_and_a_drifted_anchor_are_refused(self):
        full = patcher.patch_attention_tp_full(fixture_text())
        with self.assertRaisesRegex(ValueError, 'already grafted'):
            patcher.patch_attention_tp_sdpa_pf(full)
        drifted = fixture_text().replace('            k_chunk_size=qk_chunk,\n        )\n\n        # Pad page table',
                                         '            k_chunk_size=qk_chunk,\n            max_cores_per_head_batch=16,\n'
                                         '        )\n\n        # Pad page table')
        self.assertNotEqual(drifted, fixture_text())
        with self.assertRaisesRegex(ValueError, 'forward_prefill_paged sdpa prefill chain opt-in: expected one'):
            patcher.patch_attention_tp_full(drifted)
        with self.assertRaisesRegex(ValueError, 'helpers are not at the end'):
            patcher.unpatch_attention_tp_sdpa_pf(fixture_text())

    def test_the_constants_are_the_factorys_and_the_benchs(self):
        """scripts/ci may not import the chain directory at run time (this module must stay
        standalone: verify-t1-g0-rig.sh and the GDN card-M test copy it alone), so here."""
        if not (CHAIN_DIR / 'apply_factory_pf.py').is_file():
            self.skipTest('no sdpa_prefill_chain directory')
        sys.path.insert(0, str(CHAIN_DIR))
        try:
            import apply_factory_pf as factory
        finally:
            sys.path.remove(str(CHAIN_DIR))
        self.assertEqual(patcher.SDPA_PF_TAG, factory.PF_TAG)
        self.assertEqual(patcher.SDPA_PF_BINARY_MARKER, factory.LOG_MARKER)
        self.assertEqual(patcher.SDPA_PF_ROWS, factory.QUALIFIED_ROWS)
        self.assertEqual(patcher.SDPA_PF_CHUNK, factory.QUALIFIED_CHUNK)
        for flags in patcher.SDPA_PF_PRODUCTION_FLAGS:
            self.assertEqual(factory.decode_word(patcher.SDPA_PF_TAG | flags), (True, flags))
            self.assertEqual(flags & ~factory.PRODUCTION_FLAGS, 0)
        self.assertEqual(patcher.SDPA_PF_DEFAULT_FLAGS, factory.FLAG_KV_CHAIN | factory.FLAG_INJ_BATCH)
        for name in ('SDPA_PF_FLAG', 'SDPA_PF_FLAGS_FLAG', 'SDPA_PF_TAG', 'SDPA_PF_PRODUCTION_FLAGS',
                     'SDPA_PF_DEFAULT_FLAGS'):
            self.assertEqual(getattr(gate, name), getattr(patcher, name), name)
        self.assertEqual(gate.SDPA_PF_PINDIAG, patcher.MARKER_SDPA_PF)
        self.assertEqual(gate.SDPA_PF_FACTORY_MARKER, patcher.SDPA_PF_BINARY_MARKER)

    def test_no_new_env_read_outside_the_flag_names(self):
        helpers = (patcher.SDPA_PF_HELPERS + patcher.SDPA_PF_INIT_NEW[len(patcher.SDPA_PF_INIT_OLD):]
                   + patcher.SDPA_PF_CALL_NEW[len(patcher.SDPA_PF_CALL_OLD):])
        self.assertTrue(patcher.SDPA_PF_INIT_NEW.startswith(patcher.SDPA_PF_INIT_OLD))
        self.assertTrue(patcher.SDPA_PF_CALL_NEW.startswith(patcher.SDPA_PF_CALL_OLD))   # insertions only
        self.assertEqual(sorted(set(re.findall(r'"(QWEN_[A-Z0-9_]+)"', helpers))),
                         sorted({patcher.SDPA_PF_FLAG, patcher.SDPA_PF_FLAGS_FLAG}))
        self.assertNotIn('tp_common', helpers)


# ---------------------------------------------------------------------------------------------
# The graft executed: the real forward_prefill_paged against a recording fake ttnn
# ---------------------------------------------------------------------------------------------

class FakeTensor:
    def __init__(self, shape, ident):
        self.shape, self.ident = tuple(shape), ident

    def __repr__(self):
        return 'T%d%r' % (self.ident, self.shape)


class Config(dict):
    def __repr__(self):
        return 'SDPAProgramConfig(%s)' % ', '.join('%s=%r' % item for item in sorted(self.items()))


class Grid:
    def __repr__(self):
        return 'grid11x10'


class Mesh:
    def compute_with_storage_grid_size(self):
        return GRID

    def __repr__(self):
        return 'mesh'


GRID = Grid()


class Recorder:
    """Every ttnn call in order, as (path, args, kwargs) reprs; tensors are numbered per run. The
    program config each chunked SDPA call received is kept as a dict too."""

    def __init__(self):
        self.calls, self.count, self.sdpa_configs = [], 0, []

    def tensor(self, shape=(1, 1, 32, 32)):
        self.count += 1
        return FakeTensor(shape, self.count)

    def call(self, path, args, kwargs):
        self.calls.append((path, repr(args), repr(sorted(kwargs.items()))))
        if path == 'ttnn.SDPAProgramConfig':
            return Config(kwargs)
        if path == 'ttnn.transformer.chunked_scaled_dot_product_attention':
            self.sdpa_configs.append(dict(kwargs['program_config']))
        if path in ('ttnn.deallocate', 'ttnn.experimental.paged_fill_cache'):
            return None
        first = next((arg for arg in args if isinstance(arg, FakeTensor)), None)
        return self.tensor(first.shape if first is not None else (1, 1, 32, 32))


class FakeAttr:
    def __init__(self, recorder, path):
        self._recorder, self._path = recorder, path

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return FakeAttr(self._recorder, self._path + '.' + name)

    def __call__(self, *args, **kwargs):
        return self._recorder.call(self._path, args, kwargs)

    def __repr__(self):
        return self._path


def load_tp(source, recorder):
    """Execute attention/tp.py text with ttnn, torch and the model imports stubbed."""
    tpc = types.SimpleNamespace(COMPUTE_HIFI2='COMPUTE_HIFI2', TILE_SIZE=32)
    rope = types.SimpleNamespace(apply_partial_rope_decode=lambda *a, **k: a[0],
                                 apply_partial_rope_prefill=lambda t, *a, **k: recorder.call('rope_prefill', (t,) + a, k))
    ccl = types.SimpleNamespace(tt_all_reduce=lambda t, *a, **k: recorder.call('tt_all_reduce', (t,), k))
    package = types.SimpleNamespace(tp_common=tpc)
    modules = {'ttnn': FakeAttr(recorder, 'ttnn'), 'torch': types.SimpleNamespace(),
               'models': types.SimpleNamespace(), 'models.demos': types.SimpleNamespace(),
               'models.demos.blackhole': types.SimpleNamespace(), 'models.demos.blackhole.qwen36': types.SimpleNamespace(),
               'models.demos.blackhole.qwen36.tt': package, 'models.demos.blackhole.qwen36.tt.tp_common': tpc,
               'models.demos.blackhole.qwen36.tt.attention': types.SimpleNamespace(),
               'models.demos.blackhole.qwen36.tt.attention.rope_tp': rope,
               'models.tt_transformers': types.SimpleNamespace(), 'models.tt_transformers.tt': types.SimpleNamespace(),
               'models.tt_transformers.tt.ccl': ccl}
    namespace = {'__name__': 'qwen36_attention_tp_under_test'}
    with mock.patch.dict(sys.modules, modules):
        exec(compile(source, 'attention/tp.py', 'exec'), namespace)
    return namespace


ARGS = types.SimpleNamespace(max_batch_size=32, n_local_heads=12, n_local_kv_heads=2, head_dim=256, rope_head_dim=64,
                             ccl_topology=lambda: 'ring')


def run_prefill(source, environ, calls, *, layers=1, marker=True):
    """Construct `layers` TPAttention layers under `environ`, then run each (S, flexible, start) call
    through the first one. Returns ([(the ttnn call log, the SDPA program config) per prefill call],
    the graft's log lines, its binary probes, the layers)."""
    recorder = Recorder()
    namespace = load_tp(source, recorder)
    logged, probes = [], []
    if '_qwen_pf_log' in namespace:
        namespace['_qwen_pf_log'] = logged.append
        namespace['_qwen_pf_binary_has_marker'] = lambda: probes.append(1) or marker
    clean = {key: value for key, value in os.environ.items() if not key.startswith('QWEN_FAST_SDPA_PF')
             and key != 'QWEN_SDPA_BF8'}
    with mock.patch.dict(os.environ, dict(clean, **environ), clear=True):
        built = [namespace['TPAttention'](Mesh(), ARGS, {'q_norm': 'qn', 'k_norm': 'kn', 'wo': 'wo'}, 'ccl')
                 for _ in range(layers)]
    layer = built[0]
    layer.set_paged_kv_cache(recorder.tensor((4096, 2, 64, 256)), recorder.tensor((4096, 2, 64, 256)))
    layer._qkv = lambda x: (recorder.tensor(), recorder.tensor(), recorder.tensor())
    layer._make_heads = lambda qg, kp, vp, S: tuple(recorder.tensor((1, heads, S, 256)) for heads in (12, 1, 2, 2))
    layer._concat_heads = lambda attn: recorder.tensor()
    layer._wo_proj = lambda gated, weight: recorder.tensor()
    logs = []
    for rows, flexible, start in calls:
        recorder.calls, recorder.count, recorder.sdpa_configs = [], 10, []
        page_table = recorder.tensor((1, 32))
        start_tensor = recorder.tensor((1,)) if flexible else None
        layer.forward_prefill_paged(recorder.tensor((1, 1, rows, 5120)), 'cos', 'sin', page_table,
                                    chunk_start_idx=start, chunk_start_idx_tensor=start_tensor)
        assert len(recorder.sdpa_configs) == 1, recorder.sdpa_configs
        logs.append((list(recorder.calls), recorder.sdpa_configs[0]))
    return logs, logged, probes, built


# (S, flexible, chunk_start): flexible 2048 at 0 and mid-prompt (the page-table pad path), the legacy
# int path, the qualified 512 / 1024, and tails the chain was never swept at.
CALLS = ((2048, True, 0), (2048, True, 6144), (2048, False, 2048), (1024, True, 0), (512, True, 0),
         (1792, True, 2048), (256, True, 4096), (2048, False, 0))
ELIGIBLE = {(2048, True, 0), (2048, True, 6144), (1024, True, 0), (512, True, 0)}
BF8 = {'QWEN_SDPA_BF8': '1'}


SERVED = dict(compute_with_storage_grid_size=GRID, exp_approx_mode=False)


class GraftExecutionTests(unittest.TestCase):
    def setUp(self):
        self.decode = patcher.patch_attention_tp(fixture_text())
        self.full = patcher.patch_attention_tp_full(fixture_text())

    def test_flag_off_the_grafted_prefill_makes_exactly_the_decode_grafts_ttnn_calls(self):
        for environ in (BF8, {}, dict(BF8, QWEN_FAST_SDPA_PF='0'), dict(BF8, QWEN_FAST_SDPA_PF_FLAGS='0x7')):
            with self.subTest(environ=environ):
                before, _, _, _ = run_prefill(self.decode, environ, CALLS)
                after, logged, probes, built = run_prefill(self.full, environ, CALLS, layers=4)
                self.assertEqual(after, before)                        # every call, argument and config
                self.assertEqual((logged, probes), ([], []))           # no binary probe, no marker
                self.assertEqual([layer._sdpa_pf_word for layer in built], [None] * 4)
                for (rows, flexible, start), (_, config) in zip(CALLS, after):
                    self.assertNotIn('max_cores_per_head_batch', config)
                    if flexible:
                        self.assertEqual(config, dict(SERVED, q_chunk_size=128, k_chunk_size=128))

    def test_flag_on_only_the_qualified_calls_carry_the_word_and_nothing_else_moves(self):
        for flags, word in ((None, 0x5EFA0003), ('0x1', 0x5EFA0001), ('0x5', 0x5EFA0005), ('0x7', 0x5EFA0007)):
            environ = dict(BF8, QWEN_FAST_SDPA_PF='1')
            if flags is not None:
                environ['QWEN_FAST_SDPA_PF_FLAGS'] = flags
            with self.subTest(flags=flags):
                before, _, _, _ = run_prefill(self.decode, BF8, CALLS)
                after, logged, probes, built = run_prefill(self.full, environ, CALLS, layers=16)
                self.assertEqual(len(probes), 1)                       # one binary probe for 16 layers
                self.assertEqual(logged, ['[PINDIAG] sdpa prefill kvchain flags=%#x rows=512/1024/2048 chunk=128'
                                          % word])
                self.assertEqual({layer._sdpa_pf_word for layer in built}, {word})
                for call, (off, served), (on, sent) in zip(CALLS, before, after):
                    if call in ELIGIBLE:
                        self.assertEqual(sent, dict(served, max_cores_per_head_batch=word))
                        # One more config built (the served one, then the chain one); the rest identical.
                        configs = [entry for entry in on if entry[0] == 'ttnn.SDPAProgramConfig']
                        self.assertEqual(len(configs), 2)
                        self.assertEqual(configs[0], [entry for entry in off if entry[0] == 'ttnn.SDPAProgramConfig'][0])
                        skip = ('ttnn.SDPAProgramConfig', 'ttnn.transformer.chunked_scaled_dot_product_attention')
                        self.assertEqual([e for e in on if e[0] not in skip], [e for e in off if e[0] not in skip])
                    else:
                        self.assertEqual((on, sent), (off, served), call)   # a tail, the int path: served

    def test_bf16_mode_never_opts_in(self):
        environ = {'QWEN_FAST_SDPA_PF': '1'}                           # QWEN_SDPA_BF8 unset
        before, _, _, _ = run_prefill(self.decode, {}, CALLS)
        after, logged, _, _ = run_prefill(self.full, environ, CALLS)
        self.assertEqual(after, before)
        self.assertEqual(len(logged), 1)                               # the flag was read; no call qualified

    def test_a_stock_binary_or_a_bad_flag_set_refuses_to_construct(self):
        with self.assertRaisesRegex(RuntimeError, 'lacks the \\[QWEN-SDPA-PF\\] chain factory'):
            run_prefill(self.full, dict(BF8, QWEN_FAST_SDPA_PF='1'), CALLS, marker=False)
        for flags in ('0x2', '0x3f', '0x103', '0x203', 'x'):
            with self.subTest(flags=flags):
                with self.assertRaisesRegex(RuntimeError, 'QWEN_FAST_SDPA_PF_FLAGS must be a production flag set'):
                    run_prefill(self.full, dict(BF8, QWEN_FAST_SDPA_PF='1', QWEN_FAST_SDPA_PF_FLAGS=flags), CALLS)

    def test_the_marker_the_gate_requires_is_the_one_the_graft_logs(self):
        for flags in (None, '0x1', '0x7'):
            environ = dict(BF8, QWEN_FAST_SDPA_PF='1')
            if flags:
                environ['QWEN_FAST_SDPA_PF_FLAGS'] = flags
            with self.subTest(flags=flags):
                _, logged, _, _ = run_prefill(self.full, environ, ())
                pindiag = gate.required_flag_markers(environ, 4)['QWEN_FAST_SDPA_PF'][0]
                self.assertTrue(logged[0].startswith(pindiag), (logged, pindiag))


# ---------------------------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------------------------

def factory_line(flags, chains=16, members=96, order='raster'):
    """The K64g factory's F4 log line, from its own format string when the chain directory is here."""
    text = '"[QWEN-SDPA-PF] flags={:#x} kv_chain=1 chains={} members={} order={}"'
    if (CHAIN_DIR / 'apply_factory_pf.py').is_file():
        sys.path.insert(0, str(CHAIN_DIR))
        try:
            import apply_factory_pf as factory
            text = re.search(r'"\[QWEN-SDPA-PF\] flags=[^"]*"', factory.F4).group(0)
        finally:
            sys.path.remove(str(CHAIN_DIR))
    return '2026-09-24 | info | Op | ' + text.strip('"').format(flags, chains, members, order)


def pindiag_line(word):
    return 'INFO | [PINDIAG] sdpa prefill kvchain flags=%#x rows=512/1024/2048 chunk=128' % word


class GateMarkerTests(unittest.TestCase):
    ON = {'QWEN_FAST_SDPA_PF': '1'}

    FACTORY_2048 = '[QWEN-SDPA-PF] flags=0x3 kv_chain=1 chains=16 members=96 '

    def test_the_flag_requires_the_graft_line_and_the_factory_line_with_its_flags(self):
        """Spec 6.4's markers: the factory's line names the 2048-row topology (16 chains of 6) whenever
        a prompt holds a 2048-token chunk - every gate prompt (32768, 131072) does."""
        self.assertEqual(gate.required_flag_markers(self.ON, 4)['QWEN_FAST_SDPA_PF'],
                         ['[PINDIAG] sdpa prefill kvchain flags=0x5efa0003 ', self.FACTORY_2048])
        for prompt_tokens in (2048, 32768, 131072):
            self.assertEqual(gate.required_flag_markers(self.ON, 4, prompt_tokens)['QWEN_FAST_SDPA_PF'][1],
                             self.FACTORY_2048)
        self.assertEqual(gate.required_flag_markers(dict(self.ON, QWEN_FAST_SDPA_PF_FLAGS='0x7'), 1)['QWEN_FAST_SDPA_PF'],
                         ['[PINDIAG] sdpa prefill kvchain flags=0x5efa0007 ',
                          '[QWEN-SDPA-PF] flags=0x7 kv_chain=1 chains=16 members=96 '])
        # A prompt with no 2048-token chunk promises only the flags (its chunks are 512 or 1024 rows).
        self.assertEqual(gate.required_flag_markers(self.ON, 1, 1024)['QWEN_FAST_SDPA_PF'][1],
                         '[QWEN-SDPA-PF] flags=0x3 kv_chain=1 ')
        self.assertEqual(gate.SDPA_PF_TOPOLOGY[gate.PREFILL_CONV_CHUNK_TOKENS], (16, 96))
        self.assertNotIn('QWEN_FAST_SDPA_PF', gate.required_flag_markers({}, 4))
        self.assertNotIn('QWEN_FAST_SDPA_PF', gate.required_flag_markers({'QWEN_FAST_SDPA_PF': '0'}, 4))

    def test_the_topology_table_is_the_card_m_tests(self):
        path = CHAIN_DIR / 'test_sdpa_prefill_chain_card_m.py'
        if not path.is_file():
            self.skipTest('the chain directory is not here')
        tree = ast.parse(path.read_text(encoding='utf-8'))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'expected_chains')
        table = ast.literal_eval(next(node for node in ast.walk(function) if isinstance(node, ast.Dict)))
        self.assertEqual(table, gate.SDPA_PF_TOPOLOGY)

    def test_a_run_that_built_the_chain_passes_the_markers(self):
        log = NL.join((pindiag_line(0x5EFA0003), factory_line(0x3), factory_line(0x3, 8, 48)))
        report = gate.flag_marker_report(self.ON, 4, log, prompt_tokens=131072)
        self.assertEqual(report['missing'], [])
        self.assertTrue(all(report['found']['QWEN_FAST_SDPA_PF'].values()))
        self.assertEqual(gate.flag_marker_report(self.ON, 4, log)['missing'], [])

    def test_graft_mounted_is_not_graft_executed(self):
        """Only the Python line: the factory never built a chain program (a stock or pre-K64g .so, or
        no qualified call), so the flag did not do what it claims."""
        report = gate.flag_marker_report(self.ON, 4, pindiag_line(0x5EFA0003), prompt_tokens=32768)
        self.assertEqual(report['missing'], ['QWEN_FAST_SDPA_PF: ' + self.FACTORY_2048])
        self.assertFalse(gate.evaluate_gate(ready=True, users=4, checked=[dict(identical_prefix=True)] * 4,
                                            allow_missing_references=False, native_m3_marker_present=True,
                                            packed_phase=dict(rounds=1), binder_rounds=[{}],
                                            retired_binder_calls_nonzero=[], missing_markers=report['missing']))

    def test_only_the_short_chunks_built_the_chain(self):
        """Review: a 512- or 1024-row chain program alone must not pass a run whose 2048-row chunks
        (almost all of the saving) took the served path."""
        log = NL.join((pindiag_line(0x5EFA0003), factory_line(0x3, 4, 24), factory_line(0x3, 8, 48)))
        self.assertEqual(gate.flag_marker_report(self.ON, 1, log, prompt_tokens=131072)['missing'],
                         ['QWEN_FAST_SDPA_PF: ' + self.FACTORY_2048])
        self.assertEqual(gate.flag_marker_report(self.ON, 1, log, prompt_tokens=1536)['missing'], [])

    def test_an_unqualified_topology_fails_the_run(self):
        """A different grid or head count: the 2048-row line present but another line's (chains, members)
        is none card M qualified, or the 2048 chunks built 12 chains of 6."""
        log = NL.join((pindiag_line(0x5EFA0003), factory_line(0x3), factory_line(0x3, 12, 72)))
        self.assertEqual(gate.flag_marker_report(self.ON, 4, log, prompt_tokens=131072)['missing'],
                         ['QWEN_FAST_SDPA_PF: chains=12 members=72 is not a card-M-qualified topology '
                          '(512 rows 4/24, 1024 rows 8/48, 2048 rows 16/96)'])
        log = NL.join((pindiag_line(0x5EFA0003), factory_line(0x3, 12, 72)))
        self.assertEqual(len(gate.flag_marker_report(self.ON, 4, log, prompt_tokens=131072)['missing']), 2)

    def test_the_wrong_flags_in_the_factory_line_do_not_count(self):
        log = NL.join((pindiag_line(0x5EFA0003), factory_line(0x1)))
        self.assertEqual(gate.flag_marker_report(self.ON, 4, log)['missing'],
                         ['QWEN_FAST_SDPA_PF: ' + self.FACTORY_2048,
                          'QWEN_FAST_SDPA_PF: every chain program carries flags 0x3 (a factory line has 0x1)'])
        log = NL.join((pindiag_line(0x5EFA0003), factory_line(0x3), factory_line(0x1)))
        self.assertEqual(gate.flag_marker_report(self.ON, 4, log)['missing'],
                         ['QWEN_FAST_SDPA_PF: every chain program carries flags 0x3 (a factory line has 0x1)'])
        log = NL.join((pindiag_line(0x5EFA0007), factory_line(0x7, order='noc')))
        self.assertEqual(gate.flag_marker_report(dict(self.ON, QWEN_FAST_SDPA_PF_FLAGS='0x7'), 4, log)['missing'], [])

    def test_a_flag_set_outside_production_is_a_problem(self):
        report = gate.flag_marker_report(dict(self.ON, QWEN_FAST_SDPA_PF_FLAGS='0x103'), 4, factory_line(0x103))
        self.assertIn("QWEN_FAST_SDPA_PF_FLAGS: a production flag set (0x1, 0x3, 0x5, 0x7), not '0x103'",
                      report['missing'])

    def test_flag_off_no_chain_program_may_be_built(self):
        self.assertEqual(gate.flag_marker_report({}, 4, 'plain served log')['missing'], [])
        self.assertEqual(gate.flag_marker_report({}, 4, factory_line(0x3))['missing'],
                         ['QWEN_FAST_SDPA_PF unset: no chain program ([QWEN-SDPA-PF] flags= logged)'])

    def test_the_gate_reads_the_flags_the_arm_passes(self):
        text = arm_text()
        self.assertIn('-e QWEN_FAST_SDPA_PF=1 -e "QWEN_FAST_SDPA_PF_FLAGS=$sdpa_pf_flags"', text)
        self.assertIn('sdpa_pf_flags="${M3NATIVE_SDPA_PF_FLAGS:-0x%x}"' % gate.SDPA_PF_DEFAULT_FLAGS, text)
        self.assertIn('    0x1|0x3|0x5|0x7) ;;', text)
        self.assertEqual(gate.SDPA_PF_PRODUCTION_FLAGS, (0x1, 0x3, 0x5, 0x7))


# ---------------------------------------------------------------------------------------------
# The arm
# ---------------------------------------------------------------------------------------------

BASH = shutil.which('bash')
BLOCK_START = 'KM=""'
PF_START = 'sdpa_pf_env=()' + NL + 'if [ -n "${M3NATIVE_SDPA_PF:-}" ]; then'
PF_ARGV = '  "${sdpa_pf_env[@]}" \\' + NL


IMAGE_SDPA = {'device/sdpa_program_factory.cpp': b'factory fd8c0676', 'device/kernels/compute/sdpa.cpp': b'compute',
              'device/kernels/compute/compute_common.hpp': b'compute common',
              'device/kernels/dataflow/reader_interleaved.cpp': b'served reader',
              'device/kernels/dataflow/dataflow_common.hpp': b'dataflow common',
              'device/kernels/dataflow/chain_link.hpp': b'chain link',
              'device/kernels/dataflow/writer_interleaved.cpp': b'writer'}


def write_manifest(root):
    """build_k64g.sh's MANIFEST.sha256: sha256sum over './<path>' for every file, sorted."""
    root = Path(root)
    paths = sorted('./' + path.relative_to(root).as_posix() for path in root.rglob('*')
                   if path.is_file() and path.name != 'MANIFEST.sha256')
    (root / 'MANIFEST.sha256').write_bytes(''.join('%s  %s%s' % (hashlib.sha256((root / path[2:]).read_bytes()).hexdigest(),
                                                                  path, NL) for path in paths).encode('utf-8'))


def make_graft(root, *, reader=True, marker=True, decode=True, manifest=True):
    """A fake K64g-shaped graft directory: its sdpa/ is IMAGE_SDPA plus the chain reader."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / '_ttnncpp.so').write_bytes(b'\x7fELF..' + (b'[QWEN-SDPA-PF] flags=' if marker else b'[QWEN-SDPA] x') + b'..')
    (root / '_ttnn.so').write_bytes(b'so')
    if decode:
        for name, body in (('dataflow/reader_decode_qwen.cpp', b'reader'), ('compute/sdpa_flash_decode_qwen.cpp', b'compute')):
            path = root / 'sdpa_decode' / 'device' / 'kernels' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
    for relative, body in IMAGE_SDPA.items():
        path = root / 'sdpa' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    if reader:
        (root / 'sdpa' / 'device' / 'kernels' / 'dataflow' / 'reader_interleaved_qwen_chain.cpp').write_bytes(b'chain reader')
    if manifest:
        write_manifest(root)
    return root.as_posix()


def make_image_sdpa(root, changes=None):
    """The run image's sdpa/ as docker cp would copy it out: IMAGE_SDPA, with `changes` (None: delete)."""
    root = Path(root)
    files = dict(IMAGE_SDPA, **(changes or {}))
    for relative, body in files.items():
        if body is not None:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
    return root.as_posix()


def tree_key(directory):
    """The arm's cache key for the mounted sdpa/: sha256 over '<file sha256> ./<path>' lines, C order."""
    root = Path(directory)
    paths = sorted('./' + path.relative_to(root).as_posix() for path in root.rglob('*') if path.is_file())
    listing = ''.join('%s %s%s' % (hashlib.sha256((root / path[2:]).read_bytes()).hexdigest(), path, NL) for path in paths)
    return hashlib.sha256(listing.encode('utf-8')).hexdigest()[:12]


# A stub docker: `create` names a container (or fails under DOCKER_CREATE_FAIL=1) and records its argv,
# `cp` copies $FAKE_IMAGE_SDPA out as the image's sdpa/ and records its argv, `rm` succeeds, and `run`
# records the launched argv (NUL-separated) and prints an empty gate JSON block.
DOCKER_STUB = ('#!/usr/bin/env bash' + NL
               + 'case "$1" in' + NL
               + '  run) printf "%s' + BS + '0" "$@" > "$ARGV_OUT"; '
               + 'printf "<<<M3NATIVE_GATE_JSON_BEGIN>>>' + BS + 'n{}' + BS + 'n<<<M3NATIVE_GATE_JSON_END>>>' + BS + 'n" ;;' + NL
               + '  create) [ "${DOCKER_CREATE_FAIL:-}" = 1 ] && exit 1; '
               + 'printf "%s|" "$@" > "$DOCKER_LOG.create"; echo fakecid ;;' + NL
               + '  cp) printf "%s|" "$@" > "$DOCKER_LOG.cp"; cp -R "$FAKE_IMAGE_SDPA" "$3" ;;' + NL
               + 'esac' + NL
               + 'exit 0' + NL)


def write_stubs(directory, **scripts):
    stubs = Path(directory) / 'stubs'
    stubs.mkdir(parents=True, exist_ok=True)
    for name, body in scripts.items():
        (stubs / name).write_bytes(body.encode('utf-8'))
        (stubs / name).chmod(0o755)
    return stubs


def strip_pf(text):
    """The arm with this lever's block and argv line removed: the arm as it was before."""
    start = text.index('# M3NATIVE_SDPA_PF=1 (prefill lever #1')
    end = text.index(NL + 'fi' + NL, text.index(PF_START)) + len(NL + 'fi' + NL)
    text = text[:start] + text[end:]
    assert text.count(PF_ARGV) == 1
    return text.replace(PF_ARGV, '')


@unittest.skipUnless(BASH, 'bash not found')
class ArmBlockTests(unittest.TestCase):
    """The arm's KOPGRAFT64 block and this lever's block, executed by bash against fake grafts."""

    IMAGE = 'sha256:fakeimage'

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.stubs = write_stubs(self.tmp.name, docker=DOCKER_STUB)
        self.docker_log = (Path(self.tmp.name) / 'docker').as_posix()
        self.image_sdpa = make_image_sdpa(Path(self.tmp.name) / 'image' / 'sdpa')

    def run_blocks(self, environ, text=None):
        text = text if text is not None else arm_text()
        start = text.index(BLOCK_START)
        end = text.index(NL + 'fi' + NL, text.index(PF_START)) + len(NL + 'fi' + NL) if PF_START in text else \
            text.index(NL + 'fi' + NL, text.index('if [ -n "${KOPGRAFT64:-}" ]; then')) + len(NL + 'fi' + NL)
        script = ('set -euo pipefail' + NL + 'image=' + self.IMAGE + NL + 'sdpa_pf_env=()' + NL + text[start:end]
                  + 'printf "RESULT|%s|%s|%s" "$KM" "$kernel_cache" "${sdpa_pf_env[*]}"' + NL)
        env = dict(PATH=str(self.stubs) + os.pathsep + os.environ.get('PATH', ''), DOCKER_LOG=self.docker_log,
                   FAKE_IMAGE_SDPA=self.image_sdpa)
        for name in ('SYSTEMROOT', 'TEMP', 'TMP'):
            if name in os.environ:
                env[name] = os.environ[name]
        env.update(environ)
        try:
            result = subprocess.run([BASH, '-c', script], env=env, capture_output=True, text=True, timeout=60)
        except OSError as error:
            self.skipTest('bash unusable: %s' % error)
        if 'sha256sum' in result.stderr and 'not found' in result.stderr:
            self.skipTest('bash lacks coreutils here')
        return result

    def fields(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.split('RESULT|')[-1].split('|')

    def test_unset_the_blocks_give_exactly_what_the_arm_gave_before(self):
        graft = make_graft(Path(self.tmp.name) / 'k64g')
        for environ in ({}, dict(KOPGRAFT64=graft)):
            with self.subTest(environ=sorted(environ)):
                now = self.fields(self.run_blocks(environ))
                before = self.fields(self.run_blocks(environ, strip_pf(arm_text())))
                self.assertEqual(now, before)
                self.assertEqual(now[2], '')
                self.assertNotIn(SDPA_DIR + ':', now[0])
        self.assertFalse(Path(self.docker_log + '.create').exists())       # flag off: no image check either

    def test_set_the_sdpa_directory_the_env_and_a_tree_keyed_cache(self):
        graft = make_graft(Path(self.tmp.name) / 'k64g')
        before = self.fields(self.run_blocks(dict(KOPGRAFT64=graft)))
        result = self.run_blocks(dict(KOPGRAFT64=graft, M3NATIVE_SDPA_PF='1'))
        km, cache, env = self.fields(result)
        self.assertEqual(km, before[0] + ' -v %s/sdpa:%s:ro' % (graft, SDPA_DIR))
        self.assertEqual(cache, before[1] + '-pf-' + tree_key(Path(graft) / 'sdpa'))
        self.assertTrue(before[1].startswith('/experiment-cache/kernels-qwen-'))
        self.assertEqual(env, '-e QWEN_FAST_SDPA_PF=1 -e QWEN_FAST_SDPA_PF_FLAGS=0x3')
        self.assertIn("sdpa = image's sdpa/ + reader_interleaved_qwen_chain.cpp; manifest verified", result.stdout)
        # The image compared is THIS run's image, and its sdpa/ is what was copied out.
        self.assertEqual(Path(self.docker_log + '.create').read_text(encoding='utf-8'),
                         'create|--network|none|--entrypoint|true|%s|' % self.IMAGE)
        self.assertEqual(Path(self.docker_log + '.cp').read_text(encoding='utf-8').split('|')[:2],
                         ['cp', 'fakecid:' + SDPA_DIR])
        _, _, env = self.fields(self.run_blocks(dict(KOPGRAFT64=graft, M3NATIVE_SDPA_PF='1',
                                                     M3NATIVE_SDPA_PF_FLAGS='0x7')))
        self.assertEqual(env, '-e QWEN_FAST_SDPA_PF=1 -e QWEN_FAST_SDPA_PF_FLAGS=0x7')
        # No mount of this block reaches /experiment-scripts/ci.
        self.assertNotIn('/experiment-scripts', km[len(before[0]):])

    def test_the_cache_key_covers_every_file_of_the_mounted_tree(self):
        """Review: a later graft that changes a header the chain reader includes (or any served kernel)
        but not the reader itself must not reuse this graft's JIT binaries."""
        first = make_graft(Path(self.tmp.name) / 'first')
        _, cache, _ = self.fields(self.run_blocks(dict(KOPGRAFT64=first, M3NATIVE_SDPA_PF='1')))
        for relative in ('device/kernels/dataflow/dataflow_common.hpp', 'device/kernels/dataflow/chain_link.hpp',
                         'device/kernels/dataflow/reader_interleaved_qwen_chain.cpp'):
            with self.subTest(changed=relative):
                name = relative.replace('/', '_')
                other = make_graft(Path(self.tmp.name) / name)
                (Path(other) / 'sdpa' / relative).write_bytes(b'revised')
                write_manifest(other)
                changes = {} if relative.endswith('qwen_chain.cpp') else {relative: b'revised'}
                image = make_image_sdpa(Path(self.tmp.name) / ('image-' + name) / 'sdpa', changes)
                _, other_cache, _ = self.fields(self.run_blocks(dict(KOPGRAFT64=other, M3NATIVE_SDPA_PF='1',
                                                                     FAKE_IMAGE_SDPA=image)))
                self.assertNotEqual(other_cache, cache)
                self.assertEqual(other_cache.rsplit('-pf-', 1)[1], tree_key(Path(other) / 'sdpa'))

    def test_refusals_before_the_run(self):
        directory = self.tmp.name
        good = make_graft(Path(directory) / 'good')
        no_reader = make_graft(Path(directory) / 'no-reader', reader=False)
        stock = make_graft(Path(directory) / 'stock', marker=False)
        unlisted = make_graft(Path(directory) / 'unlisted', manifest=False)
        edited = make_graft(Path(directory) / 'edited')
        (Path(edited) / 'sdpa' / 'device' / 'kernels' / 'dataflow' / 'reader_interleaved_qwen_chain.cpp').write_bytes(b'edited')
        drifted = make_image_sdpa(Path(directory) / 'drifted' / 'sdpa',
                                  {'device/kernels/dataflow/dataflow_common.hpp': b'another image'})
        missing = make_image_sdpa(Path(directory) / 'missing' / 'sdpa', {'device/kernels/dataflow/chain_link.hpp': None})
        extra = make_image_sdpa(Path(directory) / 'extra' / 'sdpa', {'device/kernels/dataflow/new_in_image.hpp': b'x'})
        not_the_image = "sdpa is not image sha256:fakeimage's sdpa/ plus reader_interleaved_qwen_chain.cpp alone"
        cases = (
            (dict(M3NATIVE_SDPA_PF='1'), 'needs KOPGRAFT64'),
            (dict(KOPGRAFT64=no_reader, M3NATIVE_SDPA_PF='1'), 'lacks sdpa/device/kernels/dataflow/reader_interleaved_qwen_chain.cpp'),
            (dict(KOPGRAFT64=stock, M3NATIVE_SDPA_PF='1'), "_ttnncpp.so lacks '[QWEN-SDPA-PF] flags='"),
            (dict(KOPGRAFT64=good, M3NATIVE_SDPA_PF='0'), "M3NATIVE_SDPA_PF must be 1 or unset, got '0'"),
            (dict(KOPGRAFT64=good, M3NATIVE_SDPA_PF='yes'), 'must be 1 or unset'),
            (dict(KOPGRAFT64=unlisted, M3NATIVE_SDPA_PF='1'), 'MANIFEST.sha256 is missing or does not verify'),
            (dict(KOPGRAFT64=edited, M3NATIVE_SDPA_PF='1'), 'MANIFEST.sha256 is missing or does not verify'),
            (dict(KOPGRAFT64=good, M3NATIVE_SDPA_PF='1', FAKE_IMAGE_SDPA=drifted), not_the_image),
            (dict(KOPGRAFT64=good, M3NATIVE_SDPA_PF='1', FAKE_IMAGE_SDPA=missing), not_the_image),
            (dict(KOPGRAFT64=good, M3NATIVE_SDPA_PF='1', FAKE_IMAGE_SDPA=extra), not_the_image),
            (dict(KOPGRAFT64=good, M3NATIVE_SDPA_PF='1', DOCKER_CREATE_FAIL='1'), 'could not create a container of sha256:fakeimage'),
            (dict(KOPGRAFT64=good, M3NATIVE_SDPA_PF_FLAGS='0x3'),
             "M3NATIVE_SDPA_PF_FLAGS='0x3' without M3NATIVE_SDPA_PF=1 passes nothing"),
            (dict(M3NATIVE_SDPA_PF_FLAGS='0x7'), "M3NATIVE_SDPA_PF_FLAGS='0x7' without M3NATIVE_SDPA_PF=1 passes nothing"),
        ) + tuple((dict(KOPGRAFT64=good, M3NATIVE_SDPA_PF='1', M3NATIVE_SDPA_PF_FLAGS=flags),
                   "M3NATIVE_SDPA_PF_FLAGS must be a production flag set (0x1, 0x3, 0x5, 0x7), got '%s'" % flags)
                  for flags in ('0x2', '3', '0x103', '0x203', '0x0'))
        for environ, message in cases:
            with self.subTest(environ=environ):
                result = self.run_blocks(environ)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertNotIn('RESULT|', result.stdout)
        # The diff is printed with the refusal: what differs, not just that something does.
        result = self.run_blocks(dict(KOPGRAFT64=good, M3NATIVE_SDPA_PF='1', FAKE_IMAGE_SDPA=drifted))
        self.assertIn('dataflow_common.hpp', result.stderr)

    def test_the_argv_carries_the_env_before_the_entrypoint_and_one_sdpa_mount(self):
        text = arm_text()
        self.assertEqual(text.count(PF_ARGV), 1)
        self.assertLess(text.index('  $KM \\' + NL), text.index(PF_ARGV))
        self.assertLess(text.index(PF_ARGV), text.index('--entrypoint python3'))
        self.assertLess(text.index('if [ -n "${KOPGRAFT64:-}" ]; then'), text.index(PF_START))
        self.assertEqual(text.count(':/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa:ro'), 0)
        self.assertEqual(text.count(':$sdpa_pf_target:ro'), 1)                 # the one mount of the op dir
        self.assertEqual(text.count('sdpa_pf_target=' + SDPA_DIR + NL), 1)
        block = text[text.index(PF_START):text.index(NL + 'fi' + NL, text.index(PF_START))]
        self.assertNotIn('experiment-scripts', block)                 # never a mount over the evidence tree
        self.assertEqual([line.strip() for line in block.splitlines() if line.strip().startswith('KM=')],
                         ['KM="$KM -v $KOPGRAFT64/sdpa:$sdpa_pf_target:ro"'])
        self.assertIn('  sdpa_pf_target=%s%s' % (SDPA_DIR, NL), block)


@unittest.skipUnless(BASH, 'bash not found')
class ArmLaunchedArgvTests(unittest.TestCase):
    """The whole arm, run by bash with a stub docker and timeout: the launched argv (memory: read the
    launched argv). Flag off it is exactly the arm without this lever; flag on it differs by the
    sdpa mount, the two -e and the cache name, nothing else."""

    TIMEOUT = ('#!/usr/bin/env bash' + NL
               + 'while [ $# -gt 0 ]; do case "$1" in -k) shift 2 ;; -*) shift ;; *) break ;; esac; done' + NL
               + 'shift' + NL + 'exec "$@"' + NL)

    def launch(self, directory, environ, text):
        work = Path(directory) / 'work'                                # one $PWD: the mounts name it
        (work / 'draft-config').mkdir(parents=True, exist_ok=True)
        (work / 'draft-config' / 'config.json').write_text('{}', encoding='utf-8')
        stubs = write_stubs(directory, docker=DOCKER_STUB, timeout=self.TIMEOUT)
        image = Path(directory) / 'image' / 'sdpa'
        if not image.is_dir():
            make_image_sdpa(image)
        (work / 'arm.sh').write_bytes(text.encode('utf-8'))
        argv = work / 'argv.bin'
        if argv.exists():
            argv.unlink()
        env = dict(PATH=str(stubs) + os.pathsep + os.environ.get('PATH', ''), GITHUB_RUN_ID='1',
                   GITHUB_RUN_ATTEMPT='1', ARGV_OUT=argv.as_posix(), FAKE_IMAGE_SDPA=image.as_posix(),
                   DOCKER_LOG=(Path(directory) / 'docker').as_posix(), **environ)
        for name in ('SYSTEMROOT', 'TEMP', 'TMP'):
            if name in os.environ:
                env[name] = os.environ[name]
        try:
            result = subprocess.run([BASH, 'arm.sh'], cwd=str(work), env=env, capture_output=True, text=True,
                                    timeout=120)
        except OSError as error:
            self.skipTest('bash unusable: %s' % error)
        if not argv.is_file():
            self.skipTest('the stub docker did not run here: %s' % (result.stderr.strip()[-300:],))
        self.assertEqual(result.returncode, 0, result.stderr)
        return argv.read_bytes().decode('utf-8').split(chr(0))[:-1], result.stdout

    def test_flag_off_the_argv_is_the_arm_without_this_lever(self):
        with tempfile.TemporaryDirectory() as directory:
            graft = make_graft(Path(directory) / 'k64g')
            for environ in ({}, dict(KOPGRAFT64=graft)):
                with self.subTest(environ=sorted(environ)):
                    now, _ = self.launch(directory, environ, arm_text())
                    before, _ = self.launch(directory, environ, strip_pf(arm_text()))
                    self.assertEqual(now, before)
                    self.assertEqual(now[0], 'run')
                    self.assertFalse(any('QWEN_FAST_SDPA_PF' in arg for arg in now))

    def test_flag_on_the_argv_adds_the_mount_the_env_and_the_cache_key_only(self):
        with tempfile.TemporaryDirectory() as directory:
            graft = make_graft(Path(directory) / 'k64g')
            off, _ = self.launch(directory, dict(KOPGRAFT64=graft), arm_text())
            on, stdout = self.launch(directory, dict(KOPGRAFT64=graft, M3NATIVE_SDPA_PF='1'), arm_text())
            chain = tree_key(Path(graft) / 'sdpa')
            expected = []
            for index, arg in enumerate(off):
                if arg.startswith('TT_METAL_CACHE='):
                    arg += '-pf-' + chain
                expected.append(arg)
                if arg.endswith('/sdpa_decode:/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode:ro'):
                    expected += ['-v', '%s/sdpa:%s:ro' % (graft, SDPA_DIR),
                                 '-e', 'QWEN_FAST_SDPA_PF=1', '-e', 'QWEN_FAST_SDPA_PF_FLAGS=0x3']
            self.assertEqual(on, expected)
            self.assertLess(on.index('QWEN_FAST_SDPA_PF=1'), on.index('--entrypoint'))
            self.assertIn('SDPA prefill K/V chain (lever #1): sdpa op directory grafted from', stdout)
            # The image the arm compared is the one it launched.
            image = on[on.index('--entrypoint') + 2]
            self.assertEqual(Path(directory, 'docker.create').read_text(encoding='utf-8'),
                             'create|--network|none|--entrypoint|true|%s|' % image)

    def test_the_flags_alone_are_refused_before_any_docker_run(self):
        with tempfile.TemporaryDirectory() as directory:
            graft = make_graft(Path(directory) / 'k64g')
            work = Path(directory) / 'work'
            (work / 'draft-config').mkdir(parents=True)
            (work / 'draft-config' / 'config.json').write_text('{}', encoding='utf-8')
            stubs = write_stubs(directory, docker=DOCKER_STUB, timeout=self.TIMEOUT)
            (work / 'arm.sh').write_bytes(arm_text().encode('utf-8'))
            argv = work / 'argv.bin'
            env = dict(PATH=str(stubs) + os.pathsep + os.environ.get('PATH', ''), GITHUB_RUN_ID='1',
                       GITHUB_RUN_ATTEMPT='1', ARGV_OUT=argv.as_posix(), DOCKER_LOG=(Path(directory) / 'docker').as_posix(),
                       KOPGRAFT64=graft, M3NATIVE_SDPA_PF_FLAGS='0x3')
            for name in ('SYSTEMROOT', 'TEMP', 'TMP'):
                if name in os.environ:
                    env[name] = os.environ[name]
            try:
                result = subprocess.run([BASH, 'arm.sh'], cwd=str(work), env=env, capture_output=True, text=True,
                                        timeout=120)
            except OSError as error:
                self.skipTest('bash unusable: %s' % error)
            if 'without M3NATIVE_SDPA_PF=1' not in result.stderr and not argv.is_file():
                self.skipTest('the arm did not reach the block here: %s' % (result.stderr.strip()[-300:],))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('without M3NATIVE_SDPA_PF=1 passes nothing', result.stderr)
            self.assertFalse(argv.is_file())


class CiAllowlistTests(unittest.TestCase):
    def test_this_module_runs_in_the_cpu_suite(self):
        workflow = (ROOT / '.github' / 'workflows' / 'qwen-integration-cpu.yml').read_text(encoding='utf-8')
        self.assertIn('python -B -m unittest test_lever_n_m3native_sdpa_pf', workflow)


if __name__ == '__main__':
    unittest.main()
