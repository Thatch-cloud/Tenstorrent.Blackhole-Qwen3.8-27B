"""CPU checks for the V5 byte gate (gdn_v5_card_m.py, run_card_m.sh); no device, no ttnn.

  - the pure helpers: the locators (output, snapshot: token, head, tile, owner or helper), the host inputs per regime (shapes,
    determinism, the column-asymmetric RA, R3's specials and zero rows, R3b's poison, R1 equal to gdn_tp4_card_test's draws), the
    timing labels, the scope accounting, the verdict (FAIL outranks NO-DECISION, PASS only at full scope) and the verdict line;
  - the real-input maths (R5): the causal convolution against a direct loop and the CPU recurrence against a step by step
    reference, on small tensors;
  - run_card_m.sh (needs bash): the canonical qual_card block, the mounts (K5-A and V5 side by side, the probe), the launched
    environment (QWEN_FAST_TP=4, QWEN_FAST_VERIFY_T1=1), the watcher mode, the hang hint;
  - the whole flow on a fake ttnn with a page-level memory model: the sections run end to end against the REAL builders
    (gdn_seq_block.execute and gdn_seq_block_split.execute build their programs on the fake), the fake launch writes a
    deterministic function of the input pages, and each broken variant is caught by the section that must catch it: a V that
    differs on one byte (FAIL), a blind control (FAIL), a V that is wrong only in a traced replay (FAIL), a V that moves an input
    (FAIL), a wrong program-cache delta (FAIL), an unwritten page (FAIL), a section that raises (NO-DECISION), reduced scope
    (NO-DECISION), and the watchdog (exit 3, the partial report written). The device half runs on the rig only.
"""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
CI = ROOT / 'scripts' / 'ci'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CI))

import torch  # noqa: E402

import gdn_multitoken as native  # noqa: E402
import gdn_seq_block as seq  # noqa: E402
import gdn_seq_block_device_test as dev  # noqa: E402
import gdn_seq_block_split as split  # noqa: E402
import gdn_tp4_card_test as card_test  # noqa: E402
import gdn_v5_card_m as probe  # noqa: E402
import tp_shapes  # noqa: E402
import verify_trace_t1  # noqa: E402
from test_gdn_seq_block import SyntheticRoot  # noqa: E402
from test_gdn_user_batch import FakeTTNN  # noqa: E402

SCRIPT = HERE / 'run_card_m.sh'
FOUR = {'QWEN_FAST_TP': '4', 'QWEN_FAST_VERIFY_T1': '1'}


def found():
    with mock.patch.dict(os.environ, FOUR):
        return tp_shapes.geometry(4)


def script_text():
    return SCRIPT.read_text(encoding='utf-8').replace(chr(13) + chr(10), chr(10))


# ---- the pure helpers ----

class LocatorTests(unittest.TestCase):
    def test_output_locator_names_token_head_tile_and_the_half(self):
        self.assertEqual(probe.locate_output(0)['half'], 'owner')
        first = probe.locate_output(3 * 1536 + 128 * 5 + 64 + 7)
        self.assertEqual((first['token'], first['head'], first['tile'], first['half'], first['element']),
                         (3, 5, 2, 'helper', [3, 7]))
        padded = probe.locate_output(17 * 1536)
        self.assertTrue(padded['padding_row'])
        self.assertIsNone(padded['token'])

    def test_snapshot_locator_names_the_tile_pair_and_the_half(self):
        index = (5 * 12 + 7) * 128 * 128 + (96 + 3) * 128 + 70
        where = probe.locate_states(index)
        self.assertEqual((where['token'], where['head'], where['tile_i'], where['tile_j'], where['half']),
                         (5, 7, 3, 2, 'helper'))
        self.assertEqual(where['element'], [3, 6])
        low = probe.locate_states(40)
        self.assertEqual((low['tile_j'], low['half']), (1, 'owner'))


class InputTests(unittest.TestCase):
    def test_shapes_and_determinism_at_the_four_card_geometry(self):
        geometry = found()
        for regime in ('R1', 'R2', 'R3', 'R3b', 'RA'):
            a = probe.host_inputs(torch, regime, 17, 4, geometry)
            b = probe.host_inputs(torch, regime, 17, 4, geometry)
            norm_w, users, poison = a
            self.assertEqual(tuple(norm_w.shape), (1, 1, 128))
            self.assertEqual(len(users), 4)
            for values, other in zip(users, b[1]):
                self.assertEqual(tuple(values['qkv'].shape), (1, 16, 2560), regime)
                self.assertEqual(tuple(values['beta'].shape), (1, 16, 12))
                self.assertEqual(tuple(values['gate'].shape), (1, 16, 12))
                self.assertEqual(tuple(values['initial'].shape), (1, 12, 128, 128))
                self.assertEqual(tuple(values['z'].shape), (1, 16, 1536))
                for name in values:
                    self.assertEqual(values[name].dtype, torch.bfloat16)
                    self.assertTrue(torch.equal(values[name].view(torch.int16), other[name].view(torch.int16)))
        with self.assertRaises(ValueError):
            probe.host_inputs(torch, 'R9', 1, 1, found())

    def test_r1_is_the_tp4_card_tests_draws(self):
        geometry = found()
        generator = torch.Generator().manual_seed(17)
        torch.randn(1, 1, 128, generator=generator)
        expected = card_test.user_inputs(torch, geometry, generator)
        _, users, _ = probe.host_inputs(torch, 'R1', 17, 1, geometry)
        for name, value in zip(('qkv', 'beta', 'gate', 'initial', 'z'), expected):
            self.assertTrue(torch.equal(users[0][name].view(torch.int16), value.view(torch.int16)), name)

    def test_ra_makes_the_helper_columns_dominate_the_norm(self):
        geometry = found()
        _, users, _ = probe.host_inputs(torch, 'RA', 17, 1, geometry)
        v = users[0]['qkv'][..., 1024:].float().reshape(16, 12, 128)
        low, high = v[..., :64].abs().mean(), v[..., 64:].abs().mean()
        self.assertGreater(float(high / low), 100.0)
        initial = users[0]['initial'].float()
        self.assertGreater(float(initial[..., 64:].abs().mean() / initial[..., :64].abs().mean()), 100.0)
        _, plain, _ = probe.host_inputs(torch, 'R1', 17, 1, geometry)
        self.assertTrue(torch.equal(plain[0]['beta'], users[0]['beta']))

    def test_r3_has_its_specials_and_zero_rows_and_r3b_its_poison(self):
        geometry = found()
        _, users, _ = probe.host_inputs(torch, 'R3', 17, 1, geometry)
        values = users[0]
        self.assertTrue(bool((values['qkv'][0, 3, 512:1024] == 0).all()))
        self.assertTrue(bool((values['qkv'][0, 7, 0:512] == 0).all()))
        self.assertEqual(set(values['beta'].float().unique().tolist()), {0.0, float(torch.tensor(1 - 2.0 ** -8).bfloat16())})
        self.assertEqual(set(values['gate'].float().unique().tolist()), {0.0, -88.0})
        bits = values['initial'].view(torch.int16).reshape(-1)
        self.assertTrue(bool((bits == 0x0000).any()) and bool((bits == -0x8000).any()))
        _, groups, poison = probe.host_inputs(torch, 'R3b', 17, 4, geometry)
        self.assertTrue(torch.isnan(torch.tensor(poison[0])) and torch.isnan(torch.tensor(poison[1])))
        self.assertEqual((poison[2], poison[3]), (float('inf'), float('-inf')))


class VerdictTests(unittest.TestCase):
    def arguments(self, **overrides):
        values = dict(sections=list(probe.SECTIONS), regimes=list(probe.REGIMES), seeds=list(probe.PLAN_R1_SEEDS),
                      r4_launches=128, r5_layers=list(probe.PLAN_R5_LAYERS), users=4, trace_replays=100,
                      timing_launches=48, timing_rounds=25)
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_the_plan_is_full_scope(self):
        self.assertEqual(probe.scope_missing(self.arguments()), [])

    def test_each_reduction_is_named(self):
        for overrides, text in ((dict(sections=['p0']), 'section trace'), (dict(regimes=['R1']), 'regime R5'),
                                (dict(seeds=[17]), 'R1 seed 23'), (dict(r4_launches=3), 'R4 launches 3'),
                                (dict(r5_layers=[0]), 'R5 layer 23'), (dict(users=2), 'users 2'),
                                (dict(trace_replays=5), 'trace replays 5'), (dict(timing_rounds=2), 'timing')):
            with self.subTest(text=text):
                self.assertIn(text, ' | '.join(probe.scope_missing(self.arguments(**overrides))))

    def good(self):
        tally = dict(cases=10, exact_cases=10, differing=0, differing_bytes=0, unmeasured=0, first_failure=None)
        control = dict(cases=2, exact_cases=0, differing=5, differing_bytes=5, unmeasured=0, first_failure=None)
        return dict(a_qualified=True, tallies=dict(V=tally, N5r=control, N5x=control), unwritten=[], inputs_moved=[],
                    missing_scope=[], sections=dict(
                        negative=dict(error=None, N5r=dict(detects=True), N5x=dict(detects=True)),
                        stale=dict(error=None, exact=True), trace=dict(error=None, exact=True, unwritten=[]),
                        cache=dict(error=None, ok=True, problems=[]), p0=dict(error=None)))

    def test_pass_only_when_nothing_is_missing(self):
        self.assertEqual(probe.decide(self.good()), ('PASS', []))

    def test_a_differing_byte_fails_even_with_reduced_scope_or_an_error(self):
        report = self.good()
        report['tallies']['V'].update(exact_cases=9, differing_bytes=1, first_failure=dict(case='R1/seed17'))
        report['missing_scope'] = ['users 2 of 4']
        report['sections']['timing'] = dict(error='boom')
        verdict, problems = probe.decide(report)
        self.assertEqual(verdict, 'FAIL')
        self.assertIn('V differs from A', problems[0])

    def test_each_hard_miss_fails(self):
        cases = {'blind N5r': lambda r: r['sections']['negative']['N5r'].update(detects=False),
                 'blind N5x': lambda r: r['sections']['negative']['N5x'].update(detects=False),
                 'moved': lambda r: r['inputs_moved'].append(dict(user=0, tensor='qkv')),
                 'unwritten': lambda r: r['unwritten'].append(dict(arm='V', pages=1)),
                 'stale': lambda r: r['sections']['stale'].update(exact=False),
                 'trace': lambda r: r['sections']['trace'].update(exact=False),
                 'trace unwritten': lambda r: r['sections']['trace'].update(unwritten=[1]),
                 'cache': lambda r: r['sections']['cache'].update(ok=False, problems=['x'])}
        for name, damage in cases.items():
            with self.subTest(name=name):
                report = self.good()
                damage(report)
                self.assertEqual(probe.decide(report)[0], 'FAIL')

    def test_no_decision_for_a_raised_section_reduced_scope_or_an_unqualified_control(self):
        for name, damage in (('raised', lambda r: r['sections']['p0'].update(error='boom')),
                             ('reduced', lambda r: r.update(missing_scope=['users 2 of 4'])),
                             ('control', lambda r: r.update(a_qualified=False)),
                             ('no V', lambda r: r['tallies']['V'].update(cases=0))):
            with self.subTest(name=name):
                report = self.good()
                damage(report)
                self.assertEqual(probe.decide(report)[0], 'NO-DECISION')

    def test_exit_codes(self):
        self.assertEqual([probe.exit_code(v) for v in ('PASS', 'FAIL', 'NO-DECISION')], [0, 1, 4])

    def test_timing_labels(self):
        self.assertEqual(probe.timing_label(193.4, 120.0)['label'], 'proceed')
        self.assertEqual(probe.timing_label(193.4, 193.4 - 65.0)['label'], 'proceed')
        self.assertEqual(probe.timing_label(193.4, 193.4 - 50.0)['label'], 'image-build')
        self.assertEqual(probe.timing_label(193.4, 193.4 - 39.0)['label'], 'kill')
        self.assertEqual(probe.timing_label(None, 1.0)['label'], 'not-run')
        self.assertTrue(probe.timing_label(200.0, 120.0, 100.0)['write_bound'])
        self.assertFalse(probe.timing_label(200.0, 120.0, 110.0)['write_bound'])
        self.assertAlmostEqual(probe.timing_label(200.0, 140.0)['verify_ms_est'], -2.88)

    def test_the_verdict_line_has_the_documented_fields(self):
        report = self.good()
        report.update(verdict='PASS')
        report['sections']['timing'] = dict(label='proceed', a_us=193.4, v_us=100.0)
        line = probe.verdict_line(report)
        self.assertRegex(line, r'^GDN_V5 verdict=PASS scope=full p0=10/10 bytes=0 traced=exact timing=proceed '
                               r'a_us=193\.4 v_us=100\.0$')
        report['missing_scope'] = ['x']
        self.assertIn('scope=reduced', probe.verdict_line(report))


class ParseTests(unittest.TestCase):
    def test_defaults_are_the_full_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            arguments = probe.parse(['--out', str(Path(directory) / 'r.json')])
        self.assertEqual(probe.scope_missing(arguments), [])
        self.assertEqual(arguments.users, 4)

    def test_bad_arguments_are_usage_errors(self):
        for argv in (['--sections', 'nope'], ['--regimes', 'R9'], ['--timing-arms', 'V'], ['--trace-replays', '1'],
                     ['--users', '5']):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                probe.parse(['--out', 'x.json'] + argv)
            self.assertEqual(caught.exception.code, 2)


class RealInputMathTests(unittest.TestCase):
    def test_the_causal_convolution_is_the_direct_loop(self):
        generator = torch.Generator().manual_seed(3)
        x = torch.randn(9, 5, generator=generator)
        taps = torch.randn(5, 4, generator=generator)
        got = probe.causal_conv_silu(torch, x, taps)
        want = torch.zeros(9, 5)
        for t in range(9):
            for c in range(5):
                total = 0.0
                for j in range(4):
                    source = t - 3 + j
                    if source >= 0:
                        total += float(taps[c, j]) * float(x[source, c])
                want[t, c] = torch.nn.functional.silu(torch.tensor(total))
        self.assertTrue(torch.allclose(got, want, atol=1e-5))

    def test_the_recurrence_states_are_the_stepwise_ones(self):
        geometry = found()
        generator = torch.Generator().manual_seed(5)
        tokens = 24
        qkv = torch.randn(tokens, geometry.gdn_qkv, generator=generator)
        beta = torch.rand(tokens, geometry.gdn_nv, generator=generator)
        gate = -torch.rand(tokens, geometry.gdn_nv, generator=generator)
        states = probe.recurrence_states(torch, qkv, beta, gate, geometry, (0, 8, 24))
        self.assertEqual(sorted(states), [0, 8, 24])
        self.assertTrue(bool((states[0] == 0).all()))
        self.assertEqual(tuple(states[8].shape), (1, 12, 128, 128))
        again = probe.recurrence_states(torch, qkv, beta, gate, geometry, (8,))
        self.assertTrue(torch.equal(states[8].view(torch.int16), again[8].view(torch.int16)))
        self.assertFalse(torch.equal(states[8].view(torch.int16), states[24].view(torch.int16)))


# ---- run_card_m.sh ----

BASH = shutil.which('bash')


class ScriptTests(unittest.TestCase):
    def test_the_probe_and_every_module_it_imports_are_mounted_from_this_checkout(self):
        text = script_text()
        for name in ('gdn_seq_block.py', 'gdn_seq_block_compute.cpp', 'gdn_seq_block_reader.cpp', 'gdn_seq_block_writer.cpp',
                     'gdn_seq_block_split.py', 'gdn_seq_block_split_compute.cpp', 'gdn_seq_block_split_reader.cpp',
                     'gdn_seq_block_split_writer.cpp', 'gdn_seq_block_device_test.py', 'gdn_tp4_card_test.py',
                     'gdn_user_batch.py', 'gdn_user_batch_tp.py', 'gdn_multitoken.py', 'tp_shapes.py',
                     'verify_trace_t1.py'):
            self.assertIn(name, text)
            self.assertTrue((CI / name).exists(), name)
        self.assertIn('gdn_v5_card_m.py', text)
        for name in probe.MODULES:
            self.assertTrue((CI / name).exists() or name == 'gdn_v5_card_m.py', name)

    def test_the_launched_environment_is_pinned_in_the_script(self):
        text = script_text()
        self.assertIn('-e QWEN_FAST_TP=4 -e QWEN_FAST_VERIFY_T1=1', text)
        self.assertIn('--network none', text)
        self.assertRegex(text, r'--mount "type=bind,src=\$MODEL_DIR,dst=/models/hub/models--Qwen--Qwen3\.8-27B,readonly"')
        self.assertIn('TT_METAL_WATCHER=10', text)
        self.assertIn('WATCHER', text)
        self.assertEqual(len(re.findall(r'^timeout -k 30 "\$timeout_s" docker run', text, flags=re.M)), 1)
        self.assertLess(text.index('qual_card_recheck   #'), text.index('timeout -k 30 "$timeout_s" docker run'))
        self.assertIn("3|124|137)", text)

    def test_no_registry_host_or_serial_is_named_beyond_the_embedded_block(self):
        text = script_text()
        start, end = text.index('# >>> qual_card.sh'), text.index('# <<< qual_card.sh')
        outside = text[:start] + text[end:]
        for needle in ('zot.', '.local:', 'blackhole-', 'thatch', '192.168.', '10.0.'):
            self.assertNotIn(needle, outside)

    def test_the_probe_names_no_host_or_serial(self):
        text = (HERE / 'gdn_v5_card_m.py').read_text()
        for needle in ('zot.', 'blackhole-', '192.168.', 'D:\\\\', '/home/', 'C:\\\\'):
            self.assertNotIn(needle, text)

    @unittest.skipUnless(BASH, 'needs bash')
    def test_the_script_parses(self):
        subprocess.run([BASH, '-n', str(SCRIPT)], check=True)

    @unittest.skipUnless(BASH, 'needs bash')
    def test_it_refuses_without_a_card(self):
        env = {key: value for key, value in os.environ.items() if not key.startswith('QUAL_')}
        env.pop('ALLOW_SERVING_CARD', None)
        done = subprocess.run([BASH, str(SCRIPT)], capture_output=True, text=True, env=env, cwd=str(HERE))
        self.assertEqual(done.returncode, 1)
        self.assertIn('QUAL_CARD is not set', done.stderr)


# ---- the whole flow on a fake ttnn ----

class Tensor:
    """A device tensor of the page-level memory model: `pages` is int32 [pages, 512] (one 2 KiB tile or row per page), or None
    for never-written memory, or a recipe (a deterministic function a fake launch will materialise on first read)."""

    count = 0

    def __init__(self, fake, shape, dtype, layout, memory, pages=None, recipe=None):
        Tensor.count += 1
        self.fake, self.shape, self.dtype, self.layout, self._memory = fake, tuple(shape), dtype, layout, memory
        self.address = 4096 + 4096 * Tensor.count
        self._pages, self.recipe, self.name = pages, recipe, 't%d' % Tensor.count

    @property
    def pages(self):
        if self.recipe is not None:
            seed, count, flip = self.recipe
            generator = torch.Generator().manual_seed(seed)
            pages = torch.randint(-2 ** 31, 2 ** 31 - 1, (count, 512), dtype=torch.int32, generator=generator)
            if flip is not None:
                pages[flip[0], flip[1]] ^= 1
            self._pages, self.recipe = pages, None
        return self._pages

    @pages.setter
    def pages(self, value):
        self._pages, self.recipe = value, None

    def memory_config(self):
        return self._memory

    def buffer_address(self):
        return self.address

    def device(self):
        return self.fake.device


class FakeDevice:
    def __init__(self, fake):
        self.fake = fake
        self.shape = (1, 1)
        self.live_trace = None

    def compute_with_storage_grid_size(self):
        return SimpleNamespace(x=11, y=10)

    def worker_core_from_logical_core(self, core):
        return SimpleNamespace(x=core[0] + 1, y=core[1] + 2)

    def num_program_cache_entries(self):
        return len(self.fake.cache)


class Behaviour:
    """What the fake launch does wrong, if anything (each is one broken variant of the V5 launch)."""

    def __init__(self, **flags):
        self.flip_v = flags.get('flip_v', False)             # V differs from A by one bit, always
        self.flip_v_on_replay = flags.get('flip_v_on_replay', False)  # ... only inside a trace replay
        self.flip_v_third = flags.get('flip_v_third', False)  # ... only on a V launch after V has run on two sets
        self.blind_n5r = flags.get('blind_n5r', False)       # N5r behaves like A
        self.blind_n5x = flags.get('blind_n5x', False)
        self.move_input = flags.get('move_input', False)      # V overwrites its first input
        self.unwritten_v_states = flags.get('unwritten_v_states', False)
        self.extra_cache_entry = flags.get('extra_cache_entry', False)
        self.raise_in_cache = flags.get('raise_in_cache', False)
        self.hang = flags.get('hang', False)


class FakeBench(FakeTTNN):
    bfloat16, float32, uint32 = 'bf16', 'fp32', 'u32'
    TILE_LAYOUT, ROW_MAJOR_LAYOUT = 'tile', 'row-major'
    DRAM_MEMORY_CONFIG, L1_MEMORY_CONFIG = 'dram', 'l1'
    __file__ = '/fake/ttnn/__init__.py'

    def __init__(self, behaviour=None):
        super().__init__()
        self.behaviour = behaviour or Behaviour()
        self.device = FakeDevice(self)
        self.cache = set()
        self.capturing = None
        self.replaying = False
        self.v_launches = 0
        self.closed = False

    # --- device and tensors ---
    @staticmethod
    def MeshShape(rows, columns):
        return (rows, columns)

    def open_mesh_device(self, shape, **kwargs):
        self.opened = (shape, kwargs)
        return self.device

    def close_mesh_device(self, device):
        self.closed = True

    @staticmethod
    def ReplicateTensorToMesh(device):
        return 'replicate'

    def empty(self, shape, device=None, dtype=None, layout=None, memory_config=None):
        return Tensor(self, shape, dtype, layout, memory_config)

    def from_torch(self, value, device=None, dtype=None, layout=None, memory_config=None, mesh_mapper=None):
        if dtype == self.uint32:
            return Tensor(self, value.shape, dtype, layout, memory_config,
                          pages=value.reshape(-1, 512).to(torch.int32).clone())
        image = dev.pad_image(torch, value.bfloat16())
        return Tensor(self, value.shape, dtype, layout, memory_config, pages=dev.tile_image(torch, image))

    def to_torch(self, value):
        return value.pages.clone()

    def get_device_tensors(self, value):
        return [value]

    def deallocate(self, value):
        pass

    def synchronize_device(self, device):
        pass

    @staticmethod
    def SemaphoreDescriptor(id, core_ranges, initial_value):
        return SimpleNamespace(id=id, core_ranges=tuple(core_ranges), initial_value=initial_value)

    @staticmethod
    def ProgramDescriptor(kernels, cbs, semaphores=()):
        return SimpleNamespace(kernels=list(kernels), cbs=list(cbs), semaphores=list(semaphores))

    @staticmethod
    def TensorAccessorArgs(value):
        return SimpleNamespace(get_compile_time_args=lambda: [7])

    # --- traces ---
    def begin_trace_capture(self, device, cq_id=0):
        self.capturing = SimpleNamespace(ops=[])
        return self.capturing

    def end_trace_capture(self, device, handle, cq_id=0):
        self.capturing = None

    def execute_trace(self, device, handle, cq_id=0, blocking=False):
        self.replaying = True
        try:
            for tensors, program in handle.ops:
                self.run(tensors, program)
        finally:
            self.replaying = False

    def release_trace(self, device, handle):
        pass

    # --- launches ---
    def generic_op(self, tensors, program):
        for chip_program in program.values():
            key = tuple(sorted((kernel.kernel_source if len(kernel.kernel_source) < 400 else
                                hashlib.sha256(kernel.kernel_source.encode()).hexdigest(),
                                tuple(kernel.compile_time_args), tuple(kernel.core_ranges)) for kernel in chip_program.kernels))
            self.cache.add(key)
        if self.capturing is not None:
            self.capturing.ops.append((list(tensors), program))
            return
        self.run(tensors, program)

    def run(self, tensors, program):
        for chip_program in program.values():
            sources = [kernel.kernel_source for kernel in chip_program.kernels]
            if dev.RAW_COPY in sources:
                arguments = None
                for kernel in chip_program.kernels:
                    for column in kernel.runtime_args.values():
                        arguments = list(next(iter(column.values())))
                        break
                    break
                source, destination = (next(t for t in tensors if t.address == arguments[index]) for index in (0, 1))
                pages = arguments[2]
                destination.pages = source.pages[:pages].clone() if source.pages is not None else \
                    torch.full((pages, 512), 0x11111111, dtype=torch.int32)
                continue
            self.launch(tensors, chip_program)

    def launch(self, tensors, chip_program):
        header = None
        for kernel in chip_program.kernels:
            found = re.search(r'// gdn_seq_block(_split)? generated build: (.*)\n', kernel.kernel_source)
            if found:
                header = (bool(found.group(1)), found.group(2), kernel.kernel_source)
                break
        if header is None:
            raise AssertionError('a launch that is neither a raw copy nor a gdn launch')
        text = header[1]
        is_split = 'variant=' in text and 'level=' not in text
        variant = re.search(r'variant=(\w+)', text).group(1)
        diag = re.search(r'diag=(\w+)', text).group(1)
        if self.behaviour.hang and is_split and variant == 'A':
            threading.Event().wait()
        flat = list(tensors)
        users = (len(flat) - 1) // 7
        norm_w = flat[7]
        if is_split and variant == 'A' and diag == 'none':
            self.v_launches += 1
            if self.behaviour.extra_cache_entry and self.v_launches == 4:
                self.cache.add(('an entry nothing asked for', self.v_launches))
        for user in range(users):
            start = 0 if user == 0 else 7 * user + 1
            qkv, beta, gate, initial, output, states, z = flat[start:start + 7]
            digest = hashlib.blake2b(digest_size=8)
            for tensor in (qkv, beta, gate, initial, z, norm_w):
                digest.update(tensor.pages.numpy().tobytes())
            seed = int.from_bytes(digest.digest(), 'little') % (2 ** 62)
            flip_out = None
            if variant == 'N5r' and not self.behaviour.blind_n5r:
                flip_out = (0, 0)
            if variant == 'N5x' and not self.behaviour.blind_n5x:
                flip_out = (1, 1)
            if is_split and variant == 'A' and diag == 'none':
                if self.behaviour.flip_v or (self.behaviour.flip_v_on_replay and self.replaying) or \
                        (self.behaviour.flip_v_third and self.v_launches >= 7):
                    flip_out = (2, 2)
            output.recipe = (seed, dev.page_count(output.shape), flip_out)
            states.recipe = (seed + 1, dev.page_count(states.shape), None)
            if is_split and diag == 'nosnap' or (is_split and self.behaviour.unwritten_v_states and variant == 'A'):
                states.pages = torch.full((dev.page_count(states.shape), 512), dev.SENTINEL_WORD, dtype=torch.int32)
            if is_split and self.behaviour.move_input and variant == 'A' and diag == 'none' and user == 0:
                qkv.pages = qkv.pages ^ 1


@contextlib.contextmanager
def fake_runtime(behaviour=None, qualified_control=True):
    fake = FakeBench(behaviour)
    synthetic = SyntheticRoot()
    root = synthetic.__enter__()
    patches = [mock.patch.dict(sys.modules, {'ttnn': fake}), mock.patch.dict(os.environ, FOUR),
               mock.patch('gdn_multitoken.validate_handoff_runtime')]
    if qualified_control:
        patches.append(mock.patch.dict(seq.QUALIFIED, {0: seq.sha256(seq.generate(root, 0))}))
    for patch in patches:
        patch.start()
    split._ENGAGED.clear()
    verify_trace_t1.take()
    try:
        yield fake, root
    finally:
        for patch in reversed(patches):
            patch.stop()
        synthetic.__exit__(None, None, None)
        verify_trace_t1.take()


def run_probe(behaviour=None, extra=(), qualified_control=True, call_timeout='60'):
    """main() against the fake: (exit code, report, stdout lines)."""
    with tempfile.TemporaryDirectory() as directory, fake_runtime(behaviour, qualified_control) as (fake, root):
        out = Path(directory) / 'report.json'
        # One user, one regime, one seed unless a test asks for more: every case reads the 3,072 snapshot pages back
        # per arm and user, and the later options win (argparse), so a test overrides what it needs.
        argv = ['--out', str(out), '--root', str(root), '--users', '1', '--seeds', '17', '--other-seeds', '17',
                '--regimes', 'R1', '--r4-launches', '2', '--trace-replays', '3',
                '--timing-launches', '2', '--timing-rounds', '2', '--call-timeout', call_timeout, *extra]
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = probe.main(argv)
        report = json.loads(out.read_text())
        return code, report, stdout.getvalue().strip().splitlines(), fake


class FlowTests(unittest.TestCase):
    def test_a_correct_v5_runs_every_section_and_only_scope_is_missing(self):
        code, report, lines, fake = run_probe(extra=('--users', '2', '--seeds', '17,23', '--regimes', 'R1,R2,R3,R3b,RA,R4',
                                                     '--trace-replays', '4'))
        self.assertEqual(code, 4)
        self.assertEqual(report['verdict'], 'NO-DECISION')
        self.assertEqual([p for p in report['problems'] if not p.startswith('reduced scope')], [])
        self.assertEqual(sorted(report['sections']), sorted(probe.SECTIONS))
        self.assertTrue(all(section['error'] is None for section in report['sections'].values()))
        tally = report['tallies']['V']
        self.assertEqual(tally['exact_cases'], tally['cases'])
        self.assertGreater(tally['cases'], 12)
        self.assertEqual(tally['differing_bytes'], 0)
        self.assertTrue(report['sections']['negative']['N5r']['detects'])
        self.assertTrue(report['sections']['negative']['N5x']['detects'])
        self.assertTrue(report['sections']['stale']['exact'])
        self.assertTrue(report['sections']['trace']['exact'])
        self.assertEqual(report['sections']['trace']['replays'], 4)
        self.assertTrue(report['sections']['cache']['ok'], report['sections']['cache'])
        deltas = [(step['arm'], step['users'], step['delta']) for step in report['sections']['cache']['steps']]
        self.assertEqual(deltas, [('V', 2, 1), ('V', 2, 0), ('V', 1, 1), ('A', 2, 0), ('V', 2, 0), ('A', 2, 0),
                                  ('V', 2, 0), ('A', 2, 0)])
        self.assertEqual(report['unwritten'], [])
        self.assertEqual(report['inputs_moved'], [])
        timing = report['sections']['timing']
        self.assertEqual(sorted(timing['per_launch']), sorted(probe.TIMING_ARMS))
        self.assertIn(timing['label'], ('proceed', 'image-build', 'kill'))
        self.assertTrue(report['a_qualified'])
        summary = json.loads(lines[-1])
        self.assertEqual((summary['kind'], summary['verdict']), ('gdn-v5-probe', 'NO-DECISION'))
        self.assertTrue(lines[-2].startswith('GDN_V5 verdict=NO-DECISION scope=reduced'))
        self.assertEqual(report['v5_triple'].keys(), {'reader', 'writer', 'compute'})
        self.assertIsNone(summary['v5_triple'])  # only a full-scope PASS licenses the triple
        self.assertEqual(fake.opened[0], (1, 1))

    def test_the_launched_environment_is_the_one_the_probe_reads(self):
        code, report, lines, fake = run_probe()
        self.assertEqual(report['env_read']['QWEN_FAST_TP'], '4')
        self.assertEqual(report['env_read']['QWEN_FAST_VERIFY_T1'], '1')
        self.assertEqual(report['sections']['selftest']['env']['QWEN_FAST_TP'], '4')

    def test_the_probe_refuses_a_process_launched_without_the_four_card_environment(self):
        with tempfile.TemporaryDirectory() as directory, fake_runtime() as (fake, root):
            with mock.patch.dict(os.environ, {'QWEN_FAST_TP': '2'}):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    code = probe.main(['--out', str(Path(directory) / 'r.json'), '--root', str(root), '--sections', 'selftest',
                                       '--regimes', 'R1'])
            self.assertEqual(code, 4)
            self.assertIn('QWEN_FAST_TP=4 is required', stdout.getvalue())

    def test_a_v5_that_differs_by_one_bit_fails_and_names_the_location(self):
        code, report, lines, fake = run_probe(Behaviour(flip_v=True), extra=('--sections', 'p0'))
        self.assertEqual(code, 1)
        self.assertEqual(report['verdict'], 'FAIL')
        tally = report['tallies']['V']
        self.assertEqual(tally['exact_cases'], 0)
        self.assertGreater(tally['differing_bytes'], 0)
        failure = tally['first_failure']['result']['user0_output_padded']
        # the flip is in word 2 of output page 2: row 0, column 68 of the padded image - token 0, head 0, tile 2
        self.assertEqual((failure['first']['token'], failure['first']['head'], failure['first']['tile']), (0, 0, 2))
        self.assertEqual(failure['first']['half'], 'helper')
        self.assertFalse(failure['first']['padding_row'])

    def test_a_blind_control_fails_the_gate(self):
        code, report, lines, fake = run_probe(Behaviour(blind_n5r=True), extra=('--sections', 'negative'))
        self.assertEqual(code, 1)
        self.assertFalse(report['sections']['negative']['N5r']['detects'])
        self.assertTrue(any('N5r' in problem for problem in report['problems']))
        code, report, lines, fake = run_probe(Behaviour(blind_n5x=True), extra=('--sections', 'negative'))
        self.assertEqual(code, 1)
        self.assertFalse(report['sections']['negative']['N5x']['detects'])

    def test_a_replay_only_difference_is_caught_by_the_trace_section_alone(self):
        code, report, lines, fake = run_probe(Behaviour(flip_v_on_replay=True), extra=('--sections', 'stale,trace'))
        self.assertEqual(code, 1)
        self.assertFalse(report['sections']['trace']['exact'])
        self.assertTrue(report['sections']['stale']['exact'])
        self.assertTrue(any('traced replays differ' in problem for problem in report['problems']))

    def test_a_difference_after_several_launches_is_caught_by_stale_or_later_sections(self):
        code, report, lines, fake = run_probe(Behaviour(flip_v_third=True), extra=('--sections', 'stale,cache'))
        self.assertEqual(code, 1)
        self.assertEqual(report['verdict'], 'FAIL')
        self.assertFalse(report['sections']['stale']['exact'] and report['sections']['cache']['ok'])

    def test_a_moved_input_fails(self):
        code, report, lines, fake = run_probe(Behaviour(move_input=True), extra=('--sections', 'p0'))
        self.assertEqual(code, 1)
        self.assertTrue(report['inputs_moved'])
        self.assertTrue(any('input moved' in problem for problem in report['problems']))

    def test_an_unwritten_page_fails(self):
        code, report, lines, fake = run_probe(Behaviour(unwritten_v_states=True), extra=('--sections', 'p0'))
        self.assertEqual(code, 1)
        self.assertTrue(report['unwritten'])
        self.assertTrue(any('never written' in problem for problem in report['problems']))

    def test_a_wrong_program_cache_delta_fails(self):
        code, report, lines, fake = run_probe(Behaviour(extra_cache_entry=True), extra=('--sections', 'cache'))
        self.assertEqual(code, 1)
        self.assertFalse(report['sections']['cache']['ok'])
        self.assertTrue(any('new program-cache entries' in text for text in report['sections']['cache']['problems']))

    def test_a_section_that_raises_is_no_decision_and_the_others_still_run(self):
        with mock.patch.object(probe, 'real_inputs', side_effect=RuntimeError('no weights here')):
            code, report, lines, fake = run_probe(extra=('--regimes', 'R1,R5', '--r5-layers', '0', '--sections', 'p0,cache'))
        self.assertEqual(code, 4)
        self.assertIn('no weights here', report['sections']['p0']['error'])
        self.assertIsNone(report['sections']['cache']['error'])
        self.assertTrue(any('raised' in problem for problem in report['problems']))

    def test_an_unqualified_control_is_no_decision(self):
        code, report, lines, fake = run_probe(qualified_control=False, extra=('--sections', 'p0'))
        self.assertEqual(code, 4)
        self.assertFalse(report['a_qualified'])
        self.assertTrue(any('qualified K5-A' in problem for problem in report['problems']))

    def test_the_watchdog_exits_three_with_the_partial_report_written(self):
        exits = []

        class Stop(BaseException):
            pass

        def fake_exit(code):
            exits.append(code)
            raise Stop()

        with tempfile.TemporaryDirectory() as directory, fake_runtime(Behaviour(hang=True)) as (fake, root):
            out = Path(directory) / 'r.json'
            argv = ['--out', str(out), '--root', str(root), '--users', '1', '--seeds', '17', '--other-seeds', '17',
                    '--regimes', 'R1', '--sections', 'selftest,p0', '--call-timeout', '1']
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout), mock.patch.object(probe.os, '_exit', side_effect=fake_exit):
                thread = threading.Thread(target=lambda: self.run_main(argv), daemon=True)
                thread.start()
                thread.join(6)
            self.assertEqual(exits[:1], [3])
            report = json.loads(out.read_text())
            self.assertIn('watchdog', report['error'])
            self.assertIn('launch V', report['watchdog'])
            self.assertIn('WATCHDOG', stdout.getvalue())

    @staticmethod
    def run_main(argv):
        try:
            probe.main(argv)
        except BaseException:  # noqa: BLE001
            pass

    def test_the_v5_arm_runs_as_one_coalesced_launch_per_case(self):
        code, report, lines, fake = run_probe(extra=('--sections', 'p0'))
        entry = report['sections']['p0']['cases'][0]
        self.assertEqual(entry['case'], 'R1/seed17')
        self.assertEqual(report['sections']['p0']['verify_t1_counts'], {'V': {'coalesced': 1}, 'A': {'coalesced': 1}})

    def test_the_two_arms_alternate_which_goes_first(self):
        code, report, lines, fake = run_probe(extra=('--sections', 'p0', '--seeds', '17,23,29'))
        orders = [case['order'] for case in report['sections']['p0']['cases']]
        self.assertEqual(orders[:3], [['A', 'V'], ['V', 'A'], ['A', 'V']])


if __name__ == '__main__':
    unittest.main()
