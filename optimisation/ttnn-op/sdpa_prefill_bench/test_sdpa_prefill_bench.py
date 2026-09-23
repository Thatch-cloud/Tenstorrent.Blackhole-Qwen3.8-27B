"""CPU tests for the M1 chunked-SDPA bench helpers: shapes, the arm table, the slope fit and the
verdict reading rules. No ttnn needed."""

import contextlib
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sdpa_prefill_bench as bench  # noqa: E402


def results_for(slopes_ms_per_1k, intercept=5.0):
    """{arm: {start: ms}} lying exactly on the given per-call slopes."""
    return {name: {str(start): intercept + slope * start / 1000.0 for start in bench.STARTS}
            for name, slope in slopes_ms_per_1k.items()}


class ArmTableTests(unittest.TestCase):
    def test_the_seven_arms_move_one_knob_each_from_the_served_baseline(self):
        self.assertEqual(bench.ARM_NAMES, ('baseline', 'bf16_kv', 'bf16_qkv', 'exp_approx', 'fp32_off',
                                           'q256_2048', 'q256_4096'))
        base = bench.arm_by_name('baseline')
        self.assertEqual(base, dict(name='baseline', rows=2048, q_chunk=128, k_chunk=128, q_dtype='bf8',
                                    kv_dtype='bf8', exp_approx=False, fp32_dest=True))
        expected_changes = dict(bf16_kv={'kv_dtype'}, bf16_qkv={'q_dtype', 'kv_dtype'}, exp_approx={'exp_approx'},
                                fp32_off={'fp32_dest'}, q256_2048={'q_chunk'}, q256_4096={'q_chunk', 'rows'})
        for name, keys in expected_changes.items():
            with self.subTest(arm=name):
                arm = bench.arm_by_name(name)
                changed = {key for key in base if key != 'name' and arm[key] != base[key]}
                self.assertEqual(changed, keys)

    def test_unknown_arms_are_refused(self):
        with self.assertRaisesRegex(ValueError, 'unknown arm'):
            bench.arm_by_name('bf4')

    def test_arm_by_name_returns_a_copy(self):
        bench.arm_by_name('baseline')['rows'] = 1
        self.assertEqual(bench.arm_by_name('baseline')['rows'], 2048)


class GeometryTests(unittest.TestCase):
    def test_shapes_match_the_served_tp2_call(self):
        self.assertEqual(bench.q_shape(bench.arm_by_name('baseline')), (1, 12, 2048, 256))
        self.assertEqual(bench.q_shape(bench.arm_by_name('q256_4096')), (1, 12, 4096, 256))
        self.assertEqual(bench.kv_shape(2080), (2080, 2, 64, 256))
        self.assertEqual(bench.GRID, (11, 10))

    def test_sweep_starts_are_0_32k_64k_126k(self):
        self.assertEqual(bench.STARTS, (0, 32 * 1024, 64 * 1024, 126 * 1024))

    def test_the_pool_covers_the_largest_start_plus_rows_padded_to_32_blocks(self):
        arms = [bench.arm_by_name(name) for name in bench.ARM_NAMES]
        self.assertEqual(bench.pool_blocks(arms, bench.STARTS), 2080)          # 129024 + 4096 -> 2080
        self.assertEqual(bench.pool_blocks(arms[:1], bench.STARTS), 2048)     # 129024 + 2048 = 131072
        self.assertEqual(bench.blocks_for(131328), 2080)                       # the served 131k page table

    def test_busy_cores_follow_the_pair_distribution(self):
        self.assertEqual(bench.busy_cores(bench.arm_by_name('baseline')), 96)
        self.assertEqual(bench.busy_cores(bench.arm_by_name('q256_2048')), 48)
        self.assertEqual(bench.busy_cores(bench.arm_by_name('q256_4096')), 96)
        odd = dict(bench.arm_by_name('baseline'), rows=384)   # 3 q chunks: no pairing
        self.assertEqual(bench.busy_cores(odd), 36)

    def test_bf16_reads_1_88x_the_bytes_and_q256_4096_half_per_token(self):
        base = bench.arm_by_name('baseline')
        start = 129024
        ratio = bench.kv_bytes(bench.arm_by_name('bf16_kv'), start) / bench.kv_bytes(base, start)
        self.assertAlmostEqual(ratio, 2 * 1024 / 1088)
        self.assertAlmostEqual(ratio, bench.EXPECTED_BF16_RATIO)
        prefix_bytes = lambda arm: bench.kv_bytes(arm, start) - bench.kv_bytes(arm, 0)
        per_token = (prefix_bytes(bench.arm_by_name('q256_4096')) / 4096) / (prefix_bytes(base) / 2048)
        self.assertAlmostEqual(per_token, 0.5)

    def test_validate_enforces_the_flexible_path_preconditions(self):
        for name in bench.ARM_NAMES:
            bench.validate(bench.arm_by_name(name), bench.STARTS)
        with self.assertRaises(ValueError):
            bench.validate(bench.arm_by_name('q256_2048'), [128])       # not a multiple of q_chunk 256
        with self.assertRaises(ValueError):
            bench.validate(bench.arm_by_name('baseline'), [-128])
        with self.assertRaises(ValueError):
            bench.validate(dict(bench.arm_by_name('baseline'), rows=2000), [0])

    def test_schedule_interleaves_and_rotates_the_arm_order(self):
        arms = [bench.arm_by_name(name) for name in ('baseline', 'bf16_kv', 'exp_approx')]
        order = bench.schedule(arms, [0, 32768], 3)
        self.assertEqual(len(order), 3 * 3 * 2)
        self.assertEqual([order[0][0], order[6][0], order[12][0]], ['baseline', 'bf16_kv', 'exp_approx'])
        for key in {(a['name'], s) for a in arms for s in (0, 32768)}:
            self.assertEqual(order.count(key), 3)


class SlopeTests(unittest.TestCase):
    def test_fit_recovers_an_exact_line(self):
        line = bench.fit([(0, 5.0), (32768, 5.0 + 0.7 * 32.768), (129024, 5.0 + 0.7 * 129.024)])
        self.assertAlmostEqual(line['slope_ms_per_1k_keys'], 0.7)
        self.assertAlmostEqual(line['intercept_ms'], 5.0)
        self.assertEqual(line['points'], 3)

    def test_fit_needs_two_distinct_starts(self):
        self.assertIsNone(bench.fit([(0, 1.0)]))
        self.assertIsNone(bench.fit([(0, 1.0), (0, 2.0)]))
        self.assertIsNone(bench.fit([(0, 1.0), (32768, None)]))

    def test_the_4096_row_arm_is_also_given_per_2048_rows(self):
        table = bench.slopes(results_for(dict(baseline=0.4, q256_4096=0.4)))
        self.assertAlmostEqual(table['q256_4096']['slope_ms_per_1k_keys'], 0.4)
        self.assertAlmostEqual(table['q256_4096']['slope_ms_per_1k_keys_per_2048_rows'], 0.2)
        self.assertAlmostEqual(table['baseline']['slope_ms_per_1k_keys_per_2048_rows'], 0.4)

    def test_an_arm_with_failed_starts_has_no_slope(self):
        results = results_for(dict(baseline=0.4))
        results['bf16_kv'] = {str(s): None for s in bench.STARTS}
        self.assertIsNone(bench.slopes(results)['bf16_kv'])


class VerdictTests(unittest.TestCase):
    def verdict(self, **slopes):
        return bench.verdict(bench.slopes(results_for(slopes)))

    def test_bytes_bound(self):
        result = self.verdict(baseline=0.4, bf16_kv=0.4 * 1.85, bf16_qkv=0.4 * 1.9, exp_approx=0.39,
                              fp32_off=0.4, q256_2048=0.45, q256_4096=0.4 * 1.05)
        self.assertEqual(result['verdict'], 'bytes-bound')
        self.assertIn('lever #1', result['action'])
        self.assertAlmostEqual(result['ratios']['q256_4096_per_token'], 0.525)
        self.assertEqual(result['q256_2048_preregistration'], 'confirmed')

    def test_compute_bound(self):
        result = self.verdict(baseline=0.4, bf16_kv=0.42, exp_approx=0.3, fp32_off=0.39, q256_2048=0.5,
                              q256_4096=0.8)
        self.assertEqual(result['verdict'], 'compute-bound')
        self.assertIn('#1b', result['action'])

    def test_bytes_insensitive_without_a_compute_knob_moving_is_mixed(self):
        self.assertEqual(self.verdict(baseline=0.4, bf16_kv=0.41, exp_approx=0.4, fp32_off=0.4)['verdict'], 'mixed')

    def test_bf16_sensitive_but_4096_not_halving_is_mixed(self):
        result = self.verdict(baseline=0.4, bf16_kv=0.7, q256_4096=0.7)
        self.assertEqual(result['verdict'], 'mixed')

    def test_bf16_qkv_stands_in_when_bf16_kv_failed(self):
        result = self.verdict(baseline=0.4, bf16_qkv=0.75, q256_4096=0.44)
        self.assertEqual(result['verdict'], 'bytes-bound')
        self.assertEqual(result['ratios']['bf16_source'], 'bf16_qkv')

    def test_no_baseline_is_incomplete(self):
        self.assertEqual(self.verdict(bf16_kv=0.7)['verdict'], 'incomplete')

    def test_the_q256_2048_preregistration_can_be_refuted(self):
        result = self.verdict(baseline=0.4, bf16_kv=0.75, q256_2048=0.3, q256_4096=0.44)
        self.assertEqual(result['q256_2048_preregistration'], 'refuted')

    def test_the_verdict_line_is_one_parseable_line(self):
        line = bench.verdict_line(self.verdict(baseline=0.4, bf16_kv=0.75, q256_4096=0.44))
        self.assertTrue(line.startswith('M1 VERDICT: bytes-bound | bf16/bf8=1.875 (bf16_kv)'))
        self.assertNotIn(chr(10), line)
        self.assertIn('exp_approx=n/a', line)


class MainTests(unittest.TestCase):
    def test_without_ttnn_main_writes_an_error_report_and_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'm1.json'
            from contextlib import redirect_stdout
            import io
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                status = bench.main(['--out', str(out), '--arms', 'baseline', '--rounds', '1'])
            self.assertEqual(status, 1)
            report = json.loads(out.read_text(encoding='utf-8'))
        self.assertFalse(report['passed'])
        self.assertIn('Error', report['error'])
        self.assertIn('M1 VERDICT: incomplete', buffer.getvalue())

    def test_bad_arm_names_are_refused_at_parse_time(self):
        with self.assertRaises(ValueError):
            bench.main(['--out', 'unused.json', '--arms', 'nope'])

    def test_the_table_prints_every_timed_arm(self):
        results = results_for(dict(baseline=0.4, bf16_kv=0.75))
        results['bf16_kv'][str(bench.STARTS[-1])] = None
        text = bench.format_table(results, bench.slopes(results), list(bench.STARTS))
        self.assertIn('baseline', text)
        self.assertIn('ERR', text)
        self.assertEqual(len(text.splitlines()), 3)


try:
    import torch
except ImportError:  # pragma: no cover - CI installs torch; the fake-device tests need it
    torch = None


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class WatchdogTests(unittest.TestCase):
    def make(self, seconds=5.0):
        self.clock, self.exits, self.fired, self.out = FakeClock(), [], [], StringIO()
        return bench.Watchdog(seconds, on_fire=self.fired.append, clock=self.clock, exit=self.exits.append,
                              start=False, backstop=False, out=self.out)

    def test_a_call_past_its_budget_prints_watchdog_writes_the_report_and_exits_3(self):
        dog = self.make()
        with dog.guard('baseline@126976 timed'):
            self.clock.now += 4.9
            self.assertFalse(dog.check())
            self.clock.now += 0.2
            self.assertTrue(dog.check())
            self.assertFalse(dog.check())            # fires once
        self.assertEqual(self.exits, [3])
        self.assertEqual(self.fired, ['baseline@126976 timed'])
        self.assertTrue(self.out.getvalue().startswith('WATCHDOG: baseline@126976 timed did not return within 5 s'))

    def test_a_warmup_gets_the_compile_grace(self):
        dog = self.make(120)
        with dog.guard('baseline@0 warmup', extra=bench.COMPILE_GRACE_S):
            self.clock.now += 120 + bench.COMPILE_GRACE_S - 1
            self.assertFalse(dog.check())
            self.clock.now += 2
            self.assertTrue(dog.check())
        self.assertIn('within 360 s', self.out.getvalue())
        self.assertEqual(self.exits, [3])

    def test_nothing_fires_outside_a_guard_or_when_off(self):
        dog = self.make()
        with dog.guard('warmup'):
            pass
        self.clock.now += 1000
        self.assertFalse(dog.check())
        off = self.make(0)
        with off.guard('anything'):
            self.clock.now += 1e6
            self.assertFalse(off.check())
        self.assertEqual(self.exits, [])
        before = len(threading_names())
        bench.Watchdog(0)                             # the default: no poll thread when off
        self.assertEqual(len(threading_names()), before)

    def test_the_real_thread_ends_a_hung_process_with_exit_3(self):
        code = ('import sys, time; sys.path.insert(0, %r); import sdpa_prefill_bench as b\n'
                'dog = b.Watchdog(0.3, on_fire=lambda what: print("REPORT", what, flush=True), poll=0.05)\n'
                'with dog.guard("hung call"):\n'
                '    time.sleep(20)\n'
                'print("RETURNED")\n') % str(Path(__file__).resolve().parent)
        result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertIn('WATCHDOG: hung call did not return within 0 s', result.stdout)
        self.assertIn('REPORT hung call', result.stdout)
        self.assertNotIn('RETURNED', result.stdout)


def threading_names():
    import threading
    return [thread.name for thread in threading.enumerate() if thread.name == 'm1-watchdog' and thread.is_alive()]


class FakeTensor:
    def __init__(self, host):
        self.host = host


class FakeDevice:
    def compute_with_storage_grid_size(self):
        return SimpleNamespace(x=11, y=10)

    def worker_core_from_logical_core(self, core):
        return SimpleNamespace(x=core.x + 1, y=core.y + 2 if core.y < 5 else core.y + 3)


def fake_ttnn(drift=None):
    """Enough of ttnn for run(): SDPA returns q scaled by (chunk_start + 1), bit-deterministic unless
    `drift` (a set of starts) makes every second call at those starts differ in one element."""
    calls = {}

    def sdpa(input_tensor_q, chunk_start_idx_tensor=None, chunk_start_idx=None, **_):
        start = int(chunk_start_idx_tensor.host[0]) if chunk_start_idx_tensor is not None else chunk_start_idx
        calls[start] = calls.get(start, 0) + 1
        out = (input_tensor_q.host.float() * (start + 1)).to(torch.bfloat16)
        if drift and start in drift and calls[start] % 2 == 0:
            out = out.clone()
            out.view(-1)[0] = out.view(-1)[0] + 1
        return FakeTensor(out)

    module = SimpleNamespace(
        bfloat8_b='bf8', bfloat16='bf16', int32='int32', TILE_LAYOUT='tile', ROW_MAJOR_LAYOUT='rm',
        DRAM_MEMORY_CONFIG='dram', MathFidelity=SimpleNamespace(HiFi2='hifi2'),
        open_device=lambda device_id, l1_small_size: FakeDevice(), close_device=lambda device: None,
        synchronize_device=lambda device: None, deallocate=lambda tensor: None,
        from_torch=lambda host, **_: FakeTensor(host), to_torch=lambda tensor: tensor.host,
        WormholeComputeKernelConfig=lambda **kw: kw, SDPAProgramConfig=lambda **kw: kw,
        CoreCoord=lambda x, y: SimpleNamespace(x=x, y=y),
        transformer=SimpleNamespace(chunked_scaled_dot_product_attention=sdpa))
    return module


def main_with(args, module):
    """bench.main on `module` as ttnn: (status, report, stdout, coordinate fixture or None)."""
    saved = sys.modules.get('ttnn')
    sys.modules['ttnn'] = module
    try:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'm1.json'
            coords = Path(directory) / 'coords.json'
            buffer = StringIO()
            argv = ['--out', str(out), '--arms', 'baseline', '--starts', '0,128', '--rounds', '2', '--warmup', '1']
            argv += [coords.as_posix() if word == 'COORDS' else word for word in args]
            with redirect_stdout(buffer):
                status = bench.main(argv)
            report = json.loads(out.read_text(encoding='utf-8'))
            fixture = json.loads(coords.read_text(encoding='utf-8')) if coords.is_file() else None
    finally:
        if saved is None:
            sys.modules.pop('ttnn', None)
        else:
            sys.modules['ttnn'] = saved
    return status, report, buffer.getvalue(), fixture


@unittest.skipIf(torch is None, 'torch not installed')
class ShaTests(unittest.TestCase):
    def test_output_sha_is_the_int16_view_and_sees_every_bit(self):
        tensor = torch.tensor([[1.0, -0.0, float('nan')]], dtype=torch.bfloat16)
        digest, dtype, shape = bench.output_sha(torch, tensor)
        self.assertEqual(digest, hashlib.sha256(tensor.view(torch.int16).numpy().tobytes()).hexdigest())
        self.assertEqual((dtype, shape), ('torch.bfloat16', [1, 3]))
        zero = torch.tensor([[1.0, 0.0, float('nan')]], dtype=torch.bfloat16)          # -0 vs +0
        self.assertNotEqual(bench.output_sha(torch, zero)[0], digest)
        self.assertEqual(bench.output_sha(torch, tensor.t().t())[0], digest)            # layout does not matter

    def test_sha_lines(self):
        report = dict(sha256={'baseline@32768': 'b' * 64, 'baseline@0': 'a' * 64},
                      sha_stable={'baseline@0': True, 'baseline@32768': False})
        self.assertEqual(bench.sha_lines(report), ['M1 SHA baseline@0 %s stable=1' % ('a' * 64),
                                                   'M1 SHA baseline@32768 %s stable=0' % ('b' * 64)])
        self.assertEqual(bench.sha_lines(dict()), [])

    def run_main(self, args, module):
        return main_with(args, module)

    def test_sha_per_arm_and_start_on_a_fake_device(self):
        status, report, stdout, fixture = self.run_main(['--sha', '--watchdog-s', '60', '--coords-out', 'COORDS'],
                                                        fake_ttnn())
        self.assertEqual(status, 0, report.get('error'))
        self.assertEqual(sorted(report['sha256']), ['baseline@0', 'baseline@128'])
        self.assertNotEqual(report['sha256']['baseline@0'], report['sha256']['baseline@128'])
        self.assertEqual(report['sha_stable'], {'baseline@0': True, 'baseline@128': True})
        self.assertEqual(report['sha_meta']['baseline@0'], dict(dtype='torch.bfloat16', shape=[1, 12, 2048, 256]))
        self.assertIn('M1 SHA baseline@0 %s stable=1' % report['sha256']['baseline@0'], stdout)
        self.assertLess(stdout.index('M1 SHA'), stdout.index('M1 VERDICT:'))
        self.assertEqual(len(report['samples_ms']['baseline@0']), 2)
        self.assertEqual(len(fixture['cores']), 110)
        self.assertEqual(fixture['cores'][12], dict(i=12, logical=[1, 1], worker=[2, 3]))
        self.assertEqual(fixture['grid'], [11, 10])

    def test_an_unstable_output_is_flagged(self):
        _, report, stdout, _ = self.run_main(['--sha'], fake_ttnn(drift={128}))
        self.assertEqual(report['sha_stable'], {'baseline@0': True, 'baseline@128': False})
        self.assertIn('stable=0', stdout)

    def test_without_the_new_flags_the_report_and_output_are_as_before(self):
        status, report, stdout, fixture = self.run_main([], fake_ttnn())
        self.assertEqual(status, 0)
        for key in ('sha256', 'sha_stable', 'sha_meta', 'coords_out', 'coords_error'):
            self.assertNotIn(key, report)
        self.assertNotIn('M1 SHA', stdout)
        self.assertIsNone(fixture)
        self.assertTrue(report['finite_output']['baseline'])

    def test_a_coordinate_failure_never_costs_the_timing_run(self):
        module = fake_ttnn()
        module.open_device = lambda device_id, l1_small_size: SimpleNamespace(
            compute_with_storage_grid_size=FakeDevice().compute_with_storage_grid_size)
        status, report, stdout, fixture = self.run_main(['--coords-out', 'COORDS'], module)
        self.assertEqual(status, 0)
        self.assertIn('AttributeError', report['coords_error'])
        self.assertIn('COORDS ERROR', stdout)
        self.assertIsNone(fixture)


class RecordingWatchdog:
    """A watchdog that only records which guard each device phase ran under, and its extra budget."""

    def __init__(self):
        self.guards = []

    @contextlib.contextmanager
    def guard(self, what, extra=0.0):
        self.guards.append((what, extra))
        yield


def run_options(**changes):
    options = SimpleNamespace(arms=['baseline', 'bf16_kv'], starts=[0, 128], start_mode='tensor', warmup=1, rounds=1,
                              device_id=0, seed=0, fallback_scalar=True, sha=False, coords_out=None)
    for key, value in changes.items():
        setattr(options, key, value)
    return options


@contextlib.contextmanager
def fake_module(module):
    saved = sys.modules.get('ttnn')
    sys.modules['ttnn'] = module
    try:
        yield
    finally:
        if saved is None:
            sys.modules.pop('ttnn', None)
        else:
            sys.modules['ttnn'] = saved


@unittest.skipIf(torch is None, 'torch not installed')
class BudgetTests(unittest.TestCase):
    def test_compile_heavy_phases_get_their_allowance(self):
        dog = RecordingWatchdog()
        with fake_module(fake_ttnn()):
            report = bench.run(run_options(), dog)
        self.assertTrue(report['passed'])
        extra = dict(dog.guards)
        self.assertEqual(extra['open_device'], bench.COMPILE_GRACE_S)          # firmware build on a fresh cache
        self.assertEqual(extra['build_inputs'], bench.COMPILE_GRACE_S)         # host bf8 tilize of the pools
        self.assertEqual(extra['baseline@0 warmup'], bench.FIRST_COMPILE_GRACE_S)   # each arm's first compile
        self.assertEqual(extra['bf16_kv@0 warmup'], bench.FIRST_COMPILE_GRACE_S)
        self.assertEqual(extra['baseline@128 warmup'], bench.COMPILE_GRACE_S)
        self.assertEqual(extra['baseline@128 timed'], 0.0)
        self.assertEqual(extra['close_device'], 0.0)
        self.assertEqual(120 + bench.FIRST_COMPILE_GRACE_S, 900)

    def test_no_fallback_scalar_leaves_a_failing_pair_failed(self):
        module = fake_ttnn()
        served = module.transformer.chunked_scaled_dot_product_attention

        def tensor_path_broken(chunk_start_idx_tensor=None, **kwargs):
            if chunk_start_idx_tensor is not None:
                raise RuntimeError('tensor path broken')
            return served(**kwargs)

        module.transformer = SimpleNamespace(chunked_scaled_dot_product_attention=tensor_path_broken)
        with fake_module(module):
            strict = bench.run(run_options(arms=['baseline'], fallback_scalar=False), RecordingWatchdog())
            lenient = bench.run(run_options(arms=['baseline']), RecordingWatchdog())
        self.assertFalse(strict['passed'])
        self.assertEqual(sorted(strict['errors']), ['baseline@0', 'baseline@128'])
        self.assertEqual(strict['paths'], {})
        self.assertTrue(lenient['passed'])
        self.assertEqual(set(lenient['paths'].values()), {'scalar'})     # what --no-fallback-scalar forbids


def elf32(segments, note=b'', entry=0x1000):
    """A minimal little-endian ELF32: PT_LOAD per (vaddr, bytes), then one PT_NOTE over `note`
    (bytes that are in the file but never loaded, like debug info)."""
    header_size, ph_size, count = 52, 32, len(segments) + 1
    offset = header_size + ph_size * count
    phdrs, body = b'', b''
    for vaddr, data in segments:
        phdrs += struct.pack('<IIIIIIII', 1, offset + len(body), vaddr, vaddr, len(data), len(data) + 16, 5, 4)
        body += data
    phdrs += struct.pack('<IIIIIIII', 4, offset + len(body), 0, 0, len(note), len(note), 4, 4)
    body += note
    ident = bench.ELF_MAGIC + bytes((1, 1, 1)) + bytes(9)
    header = ident + struct.pack('<HHIIIIIHHHHHH', 2, 0xF3, 1, entry, header_size, 0, 0, header_size, ph_size, count,
                                 40, 0, 0)
    return header + phdrs + body


def elf64(segments, entry=0x1000):
    header_size, ph_size = 64, 56
    offset = header_size + ph_size * len(segments)
    phdrs, body = b'', b''
    for vaddr, data in segments:
        phdrs += struct.pack('<IIQQQQQQ', 1, 5, offset + len(body), vaddr, vaddr, len(data), len(data), 4)
        body += data
    ident = bench.ELF_MAGIC + bytes((2, 1, 1)) + bytes(9)
    header = ident + struct.pack('<HHIQQQIHHHHHH', 2, 0xF3, 1, entry, header_size, 0, 0, header_size, ph_size,
                                 len(segments), 64, 0, 0)
    return header + phdrs + body


class KernelElfTests(unittest.TestCase):
    CODE = [(0x4000, b'code' * 8), (0x8000, b'data' * 4)]

    def test_the_load_digest_sees_loaded_bytes_only(self):
        base = bench.elf_load_digest(elf32(self.CODE, note=b'debug-1'))
        self.assertEqual(base[1], 48)
        self.assertEqual(bench.elf_load_digest(elf32(self.CODE, note=b'debug-2 longer')), base)   # not loaded
        self.assertNotEqual(bench.elf_load_digest(elf32([(0x4000, b'code' * 7 + b'codf'), self.CODE[1]])), base)
        self.assertNotEqual(bench.elf_load_digest(elf32([(0x4004, self.CODE[0][1]), self.CODE[1]])), base)
        self.assertNotEqual(bench.elf_load_digest(elf32(self.CODE, entry=0x1004)), base)
        wide = bench.elf_load_digest(elf64(self.CODE))
        self.assertEqual(wide[1], 48)
        self.assertEqual(bench.elf_load_digest(elf64(self.CODE)), wide)

    def test_non_elf_or_truncated_input_gives_none(self):
        self.assertIsNone(bench.elf_load_digest(b'not an elf at all' * 8))
        self.assertIsNone(bench.elf_load_digest(elf32(self.CODE)[:60]))
        self.assertIsNone(bench.elf_load_digest(elf32(self.CODE)[:-10]))       # a PT_LOAD runs past the end
        self.assertIsNone(bench.elf_load_digest(b''))

    def cache(self, root, top='tt-metal-cache123', reader=None, note=b'n'):
        base = Path(root) / top / 'kernels'
        files = {
            'reader_interleaved/111/ncrisc/ncrisc.elf': reader or elf32(self.CODE, note=note),
            'reader_interleaved/111/ncrisc/ncrisck.9_99.o': b'object',
            'writer_interleaved/222/brisc/brisc.elf': elf32([(0x4000, b'writer')]),
        }
        for name, data in files.items():
            (base / name).parent.mkdir(parents=True, exist_ok=True)
            (base / name).write_bytes(data)
        (Path(root) / 'mykernels' / 'reader_interleaved').mkdir(parents=True)
        (Path(root) / 'mykernels' / 'reader_interleaved' / 'x.elf').write_bytes(elf32([(0, b'x')]))

    def report(self, **kwargs):
        with tempfile.TemporaryDirectory() as root:
            self.cache(root, **kwargs)
            return bench.kernel_elf_report(root, 'reader_interleaved')

    def test_the_report_digests_that_kernels_elfs_only(self):
        report = self.report()
        self.assertEqual([entry['path'] for entry in report['files']],
                         ['kernels/reader_interleaved/111/ncrisc/ncrisc.elf'])
        self.assertEqual(report['other_files'], ['kernels/reader_interleaved/111/ncrisc/ncrisck.9_99.o'])
        self.assertEqual(report['files'][0]['load_bytes'], 48)
        self.assertRegex(report['digest'], '^[0-9a-f]{64}$')
        self.assertEqual(self.report(top='tt-metal-cache999', note=b'other debug')['digest'], report['digest'])
        self.assertNotEqual(self.report(reader=elf32([(0x4000, b'k0c!' * 8), self.CODE[1]]))['digest'], report['digest'])
        self.assertEqual(bench.kernel_elf_line(report), 'M1 KERNEL_ELF reader_interleaved %s files=1' % report['digest'])

    def test_no_cache_or_no_elf_gives_none_and_never_raises(self):
        missing = bench.kernel_elf_report(None, 'reader_interleaved')
        self.assertIsNone(missing['digest'])
        self.assertIn('no kernel cache directory', missing['error'])
        self.assertEqual(bench.kernel_elf_line(missing), 'M1 KERNEL_ELF reader_interleaved none files=0')
        with tempfile.TemporaryDirectory() as root:
            empty = bench.kernel_elf_report(root, 'reader_interleaved')
        self.assertIsNone(empty['digest'])
        self.assertEqual(empty['files'], [])

    @unittest.skipIf(torch is None, 'torch not installed')
    def test_main_prints_the_kernel_elf_line_after_the_sha_lines(self):
        with tempfile.TemporaryDirectory() as root:
            self.cache(root)
            expected = bench.kernel_elf_report(root, 'reader_interleaved')['digest']
            saved = os.environ.get('TT_METAL_CACHE')
            os.environ['TT_METAL_CACHE'] = root
            try:
                status, report, stdout, _ = main_with(['--sha', '--kernel-elf', 'reader_interleaved'], fake_ttnn())
            finally:
                if saved is None:
                    os.environ.pop('TT_METAL_CACHE', None)
                else:
                    os.environ['TT_METAL_CACHE'] = saved
        self.assertEqual(status, 0)
        self.assertEqual(report['kernel_elf']['digest'], expected)
        line = 'M1 KERNEL_ELF reader_interleaved %s files=1' % expected
        self.assertIn(line, stdout)
        self.assertLess(stdout.rindex('M1 SHA'), stdout.index(line))


class FlagTests(unittest.TestCase):
    def test_the_new_flags_parse_and_a_negative_watchdog_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'm1.json'
            with redirect_stdout(StringIO()):
                status = bench.main(['--out', str(out), '--arms', 'baseline', '--rounds', '1', '--sha',
                                     '--watchdog-s', '0', '--coords-out', str(Path(directory) / 'c.json')])
            self.assertEqual(status, 1)          # no ttnn here: the error report, as before
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            bench.main(['--out', 'unused.json', '--watchdog-s', '-1'])


class RunnerTests(unittest.TestCase):
    """run_m1.sh: card M only, never a shared card, a fresh kernel cache, the 131k serving image."""

    def setUp(self):
        self.text = (Path(__file__).parent / 'run_m1.sh').read_text(encoding='utf-8')

    def test_card_m_only_and_refuses_a_held_device(self):
        self.assertIn('CARD_M=/dev/tenstorrent/by-id/blackhole-CEF5729692C19E6D', self.text)
        self.assertIn('--device "$node"', self.text)
        self.assertEqual(self.text.count('--device '), 1)
        self.assertIn('holds a Tenstorrent device', self.text)
        self.assertLess(self.text.index('holds a Tenstorrent device'), self.text.index('### M1 $stamp'))

    def test_fresh_kernel_cache_and_the_pinned_image(self):
        self.assertIn('kcache-m1-$stamp,dst=/kcache', self.text)
        self.assertIn('-e TT_METAL_CACHE=/kcache', self.text)
        self.assertIn('sha256:0648ca9ad663acc72e7d8ea59d9cde0f9218b583ad74a60d58ff2f31bddd6fae', self.text)

    def test_the_bench_is_mounted_and_the_verdict_surfaced(self):
        self.assertIn('src=$S/sdpa_prefill_bench.py,dst=/bench/sdpa_prefill_bench.py,readonly', self.text)
        self.assertIn('/bench/sdpa_prefill_bench.py --out', self.text)
        self.assertIn("grep -F 'M1 VERDICT:'", self.text)
        self.assertNotIn(chr(13), self.text)


if __name__ == '__main__':
    unittest.main()
