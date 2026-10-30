"""CPU tests of the K-split card-M harness (ksplit_card_m.py), its statistic (scripts/ci/ksplit_numerics.py) and its runner (run_card_m_ks.sh).

The harness runs end to end against a fake ttnn whose chunked SDPA follows the factory's reading (pk.decode_word, the envelope, the refusals, one log line per program-cache miss, the planner's
chain counts) and computes the TRUE float attention for small contexts, so the statistic, the noise-floor reference, the matrix, the refusals, the program-cache check, the cap check, the
grid pass, the factory-log check and the timing verdict are exercised without a card. Torch is needed (skipped without it).

Run from this directory:  python3 -B -m unittest test_ksplit_card
"""

import contextlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

try:
    import torch
    torch.set_num_threads(2)
except ImportError:                          # the CPU suite installs torch; a bare checkout may not have it
    torch = None

HERE = Path(__file__).resolve().parent
CI = HERE.parents[2] / 'scripts' / 'ci'
for path in (HERE, HERE.parent / 'sdpa_prefill_oneq', HERE.parent / 'sdpa_prefill_chain', CI):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import apply_factory_pf as pf  # noqa: E402
import apply_factory_pk as pk  # noqa: E402
import apply_factory_ps as ps  # noqa: E402
import ksplit_card_m as card  # noqa: E402
import ksplit_numerics as numerics  # noqa: E402
import ksplit_planner as planner  # noqa: E402
import oneq_card_m as oneq_card  # noqa: E402

RUNNER = HERE / 'run_card_m_ks.sh'
BASH = shutil.which('bash')
NL = chr(10)
RESERVED = re.search(r"^QUAL_RESERVED_CARD=(\S+)", (CI / 'qual_card.sh').read_text(encoding='utf-8'), flags=re.M).group(1)      # read from the library: a new file does not name a board
FACTORY = pk.patch((HERE.parent / 'sdpa_prefill_chain' / 'fixtures' / 'sdpa_program_factory.fd8c0676.cpp').read_bytes()).decode('utf-8')


class FakeTensor:
    def __init__(self, host, dtype=None, memory=None):
        self.host, self.dtype, self.memory = host, dtype, memory
        self.shape = tuple(host.shape)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class FakeDevice:
    def __init__(self, grid=(13, 10)):
        self.programs = set()
        self.grid = grid

    def compute_with_storage_grid_size(self):
        return SimpleNamespace(x=self.grid[0], y=self.grid[1])

    def num_program_cache_entries(self):
        return len(self.programs)


TRUE_LIMIT = 4096          # contexts up to here get the true float attention (the reference check needs it); longer ones a cheap function


def fake_ttnn(device, clock, mode='ok', silent_ks_log=False, returns_on_hang=True):
    """The factory's reading of a call. mode: 'ok' = the K-split differs from oneq by a relative 3e-4 pattern; 'broken' = by 30 percent; 'nan' = one NaN element; 'exact' = identical."""
    memo = {}

    def output(q, k, v, table, start, word):
        rows = q.shape[2]
        if start + rows <= TRUE_LIMIT and k.shape[1] == 1:
            key = (id(q), id(k), id(v), id(table), start)
            if key not in memo:           # the three programs of a case see the same inputs: one true attention per case
                memo[key] = numerics.reference_attention(q.host.float()[0], numerics.gather_pages(k.host.float(), table.host.reshape(-1), start + rows),
                                                         numerics.gather_pages(v.host.float(), table.host.reshape(-1), start + rows), start, 1.0 / 16.0, torch)[None]
            truth = memo[key]
        else:
            truth = q.host.float() * (start + 1) + float(k.host.float().mean())
        flags = (word or 0) & 0xFFFF
        if word is not None and (word & 0xFFFF0000) == pk.PF_TAG and flags & pk.FLAG_KSPLIT:
            pattern = torch.sin(torch.arange(truth.numel(), dtype=torch.float32)).reshape(truth.shape)
            if mode == 'ok':
                truth = truth * (1 + 3e-4 * pattern)
            elif mode == 'broken':
                truth = truth * (1 + 0.3 * pattern)
            elif mode == 'nan':
                truth = truth.clone()
                truth.view(-1)[5] = float('nan')
        return truth.to(torch.bfloat16)

    def chunked(input_tensor_q, input_tensor_k, input_tensor_v, page_table_tensor, compute_kernel_config, program_config,
                chunk_start_idx_tensor=None, chunk_start_idx=None):
        word = program_config.get('max_cores_per_head_batch')
        grid = tuple(program_config['compute_with_storage_grid_size'])
        rows, nqh, nkh = input_tensor_q.shape[2], input_tensor_q.shape[1], input_tensor_k.shape[1]
        key = (nqh, rows, page_table_tensor.shape[1], input_tensor_q.memory, word, chunk_start_idx_tensor is None, input_tensor_k.dtype, grid)
        ks = oneq = False
        flags = 0
        if word is not None and (word & 0xFFFF0000) == ps.PF_TAG:
            try:
                chain, flags, oneq, ks = pk.decode_word(word, os.environ.get('QWEN_SDPA_PF_TEST'))
            except ValueError as error:
                raise RuntimeError('TT_FATAL %s' % error)
        else:
            chain = False
        if key not in device.programs and chain:                  # the factory runs on a program-cache miss only
            q_chunks = rows // 128
            if chunk_start_idx_tensor is None or input_tensor_k.dtype != 'bf8' or q_chunks % 2 or nqh % nkh:
                raise RuntimeError('TT_FATAL [QWEN-SDPA-PF] kv_chain outside its qualified envelope')
            if ks:
                mine = planner.plan(nqh, nkh, rows, 128, grid, noc_order=bool(flags & 4))
                if mine['refusal']:
                    raise RuntimeError('TT_FATAL ' + mine['refusal'])
            else:
                mine = None
            base = oneq_card.planner.plan(nqh, nkh, rows, 128, grid, oneq=oneq, noc_order=bool(flags & 4))
            if oneq and base['refusal']:
                raise RuntimeError('TT_FATAL ' + base['refusal'])
            os.write(1, ('[QWEN-SDPA-PF] flags=%#x kv_chain=1 chains=%d members=%d order=%s\n'
                         % (flags, mine['chain_count'] if ks else base['chain_count'], mine['member_count'] if ks else base['member_count'],
                            'noc' if flags & 4 else 'raster')).encode())
            if oneq:
                os.write(1, ('[QWEN-SDPA-PF] oneq=1 q_chunks=%d cores=%d chunks_per_core=1 chains=%d members=%d\n'
                             % (base['total'], base['num_cores'], base['chain_count'], base['member_count'])).encode())
            if ks and not silent_ks_log:
                os.write(1, (planner.LOG_TEXT.format(mine['slices'], mine['total'], mine['num_cores'], mine['busy'], mine['chain_count'], mine['member_count']) + '\n').encode())
        if key not in device.programs:
            device.programs.add(key)
        start = int(chunk_start_idx_tensor.host[0]) if chunk_start_idx_tensor is not None else chunk_start_idx
        if ks and not (flags & pk.FLAG_KS_HANG):
            clock.now += planner.layer_us(planner.plan(nqh, nkh, rows, 128, grid), start) / 1e6
        else:
            clock.now += oneq_card.planner.layer_us(oneq_card.planner.plan(nqh, nkh, rows, 128, grid, oneq=bool(oneq)), start) / 1e6
        return FakeTensor(output(input_tensor_q, input_tensor_k, input_tensor_v, page_table_tensor, start, word))

    return SimpleNamespace(
        bfloat8_b='bf8', bfloat16='bf16', int32='int32', TILE_LAYOUT='tile', ROW_MAJOR_LAYOUT='rm', DRAM_MEMORY_CONFIG='dram', L1_MEMORY_CONFIG='l1',
        MathFidelity=SimpleNamespace(HiFi2='hifi2'), open_device=lambda **kw: device, close_device=lambda dev: None, deallocate=lambda tensor: None,
        synchronize_device=lambda dev: None, from_torch=lambda host, dtype=None, layout=None, device=None, memory_config=None: FakeTensor(host.clone(), dtype, memory_config),
        to_torch=lambda tensor: tensor.host, WormholeComputeKernelConfig=lambda **kw: kw, SDPAProgramConfig=lambda **kw: kw,
        transformer=SimpleNamespace(chunked_scaled_dot_product_attention=chunked))


class StatisticTests(unittest.TestCase):
    @unittest.skipIf(torch is None, 'torch not installed')
    def test_the_unit_is_the_bf16_spacing_below_twice_the_row_scale(self):
        scale = torch.tensor([1.0, 1.5, 2.0, 0.75, 100.0])
        unit = numerics.row_unit(scale, torch)
        self.assertEqual(unit.tolist(), [2.0 ** -7, 2.0 ** -7, 2.0 ** -6, 2.0 ** -8, 2.0 ** -1])
        self.assertEqual(float(torch.tensor(1.0).to(torch.bfloat16)) + 2.0 ** -7, float(torch.tensor(1.0078125).to(torch.bfloat16)))   # the spacing just above 1

    @unittest.skipIf(torch is None, 'torch not installed')
    def test_identical_outputs_have_zero_error_and_one_ulp_flip_of_the_largest_element_is_one_unit(self):
        a = torch.randn(1, 2, 8, 16).to(torch.bfloat16)
        stats = numerics.error_stats(a, a, torch)
        self.assertEqual((stats['max_ulps'], stats['rms_ulps'], stats['nonfinite']), (0.0, 0.0, 0))
        self.assertEqual(numerics.within_bound(stats), [])
        b = a.clone()
        row = a[0, 0, 0].abs()
        index = int(row.argmax())
        b[0, 0, 0, index] = (a[0, 0, 0, index].float() + float(numerics.row_unit(row.max().float(), torch))).to(torch.bfloat16)
        self.assertAlmostEqual(numerics.error_stats(b, a, torch)['max_ulps'], 1.0, places=5)

    @unittest.skipIf(torch is None, 'torch not installed')
    def test_the_bound_has_teeth(self):
        a = torch.randn(1, 6, 16, 256).to(torch.bfloat16)
        for factor, ok in ((1.0 + 2e-3, True), (1.3, False)):
            stats = numerics.error_stats((a.float() * factor).to(torch.bfloat16), a, torch)
            self.assertEqual(numerics.within_bound(stats) == [], ok, (factor, stats))
        broken = a.clone()
        broken.view(-1)[3] = float('nan')
        self.assertTrue(any('non-finite' in p for p in numerics.within_bound(numerics.error_stats(broken, a, torch))))
        shape_error = None
        try:
            numerics.error_stats(a, a[:, :3], torch)
        except ValueError as error:
            shape_error = str(error)
        self.assertIn('differ', shape_error)

    def test_the_noise_rule(self):
        base = dict(rms_ulps=0.30, max_ulps=2.0)
        self.assertEqual(numerics.noise_problems(dict(rms_ulps=0.36, max_ulps=2.9), base), [])
        self.assertEqual(len(numerics.noise_problems(dict(rms_ulps=0.40, max_ulps=2.0), base)), 1)
        self.assertEqual(len(numerics.noise_problems(dict(rms_ulps=0.40, max_ulps=3.5), base)), 2)

    @unittest.skipIf(torch is None, 'torch not installed')
    def test_the_reference_is_causal_attention(self):
        generator = torch.Generator().manual_seed(3)
        q = torch.randn(2, 128, 8, generator=generator)
        k = torch.randn(384, 8, generator=generator)
        v = torch.randn(384, 8, generator=generator)
        out = numerics.reference_attention(q, k, v, 256, 0.25, torch)
        for row in (0, 17, 127):
            visible = 256 + row + 1
            scores = (q[:, row] @ k[:visible].T) * 0.25
            self.assertTrue(torch.allclose(out[:, row], torch.softmax(scores, dim=-1) @ v[:visible], atol=1e-5))
        pool = torch.arange(6 * 4 * 5, dtype=torch.float32).reshape(6, 1, 4, 5)
        table = torch.tensor([3, 0, 5, 1, 2, 4])
        gathered = numerics.gather_pages(pool, table, 10)
        self.assertEqual(gathered.shape, (10, 5))
        self.assertTrue(torch.equal(gathered[:4], pool[3, 0]) and torch.equal(gathered[4:8], pool[0, 0]) and torch.equal(gathered[8:10], pool[5, 0][:2]))


class HelperTests(unittest.TestCase):
    def test_the_configs_and_the_planner(self):
        native = (13, 10)
        self.assertEqual([(name, card.config_plan(rows, grid, native)['slices']) for name, rows, grid in card.CONFIGS], [('s2', 2), ('s4', 4), ('s2g', 2)])
        self.assertEqual(card.config_plan(2048, None, native)['slices'], 1)
        self.assertIn('needs at least 2 slices', card.config_plan(2048, None, native)['refusal'])

    def test_the_words(self):
        self.assertEqual(card.word(card.ARMS['ksplit']), 0x5EFA001B)
        self.assertEqual(card.word(card.ARMS['oneq']), 0x5EFA000B)
        self.assertIsNone(card.word(card.ARMS['stock']))
        self.assertEqual(card.HANG_FLAGS, 0x41B)
        self.assertIn('%x' % card.HANG_FLAGS, card.HANG_CALL_LABEL)

    def test_every_refusal_needle_is_a_string_of_the_factory_or_the_planner(self):
        served_text = (HERE.parent / 'sdpa_prefill_chain' / 'fixtures' / 'sdpa_program_factory.fd8c0676.cpp').read_text(encoding='utf-8')
        known = FACTORY + served_text
        for name, _geometry, _change, flags, rows, needle in card.REFUSALS:
            self.assertIn(needle, known, name)
        self.assertEqual(len(card.REFUSALS), len({entry[0] for entry in card.REFUSALS}))

    def test_the_refusals_follow_the_decoder_and_the_planner(self):
        native = (13, 10)
        for name, geometry, change, flags, rows, needle in card.REFUSALS:
            if flags & ~0xFFFF or (flags in (0x10, 0x11, 0x40B)):
                with self.assertRaises(ValueError, msg=name):
                    pk.decode_word(pk.PF_TAG | flags)
                continue
            if flags == 0x41B:
                with self.assertRaises(ValueError, msg=name):
                    pk.decode_word(pk.PF_TAG | flags)
                continue
            if change or rows % 256:
                continue
            reading = card.ks_refusal(geometry, rows, change.get('grid'), native)
            self.assertIsNotNone(reading, name)

    def test_expected_log_lines_and_the_check(self):
        requested = {('tp4', 1024, 4096, 'dram', 0x1B, (13, 10)), ('tp4', 512, 4096, 'dram', 0x1B, (13, 10)), ('tp4', 1024, 4096, 'dram', 0xB, (13, 10)),
                     ('tp4', 1024, 4096, 'dram', 0x3, (13, 10))}
        flags_lines, oneq_lines, ks_lines = card.expected_programs(requested)
        self.assertEqual(sorted(ks_lines), [(2, 48, 130, 96, 16, 96), (4, 24, 130, 96, 16, 96)])
        self.assertIn((0x1B, 16, 96, 'raster'), flags_lines)              # the F4 line of a ksplit program counts ITS chains: one per slice
        self.assertIn((0xB, 8, 48, 'raster'), flags_lines)                # the oneq program of the same rows: half as many
        self.assertIn((0x3, 4, 24, 'raster'), flags_lines)                # the paired chain call at 1,024 rows: 24 busy cores, 4 chains
        text = ''.join(('[QWEN-SDPA-PF] flags=%#x kv_chain=1 chains=%d members=%d order=%s' % key + NL) * count for key, count in flags_lines.items())
        text += ''.join(('[QWEN-SDPA-PF] oneq=1 q_chunks=%d cores=%d chunks_per_core=1 chains=%d members=%d' % key + NL) * count for key, count in oneq_lines.items())
        text += ''.join((planner.LOG_TEXT.format(*key) + NL) * count for key, count in ks_lines.items())
        self.assertEqual(card.check_factory_lines(text, requested), [])
        lines = text.splitlines()
        dropped = NL.join(line for line in lines if not line.startswith('[QWEN-SDPA-PF] ksplit=1 slices=4')) + NL
        self.assertTrue(any(problem.startswith('ksplit line') for problem in card.check_factory_lines(dropped, requested)))
        self.assertTrue(any(problem.startswith('flags line') for problem in card.check_factory_lines(text.replace('chains=16 members=96 order=raster', 'chains=8 members=48 order=raster'), requested)))

    def test_the_timing_verdict(self):
        def block(ratios, config='s2'):
            timing = {config: {'dram': {'oneq': {}, 'ksplit': {}}}}
            for start, ratio in ratios.items():
                timing[config]['dram']['oneq'][str(start)] = dict(median_ms=10.0)
                timing[config]['dram']['ksplit'][str(start)] = dict(median_ms=10.0 * ratio)
            return card.timing_verdict(timing, config)['dram']
        win = block({0: 1.0, 2048: 0.8, 32768: 0.55, 65536: 0.52, 253952: 0.5})
        self.assertEqual((win['verdict'], win['small_loss']), ('KS-WIN', False))
        self.assertAlmostEqual(win['ratio'], 0.52)
        self.assertEqual(block({32768: 0.9, 65536: 0.9})['verdict'], 'KS-PARTIAL')
        self.assertEqual(block({32768: 1.0, 65536: 0.99})['verdict'], 'KS-NO-WIN')
        self.assertTrue(block({2048: 1.08, 32768: 0.5, 65536: 0.5})['small_loss'])
        self.assertEqual(block({})['verdict'], 'KS-NO-WIN')

    def test_the_argument_parser(self):
        out = Path(tempfile.gettempdir()) / 'x.json'
        args = card.parse_args(['--out', str(out)])
        self.assertEqual(args.configs, ['s2', 's4', 's2g'])
        self.assertEqual(len(args.starts), 15)
        self.assertEqual(len(args.time_starts), 8)
        for bad in (['--configs', 's9'], ['--starts', '100'], ['--variants', 'weird'], ['--time-q-memory', 'l2'], ['--starts', '999999']):
            with self.assertRaises(SystemExit, msg=bad):
                with contextlib.redirect_stderr(io.StringIO()):
                    card.parse_args(['--out', str(out)] + bad)
        self.assertTrue(card.parse_args(['--out', str(out), '--role', 'hang']).no_timing)


@unittest.skipIf(torch is None, 'torch not installed')
class HarnessTests(unittest.TestCase):
    SMALL = ['--starts', '0,128,2048', '--seeds', '0', '--variants', 'normal', '--q-memory', 'dram,l1', '--time-starts', '2048,32768,65536', '--rounds', '3', '--warmup', '1',
             '--watchdog', '0', '--no-maps', '--reference-max-start', '4096']

    def setUp(self):
        self.saved = (oneq_card.POOL_BLOCKS, oneq_card.TP2_POOL_BLOCKS, card.GRID_STARTS, oneq_card.time.perf_counter, sys.modules.get('ttnn'), os.environ.get('QWEN_SDPA_PF_TEST'))
        oneq_card.POOL_BLOCKS, oneq_card.TP2_POOL_BLOCKS, card.GRID_STARTS = 1100, 72, (0, 2048)
        os.environ.pop('QWEN_SDPA_PF_TEST', None)

    def tearDown(self):
        oneq_card.POOL_BLOCKS, oneq_card.TP2_POOL_BLOCKS, card.GRID_STARTS, oneq_card.time.perf_counter = self.saved[:4]
        if self.saved[4] is None:
            sys.modules.pop('ttnn', None)
        else:
            sys.modules['ttnn'] = self.saved[4]
        if self.saved[5] is None:
            os.environ.pop('QWEN_SDPA_PF_TEST', None)
        else:
            os.environ['QWEN_SDPA_PF_TEST'] = self.saved[5]

    def main(self, directory, extra=(), grid=(13, 10), **fake):
        clock = Clock()
        device = FakeDevice(grid)
        sys.modules['ttnn'] = fake_ttnn(device, clock, **fake)
        oneq_card.time.perf_counter = clock
        out = Path(directory) / 'ksplit.json'
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            status = card.main(['--out', str(out)] + self.SMALL + list(extra))
        return status, json.loads(out.read_text()), buffer.getvalue(), device

    def test_a_clean_run_passes_every_check(self):
        with tempfile.TemporaryDirectory() as directory:
            status, report, stdout, _ = self.main(directory)
            self.assertEqual(status, 0, report['failures'] or report.get('error'))
            self.assertTrue(report['passed'])
            self.assertEqual(len(report['cases']), 3 * 2 * 1 * 1 * 3)                 # configs x Q memory x seed x variant x start
            self.assertTrue(all(not case['problems'] for case in report['cases']))
            self.assertTrue(all(case['stats']['max_ulps'] > 0 for case in report['cases']), 'the K-split fake differs from oneq: the statistic sees it')
            referenced = [case for case in report['cases'] if 'ksplit_vs_reference' in case]
            self.assertGreater(len(referenced), 2)
            self.assertTrue(all(not case['noise_problems'] for case in referenced), [case['noise_problems'] for case in referenced])
            anchored = [case for case in report['cases'] if 'oneq_equals_stock' in case]
            self.assertTrue(anchored and all(case['oneq_equals_stock'] for case in anchored))
            self.assertEqual(len(report['grid_cases']), 4)                            # s2, s4 x 2 starts on the 11 x 10 grid
            self.assertEqual(report['cache']['after_ksplit'], report['cache']['after_oneq'] + 1)
            self.assertEqual(report['cache']['after_starts'], report['cache']['after_ksplit'])
            self.assertEqual(len(report['refusals']), len(card.REFUSALS))
            self.assertTrue(all(entry['refused'] and entry['matched'] for entry in report['refusals'].values()), report['refusals'])
            self.assertEqual(report['cap'], dict(rows=256, slices=4, busy=48))
            self.assertEqual(report['numerics']['nonfinite'], 0)
            self.assertIn('KSPLIT_CARD_M role=numerics verdict=PASS', stdout)
            verdicts = report['timing_verdict']
            self.assertEqual(verdicts['s2']['dram']['verdict'], 'KS-WIN')
            self.assertEqual(verdicts['s4']['dram']['verdict'], 'KS-WIN')
            self.assertLess(verdicts['s4']['dram']['ratio'], verdicts['s2']['dram']['ratio'])
            self.assertFalse(verdicts['s2']['dram']['small_loss'])
            self.assertIn('KSPLIT TIME_VERDICT config=s2 q=dram KS-WIN', stdout)

    def test_the_factory_log_matches_the_planner_for_every_program(self):
        with tempfile.TemporaryDirectory() as directory:
            status, report, _stdout, _ = self.main(directory, ['--no-timing', '--configs', 's2,s4'])
            self.assertEqual(status, 0, report['failures'])
            log = (Path(directory) / 'ksplit.json.native.log').read_text()
            lines = [tuple(int(v) for v in m.groups()) for m in card.KS_LINE.finditer(log)]
            self.assertIn((2, 48, 130, 96, 16, 96), lines)
            self.assertIn((4, 24, 130, 96, 16, 96), lines)
            self.assertIn((2, 48, 110, 96, 16, 96), lines)                            # the 11 x 10 grid pass
            self.assertIn((4, 24, 110, 96, 16, 96), lines)
            self.assertIn((4, 12, 130, 48, 8, 48), lines)                             # the cap check at 256 rows

    def test_a_ksplit_output_outside_the_bound_fails_the_run(self):
        with tempfile.TemporaryDirectory() as directory:
            status, report, stdout, _ = self.main(directory, ['--no-timing'], mode='broken')
            self.assertEqual(status, 1)
            self.assertFalse(report['passed'])
            self.assertTrue(all(case['problems'] for case in report['cases']))
            self.assertTrue(any('max_ulps' in failure for failure in report['failures']))
            self.assertIn('KSPLIT_CARD_M role=numerics verdict=FAIL', stdout)

    def test_a_nan_fails_the_run(self):
        with tempfile.TemporaryDirectory() as directory:
            status, report, _stdout, _ = self.main(directory, ['--no-timing', '--configs', 's2'], mode='nan')
            self.assertEqual(status, 1)
            self.assertGreater(report['numerics']['nonfinite'], 0)
            self.assertTrue(any('non-finite' in failure for failure in report['failures']))

    def test_the_noise_floor_catches_an_output_further_from_the_truth_than_oneq_even_inside_the_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            numerics.BOUND, saved = dict(numerics.BOUND, max_ulps=1e9, rms_ulps=1e9, frac_gt2=1.0), numerics.BOUND
            try:
                status, report, _stdout, _ = self.main(directory, ['--no-timing', '--configs', 's2'], mode='broken')
            finally:
                numerics.BOUND = saved
            self.assertEqual(status, 1)
            self.assertTrue(any('noise floor' in failure for failure in report['failures']), report['failures'][:3])

    def test_a_missing_ksplit_log_line_fails_the_run(self):
        with tempfile.TemporaryDirectory() as directory:
            status, report, _stdout, _ = self.main(directory, ['--no-timing', '--configs', 's2'], silent_ks_log=True)
            self.assertEqual(status, 1)
            self.assertTrue(any(failure.startswith('factory log: ksplit line') for failure in report['failures']), report['failures'])

    def test_the_planted_hang_that_returns_is_a_failure_and_needs_the_test_variable(self):
        with tempfile.TemporaryDirectory() as directory:
            status, report, _stdout, _ = self.main(directory, ['--role', 'hang', '--configs', 's2'])
            self.assertEqual(status, 1)
            self.assertTrue(any('needs QWEN_SDPA_PF_TEST=1' in failure for failure in report['failures']))
        with tempfile.TemporaryDirectory() as directory:
            os.environ['QWEN_SDPA_PF_TEST'] = '1'
            status, report, stdout, _ = self.main(directory, ['--role', 'hang', '--configs', 's2'])
            self.assertEqual(status, 1)
            self.assertTrue(any('planted ksplit hang RETURNED' in failure for failure in report['failures']), report['failures'])
            self.assertTrue(report['hang']['armed'])

    def test_the_run_keeps_a_clean_report_when_the_binary_check_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            args = ['--out', str(Path(directory) / 'r.json')] + self.SMALL
            args = [value for value in args if value != '--no-maps']
            sys.modules['ttnn'] = fake_ttnn(FakeDevice(), Clock())
            with contextlib.redirect_stdout(io.StringIO()):
                status = card.main(args)
            report = json.loads((Path(directory) / 'r.json').read_text())
            self.assertEqual(status, 1)
            self.assertTrue(report.get('error') or report['failures'])


@unittest.skipIf(BASH is None, 'bash not found')
class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.text = RUNNER.read_text(encoding='utf-8')

    def dry(self, home, **extra):
        env = {key: value for key, value in os.environ.items() if key not in ('QUAL_CARD', 'ALLOW_SERVING_CARD', 'WATCHER', 'KS_ARGS', 'CARD_B_ARGS', 'RESULTS', 'KOPGRAFT64', 'IMAGE',
                                                                              'EXPECT_TTNNCPP_SHA256', 'KS_CARD_DRY_RUN', 'QUAL_TT_GRID', 'WATCHDOG_S')}
        env.update(HOME=str(home), QUAL_CARD='blackhole-0000000000000001', KS_CARD_DRY_RUN='1')
        env.update(extra)
        args = [BASH, str(RUNNER)] + ([extra.pop('ROLE')] if 'ROLE' in extra else [])
        env.pop('ROLE', None)
        return subprocess.run(args, env=env, capture_output=True, text=True)

    def test_the_text(self):
        raw = RUNNER.read_bytes()
        self.assertNotIn(b'\r', raw)
        self.assertEqual(subprocess.run([BASH, '-n', str(RUNNER)], capture_output=True, text=True).returncode, 0)
        self.assertNotRegex(self.text, r'blackhole-[0-9A-F]{8}')
        self.assertNotIn('/dev/tenstorrent/', '\n'.join(line for line in self.text.splitlines() if not line.lstrip().startswith('#')))
        self.assertIn('. "$lib"', self.text)
        code = [line for line in self.text.splitlines() if not line.lstrip().startswith('#')]
        body = NL.join(code)
        self.assertLess(body.index('. "$lib"'), body.index('qual_card_select'))
        self.assertLess(body.index('qual_refuse_holders'), body.index('qual_card_recheck'))
        self.assertLess(body.index('qual_card_recheck'), body.index('timeout -k 30'))
        self.assertEqual(body.count('docker run'), 1)
        self.assertNotIn('qual_reset ', body)
        self.assertNotRegex(body, r'tt-smi -r')

    def test_the_dry_run_prints_a_read_only_graft_launch_on_the_named_card(self):
        with tempfile.TemporaryDirectory() as home:
            result = self.dry(home, IMAGE='img:tag', EXPECT_TTNNCPP_SHA256='ab' * 32, KS_ARGS='--configs s2')
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            out = result.stdout
            self.assertIn('### dry run: nothing launched', out)
            argv = [line for line in out.splitlines() if line.startswith('### argv: ')][0].replace('\\', '')
            self.assertIn('--device ', argv)
            self.assertIn('--network none', argv)
            for mount in ('dst=/ops/sdpa_prefill_ksplit,readonly', 'dst=/ops/sdpa_prefill_oneq,readonly', 'dst=/ops/sdpa_prefill_chain,readonly', 'dst=/ops/ci/ksplit_numerics.py,readonly',
                          'dst=/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa,readonly', 'dst=/opt/tt-metal/build_Release/lib/_ttnncpp.so,readonly'):
                self.assertIn(mount, argv)
            self.assertIn('ksplit --out /results/ksplit-numerics-', argv)
            self.assertIn('--configs s2', argv)
            self.assertNotIn('QWEN_SDPA_PF_TEST', argv)
            self.assertIn('QWEN_C2_SERVING=0', argv)

    def test_the_watcher_pass_is_narrow_and_sanitised(self):
        with tempfile.TemporaryDirectory() as home:
            result = self.dry(home, IMAGE='img:tag', WATCHER='1')
            self.assertEqual(result.returncode, 0, result.stderr)
            argv = [line for line in result.stdout.splitlines() if line.startswith('### argv: ')][0].replace('\\', '')
            self.assertIn('TT_METAL_WATCHER=5', argv)
            for word in ('--configs s2,s4', '--no-timing', '--no-grids', '--seeds 0', '--q-memory dram'):
                self.assertIn(word, argv)
            self.assertIn('WATCHER=1', result.stdout)

    def test_the_hang_role_sets_the_test_variable_and_nothing_else_changes(self):
        with tempfile.TemporaryDirectory() as home:
            result = self.dry(home, IMAGE='img:tag', ROLE='hang')
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            argv = [line for line in result.stdout.splitlines() if line.startswith('### argv: ')][0].replace('\\', '')
            self.assertIn('QWEN_SDPA_PF_TEST=1', argv)
            self.assertIn('--role hang', argv)
            self.assertIn('ksplit-hang-', argv)

    def test_the_runner_refuses_a_bad_selection_before_anything(self):
        with tempfile.TemporaryDirectory() as home:
            for env, needle in ((dict(QUAL_CARD=''), 'QUAL_CARD is not set'), (dict(QUAL_CARD=RESERVED), 'reserved for another project'),
                                (dict(QUAL_CARD='../x'), 'is not a board id')):
                result = self.dry(home, **env)
                self.assertEqual(result.returncode, 1, env)
                self.assertIn(needle, result.stderr)
            wrong = subprocess.run([BASH, str(RUNNER), 'sideways'], env=dict(os.environ, HOME=home, QUAL_CARD='blackhole-0000000000000001', KS_CARD_DRY_RUN='1'), capture_output=True, text=True)
            self.assertEqual(wrong.returncode, 1)
            self.assertIn('usage', wrong.stderr)
            noexpect = subprocess.run([BASH, str(RUNNER)], env=dict(os.environ, HOME=home, QUAL_CARD='blackhole-0000000000000001', IMAGE='x', ALLOW_SERVING_CARD='1'), capture_output=True, text=True)
            self.assertEqual(noexpect.returncode, 1)
            self.assertIn('EXPECT_TTNNCPP_SHA256 is required', noexpect.stderr)

    def test_the_graft_is_checked_when_it_exists(self):
        with tempfile.TemporaryDirectory() as home:
            graft = Path(home) / 'opgraft-K64j-OQ-KS'
            graft.mkdir()
            result = self.dry(home, IMAGE='img:tag')
            self.assertEqual(result.returncode, 1)
            self.assertIn('missing (the harness needs graft K64j-OQ-KS', result.stderr)


if __name__ == '__main__':
    unittest.main()
