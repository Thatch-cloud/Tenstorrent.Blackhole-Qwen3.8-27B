"""CPU tests of the oneq card-M harness (oneq_card_m.py), its runner (run_card_m_oq.sh) and the estimated-vs-measured report.

The harness runs end to end against a fake ttnn whose chunked SDPA follows the factory's reading (decode_word, the envelope, the oneq
refusal, one log line per program-cache miss, the planner's chain counts) and whose clock advances by the planner's modelled call time,
so the matrix, the refusals, the program-cache check, the stress, the grid and TP2 passes, the factory-log check, the timing verdict
and the report are exercised without a card. Torch is needed for the fake tensors (skipped without it).

Run from scripts/ci:  python -B -m unittest test_sdpa_oneq_card
"""

import contextlib
import io
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

try:
    import torch
except ImportError:                          # the CPU suite installs torch; a bare checkout may not have it
    torch = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ONEQ = ROOT / 'optimisation' / 'ttnn-op' / 'sdpa_prefill_oneq'
CHAIN = ROOT / 'optimisation' / 'ttnn-op' / 'sdpa_prefill_chain'
for path in (ONEQ, CHAIN, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import apply_factory_ps as ps  # noqa: E402
import oneq_card_m as card  # noqa: E402
import oneq_planner as planner  # noqa: E402
import oneq_report as report_lib  # noqa: E402

RUNNER = ONEQ / 'run_card_m_oq.sh'
NL = chr(10)


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


def fake_ttnn(device, clock, broken=None, silent_oneq_log=False, wrong_chain_count=False):
    """The factory's reading of a call. broken='oneq' changes the oneq arms' bytes; silent_oneq_log omits the oneq log line;
    wrong_chain_count logs the paired chain count for a oneq program. The clock moves by the modelled layer time of the arm."""
    def output(q, k, start, program_word):
        host = q.host.float() * (start + 1) + float(k.host.float().mean())
        if broken == 'oneq' and program_word is not None and (program_word & ps.FLAG_ONEQ) and (program_word & 0xFFFF0000) == ps.PF_TAG:
            host = host + 1
        return host.to(torch.bfloat16)

    def chunked(input_tensor_q, input_tensor_k, input_tensor_v, page_table_tensor, compute_kernel_config, program_config,
                chunk_start_idx_tensor=None, chunk_start_idx=None):
        word = program_config.get('max_cores_per_head_batch')
        grid = tuple(program_config['compute_with_storage_grid_size'])
        rows, nqh, nkh = input_tensor_q.shape[2], input_tensor_q.shape[1], input_tensor_k.shape[1]
        key = (nqh, rows, page_table_tensor.shape[1], input_tensor_q.memory, word, chunk_start_idx_tensor is None,
               input_tensor_k.dtype, grid)
        oneq = False
        if word is not None and (word & 0xFFFF0000) == ps.PF_TAG:
            try:
                chain, flags, oneq = ps.decode_word(word, os.environ.get('QWEN_SDPA_PF_TEST'))
            except ValueError as error:
                raise RuntimeError('TT_FATAL %s' % error)
        else:
            chain, flags = False, 0
        if key not in device.programs and chain:                  # the factory runs on a program-cache miss only
            q_chunks = rows // 128
            if (chunk_start_idx_tensor is None or input_tensor_k.dtype != 'bf8' or q_chunks % 2 or nqh % nkh):
                raise RuntimeError('TT_FATAL [QWEN-SDPA-PF] kv_chain outside its qualified envelope')
            the_plan = planner.plan(nqh, nkh, rows, 128, grid, oneq=oneq, noc_order=bool(flags & 4))
            if oneq and the_plan['refusal']:
                raise RuntimeError('TT_FATAL ' + the_plan['refusal'])
            chains = the_plan['chain_count'] // 2 if wrong_chain_count and oneq else the_plan['chain_count']
            os.write(1, ('[QWEN-SDPA-PF] flags=%#x kv_chain=1 chains=%d members=%d order=%s\n'
                         % (flags, chains, the_plan['member_count'], 'noc' if flags & 4 else 'raster')).encode())
            if oneq and not silent_oneq_log:
                os.write(1, ('[QWEN-SDPA-PF] oneq=1 q_chunks=%d cores=%d chunks_per_core=1 chains=%d members=%d\n'
                             % (the_plan['total'], the_plan['num_cores'], the_plan['chain_count'], the_plan['member_count'])).encode())
        if key not in device.programs:
            device.programs.add(key)
        start = int(chunk_start_idx_tensor.host[0]) if chunk_start_idx_tensor is not None else chunk_start_idx
        plan_for_time = planner.plan(nqh, nkh, rows, 128, grid, oneq=bool(chain and oneq))
        clock.now += planner.layer_us(plan_for_time, start) / 1e6
        return FakeTensor(output(input_tensor_q, input_tensor_k, start, word))

    return SimpleNamespace(
        bfloat8_b='bf8', bfloat16='bf16', int32='int32', TILE_LAYOUT='tile', ROW_MAJOR_LAYOUT='rm',
        DRAM_MEMORY_CONFIG='dram', L1_MEMORY_CONFIG='l1', MathFidelity=SimpleNamespace(HiFi2='hifi2'),
        open_device=lambda **kw: device, close_device=lambda dev: None, deallocate=lambda tensor: None,
        synchronize_device=lambda dev: None,
        from_torch=lambda host, dtype=None, layout=None, device=None, memory_config=None: FakeTensor(host.clone(), dtype, memory_config),
        to_torch=lambda tensor: tensor.host, WormholeComputeKernelConfig=lambda **kw: kw, SDPAProgramConfig=lambda **kw: kw,
        transformer=SimpleNamespace(chunked_scaled_dot_product_attention=chunked))


@unittest.skipIf(torch is None, 'torch not installed')
class HarnessTests(unittest.TestCase):
    SMALL = ['--rows', '2048,1024,512', '--starts', '0,128,2048,4096', '--seeds', '0', '--variants', 'normal,peaky',
             '--q-memory', 'dram,l1', '--alternations', '12', '--stress-starts', '0,2048', '--time-starts', '0,2048,4096',
             '--rounds', '3', '--warmup', '1', '--watchdog', '0', '--no-maps']

    def setUp(self):
        self.saved = (card.POOL_BLOCKS, card.TP2_POOL_BLOCKS, card.TP2_STARTS, card.GRID_STARTS, card.time.perf_counter,
                      sys.modules.get('ttnn'), os.environ.get('QWEN_SDPA_PF_TEST'))
        card.POOL_BLOCKS, card.TP2_POOL_BLOCKS, card.TP2_STARTS, card.GRID_STARTS = 96, 72, (0, 2048), (0, 2048)
        os.environ.pop('QWEN_SDPA_PF_TEST', None)

    def tearDown(self):
        card.POOL_BLOCKS, card.TP2_POOL_BLOCKS, card.TP2_STARTS, card.GRID_STARTS, card.time.perf_counter = self.saved[:5]
        if self.saved[5] is None:
            sys.modules.pop('ttnn', None)
        else:
            sys.modules['ttnn'] = self.saved[5]
        if self.saved[6] is not None:
            os.environ['QWEN_SDPA_PF_TEST'] = self.saved[6]

    def main(self, directory, extra=(), grid=(13, 10), **fake):
        clock = Clock()
        device = FakeDevice(grid)
        sys.modules['ttnn'] = fake_ttnn(device, clock, **fake)
        card.time.perf_counter = clock
        out = Path(directory) / 'oneq.json'
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            status = card.main(['--out', str(out)] + self.SMALL + list(extra))
        return status, json.loads(out.read_text()), buffer.getvalue(), device

    def test_a_clean_run_passes_every_check(self):
        with tempfile.TemporaryDirectory() as directory:
            status, report, stdout, _ = self.main(directory)
            self.assertEqual(status, 0, report['failures'] or report.get('error'))
            self.assertTrue(report['passed'])
            # rows x qmem x seed x variant x start = 3 x 2 x 1 x 2 x 4
            self.assertEqual(len(report['cases']), 48)
            self.assertTrue(all(case['exact'] for case in report['cases']))
            self.assertEqual(len(report['tp2_cases']), 4)             # rows 1024, 512 x 2 starts
            self.assertTrue(all(case['exact'] for case in report['tp2_cases']))
            self.assertEqual(len(report['grid_cases']), 2)            # the 11 x 10 grid on the 13 x 10 device, 2 starts
            self.assertEqual(report['stress'], dict(calls=12, drifted=0, first=[]))
            self.assertEqual(report['cache']['after_oneq'], report['cache']['after_served'] + 1)
            self.assertEqual(report['cache']['after_starts'], report['cache']['after_oneq'])
            self.assertTrue(all(entry['refused'] and entry['matched'] for entry in report['refusals'].values()), report['refusals'])
            self.assertEqual(len(report['refusals']), len(card.REFUSALS))
            self.assertEqual(report['grid'], [13, 10])
            self.assertIn('ONEQ_CARD_M verdict=PASS cases=48 exact=48 tp2=4 grid=2 failures=0', stdout)

    def test_oneq_arms_are_skipped_where_the_factory_refuses_them(self):
        with tempfile.TemporaryDirectory() as directory:
            status, report, _stdout, device = self.main(directory, ['--rows', '2048', '--no-timing'])
            self.assertEqual(status, 0, report['failures'])
            self.assertEqual(card.oneq_refusal('tp2', 2048, (13, 10)), '[QWEN-SDPA-PF] oneq needs one q chunk per core: 192 q chunks on 130 cores')
            self.assertIsNone(card.oneq_refusal('tp2', 1024, (13, 10)))
            self.assertIsNone(card.oneq_refusal('tp4', 2048, (11, 10)))
            self.assertEqual(card.oneq_refusal('tp4', 1920, (13, 10)), 'kv_chain outside its qualified envelope')
            self.assertEqual(card.arm_list(SimpleNamespace(arms=list(card.ARMS)), 'tp2', 2048, (13, 10)), ['stock', 'served'])
            self.assertEqual(card.arm_list(SimpleNamespace(arms=list(card.ARMS)), 'tp4', 2048, (13, 10)), list(card.ARMS))

    def test_the_factory_log_matches_the_planner_for_every_program(self):
        with tempfile.TemporaryDirectory() as directory:
            status, report, _stdout, _ = self.main(directory, ['--no-timing', '--rows', '2048,1024'])
            self.assertEqual(status, 0, report['failures'])
            log = (Path(directory) / 'oneq.json.native.log').read_text()
            flags, oneq = card.factory_lines(log)
            # TP4 rows 2048: paired 8 chains of 6, oneq 16 chains of 96 members in total
            self.assertIn((0x3, 8, 48, 'raster'), flags)
            self.assertIn((0xB, 16, 96, 'raster'), flags)
            self.assertIn((0xF, 16, 96, 'noc'), flags)
            self.assertIn((96, 130, 16, 96), oneq)
            self.assertIn((96, 110, 16, 96), oneq)                    # the 11 x 10 grid pass
            self.assertIn((48, 130, 8, 48), oneq)                     # 1024 rows
            self.assertIn((96, 130, 16, 96), [line for line in oneq])  # TP2 1024 rows: 12 heads x 8 chunks, 2 KV heads x 8 q = 16 chains
            self.assertEqual(card.check_factory_lines(log, set()) != [], True)     # no program requested: every line is unexpected

    def test_a_oneq_output_that_differs_fails_the_run(self):
        with tempfile.TemporaryDirectory() as directory:
            status, report, stdout, _ = self.main(directory, ['--no-timing'], broken='oneq')
            self.assertEqual(status, 1)
            self.assertFalse(report['passed'])
            bad = [case for case in report['cases'] if not case['exact']]
            self.assertTrue(bad)
            self.assertEqual({tuple(case['differing']) for case in bad}, {('oneq', 'oneq_noc')})
            self.assertTrue(any('oneq, oneq_noc differ from the stock path' in failure for failure in report['failures']))
            self.assertIn('ONEQ_CARD_M verdict=FAIL', stdout)

    def test_a_missing_oneq_log_line_or_a_wrong_chain_count_fails_the_run(self):
        with tempfile.TemporaryDirectory() as directory:
            status, report, _stdout, _ = self.main(directory, ['--no-timing', '--rows', '2048'], silent_oneq_log=True)
            self.assertEqual(status, 1)
            self.assertTrue(any(failure.startswith('factory log: oneq line') for failure in report['failures']), report['failures'])
        with tempfile.TemporaryDirectory() as directory:
            status, report, _stdout, _ = self.main(directory, ['--no-timing', '--rows', '2048'], wrong_chain_count=True)
            self.assertEqual(status, 1)
            self.assertTrue(any(failure.startswith('factory log: flags line') for failure in report['failures']), report['failures'])

    def test_timing_follows_the_model_and_the_verdict_is_a_win(self):
        with tempfile.TemporaryDirectory() as directory:
            card.POOL_BLOCKS = 1100
            status, report, stdout, _ = self.main(directory, ['--rows', '2048', '--starts', '0,2048', '--time-starts', '2048,32768,65536',
                                                              '--no-tp2', '--alternations', '0'])
            self.assertEqual(status, 0, report['failures'])
            block = report['timing']['2048']['dram']
            for start in ('2048', '32768', '65536'):
                served = block['served'][start]['median_ms']
                oneq = block['oneq'][start]['median_ms']
                self.assertAlmostEqual(served * 1e3, planner.layer_us(planner.plan(), int(start)), delta=1.0)
                self.assertAlmostEqual(oneq * 1e3, planner.layer_us(planner.plan(oneq=True), int(start)), delta=1.0)
                self.assertAlmostEqual(block['stock'][start]['median_ms'], served, delta=0.002)
            self.assertEqual(len(block['oneq']['2048']['samples']), 3)
            verdict = report['timing_verdict']['dram']
            self.assertEqual(verdict['verdict'], 'OQ-WIN')
            self.assertAlmostEqual(verdict['ratio'], 0.51, delta=0.02)
            self.assertIn('ONEQ TIME rows=2048 q=dram arm=oneq start=65536 context=67584', stdout)
            self.assertIn('ONEQ TIME_VERDICT q=dram', stdout)

    def test_the_report_module_reproduces_the_estimate_from_the_harness_report(self):
        with tempfile.TemporaryDirectory() as directory:
            starts = '0,2048,8192,32768,65536'
            card.POOL_BLOCKS = 1100
            status, report, _stdout, _ = self.main(directory, ['--rows', '2048', '--starts', '0', '--time-starts', starts, '--time-q-memory', 'dram',
                                                              '--rounds', '1', '--no-controls', '--alternations', '0', '--no-tp2'])
            self.assertEqual(status, 0, report['failures'] or report.get('error'))
            result = report_lib.summary(report['timing'], 2048, 'dram', 6, 1, tuple(report['grid']))
            by_prompt = {entry['prompt_tokens']: entry for entry in result['prompts']}
            for tokens in (32768, 131072, 253952):
                self.assertAlmostEqual(by_prompt[tokens]['measured_over_est'], 1.0, places=3)
            text = report_lib.render(result, markdown=True)
            self.assertIn('| 131072 | 64 | 7.88 | 6.09 |', text)

    def test_a_non_native_grid_larger_than_the_device_is_a_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            status, report, _stdout, _ = self.main(directory, ['--no-timing', '--grids', 'native,16x10', '--rows', '2048'], grid=(13, 10))
            self.assertEqual(status, 1)
            self.assertTrue(any('larger than the device grid' in failure for failure in report['failures']), report['failures'])

    def test_the_11x10_device_runs_the_same_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            status, report, _stdout, _ = self.main(directory, ['--no-timing', '--rows', '2048', '--grids', 'native'], grid=(11, 10))
            self.assertEqual(status, 0, report['failures'] or report.get('error'))
            self.assertEqual(report['grid'], [11, 10])
            self.assertEqual(report['grid_cases'], [])

    def test_a_binary_without_the_oneq_edits_is_refused_before_any_call(self):
        with tempfile.TemporaryDirectory() as directory:
            so = Path(directory) / '_ttnncpp.so'
            so.write_bytes(b'\x7fELF\x00[QWEN-SDPA-PF] flags=\x00')
            maps = Path(directory) / 'maps'
            maps.write_text('7f00 r-xp 0 00:00 0 %s\n' % so)
            info = card.loaded_binary(str(maps))
            self.assertEqual((info['chain'], info['oneq']), (True, False))
            so.write_bytes(so.read_bytes() + card.ONEQ_MARKER + b'\x00')
            self.assertTrue(card.loaded_binary(str(maps))['oneq'])
            other = Path(directory) / 'lib' / '_ttnncpp.so'
            other.parent.mkdir()
            other.write_bytes(so.read_bytes())
            maps.write_text('7f00 r-xp 0 00:00 0 %s\n7f01 r-xp 0 00:00 0 %s\n' % (so, other))
            with self.assertRaises(RuntimeError):
                card.loaded_binary(str(maps))

    def test_parse_args_refuses_what_cannot_be_run(self):
        for argv in (['--arms', 'served,oneq'], ['--arms', 'stock,bogus'], ['--rows', '2000'], ['--starts', '100'], ['--starts', '999936'],
                     ['--variants', 'zeroq'], ['--q-memory', 'sram']):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                card.parse_args(['--out', 'x.json'] + argv)
        args = card.parse_args(['--out', 'x.json', '--grids', 'native,11x10,12x9'])
        self.assertEqual(args.grid_list, [(11, 10), (12, 9)])
        with self.assertRaises(ValueError):
            card.parse_grids('11by10')


class HelperTests(unittest.TestCase):
    def test_schedule_covers_every_pair_each_round_with_rotated_order(self):
        arms, starts = ['a', 'b', 'c'], [0, 1]
        rounds = card.schedule(arms, starts, 4)
        self.assertEqual(len(rounds), 4 * 2 * 3)
        for index in range(4 * 2):
            chunk = rounds[index * 3:(index + 1) * 3]
            self.assertEqual(sorted(chunk), sorted((arm, starts[index % 2]) for arm in arms))
        firsts = {rounds[index * 3][0] for index in range(8)}
        self.assertEqual(firsts, {'a', 'b', 'c'})

    def test_timing_verdict_thresholds_and_the_noc_rule(self):
        def block(served, oneq, noc=None):
            entry = {'served': {str(s): dict(median_ms=ms) for s, ms in served.items()},
                     'oneq': {str(s): dict(median_ms=ms) for s, ms in oneq.items()}}
            if noc:
                entry['oneq_noc'] = {str(s): dict(median_ms=ms) for s, ms in noc.items()}
            return {'2048': {'dram': entry}}
        served = {32768: 10.0, 65536: 20.0, 129024: 40.0}
        win = card.timing_verdict(block(served, {32768: 5.5, 65536: 11.0, 129024: 22.0}))['dram']
        self.assertEqual((win['verdict'], round(win['ratio'], 2), win['best']), ('OQ-WIN', 0.55, 'oneq'))
        partial = card.timing_verdict(block(served, {32768: 8.0, 65536: 16.0, 129024: 32.0}))['dram']
        self.assertEqual(partial['verdict'], 'OQ-PARTIAL')
        none = card.timing_verdict(block(served, {32768: 10.0, 65536: 20.0, 129024: 40.0}))['dram']
        self.assertEqual(none['verdict'], 'OQ-NO-WIN')
        # noc replaces oneq only when it is at least 2 percent faster
        keep = card.timing_verdict(block(served, {32768: 5.5, 65536: 11.0, 129024: 22.0}, {32768: 5.45, 65536: 10.9, 129024: 21.8}))['dram']
        self.assertEqual(keep['best'], 'oneq')
        swap = card.timing_verdict(block(served, {32768: 5.5, 65536: 11.0, 129024: 22.0}, {32768: 5.0, 65536: 10.0, 129024: 20.0}))['dram']
        self.assertEqual((swap['best'], round(swap['ratio'], 2)), ('oneq_noc', 0.5))
        self.assertEqual(card.timing_verdict({})['x'] if False else card.timing_verdict({}), {})
        # starts below 32k do not enter the verdict
        small = card.timing_verdict(block({2048: 1.0, 8192: 2.0}, {2048: 2.0, 8192: 4.0}))['dram']
        self.assertEqual(small['verdict'], 'OQ-NO-WIN')

    def test_expected_programs_counts_one_line_per_program(self):
        requested = {('tp4', 2048, 96, 'dram', 0x3, (13, 10)), ('tp4', 2048, 96, 'l1', 0x3, (13, 10)), ('tp4', 2048, 96, 'dram', 0xB, (13, 10)),
                     ('tp2', 1024, 72, 'dram', 0xB, (11, 10))}
        flags, oneq = card.expected_programs(requested)
        self.assertEqual(flags[(0x3, 8, 48, 'raster')], 2)
        self.assertEqual(flags[(0xB, 16, 96, 'raster')], 2)         # TP4 2048 and TP2 1024 both make 16 chains of 96 members
        self.assertEqual(oneq[(96, 130, 16, 96)], 1)
        self.assertEqual(oneq[(96, 110, 16, 96)], 1)

    def test_case_labels_and_start_validation(self):
        self.assertEqual(card.case_label(2048, 'dram', 0, 'normal', 251904), 'tp4-r2048-dram-s0-normal@251904')
        self.assertEqual(card.case_label(2048, 'dram', 0, 'normal', 0, 'tp4', (11, 10)), 'tp4-r2048-dram-s0-normal-g11x10@0')
        card.validate_starts(card.STARTS, 2048)
        card.validate_starts(card.TIME_STARTS, 2048)
        self.assertEqual(max(card.STARTS) + 2048, 253952)         # the last chunk of a 254k prompt
        with self.assertRaises(ValueError):
            card.validate_starts([100], 2048)
        with self.assertRaises(ValueError):
            card.validate_starts([262144 - 1024], 2048)

    def test_the_default_matrix_covers_the_stated_contexts(self):
        contexts = sorted({start + 2048 for start in card.TIME_STARTS})
        self.assertEqual(contexts[0], 2048)
        self.assertEqual(contexts[-1], 253952)
        self.assertEqual(card.ARMS['stock'], None)
        self.assertEqual(card.word(card.ARMS['oneq']), 0x5EFA000B)
        self.assertEqual(card.word(card.ARMS['served']), 0x5EFA0003)
        self.assertEqual(card.word(card.ARMS['oneq_noc']), 0x5EFA000F)
        self.assertTrue(all(ps.decode_word(card.word(flags))[0] for name, flags in card.ARMS.items() if flags is not None))
        # every oneq arm of the default TP4 matrix is a program the planner accepts on both grids
        for grid in ((11, 10), (13, 10)):
            for rows in card.ROWS:
                self.assertIsNone(card.oneq_refusal('tp4', rows, grid), (rows, grid))

    def test_the_report_table_has_the_three_estimate_rows(self):
        result = report_lib.summary({}, 2048, 'dram')
        self.assertEqual([entry['prompt_tokens'] for entry in result['prompts']], [32768, 131072, 253952])
        self.assertEqual([round(entry['est_floor_s'], 2) for entry in result['prompts']], [0.47, 7.88, 29.80])
        self.assertTrue(all(entry['measured_s'] is None for entry in result['prompts']))
        text = report_lib.render(result)
        self.assertIn('pending', text)

    def test_fit_is_a_least_squares_line(self):
        intercept, slope = report_lib.fit([(0, 1.0), (1, 3.0), (2, 5.0)])
        self.assertAlmostEqual(intercept, 1.0)
        self.assertAlmostEqual(slope, 2.0)
        with self.assertRaises(ValueError):
            report_lib.fit([(1, 1.0)])


class RunnerTests(unittest.TestCase):
    def setUp(self):
        if not RUNNER.is_file():
            self.skipTest('run_card_m_oq.sh not written yet')
        self.text = RUNNER.read_text(encoding='utf-8')

    def test_the_text(self):
        self.assertNotIn(chr(13), self.text)
        start = self.text.index('# >>> qual_card.sh')
        end = self.text.index(NL, self.text.index('# <<< qual_card.sh')) + 1
        self.assertEqual(self.text[start:end], (HERE / 'qual_card.sh').read_text(encoding='utf-8'))
        self.assertTrue(self.text[end:].startswith('qual_card_select' + NL))
        code = self.text[:start] + self.text[end:]
        for board in ('blackhole-CEF5729692C19E6D', 'blackhole-3707293C249A5E67'):
            self.assertNotIn(board, code)
        self.assertEqual([line for line in code.splitlines() if re.search(r'/dev/tenstorrent/[0-9]', line)], [])
        for number, line in enumerate(code.splitlines(), 1):
            self.assertIsNone(re.search(r'\d{1,3}(\.\d{1,3}){3}|thatch\.local|zot\.|sha256:[0-9a-f]{16}|/home/|ssh ', line), '%d: %s' % (number, line))

    def test_the_holder_check_and_the_recheck_come_before_the_launch(self):
        end = self.text.index(NL, self.text.index('# <<< qual_card.sh')) + 1
        code = self.text[end:]
        launches = [m.start() for m in re.finditer(r'^timeout -k 30 "\$timeout_s" "\$\{argv\[@\]\}"', code, flags=re.M)]
        self.assertEqual(len(launches), 1)
        rechecks = [m.start() for m in re.finditer(r'^qual_card_recheck\b', code, flags=re.M)]
        self.assertEqual(len(rechecks), 1)
        holders = [m.start() for m in re.finditer(r'^\s*qual_refuse_holders$', code, flags=re.M)]
        self.assertTrue(holders and holders[-1] < rechecks[0] < launches[0])
        self.assertNotIn('docker run', code[rechecks[0]:launches[0]])
        self.assertNotIn('qual_card_resolve', code[rechecks[0]:launches[0]])

    def test_bash_parses_it(self):
        bash = subprocess.run(['bash', '-n', str(RUNNER)], capture_output=True, text=True)
        self.assertEqual(bash.returncode, 0, bash.stderr)

    def dry(self, home, **extra):
        env = {key: value for key, value in os.environ.items()
               if key not in ('QUAL_CARD', 'ALLOW_SERVING_CARD', 'IMAGE', 'KOPGRAFT64', 'EXPECT_TTNNCPP_SHA256', 'WATCHER', 'OQ_ARGS', 'CARD_B_ARGS',
                              'RESULTS', 'WATCHDOG_S', 'OQ_CARD_DRY_RUN')}
        env.update(dict(QUAL_CARD='blackhole-0000000000000001', OQ_CARD_DRY_RUN='1', HOME=home), **extra)
        return subprocess.run(['bash', str(RUNNER)], env=env, capture_output=True, text=True)

    def argv_of(self, result):
        import shlex
        line = [l for l in result.stdout.splitlines() if l.startswith('### argv: ')][0]
        return shlex.split(line[len('### argv: '):])

    def test_the_dry_run_prints_a_read_only_graft_launch_on_the_named_card(self):
        with tempfile.TemporaryDirectory() as home:
            result = self.dry(home, IMAGE='img:tag', EXPECT_TTNNCPP_SHA256='ab' * 32, OQ_ARGS='--rows 2048')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            argv = self.argv_of(result)
            self.assertEqual(argv[:2], ['docker', 'run'])
            self.assertNotIn('--privileged', argv)
            self.assertEqual(argv[argv.index('--network') + 1], 'none')
            self.assertEqual(argv[argv.index('--device') + 1], '/dev/tenstorrent/by-id/blackhole-0000000000000001')
            mounts = [value for flag, value in zip(argv, argv[1:]) if flag == '--mount']
            graft = [m for m in mounts if 'opgraft-K64j-OQ' in m]
            self.assertEqual(len(graft), 7)                              # _ttnn.so, _ttnncpp.so x2, four op directories
            self.assertTrue(all(m.endswith(',readonly') for m in graft))
            self.assertTrue(any(m.endswith('dst=/opt/tt-metal/build_Release/lib/_ttnncpp.so,readonly') for m in graft))
            self.assertTrue(any(m.endswith('dst=/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa,readonly') for m in graft))
            self.assertTrue(any(m.endswith('dst=/bench,readonly') for m in mounts))
            self.assertTrue(any(m.endswith('dst=/bench_pf,readonly') for m in mounts))
            tail = argv[argv.index('oneq') + 1:]
            self.assertEqual(tail[tail.index('--expect-binary-sha256') + 1], 'ab' * 32)
            self.assertIn('--rows', tail)
            self.assertNotIn('--no-timing', tail)
            self.assertIn('QWEN_C2_SERVING=0', argv)
            self.assertIn('QWEN_SDPA_TREE_SCRATCH_ROUNDS=1', argv)
            self.assertNotIn('TT_METAL_WATCHER=5', argv)
            self.assertIn('unset TT_MESH_GRAPH_DESC_PATH', argv[argv.index('-c') + 1])
            self.assertEqual(argv[argv.index('--entrypoint') + 1], 'sh')
            self.assertEqual(argv[argv.index('--entrypoint') + 2], 'img:tag')

    def test_the_watcher_pass_narrows_the_matrix_and_turns_the_sanitiser_on(self):
        with tempfile.TemporaryDirectory() as home:
            result = self.dry(home, IMAGE='img:tag', WATCHER='1')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            argv = self.argv_of(result)
            self.assertIn('TT_METAL_WATCHER=5', argv)
            tail = argv[argv.index('oneq') + 1:]
            self.assertIn('--no-timing', tail)
            self.assertEqual(tail[tail.index('--alternations') + 1], '0')
            self.assertEqual(tail[tail.index('--rows') + 1], '2048,1024')
            args = card.parse_args(tail[:tail.index('--expect-binary-sha256')] + tail[tail.index('--expect-binary-sha256') + 2:])
            self.assertTrue(args.no_timing)
            self.assertEqual(args.grid_list, [])

    def test_the_runner_refuses_a_bad_selection_before_anything(self):
        with tempfile.TemporaryDirectory() as home:
            for env, needle in ((dict(OQ_CARD_DRY_RUN='2'), 'refusing: OQ_CARD_DRY_RUN=2'),):
                result = self.dry(home, **env)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(needle, result.stderr)
            card_m = self.dry(home)
            self.assertEqual(card_m.returncode, 0)
        env = {key: value for key, value in os.environ.items() if key not in ('QUAL_CARD', 'ALLOW_SERVING_CARD')}
        for card_id, needle in (('', 'refusing: QUAL_CARD is not set'), ('blackhole-CEF5729692C19E6D', 'is card M'),
                                ('blackhole-F36F768B9A5CAFA0', 'is card B, reserved for another project')):
            result = subprocess.run(['bash', str(RUNNER)], env=dict(env, QUAL_CARD=card_id, OQ_CARD_DRY_RUN='1', HOME='/nonexistent'),
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertIn(needle, result.stderr)


if __name__ == '__main__':
    unittest.main()
