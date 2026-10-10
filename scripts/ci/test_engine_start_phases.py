"""engine_start_phases on synthetic container logs (no real log is committed): the milestones in order, both timestamp formats, lines without a timestamp, midnight, a
profile without the prefix line, a missing milestone, the lever lines, the control-against-flag table."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import engine_start_phases as phases  # noqa: E402


def loguru(second, text, minute=30, hour=22, day=4):
    return '(EngineCore pid=67) 2026-10-%02d %02d:%02d:%06.3f | INFO     | x:y:1 - %s' % (day, hour, minute, second, text)


def vllm(second, text, minute=30, hour=22):
    return '(EngineCore pid=67) INFO 10-04 %02d:%02d:%02d [x.py:1] %s' % (hour, minute, second, text)


def log(scale=1.0, prefix=True, midnight=False, lever_lines=()):
    """One engine start; every phase is `scale` times its base length (seconds)."""
    t = [0.0]

    def at(step):
        t[0] += step * scale
        total = int(t[0])
        return dict(hour=22 + (total // 3600), minute=(total // 60) % 60, second=total % 60 + (t[0] - total))
    lines = []
    def emit(step, text, fmt=loguru):
        clock = at(step)
        if midnight and clock['hour'] >= 24:
            clock['hour'] -= 24
        if fmt is loguru:
            lines.append(loguru(clock['second'], text, clock['minute'], clock['hour'], 4 if not (midnight and t[0] > 3500) else 5))
        else:
            lines.append(vllm(int(clock['second']), text, clock['minute'], clock['hour']))
    emit(1, 'Initializing a V1 LLM engine (v0.25.1) with config', vllm)
    emit(2, 'multidevice with 4 devices and grid (1, 4) is created', vllm)
    emit(4, 'Loading 64 transformer layers (indices=[0, 1, 2])')
    emit(50, '[PINDIAG] single gate/up copy: w_gate_up not built (layer 64 of this process')
    lines.append('(EngineCore pid=67) Loading layers: 100%|##########| 64/64 [00:55<00:00,  1.16it/s]')
    emit(3, 'Inferring device name: P150x4')
    emit(2, 'GPU KV cache size: 1,277,952 tokens', vllm)
    if prefix:
        emit(18, '[PINDIAG] prefix: worker block_size=64 kv_groups=1 kv_dtype=DataType.BFLOAT8_B', vllm)
    emit(8 if prefix else 26, '[MEMLEDGER] phase=P0 chip0 allocated=17.817GB')
    emit(1, '[MEMLEDGER] phase=P1 chip0 allocated=17.817GB')
    emit(1, '[MEMLEDGER] phase=P2 chip0 allocated=20.354GB')
    emit(13, '[PINDIAG] four-card eager prefill warmed before the packed traces: page_table_blocks=4096')
    emit(10, 'something during the captures')
    for line in lever_lines:
        lines.append(line)
    lines.append('(APIServer pid=1) INFO:     Application startup complete.')
    return lines


class ParseTests(unittest.TestCase):
    def test_each_phase_is_the_gap_between_its_milestones(self):
        result = phases.parse(log())
        self.assertEqual(result['missing'], [])
        expected = dict(vllm_init_to_mesh=2, config_and_state_dict=4, layers=50, lm_head_vision=5, kv_pool=18, prefix_warm=8, ledger_to_pool=1, buffer_pool=1,
                        drafter_and_eager_warm=13, captures=10)
        for name, seconds in expected.items():
            self.assertAlmostEqual(result['phases'][name], seconds, delta=1.0, msg=name)
        self.assertAlmostEqual(result['total'], 1 and sum(expected.values()) - 0, delta=2.0)

    def test_a_loaded_start_scales_the_phases_it_scales(self):
        idle, loaded = phases.parse(log(1.0)), phases.parse(log(5.0))
        self.assertAlmostEqual(loaded['phases']['layers'] / idle['phases']['layers'], 5.0, delta=0.3)

    def test_without_the_prefix_line_kv_end_falls_back_to_the_ledger(self):
        result = phases.parse(log(prefix=False))
        self.assertEqual(result['missing'], [])
        self.assertAlmostEqual(result['phases']['kv_pool'], 26, delta=1.0)
        self.assertAlmostEqual(result['phases']['prefix_warm'], 0, delta=0.001)

    def test_a_milestone_that_never_appears_is_listed_and_its_phases_dropped(self):
        lines = [line for line in log() if 'GPU KV cache size' not in line]
        result = phases.parse(lines)
        self.assertIn('kv_begin', result['missing'])
        self.assertNotIn('kv_pool', result['phases'])
        self.assertNotIn('lm_head_vision', result['phases'])

    def test_a_line_without_a_timestamp_takes_the_last_one_before_it(self):
        result = phases.parse(log())
        self.assertAlmostEqual(result['milestones']['startup'] - result['milestones']['eager_warm_end'], 10, delta=1.0)

    def test_a_start_that_crosses_midnight_is_unwrapped(self):
        lines = ['(EngineCore pid=67) INFO 10-04 23:59:50 [x.py:1] Initializing a V1 LLM engine',
                 '(EngineCore pid=67) INFO 10-04 23:59:55 [x.py:1] multidevice with 4 devices and grid (1, 4) is created',
                 '(EngineCore pid=67) 2026-10-05 00:00:05.000 | INFO | x:y:1 - Loading 64 transformer layers',
                 '(EngineCore pid=67) 2026-10-05 00:01:05.000 | INFO | x:y:1 - last layer',
                 '(EngineCore pid=67) Loading layers: 100%|##########| 64/64',
                 '(APIServer pid=1) INFO:     Application startup complete.']
        result = phases.parse(lines)
        self.assertAlmostEqual(result['phases']['vllm_init_to_mesh'], 5, delta=0.01)
        self.assertAlmostEqual(result['phases']['layers'], 60, delta=0.01)
        self.assertAlmostEqual(result['total'], 75, delta=0.01)

    def test_milestones_are_taken_in_order_and_a_missing_one_does_not_hide_the_rest(self):
        # a P2 ledger line before the P1 one is not the pool end
        lines = [loguru(0, 'Initializing a V1 LLM engine'), loguru(1, '[MEMLEDGER] phase=P2 chip0 x'), loguru(2, '[MEMLEDGER] phase=P1 chip0 x')]
        result = phases.parse(lines)
        self.assertEqual(list(result['milestones']), ['engine_init', 'pool_begin'])
        self.assertIn('pool_end', result['missing'])
        # the milestones between two found ones may be absent
        lines = [loguru(0, 'Initializing a V1 LLM engine'), loguru(9, '[MEMLEDGER] phase=P2 chip0 x')]
        self.assertEqual(phases.parse(lines)['milestones']['pool_end'], phases.parse(lines)['milestones']['engine_init'] + 9)


class LeverTests(unittest.TestCase):
    ZEROS = ['[PINDIAG] tp4 device zeros engaged summary tag=kv_cache device=32 host=0 bytes_per_card=11123294208 seconds=0.412 latched=no',
             '[PINDIAG] tp4 device zeros engaged summary tag=buffer_pool device=16 host=144 bytes_per_card=335544320 seconds=0.051 latched=no',
             '[PINDIAG] tp4 device zeros audit exact=True tag=kv_cache tensor=1 chips=4 raw=numpy',
             '[PINDIAG] tp4 device zeros audit mismatch tag=buffer_pool tensor=1 exact=False dtype=bf16 differing_bytes=2']
    LAZY = ['[PINDIAG] tp4 lazy shard loads calls=64 misses=0 hits=64 latched=no', '[PINDIAG] tp4 lazy shard loads calls=384 misses=2 hits=382 latched=no',
            '[PINDIAG] tp4 lazy shard audit exact=True name=mlp chips=4 bytes_per_chip=9 raw=numpy']

    def test_the_levers_own_lines_are_read(self):
        found = phases.levers('\n'.join(self.ZEROS + self.LAZY))
        self.assertEqual(found['device_zeros']['kv_cache'], dict(device=32, host=0, bytes_per_card=11123294208, seconds=0.412, latched='no'))
        self.assertEqual(found['device_zeros']['buffer_pool']['host'], 144)
        self.assertEqual(found['lazy_shard'], dict(calls=384, misses=2, hits=382, latched='no'))
        self.assertEqual([(entry['lever'], entry['exact']) for entry in found['audits']],
                         [('device_zeros', True), ('device_zeros', False), ('lazy_shard', True)])
        self.assertEqual(phases.levers('nothing'), {'device_zeros': {}})


class CompareTests(unittest.TestCase):
    def test_the_table_has_the_phases_both_logs_have_and_the_total(self):
        control, flag = phases.parse(log(5.0)), phases.parse(log(1.0))
        rows = phases.compare(control, flag)
        names = [row[0] for row in rows]
        self.assertEqual(names[-1], 'total')
        self.assertIn('layers', names)
        layers = [row for row in rows if row[0] == 'layers'][0]
        self.assertLess(layers[2], layers[1])
        self.assertLess(layers[3], 0)
        self.assertAlmostEqual(layers[4], 0.2, delta=0.05)
        self.assertIn('layers', phases.table(rows))

    def test_the_command_line_prints_json_per_log_and_the_comparison(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder) / 'a.log', Path(folder) / 'b.log'
            a.write_text('\n'.join(log(5.0)), encoding='utf-8')
            b.write_text('\n'.join(log(1.0, lever_lines=LeverTests.ZEROS[:2])), encoding='utf-8')
            single = subprocess.run([sys.executable, '-B', str(HERE / 'engine_start_phases.py'), str(a)], capture_output=True, text=True)
            self.assertEqual(single.returncode, 0, single.stderr)
            self.assertIn('layers', json.loads(single.stdout)['phases'])
            both = subprocess.run([sys.executable, '-B', str(HERE / 'engine_start_phases.py'), '--control', str(a), '--flag', str(b)], capture_output=True, text=True)
            self.assertEqual(both.returncode, 0, both.stderr)
            self.assertIn('phase', both.stdout.splitlines()[-len(phases.PHASES) - 2])
            self.assertEqual(json.loads(both.stdout[:both.stdout.rindex('}') + 1])['flag']['levers']['device_zeros']['kv_cache']['device'], 32)
            bad = subprocess.run([sys.executable, '-B', str(HERE / 'engine_start_phases.py'), '--control', str(a)], capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)


if __name__ == '__main__':
    unittest.main()
