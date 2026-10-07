"""QWEN_FAST_TP4_SDPA (sdpa_long_tp): the flag's parsing, the grid plans, the hook in the reader twin, the binding, the smoke check, the
profile twin and the shipment.

Flag off is the served path byte for byte: no config is rebuilt, no line is logged, the pinned reader is the one tp_addresses binds.
Flag on rebuilds each segment entry's program config on the named grid with the SAME sentinel, chunk size and exp mode, and says so.

Run at py 3.11: `py -3.11 -B -m unittest test_sdpa_long_tp` from scripts/ci.
"""

import json
import os
from pathlib import Path
import re
import unittest
from unittest.mock import patch

import c2_smoke_check
import extent_attention_fold_tp as folded
import sdpa_long_tp
import tp_addresses
from test_extent_attention_replay import WIDTH, host_table
from test_extent_attention_replay_tp import FourChipDevice, Harness, MESH, lend

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
FLAG = sdpa_long_tp.FLAG
FOUR, PAIR = {'QWEN_FAST_TP': '4'}, {}


def env(**values):
    return patch.dict(os.environ, values)


class FlagTests(unittest.TestCase):
    def test_off_values(self):
        for value in (None, '', '0', 'off', 'OFF', ' off '):
            with self.subTest(value=value):
                source = dict(FOUR) if value is None else dict(FOUR, **{FLAG: value})
                self.assertIsNone(sdpa_long_tp.selected(source))
                self.assertFalse(sdpa_long_tp.enabled(source))

    def test_off_at_the_pair_is_fine_and_on_at_the_pair_raises(self):
        self.assertIsNone(sdpa_long_tp.selected(dict(PAIR)))
        self.assertIsNone(sdpa_long_tp.selected({FLAG: '0'}))
        with self.assertRaisesRegex(ValueError, 'needs QWEN_FAST_TP=4'):
            sdpa_long_tp.selected({FLAG: 'grid8x4'})
        with self.assertRaisesRegex(ValueError, 'needs QWEN_FAST_TP=4'):
            sdpa_long_tp.selected({FLAG: 'served', 'QWEN_FAST_TP': '2'})

    def test_every_servable_name_selects_itself(self):
        for name in sdpa_long_tp.servable_names():
            self.assertEqual(sdpa_long_tp.selected(dict(FOUR, **{FLAG: name})), name)
        self.assertEqual(sdpa_long_tp.servable_names(), ('served', 'grid8x4', 'grid8x10', 'grid11x4', 'grid4x8', 'multi'))

    def test_a_typo_and_a_configuration_that_cannot_be_served_are_different_refusals(self):
        with self.assertRaisesRegex(ValueError, 'is not a configuration'):
            sdpa_long_tp.selected(dict(FOUR, **{FLAG: 'grid9x9'}))
        for name in ('rowsplit', 'ra'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'cannot be served yet: .*Servable: served, grid8x4'):
                sdpa_long_tp.selected(dict(FOUR, **{FLAG: name}))

    def test_the_table_is_complete(self):
        self.assertEqual(set(sdpa_long_tp.CONFIGS), {'served', 'grid8x4', 'grid8x10', 'grid11x4', 'grid4x8', 'multi', 'rowsplit', 'ra'})
        for name, entry in sdpa_long_tp.CONFIGS.items():
            self.assertEqual(bool(entry['why']), not entry['servable'], name)


class GridTests(unittest.TestCase):
    MESH = (11, 10)

    def test_the_four_grids_hold_the_served_program(self):
        for name in ('grid8x4', 'grid8x10', 'grid11x4', 'grid4x8'):
            with self.subTest(name=name):
                self.assertEqual(sdpa_long_tp.plan(name, self.MESH), sdpa_long_tp.CONFIGS[name]['grid'])

    def test_served_runs_on_the_meshs_own_grid(self):
        self.assertEqual(sdpa_long_tp.plan('served', (11, 10)), (11, 10))
        self.assertEqual(sdpa_long_tp.plan('served', (8, 8)), (8, 8))

    def test_problems(self):
        problem = sdpa_long_tp.grid_problem
        self.assertIsNone(problem((11, 10), self.MESH))
        self.assertIn('does not fit', problem((12, 10), self.MESH))
        self.assertIn('does not fit', problem((8, 11), self.MESH))
        self.assertIn('cores, the program needs 32', problem((4, 4), self.MESH))
        self.assertIn('twin bands', problem((5, 7), self.MESH))      # 35 cores, but ceil(16/5) * 2 = 8 rows are needed and 7 given
        self.assertIsNone(problem((4, 8), self.MESH))
        self.assertIn('empty', problem((0, 8), self.MESH))

    def test_a_grid_the_mesh_cannot_hold_raises_naming_the_flag(self):
        with self.assertRaisesRegex(ValueError, 'QWEN_FAST_TP4_SDPA=grid8x10: .*does not fit'):
            sdpa_long_tp.plan('grid8x10', (11, 8))


class ReaderHookTests(unittest.TestCase):
    def setUp(self):
        self.device = FourChipDevice()
        self.harness = Harness(self)
        self.logs = []

    def reader(self, **flags):
        segments = ((0, 16), (16, 32), (32, 48), (48, 64))
        lent = [lend(self.device, last - first) for first, last in segments]
        with env(**flags), patch('sdpa_long_tp._log', side_effect=self.logs.append):
            return folded.PackedExtentReplayReader(self.device, MESH, segments, WIDTH, [host_table(i + 1) for i in range(4)],
                                                   storage=lent, max_group_rows=8, starts=(4200, 9000, 4300, 5000))

    def configs(self, reader):
        return [entry[3] for segment in reader.readers for entry in segment.metadata]

    def test_flag_off_touches_nothing(self):
        with patch.object(sdpa_long_tp, 'plan', side_effect=AssertionError('plan ran')):
            reader = self.reader()
            reader0 = self.reader(**{FLAG: '0'})
            reader1 = self.reader(**{FLAG: 'off'})
        for built in (reader, reader0, reader1):
            self.assertEqual({vars(config)['compute_with_storage_grid_size'] for config in self.configs(built)}, {(11, 10)})
        self.assertEqual([line for line in self.logs if sdpa_long_tp.ENGAGED in line], [])

    def test_off_and_served_build_identical_configs(self):
        off = [vars(config) for config in self.configs(self.reader())]
        served = [vars(config) for config in self.configs(self.reader(**{FLAG: 'served'}))]
        self.assertEqual(off, served)

    def test_served_logs_engaged_and_a_grid_rebuilds_with_the_same_sentinel_chunk_and_mode(self):
        off = [vars(config) for config in self.configs(self.reader())]
        reader = self.reader(**{FLAG: 'grid8x4'})
        on = [vars(config) for config in self.configs(reader)]
        self.assertEqual(len(on), len(off))
        for left, right in zip(off, on):
            self.assertEqual(right['compute_with_storage_grid_size'], (8, 4))
            self.assertEqual(left['compute_with_storage_grid_size'], (11, 10))
            for key in ('q_chunk_size', 'k_chunk_size', 'exp_approx_mode'):
                self.assertEqual(left[key], right[key], key)
            self.assertEqual(right['q_chunk_size'] & 0xFF, 0x23)
            self.assertEqual(right['k_chunk_size'], 256)
            self.assertIs(right['exp_approx_mode'], False)
        engaged = [line for line in self.logs if line.startswith(sdpa_long_tp.ENGAGED)]
        self.assertEqual(engaged, ['%s config=grid8x4 grid=8x4 entries=%d' % (sdpa_long_tp.ENGAGED, len(on))])

    def test_the_rest_of_the_reader_is_untouched(self):
        off = self.reader()
        on = self.reader(**{FLAG: 'grid8x10'})
        for left, right in zip(off.readers, on.readers):
            self.assertIs(type(left), type(right))
            self.assertEqual(left.sdpa_modes_applied, right.sdpa_modes_applied)
            self.assertEqual([entry[0] for entry in left.metadata], [entry[0] for entry in right.metadata])
            self.assertEqual(len(left.metadata), len(right.metadata))

    def test_a_flag_set_the_config_is_not_qualified_for_refuses_and_closes(self):
        reader = self.reader()
        reader.readers[0].sdpa_modes_applied = (0x27, 0x27)
        with env(**{FLAG: 'grid8x4'}):
            with self.assertRaisesRegex(ValueError, 'qualified at flags 0x23 only'):
                sdpa_long_tp.apply(reader)

    def test_a_bad_value_fails_the_constructor_and_releases_the_readers(self):
        closed = []
        original = folded.PackedExtentReplayReader.close
        with patch.object(folded.PackedExtentReplayReader, 'close', lambda self: (closed.append(1), original(self))[1]):
            with self.assertRaisesRegex(ValueError, 'cannot be served yet'):
                self.reader(**{FLAG: 'rowsplit'})
        self.assertTrue(closed)

    def test_the_twin_is_a_subclass_of_the_pinned_reader(self):
        self.assertEqual(folded.PackedExtentReplayReader.__mro__[1].__module__, 'extent_attention_replay_tp')


class BindingTests(unittest.TestCase):
    def rows(self, **values):
        return tp_addresses.bound_twins(dict(FOUR, **values))[0]

    def bound(self, **values):
        return ('extent_attention_replay_tp', 'PackedExtentReplayReader') in [row[:2] for row in self.rows(**values)]

    def test_the_reader_twin_is_bound_for_either_flag_and_not_for_neither(self):
        self.assertFalse(self.bound())
        self.assertFalse(self.bound(**{FLAG: '0'}))
        self.assertFalse(self.bound(QWEN_FAST_TP4_ATTN_FOLD='0', **{FLAG: 'off'}))
        self.assertTrue(self.bound(**{FLAG: 'served'}), 'the SDPA flag alone binds the twin that carries its hook')
        self.assertTrue(self.bound(**{FLAG: 'grid8x4'}))
        self.assertTrue(self.bound(QWEN_FAST_TP4_ATTN_FOLD='1'))
        self.assertTrue(self.bound(QWEN_FAST_TP4_ATTN_FOLD='1', **{FLAG: 'grid4x8'}))

    def test_a_bad_value_raises_at_install_time(self):
        with self.assertRaises(ValueError):
            self.rows(**{FLAG: 'nope'})

    def test_the_flagged_row_names_both_flags(self):
        self.assertEqual(tp_addresses.FLAGGED_TWINS[('extent_attention_replay_tp', 'PackedExtentReplayReader')][0],
                         'QWEN_FAST_TP4_ATTN_FOLD or QWEN_FAST_TP4_SDPA')


class SmokeAndProfileTests(unittest.TestCase):
    PROFILES = json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))['profiles']
    CONTROL, TWIN = 'c2-packed-tp4-best-strace', 'c2-packed-tp4-best-sdpa'

    def test_the_twin_is_the_control_plus_exactly_the_flag_and_gate_only(self):
        control, twin = self.PROFILES[self.CONTROL], self.PROFILES[self.TWIN]
        self.assertEqual(set(twin['env']) - set(control['env']), {FLAG})
        self.assertEqual({key: value for key, value in twin['env'].items() if key != FLAG}, control['env'])
        self.assertIn(twin['env'][FLAG], sdpa_long_tp.servable_names())
        for key in set(control) | set(twin):
            if key not in ('env', 'description'):
                self.assertEqual(control.get(key), twin.get(key), key)
        self.assertIs(twin['gate_only'], True)
        self.assertIn('not for traffic', twin['description'].lower())
        self.assertIn('placeholder', twin['description'].lower())

    def test_no_traffic_profile_carries_the_flag(self):
        # tp4/sdpa-multi's two gate-only twins carry it too (test_sdpa_multi_tp holds them to their control plus exactly the flags).
        # tp4/w2's two gate-only arms carry multi too (test_tp4_w2).
        # the combined window's multi carriers are gate only too (test_tp4_w2ln_profiles)
        from test_tp4_w2ln_profiles import MULTI as WINDOW_MULTI
        allowed = {self.TWIN, 'c2-packed-tp4-8x262k-best-sdpamulti', 'c2-packed-tp4-8x262k-best-sdpamulti-audit',
                   'c2-packed-tp4-8x262k-w2', 'c2-packed-tp4-8x262k-w2-audit', 'c2-packed-tp4-8x262k-w2-nof1', 'c2-packed-tp4-8x262k-w2-nof1-audit'} | set(WINDOW_MULTI)
        for name, body in self.PROFILES.items():
            if name not in allowed:
                self.assertNotIn(FLAG, body.get('env', {}), name)
            else:
                self.assertIs(body['gate_only'], True, name)

    def test_the_smoke_check_fails_a_profile_that_asks_for_the_flag_and_logs_no_engaged_line(self):
        smoke = 'SMOKE ok'
        quiet = c2_smoke_check.sdpa_long_problems({FLAG: 'grid8x4'}, 'nothing here')
        self.assertTrue(quiet and FLAG in quiet[0] and sdpa_long_tp.ENGAGED in quiet[0], quiet)
        engaged = sdpa_long_tp.marker('grid8x4', (8, 4), 128)
        self.assertEqual(c2_smoke_check.sdpa_long_problems({FLAG: 'grid8x4'}, 'x\n%s\ny' % engaged), [])
        for value in (None, '', '0', 'off'):
            env_ = {} if value is None else {FLAG: value}
            self.assertEqual(c2_smoke_check.sdpa_long_problems(env_, 'nothing here'), [])
        self.assertEqual(c2_smoke_check.sdpa_long_problems(None, 'nothing here'), [])
        self.assertEqual(smoke, 'SMOKE ok')


class ShipmentTests(unittest.TestCase):
    def read(self, path):
        return (REPO / path).read_text(encoding='utf-8')

    def test_the_runtime_files_ship_in_the_overlay_only_and_the_cpu_allowlist_names_the_tests(self):
        # New C2 modules go in docker/qwen-c2-overlay.txt ONLY: the P8 base lists (the Dockerfile and the base workflow) are the production base's and stay as shipped.
        dockerfile = self.read('docker/qwen-fast-serving.Dockerfile')
        workflow = self.read('.github/workflows/qwen-fast-serving-image.yml')
        overlay = self.read('docker/qwen-c2-overlay.txt').splitlines()
        for name in sdpa_long_tp.RUNTIME_FILES:
            self.assertTrue((HERE / name).is_file())
            self.assertNotIn('scripts/ci/%s ' % name, dockerfile)
            self.assertNotRegex(workflow, r'for name in [^\n]*\b%s\b' % re.escape(name))
            self.assertIn('scripts/ci/%s' % name, overlay)
        cpu = self.read('.github/workflows/qwen-integration-cpu.yml')
        for module in ('test_sdpa_long_tp', 'test_sdpa_tp4_long', 'test_tp4_sdpa_long_window'):
            self.assertRegex(cpu, r'python -B -m unittest [^\n]*\b%s\b' % module)

    def test_the_importing_module_ships_beside_it(self):
        importing = [path.name for path in HERE.glob('*.py') if not path.name.startswith('test_')
                     and re.search(r'^\s*(import|from) sdpa_long_tp\b', path.read_text(encoding='utf-8'), re.M)]
        self.assertEqual(sorted(importing), ['extent_attention_fold_tp.py', 'tp_addresses.py'])
        self.assertIn('extent_attention_fold_tp.py', self.read('docker/qwen-fast-serving.Dockerfile'))

    def test_the_module_is_py37_stdlib_only(self):
        text = (HERE / 'sdpa_long_tp.py').read_text(encoding='utf-8')
        self.assertNotRegex(text, r'(?m)^import (torch|ttnn|numpy)')
        self.assertNotIn(':=', text)
        self.assertNotIn(chr(13), text)


if __name__ == '__main__':
    unittest.main()
