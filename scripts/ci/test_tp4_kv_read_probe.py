"""tp4_kv_read_probe on the CPU: the quad wrapper of the region-read card check, on a fake card module and a fake ttnn.

Run at py 3.11: `py -3.11 -B -m unittest test_tp4_kv_read_probe` from scripts/ci."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import c2_serving_job as job  # noqa: E402
import tp4_kv_read_probe as probe  # noqa: E402

WORKFLOW = (ROOT / '.github' / 'workflows' / 'qwen-c2-serving.yml').read_text(encoding='utf-8')
CARD_FILE = ROOT / 'optimisation' / 'ttnn-op' / 'kv_region_read' / 'kv_region_read_card.py'


class FakeTTNN:
    FabricConfig = type('FabricConfig', (), {'FABRIC_1D': 'f1d', 'FABRIC_1D_RING': 'f1dr'})

    def __init__(self):
        self.calls = []

    def set_fabric_config(self, value):
        self.calls.append(value)


class FakeCard:
    """kv_region_read_card.main prints one indented JSON object and returns 0 or 1."""

    def __init__(self, report, code):
        self.report, self.code, self.argv = report, code, None

    def main(self, argv=None):
        self.argv = argv
        print('some compiler chatter before the report')
        print(json.dumps(self.report, indent=1))
        return self.code


GOOD = {'ok': True, 'problems': [], 'whole_read_s': 21.5, 'whole_unpack_s': 14.0, 'program_cache_growth': 0,
        'one': {'blocks': 1, 'read_ms': 0.4, 'equal': True}, 'run': {'blocks': 512, 'read_ms': 11.0, 'equal': True},
        'scattered': {'blocks': 1248, 'read_ms': 80.0, 'equal': True}, 'shuffled_order': {'blocks': 1248, 'read_ms': 81.0, 'equal': True},
        'two_runs': {'blocks': 70, 'read_ms': 3.0, 'equal': True}}


def run_main(report, code, extra=()):
    lines = []
    card = FakeCard(report, code)
    ttnn = FakeTTNN()
    with tempfile.TemporaryDirectory() as tmp:
        output = str(Path(tmp) / 'kv-read-probe.json')
        status = probe.main(['--output', output] + list(extra), card=card, ttnn=ttnn, log=lines.append)
        written = json.loads(Path(output).read_text())
    return status, lines, written, card, ttnn


class KvreadProbeTests(unittest.TestCase):
    def test_a_passing_check_is_a_pass_line_and_exit_zero(self):
        status, lines, written, card, ttnn = run_main(GOOD, 0)
        self.assertEqual(status, 0)
        self.assertTrue(lines[0].startswith('KV_READ_PROBE verdict=PASS whole_read_s=21.5 whole_unpack_s=14.0 program_cache_growth=0'))
        self.assertIn('problems=0', lines[0])
        self.assertEqual(written['verdict'], 'PASS')
        self.assertEqual(written['check'], GOOD)
        self.assertEqual(ttnn.calls, ['f1d'])

    def test_the_check_runs_over_the_pool_sized_cache_on_four_chips(self):
        _, _, _, card, _ = run_main(GOOD, 0)
        self.assertEqual(card.argv, ['--devices', '4', '--blocks', '19968', '--heads-per-chip', '1'])
        self.assertEqual(probe.POOL_BLOCKS, 19968)

    def test_a_byte_mismatch_is_a_fail_and_exit_one(self):
        bad = dict(GOOD, ok=False, problems=['scattered: the region read differs from the whole-cache read'])
        status, lines, written, _, _ = run_main(bad, 1)
        self.assertEqual(status, 1)
        self.assertIn('verdict=FAIL', lines[0])
        self.assertIn('problems=1', lines[0])
        self.assertEqual(written['verdict'], 'FAIL')

    def test_a_check_that_says_ok_but_exits_nonzero_is_not_a_pass(self):
        status, lines, _, _, _ = run_main(GOOD, 1)
        self.assertEqual(status, 1)
        self.assertIn('verdict=FAIL', lines[0])

    def test_an_image_without_the_graft_is_a_fail(self):
        status, lines, _, _, _ = run_main({'ok': False, 'problem': 'this ttnn has no qwen_read_blocks: the graft is not in the build'}, 1)
        self.assertEqual(status, 1)
        self.assertIn('verdict=FAIL', lines[0])

    def test_a_check_that_prints_no_json_is_not_measured_and_exit_two(self):
        class Silent:
            def main(self, argv=None):
                print('Segmentation fault')
                return 139
        lines = []
        with tempfile.TemporaryDirectory() as tmp:
            output = str(Path(tmp) / 'out.json')
            status = probe.main(['--output', output], card=Silent(), ttnn=FakeTTNN(), log=lines.append)
            written = json.loads(Path(output).read_text())
        self.assertEqual(status, 2)
        self.assertEqual(written['verdict'], 'NOT-MEASURED')
        self.assertIn('verdict=NOT-MEASURED', lines[0])

    def test_a_check_that_raises_is_not_measured(self):
        class Raising:
            def main(self, argv=None):
                raise RuntimeError('mesh did not open')
        lines = []
        with tempfile.TemporaryDirectory() as tmp:
            status = probe.main(['--output', str(Path(tmp) / 'out.json')], card=Raising(), ttnn=FakeTTNN(), log=lines.append)
        self.assertEqual(status, 2)
        self.assertIn('mesh did not open', lines[0])

    def test_the_last_json_object_is_found_after_chatter_and_before_trailing_lines(self):
        self.assertEqual(probe.last_json('noise\n{\n "ok": true\n}\n'), {'ok': True})
        self.assertIsNone(probe.last_json('no report here'))


class KvreadJobTests(unittest.TestCase):
    def read(self, **extra):
        values = dict(C2_CARDS='quad', C2_ACTIONS='reset fabric', C2_IMAGE_TAG='tp4-serve-11', C2_FABRIC_PROBE='kvread')
        values.update(extra)
        return job.read_job(values, ['general'], root=str(ROOT))

    def test_the_job_parser_accepts_the_probe_and_a_box_on_the_fabric_step(self):
        outputs = self.read(C2_BOX_MINUTES='30')
        self.assertEqual(outputs['fabric_probe'], 'kvread')
        self.assertEqual(outputs['box_minutes'], '30')

    def test_the_fabric_step_ceiling_is_the_workflows_45_minutes(self):
        self.assertEqual(self.read(C2_BOX_MINUTES='45')['box_minutes'], '45')
        with self.assertRaises(job.JobError):
            self.read(C2_BOX_MINUTES='46')

    def test_the_probe_names_need_the_fabric_action(self):
        with self.assertRaises(job.JobError):
            self.read(C2_ACTIONS='reset')

    def test_the_workflow_runs_the_wrapper_and_bounds_it_by_the_box(self):
        self.assertIn('kvread) script=tp4_kv_read_probe.py; report=kv-read-probe.json', WORKFLOW)
        self.assertIn('FABRIC_PROBE|TP4_RS_TILE|MR_PROBE|KV_READ_PROBE', WORKFLOW)
        self.assertIn('BOX_MINUTES: ${{ steps.job.outputs.box_minutes }}', WORKFLOW)
        self.assertIn('timeout -k 30 "$limit" docker run', WORKFLOW)

    def test_the_card_check_the_wrapper_runs_exists_and_is_the_one_the_gate_names(self):
        self.assertTrue(CARD_FILE.is_file())
        self.assertIn('optimisation/ttnn-op/kv_region_read/kv_region_read_card.py', (HERE / 'c2_prefix_gate.py').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
