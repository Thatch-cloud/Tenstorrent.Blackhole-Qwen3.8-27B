"""The disaggregation step-1 window pack (references/tp4-disagg-jobs, docs/tp4-disagg-step1.md) on the CPU: every template parses with the job parser, the ORDER rows match
the files and the one image tag, the pair job carries no reset and comes straight after a four-card reset, the TP2 and TP4 G1 arms share their bench shapes (so r and k are read
on the same prompts), the production arm names concurrent8_steady, and no template names a host.

Run at py 3.11: `py -3.11 -B -m unittest test_tp4_disagg_jobs` from scripts/ci."""

import json
from pathlib import Path
import re
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import c2_serving_job as job  # noqa: E402

PACK = HERE / 'references' / 'tp4-disagg-jobs'
PROFILES = json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))['profiles']
TAG = 'tp4-next2-1'
SHIP = 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-traffic'


def values(name):
    return job.parse_env((PACK / (name + '.env')).read_text(encoding='utf-8'))


def parsed(name):
    return job.read_job(values(name), sorted(PROFILES), root=ROOT)


def order():
    return [line.split() for line in (PACK / 'ORDER.txt').read_text(encoding='utf-8').splitlines() if line.strip() and not line.startswith('#')]


class PackTests(unittest.TestCase):
    def test_every_template_has_an_order_row_one_tag_and_parses(self):
        names = sorted(path.name[:-4] for path in PACK.glob('*.env'))
        self.assertEqual(names, sorted(row[0] for row in order()))
        for row in order():
            with self.subTest(row=row[0]):
                self.assertEqual(len(row), 4)
                self.assertIn(row[1], ('stop', 'soft'))
                self.assertEqual(row[2], TAG)
                self.assertTrue(row[3].isdigit())
                self.assertEqual(parsed(row[0])['tag'], TAG)

    def test_the_pair_job_has_no_reset_and_follows_a_four_card_reset(self):
        names = [row[0] for row in order()]
        pair = parsed('D2-tp2-g1-bench')
        self.assertEqual((pair['cards'], pair['profile']), ('pair', 'general-2link'))
        self.assertNotIn('reset', pair['actions'].split())
        self.assertNotIn('C2_CARDS', values('D2-tp2-g1-bench'))
        before = parsed(names[names.index('D2-tp2-g1-bench') - 1])
        self.assertEqual(before['cards'], 'quad')
        self.assertIn('reset', before['actions'].split())
        self.assertEqual(names[0], 'X0-status-rescan-reset')
        self.assertEqual(order()[0][1], 'stop')

    def test_the_tp2_and_tp4_g1_arms_share_their_shapes_and_collect_agreement(self):
        tp2, tp4 = parsed('D2-tp2-g1-bench'), parsed('D1-tp4-g1-bench')
        self.assertEqual(tp4['profile'], 'general-tp4-bench')
        self.assertEqual(PROFILES['general-tp4-bench'].get('mesh_device'), 'P150x4')
        self.assertIsNone(PROFILES['general-2link'].get('engine', {}).get('additional-config', {}).get('qwen_fast_t16'))
        pair_shapes = tp2['bench_shapes'].split(',')
        quad_shapes = tp4['bench_shapes'].split(',')
        for shape in pair_shapes:
            self.assertIn(shape, quad_shapes, 'r and k are read on the same prompts: %s' % shape)
        for result in (tp2, tp4):
            self.assertEqual(result['tests'].split(','), ['warmup', 'bench', 'agreement'])
        # the pair profile's context bounds its shapes; the TP4 arm adds only what the pair cannot hold
        limit = PROFILES['general-2link']['engine']['max-model-len']
        self.assertTrue(all(int(shape.split('x')[1]) < limit - 4096 for shape in pair_shapes))
        self.assertTrue(all(int(shape.split('x')[0]) <= PROFILES['general-2link']['engine']['max-num-seqs'] for shape in pair_shapes))
        self.assertEqual(sorted(set(quad_shapes) - set(pair_shapes)), ['1x130000', '8x4096'])

    def test_the_production_arm_serves_the_ship_profile_with_the_steady_test(self):
        ship = parsed('D3-tp4-ship-bench')
        self.assertEqual((ship['cards'], ship['profile']), ('quad', SHIP))
        tests = ship['tests'].split(',')
        self.assertIn(job.STEADY_EIGHT_TEST, tests)
        self.assertIn('stall8_cold128k', tests)
        self.assertNotIn('C2_SMOKE_PARTIAL', values('D3-tp4-ship-bench'))
        cap = PROFILES[SHIP]['max_prompt_tokens']
        for shape in ship['bench_shapes'].split(','):
            streams, prompt = (int(part) for part in shape.split('x'))
            self.assertLessEqual(streams, PROFILES[SHIP]['engine']['max-num-seqs'])
            # tp_decode_bench cuts 3.6 characters per nominal token and the corpus is at least 3.7 per real token
            self.assertLessEqual(prompt * 3.6 / 3.7, cap, shape)
        bare = dict(values('D3-tp4-ship-bench'))
        bare['C2_SMOKE_TESTS'] = 'warmup,bench'
        with self.assertRaises(job.JobError):
            job.read_job(bare, sorted(PROFILES), root=ROOT)

    def test_no_template_names_a_host_an_address_or_a_home_path(self):
        pattern = re.compile(r'(/home/|/Users/|\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|\.local\b|zot\.|sha256:[0-9a-f]{16}|ghp_|token=)')
        for path in sorted(PACK.iterdir()):
            with self.subTest(path=path.name):
                self.assertIsNone(pattern.search(path.read_text(encoding='utf-8')))


if __name__ == '__main__':
    unittest.main()
