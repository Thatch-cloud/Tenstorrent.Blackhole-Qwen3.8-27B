"""The drafter checkpoint selector (drafter_checkpoint.py): the default is today's bytes and paths, a candidate is pinned or refused, and the
contract, the smoke rule and the build script carry it. No candidate id exists yet: every candidate here is a synthetic table in a temporary directory."""

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import c2_smoke_check as check  # noqa: E402
import drafter_checkpoint as dc  # noqa: E402
import serving_c2_contract as contract  # noqa: E402

PROFILES = json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))['profiles']
PRODUCTION = 'c2-packed-tp4-8x262k-ship-prefix'


def sha(data):
    return hashlib.sha256(data).hexdigest()


GEOMETRY = dict(window=2048, num_speculative_tokens=15, rows_per_user=16, target_taps=5, hidden=5120, layers=5)
CONFIG = json.dumps(dict(hidden_size=5120, num_hidden_layers=5)).encode()
MANIFEST = b'{"files": ["w.bin"]}'


def table(**changes):
    candidate = dict(revision='a' * 40, dtype='bf16', geometry=dict(GEOMETRY), config_sha256=sha(CONFIG),
                     manifests={'attention/manifest.json': sha(MANIFEST), 'layer-1/manifest.json': sha(MANIFEST)}, weights_sha256='b' * 64)
    candidate.update(changes)
    document = json.loads((HERE / 'drafter_checkpoints.json').read_text(encoding='utf-8'))
    document['checkpoints']['cand-one'] = candidate
    return document


def lay(root, config=CONFIG, manifest=MANIFEST):
    for relative, data in (('draft-configs/cand-one/config.json', config), ('experiment-dflash-fixtures/cand-one/attention/manifest.json', manifest),
                           ('experiment-dflash-fixtures/cand-one/layer-1/manifest.json', manifest)):
        path = Path(root) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def candidate_profile(name='cand-one', gate_only=True):
    profile = copy.deepcopy(PROFILES[PRODUCTION])
    profile['env'][dc.FLAG] = name
    profile['engine']['additional-config']['qwen_fast_runtime']['fixtures'] = '/experiment-dflash-fixtures/%s' % name
    profile['engine']['speculative-config']['model'] = '/draft-configs/%s' % name
    if gate_only:
        profile['gate_only'] = True
    return profile


class TableTests(unittest.TestCase):
    def test_the_committed_table_is_valid_and_its_default_is_the_baked_checkpoint(self):
        document = dc.load()
        self.assertEqual(document['default'], 'dflash2-dedf8df6')
        self.assertEqual(sorted(document['checkpoints']), ['dflash2-dedf8df6'], 'no candidate id exists yet')
        self.assertEqual(document['checkpoints']['dflash2-dedf8df6']['revision'], 'dedf8df68adfb1afeaf7b7480c0a0243108177b4')
        self.assertEqual(dc.paths('dflash2-dedf8df6'), ('/experiment-dflash-fixture', '/draft-config'))
        self.assertEqual(dc.table_problems(table()), [])

    def test_unset_and_empty_mean_the_default_and_an_unknown_id_is_an_error(self):
        self.assertEqual(dc.selected({}), 'dflash2-dedf8df6')
        self.assertEqual(dc.selected({dc.FLAG: ''}), 'dflash2-dedf8df6')
        self.assertEqual(dc.selected({dc.FLAG: 'dflash2-dedf8df6'}), 'dflash2-dedf8df6')
        with self.assertRaises(ValueError):
            dc.selected({dc.FLAG: 'cand-one'})
        self.assertEqual(dc.selected({dc.FLAG: 'cand-one'}, table()), 'cand-one')

    def test_a_candidate_must_pin_everything_and_match_the_defaults_geometry(self):
        for key in ('config_sha256', 'manifests', 'weights_sha256', 'revision'):
            document = table()
            del document['checkpoints']['cand-one'][key]
            self.assertTrue(dc.table_problems(document), key)
        wrong = table(geometry=dict(GEOMETRY, window=1024))
        self.assertTrue(any('differs from the default' in text for text in dc.table_problems(wrong)))
        self.assertTrue(dc.table_problems(table(dtype='fp8')))
        self.assertTrue(dc.table_problems(table(manifests={'../x': 'a' * 64})))
        self.assertEqual(dc.table_problems({'default': 'x', 'checkpoints': {}}), ['the default \'x\' is not among the checkpoints'])

    def test_the_table_names_no_host_path_or_private_repository(self):
        text = (HERE / 'drafter_checkpoints.json').read_text(encoding='utf-8')
        for word in ('/home', 'thatch', 'huggingface', 'zot.', 'incoai', '@'):
            self.assertNotIn(word, text)


class ProfileTests(unittest.TestCase):
    def test_every_committed_profile_selects_the_default_and_passes(self):
        for name, profile in PROFILES.items():
            with self.subTest(name=name):
                if name.endswith('-dckdefault'):
                    self.assertEqual(profile['env'][dc.FLAG], 'dflash2-dedf8df6')      # the plumbing control names the default itself
                    self.assertIs(profile['gate_only'], True)
                else:
                    self.assertNotIn(dc.FLAG, profile.get('env') or {})
                self.assertEqual(dc.profile_problems(profile), [])

    def test_the_production_profile_points_at_the_default_paths(self):
        engine = PROFILES[PRODUCTION]['engine']
        self.assertEqual(engine['speculative-config']['model'], '/draft-config')
        self.assertEqual(engine['additional-config']['qwen_fast_runtime']['fixtures'], '/experiment-dflash-fixture')

    def test_a_candidate_profile_must_point_both_paths_at_its_baked_copy(self):
        document = table()
        self.assertEqual(dc.profile_problems(candidate_profile(), document), [])
        for path in (('additional-config', 'qwen_fast_runtime', 'fixtures'), ('speculative-config', 'model')):
            profile = candidate_profile()
            node = profile['engine']
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = '/experiment-dflash-fixture' if path[-1] == 'fixtures' else '/draft-config'
            self.assertTrue(dc.profile_problems(profile, document), path)

    def test_a_default_profile_that_points_elsewhere_is_refused(self):
        profile = copy.deepcopy(PROFILES[PRODUCTION])
        profile['engine']['speculative-config']['model'] = '/draft-configs/other'
        self.assertTrue(dc.profile_problems(profile))

    def test_an_unknown_id_in_a_profile_is_refused(self):
        self.assertTrue(dc.profile_problems(candidate_profile('nope')))

    def test_the_contract_makes_a_candidate_gate_only_and_checks_its_paths(self):
        document = table()
        original = dc.load
        dc.load = lambda path=None: document
        try:
            self.assertEqual(contract.drafter_problems(candidate_profile()), [])
            self.assertTrue(any('gate-only' in text for text in contract.drafter_problems(candidate_profile(gate_only=False))))
        finally:
            dc.load = original
        self.assertEqual(contract.drafter_problems(PROFILES[PRODUCTION]), [])

    def test_an_inherited_id_is_dropped_under_a_profile_that_names_none(self):
        environ = {dc.FLAG: 'cand-one'}
        profile = copy.deepcopy(PROFILES[PRODUCTION])
        profile['name'] = PRODUCTION
        profile['mesh_graph_descriptor'] = profile.get('mesh_graph_descriptor', '/x')
        contract.apply_environment(profile, environ)
        self.assertNotIn(dc.FLAG, environ)


class AttachTests(unittest.TestCase):
    def test_the_pinned_bytes_verify_and_the_marker_is_logged(self):
        document = table()
        with tempfile.TemporaryDirectory() as root:
            lay(root)
            lines = []
            line = dc.attach_check({dc.FLAG: 'cand-one'}, document, root, log=lines.append)
            self.assertEqual(lines, [line])
            self.assertIn('id=cand-one revision=aaaaaaaaaaaa verified=1 dtype=bf16', line)
            self.assertTrue(line.startswith(check.DRAFTER_CHECKPOINT_MARKER))

    def test_the_flag_unset_logs_nothing_and_checks_nothing(self):
        self.assertIsNone(dc.attach_check({}, table(), '/nonexistent', log=lambda text: self.fail(text)))

    def test_a_changed_config_a_changed_manifest_or_a_missing_file_refuses_the_attach(self):
        document = table()
        for label, kwargs in (('config', dict(config=CONFIG + b' ')), ('manifest', dict(manifest=MANIFEST + b' '))):
            with tempfile.TemporaryDirectory() as root:
                lay(root, **kwargs)
                with self.assertRaises(ValueError, msg=label):
                    dc.attach_check({dc.FLAG: 'cand-one'}, document, root, log=lambda text: None)
        with tempfile.TemporaryDirectory() as root:
            lay(root)
            (Path(root) / 'draft-configs/cand-one/config.json').unlink()
            with self.assertRaises(ValueError):
                dc.attach_check({dc.FLAG: 'cand-one'}, document, root, log=lambda text: None)

    def test_a_config_that_disagrees_with_the_pinned_geometry_is_refused(self):
        config = json.dumps(dict(hidden_size=4096, num_hidden_layers=5)).encode()
        document = table(config_sha256=sha(config))
        with tempfile.TemporaryDirectory() as root:
            lay(root, config=config)
            with self.assertRaises(ValueError) as caught:
                dc.attach_check({dc.FLAG: 'cand-one'}, document, root, log=lambda text: None)
            self.assertIn('hidden_size', str(caught.exception))

    def test_the_default_has_no_pins_to_check(self):
        self.assertEqual(dc.attach_problems('dflash2-dedf8df6', root='/nonexistent'), [])
        line = dc.attach_check({dc.FLAG: 'dflash2-dedf8df6'}, log=lambda text: None)
        self.assertIn('id=dflash2-dedf8df6 revision=dedf8df68adf verified=1 dtype=bf8', line)


class SmokeRuleTests(unittest.TestCase):
    def test_a_flag_without_its_verified_line_is_a_problem_and_the_line_clears_it(self):
        env = {dc.FLAG: 'cand-one'}
        self.assertTrue(check.drafter_checkpoint_problems(env, 'nothing'))
        self.assertTrue(check.drafter_checkpoint_problems(env, '[PINDIAG] drafter checkpoint id=other revision=aaaaaaaaaaaa verified=1 dtype=bf16'))
        self.assertTrue(check.drafter_checkpoint_problems(env, '[PINDIAG] drafter checkpoint id=cand-one revision=aaaaaaaaaaaa verified=0 dtype=bf16'))
        self.assertEqual(check.drafter_checkpoint_problems(env, '[QWEN-C2] [PINDIAG] drafter checkpoint id=cand-one revision=aaaaaaaaaaaa verified=1 dtype=bf16'), [])

    def test_a_profile_that_names_none_needs_no_line(self):
        self.assertEqual(check.drafter_checkpoint_problems({}, ''), [])
        self.assertEqual(check.lever_engagement_problems({dc.FLAG: 'cand-one'}, ''), check.drafter_checkpoint_problems({dc.FLAG: 'cand-one'}, ''))

    def test_check_applies_the_rule_with_the_profile_env(self):
        problems, _ = check.check('', '', False, env={dc.FLAG: 'cand-one', 'QWEN_FAST_TP': '2'})
        self.assertTrue(any(check.DRAFTER_CHECKPOINT_MARKER in text for text in problems))


class BuildTests(unittest.TestCase):
    def test_the_build_stages_candidates_from_a_pinned_list_and_the_dockerfile_always_has_both_directories(self):
        script = (HERE / 'build-c2-serving-image.sh').read_text(encoding='utf-8')
        for word in ('C2_DRAFTER_CANDIDATES', 'fixtures', 'draft-configs', '.placeholder'):
            self.assertIn(word, script)
        dockerfile = (ROOT / 'docker' / 'qwen-c2-serving.Dockerfile').read_text(encoding='utf-8')
        self.assertIn('COPY fixtures/ /experiment-dflash-fixtures/', dockerfile)
        self.assertIn('COPY draft-configs/ /draft-configs/', dockerfile)
        self.assertIn('COPY fixture/ /experiment-dflash-fixture/', dockerfile, 'the default checkpoint\'s copy is untouched')

    def test_the_module_and_its_table_are_in_the_overlay_only(self):
        overlay = (ROOT / 'docker' / 'qwen-c2-overlay.txt').read_text(encoding='utf-8')
        self.assertIn('scripts/ci/drafter_checkpoint.py', overlay)
        self.assertIn('scripts/ci/drafter_checkpoints.json', overlay)
        for relative in ('docker/qwen-fast-serving.Dockerfile', '.github/workflows/qwen-fast-serving-image.yml'):
            self.assertNotIn('drafter_checkpoint', (ROOT / relative).read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
