"""make_fusion_profiles.py (docs/tp4-fusion.md): the op-fusion programme's work-package manifests (scripts/ci/fusion-wp/<WP>.json) become the twin profiles, the smoke
tables, the overlay lines, the CPU allowlist and the tp_addresses rows, deterministically, and every manifest entry lands.

The checked-in files must be what the manifests in the tree generate. The generator is also run on a synthetic set of five manifests (one per package, shaped as the packages'
are) against copies of the real files, in a scratch tree that holds the files they name."""

import ast
import copy
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c2_overlay  # noqa: E402
import c2_smoke_check  # noqa: E402
import make_fusion_profiles as gen  # noqa: E402
import profile_twins  # noqa: E402

NEW_FILES = (
    'scripts/ci/tp4_shard_argmax_fold2.cpp',
    'scripts/ci/tp4_kv_page_writer.py', 'scripts/ci/tp4_kv_page_writer_pages.cpp', 'scripts/ci/kv_page_writer_smoke.py',
    'scripts/ci/ccl_options_tp.py', 'scripts/ci/distributed_norm_gather_tp.py',
    'scripts/ci/draft_reduce_tp.py', 'scripts/ci/draft_reduce_tp.cpp', 'scripts/ci/draft_tail_tp.py', 'scripts/ci/draft_tail_tp.cpp',
    'scripts/ci/draft_permute_tp.py', 'scripts/ci/draft_permute_tp.cpp', 'scripts/ci/draft_headnorm_tp.py',
    'scripts/ci/test_tp4_shard_argmax_fold2.py', 'scripts/ci/test_tp4_kv_page_writer.py', 'scripts/ci/test_ccl_options_tp.py',
    'scripts/ci/test_draft_reduce_tp.py', 'scripts/ci/test_draft_permute_tp.py', 'scripts/ci/test_draft_permute_smoke.py',
    'optimisation/ttnn-op/ccl_sweep/test_sweep.py',
)

MANIFESTS = {
    'WP1.json': {
        'wp': 'WP1', 'branch': 'tp4/fx-wp1',
        'levers': [{'id': 's1', 'name': 'S1 shard argmax', 'flag': 'QWEN_FAST_TP4_SHARD_ARGMAX', 'audit_flag': 'QWEN_FAST_TP4_SHARD_ARGMAX_AUDIT',
                    'marker': 'tp4 shard argmax', 'reason': 'one scan over the vocabulary shard instead of untilize, argmax and gather'}],
        'image_files': ['scripts/ci/tp4_shard_argmax_fold2.cpp', 'scripts/ci/tp4_shard_argmax.py'],
        'tests': ['test_tp4_shard_argmax_fold2', 'test_tp4_shard_argmax'],
    },
    'WP2.json': {
        'wp': 'WP2', 'notes': 'free text',
        'levers': [{'id': 'kvw', 'name': 'K/V page writer', 'flag': 'QWEN_FAST_TP4_KV_PAGE_WRITER', 'audit_flag': 'QWEN_FAST_TP4_KV_PAGE_WRITER_AUDIT',
                    'marker': 'tp4 kv page writer', 'env': {'QWEN_FAST_TP4_KV_PAGE_WIDTH': 64}, 'reason': 'page-parallel K/V writer'}],
        'image_files': [{'path': 'scripts/ci/tp4_kv_page_writer.py', 'reason': 'the writer'}, {'path': 'scripts/ci/tp4_kv_page_writer_pages.cpp', 'reason': 'its kernel'},
                        'scripts/ci/kv_page_writer_smoke.py'],
        'tests': ['test_tp4_kv_page_writer'],
    },
    'WP5.json': {
        'wp': 'WP5',
        'levers': [{'id': 'ccl', 'name': 'CCL per-call options', 'flag': 'QWEN_FAST_TP4_CCL_OPTIONS', 'value': 'rs2', 'marker': 'tp4 ccl options',
                    'reason': 'reduce-scatter and all-gather per-call options'}],
        'image_copy': ['scripts/ci/ccl_options_tp.py', 'scripts/ci/distributed_norm_gather_tp.py'],
        'cpu_tests': ['test_ccl_options_tp', {'discover': 'optimisation/ttnn-op/ccl_sweep'}],
    },
    'WP6.json': {
        'wp': 'WP6',
        'levers': [{'id': 'dr', 'name': 'drafter local reduce', 'flag': 'QWEN_FAST_TP4_DRAFT_REDUCE', 'audit_flag': 'QWEN_FAST_TP4_DRAFT_REDUCE_AUDIT',
                    'markers': {'engaged': '[PINDIAG] tp4 draft reduce engaged', 'fell_back': '[PINDIAG] tp4 draft reduce fell back',
                                'audit': '[PINDIAG] tp4 draft reduce audit'}, 'reason': 'one local reduce for the drafter chains'},
                   {'id': 'tail', 'name': 'drafter residual and SwiGLU kernels', 'flag': 'QWEN_FAST_TP4_DRAFT_TAIL', 'marker': 'tp4 draft tail',
                    'reason': 'residual add and SwiGLU in one launch'}],
        'profiles': [{'name': 'dr-tail', 'env': {'QWEN_FAST_TP4_DRAFT_REDUCE': '1', 'QWEN_FAST_TP4_DRAFT_TAIL': '1'}, 'reason': 'both drafter levers together'}],
        'image_files': ['scripts/ci/draft_reduce_tp.py', 'scripts/ci/draft_reduce_tp.cpp', 'scripts/ci/draft_tail_tp.py', 'scripts/ci/draft_tail_tp.cpp'],
        'tests': ['test_draft_reduce_tp'],
    },
    'WP7.json': {
        'wp': 'WP7',
        'levers': [{'id': 'dperm', 'name': 'drafter permutation kernels', 'flag': 'QWEN_FAST_TP4_DRAFT_PERMUTE', 'audit_flag': 'QWEN_FAST_TP4_DRAFT_PERMUTE_AUDIT',
                    'marker': 'tp4 draft permute', 'reason': 'the K/V assembly, query fold and unfold as one kernel each'}],
        'smoke': [{'flag': 'QWEN_FAST_TP4_DRAFT_QKV1', 'marker': 'tp4 draft qkv1', 'what': 'fused drafter q, k and v projection'}],
        'image_files': ['scripts/ci/draft_permute_tp.py', 'scripts/ci/draft_permute_tp.cpp', 'scripts/ci/draft_headnorm_tp.py'],
        'tests': ['test_draft_permute_tp', 'test_draft_permute_smoke.PermuteSmokeTests'],
        'tp_addresses': {'module_twins': [['quad_draft', 'quad_draft_tp'], ['draft_kv_projection', 'draft_kv_projection_tp']],
                         'flagged_module_twins': [{'module': 'draft_kv_projection', 'flag': 'QWEN_FAST_TP4_DRAFT_QKV1', 'gate': 'draft_permute_tp:qkv1_enabled'}]},
    },
}

PARENT_NAME = gen.PARENT


def scratch_root():
    """A tree holding the files the synthetic manifests name (empty files) beside the real ones they also name."""
    root = Path(tempfile.mkdtemp(prefix='fusion-wp-'))
    for name in NEW_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('', encoding='utf-8')
    for real in ('scripts/ci/tp4_shard_argmax.py', 'scripts/ci/test_tp4_shard_argmax.py', 'scripts/ci/quad_draft_tp.py', 'scripts/ci/draft_kv_projection_tp.py'):
        path = root / real
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('', encoding='utf-8')
    return root


def write_manifests(directory, manifests, reverse=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    items = list(manifests.items())
    for name, raw in (reversed(items) if reverse else items):
        (directory / name).write_text(json.dumps(raw, indent=1), encoding='utf-8')
    return directory


def plan_of(manifests, reverse=False):
    directory = write_manifests(tempfile.mkdtemp(prefix='fusion-wp-m-'), manifests, reverse)
    return gen.normalise(gen.read_manifests(directory))


def real_texts():
    return gen.read_texts()


def generate(manifests, texts=None, root=None, reverse=False):
    return gen.generate(texts or real_texts(), plan_of(manifests, reverse), root or ROOT)


ROOT = scratch_root()
OUT, REPORT = generate(MANIFESTS)


def block(text, name=None):
    """The lines between the fence lines of the (named) block."""
    lines = text.split('\n')
    start = next(i for i, line in enumerate(lines) if gen.BEGIN in line and (name is None or line.strip().endswith('[%s]' % name)))
    stop = next(i for i in range(start + 1, len(lines)) if lines[i].strip().lstrip('#').strip() == gen.END)
    return [line.strip() for line in lines[start + 1:stop]]


def literal(text, name):
    """The value of a module-level `name = <literal>` assignment of python source."""
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


class CheckedInTests(unittest.TestCase):
    def test_the_checked_in_files_are_what_the_manifests_in_the_tree_generate(self):
        plan = gen.normalise(gen.read_manifests())
        texts = real_texts()
        new, report = gen.generate(texts, plan, gen.REPO)
        gen.validate(new)
        for name in gen.TARGETS:
            self.assertEqual(new[name], texts[name], '%s: run make_fusion_profiles.py --write' % gen.PATHS[name])
        self.assertEqual({row[3] for row in report} - {'landed', 'existing'}, set())

    def test_the_command_line_check_agrees(self):
        self.assertEqual(gen.main(['--check']), 0)

    def test_the_five_files_carry_one_fenced_block_each_and_tp_addresses_four(self):
        texts = real_texts()
        for name in ('smoke', 'overlay', 'cpu'):
            lines = texts[name].split('\n')
            self.assertEqual(sum(1 for line in lines if gen.BEGIN in line), 1, name)
            self.assertEqual(sum(1 for line in lines if line.strip().lstrip('#').strip() == gen.END), 1, name)
        lines = texts['addresses'].split('\n')
        self.assertEqual([line.strip().rsplit('[', 1)[1] for line in lines if gen.BEGIN in line],
                         ['twins]', 'module_twins]', 'flagged_twins]', 'flagged_module_twins]'])

    def test_the_twin_names_are_registered_with_the_census_exemptions(self):
        names = gen.twin_names()
        self.assertEqual(sorted(set(names)), sorted(names))
        for name in names:
            self.assertIn(name, profile_twins.twin_names())
            self.assertIn(name, json.loads(real_texts()['profiles'])['profiles'])

    def test_with_no_manifest_every_block_is_empty_and_nothing_else_moves(self):
        texts = real_texts()
        new, report = gen.generate(texts, gen.normalise([]), gen.REPO)
        self.assertEqual(report, [])
        profiles = json.loads(new['profiles'])['profiles']
        self.assertFalse([name for name in profiles if name.startswith(gen.NAMESPACE)])
        for name in ('smoke', 'overlay', 'cpu', 'addresses'):
            self.assertEqual(gen.outside(new[name]), gen.outside(texts[name]), name)
        self.assertEqual(block(new['smoke']), ['FUSION_LEVERS = (', ')', 'FUSION_AUDITS = (', ')', 'FUSION_RULES = (', ')'])
        self.assertEqual(block(new['overlay']), [])
        for key in gen.TP_KEYS:
            self.assertEqual(block(new['addresses'], key), [], key)
        self.assertEqual([line for line in block(new['cpu']) if line], [])


class DeterminismTests(unittest.TestCase):
    def test_two_generations_are_identical_and_the_file_order_of_the_manifests_does_not_matter(self):
        again, report = generate(MANIFESTS)
        reversed_out, reversed_report = generate(MANIFESTS, reverse=True)
        self.assertEqual(again, OUT)
        self.assertEqual(report, REPORT)
        self.assertEqual(reversed_out, OUT)
        self.assertEqual(reversed_report, REPORT)

    def test_a_generation_is_idempotent(self):
        texts = real_texts()
        once = dict(texts, **OUT)
        twice, _ = gen.generate(once, plan_of(MANIFESTS), ROOT)
        self.assertEqual(twice, OUT)

    def test_a_dropped_manifest_drops_its_twins_and_its_rows(self):
        once = dict(real_texts(), **OUT)
        smaller = dict((k, v) for k, v in MANIFESTS.items() if k != 'WP2.json')
        out, _ = gen.generate(once, plan_of(smaller), ROOT)
        profiles = json.loads(out['profiles'])['profiles']
        self.assertFalse([name for name in profiles if name.endswith('-fx-kvw') or name.endswith('-fx-kvw-audit')])
        self.assertNotIn('QWEN_FAST_TP4_KV_PAGE_WRITER', out['smoke'])
        self.assertNotIn('tp4_kv_page_writer', out['overlay'])
        self.assertNotIn('test_tp4_kv_page_writer', out['cpu'])

    def test_only_the_fenced_blocks_and_the_twins_change(self):
        texts = real_texts()
        for name in ('smoke', 'overlay', 'cpu', 'addresses'):
            self.assertEqual(gen.outside(OUT[name]), gen.outside(texts[name]), name)
        before = json.loads(texts['profiles'])['profiles']
        after = json.loads(OUT['profiles'])['profiles']
        self.assertEqual([n for n in after if not n.startswith(gen.NAMESPACE)], [n for n in before if not n.startswith(gen.NAMESPACE)])
        for name, profile in before.items():
            if not name.startswith(gen.NAMESPACE):
                self.assertEqual(after[name], profile, name)
        # the twins sit just before the region-read audit twins that close the file
        import make_kvread_profiles

        names = list(after)
        closing = list(make_kvread_profiles.twin_names())
        self.assertEqual(names[-len(closing):], closing)
        self.assertTrue(names[-len(closing) - 1].startswith(gen.NAMESPACE))

    def test_the_other_generators_still_agree_on_the_generated_file(self):
        import make_kvread_profiles
        import make_octo2_profiles
        import make_octo_profiles
        import make_parked_profiles
        import make_round_host_profiles
        import make_w2_kill_profiles

        data = json.loads(OUT['profiles'])
        for other in (make_kvread_profiles, make_octo2_profiles, make_octo_profiles, make_parked_profiles, make_round_host_profiles, make_w2_kill_profiles):
            self.assertEqual(other.render(other.generate(copy.deepcopy(data))), OUT['profiles'], other.__name__)


class EveryEntryLandsTests(unittest.TestCase):
    def test_every_entry_has_a_report_row_and_none_is_unplaced(self):
        self.assertEqual({row[3] for row in REPORT}, {'landed', 'existing'})
        plan = plan_of(MANIFESTS)
        expected = (sum(2 if lever['audit_flag'] else 1 for lever in plan['levers']) + len(plan['profiles'])      # profiles
                    + len(plan['image_files']) + len(plan['tests'])
                    + sum(len(rows) for rows in plan['tp'].values()))
        placed = [row for row in REPORT if row[1].startswith(('profile ', 'image ', 'tests ', 'tp_addresses '))]
        self.assertEqual(len(placed), expected)
        smoke_rows = [row for row in REPORT if row[2] == 'c2_smoke_check.py']
        self.assertEqual(len(smoke_rows), sum(1 + (1 if lever['audit_flag'] else 0) for lever in plan['levers'])
                         + sum(1 + (1 if item['audit_flag'] else 0) for item in plan['smoke']))

    def test_the_twin_profiles_are_in_the_file_with_the_flag_and_nothing_else(self):
        profiles = json.loads(OUT['profiles'])['profiles']
        parent = profiles[PARENT_NAME]
        want = {
            'fx-s1': {'QWEN_FAST_TP4_SHARD_ARGMAX': '1'},
            'fx-s1-audit': {'QWEN_FAST_TP4_SHARD_ARGMAX': '1', 'QWEN_FAST_TP4_SHARD_ARGMAX_AUDIT': '1'},
            'fx-kvw': {'QWEN_FAST_TP4_KV_PAGE_WRITER': '1', 'QWEN_FAST_TP4_KV_PAGE_WIDTH': '64'},
            'fx-kvw-audit': {'QWEN_FAST_TP4_KV_PAGE_WRITER': '1', 'QWEN_FAST_TP4_KV_PAGE_WIDTH': '64', 'QWEN_FAST_TP4_KV_PAGE_WRITER_AUDIT': '1'},
            'fx-ccl': {'QWEN_FAST_TP4_CCL_OPTIONS': 'rs2'},
            'fx-dr': {'QWEN_FAST_TP4_DRAFT_REDUCE': '1'},
            'fx-dr-audit': {'QWEN_FAST_TP4_DRAFT_REDUCE': '1', 'QWEN_FAST_TP4_DRAFT_REDUCE_AUDIT': '1'},
            'fx-tail': {'QWEN_FAST_TP4_DRAFT_TAIL': '1'},
            'fx-dr-tail': {'QWEN_FAST_TP4_DRAFT_REDUCE': '1', 'QWEN_FAST_TP4_DRAFT_TAIL': '1'},
            'fx-dperm': {'QWEN_FAST_TP4_DRAFT_PERMUTE': '1'},
            'fx-dperm-audit': {'QWEN_FAST_TP4_DRAFT_PERMUTE': '1', 'QWEN_FAST_TP4_DRAFT_PERMUTE_AUDIT': '1'},
        }
        generated = [name for name in profiles if name.startswith(gen.NAMESPACE)]
        self.assertEqual(sorted(name[len(gen.BASE) + 1:] for name in generated), sorted(want))
        for suffix, env in want.items():
            twin = copy.deepcopy(profiles[gen.BASE + '-' + suffix])
            self.assertIs(twin.pop('gate_only'), True, suffix)
            self.assertTrue(twin.pop('description').startswith('GATE ONLY, UNQUALIFIED (the op-fusion programme'), suffix)
            mine = twin.pop('env')
            for name, value in env.items():
                self.assertEqual(mine.pop(name), value, (suffix, name))
            theirs = copy.deepcopy(parent)
            self.assertIn(gen.WAIVER_FIELD, theirs)
            theirs.pop(gen.WAIVER_FIELD)
            theirs.pop('description')
            self.assertEqual(mine, theirs.pop('env'), 'nothing but the named env differs: ' + suffix)
            self.assertEqual(twin, theirs, 'nothing but env, gate_only, the waiver and the description differs: ' + suffix)
        self.assertNotIn('gate_only', parent)
        self.assertIn(gen.WAIVER_FIELD, parent)

    def test_the_descriptions_name_the_parent_the_control_and_the_manifest(self):
        profiles = json.loads(OUT['profiles'])['profiles']
        text = profiles[gen.BASE + '-fx-s1-audit']['description']
        for word in (PARENT_NAME, 'scripts/ci/fusion-wp/WP1.json', 'QWEN_FAST_TP4_SHARD_ARGMAX_AUDIT=1', 'exact=True', 'not a timing arm', 'never hand-merge',
                     'The lever is default off'):
            self.assertIn(word, text)
        self.assertIn('S1 shard argmax', profiles[gen.BASE + '-fx-s1']['description'])

    def test_the_smoke_tables_hold_each_new_lever_and_leave_the_static_ones_alone(self):
        levers = literal('\n'.join(block(OUT['smoke'])), 'FUSION_LEVERS')
        audits = literal('\n'.join(block(OUT['smoke'])), 'FUSION_AUDITS')
        flags = [row[0] for row in levers]
        self.assertNotIn('QWEN_FAST_TP4_SHARD_ARGMAX', flags, 'the sampdraft table already dispatches S1')
        self.assertEqual(flags, ['QWEN_FAST_TP4_KV_PAGE_WRITER', 'QWEN_FAST_TP4_CCL_OPTIONS', 'QWEN_FAST_TP4_DRAFT_REDUCE', 'QWEN_FAST_TP4_DRAFT_TAIL',
                                 'QWEN_FAST_TP4_DRAFT_PERMUTE', 'QWEN_FAST_TP4_DRAFT_QKV1'])
        rows = dict((row[0], row) for row in levers)
        self.assertEqual(rows['QWEN_FAST_TP4_CCL_OPTIONS'][:4], ('QWEN_FAST_TP4_CCL_OPTIONS', 'rs2', '[PINDIAG] tp4 ccl options engaged',
                                                                 '[PINDIAG] tp4 ccl options fell back'))
        self.assertEqual(rows['QWEN_FAST_TP4_DRAFT_REDUCE'][2:4], ('[PINDIAG] tp4 draft reduce engaged', '[PINDIAG] tp4 draft reduce fell back'))
        self.assertEqual([row[0] for row in audits], ['QWEN_FAST_TP4_KV_PAGE_WRITER_AUDIT', 'QWEN_FAST_TP4_DRAFT_REDUCE_AUDIT', 'QWEN_FAST_TP4_DRAFT_PERMUTE_AUDIT'])
        self.assertEqual(dict((row[0], row[1]) for row in audits)['QWEN_FAST_TP4_DRAFT_PERMUTE_AUDIT'], '[PINDIAG] tp4 draft permute audit')
        # the static tables above the fence are untouched
        self.assertEqual(literal(real_texts()['smoke'], 'SAMPDRAFT_LEVERS'), literal(OUT['smoke'], 'SAMPDRAFT_LEVERS'))

    def test_the_overlay_lists_each_new_file_once_and_the_parser_accepts_it(self):
        entries = c2_overlay.parse_manifest(OUT['overlay'])
        sources = [entry.source for entry in entries]
        self.assertEqual(len(sources), len(set(sources)))
        added = set(block(OUT['overlay'])) - set(line for line in block(OUT['overlay']) if line.startswith('#'))
        self.assertEqual(sorted(added), sorted(set(
            ['scripts/ci/tp4_shard_argmax_fold2.cpp', 'scripts/ci/tp4_kv_page_writer.py', 'scripts/ci/tp4_kv_page_writer_pages.cpp', 'scripts/ci/kv_page_writer_smoke.py',
             'scripts/ci/ccl_options_tp.py', 'scripts/ci/distributed_norm_gather_tp.py', 'scripts/ci/draft_reduce_tp.py', 'scripts/ci/draft_reduce_tp.cpp',
             'scripts/ci/draft_tail_tp.py', 'scripts/ci/draft_tail_tp.cpp', 'scripts/ci/draft_permute_tp.py', 'scripts/ci/draft_permute_tp.cpp',
             'scripts/ci/draft_headnorm_tp.py'])))
        self.assertNotIn('scripts/ci/tp4_shard_argmax.py', added, 'already listed above the fence: named once')
        for source in added:
            self.assertIn(source, sources)

    def test_the_cpu_allowlist_names_each_new_module_in_the_regression_step(self):
        import yaml

        data = yaml.safe_load(OUT['cpu'])
        step = next(s for job in data['jobs'].values() for s in job['steps'] if s.get('name') == gen.SUITE_STEP)
        lines = [line.strip() for line in step['run'].splitlines() if line.strip()]
        words = set()
        for line in lines:
            parts = shlex.split(line)
            if parts[:4] == ['python', '-B', '-m', 'unittest']:
                words.update(parts[4:])
        for module in ('test_tp4_shard_argmax_fold2', 'test_tp4_kv_page_writer', 'test_ccl_options_tp', 'test_draft_reduce_tp', 'test_draft_permute_tp',
                       'test_draft_permute_smoke.PermuteSmokeTests'):
            self.assertIn(module, words)
        self.assertIn("python -B -m unittest discover -s optimisation/ttnn-op/ccl_sweep -p 'test_*.py'", lines)
        # the generated lines name every module once, and test_tp4_shard_argmax (already allowlisted) is not added twice
        self.assertEqual(sum(1 for line in lines if 'test_tp4_shard_argmax ' in line + ' '), 1)
        self.assertEqual(sum(line.count('test_tp4_kv_page_writer') for line in lines), 1)

    def test_every_line_of_the_step_splits_the_way_the_any_ref_cpu_suite_splits_it(self):
        """cpu_suite_plan.split runs shlex.split on every line of the regression step: a stray quote in a fence comment would kill the whole plan."""
        import yaml

        data = yaml.safe_load(OUT['cpu'])
        step = next(s for job in data['jobs'].values() for s in job['steps'] if s.get('name') == gen.SUITE_STEP)
        for line in step['run'].splitlines():
            if line.strip():
                shlex.split(line.strip())

    def test_the_tp_addresses_rows_are_in_their_blocks_and_the_module_executes(self):
        self.assertEqual(block(OUT['addresses'], 'twins'), [])
        self.assertEqual(block(OUT['addresses'], 'flagged_twins'), [])
        self.assertEqual(block(OUT['addresses'], 'module_twins'), ["('draft_kv_projection', 'draft_kv_projection_tp'),  # WP7"])
        self.assertEqual(block(OUT['addresses'], 'flagged_module_twins'),
                         ["'draft_kv_projection': ('QWEN_FAST_TP4_DRAFT_QKV1', _fusion_gate('draft_permute_tp:qkv1_enabled')),  # WP7"])
        namespace = {'__name__': 'tp_addresses_fusion_trial'}
        exec(compile(OUT['addresses'], 'tp_addresses.py', 'exec'), namespace)
        self.assertIn(('draft_kv_projection', 'draft_kv_projection_tp'), namespace['MODULE_TWINS'])
        self.assertEqual(sum(1 for row in namespace['MODULE_TWINS'] if row == ('quad_draft', 'quad_draft_tp')), 1, 'a row the file already has is not repeated')
        self.assertIn('draft_kv_projection', namespace['FLAGGED_MODULE_TWINS'])
        self.assertEqual(namespace['FLAGGED_MODULE_TWINS']['draft_kv_projection'][0], 'QWEN_FAST_TP4_DRAFT_QKV1')

    def test_a_flagged_row_binds_only_when_its_gate_says_so_and_imports_nothing_until_asked(self):
        import tp_addresses

        module = types.ModuleType('fusion_gate_probe')
        module.on = lambda environ: (environ or {}).get('X') == '1'
        gate = tp_addresses._fusion_gate('fusion_gate_probe:on')
        self.assertNotIn('fusion_gate_probe', sys.modules)
        with mock.patch.dict(sys.modules, {'fusion_gate_probe': module}):
            self.assertIs(gate({'X': '1'}), True)
            self.assertIs(gate({}), False)

    def test_the_static_tp_addresses_binding_is_unchanged_by_the_fence(self):
        import tp_addresses

        text = (HERE / 'tp_addresses.py').read_text(encoding='utf-8')
        static = {}
        exec(compile(gen.outside(text), 'tp_addresses.py', 'exec'), static)
        for name in ('TWINS', 'MODULE_TWINS'):
            self.assertEqual(getattr(tp_addresses, name)[:len(static[name])], static[name], name)
        self.assertEqual(set(tp_addresses.FLAGGED_TWINS), set(static['FLAGGED_TWINS']) | set(
            key for key in tp_addresses.FLAGGED_TWINS if key not in static['FLAGGED_TWINS']))
        rows, modules = tp_addresses.bound_twins({'QWEN_FAST_TP': '4'})
        self.assertIn(('quad_draft', 'quad_draft_tp'), modules)


class SmokeRulesTests(unittest.TestCase):
    LEVERS = (('QWEN_FAST_X', '1', '[PINDIAG] tp4 x engaged', '[PINDIAG] tp4 x fell back', 'x lever'),
              ('QWEN_FAST_Y', 'rs2', '[PINDIAG] tp4 y engaged', '[PINDIAG] tp4 y fell back', 'y lever'))
    AUDITS = (('QWEN_FAST_X_AUDIT', '[PINDIAG] tp4 x audit', 'x lever'),)

    def problems(self, text, env):
        with mock.patch.object(c2_smoke_check, 'FUSION_LEVERS', self.LEVERS), mock.patch.object(c2_smoke_check, 'FUSION_AUDITS', self.AUDITS):
            return c2_smoke_check.fusion_problems(text, env)

    def test_a_profile_that_asks_for_a_lever_and_logs_no_engaged_line_fails(self):
        self.assertEqual(len(self.problems('nothing\n', {'QWEN_FAST_X': '1'})), 1)
        self.assertEqual(self.problems('[PINDIAG] tp4 x engaged site=a\n', {'QWEN_FAST_X': '1'}), [])

    def test_the_value_must_match_and_a_profile_without_the_flag_is_not_judged(self):
        self.assertEqual(self.problems('nothing\n', {'QWEN_FAST_Y': '1'}), [])
        self.assertEqual(len(self.problems('nothing\n', {'QWEN_FAST_Y': 'rs2'})), 1)
        self.assertEqual(self.problems('nothing\n', {}), [])
        self.assertEqual(self.problems('nothing\n', None), [])

    def test_a_fell_back_line_fails_whatever_the_profile(self):
        self.assertEqual(len(self.problems('[PINDIAG] tp4 y fell back reason=shape\n', {})), 1)
        self.assertEqual(len(self.problems('[PINDIAG] tp4 y fell back reason=shape\n', None)), 1)

    def test_an_audit_flag_needs_a_passing_line(self):
        env = {'QWEN_FAST_X': '1', 'QWEN_FAST_X_AUDIT': '1'}
        engaged = '[PINDIAG] tp4 x engaged\n'
        self.assertEqual(len(self.problems(engaged, env)), 1)
        self.assertEqual(len(self.problems(engaged + '[PINDIAG] tp4 x audit 3 mismatch\n', env)), 1)
        self.assertEqual(self.problems(engaged + '[PINDIAG] tp4 x audit 3 exact=True\n', env), [])

    def test_check_and_the_gate_arms_both_run_the_rule(self):
        with mock.patch.object(c2_smoke_check, 'FUSION_LEVERS', self.LEVERS), mock.patch.object(c2_smoke_check, 'FUSION_AUDITS', self.AUDITS):
            self.assertTrue(any('QWEN_FAST_X' in p for p in c2_smoke_check.lever_engagement_problems({'QWEN_FAST_X': '1'}, 'nothing\n')))
            problems, _facts = c2_smoke_check.check('', 'nothing\n', False, env={'QWEN_FAST_X': '1'})
            self.assertTrue(any('QWEN_FAST_X' in p for p in problems))

    def test_each_rule_is_called_once_by_check_and_once_by_the_gate_arms(self):
        source = (HERE / 'c2_smoke_check.py').read_text(encoding='utf-8')
        self.assertEqual(source.count('problems += fusion_problems(container_text, env)'), 1)
        self.assertEqual(source.count('problems.extend(fusion_problems(container_text, env))'), 1)

    def test_the_real_tables_are_well_formed(self):
        for flag, value, engaged, fell_back, what in c2_smoke_check.FUSION_LEVERS:
            self.assertTrue(flag.startswith('QWEN_') and value and engaged.startswith('[PINDIAG] ') and fell_back.startswith('[PINDIAG] ') and what)
        for flag, marker, what in c2_smoke_check.FUSION_AUDITS:
            self.assertTrue(flag.endswith('_AUDIT') and marker.startswith('[PINDIAG] ') and what)


class RefusalTests(unittest.TestCase):
    def refuse(self, edit, needle, manifests=None):
        manifests = copy.deepcopy(manifests or MANIFESTS)
        edit(manifests)
        with self.assertRaises(gen.ManifestError) as caught:
            out, _report = generate(manifests)
            gen.validate(out)
        self.assertIn(needle, str(caught.exception))

    def test_an_unknown_key_is_refused_by_name_and_file(self):
        self.refuse(lambda m: m['WP1.json'].update(frobnicate=[1]), "fusion-wp/WP1.json: the key 'frobnicate'")

    def test_two_spellings_of_one_key_are_refused(self):
        self.refuse(lambda m: m['WP5.json'].update(tests=['test_x']), "'tests' and 'cpu_tests'", MANIFESTS)

    def test_two_packages_cannot_name_one_lever_id_flag_or_profile(self):
        self.refuse(lambda m: m['WP2.json']['levers'][0].update(id='s1'), 'both name the lever id s1')
        self.refuse(lambda m: m['WP2.json']['levers'][0].update(flag='QWEN_FAST_TP4_SHARD_ARGMAX'), 'both name the lever flag QWEN_FAST_TP4_SHARD_ARGMAX')
        self.refuse(lambda m: m['WP7.json'].update(profiles=[{'name': 'dr-tail', 'env': {'QWEN_FAST_X': '1'}, 'reason': 'x'}]), 'both generate the profile')

    def test_a_flag_the_parent_already_names_is_refused(self):
        self.refuse(lambda m: m['WP2.json']['levers'][0].update(flag='QWEN_FAST_TP4_SDPA'), 'already names QWEN_FAST_TP4_SDPA')

    def test_a_missing_parent_is_refused(self):
        self.refuse(lambda m: m['WP2.json']['levers'][0].update(parent='no-such-profile'), 'no-such-profile is not defined')

    def test_a_file_the_manifest_names_but_the_tree_lacks_is_refused(self):
        self.refuse(lambda m: m['WP6.json']['image_files'].append('scripts/ci/not_there_tp.py'), 'scripts/ci/not_there_tp.py is not in the tree')
        self.refuse(lambda m: m['WP6.json']['tests'].append('test_not_there'), 'test module test_not_there is not in scripts/ci')
        self.refuse(lambda m: m['WP5.json']['cpu_tests'].append({'discover': 'optimisation/ttnn-op/not_there'}), 'test directory optimisation/ttnn-op/not_there')
        self.refuse(lambda m: m['WP7.json']['tp_addresses']['module_twins'].append(['a', 'not_there_tp']), 'tp_addresses names the module not_there_tp')

    def test_a_path_the_overlay_cannot_carry_is_refused(self):
        self.refuse(lambda m: m['WP6.json']['image_files'].append('docs/x.md'), 'must be a file under')
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'scripts' / 'ci').mkdir(parents=True)
            (Path(directory) / 'scripts' / 'ci' / 'tp_common.py').write_text('', encoding='utf-8')
            manifests = {'WP9.json': {'wp': 'WP9', 'image_files': ['scripts/ci/tp_common.py']}}
            out, _report = gen.generate(real_texts(), plan_of(manifests), directory)
            with self.assertRaises(gen.ManifestError) as caught:
                gen.validate(out)
            self.assertIn('never overlaid', str(caught.exception))

    def test_a_malformed_entry_is_refused(self):
        self.refuse(lambda m: m['WP1.json']['levers'][0].update(flag='lower'), 'is not an environment variable name')
        self.refuse(lambda m: m['WP1.json']['levers'][0].update(id='Bad Id'), 'lever id')
        self.refuse(lambda m: m['WP1.json']['levers'][0].pop('marker'), 'name the engaged marker')
        self.refuse(lambda m: m['WP1.json']['levers'][0].update(marker='tp4 é'), 'must be a [PINDIAG] marker')
        self.refuse(lambda m: m['WP1.json']['levers'][0].update(env={'QWEN_X': True}), 'must be a string or an integer')
        self.refuse(lambda m: m['WP1.json']['levers'][0].update(surprise=1), 'a lever has no key surprise')
        self.refuse(lambda m: m['WP7.json']['tp_addresses']['flagged_module_twins'][0].update(gate='no colon'), 'module:function')
        self.refuse(lambda m: m['WP1.json'].update(tests=['not_a_test']), 'not a test module name')

    def test_a_manifest_that_is_not_json_is_refused_by_file_name(self):
        directory = Path(tempfile.mkdtemp())
        (directory / 'WP3.json').write_text('{oops', encoding='utf-8')
        with self.assertRaises(gen.ManifestError) as caught:
            gen.read_manifests(directory)
        self.assertIn('WP3.json: not JSON', str(caught.exception))
        (directory / 'WP3.json').write_text('[1]', encoding='utf-8')
        with self.assertRaises(gen.ManifestError):
            gen.read_manifests(directory)

    def test_a_file_without_the_fence_is_refused_by_name(self):
        texts = dict(real_texts(), smoke=real_texts()['smoke'].replace(gen.BEGIN, 'gone'))
        with self.assertRaises(gen.ManifestError) as caught:
            gen.generate(texts, plan_of(MANIFESTS), ROOT)
        self.assertIn('c2_smoke_check.py: needs exactly one', str(caught.exception))

    def test_a_smoke_flag_the_static_tables_dispatch_with_other_markers_is_refused(self):
        self.refuse(lambda m: m['WP1.json']['levers'][0].update(marker='tp4 other name'), 'already dispatched by c2_smoke_check.py')


class ManifestShapeTests(unittest.TestCase):
    """The shapes a package is likely to write that the schema's own examples do not show."""

    def plan(self, **keys):
        manifest = dict(wp='WP3', **keys)
        return plan_of({'WP3.json': manifest})

    def test_a_single_smoke_object_with_a_marker_list_and_a_module(self):
        plan = self.plan(smoke={'flag': 'QWEN_FAST_TP4_X', 'audit_flag': 'QWEN_FAST_TP4_X_AUDIT', 'module': 'x_smoke.py',
                                'markers': ['[PINDIAG] tp4 x engaged site=a', '[PINDIAG] tp4 x fell back', '[PINDIAG] tp4 x audit 3 exact=True']})
        entry = plan['smoke'][0]
        self.assertEqual((entry['engaged'], entry['fell_back'], entry['audit']), ('[PINDIAG] tp4 x engaged', '[PINDIAG] tp4 x fell back', '[PINDIAG] tp4 x audit'))
        self.assertIn(('WP3', 'scripts/ci/x_smoke.py', 'the smoke module of QWEN_FAST_TP4_X'), plan['image_files'])

    def test_a_marker_list_line_that_is_none_of_the_three_kinds_is_refused(self):
        with self.assertRaises(gen.ManifestError) as caught:
            self.plan(smoke={'flag': 'QWEN_FAST_TP4_X', 'markers': ['[PINDIAG] tp4 x something']})
        self.assertIn('none of engaged, fell back or audit', str(caught.exception))

    def test_a_smoke_entry_without_a_flag_is_refused_by_name(self):
        with self.assertRaises(gen.ManifestError) as caught:
            self.plan(smoke=[{'marker': 'tp4 x'}])
        self.assertIn('names its flag', str(caught.exception))

    def test_image_files_and_tests_may_be_objects_of_name_to_reason(self):
        plan = self.plan(image_files={'scripts/ci/a_tp.py': 'the module', 'scripts/ci/b_tp.cpp': 'its kernel'}, tests={'test_a_tp': 'its test', 'test_b_tp.Cls': 'more'})
        self.assertEqual([(path, reason) for _wp, path, reason in plan['image_files']], [('scripts/ci/a_tp.py', 'the module'), ('scripts/ci/b_tp.cpp', 'its kernel')])
        self.assertEqual([module for _wp, module, _reason in plan['tests']], ['test_a_tp', 'test_b_tp.Cls'])

    def test_a_lever_module_is_an_image_file_too(self):
        plan = self.plan(levers=[{'id': 'x', 'name': 'x', 'flag': 'QWEN_FAST_TP4_X', 'marker': 'tp4 x', 'module': 'x_smoke.py'}])
        self.assertIn('scripts/ci/x_smoke.py', [path for _wp, path, _reason in plan['image_files']])

    def test_a_smoke_flag_with_a_lever_of_the_same_flag_is_dispatched_once(self):
        manifests = {'WP3.json': {'wp': 'WP3', 'levers': [{'id': 'x', 'name': 'x', 'flag': 'QWEN_FAST_TP4_X', 'audit_flag': 'QWEN_FAST_TP4_X_AUDIT', 'marker': 'tp4 x'}],
                                  'smoke': [{'flag': 'QWEN_FAST_TP4_X', 'audit_flag': 'QWEN_FAST_TP4_X_AUDIT', 'marker': 'tp4 x'}]}}
        out, report = gen.generate(real_texts(), plan_of(manifests), ROOT)
        levers = literal('\n'.join(block(out['smoke'])), 'FUSION_LEVERS')
        audits = literal('\n'.join(block(out['smoke'])), 'FUSION_AUDITS')
        self.assertEqual([row[0] for row in levers], ['QWEN_FAST_TP4_X'])
        self.assertEqual([row[0] for row in audits], ['QWEN_FAST_TP4_X_AUDIT'])
        self.assertEqual({row[3] for row in report}, {'landed', 'existing'})


class SmokeRuleModuleTests(unittest.TestCase):
    """A package's own stricter smoke rule: a host-side module with problems(env, container_text), named under smoke_rules."""

    def manifests(self):
        return {'WP1.json': dict(MANIFESTS['WP1.json'], smoke_rules=['tp4_shard_argmax_smoke'])}

    def root(self):
        root = scratch_root()
        (root / 'scripts' / 'ci' / 'tp4_shard_argmax_smoke.py').write_text('def problems(env, container_text):\n    return []\n', encoding='utf-8')
        return root

    def test_a_rule_module_lands_in_the_rules_table_once(self):
        out, report = gen.generate(real_texts(), plan_of(self.manifests()), self.root())
        self.assertEqual(literal('\n'.join(block(out['smoke'])), 'FUSION_RULES'), ('tp4_shard_argmax_smoke',))
        self.assertIn(('WP1', 'smoke_rules tp4_shard_argmax_smoke', 'c2_smoke_check.py', 'landed', 'FUSION_RULES'), report)

    def test_a_rule_module_the_tree_lacks_or_without_problems_is_refused(self):
        with self.assertRaises(gen.ManifestError) as caught:
            gen.generate(real_texts(), plan_of(self.manifests()), scratch_root())
        self.assertIn('smoke rule module tp4_shard_argmax_smoke is not in scripts/ci', str(caught.exception))
        root = self.root()
        (root / 'scripts' / 'ci' / 'tp4_shard_argmax_smoke.py').write_text('x = 1\n', encoding='utf-8')
        with self.assertRaises(gen.ManifestError) as caught:
            gen.generate(real_texts(), plan_of(self.manifests()), root)
        self.assertIn('has no problems(env, container_text)', str(caught.exception))

    def test_fusion_problems_calls_each_rule_with_the_env_and_the_log_and_reports_a_rule_that_cannot_run(self):
        module = types.ModuleType('fusion_rule_probe')
        module.problems = lambda env, text: ['probe saw %s %d' % (sorted((env or {}).items()), len(text))]
        with mock.patch.dict(sys.modules, {'fusion_rule_probe': module}), mock.patch.object(c2_smoke_check, 'FUSION_RULES', ('fusion_rule_probe', 'fusion_rule_missing')):
            found = c2_smoke_check.fusion_problems('abc', {'A': '1'})
            self.assertEqual(found[0], "probe saw [('A', '1')] 3")
            self.assertEqual(len(found), 2)
            self.assertIn('smoke rule of fusion_rule_missing could not run', found[1])
            self.assertTrue(any('probe saw' in p for p in c2_smoke_check.lever_engagement_problems({'A': '1'}, 'abc')))
            problems, _facts = c2_smoke_check.check('', 'abc', False, env={'A': '1'})
            self.assertTrue(any('probe saw' in p for p in problems))


class SampdraftCompanionTests(unittest.TestCase):
    """QWEN_FAST_TP4_SHARD_ARGMAX_FOLD2 (WP1) is strict and needs its lever; nothing changes while it is unset."""

    def test_the_companion_without_its_lever_is_refused_at_attach_and_with_it_is_fine(self):
        import tp4_sampdraft as sd

        base = {'QWEN_FAST_TP': '4'}
        sd.validate(dict(base))
        sd.validate(dict(base, **{sd.SHARD_ARGMAX: '1', sd.SHARD_ARGMAX_FOLD2: '1'}))
        sd.validate(dict(base, **{sd.SHARD_ARGMAX: '1', sd.SHARD_ARGMAX_FOLD2: '0'}))
        with self.assertRaises(ValueError) as caught:
            sd.validate(dict(base, **{sd.SHARD_ARGMAX_FOLD2: '1'}))
        self.assertIn('needs QWEN_FAST_TP4_SHARD_ARGMAX=1', str(caught.exception))
        with self.assertRaises(ValueError):
            sd.validate(dict(base, **{sd.SHARD_ARGMAX: '1', sd.SHARD_ARGMAX_FOLD2: 'yes'}))

    def test_the_companion_is_a_tp4_flag_like_the_levers(self):
        import tp4_sampdraft as sd

        with self.assertRaises(ValueError):
            sd.validate({'QWEN_FAST_TP': '2', sd.SHARD_ARGMAX_FOLD2: '1'})
        self.assertIn(sd.SHARD_ARGMAX_FOLD2, sd.COMPANIONS)
        self.assertNotIn(sd.SHARD_ARGMAX_FOLD2, sd.ALL_FLAGS, 'the census tests of the levers and audits iterate ALL_FLAGS')
        self.assertEqual(sd.LEVERS, ('QWEN_FAST_TP4_SHARD_ARGMAX', 'QWEN_FAST_TP4_DRAFT_CONV', 'QWEN_FAST_TP4_DRAFT_HEADS'), 'the lever tuple is unchanged')


class SchemaTests(unittest.TestCase):
    def test_the_schema_document_names_every_key_the_generator_places(self):
        text = (HERE / 'fusion-wp' / 'SCHEMA.md').read_text(encoding='utf-8')
        for key in gen.KEYS + gen.TP_KEYS:
            self.assertIn(key, text)
        for word in ('--write', '--report', '--check', 'QWEN_FAST_TP4_SHARD_ARGMAX'):
            self.assertIn(word, text)

    def test_the_aliases_mean_their_canonical_keys(self):
        for alias, key in gen.ALIASES.items():
            self.assertIn(key, gen.KEYS, alias)
        plan = plan_of(MANIFESTS)
        self.assertIn('scripts/ci/ccl_options_tp.py', [path for _wp, path, _reason in plan['image_files']])
        self.assertIn(('WP5', ('discover', 'optimisation/ttnn-op/ccl_sweep', 'test_*.py'), 'named by WP5'), plan['tests'])

    def test_no_tracked_text_names_a_host_or_a_path_outside_the_repo(self):
        for path in (HERE / 'make_fusion_profiles.py', HERE / 'fusion-wp' / 'SCHEMA.md'):
            text = path.read_text(encoding='utf-8')
            for word in ('/home/', '/tmp/', '192.168', '172.16', 'thatch@'):
                self.assertNotIn(word, text, path.name)


if __name__ == '__main__':
    unittest.main()
