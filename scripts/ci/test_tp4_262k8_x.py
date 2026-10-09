"""tp4/262k8-x: the experiments image. The eight-seat 262k stack with one change per arm (profiles) and the window that times them (references/tp4-262k8-x-jobs).

Merged here: tp4/262k8 (the control), tp4/samp-draft (S1 shard argmax, D2a drafter conv, D2c drafter heads), tp4/drafter-bf16, tp4/lookup, tp4/v5split (inert: no profile of
this branch sets its flag). Each timed arm is c2-packed-tp4-8x262k-best-time-gate plus or minus exactly one change, each audited twin is c2-packed-tp4-8x262k-best-audit plus the
same. These tests hold the deltas, the per-lever code facts the profiles rely on at two blocks, the image copy lists, and the pack. The templates are public: no rig, card,
address, registry or digest."""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_job as job  # noqa: E402
import c2_smoke_check  # noqa: E402
import draft_mlp_branch  # noqa: E402
import packed_verifier  # noqa: E402
import prompt_lookup  # noqa: E402
import tp4_sampdraft as sd  # noqa: E402
import tp4_shard_argmax  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FOLDER = os.path.join(HERE, 'references', 'tp4-262k8-x-jobs')
IMAGE = 'tp4-262k8-x-1'
with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as _handle:
    DOCUMENT = json.load(_handle)
PROFILES = DOCUMENT['profiles']
NAMES = sorted(PROFILES)
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|/dev/tenstorrent|home/|zot\.|@[A-Z0-9_]+@')

TIMED, AUDITED = 'c2-packed-tp4-8x262k-best-time-gate', 'c2-packed-tp4-8x262k-best-audit'
HANG = 'QWEN_FAST_PACKED_SAMPLER_IN_TRACE'
SARG, SARG_AUDIT = sd.SHARD_ARGMAX, sd.SHARD_ARGMAX_AUDIT
CONV, CONV_AUDIT, HEADS, HEADS_AUDIT = sd.DRAFT_CONV, sd.DRAFT_CONV_AUDIT, sd.DRAFT_HEADS, sd.DRAFT_HEADS_AUDIT
BF16, LOOKUP = 'QWEN_FAST_DRAFTER_BF16', 'QWEN_FAST_LOOKUP_DRAFT'
TOKENS = 'QWEN36_MAX_TOKENS_ALL_USERS'

# name -> (base, env keys removed, env keys added or changed, engine keys changed)
ARMS = {
    TIMED + '-nosamp': (TIMED, (HANG,), {}, {}),
    TIMED + '-s1': (TIMED, (), {SARG: '1'}, {}),
    TIMED + '-d2': (TIMED, (), {CONV: '1', HEADS: '1'}, {}),
    TIMED + '-dbf16': (TIMED, (), {BF16: '1', TOKENS: '1228288'}, {'num-gpu-blocks-override': 19200}),
    TIMED + '-lookup': (TIMED, (), {LOOKUP: 'n3m12'}, {}),
    TIMED + '-stack': (TIMED, (HANG,), {SARG: '1', CONV: '1', HEADS: '1'}, {}),
    'c2-packed-tp4-8x262k-best-nosamp-audit': (AUDITED, (HANG,), {}, {}),
    'c2-packed-tp4-8x262k-best-stack-audit': (AUDITED, (HANG,), {SARG: '1', SARG_AUDIT: '1', CONV: '1', CONV_AUDIT: '1', HEADS: '1', HEADS_AUDIT: '1'}, {}),
}
TIMED_ARMS = [name for name in ARMS if name.startswith(TIMED)]
LEVER_PREFIX = {'nosamp': 'TN', 's1': 'TS', 'd2': 'TD', 'stack': 'TK', 'dbf16': 'TB', 'lookup': 'TL'}
LEVER_ORDER = ('nosamp', 's1', 'd2', 'stack', 'dbf16', 'lookup')
AGENT_ACTIONS = {'agentstop', 'agentstart', 'unserve', 'platform', 'replay', 'priority', 'cardm'}
TESTS_TIMED = ['warmup', 'coding', 'concurrent8_steady', 'concurrent8_code_32k', 'concurrent8_code_128k']


def text_of(name):
    with open(os.path.join(FOLDER, name + '.env'), encoding='utf-8', newline='') as handle:
        return handle.read()


def parsed(name):
    return job.read_job(job.parse_env(text_of(name)), NAMES, root=ROOT)


def order_lines():
    with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
        return [line.split() for line in handle.read().splitlines() if line.strip() and not line.startswith('#')]


def order_text():
    with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
        return handle.read()


def tests_of(name):
    return parsed(name)['tests'].replace(' ', ',').split(',')


class ProfileTests(unittest.TestCase):
    def test_each_arm_is_its_base_plus_or_minus_exactly_its_change(self):
        for name, (base, removed, added, engine) in ARMS.items():
            with self.subTest(name):
                mine, control = PROFILES[name], PROFILES[base]
                want_env = {key: value for key, value in control['env'].items() if key not in removed}
                want_env.update(added)
                self.assertEqual(mine['env'], want_env)
                want_engine = dict(control['engine'])
                want_engine.update(engine)
                self.assertEqual(mine['engine'], want_engine)
                for key in set(mine) | set(control):
                    if key not in ('description', 'env', 'engine'):
                        self.assertEqual(mine.get(key), control.get(key), key)
                self.assertIs(mine['gate_only'], True)
                self.assertTrue(mine['description'].startswith('GATE ONLY'))
                self.assertIn('UNVERIFIED on hardware', mine['description'])
                self.assertIsNone(BANNED.search(mine['description']))
                self.assertEqual(mine['env']['QWEN_FAST_262K_EVIDENCE_WAIVER'], '1')

    def test_the_removed_flag_is_removed_not_zeroed_and_every_other_arm_keeps_it(self):
        for name in ARMS:
            self.assertEqual(HANG in PROFILES[name]['env'], not (name.endswith('-nosamp') or name.endswith('-stack') or name.endswith('-nosamp-audit') or name.endswith('-stack-audit')), name)
        self.assertEqual(PROFILES[TIMED][ 'env'][HANG], '1')

    def test_the_timed_arms_have_every_audit_off_and_the_audited_arms_every_lever_audit_on(self):
        for name in TIMED_ARMS:
            env = PROFILES[name]['env']
            for audit in ('QWEN_FAST_VERIFY_T1_AUDIT', 'QWEN_FAST_VERIFY_T2_AUDIT'):
                self.assertEqual(env[audit], '0', (name, audit))
            for audit in ('QWEN_FAST_FUSED_COMMIT_AUDIT', 'QWEN_FAST_DRAFT_SINGLES_AUDIT', 'QWEN_FAST_TP4_VGLUE_AUDIT') + tuple(sd.AUDITS):
                self.assertNotIn(audit, env, (name, audit))
        for name in ('c2-packed-tp4-8x262k-best-nosamp-audit', 'c2-packed-tp4-8x262k-best-stack-audit'):
            env = PROFILES[name]['env']
            self.assertEqual((env['QWEN_FAST_VERIFY_T1_AUDIT'], env['QWEN_FAST_VERIFY_T2_AUDIT'], env['QWEN_FAST_FUSED_COMMIT_AUDIT'],
                              env['QWEN_FAST_TP4_VGLUE_AUDIT'], env['QWEN_FAST_DRAFT_SINGLES_AUDIT']), ('1', '1', '1', '1', 'all'), name)

    def test_every_arm_serves_two_quad_blocks_with_the_request_warm_and_the_four_card_width(self):
        for name in ARMS:
            env = PROFILES[name]['env']
            self.assertEqual((env['QWEN_FAST_QUAD_DRAFT_BLOCKS'], env['QWEN_FAST_M3_BLOCKS'], env['QWEN_FAST_M3_REQUEST_WARM'], env['QWEN_FAST_TP']),
                             ('2', '2', '1', '4'), name)
            self.assertEqual(PROFILES[name]['engine']['max-num-seqs'], 8, name)
            self.assertEqual(PROFILES[name]['engine']['max-model-len'], 262144, name)
            self.assertEqual(env['QWEN_FAST_EXTENT_REPLAY'], '1', name)

    def test_the_dbf16_pool_is_the_one_405_mb_of_weights_leave_and_the_pair_the_contract_holds_equal(self):
        arm, control = PROFILES[TIMED + '-dbf16'], PROFILES[TIMED]
        blocks = arm['engine']['num-gpu-blocks-override']
        self.assertEqual(int(arm['env'][TOKENS]), (blocks - 8) * 64)
        self.assertEqual(blocks % 64, 0)
        gave_back = (control['engine']['num-gpu-blocks-override'] - blocks) * 557056
        self.assertGreaterEqual(gave_back, 405e6 * 0.9)
        self.assertLessEqual(gave_back, 405e6 * 1.3)
        self.assertGreaterEqual((blocks - 8) // 4097, 4, 'still four full 262,144-token windows')
        for other in ARMS:
            if other != TIMED + '-dbf16':
                self.assertEqual(PROFILES[other]['engine']['num-gpu-blocks-override'], PROFILES[TIMED if other.startswith(TIMED) else AUDITED]['engine']['num-gpu-blocks-override'])

    def test_the_control_and_every_older_profile_are_untouched_and_production_stays_the_default(self):
        self.assertEqual(DOCUMENT['default'], 'c2-packed-tp4')
        self.assertEqual(PROFILES['c2-packed-tp4']['env'][HANG], '1', 'production keeps the hang fix')
        for flag in sd.ALL_FLAGS + (BF16, LOOKUP, 'QWEN_FAST_GDN_SPLIT_V'):
            for name, profile in PROFILES.items():
                if flag in profile['env'] and name not in ('c2-packed-tp4-8x262k-ship-prefix', 'c2-packed-tp4-8x262k-ship-prefix-levern-traffic', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-traffic'):     # the traffic profiles carrying the W1 levers (ship/262k-prefix, and its Lever N twin)
                    self.assertTrue(profile.get('gate_only'), (name, flag))
        # the v5 split is merged and inert: no arm of this window sets it
        for name in ARMS:
            self.assertNotIn('QWEN_FAST_GDN_SPLIT_V', PROFILES[name]['env'], name)
        base = {name: body for name, body in PROFILES.items() if name not in ARMS}
        self.assertEqual(len(PROFILES) - len(base), len(ARMS))

    def test_the_stack_is_exactly_the_union_of_the_three_exact_levers(self):
        stack, control = PROFILES[TIMED + '-stack']['env'], PROFILES[TIMED]['env']
        self.assertEqual(set(stack) ^ set(control), {SARG, CONV, HEADS, HANG})
        self.assertEqual(set(PROFILES['c2-packed-tp4-8x262k-best-stack-audit']['env']) ^ set(PROFILES[AUDITED]['env']),
                         {HANG} | set(sd.ALL_FLAGS))


class LeverCodeTests(unittest.TestCase):
    """What each lever does at two 64-row blocks and beside the request-width warm, held in code (nothing here needs a card)."""

    def tearDown(self):
        tp4_shard_argmax._RESERVED.clear()

    def test_the_sampler_diet_is_the_flag_alone_and_nothing_refuses_without_it(self):
        env = {key: value for key, value in PROFILES[TIMED + '-nosamp']['env'].items()}
        self.assertFalse(packed_verifier.sampler_arm_enabled(packed_verifier.SAMPLER_IN_TRACE_FLAG, env))
        self.assertFalse(packed_verifier.sampler_arm_requested(env))
        self.assertTrue(packed_verifier.sampler_arm_requested(PROFILES[TIMED]['env']))
        # no attach check reads the hang-fix flag: it appears in packed_verifier only as the arm's own enable
        with open(os.path.join(HERE, 'packed_verifier.py'), encoding='utf-8') as handle:
            source = handle.read()
        self.assertEqual(source.count('SAMPLER_IN_TRACE_FLAG'), 3)
        self.assertIn('self.sampler_in_trace = self.shard_argmax and not self.shard_audit and sampler_arm_enabled(SAMPLER_IN_TRACE_FLAG)', source)
        # the tokens never come from the pinned sampler's second call: shard_predictions reads output[1] and output[2]
        self.assertIn('ids, values = self.output[1], self.output[2]', source)

    def test_the_request_warm_does_not_call_the_shard_sampler_so_the_diet_and_the_warm_do_not_meet(self):
        with open(os.path.join(HERE, 'request_width_warm.py'), encoding='utf-8') as handle:
            self.assertNotIn('sample_shards', handle.read())

    def test_the_shard_argmax_partials_buffer_is_shared_by_two_blocks_and_freed_by_the_last_close(self):
        freed = []

        class Operations:
            uint32, ROW_MAJOR_LAYOUT, DRAM_MEMORY_CONFIG = 'u32', 'rm', 'dram'

            def empty(self, shape, **options):
                return {'shape': shape}

            def deallocate(self, tensor):
                freed.append(tensor)

        operations, mesh = Operations(), object()
        first = tp4_shard_argmax.reserve(operations, mesh)
        second = tp4_shard_argmax.reserve(operations, mesh)
        self.assertIs(first, second, 'block 1 takes a holder, it allocates nothing')
        tp4_shard_argmax.release_reserved(operations, mesh)
        self.assertEqual(freed, [], 'block 0 closing keeps the buffer block 1 still reads')
        tp4_shard_argmax.release_reserved(operations, mesh)
        self.assertEqual(freed, [first])

    def test_the_shard_argmax_plan_fits_a_64_row_block_on_the_110_core_scan(self):
        tasks, per_tile_row, tile_rows = tp4_shard_argmax.plan(64, 1940)
        self.assertEqual(tile_rows, 2)
        self.assertEqual(len(tasks), tp4_shard_argmax.TASKS)

    def test_every_new_arm_passes_the_strict_flag_validation_at_four_cards(self):
        for name in ARMS:
            env = PROFILES[name]['env']
            sd.validate(env)
            for audit, lever in sd.AUDITS.items():
                if env.get(audit) == '1':
                    self.assertEqual(env.get(lever), '1', (name, audit))
            if LOOKUP in env:
                self.assertEqual(repr(prompt_lookup.parse_policy(env[LOOKUP])), repr(prompt_lookup.parse_policy('n3m12')))
        self.assertEqual(draft_mlp_branch.draft_projection_dtype(type('Ops', (), {'bfloat16': 'bf16', 'bfloat8_b': 'bf8'}), PROFILES[TIMED + '-dbf16']['env']), 'bf16')
        control_dtype = draft_mlp_branch.draft_projection_dtype(type('Ops', (), {'bfloat16': 'bf16', 'bfloat8_b': 'bf8'}), dict(PROFILES[TIMED]['env'], QWEN_FAST_DRAFT_BF8='1'))
        self.assertEqual(control_dtype, 'bf8', 'the image bakes QWEN_FAST_DRAFT_BF8=1: the control keeps the bfloat8 drafter')

    def test_the_smoke_check_is_armed_for_each_lever_an_arm_sets(self):
        for name in (TIMED + '-s1', TIMED + '-d2', TIMED + '-stack', 'c2-packed-tp4-8x262k-best-stack-audit'):
            problems = c2_smoke_check.sampdraft_problems('', PROFILES[name]['env'])
            self.assertTrue(problems, name)
        self.assertEqual(c2_smoke_check.sampdraft_problems('', PROFILES[TIMED]['env']), [])
        self.assertEqual(c2_smoke_check.sampdraft_problems('', PROFILES[TIMED + '-nosamp']['env']), [])
        self.assertTrue(c2_smoke_check.lookup_problems(PROFILES[TIMED + '-lookup']['env'], ''))
        self.assertEqual(c2_smoke_check.lookup_problems(PROFILES[TIMED]['env'], ''), [])

    def test_the_drafter_audits_cover_two_buckets_a_site_so_the_second_quad_block_rests_on_the_singles_audit(self):
        import tp4_draft_conv

        self.assertEqual(tp4_draft_conv.AUDIT_SCOPES, 2)
        env = PROFILES['c2-packed-tp4-8x262k-best-stack-audit']['env']
        self.assertEqual(env['QWEN_FAST_DRAFT_SINGLES_AUDIT'], 'all')


class ImageTests(unittest.TestCase):
    MODULES = tuple(sd.RUNTIME_FILES) + ('prompt_lookup.py', 'gdn_seq_block_split.py', 'gdn_seq_block_split_compute.cpp',
                                         'gdn_seq_block_split_reader.cpp', 'gdn_seq_block_split_writer.cpp')

    def read(self, *parts):
        with open(os.path.join(ROOT, *parts), encoding='utf-8') as handle:
            return handle.read()

    def test_every_served_lever_module_is_in_both_image_copy_lists_and_the_overlay(self):
        docker = self.read('docker', 'qwen-fast-serving.Dockerfile')
        workflow = self.read('.github', 'workflows', 'qwen-fast-serving-image.yml')
        overlay = self.read('docker', 'qwen-c2-overlay.txt')
        for name in self.MODULES:
            with self.subTest(name):
                self.assertIn('scripts/ci/' + name, docker)
                self.assertRegex(workflow, r'for name in [^\n]*\b' + re.escape(name) + r'\b[^\n]*; do')
                self.assertRegex(overlay, r'(?m)^scripts/ci/' + re.escape(name) + r'$')
                self.assertTrue(os.path.exists(os.path.join(HERE, name)), name)

    def test_the_new_test_module_is_on_the_cpu_allowlist(self):
        self.assertIn('test_tp4_262k8_x', self.read('.github', 'workflows', 'qwen-integration-cpu.yml'))


class OrderTests(unittest.TestCase):
    def expected(self):
        names = ['X0-status-rescan-reset', 'B0-build', 'A1-stack-audited-attach']

        def pair(lever):
            prefix = LEVER_PREFIX[lever]
            return ['%s%d-timed-%s-%s' % (prefix, number, 'A' if number % 2 else 'B', 'control' if number % 2 else lever) for number in (1, 2, 3, 4)]
        # the pairs that need no hang gate first, then the nosamp gate and pair, then the stack gate and pair
        for lever in ('s1', 'd2', 'dbf16', 'lookup'):
            names += pair(lever)
        names += ['H%d-hang-shapes-nosamp' % number for number in (1, 2, 3, 4, 5)] + pair('nosamp')
        names += ['HK%d-hang-shapes-stack' % number for number in (1, 2, 3, 4, 5)] + pair('stack')
        return names + ['Z-reset']

    def test_the_order_lists_exactly_the_templates_in_the_asked_order_with_four_columns(self):
        lines = order_lines()
        self.assertTrue(all(len(line) == 4 for line in lines))
        self.assertEqual([line[0] for line in lines], self.expected())
        files = sorted(name[:-4] for name in os.listdir(FOLDER) if name.endswith('.env'))
        self.assertEqual(files, sorted(self.expected()))
        for name, mode, image, minutes in lines:
            self.assertEqual(image, IMAGE, name)
            self.assertTrue(minutes.isdigit() and int(minutes) > 0, name)
            self.assertEqual(mode, 'stop' if name[0] in 'XBAH' else 'soft', name)

    def test_the_first_quad_job_rescans_before_it_resets_and_the_window_ends_with_a_reset_and_never_touches_the_agent(self):
        first = parsed('X0-status-rescan-reset')['actions'].split()
        self.assertEqual(first, ['status', 'rescan', 'reset'])
        self.assertEqual(order_lines()[-1][0], 'Z-reset')
        self.assertEqual(parsed('Z-reset')['actions'], 'status reset')
        for name in self.expected():
            with self.subTest(name):
                result = parsed(name)
                self.assertEqual((result['cards'], result['tag']), ('quad', IMAGE))
                self.assertFalse(AGENT_ACTIONS & set(result['actions'].split()))
                self.assertEqual(result['bake_default_profile'] or '', '')
        self.assertNotIn(IMAGE, job.PROTECTED)
        self.assertFalse(IMAGE.startswith(job.PROTECTED_PREFIXES))

    def test_the_build_is_the_second_job_opens_no_card_and_the_audited_attach_runs_the_quad_steady_mix(self):
        build = parsed('B0-build')
        self.assertEqual((build['actions'], build['profile']), ('build', 'c2-packed-tp4'))
        attach = parsed('A1-stack-audited-attach')
        self.assertEqual((attach['actions'], attach['profile']), ('reset smoke', 'c2-packed-tp4-8x262k-best-stack-audit'))
        for test in ('concurrent8_steady', 'concurrent8_code_equal'):
            self.assertIn(test, tests_of('A1-stack-audited-attach'))

    def test_the_hang_shapes_run_the_eight_seat_shapes_on_the_nosamp_and_stack_arms_five_times_each(self):
        gates = [('H%d-hang-shapes-nosamp' % number, '-nosamp') for number in (1, 2, 3, 4, 5)]
        gates += [('HK%d-hang-shapes-stack' % number, '-stack') for number in (1, 2, 3, 4, 5)]
        for name, suffix in gates:
            self.assertEqual(parsed(name)['profile'], TIMED + suffix)
            for shape in ('concurrent4_steady', 'concurrent8_steady', 'steady_resend', 'replay_concurrent8', 'concurrent8_code_equal'):
                self.assertIn(shape, tests_of(name), name)
            env = PROFILES[parsed(name)['profile']]['env']
            self.assertEqual((env['QWEN_FAST_VERIFY_T1_AUDIT'], env['QWEN_FAST_VERIFY_T2_AUDIT']), ('0', '0'))
            self.assertNotIn(HANG, env)

    def test_every_lever_has_an_abab_against_the_control_on_the_same_tests_and_concurrent8_steady_is_among_them(self):
        for lever in LEVER_ORDER:
            names = [name for name in self.expected() if name.startswith(LEVER_PREFIX[lever])]
            self.assertEqual(len(names), 4)
            profile = TIMED + '-' + lever
            self.assertEqual([parsed(name)['profile'] for name in names], [TIMED, profile, TIMED, profile], lever)
            self.assertEqual(len({parsed(name)['tests'] for name in names}), 1, lever)
            self.assertEqual(tests_of(names[0]), TESTS_TIMED, lever)

    def test_the_dependencies_name_the_audited_attach_and_the_hang_shapes_where_the_trace_changes(self):
        needs = {}
        for line in order_text().splitlines():
            match = re.match(r'# NEEDS (.+) <- (.+)$', line)
            if match:
                for name in match.group(1).split():
                    needs.setdefault(name, set()).update(match.group(2).split())
        nosamp_gate = {'A1', 'H1', 'H2', 'H3', 'H4', 'H5'}
        stack_gate = nosamp_gate | {'HK1', 'HK2', 'HK3', 'HK4', 'HK5'}
        for number in (1, 2, 3, 4):
            self.assertEqual(needs['TN%d' % number], nosamp_gate)
            self.assertEqual(needs['TK%d' % number], stack_gate)
            for lever in ('s1', 'd2', 'dbf16', 'lookup'):
                self.assertEqual(needs['%s%d' % (LEVER_PREFIX[lever], number)], {'A1'})
        for number in (1, 2, 3, 4, 5):
            self.assertEqual(needs['H%d' % number], {'A1'})
            self.assertEqual(needs['HK%d' % number], nosamp_gate)
        self.assertEqual(needs['A1'], {'B0', 'X0'})

    def test_the_independent_pairs_run_before_either_hang_gate_so_a_stall_never_costs_them(self):
        names = [line[0] for line in order_lines()]
        first_gate = names.index('H1-hang-shapes-nosamp')
        for lever in ('s1', 'd2', 'dbf16', 'lookup'):
            for name in names:
                if name.startswith(LEVER_PREFIX[lever]):
                    self.assertLess(names.index(name), first_gate, name)
        self.assertLess(names.index('H5-hang-shapes-nosamp'), names.index('TN1-timed-A-control'))
        self.assertLess(names.index('HK5-hang-shapes-stack'), names.index('TK1-timed-A-control'))

    def test_every_template_is_lf_parses_and_names_no_card_host_address_registry_or_digest(self):
        for name in self.expected():
            with self.subTest(name):
                text = text_of(name)
                self.assertNotIn('\r', text)
                self.assertIsNone(BANNED.search(text))
                self.assertIn('C2_IMAGE_TAG=' + IMAGE, text)
                self.assertNotIn('experiment/c2-serving-v' + '0', text)
        self.assertNotIn('\r', order_text())
        self.assertIsNone(BANNED.search(order_text()))

    def test_the_order_names_the_levers_left_out_the_control_and_the_stop_rules(self):
        text = order_text()
        for needle in ('V5', 'M8', 'SDPA multi', 'NO agentstop', 'NO agentstart', 'PAIRED', 'c2-packed-tp4-8x262k-best-time-gate', 'NEEDS', 'UNQUALIFIED'):
            self.assertIn(needle, text)

    def test_the_report_commands_name_scripts_that_exist(self):
        for name in self.expected():
            for script in re.findall(r'python scripts/ci/([a-z_0-9]+\.py)', text_of(name)):
                self.assertTrue(os.path.exists(os.path.join(HERE, script)), (name, script))


if __name__ == '__main__':
    unittest.main()
