"""C2_CARDS: the card set a hardware step opens (pair: cards M and A, the default; quad: every Blackhole board
present, the four-card (1, 4) mesh) - the job file's refusals, the gates' --cards option and device resolution, and
the workflow's quad steps. Stdlib only; the shell halves are test_card_set.py."""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_prefix_gate as prefix_gate  # noqa: E402
import c2_serving_gate as gate  # noqa: E402
import c2_serving_job as job  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WORKFLOW = os.path.join(ROOT, '.github', 'workflows', 'qwen-c2-serving.yml')
JOB_FILE = os.path.join(ROOT, '.github', 'c2-serving-job.env')
with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as _handle:
    PROFILES = json.load(_handle)
NAMES = sorted(PROFILES['profiles'])
TP4 = sorted(name for name, body in PROFILES['profiles'].items() if body.get('mesh_device') == 'P150x4')
PAIR = sorted(set(NAMES) - set(TP4))


def read(**values):
    base = dict(C2_IMAGE_TAG='tp4-test')
    base.update(values)
    return job.read_job(base, NAMES)


class JobTests(unittest.TestCase):
    def test_the_default_is_the_pair_and_the_fabric_default_is_1d(self):
        outputs = read()
        self.assertEqual((outputs['cards'], outputs['fabric']), ('pair', 'FABRIC_1D'))

    def test_the_tp4_profiles_are_the_profiles_that_name_the_mesh(self):
        self.assertEqual(TP4, ['c2-packed-tp4', 'c2-packed-tp4-8', 'c2-packed-tp4-8-best', 'c2-packed-tp4-8-diag-strace', 'c2-packed-tp4-8-diag-strace-nowarm', 'c2-packed-tp4-8-diag-strace-rshard', 'c2-packed-tp4-8-gate', 'c2-packed-tp4-8-time-gate', 'c2-packed-tp4-best', 'c2-packed-tp4-best-gate', 'c2-packed-tp4-best-gate-tpub', 'c2-packed-tp4-best-rshard', 'c2-packed-tp4-best-ship', 'c2-packed-tp4-best-ship-warm4', 'c2-packed-tp4-best-strace', 'c2-packed-tp4-best-strace-tpub', 'c2-packed-tp4-diag', 'c2-packed-tp4-diag-rshard', 'c2-packed-tp4-diag-sprewarm',
                               'c2-packed-tp4-diag-strace', 'c2-packed-tp4-diag-t1', 'c2-packed-tp4-diag-t1-rshard-audit', 'c2-packed-tp4-diag-t2', 'c2-packed-tp4-f12',
                               'c2-packed-tp4-f2', 'c2-packed-tp4-gate', 'c2-packed-tp4-gate-bf16', 'c2-packed-tp4-gate-draftwide', 'c2-packed-tp4-gate-fcommit',
                               'c2-packed-tp4-gate-fcommit-live', 'c2-packed-tp4-gate-fcommit-quad', 'c2-packed-tp4-gate-noslide',
                               'c2-packed-tp4-gate-pairs', 'c2-packed-tp4-gate-pairslice', 'c2-packed-tp4-gate-quad', 'c2-packed-tp4-gate-ring', 'c2-packed-tp4-gate-rshard-audit', 'c2-packed-tp4-gate-vglue',
                               'c2-packed-tp4-lanes-gate', 'c2-packed-tp4-lanes-time-gate', 'c2-packed-tp4-solo-gate',
                               'c2-packed-tp4-solo-time-gate', 'c2-packed-tp4-speed', 'c2-packed-tp4-speed-fcommit',
                               'c2-packed-tp4-speed-fcommit-oop', 'c2-packed-tp4-speed-fcommit-quad', 'c2-packed-tp4-speed-fix',
                               'c2-packed-tp4-speed-noslide', 'c2-packed-tp4-speed-pairs', 'c2-packed-tp4-speed-quad',
                               'c2-packed-tp4-speed-rshard', 'c2-packed-tp4-speed-sprewarm', 'c2-packed-tp4-speed-strace', 'c2-packed-tp4-speed-strace-dispatchdiag', 'c2-packed-tp4-speed-strace-draftwide', 'c2-packed-tp4-speed-strace-fcommit', 'c2-packed-tp4-speed-strace-pairslice', 'c2-packed-tp4-speed-strace-ring', 'c2-packed-tp4-speed-vglue',
                               'c2-packed-tp4-speed-vglue-c1a', 'c2-packed-tp4-speed-vglue-v1', 'c2-packed-tp4-speed-vglue-v2',
                               'c2-packed-tp4-speed-vglue-v3a', 'c2-packed-tp4-speed-vglue-v4a', 'c2-packed-tp4-speed-warm4', 'c2-packed-tp4-time-gate', 'c2-packed-tp4-warm4-control', 'c2-packed-tp4-warm4-diag', 'c2-packed-tp4-warm4-diag-oldtail', 'c2-packed-tp4-warm4-even-diag', 'c2-packed-tp4-warm4-gate', 'general-prefix-tp4',
                               'general-prefix-tp4-131k', 'general-tp4', 'general-tp4-131k',
                               'general-tp4-bench', 'general-tp4-mmrs', 'general-tp4-ring-mmrs'])
        self.assertIn('general', PAIR)

    def test_quad_takes_a_tp4_profile_and_refuses_a_pair_profile(self):
        outputs = read(C2_CARDS='quad', C2_ACTIONS='reset gate', C2_PROFILE='general-tp4')
        self.assertEqual(outputs['cards'], 'quad')
        with self.assertRaisesRegex(job.JobError, 'C2_PROFILE general opens the \\(1, 2\\) pair, but C2_CARDS=quad'):
            read(C2_CARDS='quad', C2_ACTIONS='gate', C2_PROFILE='general')

    def test_the_pair_refuses_a_tp4_profile_on_the_steps_that_serve_it(self):
        with self.assertRaisesRegex(job.JobError, 'general-tp4 opens the four-card'):
            read(C2_ACTIONS='gate', C2_PROFILE='general-tp4')
        # a profile the job's steps never serve is not judged: status with a TP4 default profile is only a status
        self.assertEqual(read(C2_ACTIONS='status', C2_PROFILE='general-tp4')['cards'], 'pair')

    def test_the_prefix_action_checks_both_its_profiles(self):
        good = read(C2_CARDS='quad', C2_ACTIONS='prefix', C2_PREFIX_PROFILE='general-prefix-tp4',
                    C2_PREFIX_BASELINE='general-tp4')
        self.assertEqual(good['prefix_profile'], 'general-prefix-tp4')
        with self.assertRaisesRegex(job.JobError, 'C2_PREFIX_BASELINE general opens the'):
            read(C2_CARDS='quad', C2_ACTIONS='prefix', C2_PREFIX_PROFILE='general-prefix-tp4')
        with self.assertRaisesRegex(job.JobError, 'C2_PREFIX_PROFILE general-prefix-tp4 opens the four-card'):
            read(C2_ACTIONS='prefix', C2_PREFIX_PROFILE='general-prefix-tp4', C2_PREFIX_BASELINE='general-tp4')

    def test_quad_refuses_the_pair_shaped_steps(self):
        for action in ('cardm', 'priority'):
            with self.assertRaisesRegex(job.JobError, 'pair-shaped'):
                read(C2_CARDS='quad', C2_ACTIONS=action)

    def test_quad_takes_the_replay_with_a_four_card_profile_only(self):
        """serving/tp4-s2 (E4): the agent's serving sequence on all four boards needs C2_REPLAY_PROFILE, a P150x4 profile."""
        self.assertIn('replay', job.QUAD_ACTIONS)
        outputs = read(C2_CARDS='quad', C2_ACTIONS='reset replay', C2_REPLAY_PROFILE='c2-packed-tp4',
                       C2_PLATFORM_IMAGE='thatch-serving-tt:abcdef123456')
        self.assertEqual((outputs['cards'], outputs['replay_profile']), ('quad', 'c2-packed-tp4'))
        with self.assertRaisesRegex(job.JobError, 'needs C2_REPLAY_PROFILE'):
            read(C2_CARDS='quad', C2_ACTIONS='reset replay')
        with self.assertRaisesRegex(job.JobError, 'C2_REPLAY_PROFILE c2-packed opens the \\(1, 2\\) pair, but C2_CARDS=quad'):
            read(C2_CARDS='quad', C2_ACTIONS='replay', C2_REPLAY_PROFILE='c2-packed')
        with self.assertRaisesRegex(job.JobError, 'C2_REPLAY_PROFILE c2-packed-tp4 opens the four-card'):
            read(C2_ACTIONS='replay', C2_REPLAY_PROFILE='c2-packed-tp4')
        # the pair's replay stays as it was: the profile is optional
        self.assertEqual(read(C2_ACTIONS='replay')['cards'], 'pair')

    def test_fabric_needs_quad(self):
        with self.assertRaisesRegex(job.JobError, 'fabric probe needs C2_CARDS=quad'):
            read(C2_ACTIONS='fabric')
        with self.assertRaisesRegex(job.JobError, 'C2_FABRIC needs C2_CARDS=quad'):
            read(C2_FABRIC='FABRIC_1D_RING')
        self.assertEqual(read(C2_CARDS='quad', C2_ACTIONS='fabric', C2_FABRIC='FABRIC_1D_RING')['fabric'],
                         'FABRIC_1D_RING')
        with self.assertRaisesRegex(job.JobError, 'C2_FABRIC must be one of'):
            read(C2_CARDS='quad', C2_FABRIC='FABRIC_2D')

    def test_the_fabric_probe_choice_is_the_reduction_order_spike_or_the_fabric_probe(self):
        quad = dict(C2_CARDS='quad', C2_ACTIONS='reset fabric', C2_PROFILE='general-tp4')
        self.assertEqual(read(**quad)['fabric_probe'], 'fabric')
        self.assertEqual(read(C2_FABRIC_PROBE='rs-tile', **quad)['fabric_probe'], 'rs-tile')
        with self.assertRaisesRegex(job.JobError, 'C2_FABRIC_PROBE must be one of'):
            read(C2_FABRIC_PROBE='ring', **quad)
        with self.assertRaisesRegex(job.JobError, 'no fabric'):
            read(C2_CARDS='quad', C2_ACTIONS='status', C2_FABRIC_PROBE='rs-tile')

    def test_the_fabric_probe_runs_alone(self):
        # the probe closes its mesh, and a second open in one job is what the ethernet-core teardown wedge punishes
        for beside in ('smoke', 'gate', 'prefix'):
            with self.assertRaisesRegex(job.JobError, 'fabric with %s' % beside):
                read(C2_CARDS='quad', C2_ACTIONS='reset fabric ' + beside, C2_PROFILE='general-tp4',
                     C2_PREFIX_PROFILE='general-prefix-tp4', C2_PREFIX_BASELINE='general-tp4')
        self.assertEqual(read(C2_CARDS='quad', C2_ACTIONS='status reset fabric', C2_PROFILE='general-tp4')['cards'],
                         'quad')

    def test_the_two_link_pair_profile_is_a_pair_profile_the_quad_refuses(self):
        self.assertEqual(PROFILES['profiles']['general-2link']['mesh_device'], 'P300')
        self.assertIn('general-2link', PAIR)
        self.assertEqual(read(C2_ACTIONS='smoke', C2_PROFILE='general-2link')['cards'], 'pair')
        with self.assertRaisesRegex(job.JobError, 'opens the'):
            read(C2_CARDS='quad', C2_ACTIONS='smoke', C2_PROFILE='general-2link')
        self.assertIsNone(gate.cards_problem('pair', PROFILES, ['general-2link', 'general']))
        self.assertIn('opens the (1, 2) pair', gate.cards_problem('quad', PROFILES, ['general-2link']))

    def test_the_job_templates_run_the_pair_jobs_without_the_pair_reset(self):
        # a pair-only reset leaves M's and A's links to B and C untrained on the full-mesh cabling: the pair jobs
        # follow a four-card reset job instead
        folder = os.path.join(HERE, 'references', 'tp4-jobs')
        for name in ('J2r-tp2-reference.env', 'J3r-tp2-bench.env'):
            with open(os.path.join(folder, name), encoding='utf-8') as handle:
                values = dict(line.split('=', 1) for line in handle.read().splitlines()
                              if line and not line.startswith('#'))
            self.assertEqual(values['C2_PROFILE'], 'general-2link', name)
            self.assertNotIn('reset', values['C2_ACTIONS'].split(), name)
            self.assertNotIn('C2_CARDS', values, name)
        with open(os.path.join(folder, 'Jr-quad-reset.env'), encoding='utf-8') as handle:
            text = handle.read()
        self.assertIn('C2_CARDS=quad', text)
        self.assertIn('C2_ACTIONS=reset', text)

    def test_an_unknown_card_set_is_refused(self):
        with self.assertRaisesRegex(job.JobError, 'C2_CARDS must be one of pair, quad'):
            read(C2_CARDS='trio')

    def test_the_fabric_configs_are_the_probes(self):
        import tp4_mesh
        self.assertEqual(job.FABRIC_CONFIGS, tp4_mesh.FABRIC_CONFIGS)
        self.assertEqual(job.TP4_MESH_DEVICE, tp4_mesh.MESH_DEVICE)


class GateCardTests(unittest.TestCase):
    BOARDS = ['blackhole-AAAA', 'blackhole-BBBB', 'blackhole-CCCC', 'blackhole-DDDD']

    def card_set(self, boards, expect=4, nodes=None):
        nodes = nodes or dict((board, '/dev/tenstorrent/%d' % n) for n, board in enumerate(boards))
        root = '/dev/tenstorrent/by-id'
        return gate.card_set(root, expect, listdir=lambda _: list(boards) + ['other-thing'],
                             realpath=lambda path: nodes.get(os.path.basename(path), path),
                             is_device=lambda path: True)

    def test_every_present_blackhole_board_is_the_set_in_board_id_order(self):
        shuffled = [self.BOARDS[2], self.BOARDS[0], self.BOARDS[3], self.BOARDS[1]]
        nodes = dict(zip(self.BOARDS, ['/dev/tenstorrent/3', '/dev/tenstorrent/0', '/dev/tenstorrent/2',
                                       '/dev/tenstorrent/1']))
        self.assertEqual(self.card_set(shuffled, nodes=nodes), ['/dev/tenstorrent/3', '/dev/tenstorrent/0',
                                                                '/dev/tenstorrent/2', '/dev/tenstorrent/1'])

    def test_three_or_five_boards_are_refused_never_guessed(self):
        for boards in (self.BOARDS[:3], self.BOARDS + ['blackhole-EEEE']):
            with self.assertRaisesRegex(RuntimeError, 'not 4'):
                self.card_set(boards)

    def test_a_board_without_a_node_or_two_on_one_node_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, 'no device node'):
            gate.card_set('/dev/tenstorrent/by-id', 4, listdir=lambda _: self.BOARDS, realpath=lambda path: path,
                          is_device=lambda path: True)
        nodes = dict((board, '/dev/tenstorrent/0') for board in self.BOARDS)
        with self.assertRaisesRegex(RuntimeError, 'two boards resolve'):
            self.card_set(self.BOARDS, nodes=nodes)

    def test_devices_for_refuses_an_unknown_set(self):
        with self.assertRaises(ValueError):
            gate.devices_for('trio')

    def test_cards_problem_reads_the_profiles_mesh(self):
        self.assertIsNone(gate.cards_problem('pair', PROFILES, ('general', 'none', None, 'general-prefix')))
        self.assertIsNone(gate.cards_problem('quad', PROFILES, ('general-tp4', 'general-prefix-tp4')))
        self.assertIn('C2_CARDS=quad gives all four cards', gate.cards_problem('quad', PROFILES, ('general',)))
        self.assertIn('the four-card (1, 4) mesh, but C2_CARDS=pair', gate.cards_problem('pair', PROFILES,
                                                                                          ('general-tp4',)))
        self.assertIsNone(gate.cards_problem('quad', PROFILES, ('not-a-profile',)))

    def test_both_gates_refuse_before_any_container_and_dry_run_names_four_cards(self):
        lines = []
        status = gate.main(['--image', 'img', '--profile', 'general', '--plan', 'bringup', '--results', 'r',
                            '--cards', 'quad', '--dry-run', '--profiles', os.path.join(HERE, 'qwen_c2_profiles.json')],
                           log=lines.append)
        self.assertEqual(status, 2)
        self.assertTrue(any('refused: profile general opens the (1, 2) pair' in line for line in lines), lines)
        lines = []
        status = prefix_gate.main(['--image', 'img', '--profile', 'general-prefix', '--baseline', 'general',
                                   '--plan', 'bringup', '--results', 'r', '--cards', 'quad', '--dry-run',
                                   '--profiles', os.path.join(HERE, 'qwen_c2_profiles.json')], log=lines.append)
        self.assertEqual(status, 2)
        self.assertTrue(any('refused:' in line for line in lines), lines)

    def test_a_dry_run_on_quad_mounts_four_devices_and_the_pair_still_two(self):
        for cards, count in (('quad', 4), ('pair', 2)):
            lines = []
            profile = 'general-tp4' if cards == 'quad' else 'general'
            status = gate.main(['--image', 'img', '--profile', profile, '--plan', 'bringup', '--results', 'r',
                                '--cards', cards, '--dry-run', '--profiles',
                                os.path.join(HERE, 'qwen_c2_profiles.json')], log=lines.append)
            self.assertEqual(status, 0, lines)
            arms = [json.loads(line) for line in lines if line.startswith('{"arm"')]
            self.assertTrue(arms, lines)
            for arm in arms:
                self.assertEqual(arm['docker'].count('--device'), count)
                self.assertIn('QWEN_C2_PROFILE=%s' % profile, arm['docker'])


def step(name):
    with open(WORKFLOW, encoding='utf-8') as handle:
        text = handle.read()
    start = text.index('      - name: ' + name)
    end = text.find('\n      - name: ', start + 1)
    return text[start:end if end != -1 else len(text)]


class WorkflowTests(unittest.TestCase):
    def test_the_pair_reset_steps_aside_for_quad_and_the_quad_reset_takes_over(self):
        self.assertIn("steps.job.outputs.cards != 'quad'", step('Reset cards M and A'))
        quad = step('Reset all four cards')
        self.assertIn("steps.job.outputs.cards == 'quad'", quad)

    def test_the_quad_reset_is_one_tt_smi_call_over_the_resolved_set_then_the_heal(self):
        quad = step('Reset all four cards')
        self.assertEqual(len(re.findall(r'"\$smi" -r ', quad)), 1)
        self.assertIn('-r "${CARD_SET_NODES[@]}"', quad)
        self.assertLess(quad.index('card_set_resolve 60'), quad.index('card_set_unheld'))
        self.assertLess(quad.index('card_set_unheld'), quad.index('"$smi" -r'))
        self.assertLess(quad.index('"$smi" -r'), quad.index('card_set_heal'))
        self.assertLess(quad.index('card_set_heal'), quad.index('card_set_resolve 180'))
        self.assertNotIn('blackhole-', quad)
        self.assertNotIn('/dev/tenstorrent/', quad)
        # the heal's last-start second and the reset's timeout both fit the step
        minutes = int(re.search(r'timeout-minutes: ([0-9]+)', quad).group(1))
        self.assertGreater(minutes * 60, 600 + 20 + 1000)

    def test_the_fabric_probe_opens_all_four_cards_once_under_the_ring_descriptor(self):
        probe = step('Four-card fabric probe (all four cards, inside the image)')
        self.assertIn("contains(steps.job.outputs.actions, 'fabric')", probe)
        self.assertEqual(probe.count('docker run'), 1, 'one device-opening container per step')
        self.assertIn('"${devices[@]}"', probe)
        self.assertIn('-e MESH_DEVICE=P150x4', probe)
        self.assertIn('qwen_p150x4_ring_mesh_graph_descriptor.textproto', probe)
        self.assertIn('--fabric "$FABRIC"', probe)
        # the image bakes QWEN_C2_SERVING=1, whose boot would replace the ring descriptor with the pair's before
        # ttnn loads: the probe runs with the contract off, and the ring descriptor is the -e that reaches python3
        self.assertIn('-e QWEN_C2_SERVING=0', probe)
        self.assertLess(probe.index('-e QWEN_C2_SERVING=0'), probe.index('--entrypoint python3'))
        self.assertLess(probe.index('-e TT_MESH_GRAPH_DESC_PATH='), probe.index('--entrypoint python3'))
        # the image's TT_METAL_CACHE and TT_CACHE_PATH sit under /experiment-cache, a link to /models/.qwen-c2 that
        # dangles when /models is not mounted: the fabric router JIT then fails with "cannot create directories:
        # File exists" (run 36669205193). The probe compiles into its own tmpfs instead.
        self.assertIn('-e TT_METAL_CACHE=/root/.cache/tt-metal-cache', probe)
        self.assertIn('--tmpfs /root/.cache/tt-metal-cache:', probe)
        self.assertNotIn('/experiment-cache', probe)
        self.assertIn('-v "$PWD:/c2:ro"', probe)
        self.assertNotIn('blackhole-', probe)
        # C2_FABRIC_PROBE picks the script and the report; both are this checkout's, mounted read-only
        self.assertIn('PROBE: ${{ steps.job.outputs.fabric_probe }}', probe)
        self.assertIn('fabric) script=tp4_fabric_probe.py; report=fabric-probe.json', probe)
        self.assertIn('rs-tile) script=tp4_rs_tile_spike.py; report=rs-tile-spike.json', probe)
        self.assertIn('-B "/c2/scripts/ci/$script"', probe)
        self.assertIn('TP4_RS_TILE', probe)
        for script in ('tp4_fabric_probe.py', 'tp4_rs_tile_spike.py', 'tp4_mesh.py'):
            self.assertTrue(os.path.isfile(os.path.join(HERE, script)), script)
        self.assertLess(int(re.search(r'timeout-minutes: ([0-9]+)', probe).group(1)) * 60, 600 * 60)

    def test_the_quad_status_prints_the_bring_up_preconditions(self):
        status = step('Status of the four-card set')
        for needle in ('hugepages-1048576kB', 'free 1 GiB hugepages', 'dmesg', '-121', 'fw_bundle',
                       'not readable and writable'):
            self.assertIn(needle, status)
        self.assertNotIn('blackhole-', status)

    def test_the_quad_smoke_maps_the_resolved_set_and_the_pair_smoke_steps_aside(self):
        self.assertIn("steps.job.outputs.cards != 'quad'", step('Smoke on cards M+A'))
        quad = step('Smoke on the four-card set')
        self.assertIn("steps.job.outputs.cards == 'quad'", quad)
        self.assertEqual(quad.count('docker run'), 1)
        self.assertIn('"${devices[@]}"', quad)
        self.assertNotIn('blackhole-', quad)
        self.assertIn('AGREEMENT_OUT="$results/agreement-$PROFILE.json"', quad)
        self.assertLess(quad.index('card_set_unheld'), quad.index('docker run'))

    def test_gate_and_prefix_pass_the_card_set_and_check_all_four_holders(self):
        for name, flag in (('Run the gate in the agent\'s container shape', '--cards "$CARDS"'),
                           ('Prefix-reuse gates in the agent\'s container shape', '--cards "$CARDS"')):
            text = step(name)
            self.assertIn('CARDS: ${{ steps.job.outputs.cards }}', text)
            self.assertIn(flag, text)
            self.assertIn('if [ "$CARDS" = quad ]; then', text)
            self.assertIn('card_set_unheld', text)

    def test_one_runner_one_group_covers_the_four_card_job(self):
        with open(WORKFLOW, encoding='utf-8') as handle:
            text = handle.read()
        self.assertEqual(text.count('runs-on:'), 1)
        self.assertIn('runs-on: [self-hosted, Linux, X64, thatch-qwen-p150a-pair]', text)
        self.assertIn('group: qwen-two-p150a-exclusive', text)

    def test_the_job_file_documents_the_card_set(self):
        with open(JOB_FILE, encoding='utf-8') as handle:
            text = handle.read()
        self.assertIn('C2_CARDS', text)


JOBS = os.path.join(HERE, 'references', 'tp4-jobs')


class TemplateTests(unittest.TestCase):
    """The committed TP4 job templates parse, agree with their card set, and name nothing of the rig."""

    def templates(self):
        names = sorted(name for name in os.listdir(JOBS) if name.endswith('.env'))
        self.assertGreaterEqual(len(names), 8)
        return names

    def parsed(self, name):
        with open(os.path.join(JOBS, name), encoding='utf-8') as handle:
            text = handle.read()
        return text, job.read_job(job.parse_env(text), NAMES)

    def test_every_template_parses(self):
        for name in self.templates():
            text, outputs = self.parsed(name)
            self.assertIn(outputs['cards'], job.CARD_SETS, name)

    def test_the_jobs_are_the_plans_j0_to_j4(self):
        names = self.templates()
        for prefix in ('J0-', 'J0b-', 'J1-', 'J2-', 'J2r-', 'J3-', 'J3r-', 'J4-'):
            self.assertTrue(any(name.startswith(prefix) for name in names), prefix)

    def test_the_reference_jobs_run_on_the_pair_and_the_tp4_jobs_on_the_quad(self):
        for name in self.templates():
            _, outputs = self.parsed(name)
            if name.startswith(('J2r', 'J3r')):
                self.assertEqual((outputs['cards'], outputs['profile']), ('pair', 'general-2link'), name)
            elif name.startswith(('J0', 'J2-', 'J2m', 'J3-', 'J4', 'Jr-')):
                self.assertEqual(outputs['cards'], 'quad', name)
                self.assertIn(outputs['profile'], TP4, name)

    def test_the_bench_shapes_fit_their_profiles(self):
        for name, seats, context in (('J3-tp4-bench.env', 8, 131072), ('J3r-tp2-bench.env', 4, 65536)):
            _, outputs = self.parsed(name)
            for shape in outputs['bench_shapes'].split(','):
                streams, prompt = (int(part) for part in shape.split('x'))
                self.assertLessEqual(streams, seats, shape)
                self.assertLess(prompt + 256, context + 1, shape)

    def test_bad_bench_shapes_are_refused(self):
        with self.assertRaisesRegex(job.JobError, 'C2_BENCH_SHAPES'):
            read(C2_BENCH_SHAPES='4 x 4096')
        self.assertEqual(read(C2_BENCH_SHAPES='4x4096,1x131072')['bench_shapes'], '4x4096,1x131072')

    def test_the_templates_name_no_card_host_or_address(self):
        banned = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|'
                            r'/dev/tenstorrent|home/')
        for name in self.templates():
            text, _ = self.parsed(name)
            self.assertIsNone(banned.search(text), name)


if __name__ == '__main__':
    unittest.main()
