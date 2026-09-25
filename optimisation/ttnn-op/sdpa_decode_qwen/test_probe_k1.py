"""CPU checks for the S0.K1 probe (probe_k1_card_b.py, run_probe_k1.sh); no device, no ttnn.

  - the pure helpers: busiest-core chunk counts, the slopes (they reproduce the design's card-M figures:
    19.13 / 21.32 us per chunk and the 112.3 us intercept for G8 share, 15.05 / 15.39 / 71.2 for G4 B1
    tail), section 3's decision table on card-M anchors (A <= 625 us, C >= 759 us) and at every edge, the
    K1a-alone rules, the spot check and the verdict line;
  - the contract: the proxy's per-core work is the head-sliced G8 call's (the design's slice rule gives 2
    tiles; B, cores and the CB bytes match K1's 796,736), the served shape and flags are every current
    gate arm's (K64g, 8-row groups, tail,share), the runner's default image is the gate's, and the served
    kernel and binary hashes agree across the probe, the runner, make_qwen_kernels and stage3/;
  - the runner (needs bash): the dry run launches on card B by board id with exactly the arm's graft
    mounts, the arm's scratch setting and a fresh kernel cache; the watcher pass; the pre-launch graft
    checks refuse a wrong binary, a wrong kernel, a manifest that does not verify and a missing directory;
  - a full dry run of the device flow against a fake ttnn with a DRAM allocator (freed addresses are
    reused, bytes stay stale), the graft's semantics, a model clock and the factory's log lines: scenario A
    end to end on both bases, and the controls are live - rows a kernel leaves unwritten show through the
    poison (NO-DECISION), a qualified shape that differs fails, the spot check flags the proxy, a graft whose
    factory never logs (mounted, not executed) fails, and so do a wrong binary, a wrong kernel and a missing
    scratch setting.

    py -3.11 -B -m unittest test_probe_k1      (from this directory)
"""

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CI = ROOT / 'scripts' / 'ci'
for path in (str(HERE), str(CI)):
    if path not in sys.path:
        sys.path.insert(0, path)

import probe_k1_card_b as probe  # noqa: E402
import test_sdpa_decode_qwen_card_m as card  # noqa: E402

RUNNER = HERE / 'run_probe_k1.sh'
ARM = CI / 'lever_n_m3native_run_arm.sh'
GATE = ROOT / '.github' / 'workflows' / 'qwen-lever-n-m3native-gate.yml'
STAGE3 = HERE / 'stage3'
CARD_B = 'blackhole-F36F768B9A5CAFA0'
CARD_M = 'blackhole-CEF5729692C19E6D'
NL = chr(10)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return path.read_text(encoding='utf-8')


# Card M's figures (the design's section 1 tables), per busiest-core chunk count.
G8_SHARE = {2304: 151.2, 33024: 304.2, 131328: 815.8}
G4B1_TAIL = {2304: 89.3, 33024: 209.7, 131328: 579.0}


class HelperTests(unittest.TestCase):
    def test_busiest_core_chunks_at_the_four_capacities(self):
        self.assertEqual([probe.busiest_chunks(c) for c in probe.CAPACITIES], [1, 9, 17, 33])
        self.assertEqual(probe.busiest_chunks(131328, cores_per_head=1), 513)
        with self.assertRaises(ValueError):
            probe.busiest_chunks(1000)

    def test_the_slopes_reproduce_the_designs_card_m_figures(self):
        g8 = probe.slopes(G8_SHARE)
        self.assertAlmostEqual(g8['slope_1_9'], 19.13, places=2)
        self.assertAlmostEqual(g8['slope_9_33'], 21.32, places=2)
        self.assertAlmostEqual(g8['intercept_9_33'], 112.3, places=1)
        self.assertAlmostEqual(g8['rise'], 21.3167 / 19.125 - 1, places=3)      # the 11% rise
        self.assertGreater(g8['rise'], 0.10)
        g4 = probe.slopes(G4B1_TAIL)
        self.assertAlmostEqual(g4['slope_1_9'], 15.05, places=2)
        self.assertAlmostEqual(g4['slope_9_33'], 15.39, places=2)
        self.assertAlmostEqual(g4['intercept_9_33'], 71.2, places=1)
        self.assertLess(abs(g4['rise']), 0.03)
        # With two multi-chunk points the fit is the two-point line.
        self.assertAlmostEqual(g8['fit_slope'], g8['slope_9_33'])
        self.assertEqual(g8['fit_points'], 2)
        self.assertAlmostEqual(g8['fit_max_residual'], 0.0)

    def test_the_fit_uses_the_multi_chunk_points_and_reads_curvature(self):
        line = {capacity: 100.0 + 20.0 * probe.busiest_chunks(capacity) for capacity in probe.CAPACITIES}
        fit = probe.slopes(line)
        self.assertEqual(fit['points'], [[1, 120.0], [9, 280.0], [17, 440.0], [33, 760.0]])
        self.assertAlmostEqual(fit['fit_slope'], 20.0)
        self.assertAlmostEqual(fit['fit_intercept'], 100.0)
        self.assertAlmostEqual(fit['rise'], 0.0)
        self.assertAlmostEqual(fit['curvature'], 0.0)
        self.assertEqual(fit['fit_points'], 3)
        bent = dict(line)
        bent[131328] = 800.0
        self.assertGreater(probe.slopes(bent)['curvature'], 0.1)
        self.assertGreater(probe.slopes(bent)['fit_max_residual'], 0.0)
        self.assertNotIn('fit_slope', probe.slopes({2304: 100.0, 33024: 200.0}))    # one multi-chunk point

    def test_the_slope_lines(self):
        report = dict(slopes={probe.SERVED: {'eager': probe.slopes(G8_SHARE), 'trace': None},
                              probe.PROXY: {'eager': probe.slopes({2304: 100.0})}},
                      card_b_over_card_m={probe.SERVED: {131328: 1.0, 2304: 0.9}})
        lines = probe.slope_lines(report)
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith('slope G8B2_tail_share eager fit=21.32us/chunk intercept=112.'), lines[0])
        self.assertIn(' residual=0.0% 1->9=19.1', lines[0])                  # 19.125 us per chunk
        self.assertTrue(lines[0].endswith(' rise=+11.5% cardB/cardM=2304:0.900,131328:1.000'), lines[0])

    def times(self, proxy, compute=579.0, served=815.8, dram=900.0, capacity=131328):
        return {probe.PROXY: {33024: 200.0, capacity: proxy}, probe.COMPUTE: {33024: 209.7, capacity: compute},
                probe.SERVED: {33024: 304.2, capacity: served}, probe.DRAM: {capacity: dram}}

    def fits(self, proxy_slope, compute_slope=15.39, served_slope=21.32):
        return {probe.PROXY: dict(fit_slope=proxy_slope, rise=0.0), probe.COMPUTE: dict(fit_slope=compute_slope),
                probe.SERVED: dict(fit_slope=served_slope, rise=0.11)}

    def test_the_thresholds_on_card_m_anchors_are_the_designs(self):
        """Section 3: 'card M: <~ 625 us' for A and '>~ 760 us' for C."""
        decision = probe.decide(self.times(600.0), self.fits(15.5))
        self.assertAlmostEqual(decision['a_limit_us'], 625.3, places=1)
        self.assertAlmostEqual(decision['c_limit_us'], 758.7, places=1)

    def test_scenario_a_needs_the_time_and_the_slope(self):
        a = probe.decide(self.times(610.0), self.fits(15.9))
        self.assertEqual((a['verdict'], a['scenario'], a['plan'], a['enable']), ('GO', 'A', 'K1a', '0x4'))
        self.assertAlmostEqual(a['saving_us_per_call'], 205.8, places=1)
        self.assertAlmostEqual(a['saving_ms_per_replay_card'], 205.8 * 64 / 1000, places=3)
        self.assertAlmostEqual(a['saving_ms_per_replay_trace'], 205.8 * 64 / 1000 * 761 / 815.8, places=3)
        self.assertEqual(a['k1a_alone'], 'meets-acceptance')
        # In time but the slope is 6% steeper: not A. Neither is it C, so B.
        b = probe.decide(self.times(610.0), self.fits(15.39 * 1.06))
        self.assertEqual((b['scenario'], b['plan'], b['enable']), ('B', 'K1a+K1b', '0x4,0x8'))
        # The slope matches but the time is 1.09 x: B.
        self.assertEqual(probe.decide(self.times(579.0 * 1.09), self.fits(15.39))['scenario'], 'B')
        self.assertEqual(probe.decide(self.times(579.0 * 1.08), self.fits(15.39 * 0.96))['scenario'], 'A')

    def test_scenarios_b_and_c(self):
        b = probe.decide(self.times(700.0), self.fits(18.5))
        self.assertEqual((b['verdict'], b['scenario']), ('GO', 'B'))
        self.assertEqual(b['k1a_alone'], 'between')
        c = probe.decide(self.times(0.93 * 815.8), self.fits(21.0))
        self.assertEqual((c['verdict'], c['scenario'], c['plan'], c['enable']), ('GO', 'C', 'K1b-led', '0x4,0x8'))
        self.assertIn('0xB', c['note'])
        self.assertEqual(c['k1a_alone'], 'stop-rule')
        self.assertEqual(probe.k1a_alone(0.78, 1.0), 'meets-acceptance')
        self.assertEqual(probe.k1a_alone(0.90, 1.0), 'between')
        self.assertEqual(probe.k1a_alone(0.901, 1.0), 'stop-rule')

    def test_no_decision_on_bad_controls_or_missing_data(self):
        overlap = probe.decide(self.times(650.0, compute=700.0, served=800.0), self.fits(15.5))
        self.assertEqual(overlap['verdict'], 'NO-DECISION')
        self.assertIn('do not separate A from C', overlap['reasons'][0])
        inverted = probe.decide(self.times(650.0, compute=900.0, served=800.0), self.fits(15.5))
        self.assertIn('controls inverted', inverted['reasons'][0])
        missing = probe.decide({probe.PROXY: {33024: 1.0}}, {})
        self.assertIn('not measured at 131328 keys', missing['reasons'][0])
        noslope = probe.decide(self.times(610.0), {probe.PROXY: {}, probe.COMPUTE: {}})
        self.assertIn('no fit slope', noslope['reasons'][0])
        invalid = probe.decide(self.times(610.0), self.fits(15.5), ['measurement invalid: 1 failures'])
        self.assertEqual((invalid['verdict'], invalid['scenario']), ('NO-DECISION', None))
        self.assertAlmostEqual(invalid['proxy_us'], 610.0)          # the numbers are still reported

    def test_the_spot_check_and_the_verdict_line(self):
        entries = [dict(shape=probe.PROXY, capacity=2304, nan_elements=0, differing_from_legacy=0,
                        poison_address_reused=True),
                   dict(shape=probe.SERVED, capacity=2304, nan_elements=0, differing_from_legacy=5)]
        self.assertEqual(probe.spot_check(entries), 'equal')
        self.assertEqual(probe.spot_check(entries[1:]), 'none')
        # Equal, but an output that did not take the poisoned address could hide an unwritten row.
        entries.append(dict(shape=probe.PROXY, capacity=66048, nan_elements=0, differing_from_legacy=0,
                            poison_address_reused=False))
        self.assertEqual(probe.spot_check(entries), 'equal-unpoisoned(1/2)')
        entries.append(dict(shape=probe.PROXY, capacity=33024, nan_elements=0, differing_from_legacy=7))
        self.assertEqual(probe.spot_check(entries), 'DIFFERS(7)')
        entries.append(dict(shape=probe.PROXY, capacity=131328, nan_elements=3, differing_from_legacy=3))
        self.assertEqual(probe.spot_check(entries), 'UNWRITTEN(3)')
        decision = probe.decide(self.times(610.0), self.fits(15.9))
        line = probe.verdict_line(decision, 'eager', probe.decide(self.times(640.0), self.fits(16.0)), 'equal')
        self.assertTrue(line.startswith('K1_PROBE verdict=GO scenario=A plan=K1a enable=0x4 basis=eager cap=131328 '
                                        'proxy=610.0us compute=579.0us served=815.8us '), line)
        for word in ('proxy/compute=1.054(A<=1.08)', 'proxy/served=0.748(C>=0.93)', 'slope=15.90/15.39us/chunk(+3.3%,A<=5%)',
                     'saving=205.8us/call', 'replay=-13.2ms(card)/-12.3ms(trace-scaled)', 'k1a_alone=meets-acceptance',
                     'spot_check=equal', 'other_basis=B'):
            self.assertIn(' ' + word, line)
        self.assertNotIn('reasons=', line)
        # With a measured trace decision beside it, the eager line also carries the measured trace saving.
        traced = probe.decide(self.times(560.0, served=760.0), self.fits(15.9), basis='trace')
        line = probe.verdict_line(decision, 'eager', traced, 'equal')
        self.assertIn(' replay=-13.2ms(card)/-12.3ms(trace-scaled)/-12.8ms(trace) ', line)

    def test_the_trace_basis_saving_is_measured_and_never_scaled_again(self):
        """On the trace basis the per-call times are already trace times: (served - proxy) x 64, no 761/815.8."""
        eager = probe.decide(self.times(610.0), self.fits(15.9))
        self.assertFalse(eager['saving_trace_measured'])
        self.assertAlmostEqual(eager['saving_ms_per_replay_card'], 205.8 * 64 / 1000, places=3)
        self.assertAlmostEqual(eager['saving_ms_per_replay_trace'], 205.8 * 64 / 1000 * probe.TRACE_BASIS, places=3)
        trace = probe.decide(self.times(560.0, served=760.0), self.fits(15.9), basis='trace')
        self.assertTrue(trace['saving_trace_measured'])
        self.assertIsNone(trace['saving_ms_per_replay_card'])
        self.assertAlmostEqual(trace['saving_ms_per_replay_trace'], 200.0 * 64 / 1000, places=6)
        line = probe.verdict_line(trace, 'trace', eager, 'equal')
        self.assertIn(' replay=-12.8ms(trace) ', line)
        self.assertNotIn('(card)', line)

    def test_the_32k_cross_check_is_recorded_not_decided_on(self):
        times = self.times(610.0)
        times[probe.PROXY][33024], times[probe.SERVED][33024] = 240.0, 304.2
        decision = probe.decide(times, self.fits(15.9))
        self.assertEqual(decision['scenario'], 'A')
        check = decision['cross_check']
        self.assertEqual((check['capacity'], check['proxy_us'], check['served_us']), (33024, 240.0, 304.2))
        self.assertAlmostEqual(check['saving_ms_per_replay'], 64.2 * 64 / 1000, places=6)
        del times[probe.SERVED][33024]
        self.assertNotIn('cross_check', probe.decide(times, self.fits(15.9)))
        empty = probe.verdict_line(probe.decide({}, {}, ['no timing (--no-timing: the watcher pass)']), 'eager')
        self.assertTrue(empty.startswith('K1_PROBE verdict=NO-DECISION basis=eager spot_check=none reasons=['), empty)

    def test_the_arguments(self):
        args = probe.parse_args(['--out', 'x.json'])
        self.assertEqual((args.capacities, args.shapes, args.iters, args.rounds, args.trace_calls, args.basis),
                         (list(probe.CAPACITIES), list(probe.DEFAULT_SHAPES), 20, 2, 16, 'eager'))
        self.assertEqual(args.kernel_root, probe.KERNEL_ROOT)
        self.assertEqual(set(probe.REQUIRED) - set(args.shapes), set())
        for bad in (['--shapes', 'G4B2_tail_share,nope'], ['--shapes', 'G4B1_tail,G4B1_tail'],
                    ['--capacities', '33024,33280'], ['--capacities', '1000'], ['--start', '241'],
                    ['--iters', '0'], ['--expect-binary-sha256', 'abc']):
            with self.subTest(bad=bad), self.assertRaises(SystemExit), mock.patch('sys.stderr'):
                probe.parse_args(['--out', 'x.json'] + bad)

    def test_the_watchdog_fires_past_the_deadline_and_a_span_is_one_deadline(self):
        exits, fired = [], []
        dog = probe.Watchdog(5, on_fire=fired.append, backstop=False, exit=exits.append)
        self.assertFalse(dog.check())
        with dog.op('outer'):
            self.assertFalse(dog.check())
            with dog.span('timed loop', 30):
                self.assertTrue(dog.coarse)
                with dog.op('inner call'):
                    self.assertEqual(dog.label, 'timed loop')         # no per-call arming inside a span
                dog.deadline = time.monotonic() - 1
                with mock.patch('sys.stdout'):
                    self.assertTrue(dog.check())
            self.assertEqual(dog.label, 'outer')
        self.assertEqual((fired, exits), (['timed loop'], [3]))
        self.assertIsNone(dog.label)
        off = probe.Watchdog(0)
        with off.op('x'):
            self.assertIsNone(off.label)


class ContractTests(unittest.TestCase):
    def test_the_proxy_does_the_head_sliced_g8_calls_per_core_work(self):
        """The design's rule: start(h) = floor(h*G/32), slice = max_h(ceil((h+1)*G/32) - start(h)), G = rows per KV head."""
        served, proxy = probe.shape_geometry(probe.SERVED), probe.shape_geometry(probe.PROXY)
        rows_per_kv = served['folded_rows'] // card.KV_HEADS
        starts = [h * rows_per_kv // 32 for h in range(card.KV_HEADS)]
        sliced = max(-(-((h + 1) * rows_per_kv) // 32) - starts[h] for h in range(card.KV_HEADS))
        self.assertEqual((rows_per_kv, starts, sliced, served['PNHt']), (48, [0, 1], 2, 3))
        self.assertEqual(proxy['PNHt'], sliced)
        self.assertEqual((proxy['B'], proxy['cores'], proxy['flags']), (served['B'], served['cores'], served['flags']))
        self.assertEqual((served['cores'], served['flags']), (64, 0x3))
        # The same L1 layout K1 would have: section 6's 796,736 B (and today's 1,048,640 for the served call).
        self.assertEqual(card.CB_BYTES[(131328, proxy['PNHt'])], 796736)
        self.assertEqual(card.CB_BYTES[(131328, served['PNHt'])], 1048640)
        # The controls: G4 B1 tail is half the cores at the proxy's tiles; G4 B2 tail is the proxy without share.
        compute, dram = probe.shape_geometry(probe.COMPUTE), probe.shape_geometry(probe.DRAM)
        self.assertEqual((compute['PNHt'], compute['cores'], compute['flags']), (2, 32, 0x1))
        self.assertEqual((dram['PNHt'], dram['cores'], dram['flags']), (2, 64, 0x1))
        self.assertEqual(set(probe.QUALIFIED) & {probe.PROXY, probe.DRAM}, set())

    def gate_line(self, tag):
        lines = [line for line in read(GATE).splitlines() if re.search(r'\*-%s[)|]' % tag, line) and 'export ' in line]
        self.assertEqual(len(lines), 1, tag)
        return lines[0].split()

    def test_the_served_shape_is_every_current_gate_arms(self):
        import pooled_attention_replay as replay
        for tag in ('v192', 'v194', 'v195'):
            with self.subTest(tag=tag):
                words = self.gate_line(tag)
                for word in ('KOPGRAFT64=/home/thatch/opgraft-K64g', 'M3NATIVE_REPLAY_GROUP_ROWS=8',
                             'M3NATIVE_SDPA_MODES=tail,share', 'M3NATIVE_SDPA_PF=1'):
                    self.assertIn(word, words)
        rows, batches, sentinel, offsets, role = probe.SHAPES[probe.SERVED]
        self.assertEqual((rows, batches, offsets, role), (8, 2, (0, 8), 'served'))
        self.assertEqual(replay.QWEN_DECODE_MAGIC | replay.mode_flags(('tail', 'share'), 2), sentinel)
        self.assertEqual(sentinel, card.TAIL_SHARE)

    def test_the_runner_defaults_to_the_gates_image_and_the_served_hashes(self):
        runner = read(RUNNER)
        image = re.search(r'^IMAGE=\$\{IMAGE:-(sha256:[0-9a-f]{64})\}$', runner, flags=re.M).group(1)
        gate = [line for line in read(GATE).splitlines() if '*-v194|' in line and 'image=sha256:' in line]
        self.assertEqual(len(gate), 1)
        self.assertIn('image=%s ' % image, gate[0] + ' ')
        self.assertIn('K64G_TTNNCPP_SHA256=%s' % probe.K64G_TTNNCPP_SHA256 + NL, runner)
        self.assertIn('G=${KOPGRAFT64:-$HOME/opgraft-K64g}' + NL, runner)
        self.assertIn('READER_QWEN=%s' % probe.SERVED_KERNELS['dataflow/reader_decode_qwen.cpp'] + NL, runner)
        self.assertIn('COMPUTE_QWEN=%s' % probe.SERVED_KERNELS['compute/sdpa_flash_decode_qwen.cpp'] + NL, runner)
        self.assertNotIn(chr(13), runner)

    def test_the_served_kernels_are_stage_three(self):
        import make_qwen_kernels as kernels
        self.assertEqual(probe.SERVED_KERNELS['dataflow/reader_decode_qwen.cpp'], kernels.OUTPUTS_STAGE3[kernels.READER_NAME])
        self.assertEqual(probe.SERVED_KERNELS['compute/sdpa_flash_decode_qwen.cpp'], kernels.OUTPUTS_STAGE3[kernels.COMPUTE_NAME])
        for name, expected in probe.SERVED_KERNELS.items():
            self.assertEqual(sha((STAGE3 / Path(name).name).read_bytes()), expected)
        self.assertEqual(probe.KERNEL_ROOT, kernels.DUMP_PREFIX.rstrip('/'))


def find_bash():
    candidates = []
    if os.name == 'nt':
        for root in (os.environ.get('ProgramW6432'), os.environ.get('ProgramFiles'), 'C:/Program Files'):
            if root:
                candidates.append(Path(root) / 'Git' / 'bin' / 'bash.exe')
    found = shutil.which('bash')
    if found and not (os.name == 'nt' and ('system32' in found.lower() or 'windowsapps' in found.lower())):
        candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


BASH = find_bash()
SCRUB = ('QUAL_CARD', 'ALLOW_SERVING_CARD', 'IMAGE', 'RESULTS', 'WATCHER', 'WATCHDOG_S', 'KOPGRAFT64',
         'PROBE_DRY_RUN', 'EXPECT_TTNNCPP_SHA256', 'CARD_B_ARGS', 'MSYS')


def make_graft(root, binary=b'fake _ttnncpp.so'):
    """A graft directory shaped like ~/opgraft-K64g: the .so files, the op directories with the served qwen
    kernels, and a MANIFEST.sha256 in build_k64g.sh's format."""
    graft = Path(root) / 'graft'
    (graft / 'sdpa_decode' / 'device' / 'kernels' / 'dataflow').mkdir(parents=True)
    (graft / 'sdpa_decode' / 'device' / 'kernels' / 'compute').mkdir(parents=True)
    for name in ('attn_prep', 'nlp_concat_heads_decode', 'sdpa'):
        (graft / name).mkdir()
        (graft / name / 'placeholder.txt').write_bytes(name.encode())
    (graft / '_ttnn.so').write_bytes(b'fake _ttnn.so')
    (graft / '_ttnncpp.so').write_bytes(binary)
    kernels = graft / 'sdpa_decode' / 'device' / 'kernels'
    shutil.copyfile(STAGE3 / 'reader_decode_qwen.cpp', kernels / 'dataflow' / 'reader_decode_qwen.cpp')
    shutil.copyfile(STAGE3 / 'sdpa_flash_decode_qwen.cpp', kernels / 'compute' / 'sdpa_flash_decode_qwen.cpp')
    write_manifest(graft)
    return graft


def write_manifest(graft):
    files = sorted('./' + path.relative_to(graft).as_posix() for path in graft.rglob('*')
                   if path.is_file() and path.name != 'MANIFEST.sha256')
    (graft / 'MANIFEST.sha256').write_bytes(''.join('%s  %s\n' % (sha((graft / name).read_bytes()), name)
                                                    for name in files).encode())


@unittest.skipUnless(BASH, 'bash not found')
class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_runner(self, *args, **env):
        environ = {key: value for key, value in os.environ.items() if key not in SCRUB}
        environ.update(HOME=self.dir.as_posix(), RESULTS=(self.dir / 'results').as_posix(), PROBE_DRY_RUN='1')
        environ.update(env)
        return subprocess.run([BASH, RUNNER.as_posix()] + list(args), env=environ, capture_output=True, text=True,
                              encoding='utf-8', errors='replace', timeout=120)

    def argv(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        lines = [line for line in result.stdout.splitlines() if line.startswith('### argv: ')]
        self.assertEqual(len(lines), 1, result.stdout)
        return shlex.split(lines[0][len('### argv: '):])

    def mounts(self, argv):
        out = []
        for index, word in enumerate(argv):
            if word == '--mount':
                fields = dict(part.split('=', 1) if '=' in part else (part, True) for part in argv[index + 1].split(','))
                out.append(fields)
        return out

    def arm_graft_mounts(self):
        """What lever_n_m3native_run_arm.sh mounts from KOPGRAFT64 for the gate's argv (M3NATIVE_SDPA_PF=1)."""
        arm = read(ARM)
        pairs = set(re.findall(r'-v \$KOPGRAFT64/([A-Za-z0-9_.]+):(/opt/[^:" ]+):ro', arm))
        self.assertIn('-v $KOPGRAFT64/sdpa:$sdpa_pf_target:ro', arm)
        target = re.search(r'^\s*sdpa_pf_target=(\S+)$', arm, flags=re.M).group(1)
        pairs.add(('sdpa', target))
        self.assertEqual(len(pairs), 7)
        return pairs

    def test_the_dry_run_launches_on_card_b_with_the_arms_graft_mounts(self):
        graft = make_graft(self.dir)
        result = self.run_runner(KOPGRAFT64=graft.as_posix(), EXPECT_TTNNCPP_SHA256=sha(b'fake _ttnncpp.so'))
        argv = self.argv(result)
        self.assertEqual(result.stderr, '')
        self.assertIn('### graft %s: _ttnncpp.so %s, qwen kernels 280a847f / 8776fcc7, manifest verified'
                      % (graft.as_posix(), sha(b'fake _ttnncpp.so')[:16]), result.stdout)
        self.assertEqual(argv[:3], ['docker', 'run', '--rm'])
        self.assertEqual(argv[argv.index('--device') + 1], '/dev/tenstorrent/by-id/' + CARD_B)
        self.assertEqual(argv.count('--device'), 1)
        self.assertEqual(argv[argv.index('--name') + 1], 'qwen-k1probe-card-b')
        self.assertIn('card=%s (card-b)' % CARD_B, result.stdout)
        mounts = self.mounts(argv)
        prefix = graft.as_posix() + '/'
        grafted = {(m['src'][len(prefix):], m['dst']) for m in mounts if m['src'].startswith(prefix)}
        self.assertEqual(grafted, self.arm_graft_mounts())
        self.assertTrue(all(m.get('readonly') for m in mounts if m['src'].startswith(prefix)))
        bench = {m['dst']: m['src'] for m in mounts if m['dst'].startswith('/bench/')}
        self.assertEqual(sorted(bench), ['/bench/probe_k1_card_b.py', '/bench/test_sdpa_decode_qwen_card_m.py'])
        for dst, src in bench.items():
            self.assertTrue(src.endswith('/optimisation/ttnn-op/sdpa_decode_qwen/' + dst[len('/bench/'):]), src)
        cache = [m for m in mounts if m['dst'] == '/kcache']
        self.assertEqual(len(cache), 1)
        self.assertRegex(cache[0]['src'], r'/results/kcache-probe-[0-9]{8}T[0-9]{6}$')   # fresh per run
        env = [argv[i + 1] for i, word in enumerate(argv) if word == '-e']
        for value in ('QWEN_SDPA_TREE_SCRATCH_ROUNDS=1', 'TT_METAL_CACHE=/kcache', 'TT_METAL_HOME=/opt/tt-metal'):
            self.assertIn(value, env)
        self.assertIn('-e QWEN_SDPA_TREE_SCRATCH_ROUNDS=1', read(ARM))                   # as the arm sets it
        self.assertNotIn('TT_METAL_WATCHER=5', env)
        entry = argv.index('--entrypoint')
        self.assertEqual(argv[entry + 1:entry + 4], ['sh', 'sha256:70c27e7539db757e3b166f0d14ccdc32edf86532017b6f2f50310fcaad855c3f', '-c'])
        self.assertIn('exec python3 -B /bench/probe_k1_card_b.py "$@"', argv[entry + 4])
        tail = argv[entry + 5:]
        self.assertEqual(tail[0], 'probe')
        args = probe.parse_args(tail[1:])
        self.assertEqual((args.expect_binary_sha256, args.watchdog, args.no_timing, args.capacities),
                         (sha(b'fake _ttnncpp.so'), 300.0, False, list(probe.CAPACITIES)))
        self.assertRegex(args.out.as_posix(), r'^/results/probe-[0-9]{8}T[0-9]{6}\.json$')

    def test_the_default_expectation_is_k64g_and_the_watcher_pass(self):
        result = self.run_runner(WATCHER='1', CARD_B_ARGS='--rounds 1')
        argv = self.argv(result)
        self.assertIn('does not exist here; the graft was not checked', result.stdout)
        env = [argv[i + 1] for i, word in enumerate(argv) if word == '-e']
        self.assertIn('TT_METAL_WATCHER=5', env)
        self.assertTrue(any(m['dst'] == '/opt/tt-metal/generated/watcher' for m in self.mounts(argv)))
        args = probe.parse_args(argv[argv.index('probe') + 1:])
        self.assertEqual((args.expect_binary_sha256, args.watchdog, args.no_timing, args.capacities, args.rounds),
                         (probe.K64G_TTNNCPP_SHA256, 120.0, True, [2304, 33024], 1))
        self.assertIn('WATCHER=1: TT_METAL_WATCHER=5, per-call watchdog 120 s, container timeout 900 s', result.stdout)

    def test_an_empty_expectation_skips_the_binary_hash_only(self):
        graft = make_graft(self.dir)
        argv = self.argv(self.run_runner(KOPGRAFT64=graft.as_posix(), EXPECT_TTNNCPP_SHA256=''))
        self.assertEqual(argv[argv.index('--expect-binary-sha256') + 1], '')
        self.assertEqual(probe.parse_args(argv[argv.index('probe') + 1:]).expect_binary_sha256, '')

    def refused(self, result, needle):
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn(needle, result.stderr)
        self.assertNotIn('### argv: ', result.stdout)

    def test_the_graft_checks_refuse_anything_but_the_served_graft(self):
        graft = make_graft(self.dir)
        self.refused(self.run_runner(KOPGRAFT64=graft.as_posix()),
                     'refusing: %s/_ttnncpp.so is %s, not 134bc8347d6533b3' % (graft.as_posix(), sha(b'fake _ttnncpp.so')[:16]))
        expect = dict(KOPGRAFT64=graft.as_posix(), EXPECT_TTNNCPP_SHA256=sha(b'fake _ttnncpp.so'))
        reader = graft / 'sdpa_decode' / 'device' / 'kernels' / 'dataflow' / 'reader_decode_qwen.cpp'
        reader.write_bytes(reader.read_bytes() + b'// K1\n')
        self.refused(self.run_runner(**expect), 'refusing: %s/MANIFEST.sha256 does not verify' % graft.as_posix())
        write_manifest(graft)
        self.refused(self.run_runner(**expect), 'refusing: %s is %s, not the served 280a847f'
                     % (reader.as_posix(), sha(reader.read_bytes())))
        shutil.copyfile(STAGE3 / 'reader_decode_qwen.cpp', reader)
        write_manifest(graft)
        self.assertEqual(self.run_runner(**expect).returncode, 0)
        shutil.rmtree(graft / 'sdpa')
        self.refused(self.run_runner(**expect), 'refusing: %s/sdpa missing' % graft.as_posix())

    def test_the_serving_pair_and_an_unknown_role_are_refused(self):
        result = self.run_runner(QUAL_CARD=CARD_M)
        self.refused(result, 'refusing: QUAL_CARD=%s is card M' % CARD_M)
        self.refused(self.run_runner('candidate'), 'unknown role candidate')

    def test_the_runner_echoes_the_verdict_line_not_the_summary_line(self):
        """The harness prints 'K1_PROBE verdict=...' and then 'SDPA_K1_PROBE passed=...', which also contains
        'K1_PROBE '; the runner's '### K1_PROBE' line (what the rig sequence greps) must be the verdict."""
        lines = [line for line in read(RUNNER).splitlines() if line.startswith('echo ') and 'K1_PROBE' in line]
        self.assertEqual(len(lines), 1, lines)
        log = self.dir / 'probe.log'
        with self.subTest('both lines'):
            log.write_bytes(b'checksum cap=2304 x\nK1_PROBE verdict=GO scenario=A enable=0x4\n'
                            b'SDPA_K1_PROBE passed=True checksums=20\n### exit 0\n')
            result = subprocess.run([BASH, '-c', 'set -o pipefail; log=%s; %s' % (shlex.quote(log.as_posix()), lines[0])],
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.stdout, '### K1_PROBE verdict=GO scenario=A enable=0x4' + NL, result.stderr)
        with self.subTest('no verdict'):
            log.write_bytes(b'SDPA_K1_PROBE passed=False checksums=0\n')
            result = subprocess.run([BASH, '-c', 'set -o pipefail; log=%s; %s' % (shlex.quote(log.as_posix()), lines[0])],
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.stdout, '### no K1_PROBE line' + NL, result.stderr)

    def test_a_real_run_resolves_the_board_before_anything_else(self):
        if Path('/dev/tenstorrent/by-id', CARD_B).exists():
            self.skipTest('card B is present on this host: the probe would run')
        result = self.run_runner(PROBE_DRY_RUN='0')
        self.refused(result, 'refusing: %s (card B, the qualification card) has no device node here' % CARD_B)
        self.assertFalse((self.dir / 'results').exists())


# ---------------------------------------------------------------------------------------------
# The device flow on a fake ttnn.
# ---------------------------------------------------------------------------------------------

SHAPE_OF = {(96, 2, 3): 'G8B2_tail_share', (48, 2, 3): 'G4B2_tail_share', (48, 2, 1): 'G4B2_tail',
            (48, 1, 1): 'G4B1_tail', (96, 2, 1): 'G8B2_tail', (48, 3, 3): 'G4B3_tail_share'}
# Per-call models, intercept + slope x busiest-core chunks (us): card M's lines; the proxy in scenario A.
MODEL_A = {'G8B2_tail_share': (112.3, 21.32), 'G4B2_tail_share': (89.0, 15.6), 'G4B2_tail': (72.7, 23.0),
           'G4B1_tail': (71.2, 15.39), 'G8B2_tail': (105.4, 23.36), 'G4B3_tail_share': (88.7, 30.61), None: (50.0, 30.0)}
ELEMENT = {'bf16': 2, 'bf8': 1, 'int32': 4}


class FakeTensor:
    def __init__(self, address, shape, dtype, nbytes):
        self.address, self._shape, self.dtype, self.nbytes = address, tuple(shape), dtype, nbytes
        self.freed = False

    @property
    def shape(self):
        return self._shape

    def buffer_address(self):
        return self.address


class FakeTtnn:
    """The ttnn surface the probe touches. DRAM: a size-keyed LIFO free list (a freed address is the next
    same-size allocation's) whose bytes stay stale. The op has the graft's semantics (share: every entry
    reads page-table row 0; tail: only the mask's last 256 columns) and prints the F4 line once per new qwen
    program to fd 1; its time is the model's, on a model clock; trace capture records the launches and
    execute_trace replays them. unwritten=<shape> leaves the last 8 folded rows of the last entry unwritten;
    wrong=<shape> adds 1 to that shape's qwen output; silent=True never logs (the graft is not executed)."""

    int32, bfloat16, bfloat8_b = 'int32', 'bf16', 'bf8'
    ROW_MAJOR_LAYOUT, TILE_LAYOUT, DRAM_MEMORY_CONFIG = 'rm', 'tile', 'dram'

    def __init__(self, torch, model=None, unwritten=None, wrong=None, silent=False, reuse=True):
        self.torch, self.model = torch, dict(MODEL_A if model is None else model)
        self.unwritten, self.wrong, self.silent, self.reuse = unwritten, wrong, silent, reuse
        self.transformer = self
        self.memory, self.free, self.next = {}, {}, 0x10000
        self.now = 0.0
        self.programs, self.traces, self.capturing = set(), {}, None
        self.launches = {}

    def clock(self):
        return self.now

    # device -----------------------------------------------------------------------------
    def open_device(self, **options):
        self.options = options
        return self

    def close_device(self, device):
        self.closed = True

    def enable_program_cache(self):
        pass

    def compute_with_storage_grid_size(self):
        return type('Grid', (), {'x': 11, 'y': 10})()

    def synchronize_device(self, device):
        pass

    def SDPAProgramConfig(self, **options):  # noqa: N802 - mirrors ttnn
        return options

    # memory -----------------------------------------------------------------------------
    def allocate(self, shape, dtype):
        nbytes = int(self.torch.Size(shape).numel()) * ELEMENT[dtype]
        pool = self.free.get(nbytes) if self.reuse else None
        if pool:
            address = pool.pop()
        else:
            address, self.next = self.next, self.next + nbytes
        return FakeTensor(address, shape, dtype, nbytes)

    def from_torch(self, host, *, device, dtype, layout, memory_config):
        tensor = self.allocate(host.shape, dtype)
        self.memory[tensor.address] = host.clone()
        return tensor

    def read(self, tensor):
        if tensor.freed:
            raise RuntimeError('read of a freed tensor')
        return self.memory[tensor.address]

    def to_torch(self, tensor):
        return self.read(tensor).clone()

    def deallocate(self, tensor):
        if not tensor.freed:
            tensor.freed = True
            self.free.setdefault(tensor.nbytes, []).append(tensor.address)

    # trace ------------------------------------------------------------------------------
    def begin_trace_capture(self, device, cq_id):
        self.capturing = len(self.traces) + 1
        self.traces[self.capturing] = []
        return self.capturing

    def end_trace_capture(self, device, trace, cq_id):
        self.capturing = None

    def execute_trace(self, device, trace, cq_id, blocking):
        for write, cost in self.traces[trace]:
            write()
            self.now += cost * 1e-6

    def release_trace(self, device, trace):
        del self.traces[trace]

    # the op -----------------------------------------------------------------------------
    def paged_scaled_dot_product_attention_decode(self, query, k, v, pages, *, is_causal, scale, program_config,
                                                  memory_config, attn_mask=None, cur_pos_tensor=None):
        torch = self.torch
        sentinel = program_config['q_chunk_size']
        qwen = (sentinel & 0xFFFFFF00) == card.MAGIC
        flags = sentinel & 0xFF if qwen else 0
        batches, rows = query.shape[1], query.shape[2]
        table = self.read(pages)
        capacity = table.shape[1] * card.PAGE
        if qwen and flags & 0x1 and attn_mask.shape[3] != capacity:
            raise RuntimeError('TT_FATAL [QWEN-SDPA] tail mask must be full width')
        if qwen and not self.silent:
            key = (capacity, batches, -(-rows // 32), attn_mask.shape[3] // 32, flags)
            if key not in self.programs:
                self.programs.add(key)
                os.write(1, ('Op | INFO | [QWEN-SDPA] flags=0x%x B=%d PNHt=%d St=%d mask_width_t=%d kv_share=%s '
                             'scratch_slots=4 cb_bytes=%d\n' % (flags, batches, key[2], capacity // 32, key[3],
                                                                'true' if flags & 0x2 and batches > 1 else 'false',
                                                                card.CB_BYTES.get((capacity, key[2]), 777))).encode())
        name = SHAPE_OF.get((rows, batches, flags)) if qwen else None
        intercept, slope = self.model[name]
        cost = intercept + slope * probe.busiest_chunks(capacity)
        self.launches[name] = self.launches.get(name, 0) + 1
        q, keys, mask = self.read(query).float(), self.read(k), self.read(attn_mask).float()
        share, tail = bool(flags & 0x2) and batches > 1, bool(flags & 0x1)

        def compute():
            out = q.clone()
            for b in range(batches):
                row = table[0 if share else b].long()[:4]
                out[0, b] += keys[row].float().mean()
                seen = mask[b, 0][:, -256:] if tail else mask[b, 0]
                out[0, b] += torch.isinf(seen).sum(dim=1, keepdim=True).float() / 1000
            if qwen and self.wrong == name:
                out += 1
            return out.to(torch.bfloat16)

        output = self.allocate((1, batches, rows, card.HEAD_DIM), 'bf16')
        stale = self.memory.get(output.address)
        if stale is None or tuple(stale.shape) != output.shape:
            stale = torch.zeros(output.shape, dtype=torch.bfloat16)
        self.memory[output.address] = stale

        def write():
            data = compute()
            if qwen and self.unwritten == name:
                data[0, batches - 1, rows - 8:] = self.memory[output.address][0, batches - 1, rows - 8:]
            self.memory[output.address] = data

        if self.capturing is not None:
            self.traces[self.capturing].append((write, cost))
        else:
            write()
            self.now += cost * 1e-6
        return output


class DryRunTests(unittest.TestCase):
    """The probe end to end on the fake. The capacities 256 / 2,304 / 4,352 / 8,448 with one core per head
    give the design's 1 / 9 / 17 / 33 busiest-core chunks at a fraction of the host memory."""

    CAPS = '256,2304,4352,8448'
    ARGS = ['--capacities', CAPS, '--warmup', '1', '--iters', '3', '--rounds', '2', '--replays', '2',
            '--trace-calls', '4', '--trace-warmup', '1']
    BINARY = b'fake K64g _ttnncpp.so'

    def setUp(self):
        import torch
        self.torch = torch
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.kernels = self.dir / 'kernels'
        for name in probe.SERVED_KERNELS:
            (self.kernels / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(STAGE3 / Path(name).name, self.kernels / name)
        self.binary = self.dir / '_ttnncpp.so'
        self.binary.write_bytes(self.BINARY)

    def tearDown(self):
        self.tmp.cleanup()

    def run_probe(self, fake, extra=(), markers=None, scratch='1', name='probe'):
        out = self.dir / ('%s.json' % name)
        markers = dict(flags=True, share=True, stage1=False) if markers is None else markers
        argv = ['--out', str(out), '--kernel-root', str(self.kernels), '--expect-binary-sha256', sha(self.BINARY)]
        environ = {card.SCRATCH_ENV: scratch} if scratch is not None else {}
        with mock.patch.dict(sys.modules, {'ttnn': fake}), \
                mock.patch.object(card, 'loaded_binary', return_value=(str(self.binary), markers)), \
                mock.patch.dict(os.environ, environ), \
                mock.patch.multiple(probe, CORES_PER_HEAD=1, DECISION_CAPACITY=8448, clock=fake.clock), \
                mock.patch.object(card, 'WATCHDOG', card.WATCHDOG), mock.patch.object(probe, 'WATCHDOG', probe.WATCHDOG), \
                mock.patch.object(probe, 'print', create=True), mock.patch('sys.stdout'):
            if scratch is None:
                os.environ.pop(card.SCRATCH_ENV, None)
            status = probe.main(argv + self.ARGS + list(extra))
        return status, json.loads(out.read_text())

    def test_scenario_a_end_to_end(self):
        fake = FakeTtnn(self.torch)
        status, report = self.run_probe(fake)
        self.assertEqual((report.get('error'), report['failures']), (None, []))
        self.assertEqual((status, report['passed']), (0, True))
        self.assertEqual(report['warnings'], [])
        self.assertEqual(report['binary']['sha256'], sha(self.BINARY))
        self.assertEqual(report['kernels']['sha256'], probe.SERVED_KERNELS)
        self.assertEqual(report['env'][card.SCRATCH_ENV], '1')
        # Checksums: 4 capacities x 5 shapes, each poisoned (the address reused), all equal to legacy.
        self.assertEqual(len(report['checksums']), 20)
        self.assertTrue(all(entry['equal_legacy'] and entry['poison_address_reused'] and not entry['nan_elements']
                            for entry in report['checksums']))
        self.assertEqual([entry['chunks'] for entry in report['checksums'][::5]], [1, 9, 17, 33])
        # Timing: the model's per-call times on both bases, the rounds interleaved.
        self.assertEqual(len(report['timing']), 20)
        for row in report['timing']:
            intercept, slope = MODEL_A[row['shape']]
            expected = intercept + slope * row['chunks']
            self.assertAlmostEqual(row['eager']['median_us'], expected, places=3)
            self.assertAlmostEqual(row['trace']['median_us'], expected, places=3)
            self.assertEqual((row['eager']['n'], len(row['eager_round_medians']), row['trace']['n']), (6, 2, 2))
            self.assertTrue(row['trace_output_matches_eager'])
        self.assertAlmostEqual(report['slopes'][probe.SERVED]['eager']['fit_slope'], 21.32, places=3)
        self.assertAlmostEqual(report['slopes'][probe.PROXY]['trace']['fit_intercept'], 89.0, places=3)
        # Every qwen program the probe ran has its factory line: the graft was executed.
        self.assertEqual(len(report['factory_lines']), 20)
        self.assertEqual(len(report['requested_programs']), 20)
        self.assertIn('cap=8448 B=2 PNHt=2 mask_width_t=264 flags=0x3', report['requested_programs'])
        self.assertIn('cap=8448 B=2 PNHt=3 mask_width_t=264 flags=0x3', report['requested_programs'])
        self.assertIn('cap=8448 B=1 PNHt=2 mask_width_t=264 flags=0x1', report['requested_programs'])
        # The decision.
        for basis in ('eager', 'trace'):
            decision = report['decision'][basis]
            self.assertEqual((decision['verdict'], decision['scenario'], decision['enable']), ('GO', 'A', '0x4'))
        decision = report['decision']['eager']
        self.assertAlmostEqual(decision['proxy_us'], 89.0 + 15.6 * 33, places=3)
        self.assertAlmostEqual(decision['saving_us_per_call'], (112.3 + 21.32 * 33) - (89.0 + 15.6 * 33), places=3)
        self.assertEqual(report['spot_check'], 'equal')
        self.assertTrue(report['verdict_line'].startswith('K1_PROBE verdict=GO scenario=A plan=K1a enable=0x4 basis=eager '
                                                          'cap=8448 '), report['verdict_line'])
        self.assertIn(' other_basis=A', report['verdict_line'])
        # Eager: the card figure, the design's 761/815.8 scaling, and the measured trace figure beside them.
        self.assertIn(' replay=-13.6ms(card)/-12.7ms(trace-scaled)/-13.6ms(trace) ', report['verdict_line'])
        self.assertTrue(report['decision']['trace']['saving_trace_measured'])
        self.assertEqual(report['geometry'][probe.PROXY]['PNHt'], 2)
        self.assertEqual(fake.options['trace_region_size'], 16 << 20)

    def test_scenario_c_on_the_trace_basis(self):
        model = dict(MODEL_A, **{probe.PROXY: (112.0, 21.0)})
        status, report = self.run_probe(FakeTtnn(self.torch, model=model), ['--basis', 'trace'])
        self.assertEqual(status, 0)
        self.assertEqual(report['decision']['trace']['scenario'], 'C')
        self.assertTrue(report['verdict_line'].startswith('K1_PROBE verdict=GO scenario=C plan=K1b-led enable=0x4,0x8 '
                                                          'basis=trace '), report['verdict_line'])
        self.assertIn(' k1a_alone=stop-rule', report['verdict_line'])

    def test_rows_left_unwritten_show_through_the_poison(self):
        status, report = self.run_probe(FakeTtnn(self.torch, unwritten=probe.PROXY))
        self.assertEqual(report['failures'], [])
        proxy = [entry for entry in report['checksums'] if entry['shape'] == probe.PROXY]
        self.assertTrue(all(entry['poison_address_reused'] and entry['nan_elements'] == 8 * 256 for entry in proxy))
        self.assertEqual(report['spot_check'], 'UNWRITTEN(%d)' % (4 * 8 * 256))
        self.assertEqual(report['decision']['eager']['verdict'], 'NO-DECISION')
        self.assertIn('left output rows unwritten', report['verdict_line'])
        self.assertTrue(any('rows the kernels left unwritten' in warning for warning in report['warnings']))

    def test_an_output_that_missed_the_poison_is_flagged(self):
        """An allocator that never reuses a freed address: every output misses the poison, so an unwritten
        row could hide behind stale bytes. Warned per output and marked in the spot check."""
        status, report = self.run_probe(FakeTtnn(self.torch, reuse=False), ['--capacities', '2304,4352', '--no-timing'])
        self.assertEqual((status, report['failures']), (0, []))
        self.assertTrue(all(entry['poison_address_reused'] is False for entry in report['checksums']))
        self.assertEqual(report['spot_check'], 'equal-unpoisoned(2/2)')
        self.assertEqual(sum('did not take the poisoned address' in warning for warning in report['warnings']), 10)
        self.assertIn(' spot_check=equal-unpoisoned(2/2) ', report['verdict_line'])

    def test_the_spot_check_and_a_qualified_shape(self):
        status, report = self.run_probe(FakeTtnn(self.torch, wrong=probe.PROXY), name='spot')
        self.assertEqual((status, report['failures']), (0, []))                 # not a failure of the run
        self.assertTrue(report['spot_check'].startswith('DIFFERS('), report['spot_check'])
        # A proxy whose bytes are wrong ran the share protocol wrongly: its time decides nothing, but the
        # numbers are still reported.
        decision = report['decision']['eager']
        self.assertEqual((decision['verdict'], decision['scenario']), ('NO-DECISION', None))
        self.assertAlmostEqual(decision['proxy_us'], 89.0 + 15.6 * 33, places=3)
        self.assertTrue(any("the design's spot check" in warning for warning in report['warnings']))
        self.assertTrue(report['verdict_line'].startswith('K1_PROBE verdict=NO-DECISION basis=eager cap=8448 '),
                        report['verdict_line'])
        self.assertIn(' spot_check=DIFFERS(', report['verdict_line'])
        self.assertIn('differs from legacy', report['verdict_line'])
        status, report = self.run_probe(FakeTtnn(self.torch, wrong=probe.SERVED), name='served')
        self.assertEqual(status, 1)
        self.assertTrue(any('card M qualified this shape on K64f' in failure for failure in report['failures']))
        self.assertEqual(report['decision']['eager']['verdict'], 'NO-DECISION')

    def test_a_graft_that_never_logs_was_not_executed(self):
        status, report = self.run_probe(FakeTtnn(self.torch, silent=True), ['--capacities', '2304,8448'])
        self.assertEqual(status, 1)
        self.assertEqual(len(report['failures']), 10)
        self.assertTrue(all(failure.startswith('factory log: ') and 'no [QWEN-SDPA] line' in failure
                            for failure in report['failures']))
        self.assertTrue(report['verdict_line'].startswith('K1_PROBE verdict=NO-DECISION'))
        self.assertIn('measurement invalid: 10 failures', report['verdict_line'])

    def test_the_binary_the_kernels_and_the_scratch_are_checked_first(self):
        fake = FakeTtnn(self.torch)
        status, report = self.run_probe(fake, ['--expect-binary-sha256', 'f' * 64], name='sha')
        self.assertEqual(status, 1)
        self.assertIn('not the expected ffffffffffffffff', report['failures'][0])
        self.assertEqual((report['checksums'], fake.launches), ([], {}))
        status, report = self.run_probe(FakeTtnn(self.torch), markers=dict(flags=True, share=False, stage1=True),
                                        name='stage1')
        self.assertIn('stage 1, not a stage-3 (KV share) build', report['failures'][0])
        reader = self.kernels / 'dataflow' / 'reader_decode_qwen.cpp'
        reader.write_bytes(reader.read_bytes() + b'// changed\n')
        status, report = self.run_probe(FakeTtnn(self.torch), name='kernel')
        self.assertEqual(status, 1)
        self.assertIn("not the served 280a847fae833891 (the op directory is not K64g's)", report['failures'][0])
        shutil.copyfile(STAGE3 / 'reader_decode_qwen.cpp', reader)
        status, report = self.run_probe(FakeTtnn(self.torch), scratch=None, name='scratch')
        self.assertEqual(status, 1)
        self.assertIn('QWEN_SDPA_TREE_SCRATCH_ROUNDS=1 is required', report['failures'][0])

    def test_the_watcher_pass_records_checksums_and_decides_nothing(self):
        fake = FakeTtnn(self.torch)
        status, report = self.run_probe(fake, ['--capacities', '2304,4352', '--no-timing'])
        self.assertEqual((status, report['passed'], report['failures']), (0, True, []))
        self.assertEqual((len(report['checksums']), report['timing']), (10, []))
        self.assertNotIn('trace_region_size', fake.options)
        self.assertEqual(report['decision']['eager']['verdict'], 'NO-DECISION')
        self.assertIn('no timing (--no-timing: the watcher pass)', report['verdict_line'])


if __name__ == '__main__':
    unittest.main()
