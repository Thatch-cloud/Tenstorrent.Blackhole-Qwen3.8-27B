"""CPU checks for the K64i card-B qualification (sdpa_decode_slice_card_b.py, run_card_b.sh); no device, no ttnn.

  - the pure helpers: sentinels, twins, the reference role's modes, the factory-line and q-slice-line keys and
    the log check (requested / stray / wrong fields), the timing analysis and section 7's verdict at every
    edge, the verdict line, the arguments;
  - the runner (needs bash): the dry runs launch on card B by board id with exactly the arm's graft mounts, the
    four harness files, the arm's scratch setting and a fresh kernel cache, for both roles; the watcher pass;
    the pre-launch graft checks refuse a manifest that does not verify, a wrong kernel, a candidate .so without
    the stage-4 literal and a reference .so with it; the K1_CARD echo is the verdict line;
  - a full dry run of the device flow on a fake ttnn whose op is the index model (slice_index_model.output_rows:
    the reader's Q and mask rows, the writer's L1 slot and DRAM tile) with the factory's refusals, log lines and
    CB bytes, a DRAM allocator that reuses freed addresses (stale bytes stay), a model clock and trace replay:
    the candidate passes end to end and reads section 7's verdict; the reference round-trips its shas; and the
    controls are live - a writer without the W3 rebase, both R8 slips (row offset and batch stride), an R7
    slip, rows left unwritten (the poison), an allocator that never reuses the poisoned address, a graft that
    never prints its q-slice line, a refusal the factory lacks and a tampered reference all fail.

    py -3.11 -B -m unittest test_sdpa_decode_slice_card_b      (from this directory)
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
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
OPS = HERE.parent
ROOT = HERE.parents[2]
CI = ROOT / 'scripts' / 'ci'
QWEN = OPS / 'sdpa_decode_qwen'
for path in (str(HERE), str(QWEN), str(CI)):
    if path not in sys.path:
        sys.path.insert(0, path)

import make_slice_kernels as kernels  # noqa: E402
import probe_k1_card_b as probe  # noqa: E402
import sdpa_decode_slice_card_b as harness  # noqa: E402
import slice_index_model as model  # noqa: E402
import test_sdpa_decode_qwen_card_m as card  # noqa: E402

RUNNER = HERE / 'run_card_b.sh'
ARM = CI / 'lever_n_m3native_run_arm.sh'
STAGE3 = QWEN / 'stage3'
CARD_B = 'blackhole-F36F768B9A5CAFA0'
CARD_M = 'blackhole-CEF5729692C19E6D'
NL = chr(10)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return path.read_text(encoding='utf-8')


class HelperTests(unittest.TestCase):
    def test_sentinels_twins_and_the_reference_modes(self):
        self.assertEqual((harness.sentinel(None), harness.sentinel(0x7)), (0, 0x51DEC007))
        self.assertEqual([harness.twin_of(flags) for flags in (0x4, 0x5, 0x6, 0x7, 0xA, 0xB, 0xF)],
                         [0x0, 0x1, 0x2, 0x3, 0x2, 0x3, 0x3])
        self.assertEqual(harness.reference_modes('G8B2'), (None, 0x0, 0x1, 0x2, 0x3))
        self.assertEqual(harness.reference_modes('G16B1'), ())
        self.assertEqual(harness.head_rows('G7B2', 1), (42, 84))
        for shape, modes in harness.MODES.items():
            rows = harness.folded_rows(shape)
            for flags in modes:
                if flags & harness.SLICE:
                    self.assertIsNone(model.slice_rule(rows)['refusal'], (shape, flags))
                if flags & harness.READAHEAD:
                    self.assertTrue(flags & harness.SHARE, 'the harness never asks for 0x8 without 0x2 but in N-S3')
        # The served call's slice program: G8 B=2, 3 -> 2 tiles, the design's CB bytes.
        key = harness.program_key(131328, 2, 96, 131328, 0x7)
        self.assertEqual(key, (131328, 2, 2, 4104, 0x7))
        self.assertEqual(model.cb_bytes(131328, key[2]), 796736)
        self.assertEqual(harness.slice_key(96, 2, 0xF), (48, 3, 2, True))
        self.assertEqual(harness.slice_key(96, 1, 0xF), (48, 3, 2, False))      # 0x8 is inert at B = 1
        self.assertEqual(harness.slice_key(96, 2, 0xB), (48, 3, 3, True))       # read-ahead alone: slice = full
        self.assertIsNone(harness.slice_key(96, 2, 0x3))
        self.assertEqual(harness.slice_key(84, 2, 0x7), (42, 3, 2, False))

    def line(self, flags, batches, pnht, capacity=33024, width=None, cb=None):
        return ('Op | INFO | [QWEN-SDPA] flags=0x%x B=%d PNHt=%d St=%d mask_width_t=%d kv_share=%s scratch_slots=4 '
                'cb_bytes=%d\n' % (flags, batches, pnht, capacity // 32, (width or capacity) // 32,
                                   'true' if flags & 2 and batches > 1 else 'false',
                                   model.cb_bytes(capacity, pnht) if cb is None else cb))

    def test_the_log_check(self):
        requested = {harness.program_key(33024, 2, 96, 33024, 0x7), harness.program_key(33024, 2, 96, 33024, 0x3)}
        slices = {harness.slice_key(96, 2, 0x7)}
        good = self.line(0x7, 2, 2) + self.line(0x3, 2, 3) + '[QWEN-SDPA] q-slice rows_per_kv=48 pnht_full=3 slice_tiles=2 readahead=false\n'
        self.assertEqual(harness.check_logs(good, requested, slices), [])
        self.assertEqual(harness.check_logs(good + good, requested, slices), [])          # repeats (cache off)
        missing = harness.check_logs(self.line(0x3, 2, 3), requested, slices)
        self.assertTrue(any('flags=0x7: no [QWEN-SDPA] flags= line' in problem for problem in missing), missing)
        self.assertTrue(any('no [QWEN-SDPA] q-slice line' in problem for problem in missing), missing)
        wrong = harness.check_logs(self.line(0x7, 2, 2, cb=1) + self.line(0x3, 2, 3)
                                   + '[QWEN-SDPA] q-slice rows_per_kv=48 pnht_full=3 slice_tiles=2 readahead=false\n',
                                   requested, slices)
        self.assertTrue(any("'cb_bytes': (1, 790592)" in problem for problem in wrong), wrong)
        stray = harness.check_logs(good + self.line(0xF, 2, 2)
                                   + '[QWEN-SDPA] q-slice rows_per_kv=48 pnht_full=3 slice_tiles=2 readahead=true\n',
                                   requested, slices)
        self.assertEqual(len(stray), 2, stray)

    def rows(self, served, k1a, k1b, k1ab, capacity=131328):
        out = []
        for flags, us in (('0x3', served), ('0x7', k1a), ('0xb', k1b), ('0xf', k1ab)):
            out.append(dict(capacity=capacity, shape='G8B2', flags=flags, eager=dict(median_us=us),
                            trace=dict(median_us=us * 0.95)))
        return out

    def test_the_timing_verdict_at_every_edge(self):
        for k1ab, expected in ((0.78 * 800, 'ACCEPT'), (0.7801 * 800, 'BETWEEN'), (0.90 * 800, 'BETWEEN'),
                               (0.9001 * 800, 'STOP')):
            verdict = harness.analyse_timing(self.rows(800.0, 900.0, 850.0, k1ab))['verdict']['eager']
            self.assertEqual((verdict['acceptance'], verdict['best']), (expected, 'k1ab'), k1ab)
        verdict = harness.analyse_timing(self.rows(800.0, 600.0, 850.0, 620.0))['verdict']['eager']
        self.assertEqual((verdict['best'], verdict['acceptance']), ('k1a', 'ACCEPT'))
        self.assertAlmostEqual(verdict['saving_ms_per_replay'], 200.0 * 64 / 1000)
        self.assertAlmostEqual(verdict['k1b_ratio'], 850.0 / 800.0)
        self.assertAlmostEqual(verdict['served_over_probe'], 800.0 / 796.5)
        self.assertEqual(harness.analyse_timing(self.rows(800.0, 600.0, 850.0, 620.0, capacity=33024))['verdict']['eager']
                         ['acceptance'], 'none')
        report = dict(role='candidate', passed=True, cases=[1], failures=[],
                      timing_analysis=harness.analyse_timing(self.rows(796.5, 650.0, 700.0, 600.0)))
        line = harness.verdict_line(report)
        self.assertTrue(line.startswith('K1_CARD role=candidate exact=PASS cases=1 failures=0 timing=ACCEPT basis=eager '
                                        'cap=131328 served_0x3=796.5us k1a_0x7=650.0us(r=0.816) k1b_0xB=700.0us(r=0.879) '
                                        'k1ab_0xF=600.0us(r=0.753) saving=196.5us/call replay=-12.6ms '), line)
        self.assertIn('probe_k1a=696.3us(r=0.874)', line)
        self.assertTrue(line.endswith(' reference=none'), line)       # exact=PASS without the served bytes
        report['reference'] = '/results/reference-20260925T010203.json'
        self.assertTrue(harness.verdict_line(report).endswith(' reference=reference-20260925T010203.json'))
        self.assertEqual(harness.verdict_line(dict(role='reference', passed=True, cases=[1], failures=[])),
                         'K1_CARD role=reference exact=PASS cases=1 failures=0 timing=none')

    def test_the_arguments(self):
        args = harness.parse_args(['--out', 'x.json'])
        self.assertEqual((args.capacities, args.timing_capacities, args.role, args.shapes),
                         ([2304, 33024, 66048, 131328], [2304, 33024, 66048, 131328], 'candidate', list(harness.SHAPES)))
        self.assertEqual((args.seeds, args.variants, args.starts), ([0, 1, 2, 3, 4], ['normal', 'peaky', 'zeroq'], [0, 7, 240]))
        self.assertEqual([probe.busiest_chunks(capacity) for capacity in args.capacities], [1, 9, 17, 33])
        for bad in (['--shapes', 'G9B2'], ['--capacities', '512'], ['--starts', '241'], ['--role', 'reference', '--reference', 'r.json'],
                    ['--expect-binary-sha256', 'abc']):
            with self.subTest(bad=bad), mock.patch('sys.stderr'), self.assertRaises(SystemExit):
                harness.parse_args(['--out', 'x.json'] + bad)
        # --reference: only a passing reference-role report with shas is the served bytes.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reference.json'
            good = dict(role='reference', passed=True, shas={'G8B2/x/legacy': '0' * 64}, binary=dict(sha256='ab' * 32))
            path.write_text(json.dumps(good))
            args = harness.parse_args(['--out', 'x.json', '--reference', str(path)])
            self.assertEqual((args.reference_shas, args.reference_binary), (good['shas'], 'ab' * 32))
            for bad in (dict(good, passed=False), dict(good, role='candidate'), dict(good, shas={}),
                        {key: value for key, value in good.items() if key != 'passed'}):
                path.write_text(json.dumps(bad))
                with self.subTest(bad=sorted(bad.items())), mock.patch('sys.stderr'), self.assertRaises(SystemExit):
                    harness.parse_args(['--out', 'x.json', '--reference', str(path)])


# ---------------------------------------------------------------------------------------------
# The runner (bash).
# ---------------------------------------------------------------------------------------------

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
         'KOPGRAFT64_REFERENCE', 'K64I_DRY_RUN', 'EXPECT_TTNNCPP_SHA256', 'CARD_B_ARGS', 'MSYS')


def make_graft(root, name, *, binary, slice_kernels):
    """A graft directory shaped like ~/opgraft-K64g (slice_kernels=False) or ~/opgraft-K64i (True)."""
    graft = Path(root) / name
    kernels_dir = graft / 'sdpa_decode' / 'device' / 'kernels'
    (kernels_dir / 'dataflow').mkdir(parents=True)
    (kernels_dir / 'compute').mkdir(parents=True)
    for op in ('attn_prep', 'nlp_concat_heads_decode', 'sdpa'):
        (graft / op).mkdir()
        (graft / op / 'placeholder.txt').write_bytes(op.encode())
    (graft / '_ttnn.so').write_bytes(b'fake _ttnn.so')
    (graft / '_ttnncpp.so').write_bytes(binary)
    shutil.copyfile(STAGE3 / 'reader_decode_qwen.cpp', kernels_dir / 'dataflow' / 'reader_decode_qwen.cpp')
    shutil.copyfile(STAGE3 / 'sdpa_flash_decode_qwen.cpp', kernels_dir / 'compute' / 'sdpa_flash_decode_qwen.cpp')
    if slice_kernels:
        for name in (kernels.READER_SLICE_NAME, kernels.WRITER_SLICE_NAME):
            shutil.copyfile(HERE / name, kernels_dir / 'dataflow' / name)
    write_manifest(graft)
    return graft


def write_manifest(graft):
    files = sorted('./' + path.relative_to(graft).as_posix() for path in graft.rglob('*')
                   if path.is_file() and path.name != 'MANIFEST.sha256')
    (graft / 'MANIFEST.sha256').write_bytes(''.join('%s  %s\n' % (sha((graft / name).read_bytes()), name)
                                                    for name in files).encode())


K64I_BINARY = b'fake K64i _ttnncpp.so [QWEN-SDPA] q-slice rows_per_kv={} pnht_full={}'
K64G_BINARY = b'fake K64g _ttnncpp.so'


@unittest.skipUnless(BASH, 'bash not found')
class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_runner(self, *args, **env):
        environ = {key: value for key, value in os.environ.items() if key not in SCRUB}
        environ.update(HOME=self.dir.as_posix(), RESULTS=(self.dir / 'results').as_posix(), K64I_DRY_RUN='1')
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
                out.append(dict(part.split('=', 1) if '=' in part else (part, True) for part in argv[index + 1].split(',')))
        return out

    def arm_graft_mounts(self):
        """What lever_n_m3native_run_arm.sh mounts from KOPGRAFT64 for the gate's argv (M3NATIVE_SDPA_PF=1)."""
        arm = read(ARM)
        pairs = set(re.findall(r'-v \$KOPGRAFT64/([A-Za-z0-9_.]+):(/opt/[^:" ]+):ro', arm))
        target = re.search(r'^\s*sdpa_pf_target=(\S+)$', arm, flags=re.M).group(1)
        pairs.add(('sdpa', target))
        self.assertEqual(len(pairs), 7)
        return pairs

    def check_launch(self, result, graft, role):
        argv = self.argv(result)
        self.assertEqual(result.stderr, '')
        self.assertEqual(argv[:3], ['docker', 'run', '--rm'])
        self.assertEqual(argv[argv.index('--device') + 1], '/dev/tenstorrent/by-id/' + CARD_B)
        self.assertEqual(argv.count('--device'), 1)
        self.assertEqual(argv[argv.index('--name') + 1], 'qwen-k64i-card-b')
        self.assertIn('card=%s (card-b)' % CARD_B, result.stdout)
        mounts = self.mounts(argv)
        prefix = graft.as_posix() + '/'
        grafted = {(m['src'][len(prefix):], m['dst']) for m in mounts if m['src'].startswith(prefix)}
        self.assertEqual(grafted, self.arm_graft_mounts())
        self.assertTrue(all(m.get('readonly') for m in mounts if m['src'].startswith(prefix)))
        bench = {m['dst']: m['src'] for m in mounts if m['dst'].startswith('/bench/')}
        self.assertEqual(sorted(bench), ['/bench/probe_k1_card_b.py', '/bench/sdpa_decode_slice_card_b.py',
                                         '/bench/slice_index_model.py', '/bench/test_sdpa_decode_qwen_card_m.py'])
        for dst, src in bench.items():
            home = 'sdpa_decode_slice' if dst in ('/bench/sdpa_decode_slice_card_b.py', '/bench/slice_index_model.py')                 else 'sdpa_decode_qwen'
            self.assertTrue(src.endswith('/optimisation/ttnn-op/%s/%s' % (home, Path(dst).name)), src)
        cache = [m for m in mounts if m['dst'] == '/kcache']
        self.assertEqual(len(cache), 1)
        self.assertRegex(cache[0]['src'], r'/results/kcache-%s-[0-9]{8}T[0-9]{6}$' % role)
        env = [argv[i + 1] for i, word in enumerate(argv) if word == '-e']
        for value in ('QWEN_SDPA_TREE_SCRATCH_ROUNDS=1', 'TT_METAL_CACHE=/kcache', 'TT_METAL_HOME=/opt/tt-metal'):
            self.assertIn(value, env)
        entry = argv.index('--entrypoint')
        self.assertEqual(argv[entry + 1:entry + 4],
                         ['sh', 'sha256:70c27e7539db757e3b166f0d14ccdc32edf86532017b6f2f50310fcaad855c3f', '-c'])
        self.assertIn('exec python3 -B /bench/sdpa_decode_slice_card_b.py "$@"', argv[entry + 4])
        self.assertEqual(argv[entry + 5], role)                  # $0 of sh -c
        return argv, harness.parse_args(argv[entry + 6:])

    def test_the_candidate_dry_run(self):
        graft = make_graft(self.dir, 'k64i', binary=K64I_BINARY, slice_kernels=True)
        result = self.run_runner('candidate', KOPGRAFT64=graft.as_posix())
        argv, args = self.check_launch(result, graft, 'candidate')
        self.assertIn('### graft %s (candidate): _ttnncpp.so %s, 4 qwen kernels as recorded, manifest verified'
                      % (graft.as_posix(), sha(K64I_BINARY)[:16]), result.stdout)
        self.assertEqual((args.role, args.expect_binary_sha256, args.watchdog, args.no_timing, args.controls, args.reference),
                         ('candidate', '', 300.0, False, True, None))
        self.assertIn('no reference report in', result.stdout)
        self.assertRegex(args.out.as_posix(), r'^/results/candidate-[0-9]{8}T[0-9]{6}\.json$')
        # With a reference report in the results directory the newest is passed.
        (self.dir / 'results').mkdir(exist_ok=True)
        (self.dir / 'results' / 'reference-20260925T000000.json').write_text('{}')
        result = self.run_runner(KOPGRAFT64=graft.as_posix())
        argv = self.argv(result)
        self.assertEqual(argv[argv.index('--reference') + 1], '/results/reference-20260925T000000.json')
        self.assertIn('legacy and 0x0-0x3 outputs are compared against', result.stdout)

    def test_the_reference_dry_run_and_the_watcher_pass(self):
        graft = make_graft(self.dir, 'k64g', binary=K64G_BINARY, slice_kernels=False)
        result = self.run_runner('reference', KOPGRAFT64_REFERENCE=graft.as_posix(), EXPECT_TTNNCPP_SHA256=sha(K64G_BINARY))
        _argv, args = self.check_launch(result, graft, 'reference')
        self.assertEqual((args.role, args.controls, args.no_timing, args.expect_binary_sha256),
                         ('reference', False, True, sha(K64G_BINARY)))
        result = self.run_runner('candidate', WATCHER='1', CARD_B_ARGS='--seeds 0,1')
        argv = self.argv(result)
        self.assertIn('does not exist here; the graft was not checked', result.stdout)
        env = [argv[i + 1] for i, word in enumerate(argv) if word == '-e']
        self.assertIn('TT_METAL_WATCHER=5', env)
        self.assertTrue(any(m['dst'] == '/opt/tt-metal/generated/watcher' for m in self.mounts(argv)))
        args = harness.parse_args(argv[argv.index('--entrypoint') + 6:])
        self.assertEqual((args.capacities, args.seeds, args.variants, args.starts, args.no_timing, args.watchdog,
                          args.alternations, args.trace_soak),
                         ([2304, 33024], [0, 1], ['normal', 'zeroq'], [0, 240], True, 120.0, 100, 50))
        self.assertIn('WATCHER=1: TT_METAL_WATCHER=5, per-call watchdog 120 s, container timeout 1800 s', result.stdout)

    def refused(self, result, needle):
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn(needle, result.stderr)
        self.assertNotIn('### argv: ', result.stdout)

    def test_the_graft_checks(self):
        graft = make_graft(self.dir, 'k64i', binary=K64I_BINARY, slice_kernels=True)
        env = dict(KOPGRAFT64=graft.as_posix())
        writer = graft / 'sdpa_decode' / 'device' / 'kernels' / 'dataflow' / kernels.WRITER_SLICE_NAME
        writer.write_bytes(writer.read_bytes() + b'// changed\n')
        self.refused(self.run_runner(**env), 'refusing: %s/MANIFEST.sha256 does not verify' % graft.as_posix())
        write_manifest(graft)
        self.refused(self.run_runner(**env), 'refusing: %s is %s, not the recorded %s'
                     % (writer.as_posix(), sha(writer.read_bytes()), kernels.OUTPUTS[kernels.WRITER_SLICE_NAME]))
        shutil.copyfile(HERE / kernels.WRITER_SLICE_NAME, writer)
        (graft / '_ttnncpp.so').write_bytes(K64G_BINARY)
        write_manifest(graft)
        self.refused(self.run_runner(**env), "lacks '[QWEN-SDPA] q-slice rows_per_kv=' (not a K64i build)")
        self.refused(self.run_runner(**env, EXPECT_TTNNCPP_SHA256='f' * 64), 'refusing: %s/_ttnncpp.so is %s, not ffffffffffffffff'
                     % (graft.as_posix(), sha(K64G_BINARY)[:16]))
        reference = make_graft(self.dir, 'k64g', binary=K64I_BINARY, slice_kernels=False)
        self.refused(self.run_runner('reference', KOPGRAFT64_REFERENCE=reference.as_posix(), EXPECT_TTNNCPP_SHA256=''),
                     'the reference must be the served K64g')
        self.refused(self.run_runner('reference', KOPGRAFT64_REFERENCE=reference.as_posix()),
                     'not 134bc8347d6533b3 (EXPECT_TTNNCPP_SHA256= skips)')
        shutil.rmtree(graft / 'sdpa')
        self.refused(self.run_runner(**env), 'refusing: %s/sdpa missing' % graft.as_posix())

    def test_the_serving_pair_and_an_unknown_role_are_refused(self):
        self.refused(self.run_runner(QUAL_CARD=CARD_M), 'refusing: QUAL_CARD=%s is card M' % CARD_M)
        self.refused(self.run_runner('probe'), 'unknown role probe')

    def test_the_runner_echoes_the_verdict_line_not_the_summary_line(self):
        lines = [line for line in read(RUNNER).splitlines() if line.startswith('echo ') and 'K1_CARD' in line]
        self.assertEqual(len(lines), 1, lines)
        log = self.dir / 'run.log'
        log.write_bytes(b'x\nK1_CARD role=candidate exact=PASS timing=ACCEPT\nSDPA_K1_CARD passed=True\n')
        result = subprocess.run([BASH, '-c', 'set -o pipefail; log=%s; %s' % (shlex.quote(log.as_posix()), lines[0])],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.stdout, '### K1_CARD role=candidate exact=PASS timing=ACCEPT' + NL, result.stderr)
        log.write_bytes(b'SDPA_K1_CARD passed=False\n')
        result = subprocess.run([BASH, '-c', 'set -o pipefail; log=%s; %s' % (shlex.quote(log.as_posix()), lines[0])],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.stdout, '### no K1_CARD line' + NL, result.stderr)

    def test_a_real_run_resolves_the_board_before_anything_else(self):
        if Path('/dev/tenstorrent/by-id', CARD_B).exists():
            self.skipTest('card B is present on this host: the qualification would run')
        result = self.run_runner(K64I_DRY_RUN='0')
        self.refused(result, 'refusing: %s (card B, the qualification card) has no device node here' % CARD_B)
        self.assertFalse((self.dir / 'results').exists())


# ---------------------------------------------------------------------------------------------
# The device flow on a fake ttnn whose op is the index model.
# ---------------------------------------------------------------------------------------------

ELEMENT = {'bf16': 2, 'bf8': 1, 'int32': 4}
L1_LIMIT = 1_500_000
# Per-call models (intercept, slope per busiest-core chunk), by the flags' stage-4 bits: 0x3 as served, K1a,
# K1b alone, both (a scenario-B card: 0xF beats 0x7).
MODEL = {0x0: (112.3, 21.32), 0x4: (95.0, 17.8), 0x8: (105.0, 19.5), 0xC: (89.0, 15.6)}


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
    """The ttnn surface the harness touches. DRAM: a size-keyed LIFO free list (a freed address is the next same-size
    allocation's) whose bytes stay stale. The op: validate's and the stage-4 factory's refusals; the L1 limit; the
    F4 and F18 lines once per new program (to fd 1); and the output the kernels write, from
    slice_index_model.output_rows - each (entry, KV head)'s cores compute the Q rows their reader loaded (R7) with
    the mask rows loaded beside them (R8), and the writer copies each kept row from its L1 slot (W2/W3). A row's
    value is row-local: its Q row, its KV head, the page-table row it reads (row 0 for every entry under share)
    and its mask row (the last 256 columns under tail). Injected faults: bugs (the index model's), skip_rows (the
    slice writer skips head 1's last 8 rows of the last entry), reuse=False, silent_slice (no F18 line), lax
    (the factory accepts 0x8 without 0x2)."""

    int32, bfloat16, bfloat8_b = 'int32', 'bf16', 'bf8'
    ROW_MAJOR_LAYOUT, TILE_LAYOUT, DRAM_MEMORY_CONFIG = 'rm', 'tile', 'dram'

    def __init__(self, torch, *, bugs=(), skip_rows=False, reuse=True, silent_slice=False, lax=False, model_us=None):
        self.torch, self.bugs, self.skip_rows, self.reuse = torch, tuple(bugs), skip_rows, reuse
        self.silent_slice, self.lax, self.model_us = silent_slice, lax, dict(MODEL if model_us is None else model_us)
        self.transformer = self
        self.memory, self.free, self.next = {}, {}, 0x10000
        self.now = 0.0
        self.programs, self.traces, self.capturing = set(), {}, None
        self.calls = {}
        self.mask_cache = {}

    def clock(self):
        return self.now

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

    def allocate(self, shape, dtype):
        rows = -(-shape[-2] // 32) * 32 if len(shape) >= 2 else 1
        nbytes = int(self.torch.Size(shape[:-2]).numel()) * rows * shape[-1] * ELEMENT[dtype] if len(shape) >= 2 else 4
        pool = self.free.get(nbytes) if self.reuse else None
        if pool:
            address = pool.pop()
        else:
            address, self.next = self.next, self.next + nbytes
        return FakeTensor(address, shape, dtype, nbytes)

    def from_torch(self, host, *, device, dtype, layout, memory_config):
        tensor = self.allocate(tuple(host.shape), dtype)
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

    def mask_terms(self, attn_mask, tail):
        """Per (entry, row) of a mask: its -inf count and its finite sum, over the last 256 columns under tail
        (cached per uploaded mask: every call of a case reads the same one)."""
        if attn_mask is None:
            return None, None
        host = self.read(attn_mask)
        cached = self.mask_cache.get((attn_mask.address, tail))
        if cached is not None and cached[0] is host:
            return cached[1], cached[2]
        seen = host.float()[:, 0]
        seen = seen[..., -256:] if tail else seen
        infs = self.torch.isinf(seen).sum(dim=-1).float()
        finite = self.torch.where(self.torch.isfinite(seen), seen, self.torch.zeros_like(seen)).sum(dim=-1)
        self.mask_cache[(attn_mask.address, tail)] = (host, infs, finite)
        return infs, finite

    def paged_scaled_dot_product_attention_decode(self, query, k, v, pages, *, is_causal, scale, program_config,
                                                  memory_config, attn_mask=None, cur_pos_tensor=None):
        torch = self.torch
        sentinel = program_config['q_chunk_size']
        qwen = (sentinel & 0xFFFFFF00) == card.MAGIC
        flags = sentinel & 0xFF if qwen else 0
        batches, rows = query.shape[1], query.shape[2]
        table = self.read(pages)
        capacity = table.shape[1] * card.PAGE
        # validate()
        if is_causal:
            if attn_mask is not None:
                raise RuntimeError('TT_FATAL Must not have attn_mask tensor for non-causal attention')
            if cur_pos_tensor is None:
                raise RuntimeError('TT_FATAL Must have cur_pos tensor for paged attention in causal mode')
        elif attn_mask is not None and -(-attn_mask.shape[2] // 32) != -(-rows // 32):
            raise RuntimeError('TT_FATAL Expect same number of padded heads in mask as in Q, got %d and %d'
                               % (attn_mask.shape[2], rows))
        # the factory
        rule = model.slice_rule(rows)
        if qwen:
            if flags & ~0xF:
                raise RuntimeError('TT_FATAL [QWEN-SDPA] unknown flags 0x%x' % flags)
            if flags & 0x8 and not flags & 0x2 and not self.lax:
                raise RuntimeError('TT_FATAL [QWEN-SDPA] KV read-ahead needs KV share (0x2), flags 0x%x' % flags)
            if flags & 0x4 and rule['refusal']:
                raise RuntimeError('TT_FATAL %s' % rule['refusal'])
            if is_causal:
                raise RuntimeError('TT_FATAL [QWEN-SDPA] modes are non-causal, full-window and take no cur_pos tensor')
            if flags & 0x1 and attn_mask.shape[3] not in (capacity, 256):
                raise RuntimeError('TT_FATAL [QWEN-SDPA] tail mask must be full width')
        sliced = bool(qwen and flags & 0x4)
        pnht = rule['slice_tiles'] if sliced else rule['pnht_full']
        if model.cb_bytes(capacity, pnht) > L1_LIMIT:
            raise RuntimeError('TT_THROW Statically allocated circular buffers grow to %d B, beyond max L1 size'
                               % model.cb_bytes(capacity, pnht))
        share = bool(flags & 0x2) and batches > 1
        readahead = bool(flags & 0x8) and share
        if qwen:
            width = attn_mask.shape[3] if attn_mask is not None else capacity
            key = (capacity, batches, rows, width, flags)
            if key not in self.programs:
                self.programs.add(key)
                text = ('Op | INFO | [QWEN-SDPA] flags=0x%x B=%d PNHt=%d St=%d mask_width_t=%d kv_share=%s scratch_slots=4 '
                        'cb_bytes=%d\n' % (flags, batches, pnht, capacity // 32, width // 32, str(share).lower(),
                                           model.cb_bytes(capacity, pnht)))
                if (sliced or readahead) and not self.silent_slice:
                    text += ('Op | INFO | [QWEN-SDPA] q-slice rows_per_kv=%d pnht_full=%d slice_tiles=%d readahead=%s\n'
                             % (rule['rows_per_kv'], rule['pnht_full'], pnht, str(readahead).lower()))
                os.write(1, text.encode())
        self.calls[flags if qwen else None] = self.calls.get(flags if qwen else None, 0) + 1
        intercept, slope = self.model_us[flags & 0xC]
        cost = (intercept + slope * probe.busiest_chunks(capacity)) * (1 + 0.3 * (batches - 2))
        tail = bool(flags & 0x1)
        # A row's value, row-local, with each input in its own elements so bf16 keeps every difference exact:
        # 0-249 the Q row x 0.5 plus the KV head's keys; 250 the -inf count of the mask row it was loaded with
        # (the last 256 columns under tail); 251 that row's finite mask sum / 100 (a +320 spike shows); 252 the
        # KV head; 253 the first page of the page-table row it read (row 0 for every entry under share). The index
        # model says which Q row, mask row and KV head each written row came from; the values are gathered at once.
        q_obj, keys = self.read(query), self.read(k)
        infs, finite = self.mask_terms(attn_mask, tail)
        first_page = torch.tensor([float(table[0 if share else entry][0]) for entry in range(batches)])
        kv = torch.tensor([[float(keys[table[0 if share else entry].long()[:4], head].float().mean()) for head in range(2)]
                           for entry in range(batches)])

        output = self.allocate((1, batches, rows, card.HEAD_DIM), 'bf16')
        stale = self.memory.get(output.address)
        if stale is None or tuple(stale.shape) != output.shape:
            stale = torch.zeros(output.shape, dtype=torch.bfloat16)
        self.memory[output.address] = stale

        def write():
            written = model.output_rows(batches, rows, sliced, lambda *source: source, bugs=self.bugs if sliced else (),
                                        garbage='garbage')
            data = self.memory[output.address].clone()
            targets, sources, garbage = [], [], []
            for (entry, row), source in written.items():
                if sliced and self.skip_rows and entry == batches - 1 and row >= rows - 8:
                    continue
                if source == 'garbage':
                    garbage.append((entry, row))
                else:
                    targets.append((entry, row))
                    sources.append(source)
            values = torch.zeros(len(sources), card.HEAD_DIM)
            if sources:
                entries = torch.tensor([source[0] for source in sources])
                heads = torch.tensor([source[1] for source in sources])
                q_rows = torch.tensor([source[2] for source in sources])
                mask_entries = torch.tensor([source[3][0] for source in sources])
                mask_rows = torch.tensor([source[3][1] for source in sources])
                real = q_rows < rows
                q = q_obj.float()[0][entries, q_rows.clamp(max=rows - 1), :250] * 0.5 + kv[entries, heads][:, None]
                values[:, :250] = torch.where(real[:, None], q, torch.zeros_like(q))
                if infs is not None:
                    visible = real & (mask_entries < batches) & (mask_rows < rows)
                    me, mr = mask_entries.clamp(max=batches - 1), mask_rows.clamp(max=rows - 1)
                    values[:, 250] = torch.where(visible, infs[me, mr], torch.zeros(len(sources)))
                    values[:, 251] = torch.where(visible, finite[me, mr] / 100, torch.zeros(len(sources)))
                values[:, 252] = torch.where(real, heads.float(), torch.zeros(len(sources)))
                values[:, 253] = torch.where(real, first_page[entries], torch.zeros(len(sources)))
                index = torch.tensor(targets)
                data[0, index[:, 0], index[:, 1]] = values.to(torch.bfloat16)
            for entry, row in garbage:
                data[0, entry, row] = 7.0
            self.memory[output.address] = data

        if self.capturing is not None:
            self.traces[self.capturing].append((write, cost))
        else:
            write()
            self.now += cost * 1e-6
        return output


class DryRunTests(unittest.TestCase):
    """The harness end to end on the fake. Capacities 768 / 1,024 / 2,304 with one core per head give 3 / 4 / 9
    busiest-core chunks at a fraction of the host memory; the decision is read at 2,304."""

    SMALL = ['--capacities', '768,2304', '--seeds', '0', '--variants', 'normal,zeroq', '--starts', '0,240',
             '--alternations', '3', '--trace-soak', '2', '--warmup', '1', '--iters', '3', '--rounds', '2',
             '--replays', '2', '--trace-calls', '4', '--trace-warmup', '1', '--decision-capacity', '2304']

    def setUp(self):
        import torch
        self.torch = torch
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.kernels = self.dir / 'kernels'
        for name in list(harness.KERNELS) + list(harness.SLICE_KERNELS):
            (self.kernels / name).parent.mkdir(parents=True, exist_ok=True)
            source = STAGE3 / Path(name).name if name in harness.KERNELS else HERE / Path(name).name
            shutil.copyfile(source, self.kernels / name)
        self.k64i = self.dir / 'k64i.so'
        self.k64i.write_bytes(K64I_BINARY)
        self.k64g = self.dir / 'k64g.so'
        self.k64g.write_bytes(K64G_BINARY)

    def tearDown(self):
        self.tmp.cleanup()

    def run_card(self, fake, extra=(), *, binary=None, role='candidate', name='candidate', scratch='1'):
        out = self.dir / ('%s.json' % name)
        binary = binary or (self.k64i if role == 'candidate' else self.k64g)
        markers = dict(flags=True, share=True, stage1=False)
        argv = ['--out', str(out), '--role', role, '--kernel-root', str(self.kernels)] + self.SMALL + list(extra)
        environ = {card.SCRATCH_ENV: scratch} if scratch is not None else {}
        with mock.patch.dict(sys.modules, {'ttnn': fake}), \
                mock.patch.object(card, 'loaded_binary', return_value=(str(binary), markers)), \
                mock.patch.dict(os.environ, environ), \
                mock.patch.multiple(probe, CORES_PER_HEAD=1, clock=fake.clock), \
                mock.patch.object(card, 'WATCHDOG', card.WATCHDOG), mock.patch.object(probe, 'WATCHDOG', probe.WATCHDOG), \
                mock.patch.object(harness, 'WATCHDOG', harness.WATCHDOG), \
                mock.patch.object(harness, 'print', create=True), mock.patch.object(probe, 'print', create=True), \
                mock.patch('sys.stdout'):
            if scratch is None:
                os.environ.pop(card.SCRATCH_ENV, None)
            status = harness.main(argv)
        return status, json.loads(out.read_text())

    def test_the_candidate_passes_end_to_end_and_reads_the_verdict(self):
        fake = FakeTtnn(self.torch)
        status, report = self.run_card(fake)
        self.assertEqual((report.get('error'), report['failures']), (None, []))
        self.assertEqual((status, report['passed']), (0, True))
        # No --reference here: the one warning says the served bytes were not compared, and so does the verdict.
        self.assertEqual(len(report['warnings']), 1, report['warnings'])
        self.assertTrue(report['warnings'][0].startswith('no --reference: legacy and 0x0-0x3 were NOT compared'))
        self.assertTrue(report['verdict_line'].endswith(' reference=none'), report['verdict_line'])
        self.assertEqual(report['kernels']['sha256'], dict(harness.KERNELS, **harness.SLICE_KERNELS))
        # E-S1: 2 capacities x 1 seed x 2 variants x 2 starts, every shape, every mode, both heads equal.
        self.assertEqual(len(report['cases']), 8)
        for case in report['cases']:
            self.assertEqual(sorted(case['shapes']), sorted(harness.SHAPES))
            for shape, result in case['shapes'].items():
                self.assertEqual(sorted(result['modes']), sorted('0x%x' % flags for flags in harness.MODES[shape]))
                for mode in result['modes'].values():
                    self.assertTrue(mode['poison_address_reused'])
                    self.assertEqual(mode['nan_elements'], 0)
                    self.assertIn(mode.get('legacy_differing_per_head', [0, 0]), ([0, 0],))
        self.assertTrue(all(item['differing'] == 0 for item in report['e_s1b']))
        self.assertEqual(len(report['e_s1b']), 8 * 2)
        # The controls at both capacities.
        self.assertEqual(len(report['controls']), 2)
        for control in report['controls']:
            self.assertEqual(len(control['planted']), 6)
            self.assertTrue(all(item['legacy_moved_rows'] == [[item['entry'], item['row']]] for item in control['planted']))
            self.assertEqual(set(control['refusals']), {name for name, *_ in harness.REFUSALS})
            self.assertTrue(all(outcome['refused'] and outcome['matched'] for outcome in control['refusals'].values()))
            self.assertEqual([item['modes_differ'] for item in control['alternations']], [True, False, False])
            self.assertEqual(control['trace']['mismatch_count'], 0)
        # The factory's lines for every program the run asked for, the slice lines included.
        self.assertIn('cap=2304 B=2 PNHt=2 mask_width_t=72 flags=0x7', report['requested_programs'])
        self.assertIn('cap=2304 B=1 PNHt=3 mask_width_t=72 flags=0x5', report['requested_programs'])     # G16: 6 -> 3
        self.assertEqual(report['requested_slice_programs'],
                         sorted(harness.describe_slice(key) for key in {
                             (48, 3, 2, False), (48, 3, 2, True), (48, 3, 3, True), (42, 3, 2, False),
                             (42, 3, 2, True), (42, 3, 3, True), (96, 6, 3, False)}))
        self.assertGreaterEqual(report['slice_line_count'], len(report['requested_slice_programs']))
        # Timing: the model's per-call times; section 7's verdict at 2,304 keys (9 chunks).
        verdict = report['timing_analysis']['verdict']['eager']
        served = 112.3 + 21.32 * 9
        both = 89.0 + 15.6 * 9
        self.assertAlmostEqual(verdict['served_us'], served, places=3)
        self.assertAlmostEqual(verdict['k1ab_us'], both, places=3)
        self.assertEqual((verdict['best'], verdict['acceptance']), ('k1ab', 'ACCEPT' if both / served <= 0.78 else 'BETWEEN'))
        self.assertTrue(report['verdict_line'].startswith('K1_CARD role=candidate exact=PASS cases=8 failures=0 timing='))
        self.assertEqual(fake.options['trace_region_size'], 32 << 20)
        self.assertNotIn(None, [fake.calls.get(flags) for flags in (0x4, 0x5, 0x6, 0x7, 0xA, 0xB, 0xF)])

    def test_the_reference_round_trips_and_a_tampered_one_fails(self):
        status, reference = self.run_card(FakeTtnn(self.torch), ['--no-controls', '--no-timing'], role='reference',
                                          name='reference')
        self.assertEqual((status, reference['failures']), (0, []))
        self.assertTrue(reference['shas'])
        self.assertFalse(any(label.endswith(('/0x7', '/0xf', '/0xb', '/0x4')) for label in reference['shas']))
        path = self.dir / 'reference.json'
        status, report = self.run_card(FakeTtnn(self.torch), ['--reference', str(path), '--no-controls', '--no-timing'])
        self.assertEqual((status, report['failures'], report['warnings']), (0, [], []))
        self.assertEqual({label: report['shas'][label] for label in reference['shas']}, reference['shas'])
        self.assertTrue(report['verdict_line'].endswith(' reference=reference.json'), report['verdict_line'])
        self.assertEqual(report['reference_binary_sha256'], reference['binary']['sha256'])
        tampered = json.loads(path.read_text())
        label = sorted(tampered['shas'])[0]
        tampered['shas'][label] = '0' * 64
        path.write_text(json.dumps(tampered))
        status, report = self.run_card(FakeTtnn(self.torch), ['--reference', str(path), '--no-controls', '--no-timing'],
                                       name='tampered')
        self.assertEqual(status, 1)
        self.assertEqual(report['failures'], ['%s: differs from the reference (K64g) bytes' % label])

    def failures(self, fake, extra=('--no-timing',), **options):
        status, report = self.run_card(fake, list(extra), **options)
        self.assertEqual(status, 1)
        return report['failures']

    def test_a_writer_without_the_w3_rebase_fails_e_s1(self):
        failures = self.failures(FakeTtnn(self.torch, bugs=('w3',)))
        self.assertTrue(any('0x7 differs from legacy (per head [0,' in failure for failure in failures), failures[:5])
        self.assertTrue(any('from its non-slice twin 0x3' in failure for failure in failures))

    def test_both_r8_slips_and_the_r7_slip_fail(self):
        row = self.failures(FakeTtnn(self.torch, bugs=('r8_row',)))
        self.assertTrue(any('the slice read the wrong mask rows (R8)' in failure for failure in row), row[:5])
        self.assertTrue(any(re.search(r'G8B2/.* 0x7 differs from legacy \(per head \[0, [1-9]', failure) for failure in row))
        stride = self.failures(FakeTtnn(self.torch, bugs=('r8_stride',)))
        self.assertTrue(any('planted at entry 1' in failure for failure in stride), stride[:5])
        # The batch-stride slip moves entry 1's mask rows only: the E-S1 differences are all in entry 1's output.
        self.assertTrue(any(re.search(r'G8B2/.* 0x7 differs from legacy', failure) for failure in stride))
        q = self.failures(FakeTtnn(self.torch, bugs=('r7',)))
        self.assertTrue(any('0x4 differs from legacy' in failure for failure in q), q[:5])

    def test_rows_left_unwritten_show_through_the_poison(self):
        failures = self.failures(FakeTtnn(self.torch, skip_rows=True))
        self.assertTrue(any('output elements are NaN - rows the kernels left unwritten (N-S0)' in failure
                            for failure in failures), failures[:5])

    def test_an_output_that_never_takes_the_poisoned_address_fails_the_slice(self):
        failures = self.failures(FakeTtnn(self.torch, reuse=False), ['--no-timing', '--no-controls'])
        self.assertTrue(all('never took the poisoned address' in failure for failure in failures), failures[:5])
        self.assertTrue(all(re.search(r'/0x[4-9a-f]: ', failure) for failure in failures), failures[:5])

    def test_a_graft_without_its_q_slice_line_was_not_executed(self):
        failures = self.failures(FakeTtnn(self.torch, silent_slice=True), ['--no-timing', '--no-controls'])
        self.assertTrue(failures)
        self.assertTrue(all('no [QWEN-SDPA] q-slice line' in failure for failure in failures), failures)

    def test_a_refusal_the_factory_lacks_fails(self):
        failures = self.failures(FakeTtnn(self.torch, lax=True))
        self.assertTrue(any('read-ahead without share (0x9) was not refused' in failure for failure in failures), failures)

    def test_the_binary_the_kernels_and_the_scratch_are_checked_first(self):
        fake = FakeTtnn(self.torch)
        failures = self.failures(fake, binary=self.k64g)
        self.assertEqual(failures, ["the loaded _ttnncpp.so lacks '[QWEN-SDPA] q-slice rows_per_kv=': it is not K64i "
                                    "(read the launched argv)"])
        self.assertEqual(fake.calls, {})
        failures = self.failures(FakeTtnn(self.torch), binary=self.k64i, role='reference', name='reference')
        self.assertEqual(failures, ['the reference role needs the served K64g, but the loaded binary is a stage-4 build'])
        writer = self.kernels / 'dataflow' / kernels.WRITER_SLICE_NAME
        writer.write_bytes(writer.read_bytes() + b'// changed\n')
        failures = self.failures(FakeTtnn(self.torch))
        self.assertTrue(failures[0].endswith('not the recorded %s' % kernels.OUTPUTS[kernels.WRITER_SLICE_NAME][:16]), failures)
        shutil.copyfile(HERE / kernels.WRITER_SLICE_NAME, writer)
        failures = self.failures(FakeTtnn(self.torch), scratch=None)
        self.assertIn('QWEN_SDPA_TREE_SCRATCH_ROUNDS=1 is required', failures[0])


if __name__ == '__main__':
    unittest.main()
