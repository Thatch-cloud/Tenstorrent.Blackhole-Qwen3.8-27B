"""CPU tests for the prefill-chain additions to the M1 bench and runner (sdpa-prefill-share-spec.md
3.7 / 5.3 item 8; optimisation/ttnn-op/sdpa_prefill_chain): the chain arms and --program-word words
reach SDPAProgramConfig.max_cores_per_head_batch, --q-memory / --page-blocks / --verify-log /
the Q2 verdict, and run_m1.sh's KOPGRAFT_PF / WATCHER / QWEN_SDPA_PF_TEST dry runs. No ttnn."""

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
CHAIN_DIR = HERE.parent / 'sdpa_prefill_chain'
for path in (str(HERE), str(CHAIN_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import apply_factory_pf as factory  # noqa: E402
import make_pf_reader  # noqa: E402
import sdpa_prefill_bench as bench  # noqa: E402
from test_k0_readers import BASH, posix, run_bash  # noqa: E402

try:
    import torch
except ImportError:  # pragma: no cover - CI installs torch
    torch = None

RUN_M1 = HERE / 'run_m1.sh'
IMAGE_A2 = 'sha256:eceb2daa744c3345368a638488a804f0ffe8b76f6b0680945a47bb527bdb55a9'
IMAGE_M1 = 'sha256:0648ca9ad663acc72e7d8ea59d9cde0f9218b583ad74a60d58ff2f31bddd6fae'


class ArmTests(unittest.TestCase):
    def test_chain_arms_are_the_baseline_plus_one_word(self):
        base = bench.arm_by_name('baseline')
        words = dict(chain=0x5EFA0001, chain_b=0x5EFA0003, chain_o=0x5EFA0005, chain_bo=0x5EFA0007)
        self.assertEqual(bench.CHAIN_ARM_NAMES, tuple(words))
        for name, word in words.items():
            arm = bench.arm_by_name(name)
            self.assertEqual(arm.pop('program_word'), word)
            self.assertEqual({key: value for key, value in arm.items() if key != 'name'},
                             {key: value for key, value in base.items() if key != 'name'})
            self.assertEqual(factory.decode_word(word), (True, word & 0xFFFF))
        self.assertNotIn('chain', bench.ARM_NAMES)                   # never in the default --arms
        self.assertEqual(bench.PF_TAG, factory.PF_TAG)

    def test_program_word_arms(self):
        arm = bench.word_arm(0x5EFA0203)
        self.assertEqual(arm['name'], 'word_0x5efa0203')
        self.assertEqual(bench.arm_by_name('word_0x5efa0203'), arm)
        with self.assertRaises(ValueError):
            bench.word_arm(1 << 32)
        with self.assertRaisesRegex(ValueError, 'unknown arm'):
            bench.arm_by_name('chainz')

    def test_page_blocks(self):
        arms = [bench.arm_by_name('baseline')]
        self.assertEqual(bench.page_table_blocks(arms, [0, 126976]), 2016)
        self.assertEqual(bench.page_table_blocks(arms, [0, 126976], 2080), 2080)
        for bad in (2000, 2081):
            with self.assertRaisesRegex(ValueError, 'page-blocks'):
                bench.page_table_blocks(arms, [0, 126976], bad)


class Q2Tests(unittest.TestCase):
    def table(self, **slopes):
        return {name: dict(slope_ms_per_1k_keys=slope, intercept_ms=1.2) for name, slope in slopes.items()}

    def shas(self, arms, starts=(0, 128), differ=()):
        return {'%s@%d' % (arm, start): ('bad' if (arm, start) in differ else 'good%d' % start)
                for arm in arms for start in starts}

    def test_the_rules(self):
        arms = ('baseline', 'chain', 'chain_b', 'chain_o')
        q2 = bench.q2_verdict(self.table(baseline=0.2838, chain=0.2200, chain_b=0.2210, chain_o=0.2150), bench.K0B32_RATIO,
                              self.shas(arms))
        self.assertEqual(q2['best'], 'chain_o')
        self.assertTrue(q2['passed'])
        self.assertEqual(q2['rules'], dict(a=True, b=True, c=True, d=True))
        self.assertAlmostEqual(q2['step_us']['baseline'], 0.2838 * 64)
        self.assertEqual(q2['beat_chain_by_2pct'], ['chain_o'])
        self.assertIn('M1 Q2: PASS | best=chain_o', bench.q2_line(q2))
        self.assertIn('d(exact)=1', bench.q2_line(q2))
        slow = bench.q2_verdict(self.table(baseline=0.2838, chain=0.2400), bench.K0B32_RATIO,
                                self.shas(('baseline', 'chain')))
        self.assertEqual(slow['rules'], dict(a=False, b=True, c=True, d=True))  # 0.2400/0.2838 > 1.10 x 0.746
        self.assertFalse(slow['passed'])

    def test_q2_never_passes_an_inexact_or_unchecked_chain(self):
        """Review 2 L3: rule (d), every production chain arm's output equals the baseline's."""
        table = self.table(baseline=0.2838, chain=0.2200, chain_o=0.2150)
        wrong = bench.q2_verdict(table, bench.K0B32_RATIO, self.shas(('baseline', 'chain', 'chain_o'),
                                                                      differ={('chain_o', 128)}))
        self.assertEqual(wrong['rules']['d'], False)
        self.assertFalse(wrong['passed'])
        self.assertIn('chain_o@128 differs from the baseline output', wrong['exact_problems'])
        self.assertIn('M1 Q2: FAIL', bench.q2_line(wrong))
        self.assertIn('d(exact)=0', bench.q2_line(wrong))
        unchecked = bench.q2_verdict(table)
        self.assertIsNone(unchecked['rules']['d'])
        self.assertFalse(unchecked['passed'])
        self.assertIn('d(exact)=-', bench.q2_line(unchecked))
        unstable = bench.q2_verdict(table, bench.K0B32_RATIO, self.shas(('baseline', 'chain', 'chain_o')),
                                    {'chain@0': False})
        self.assertFalse(unstable['passed'])
        missing = bench.q2_verdict(table, bench.K0B32_RATIO, self.shas(('baseline', 'chain')))
        self.assertIn('chain_o@0: no digest', missing['exact_problems'])
        self.assertIsNone(bench.q2_verdict(self.table(baseline=0.28)))
        self.assertIsNone(bench.q2_verdict(self.table(chain=0.22)))
        tested = bench.q2_verdict(dict(self.table(baseline=0.28, chain=0.25), **{'word_0x5efa0203': dict(
            slope_ms_per_1k_keys=0.01, intercept_ms=1.0)}), bench.K0B32_RATIO, self.shas(('baseline', 'chain')))
        self.assertEqual(tested['best'], 'chain')                     # test-flag words never count
        self.assertTrue(tested['rules']['d'])                         # nor does their (different) output

    # Q2 on card M (graft K64g, Q in L1, the model's 2080-block page table), as per-step times in us.
    CARD_M_Q2_STEP_US = dict(baseline=21.4, chain=20.7, chain_b=16.5, chain_o=22.3, chain_bo=18.9)

    def test_rule_a_compares_ratios_to_each_runs_own_baseline(self):
        """K0 (stock 0.2838) and Q2 (baseline 0.335) ran different bench configs. The absolute rule
        failed Q2's 0x3 arm (16.5 us = 0.2578 > 1.10 x 0.2116 = 0.2328) though it cut the step to
        0.770x its own baseline - inside 1.10 x K0b32 / stock = 1.10 x 0.746 = 0.820."""
        self.assertAlmostEqual(bench.K0B32_RATIO, 0.2116 / 0.2838)
        self.assertAlmostEqual(round(bench.K0B32_RATIO, 3), 0.746)
        slopes = {name: step / bench.STEP_US_PER_SLOPE for name, step in self.CARD_M_Q2_STEP_US.items()}
        arms = tuple(slopes)
        q2 = bench.q2_verdict(self.table(**slopes), bench.K0B32_RATIO, self.shas(arms))
        self.assertEqual(q2['best'], 'chain_b')
        self.assertAlmostEqual(q2['best_over_baseline'], 16.5 / 21.4)
        self.assertAlmostEqual(q2['a_limit'], 1.10 * 0.2116 / 0.2838)
        self.assertEqual(q2['rules'], dict(a=True, b=True, c=True, d=True))
        self.assertTrue(q2['passed'])
        self.assertGreater(q2['best_slope'], 1.10 * bench.K0B32_SLOPE)   # the old absolute rule would FAIL it
        self.assertEqual(q2['beat_chain_by_2pct'], ['chain_b', 'chain_bo'])
        self.assertEqual((q2['production'], q2['production_flags']), ('chain_b', 0x3))   # the production default
        self.assertAlmostEqual(q2['step_us']['chain_b'], 16.5)
        line = bench.q2_line(q2)
        self.assertIn('M1 Q2: PASS | best=chain_b', line)
        self.assertIn('(x0.771 baseline; a: <= 0.820 = 1.10 x K0b32/stock 0.746)', line)
        self.assertIn('production=chain_b(0x3)', line)

    def test_rule_a_is_scale_free_and_still_bites(self):
        """Scaling every slope of a run by the same factor (another bench config) cannot change (a);
        a chain arm whose step is over 1.10 x 0.746 of its baseline still fails it."""
        slopes = {name: step / bench.STEP_US_PER_SLOPE for name, step in self.CARD_M_Q2_STEP_US.items()}
        for scale in (0.5, 0.2838 / 0.335, 3.0):
            with self.subTest(scale=scale):
                q2 = bench.q2_verdict(self.table(**{k: v * scale for k, v in slopes.items()}), bench.K0B32_RATIO,
                                      self.shas(tuple(slopes)))
                self.assertTrue(q2['rules']['a'])
        slow = bench.q2_verdict(self.table(baseline=0.335, chain=0.335 * 0.83), bench.K0B32_RATIO,
                                self.shas(('baseline', 'chain')))
        self.assertEqual(slow['rules'], dict(a=False, b=True, c=True, d=True))
        self.assertEqual((slow['production'], slow['production_flags']), ('chain', 0x1))
        edge = bench.q2_verdict(self.table(baseline=1.0, chain=1.10 * bench.K0B32_RATIO), bench.K0B32_RATIO,
                                self.shas(('baseline', 'chain')))
        self.assertTrue(edge['rules']['a'])
        with self.assertRaisesRegex(ValueError, 'must be positive'):
            bench.q2_verdict(self.table(baseline=1.0, chain=0.5), 0.0)

    def test_the_production_choice_needs_a_2pct_win_over_0x1(self):
        q2 = bench.q2_verdict(self.table(baseline=0.335, chain=0.250, chain_b=0.246, chain_bo=0.240),
                              bench.K0B32_RATIO, self.shas(('baseline', 'chain', 'chain_b', 'chain_bo')))
        self.assertEqual(q2['beat_chain_by_2pct'], ['chain_bo'])       # 0x3 only 1.6% faster than 0x1
        self.assertEqual((q2['production'], q2['production_flags']), ('chain_bo', 0x7))
        alone = bench.q2_verdict(self.table(baseline=0.335, chain_b=0.25), bench.K0B32_RATIO,
                                 self.shas(('baseline', 'chain_b')))
        self.assertEqual((alone['production'], alone['production_flags']), ('chain_b', 0x3))   # no 0x1 arm timed

    def test_0x7_replaces_0x3_only_with_a_2pct_win(self):
        """Spec 6.3: '0x7 over 0x3 only if its slope is >= 2% lower'. Both beat 0x1 by >= 2% here; a 0x7
        only 1.25% under 0x3 is the fastest arm but not the production choice (the old fastest-of-those
        rule chose it), a 4.2% one is. 0x3 and 0x5 within 2% of each other: the lower flags."""
        arms = ('baseline', 'chain', 'chain_b', 'chain_bo')
        near = bench.q2_verdict(self.table(baseline=0.335, chain=0.250, chain_b=0.240, chain_bo=0.237),
                                bench.K0B32_RATIO, self.shas(arms))
        self.assertEqual(near['beat_chain_by_2pct'], ['chain_b', 'chain_bo'])
        self.assertEqual(near['best'], 'chain_bo')
        self.assertEqual((near['production'], near['production_flags']), ('chain_b', 0x3))
        self.assertIn('production=chain_b(0x3)', bench.q2_line(near))
        far = bench.q2_verdict(self.table(baseline=0.335, chain=0.250, chain_b=0.240, chain_bo=0.230),
                               bench.K0B32_RATIO, self.shas(arms))
        self.assertEqual((far['production'], far['production_flags']), ('chain_bo', 0x7))
        pair = bench.q2_verdict(self.table(baseline=0.335, chain=0.250, chain_b=0.240, chain_o=0.238),
                                bench.K0B32_RATIO, self.shas(('baseline', 'chain', 'chain_b', 'chain_o')))
        self.assertEqual((pair['production'], pair['production_flags']), ('chain_b', 0x3))
        flat = bench.q2_verdict(self.table(baseline=0.335, chain=0.240, chain_b=0.237, chain_bo=0.236),
                                bench.K0B32_RATIO, self.shas(arms))
        self.assertEqual((flat['production'], flat['production_flags']), ('chain', 0x1))   # nothing 2% under 0x1


class FakeTensor:
    def __init__(self, host, memory=None):
        self.host, self.memory = host, memory


class FakeDevice:
    def compute_with_storage_grid_size(self):
        return SimpleNamespace(x=11, y=10)


def fake_ttnn(configs, uploads, factory_lines=True):
    """SDPA exact on the chain (the word only names a program); a new chain word writes the
    factory's line to fd 1, as tt-logger does, once per (word) program."""
    built = set()

    def sdpa(input_tensor_q, program_config, chunk_start_idx_tensor=None, chunk_start_idx=None, **_):
        configs.append(program_config)
        word = program_config.get('max_cores_per_head_batch')
        if word is not None and factory_lines and word not in built:
            built.add(word)
            flags = word & 0xFFFF
            os.write(1, ('[QWEN-SDPA-PF] flags=%#x kv_chain=1 chains=16 members=96 order=%s\n'
                         % (flags, 'noc' if flags & 4 else 'raster')).encode())
        start = int(chunk_start_idx_tensor.host[0]) if chunk_start_idx_tensor is not None else chunk_start_idx
        return FakeTensor((input_tensor_q.host.float() * (start + 1)).to(torch.bfloat16))

    def from_torch(host, memory_config=None, **_):
        uploads.append((tuple(host.shape), memory_config))
        return FakeTensor(host, memory_config)

    return SimpleNamespace(
        bfloat8_b='bf8', bfloat16='bf16', int32='int32', TILE_LAYOUT='tile', ROW_MAJOR_LAYOUT='rm',
        DRAM_MEMORY_CONFIG='dram', L1_MEMORY_CONFIG='l1', MathFidelity=SimpleNamespace(HiFi2='hifi2'),
        open_device=lambda device_id, l1_small_size: FakeDevice(), close_device=lambda device: None,
        synchronize_device=lambda device: None, deallocate=lambda tensor: None, from_torch=from_torch,
        to_torch=lambda tensor: tensor.host, WormholeComputeKernelConfig=lambda **kw: kw,
        SDPAProgramConfig=lambda **kw: kw, CoreCoord=lambda x, y: SimpleNamespace(x=x, y=y),
        transformer=SimpleNamespace(chunked_scaled_dot_product_attention=sdpa))


@unittest.skipIf(torch is None, 'torch not installed')
class MainTests(unittest.TestCase):
    def main(self, args, factory_lines=True):
        configs, uploads = [], []
        saved = sys.modules.get('ttnn')
        sys.modules['ttnn'] = fake_ttnn(configs, uploads, factory_lines)
        try:
            with tempfile.TemporaryDirectory() as directory:
                out = Path(directory) / 'm1.json'
                argv = ['--out', str(out), '--starts', '0,128', '--rounds', '2', '--warmup', '1'] + args
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    status = bench.main(argv)
                report = json.loads(out.read_text(encoding='utf-8'))
        finally:
            if saved is None:
                sys.modules.pop('ttnn', None)
            else:
                sys.modules['ttnn'] = saved
        return status, report, buffer.getvalue(), configs, uploads

    def test_the_words_reach_the_program_config_and_the_outputs_match(self):
        status, report, stdout, configs, _ = self.main(['--arms', 'baseline,chain,chain_bo', '--sha',
                                                        '--program-word', '0x5EFA0203'])
        self.assertEqual(status, 0, report.get('error'))
        words = {config.get('max_cores_per_head_batch') for config in configs}
        self.assertEqual(words, {None, 0x5EFA0001, 0x5EFA0007, 0x5EFA0203})
        self.assertEqual([arm['name'] for arm in report['arms']], ['baseline', 'chain', 'chain_bo', 'word_0x5efa0203'])
        for start in (0, 128):
            shas = {report['sha256']['%s@%d' % (name, start)] for name in ('baseline', 'chain', 'chain_bo')}
            self.assertEqual(len(shas), 1)
        self.assertIn('M1 Q2:', stdout)
        self.assertTrue(report['q2']['rules']['d'])                   # --sha: exactness checked end to end
        self.assertIn('chain_bo', report['table'])
        self.assertEqual(report['q_memory'], 'dram')
        self.assertAlmostEqual(report['q2']['k0b32_ratio'], bench.K0B32_RATIO)

    def test_rule_a_takes_its_ratio_from_one_k0_session(self):
        status, report, stdout, _, _ = self.main(['--arms', 'baseline,chain', '--sha', '--k0b32-slope', '0.2',
                                                  '--k0-stock-slope', '0.4'])
        self.assertEqual(status, 0, report.get('error'))
        self.assertAlmostEqual(report['q2']['k0b32_ratio'], 0.5)
        self.assertAlmostEqual(report['q2']['a_limit'], 0.55)
        self.assertIn('1.10 x K0b32/stock 0.500', stdout)
        for bad in (['--k0b32-slope', '0'], ['--k0-stock-slope', '-1']):
            with self.subTest(args=bad):
                with self.assertRaises(SystemExit):
                    with contextlib.redirect_stderr(io.StringIO()):
                        self.main(['--arms', 'baseline'] + bad)

    def test_q_in_l1_and_the_model_page_width(self):
        status, report, _, _, uploads = self.main(['--arms', 'baseline', '--q-memory', 'l1', '--page-blocks', '64'])
        self.assertEqual(status, 0, report.get('error'))
        self.assertIn(((1, 12, 2048, 256), 'l1'), uploads)
        self.assertIn(((64, 2, 64, 256), 'dram'), uploads)
        self.assertIn(((1, 64), 'dram'), uploads)
        self.assertEqual(report['page_blocks'], 64)
        status, report, _, _, _ = self.main(['--arms', 'baseline', '--page-blocks', '40'])
        self.assertEqual(status, 1)
        self.assertIn('page-blocks', report['error'])

    def test_verify_log_counts_one_line_per_chain_program(self):
        status, report, stdout, _, _ = self.main(['--arms', 'baseline,chain,chain_o', '--verify-log'])
        self.assertEqual(status, 0, report.get('pf_log'))
        self.assertEqual([line['flags'] for line in report['pf_log']['lines']], [0x1, 0x5])
        self.assertEqual(report['pf_log']['problems'], [])
        self.assertIn('M1 PF_LOG flags=0x1 chains=16 members=96 order=raster', stdout)
        self.assertIn('M1 PF_LOG ok', stdout)
        status, report, stdout, _, _ = self.main(['--arms', 'baseline,chain', '--verify-log'], factory_lines=False)
        self.assertEqual(status, 1)
        self.assertIn('flags 0x1: 0 factory lines, expected exactly 1', report['pf_log']['problems'])

    def test_check_pf_log(self):
        arms = [bench.arm_by_name('chain')]
        line = dict(flags=1, chains=16, members=96, order='raster')
        self.assertEqual(bench.check_pf_log([line], arms), [])
        self.assertTrue(bench.check_pf_log([line, line], arms))                           # rebuilt
        self.assertTrue(bench.check_pf_log([line, dict(line, flags=3)], arms))            # unrequested
        self.assertTrue(bench.check_pf_log([dict(line, chains=8, members=48)], arms))     # wrong topology
        self.assertTrue(bench.check_pf_log([dict(line, order='noc')], arms))              # 0x4 not set


@unittest.skipUnless(BASH, 'bash not found')
class RunnerTests(unittest.TestCase):
    def graft(self, root, chain=True):
        g = Path(root) / 'graft'
        (g / 'sdpa/device/kernels/dataflow').mkdir(parents=True)
        for op in ('attn_prep', 'sdpa_decode', 'nlp_concat_heads_decode'):
            (g / op).mkdir()
        (g / '_ttnn.so').write_bytes(b'so')
        (g / '_ttnncpp.so').write_bytes(b'cpp')
        if chain:
            (g / 'sdpa/device/kernels/dataflow/reader_interleaved_qwen_chain.cpp').write_bytes(b'r')
        return g

    def dry(self, **env):
        with tempfile.TemporaryDirectory() as directory:
            base = dict(M1_DRY_RUN='1', M1_SRC=posix(HERE), RESULTS=posix(Path(directory) / 'r'),
                        M1_ARGS='--arms baseline,chain --sha', KOPGRAFT_PF='', WATCHER='', QWEN_SDPA_PF_TEST='')
            base.update({key: (value(directory) if callable(value) else value) for key, value in env.items()})
            return run_bash(RUN_M1, base)

    def argv(self, result):
        return [line for line in result.stdout.splitlines() if line.startswith('### argv:')][0]

    def test_kopgraft_pf_mounts_the_graft_like_the_arm(self):
        result = self.dry(KOPGRAFT_PF=lambda d: posix(self.graft(d)), IMAGE=IMAGE_A2)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('-e M1_REQUIRE_SOURCES=1', self.argv(result))   # the seven sources are enforced
        argv = self.argv(result)
        ops = '/opt/tt-metal/ttnn/cpp/ttnn/operations/'
        for dst in ('/opt/tt-metal/ttnn/ttnn/_ttnn.so', '/opt/tt-metal/build_Release/ttnn/_ttnncpp.so',
                    '/opt/tt-metal/build_Release/lib/_ttnncpp.so', ops + 'transformer/attn_prep',
                    ops + 'experimental/transformer/nlp_concat_heads_decode', ops + 'transformer/sdpa_decode',
                    ops + 'transformer/sdpa'):
            self.assertIn('dst=%s\\,readonly' % dst, argv)
        self.assertIn('/results/m1-', argv)
        self.assertIn('graft.sha256', argv)                           # the in-container .so check
        self.assertIn('reader_interleaved_qwen_chain.cpp', argv)      # the chain reader joins the source list
        self.assertNotIn('TT_METAL_WATCHER', argv)
        self.assertNotIn('QWEN_SDPA_PF_TEST', argv)

    def test_the_chain_reader_sha_is_the_recorded_output(self):
        text = RUN_M1.read_text(encoding='utf-8')
        self.assertIn('CHAIN_READER_SHA=%s' % make_pf_reader.READER_OUTPUT, text)
        self.assertNotIn(chr(13), text)

    def test_refusals(self):
        both = self.dry(KOPGRAFT_PF=lambda d: posix(self.graft(d)),
                        M1_READER=posix(CHAIN_DIR / 'reader_interleaved_qwen_chain.cpp'))
        self.assertNotEqual(both.returncode, 0)
        self.assertIn('M1_READER and KOPGRAFT_PF together', both.stderr)
        missing = self.dry(KOPGRAFT_PF=lambda d: posix(self.graft(d, chain=False)), IMAGE=IMAGE_A2)
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn('build it with build_k64g.sh', missing.stderr)

    def test_the_graft_runs_only_in_an_image_the_build_compared(self):
        """Review 2 M5: KOPGRAFT_PF on run_m1.sh's 0648ca9a default was accepted with a warning only."""
        implicit = self.dry(KOPGRAFT_PF=lambda d: posix(self.graft(d)))
        self.assertNotEqual(implicit.returncode, 0)
        self.assertIn('KOPGRAFT_PF needs IMAGE set explicitly', implicit.stderr)
        foreign = self.dry(KOPGRAFT_PF=lambda d: posix(self.graft(d)), IMAGE=IMAGE_M1)
        self.assertNotEqual(foreign.returncode, 0)
        self.assertIn('which build_k64g.sh did not compare', foreign.stderr)
        plain = self.dry()                                          # no graft: the default image, sources not forced
        self.assertEqual(plain.returncode, 0, plain.stderr)
        self.assertIn(IMAGE_M1, self.argv(plain))
        self.assertIn('-e M1_REQUIRE_SOURCES=0', self.argv(plain))

    def test_hangs_holders_and_the_reset_hint(self):
        text = RUN_M1.read_text(encoding='utf-8')
        tail = text[text.index('hung=0'):]
        self.assertIn("3|124|137) hung=1 ;;", tail)
        self.assertIn("1) grep -qF 'Timeout (' \"$R/m1-$stamp.log\" && hung=1 ;;", tail)   # the faulthandler backstop
        self.assertIn('/sys/dev/char/', tail)
        self.assertIn('tt-smi -ls', tail)
        for token in ('privileged={{.HostConfig.Privileged}}', '"Source":"/dev"', 'fuser -v "$node"'):
            self.assertIn(token, text)

    def test_watcher_and_test_env(self):
        result = self.dry(KOPGRAFT_PF=lambda d: posix(self.graft(d)), WATCHER='1', QWEN_SDPA_PF_TEST='1', IMAGE=IMAGE_A2)
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = self.argv(result)
        self.assertIn('-e TT_METAL_WATCHER=5', argv)
        self.assertIn('dst=/opt/tt-metal/generated/watcher', argv)
        self.assertIn('-e QWEN_SDPA_PF_TEST=1', argv)
        self.assertIn('container cap 900 s', result.stdout)
        text = RUN_M1.read_text(encoding='utf-8')
        self.assertIn('timeout -k 30 "$timeout_s"', text)

    def test_plain_runs_are_unchanged(self):
        result = self.dry()
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = self.argv(result)
        self.assertNotIn('_ttnncpp.so', argv)
        self.assertNotIn('TT_METAL_WATCHER', argv)
        self.assertIn('graft=none watcher=0', result.stdout)


if __name__ == '__main__':
    unittest.main()
