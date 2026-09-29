"""The C2 gate's llk-* plans (llk_profile_plan, wired into c2_serving_gate and c2_serving_job): argv snapshots,
the refusals, prepare/finish around a fake arm, the verdicts, the no-card preflight, the workflow steps, and one
driver run of the decode arms end to end against a fake docker. No device, no image, no network."""
import ast
import json
import os
import re
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import c2_serving_gate as driver  # noqa: E402
import c2_serving_job as job  # noqa: E402
import lever_n_m3native_gate as gate  # noqa: E402
import llk_kernels as kernels  # noqa: E402
import llk_profile_plan as plan  # noqa: E402
import llk_zones as zones  # noqa: E402

with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as _handle:
    PROFILES = json.load(_handle)
PROFILE = 'c2-packed'
IMAGE = 'registry.example/tt-vllm:qwen38-c2-test'
J3 = ('llk-decode-twin', 'llk-prefill-twin', 'llk-decode-zones', 'llk-prefill-zones', 'llk-decode-counters',
      'llk-prefill-counters')
HEADER = ('ARCH: blackhole, CHIP_FREQ[MHz]: 1350, Max Compute Cores: 140\n'
          'PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset], data, run host ID, '
          'trace id, trace id counter, zone name, type, source line, source file, meta data\n')
PROBE_TEXTS = {
    'tools/tracy/__main__.py': ' '.join('"%s"' % option for option in plan.TRACY_OPTIONS + (
        '--disable-device-data-dump-to-files',)),
    'tools/tracy/common.py': 'if "TT_METAL_PROFILER_DIR" in ENVS.keys():',
    'tt_metal/tools/profiler/kernel_profiler.hpp': '#define DeviceZoneScopedN(name)\n#define DeviceZoneScopedSumN1(name)\n'
                                                   '#define DeviceZoneScopedSumN2(name)\n',
    'tt_metal/hostdevcommon/api/hostdevcommon/profiler_common.h':
        'constexpr static std::uint32_t PROFILER_L1_OPTIONAL_MARKER_COUNT = 250;\nstatic constexpr int SUM_COUNT = 2;\n',
    'tt_metal/tools/profiler/perf_counters.hpp': 'constexpr uint16_t PERF_COUNTER_PROFILER_ID = 9090;\n',
}


def read(path):
    with open(os.path.join(ROOT, path), 'rb') as handle:
        return zones.decode(handle.read())


def fake_reader(probes=PROBE_TEXTS, tracy='/opt/tt-metal/tools/tracy/__init__.py', drop=()):
    """The image as this checkout's pinned copies and a synthetic profiler source tree."""
    def reader(image, paths, patterns, probe_paths):
        files = {}
        for entry in kernels.KERNELS:
            if entry['route'] == 'file' and entry.get('repo') and entry['path'] in paths and entry['key'] not in drop:
                files[entry['path']] = read(entry['repo'])
        return dict(files=files, globs={}, probes=dict((path, probes.get(path)) for path in probe_paths),
                    tracy_module=tracy)
    return reader


def arms(name):
    return plan.plan_arms(name, PROFILE, PROFILES, driver)


class PlanTests(unittest.TestCase):
    def test_every_plan_is_one_arm(self):
        for name in plan.PLANS:
            with self.subTest(plan=name):
                spec, = arms(name)
                phase, kind = plan.PLAN_SHAPES[name]
                self.assertEqual(spec[0], name)
                self.assertEqual(spec[2], plan.ARM_SECONDS[name])
                self.assertFalse(spec.judged)
                self.assertFalse(spec.rerun)
                self.assertEqual(spec.env, ())
                llk = spec.extra['llk']
                self.assertEqual((llk['phase'], llk['kind'], llk['level']), (phase, kind, plan.LEVEL[kind]))
                shape = plan.SHAPES[phase]
                args = spec[1]
                self.assertEqual(args[args.index('--users') + 1], str(shape['users']))
                self.assertEqual(args[args.index('--prompt-tokens') + 1], '32768')
                self.assertEqual(args[args.index('--max-tokens') + 1], str(shape['max_tokens']))
                self.assertEqual(args[args.index('--expect-profile') + 1], PROFILE)
                if kind == 'twin':
                    self.assertEqual((llk['env'], llk['tracy']), ((), None))
                else:
                    env = dict(llk['env'])
                    self.assertEqual(env['QWEN_LLK_ZONES'], llk['level'])
                    self.assertEqual(env['TT_METAL_CACHE'], '/root/.cache/tt-metal-cache')
                    self.assertEqual(env['TT_METAL_DEVICE_PROFILER'], '1')
                    tracy = llk['tracy']
                    self.assertEqual(tracy[tracy.index('--op-support-count') + 1], '20000')
                    self.assertNotIn('--disable-device-data-dump-to-files', tracy)
                    self.assertEqual('--enable-sum-profiling' in tracy, kind == 'zones')
                    self.assertEqual('--profiler-capture-perf-counters' in tracy, kind == 'counters')

    def test_the_window_list_fits_the_gate_step(self):
        arms_of = dict((name, arms(name)) for name in J3)
        self.assertLessEqual(driver.worst_case_seconds(list(J3), arms_of), 380 * 60 - 600)

    def test_counter_mask(self):
        bits = dict(fpu=1, pack=2, unpack=4, l1_0=8, l1_1=16, instrn=32)
        self.assertEqual(sum(bits[group] for group in plan.COUNTER_GROUPS), plan.COUNTER_MASK)

    def test_refusals(self):
        with self.assertRaisesRegex(plan.LlkPlanError, 'op-support count 200000 refused'):
            plan.check_tracy(['-p', '--op-support-count', '200000', '-o', plan.PROFILE_DIR])
        with self.assertRaisesRegex(plan.LlkPlanError, 'suppresses profile_log_device.csv'):
            plan.check_tracy(['--disable-device-data-dump-to-files', '--op-support-count', '20000', '-o', plan.PROFILE_DIR])
        with self.assertRaisesRegex(plan.LlkPlanError, 'states its op-support count'):
            plan.check_tracy(['-p', '-o', plan.PROFILE_DIR])
        with self.assertRaisesRegex(plan.LlkPlanError, 'tracy output must go under'):
            plan.check_tracy(['--op-support-count', '20000', '-o', '/bench/out'])
        with self.assertRaisesRegex(plan.LlkPlanError, 'not an environment an LLK arm may add'):
            plan.check_env((('LD_PRELOAD', 'x'),))
        carrying = json.loads(json.dumps(PROFILES))
        carrying['profiles']['c2']['env']['QWEN_LLK_ZONES'] = 'stages'
        with self.assertRaisesRegex(driver.PlanError, 'profile c2 carries QWEN_LLK_ZONES'):
            plan.plan_arms('llk-decode-zones', PROFILE, carrying, driver)
        carrying['profiles']['c2']['env'] = dict(TT_METAL_DEVICE_PROFILER='1')
        with self.assertRaisesRegex(driver.PlanError, 'never runs the profiler'):
            plan.plan_arms('llk-decode-twin', PROFILE, carrying, driver)
        with self.assertRaisesRegex(driver.PlanError, 'not in the image'):
            plan.plan_arms('llk-decode-twin', 'no-such', PROFILES, driver)

    def test_a_non_llk_arm_cannot_carry_profiler_variables(self):
        with self.assertRaisesRegex(driver.PlanError, 'TT_METAL_DEVICE_PROFILER is not an environment an arm may add'):
            driver.agent_shape(IMAGE, 'n', PROFILE, ['<M>', '<A>'], env=(('TT_METAL_DEVICE_PROFILER', '1'),))


class ArgvTests(unittest.TestCase):
    def test_off_the_argv_is_unchanged(self):
        """llk None (every non-LLK arm, and the twins): the pre-LLK composition, byte for byte."""
        args = ['--users', '4']
        before = driver.agent_shape(IMAGE, 'n', PROFILE, ['<M>', '<A>'], env=())
        for script in driver.BENCH_SCRIPTS:
            before += ['--mount', 'type=bind,src=%s,dst=/bench/%s,readonly' % (
                os.path.join('/co', 'scripts', 'ci', script), script)]
        before += ['--mount', 'type=bind,src=/r/a,dst=%s' % driver.RESULTS_IN_CONTAINER]
        before += ['--entrypoint', 'python3', IMAGE, '-B', '/bench/lever_n_m3native_gate.py'] + args
        self.assertEqual(driver.gate_run(IMAGE, 'n', PROFILE, ['<M>', '<A>'], '/co', '/r/a', args), before)
        twin = plan.prepare_arm(IMAGE, '/r/a', arms('llk-decode-twin')[0].extra['llk'], fake_reader())
        self.assertEqual(driver.gate_run(IMAGE, 'n', PROFILE, ['<M>', '<A>'], '/co', '/r/a', args, llk=twin), before)
        self.assertFalse([word for word in before if 'PROFILER' in word or 'QWEN_LLK' in word or 'tracy' in word])

    def test_a_profiled_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared = plan.prepare_arm(IMAGE, directory, arms('llk-decode-zones')[0].extra['llk'], fake_reader(), log=[].append)
            argv = driver.gate_run(IMAGE, 'n', PROFILE, ['<M>', '<A>'], '/co', directory, ['--users', '4'], llk=prepared)
            entry = argv[argv.index('--entrypoint'):]
            self.assertEqual(entry[:6], ['--entrypoint', 'python3', IMAGE, '-B', '-m', 'tracy'])
            self.assertEqual(entry[-3:], ['/bench/lever_n_m3native_gate.py', '--users', '4'])
            env = [argv[i + 1] for i, word in enumerate(argv) if word == '-e']
            self.assertIn('QWEN_LLK_ZONES=stages', env)
            self.assertIn('TT_METAL_PROFILER_DIR=/opt/tt-metal/generated/profiler', env)
            mounts = [argv[i + 1] for i, word in enumerate(argv) if word == '--mount']
            destinations = [re.search(r'dst=([^,]+)', mount).group(1) for mount in mounts]
            self.assertIn('/opt/tt-metal/generated/profiler', destinations)
            sdpa = '/opt/tt-metal/' + kernels.BY_KEY['SDPA_DEC']['path']
            self.assertIn(sdpa, destinations)
            mount = mounts[destinations.index(sdpa)]
            self.assertTrue(mount.endswith(',readonly'))
            source = re.search(r'src=([^,]+)', mount).group(1)
            with open(source, 'rb') as handle:
                copy = zones.decode(handle.read())
            self.assertEqual(zones.remove(copy), read(kernels.BY_KEY['SDPA_DEC']['repo']))
            # Only kernel sources are mounted over the image; never a model, graft or pinned file.
            for destination in destinations:
                if destination != plan.PROFILE_DIR and not destination.startswith('/bench/') and \
                        destination != driver.RESULTS_IN_CONTAINER and destination not in ('/models', '/dev/hugepages-1G'):
                    kernels.check_destination(destination[len('/opt/tt-metal/'):])

    def test_dry_run_shows_the_planned_mounts(self):
        lines = []
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'profiles.json')
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump(PROFILES, handle)
            code = driver.main(['--image', IMAGE, '--profile', PROFILE, '--plan', 'llk-decode-twin,llk-decode-zones',
                                '--results', os.path.join(directory, 'r'), '--profiles', path, '--dry-run'],
                               devices=['<M>', '<A>'], log=lines.append)
        self.assertEqual(code, 0)
        planned = [json.loads(line) for line in lines[1:]]
        self.assertEqual([entry['arm'] for entry in planned], ['llk-decode-twin', 'llk-decode-zones'])
        self.assertNotIn('tracy', planned[0]['docker'])
        self.assertIn('tracy', planned[1]['docker'])
        self.assertTrue(any('dst=<each instrumented kernel path>' in word for word in planned[1]['docker']))


class CapabilityTests(unittest.TestCase):
    def test_everything_present(self):
        caps = plan.capabilities(fake_reader()(IMAGE, [], [], plan.PROBES))
        self.assertTrue(all(caps['tracy_options'].values()))
        self.assertTrue(caps['dump_to_files_flag'])
        self.assertEqual((caps['zone_macro'], caps['sum_zones'], caps['perf_counters']), (True, True, True))
        self.assertEqual((caps['optional_markers'], caps['sum_count']), (250, 2))
        self.assertEqual(caps['unknown'], [])
        for kind in ('twin', 'zones', 'counters'):
            self.assertIsNone(plan.unsupported(kind, caps))

    def test_unknown_is_none_never_false(self):
        caps = plan.capabilities(fake_reader(probes={})(IMAGE, [], [], plan.PROBES))
        self.assertEqual(caps['zone_macro'], None)
        self.assertEqual(set(caps['tracy_options'].values()), {None})
        self.assertEqual(len(caps['unknown']), 5)
        self.assertIn('lacks', plan.unsupported('zones', caps))

    def test_counters_unsupported(self):
        probes = dict(PROBE_TEXTS)
        probes['tt_metal/tools/profiler/perf_counters.hpp'] = None
        caps = plan.capabilities(fake_reader(probes=probes)(IMAGE, [], [], plan.PROBES))
        self.assertIsNone(plan.unsupported('zones', caps))
        self.assertTrue(plan.unsupported('counters', caps).startswith('UNSUPPORTED'))
        self.assertIn('not importable', plan.unsupported('zones', plan.capabilities(fake_reader(tracy=None)(
            IMAGE, [], [], plan.PROBES))))


def device_log(chips=(0, 1), counters=False, kernels_=('K5A', 'SDPA_DEC'), extra=''):
    rows = []
    for chip in chips:
        for index, kernel in enumerate(kernels_):
            core = (1 + index, 2)
            for offset, risc in enumerate(('TRISC_0', 'TRISC_1', 'TRISC_2')):
                start = 1000 * (index + 1) + offset
                rows.append('%d,%d,%d,%s,1,%d,0,64,3,7,QWEN_LLK_%s,ZONE_START,1,k.cpp,\n' % (chip, core[0], core[1], risc,
                                                                                          start, kernel))
                rows.append('%d,%d,%d,%s,1,%d,0,64,3,7,QWEN_LLK_%s,ZONE_END,1,k.cpp,\n' % (chip, core[0], core[1], risc,
                                                                                        start + 800, kernel))
                if not counters:
                    rows.append('%d,%d,%d,%s,3,%d,%d,64,3,7,QWEN_LLK_WAIT_IN,ZONE_TOTAL,1,k.cpp,\n' % (
                        chip, core[0], core[1], risc, start + 800, 100 * (offset + 1)))
            if counters:
                rows.append('%d,%d,%d,BRISC,9090,9000,5,64,3,7,,TS_DATA_16B,0,,{"counter type":"FPU_COUNTER";"ref cnt":10;'
                            '"value":5}\n' % (chip, core[0], core[1]))
    return HEADER + ''.join(rows) + extra


class FinishTests(unittest.TestCase):
    def prepared(self, directory, name='llk-decode-zones'):
        return plan.prepare_arm(IMAGE, directory, arms(name)[0].extra['llk'], fake_reader(), log=[].append)

    def write_log(self, directory, text):
        logs = os.path.join(directory, plan.PROFILE_SUBDIR, '.logs')
        os.makedirs(logs, exist_ok=True)
        with open(os.path.join(logs, 'profile_log_device.csv'), 'w', encoding='utf-8', newline='') as handle:
            handle.write(text)

    def console(self, level='stages'):
        import test_gdn_seq_block
        _, records = kernels.instrument_generated(dict(test_gdn_seq_block.build()), level)
        return ''.join('2026-09-29 | INFO | serving_runtime - [LLK] record %s\n' % json.dumps(record) for record in records)

    def test_export_report_coverage_and_prune(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared = self.prepared(directory)
            self.assertEqual(sorted(os.listdir(directory)), ['llk-manifest.json', 'llk-overlay', 'llk-profile'])
            self.write_log(directory, device_log())
            handed = []
            self.assertEqual(plan.handback_arm(IMAGE, directory, prepared, lambda image, path: handed.append(path) or 0), 0)
            self.assertEqual(handed, [os.path.join(directory, plan.PROFILE_SUBDIR)])
            lines = []
            with mock.patch.object(plan, 'KEEP_BYTES', 100):
                summary = plan.finish_arm(IMAGE, directory, prepared, console=self.console(), log=lines.append)
            self.assertEqual(summary['coverage'], [])
            self.assertTrue(summary['complete'])
            self.assertEqual(sorted(summary['kernels']), ['K5A', 'SDPA_DEC'])
            self.assertEqual([entry['path'] for entry in summary['pruned']], [os.path.join('.logs', 'profile_log_device.csv')])
            self.assertFalse(os.path.exists(os.path.join(directory, plan.PROFILE_SUBDIR, '.logs', 'profile_log_device.csv')))
            for name in (plan.REPORT, plan.EXPORT, plan.PRUNED, plan.MANIFEST):
                self.assertTrue(os.path.isfile(os.path.join(directory, name)), name)
            with open(os.path.join(directory, plan.MANIFEST), encoding='utf-8') as handle:
                manifest = json.load(handle)
            self.assertEqual(sorted(record['key'] for record in manifest['generated']), ['K5A', 'K5A_RD', 'K5A_WR'])
            self.assertTrue(any('#1' in line for line in lines))

    def test_missing_chip_and_missing_log(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared = self.prepared(directory)
            self.write_log(directory, device_log(chips=(0,)))
            summary = plan.finish_arm(IMAGE, directory, prepared, console=self.console(), log=[].append)
            self.assertEqual(summary['coverage'], ['K5A: no zone on chip 1', 'SDPA_DEC: no zone on chip 1'])
        with tempfile.TemporaryDirectory() as directory:
            prepared = self.prepared(directory)
            summary = plan.finish_arm(IMAGE, directory, prepared, console=self.console(), log=[].append)
            self.assertEqual(summary['problem'], 'no profile_log_device.csv under llk-profile')

    def test_a_broken_log_is_a_problem_not_a_crash(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared = self.prepared(directory)
            self.write_log(directory, 'garbage\n')
            summary = plan.finish_arm(IMAGE, directory, prepared, console=self.console(), log=[].append)
            self.assertIn('not a tt-metal device log', summary['problem'])

    def test_a_skipped_arm_writes_its_reason(self):
        probes = dict(PROBE_TEXTS)
        probes['tt_metal/tools/profiler/perf_counters.hpp'] = None
        with tempfile.TemporaryDirectory() as directory:
            prepared = plan.prepare_arm(IMAGE, directory, arms('llk-decode-counters')[0].extra['llk'],
                                        fake_reader(probes=probes))
            self.assertTrue(prepared['skip'].startswith('UNSUPPORTED'))
            with open(os.path.join(directory, plan.MANIFEST), encoding='utf-8') as handle:
                self.assertEqual(json.load(handle)['skipped'], prepared['skip'])
            self.assertIsNone(plan.handback_arm(IMAGE, directory, prepared, lambda image, path: 1 / 0))

    def test_without_sum_zones_the_tracy_flag_goes_too(self):
        probes = dict(PROBE_TEXTS)
        probes['tt_metal/tools/profiler/kernel_profiler.hpp'] = '#define DeviceZoneScopedN(name)\n'
        with tempfile.TemporaryDirectory() as directory:
            prepared = plan.prepare_arm(IMAGE, directory, arms('llk-decode-zones')[0].extra['llk'],
                                        fake_reader(probes=probes), log=[].append)
            self.assertNotIn('--enable-sum-profiling', prepared['tracy'])
            self.assertFalse(prepared['sums'])
            sdpa = [record for record in prepared['manifest']['files'] if record.get('key') == 'SDPA_DEC'][0]
            self.assertFalse(sdpa['sync']['enabled'])

    def test_generated_records_parse_loguru_lines(self):
        console = 'noise\n2026 | INFO | x - [LLK] record {"key": "K5A", "zones": []}\n[LLK] record {broken\n'
        self.assertEqual(plan.generated_records(console), [dict(key='K5A', zones=[])])


def served_report(configuration, tokens=('a', 'b', 'c', 'd'), trace_ms=150.0):
    return dict(streams=[dict(text_sha256=token, completion_tokens=48, finish_reason='length') for token in tokens],
                qwen_configuration=configuration, packed_phase=dict(rounds=4, trace_ms_mean=trace_ms), gate_passed=True)


class VerdictTests(unittest.TestCase):
    def test_twin(self):
        result = plan.verdict('llk-decode-twin', served_report({}), dict(exit=0), None)
        self.assertEqual(result['verdict'], 'PASS')
        self.assertEqual(plan.verdict('llk-decode-twin', None, dict(exit=125), None)['verdict'], 'FAIL')
        broken = served_report({}, tokens=('a',))
        self.assertIn('1 streams, asked 4', plan.verdict('llk-decode-twin', broken, {}, None)['reason'])

    def arm(self, **llk):
        env = arms('llk-decode-zones')[0].extra['llk']['env']
        base = dict(coverage=[], drops=dict(unmatched_end=0, unmatched_start=0, console=False), counters_seen=0,
                    ranking=[dict(rank=1, kernel='K5A', bound='llk-math', reason='r')])
        base.update(llk)
        return dict(exit=0, llk=base, llk_env=[list(pair) for pair in env])

    def configuration(self):
        return dict((name, value) for name, value in arms('llk-decode-zones')[0].extra['llk']['env']
                    if plan.CONFIGURATION_PREFIX.match(name))

    def test_zones(self):
        twin = served_report({})
        report = served_report(self.configuration())
        report['c2_gate_problems'] = ['S2: something the judged arms check']
        result = plan.verdict('llk-decode-zones', report, self.arm(), twin)
        self.assertEqual(result['verdict'], 'PASS', result)
        self.assertIn('#1 K5A llk-math: r', result['lines'])
        self.assertIn('note: S2: something the judged arms check', result['lines'])
        differing = plan.verdict('llk-decode-zones', served_report(self.configuration(), tokens=('a', 'b', 'c', 'x')),
                                 self.arm(), twin)
        self.assertEqual(differing['verdict'], 'FAIL')
        self.assertIn('tokens differ from llk-decode-twin', differing['reason'])
        missing = plan.verdict('llk-decode-zones', report, self.arm(coverage=['SDPA_DEC: no zone on chip 1']), twin)
        self.assertEqual(missing['verdict'], 'NOT_EXERCISED')
        dropped = plan.verdict('llk-decode-zones', report, self.arm(drops=dict(unmatched_end=2)), twin)
        self.assertIn('dropped markers', dropped['reason'])
        unshown = plan.verdict('llk-decode-zones', served_report({}), self.arm(), twin)
        self.assertEqual(unshown['verdict'], 'FAIL')
        self.assertIn('never reached the container', unshown['reason'])
        alone = plan.verdict('llk-decode-zones', report, self.arm(), None)
        self.assertIn('no llk-decode-twin report: tokens not compared', alone['lines'])

    def test_counters(self):
        configuration = dict((name, value) for name, value in arms('llk-decode-counters')[0].extra['llk']['env']
                             if plan.CONFIGURATION_PREFIX.match(name))
        report = served_report(configuration)
        arm = self.arm()
        arm['llk_env'] = [list(pair) for pair in arms('llk-decode-counters')[0].extra['llk']['env']]
        result = plan.verdict('llk-decode-counters', report, arm, served_report({}))
        self.assertEqual(result['verdict'], 'NOT_EXERCISED')
        self.assertIn('UNSUPPORTED: no counter rows', result['reason'])
        arm['llk']['counters_seen'] = 12
        self.assertEqual(plan.verdict('llk-decode-counters', report, arm, served_report({}))['verdict'], 'PASS')


class HandbackAndPreflightTests(unittest.TestCase):
    def test_handback_results(self):
        with tempfile.TemporaryDirectory() as directory:
            for arm in ('llk-decode-zones', 'bringup-concurrent'):
                os.makedirs(os.path.join(directory, arm))
            profile = os.path.join(directory, 'llk-decode-zones', plan.PROFILE_SUBDIR, '.logs')
            os.makedirs(profile)
            with open(os.path.join(profile, 'profile_log_device.csv'), 'wb') as handle:
                handle.write(b'x' * 64)
            calls = []
            with mock.patch.object(plan, 'KEEP_BYTES', 16):
                done = plan.handback_results(directory, IMAGE, handback=lambda image, path: calls.append(path),
                                             log=[].append)
            self.assertEqual(list(done), ['llk-decode-zones'])
            self.assertEqual(done['llk-decode-zones'][0]['bytes'], 64)
            self.assertEqual(len(calls), 1)

    def generated(self, image, checkout):
        import test_gdn_seq_block
        build = test_gdn_seq_block.build()
        return dict(qualified=False, stages=kernels.instrument_generated(dict(build), 'stages')[1],
                    tag=kernels.instrument_generated(dict(build), 'tag')[1])

    def test_preflight_passes_on_the_pinned_copies(self):
        with tempfile.TemporaryDirectory() as directory:
            result = plan.preflight(IMAGE, ROOT, directory, reader=fake_reader(), generated=self.generated)
            self.assertEqual(result['problems'], [])
            self.assertTrue(os.path.isfile(os.path.join(directory, 'llk-preflight.json')))
            decode = result['phases']['decode']['levels']['stages']
            self.assertTrue(any(record.get('key') == 'SDPA_DEC' and record.get('pin_match') for record in decode))

    def test_preflight_names_what_the_window_would_lose(self):
        with tempfile.TemporaryDirectory() as directory:
            result = plan.preflight(IMAGE, ROOT, directory, reader=fake_reader(drop=('SDPA_DEC',)),
                                    generated=lambda image, checkout: dict(error='ImportError: no gdn_seq_block'))
        self.assertIn('decode SDPA_DEC at level stages: not in the image', result['problems'])
        self.assertIn('K5A: the image\'s build could not be instrumented: ImportError: no gdn_seq_block', result['problems'])


class FakeLlkDocker(object):
    """An arm as the container would leave it: the harness report (with the -e variables that reached it), a server
    log with the override's record lines on profiled arms, and a device log in the mounted profile tree."""

    def __init__(self, tokens=('a', 'b', 'c', 'd')):
        self.calls, self.tokens = [], tokens

    def __call__(self, arguments, stdout_path, timeout, name):
        arm = name[len(driver.CONTAINER_PREFIX):]
        self.calls.append(dict(arm=arm, arguments=arguments))
        env = dict(arguments[i + 1].split('=', 1) for i, word in enumerate(arguments) if word == '-e')
        configuration = dict((key, value) for key, value in env.items() if re.match(r'(?:QWEN[0-9]*_|TT_)', key))
        report = served_report(configuration, tokens=self.tokens)
        arm_dir = os.path.dirname(stdout_path)
        with open(stdout_path, 'w') as handle:
            handle.write('%s\n%s\n%s\n' % (gate.BEGIN, json.dumps(report), gate.END))
        log = 'INFO ' + driver.QUARANTINE_LIVE + '\n'
        if env.get('QWEN_LLK_ZONES'):
            import test_gdn_seq_block
            _, records = kernels.instrument_generated(dict(test_gdn_seq_block.build()), env['QWEN_LLK_ZONES'])
            log += ''.join('INFO [LLK] record %s\n' % json.dumps(record) for record in records)
            logs = os.path.join(arm_dir, plan.PROFILE_SUBDIR, '.logs')
            os.makedirs(logs, exist_ok=True)
            with open(os.path.join(logs, 'profile_log_device.csv'), 'w', encoding='utf-8', newline='') as handle:
                handle.write(device_log(counters=env['QWEN_LLK_ZONES'] == 'tag'))
        with open(os.path.join(arm_dir, 'server.log'), 'w') as handle:
            handle.write(log)
        return 0


class DriverTests(unittest.TestCase):
    def run_driver(self, plans, docker=None):
        docker = docker or FakeLlkDocker()
        handed = []
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'profiles.json')
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump(PROFILES, handle)
            results = os.path.join(directory, 'results')
            lines = []
            code = driver.main(['--image', IMAGE, '--profile', PROFILE, '--plan', ','.join(plans), '--results', results,
                                '--profiles', path], execute=docker, devices=['/dev/tenstorrent/3', '/dev/tenstorrent/1'],
                               log=lines.append, containers=lambda: [], corpus=lambda: {},
                               llk_reader=fake_reader(), llk_handback=lambda image, path: handed.append(path) or 0)
            with open(os.path.join(results, 'c2-gate-summary.json'), encoding='utf-8') as handle:
                summary = json.load(handle)
            files = dict((arm, sorted(os.listdir(os.path.join(results, arm)))) for arm in plans)
        return code, summary, docker.calls, lines, files, handed

    def test_the_decode_arms_end_to_end(self):
        code, summary, calls, lines, files, handed = self.run_driver(
            ['llk-decode-twin', 'llk-decode-zones', 'llk-decode-counters'])
        verdicts = dict((name, result['verdict']) for name, result in summary['results'].items())
        self.assertEqual(verdicts, {'llk-decode-twin': 'PASS', 'llk-decode-zones': 'PASS', 'llk-decode-counters': 'PASS'},
                         json.dumps(summary['results'], indent=1)[:3000])
        self.assertEqual(code, 0)
        self.assertEqual([call['arm'] for call in calls], ['llk-decode-twin', 'llk-decode-zones', 'llk-decode-counters'])
        self.assertNotIn('tracy', calls[0]['arguments'])
        self.assertIn('--enable-sum-profiling', calls[1]['arguments'])
        self.assertIn('--profiler-capture-perf-counters', calls[2]['arguments'])
        self.assertEqual(len(handed), 2)
        for arm in ('llk-decode-zones', 'llk-decode-counters'):
            for name in (plan.MANIFEST, plan.REPORT, plan.EXPORT, plan.PRUNED, 'docker-run.json', 'llk-overlay'):
                self.assertIn(name, files[arm], arm)
        self.assertNotIn(plan.MANIFEST, files['llk-decode-twin'])
        zones_arm = summary['arms']['llk-decode-zones']['llk']
        self.assertEqual(sorted(zones_arm['kernels']), ['K5A', 'SDPA_DEC'])
        self.assertEqual(summary['arms']['llk-decode-counters']['llk']['counters_seen'], 4)
        self.assertTrue(any('[C2-GATE] llk-decode-zones: LLK profile: 2 kernels' in line for line in lines))

    def test_token_drift_fails_the_profiled_arm(self):
        class Drifting(FakeLlkDocker):
            def __call__(self, arguments, stdout_path, timeout, name):
                self.tokens = ('a', 'b', 'c', 'x') if 'tracy' in arguments else ('a', 'b', 'c', 'd')
                return FakeLlkDocker.__call__(self, arguments, stdout_path, timeout, name)
        code, summary, _, _, _, _ = self.run_driver(['llk-decode-twin', 'llk-decode-zones'], Drifting())
        self.assertEqual(summary['results']['llk-decode-zones']['verdict'], 'FAIL')
        self.assertEqual(code, 1)

    def test_unsupported_counters_skip_the_other_phase_without_a_container(self):
        probes = dict(PROBE_TEXTS)
        probes['tt_metal/tools/profiler/perf_counters.hpp'] = None
        docker = FakeLlkDocker()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'profiles.json')
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump(PROFILES, handle)
            results = os.path.join(directory, 'results')
            driver.main(['--image', IMAGE, '--profile', PROFILE, '--plan', 'llk-decode-counters,llk-prefill-counters',
                         '--results', results, '--profiles', path], execute=docker, devices=['<M>', '<A>'],
                        log=[].append, containers=lambda: [], corpus=lambda: {}, llk_reader=fake_reader(probes=probes),
                        llk_handback=lambda image, path: 0)
            with open(os.path.join(results, 'c2-gate-summary.json'), encoding='utf-8') as handle:
                summary = json.load(handle)
        self.assertEqual(docker.calls, [])
        self.assertTrue(summary['results']['llk-decode-counters']['reason'].startswith('UNSUPPORTED'))
        self.assertTrue(summary['results']['llk-prefill-counters']['reason'].startswith('skipped: UNSUPPORTED'))


class JobTests(unittest.TestCase):
    def read(self, **values):
        base = dict(C2_IMAGE_TAG='s2p-abc1234', C2_ACTIONS='status reset gate', C2_PROFILE=PROFILE)
        base.update(values)
        return job.read_job(base, sorted(PROFILES['profiles']))

    def test_llk_plans_and_actions(self):
        outputs = self.read(C2_GATE_PLAN=','.join(J3))
        self.assertEqual(outputs['gate_plan'], ','.join(J3))
        self.assertEqual(self.read(C2_ACTIONS='status build llkcheck')['actions'], 'status build llkcheck')
        self.assertEqual(self.read(C2_GATE_PLAN='warm,mixed,llk-decode-twin,llk-decode-zones')['gate_plan'],
                         'warm,mixed,llk-decode-twin,llk-decode-zones')

    def test_refusals(self):
        with self.assertRaisesRegex(job.JobError, 'llk-k5a-clock is refused: the K5-A clock-page adapter'):
            self.read(C2_GATE_PLAN='llk-decode-twin,llk-k5a-clock')
        with self.assertRaisesRegex(job.JobError, 'mixed runs after llk-decode-twin'):
            self.read(C2_GATE_PLAN='llk-decode-twin,mixed')
        with self.assertRaisesRegex(job.JobError, 'names llk-decode-zones twice'):
            self.read(C2_GATE_PLAN='llk-decode-zones,llk-decode-zones')


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(ROOT, '.github', 'workflows', 'qwen-c2-serving.yml'), encoding='utf-8') as handle:
            self.text = handle.read()

    def test_the_steps(self):
        import yaml
        steps = list(yaml.safe_load(self.text)['jobs'].values())[0]['steps']
        names = [step.get('name') for step in steps]
        preflight = steps[names.index('LLK profiling preflight (no card)')]
        handback = steps[names.index('Hand back the LLK profiler output')]
        self.assertEqual(preflight['if'], "contains(steps.job.outputs.actions, 'llkcheck')")
        self.assertIn('llk_profile_plan.py preflight', preflight['run'])
        self.assertTrue(handback['if'].startswith('always() && '))
        self.assertIn("contains(steps.job.outputs.gate_plan, 'llk-')", handback['if'])
        self.assertIn('llk_profile_plan.py handback --results "$RUNNER_TEMP/c2-results/gate"', handback['run'])
        self.assertLess(names.index("Run the gate in the agent's container shape"), names.index('Hand back the LLK profiler output'))
        self.assertLess(names.index('Hand back the LLK profiler output'), names.index('Upload results'))
        self.assertNotIn('llk', ' '.join(str(step.get('run', '')) for step in steps if step.get('name') == 'Upload results'))

    def test_the_cpu_suite_runs_these_tests(self):
        with open(os.path.join(ROOT, '.github', 'workflows', 'qwen-integration-cpu.yml'), encoding='utf-8') as handle:
            text = handle.read()
        for module in ('test_llk_zones', 'test_llk_kernels', 'test_llk_profile_report', 'test_llk_zone_override',
                       'test_llk_profile_plan'):
            self.assertTrue(re.search(r'python -B -m unittest [^\n]*\b%s\b' % module, text), module)


class SyntaxTests(unittest.TestCase):
    def test_gate_side_modules_parse_as_python_37(self):
        for name in ('llk_profile_plan.py', 'c2_serving_job.py', 'c2_serving_gate.py'):
            with open(os.path.join(HERE, name), encoding='utf-8') as handle:
                ast.parse(handle.read(), name, feature_version=(3, 7))


if __name__ == '__main__':
    unittest.main()
