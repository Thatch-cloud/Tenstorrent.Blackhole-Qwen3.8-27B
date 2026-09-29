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
import llk_profile_report as report_module  # noqa: E402
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


# A compute kernel for the file entries this repo holds no copy of (SDPA_PF, MM): every sync call of the table.
SYNTHETIC_COMPUTE = ('#include "compute_kernel_api.h"\n\nnamespace NAMESPACE {\nvoid MAIN {\n    cb_wait_front(0, 1);\n'
                     '    tile_regs_acquire();\n    tile_regs_commit();\n    tile_regs_wait();\n    cb_reserve_back(16, 1);\n'
                     '    pack_tile(0, 16);\n    tile_regs_release();\n    cb_push_back(16, 1);\n    cb_pop_front(0, 1);\n}\n'
                     '}  // namespace NAMESPACE\n')


def fake_reader(probes=PROBE_TEXTS, tracy='/opt/tt-metal/tools/tracy/__init__.py', drop=()):
    """The image as this checkout's pinned copies, SYNTHETIC_COMPUTE for the unpinned compute kernels (not the
    header) and a synthetic profiler source tree."""
    def reader(image, paths, patterns, probe_paths):
        files = {}
        for entry in kernels.KERNELS:
            if entry['route'] != 'file' or entry['path'] not in paths or entry['key'] in drop:
                continue
            if entry.get('repo'):
                files[entry['path']] = read(entry['repo'])
            elif not entry.get('header'):
                files[entry['path']] = SYNTHETIC_COMPUTE
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
                # Every arm's docker limit covers the readiness allowance and the stream: a slow cold compile ends
                # in the harness's own report, never in a docker kill.
                self.assertEqual(args[args.index('--readiness-seconds') + 1], str(plan.READINESS_SECONDS))
                stream = int(args[args.index('--stream-timeout') + 1])
                self.assertGreaterEqual(spec[2], plan.READINESS_SECONDS + stream)
                if kind == 'twin':
                    self.assertEqual((llk['env'], llk['tracy']), ((), None))
                else:
                    env = dict(llk['env'])
                    self.assertEqual(env['QWEN_LLK_ZONES'], llk['level'])
                    self.assertEqual(env['TT_METAL_CACHE'], '/root/.cache/tt-metal-cache')
                    self.assertEqual(env['TT_METAL_DEVICE_PROFILER'], '1')
                    tracy = llk['tracy']
                    self.assertEqual(tracy[tracy.index('--op-support-count') + 1], '20000')
                    # tracy hands its -o to the child as TT_METAL_PROFILER_DIR: one path, the bind mount.
                    self.assertEqual(tracy[tracy.index('-o') + 1], env['TT_METAL_PROFILER_DIR'])
                    self.assertEqual(env['TT_METAL_PROFILER_DIR'], plan.PROFILE_DIR)
                    self.assertEqual(env[plan.FLUSH_FLAG], '1')
                    self.assertEqual(env['QWEN_FAST_PROFILE_DUMP_ROUND'], str(plan.DUMP_ROUND))
                    self.assertNotIn('--disable-device-data-dump-to-files', tracy)
                    self.assertEqual('--enable-sum-profiling' in tracy, kind == 'zones')
                    self.assertEqual('--profiler-capture-perf-counters' in tracy, kind == 'counters')

    def test_the_window_list_fits_the_gate_step(self):
        arms_of = dict((name, arms(name)) for name in J3)
        self.assertLessEqual(driver.worst_case_seconds(list(J3), arms_of), 380 * 60 - 600)

    def test_every_phase_requires_a_kernel(self):
        self.assertEqual(plan.REQUIRED, dict(decode=('K5A', 'SDPA_DEC'), prefill=('SDPA_PF',)))

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
        with self.assertRaisesRegex(plan.LlkPlanError, 'tracy output must be /opt/tt-metal/generated/profiler'):
            plan.check_tracy(['--op-support-count', '20000', '-o', '/bench/out'])
        with self.assertRaisesRegex(plan.LlkPlanError, 'and its TT_METAL_PROFILER_DIR'):
            plan.check_tracy(['--op-support-count', '20000', '-o', plan.PROFILE_DIR + '/tracy'])
        self.assertTrue(plan.check_tracy(['--op-support-count', '20000', '-o', plan.PROFILE_DIR + '/']))
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


def device_log(chips=(0, 1), counters=False, kernels_=('K5A', 'SDPA_DEC'), replays=(7, 8), extra='', sums=True):
    """A profiled arm's device log: each kernel on its own core, on the three TRISCs of each chip, once per trace
    replay session (replays: trace id counters of trace 3; (None,) writes untraced rows, a prefill's). A zones arm's
    rows carry the sum of every slot each thread waits in (llk_profile_report.THREAD_SLOTS) unless sums is False; a
    counters arm's carry one counter row per kernel core and session instead."""
    rows = []
    slots = (('wait_in', 3, zones.WAIT_IN), ('wait_out', 4, zones.WAIT_OUT))
    for replay in replays:
        trace, counter = ('', '') if replay is None else ('3', str(replay))
        base = 0 if replay is None else 100000 * replay
        for chip in chips:
            for index, kernel in enumerate(kernels_):
                core = (1 + index, 2)
                where = (chip, core[0], core[1])
                for offset, risc in enumerate(('TRISC_0', 'TRISC_1', 'TRISC_2')):
                    start = base + 1000 * (index + 1) + offset
                    rows.append('%d,%d,%d,%s,1,%d,0,64,%s,%s,QWEN_LLK_%s,ZONE_START,1,k.cpp,\n' % (
                        where + (risc, start, trace, counter, kernel)))
                    rows.append('%d,%d,%d,%s,1,%d,0,64,%s,%s,QWEN_LLK_%s,ZONE_END,1,k.cpp,\n' % (
                        where + (risc, start + 800, trace, counter, kernel)))
                    if counters or not sums:
                        continue
                    for slot, timer, zone in slots:
                        if slot in report_module.THREAD_SLOTS[risc]:
                            rows.append('%d,%d,%d,%s,%d,%d,%d,64,%s,%s,%s,ZONE_TOTAL,1,k.cpp,\n' % (
                                where + (risc, timer, start + 800, 100 * (offset + 1), trace, counter, zone)))
                if counters:
                    rows.append('%d,%d,%d,BRISC,9090,%d,5,64,%s,%s,,TS_DATA_16B,0,,{"counter type":"FPU_COUNTER";'
                                '"ref cnt":10;"value":5}\n' % (where + (base + 9000, trace, counter)))
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
            self.assertEqual(summary['selection']['window'], 'traced')
            self.assertEqual(summary['selection']['chips']['0'], dict(sessions=2, dropped=0, short=0, used=2))
            self.assertEqual([entry['path'] for entry in summary['pruned']], [os.path.join('.logs', 'profile_log_device.csv')])
            self.assertFalse(os.path.exists(os.path.join(directory, plan.PROFILE_SUBDIR, '.logs', 'profile_log_device.csv')))
            for name in (plan.REPORT, plan.EXPORT, plan.PRUNED, plan.MANIFEST):
                self.assertTrue(os.path.isfile(os.path.join(directory, name)), name)
            with open(os.path.join(directory, plan.MANIFEST), encoding='utf-8') as handle:
                manifest = json.load(handle)
            self.assertEqual(sorted(record['key'] for record in manifest['generated']), ['K5A', 'K5A_RD', 'K5A_WR'])
            self.assertTrue(any('#1' in line for line in lines))

    def test_the_prune_reuses_the_export_hash(self):
        """A decode log is tens of GB: the export already hashed it, so the prune does not read it again."""
        with tempfile.TemporaryDirectory() as directory:
            prepared = self.prepared(directory)
            self.write_log(directory, device_log())
            with mock.patch.object(plan, 'KEEP_BYTES', 100), \
                    mock.patch.object(plan, 'file_sha256', side_effect=AssertionError('hashed twice')):
                summary = plan.finish_arm(IMAGE, directory, prepared, console=self.console(), log=[].append)
            pruned, = summary['pruned']
            self.assertEqual(pruned['sha256'], summary['export']['source']['sha256'])
            self.assertEqual(pruned['bytes'], summary['export']['source']['bytes'])

    def test_a_decode_arm_needs_two_complete_sessions_per_chip(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared = self.prepared(directory)
            self.write_log(directory, device_log(replays=(7,)))
            summary = plan.finish_arm(IMAGE, directory, prepared, console=self.console(), log=[].append)
        self.assertEqual(summary['coverage'], ['K5A: 1 complete traced sessions on chip 0, 2 needed',
                                               'K5A: 1 complete traced sessions on chip 1, 2 needed',
                                               'SDPA_DEC: 1 complete traced sessions on chip 0, 2 needed',
                                               'SDPA_DEC: 1 complete traced sessions on chip 1, 2 needed'])

    def test_a_zones_arm_without_its_wait_sums_is_not_exercised(self):
        """Sums asked for, no ZONE_TOTAL row came back: the waits are undetermined, never zero (which would read
        every kernel as LLK-bound)."""
        with tempfile.TemporaryDirectory() as directory:
            prepared = self.prepared(directory)
            self.write_log(directory, device_log(sums=False))
            summary = plan.finish_arm(IMAGE, directory, prepared, console=self.console(), log=[].append)
            with open(os.path.join(directory, plan.REPORT), encoding='utf-8') as handle:
                analysed = json.load(handle)
        self.assertTrue(any(problem.startswith('K5A: wait sums missing: TRISC_0 wait_in') for problem in summary['coverage']),
                        summary['coverage'])
        self.assertEqual(set(kernel['bound'] for kernel in analysed['kernels']), {'undetermined'})
        self.assertFalse(any(kernel['llk_bound'] for kernel in analysed['kernels']))

    def test_a_prefill_arm_reads_the_untraced_rows_and_needs_sdpa_pf(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared = self.prepared(directory, 'llk-prefill-zones')
            self.assertIn('SDPA_PF', [record.get('key') for record in prepared['manifest']['files']
                                      if 'refused' not in record])
            self.write_log(directory, device_log(kernels_=('SDPA_PF',), replays=(None,)) +
                           device_log(kernels_=('SDPA_PF',), replays=(7,))[len(HEADER):])
            summary = plan.finish_arm(IMAGE, directory, prepared, console=self.console(), log=[].append)
        self.assertEqual(summary['selection']['window'], 'untraced')
        self.assertEqual(summary['coverage'], [])
        self.assertEqual(summary['kernels'], ['SDPA_PF'])
        with tempfile.TemporaryDirectory() as directory:
            prepared = self.prepared(directory, 'llk-prefill-zones')
            self.write_log(directory, device_log(kernels_=('SDPA_PF',), replays=(7,)))
            summary = plan.finish_arm(IMAGE, directory, prepared, console=self.console(), log=[].append)
        self.assertEqual(summary['coverage'], ['SDPA_PF: no QWEN_LLK_SDPA_PF zone at all (not executed, not instrumented, '
                                               'or its markers were dropped)'])

    def test_counter_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared = self.prepared(directory, 'llk-decode-counters')
            text = device_log(counters=True)
            self.write_log(directory, ''.join(line for line in text.splitlines(True)
                                              if not (line.startswith('1,') and ',9090,' in line)))
            summary = plan.finish_arm(IMAGE, directory, prepared, console=self.console('tag'), log=[].append)
        self.assertEqual(summary['coverage'], ['K5A: no counter rows on chip 1', 'SDPA_DEC: no counter rows on chip 1'])

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


def served_report(configuration, tokens=('a', 'b', 'c', 'd'), trace_ms=150.0, missing=()):
    """The harness report as lever_n_m3native_gate writes it: each stream carries its text (v231's keys), and
    there is no digest of it."""
    streams = [dict(chunk_s=[0.1], chunk_tokens=[48], completion_tokens=48, finish_reason='length', gaps_ms=[20.0],
                    prompt_tokens=32768, request_id='r%d' % index, started_s=float(index), text='the answer %s' % token,
                    tokens=48, ttft_s=3.0, wall_s=9.0)
               for index, token in enumerate(tokens)]
    return dict(streams=streams, qwen_configuration=configuration, ready=True, gate_passed=True,
                packed_phase=dict(rounds=4, trace_ms_mean=trace_ms), flag_markers=dict(found={}, missing=list(missing)))


class VerdictTests(unittest.TestCase):
    def test_twin(self):
        result = plan.verdict('llk-decode-twin', served_report({}), dict(exit=0), None)
        self.assertEqual(result['verdict'], 'PASS')
        self.assertEqual(plan.verdict('llk-decode-twin', None, dict(exit=125), None)['verdict'], 'FAIL')
        broken = served_report({}, tokens=('a',))
        self.assertIn('1 streams, asked 4', plan.verdict('llk-decode-twin', broken, {}, None)['reason'])
        self.assertEqual(result['tokens'], [plan.text_digest('the answer %s' % token) for token in 'abcd'])

    def test_a_stream_without_text_is_refused_never_equal(self):
        """The old check compared a key the harness never writes (text_sha256): both sides were [None] * 4 and
        always equal. A stream without text is now a FAIL, and a twin without text leaves the arm unjudged."""
        digest_only = served_report({})
        for stream in digest_only['streams']:
            stream['text_sha256'] = 'x'
            del stream['text']
        twin = plan.verdict('llk-decode-twin', digest_only, dict(exit=0), None)
        self.assertEqual(twin['verdict'], 'FAIL')
        self.assertIn("this arm's stream 0 carries no text", twin['reason'])
        zones_arm = plan.verdict('llk-decode-zones', served_report(self.configuration()), self.arm(), digest_only)
        self.assertEqual(zones_arm['verdict'], 'NOT_EXERCISED')
        self.assertIn('tokens not compared: llk-decode-twin stream 0 carries no text', zones_arm['reason'])
        both = plan.verdict('llk-decode-zones', dict(digest_only, qwen_configuration=self.configuration()), self.arm(),
                            digest_only)
        self.assertEqual(both['verdict'], 'FAIL')
        # An empty text after a token (a prefill arm's one token can detokenise to nothing) is a value.
        single = served_report({}, tokens=('a',))
        single['streams'][0]['text'] = ''
        self.assertEqual(plan.text_problems(single, 'x'), [])
        single['streams'][0].update(completion_tokens=0, tokens=0)
        self.assertEqual(plan.text_problems(single, 'x'), ['x stream 0 carries no text'])

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
        dropped = plan.verdict('llk-decode-zones', report, self.arm(complete=False, drops=dict(unmatched_end=2)), twin)
        self.assertEqual(dropped['verdict'], 'NOT_EXERCISED')
        self.assertIn('dropped markers in the analysed window', dropped['reason'])
        unshown = plan.verdict('llk-decode-zones', served_report({}), self.arm(), twin)
        self.assertEqual(unshown['verdict'], 'FAIL')
        self.assertIn('never reached the container', unshown['reason'])
        alone = plan.verdict('llk-decode-zones', report, self.arm(), None)
        self.assertEqual(alone['verdict'], 'NOT_EXERCISED')
        self.assertIn('tokens not compared: no llk-decode-twin report', alone['reason'])
        undrained = plan.verdict('llk-decode-zones', served_report(self.configuration(), missing=(
            plan.FLUSH_FLAG + ': ' + gate.PREFILL_FLUSH_MARKER,)), self.arm(), twin)
        self.assertEqual(undrained['verdict'], 'NOT_EXERCISED')
        self.assertIn('the prefill profiler drain was not seen', undrained['reason'])
        guarded = plan.verdict('llk-decode-zones', None, self.arm(disk_guard='the profile tree reached 65.0 GB'), twin)
        self.assertEqual(guarded['verdict'], 'NOT_EXERCISED')
        self.assertIn('the disk guard stopped the arm', guarded['reason'])
        empty = plan.verdict('llk-prefill-zones', served_report(self.configuration(), tokens=('a',)),
                             self.arm(coverage=report_module.coverage(dict(kernels=[]), ())), served_report({}, tokens=('a',)))
        self.assertEqual(empty['verdict'], 'NOT_EXERCISED')
        self.assertIn('no required kernel named', empty['reason'])

    def test_the_profiler_dir_may_come_back_as_tracy_writes_it(self):
        env = arms('llk-decode-zones')[0].extra['llk']['env']
        for carried, ok in ((plan.PROFILE_DIR, True), (plan.PROFILE_DIR + '/', True), (plan.PROFILE_DIR + '/.logs', True),
                            ('/opt/tt-metal/generated/profiler-other', False), ('/tmp/profiler', False), (None, False)):
            with self.subTest(carried=carried):
                configuration = dict(self.configuration(), TT_METAL_PROFILER_DIR=carried)
                problems = plan.arrival_problems(dict(qwen_configuration=configuration), env)
                self.assertEqual(problems == [], ok, problems)

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
        # A counters arm is held to its coverage and its window too (prefill-only rows no longer pass it).
        arm['llk']['coverage'] = ['K5A: no QWEN_LLK_K5A zone in a complete traced session (not executed, ...)']
        self.assertEqual(plan.verdict('llk-decode-counters', report, arm, served_report({}))['verdict'], 'NOT_EXERCISED')
        arm['llk'].update(coverage=[], complete=False)
        self.assertEqual(plan.verdict('llk-decode-counters', report, arm, served_report({}))['verdict'], 'NOT_EXERCISED')

    def test_never_ready_is_a_readiness_timeout_only(self):
        timed_out = dict(ready=False, fatal='TimeoutError: readiness exceeded 1800s')
        self.assertIn('never became ready within 1800 s', plan.never_ready(timed_out))
        self.assertIsNone(plan.never_ready(dict(ready=False, fatal='RuntimeError: server exited before readiness: 1')))
        self.assertIsNone(plan.never_ready(dict(ready=True)))
        self.assertIsNone(plan.never_ready(None))


class BudgetTests(unittest.TestCase):
    def prepared(self, directory, name):
        return plan.prepare_arm(IMAGE, directory, arms(name)[0].extra['llk'], fake_reader(), log=[].append)

    def test_the_arms_fit_the_profiler_buffer(self):
        for name in ('llk-decode-zones', 'llk-decode-counters', 'llk-prefill-zones', 'llk-prefill-counters'):
            with self.subTest(plan=name), tempfile.TemporaryDirectory() as directory:
                prepared = self.prepared(directory, name)
                self.assertNotIn('skip', prepared)
                budget = prepared['manifest']['budget']
                self.assertTrue(budget['ok'], budget)
                self.assertLessEqual(budget['fraction'], plan.BUDGET_FRACTION)
        with tempfile.TemporaryDirectory() as directory:
            budget = self.prepared(directory, 'llk-decode-zones')['manifest']['budget']
        # K5-A's 168 stage markers per program, 48 GDN layers a round, DUMP_ROUND rounds between two drains.
        self.assertEqual(budget['kernels']['K5A'], dict(markers_per_program=168, invocations=48 * plan.DUMP_ROUND))
        self.assertNotIn('MM', budget['kernels'])
        self.assertEqual(budget['window_programs'], plan.DUMP_ROUND * plan.ROUND_PROGRAMS +
                         plan.FLUSH_LAYERS * plan.LAYER_PROGRAMS)

    def test_a_window_past_the_budget_skips_that_arm_only(self):
        """v138's shape without the drains: four rounds on top of an undrained window overflow the buffer."""
        with mock.patch.object(plan, 'DUMP_ROUND', 4), tempfile.TemporaryDirectory() as directory:
            prepared = self.prepared(directory, 'llk-decode-zones')
            self.assertIn('marker budget', prepared['skip'])
            self.assertEqual(prepared['skip_scope'], 'arm')
            with open(os.path.join(directory, plan.MANIFEST), encoding='utf-8') as handle:
                self.assertEqual(json.load(handle)['skipped'], prepared['skip'])
            result = plan.preflight(IMAGE, ROOT, directory, reader=fake_reader(),
                                    generated=HandbackAndPreflightTests.generated)
        self.assertTrue(any(problem.startswith('decode at level stages: marker budget') for problem in result['problems']),
                        result['problems'])


class DiskTests(unittest.TestCase):
    def test_disk_problem(self):
        gb = plan.GB
        self.assertIsNone(plan.disk_problem('/r', 48 * gb, lambda path: (1000 * gb, 500 * gb, 500 * gb)))
        self.assertIn('free space first', plan.disk_problem('/r', 48 * gb, lambda path: (1000 * gb, 760 * gb, 240 * gb)))
        self.assertIn('free space first', plan.disk_problem('/r', 48 * gb, lambda path: (100 * gb, 10 * gb, 40 * gb)))

    def test_the_guard(self):
        gb = plan.GB
        stops = []
        calm = plan.DiskGuard('/p', 64 * gb, lambda: stops.append(1), usage=lambda path: (1000 * gb, 500 * gb, 500 * gb),
                              size=lambda path: 10 * gb)
        self.assertIsNone(calm.check())
        big = plan.DiskGuard('/p', 64 * gb, lambda: stops.append(1), usage=lambda path: (1000 * gb, 500 * gb, 500 * gb),
                             size=lambda path: 65 * gb)
        self.assertIn('the profile tree reached 65.0 GB', big.check())
        full = plan.DiskGuard('/p', 64 * gb, lambda: stops.append(1), usage=lambda path: (1000 * gb, 860 * gb, 140 * gb),
                              size=lambda path: gb)
        self.assertIn('the disk reached 86.0% used', full.check())
        lines = []
        tripping = plan.DiskGuard('/p', 64 * gb, lambda: stops.append('stop'), usage=lambda path: (1000 * gb, 0, 1000 * gb),
                                  size=lambda path: 70 * gb, interval=0.01, log=lines.append).start()
        tripping._thread.join(5)
        self.assertIn('profile tree', tripping.finish())
        self.assertEqual(stops, ['stop'])
        self.assertTrue(lines[0].startswith('[C2-GATE] disk guard: the profile tree reached'))
        broken = plan.DiskGuard('/p', 64 * gb, lambda: stops.append('never'), size=lambda path: 1 / 0, interval=0.01,
                                log=lines.append).start()
        self.assertIsNone(broken.finish())
        self.assertEqual(stops, ['stop'])


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

    @staticmethod
    def generated(image, checkout):
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
        with tempfile.TemporaryDirectory() as directory:
            result = plan.preflight(IMAGE, ROOT, directory, reader=fake_reader(drop=('SDPA_PF',)), generated=self.generated)
        self.assertEqual(result['problems'], ['prefill SDPA_PF at level tag: not in the image',
                                              'prefill SDPA_PF at level stages: not in the image'])


class FakeLlkDocker(object):
    """An arm as the container would leave it: the harness report (with the -e variables that reached it), a server
    log with the override's record lines on profiled arms, and a device log in the mounted profile tree."""

    def __init__(self, tokens=('a', 'b', 'c', 'd'), never_ready=()):
        self.calls, self.tokens, self.never_ready = [], tokens, never_ready

    def __call__(self, arguments, stdout_path, timeout, name):
        arm = name[len(driver.CONTAINER_PREFIX):]
        self.calls.append(dict(arm=arm, arguments=arguments))
        env = dict(arguments[i + 1].split('=', 1) for i, word in enumerate(arguments) if word == '-e')
        configuration = dict((key, value) for key, value in env.items() if re.match(r'(?:QWEN[0-9]*_|TT_)', key))
        prefill = 'prefill' in arm
        report = served_report(configuration, tokens=self.tokens[:1] if prefill else self.tokens)
        if arm in self.never_ready:
            report.update(ready=False, fatal='TimeoutError: readiness exceeded 1800s', streams=[])
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
                handle.write(device_log(counters=env['QWEN_LLK_ZONES'] == 'tag',
                                        kernels_=('SDPA_PF',) if prefill else ('K5A', 'SDPA_DEC'),
                                        replays=(None,) if prefill else (7, 8)))
        with open(os.path.join(arm_dir, 'server.log'), 'w') as handle:
            handle.write(log)
        return 0


class DriverTests(unittest.TestCase):
    def run_driver(self, plans, docker=None, disk=None):
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
                               llk_reader=fake_reader(), llk_handback=lambda image, path: handed.append(path) or 0,
                               llk_disk_usage=disk)
            with open(os.path.join(results, 'c2-gate-summary.json'), encoding='utf-8') as handle:
                summary = json.load(handle)
            # An arm refused before any container has no directory.
            files = dict((arm, sorted(os.listdir(os.path.join(results, arm)))) for arm in plans
                         if os.path.isdir(os.path.join(results, arm)))
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
        self.assertEqual(summary['arms']['llk-decode-counters']['llk']['counters_seen'], 8)
        self.assertIsNone(zones_arm['disk_guard'])
        self.assertTrue(any('[C2-GATE] llk-decode-zones: LLK profile: 2 kernels' in line for line in lines))

    def test_the_prefill_arms_end_to_end(self):
        code, summary, calls, lines, files, handed = self.run_driver(
            ['llk-prefill-twin', 'llk-prefill-zones', 'llk-prefill-counters'])
        verdicts = dict((name, result['verdict']) for name, result in summary['results'].items())
        self.assertEqual(verdicts, {'llk-prefill-twin': 'PASS', 'llk-prefill-zones': 'PASS', 'llk-prefill-counters': 'PASS'},
                         json.dumps(summary['results'], indent=1)[:3000])
        self.assertEqual(summary['arms']['llk-prefill-zones']['llk']['kernels'], ['SDPA_PF'])
        self.assertEqual(summary['arms']['llk-prefill-zones']['llk']['selection']['window'], 'untraced')
        self.assertEqual(code, 0)

    def test_a_full_disk_refuses_the_profiled_arms_without_a_container(self):
        gb = plan.GB
        code, summary, calls, _, _, handed = self.run_driver(
            ['llk-decode-twin', 'llk-decode-zones', 'llk-decode-counters'],
            disk=lambda path: (1000 * gb, 790 * gb, 210 * gb))
        self.assertEqual([call['arm'] for call in calls], ['llk-decode-twin'])
        for name in ('llk-decode-zones', 'llk-decode-counters'):
            self.assertEqual(summary['results'][name]['verdict'], 'NOT_EXERCISED')
            self.assertIn('free space first', summary['results'][name]['reason'])
        self.assertEqual(handed, [])

    def test_a_cold_compile_past_readiness_skips_the_later_profiled_arms(self):
        docker = FakeLlkDocker(never_ready=('llk-decode-zones',))
        code, summary, calls, _, _, _ = self.run_driver(
            ['llk-decode-twin', 'llk-prefill-twin', 'llk-decode-zones', 'llk-prefill-zones'], docker)
        self.assertEqual([call['arm'] for call in calls], ['llk-decode-twin', 'llk-prefill-twin', 'llk-decode-zones'])
        self.assertEqual(summary['results']['llk-decode-zones']['verdict'], 'FAIL')
        skipped = summary['results']['llk-prefill-zones']
        self.assertEqual(skipped['verdict'], 'NOT_EXERCISED')
        self.assertIn('skipped: a profiled arm\'s server never became ready', skipped['reason'])

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
        # Profiling only: its failure never keeps the image from the push every other job needs, and its limit
        # covers the three throwaway containers of up to 600 s each.
        self.assertIs(preflight['continue-on-error'], True)
        self.assertGreaterEqual(preflight['timeout-minutes'], 30)
        self.assertLess(names.index('LLK profiling preflight (no card)'), names.index('Push'))
        # A gate container a killed step left running is removed before the tree is handed back.
        self.assertIn("grep '^%sllk-'" % driver.CONTAINER_PREFIX, handback['run'])
        self.assertLess(handback['run'].index('docker rm -f'), handback['run'].index('llk_profile_plan.py handback'))
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
