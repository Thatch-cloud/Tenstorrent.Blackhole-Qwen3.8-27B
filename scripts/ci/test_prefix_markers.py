"""prefix_markers: what a general-prefix engine logs, parsed - held on CPU against the lines the
scheduler graft and the serving contract actually print (their own functions write them here)."""

import copy
import io
import json
import os
import sys
import unittest
from contextlib import redirect_stderr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import prefix_markers as pm  # noqa: E402
import qwen_prefix_registry as graft  # noqa: E402
import serving_c2_contract as contract  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as _handle:
    PROFILES = json.load(_handle)
PLATFORM = ['/usr/lib/python3/dist-packages/vllm/entrypoints/openai/api_server.py', '--model', 'Qwen/Qwen3.8-27B',
            '--served-model-name', 'Qwen/Qwen3.8-27B', '--host', '0.0.0.0', '--port', '8000', '--reasoning-parser',
            'qwen3', '--tool-call-parser', 'qwen3_xml', '--enable-auto-tool-choice', '--max-model-len', '65536',
            '--max-num-seqs', '2', '--block-size', '64', '--no-enable-prefix-caching', '--additional-config', '{}']


def prefix_profile():
    """general plus the design's general-prefix changes (2.0.1 item 5): the prefix cache on, chunked
    prefill on in the argv (the platform turns it off again), QWEN_PREFIX_REUSE=1."""
    profile = copy.deepcopy(PROFILES['profiles']['general'])
    engine = profile['engine']
    engine.pop('no-enable-prefix-caching')
    engine.pop('no-enable-chunked-prefill')
    engine['enable-prefix-caching'] = True
    engine['enable-chunked-prefill'] = True
    profile['env']['QWEN_PREFIX_REUSE'] = '1'
    return profile


def launched_argv(profile):
    return contract.rewrite_argv(PLATFORM, profile, profile['snapshots'][0])[1:]


def graft_line(message, *values):
    buffer = io.StringIO()
    with redirect_stderr(buffer):
        graft.log(message, *values)
    return buffer.getvalue().rstrip('\n')


def install_line():
    return graft_line('install scheduler=%s.%s plugin=%s coordinator=%s block_size=%d blocks=%d kv_spec_dtype=%s '
                      'QWEN_SDPA_BF8=%s store_gib=%.1f kill_switch=%s', 'vllm_tt_plugin.scheduler', 'TTScheduler',
                      '/opt/qwen-fast-plugin/src/vllm_tt_plugin/scheduler.py', 'UnitaryKVCacheCoordinator', 64, 4096,
                      'torch.bfloat16', '1', 8.0, graft.KILL_SWITCH_PATH)


def grant_line(req, h, q, plan):
    grant = graft.Grant(req, q, h, None, None, [(pos, None) for pos in plan], None)
    return graft_line('grant %s', grant.describe())


class LineTests(unittest.TestCase):
    def test_docker_timestamps_are_split_off(self):
        self.assertEqual(pm.split_timestamp('2026-09-26T10:00:01.123456789Z hello'),
                         ('2026-09-26T10:00:01.123456789Z', 'hello'))
        self.assertEqual(pm.split_timestamp('hello'), (None, 'hello'))

    def test_fields(self):
        self.assertEqual(pm.fields('a=1 b=2.5 c=[4096,8192] d=x e=[] f=-3'),
                         dict(a=1, b=2.5, c=[4096, 8192], d='x', e=[], f=-3))

    def test_request_tag_undoes_the_servers_two_decorations(self):
        """chatcmpl-<X-Request-Id> (serving.py) and '-<8 random>' (input_processor.assign_request_id)."""
        self.assertEqual(pm.request_tag('chatcmpl-pfx-arm-0003-hit-1a2B3c4d'), 'pfx-arm-0003-hit')
        self.assertEqual(pm.request_tag('pfx-arm-0003-hit'), 'pfx-arm-0003-hit')
        self.assertEqual(pm.request_tag('chatcmpl-abc'), 'abc')
        self.assertEqual(pm.request_tag(None), '')


class ScanTests(unittest.TestCase):
    def log(self):
        profile = prefix_profile()
        return [
            '2026-09-26T10:00:00.000000001Z (APIServer pid=1) [QWEN-C2] profile general-prefix: vLLM argv %s'
            % json.dumps(launched_argv(profile)),
            'INFO platform.py:83] Chunked prefill is not supported for `model_type=qwen3_5`; disabling it.',
            'INFO platform.py:1153] Automatic prefix caching is enabled',
            'INFO kv_cache_utils.py:2146] GPU KV cache size: 262,144 tokens',
            '(EngineCore pid=9) ' + install_line(),
            '(EngineCore pid=9) 2026-09-26 12:00:30.100 | INFO     | models.demos.blackhole.qwen36.tt.model:'
            '_qwen_prefix_dram:180 - [PINDIAG] dram after registry: chip0 allocated=25.90GB free=8.01GB '
            'largest_free=7877.5MB of 33.91GB; chip1 allocated=25.90GB free=8.01GB largest_free=7877.5MB of 33.91GB',
            grant_line('chatcmpl-pfx-a-0002-hit-deadbeef', 4160, 4096, [8192]),
            '[PREFIX] req=chatcmpl-pfx-a-0002-hit-deadbeef Q=4096 L=9000 path=traced restored_ms=210.5 '
            'captured=[8192] capture_ms=380.0 programs=1234',
            '[PREFIX] row=0 Q=0 L=77 restored_ms=0 captured=[] ms=0',
            '[PREFIX-AUDIT] req=chatcmpl-pfx-a-0002-hit-deadbeef Q=4096 L=9000 kv_range=0:9000 kv_sha=aa slot_sha=bb',
            graft_line('commit refused req=%s start_pos=%d Q=%d', 'chatcmpl-pfx-a-0009-hit-00000000', 4096, 8192),
            graft_line('capture skipped req=%s pos=%s: %s', 'chatcmpl-pfx-a-0002-hit-deadbeef', 8192, 'MemoryError()'),
            graft_line('kill switch %s present: no grants', graft.KILL_SWITCH_PATH),
            '[PINDIAG] prefix: stats {"grants": 3, "pins": 0}',
            'INFO [TP chunk-replay] 4/4 chunks',
            'ERROR llrt.cpp:594 Timed out while waiting for active ethernet core 31-25 to become active again',
        ]

    def test_every_marker_is_read(self):
        out = pm.scan(self.log())
        self.assertEqual(len(out['launches']), 1)
        self.assertEqual(out['launches'][0]['profile'], 'general-prefix')
        self.assertEqual(out['launches'][0]['time'], '2026-09-26T10:00:00.000000001Z')
        self.assertEqual(out['apc'], ['enabled'])
        self.assertEqual((out['chunking_off'], out['chunk_replay'], out['kv_tokens']), (1, 1, 262144))
        install, = out['installs']
        self.assertEqual((install['scheduler'], install['block_size'], install['QWEN_SDPA_BF8'], install['store_gib']),
                         ('vllm_tt_plugin.scheduler.TTScheduler', 64, 1, 8.0))
        grant, = out['grants']
        self.assertEqual((grant['tag'], grant['h'], grant['q'], grant['plan']), ('pfx-a-0002-hit', 4160, 4096, [8192]))
        tagged, untagged = out['rows']
        self.assertEqual((tagged['tag'], tagged['q'], tagged['l'], tagged['captured'], tagged['programs'],
                          tagged['path'], tagged['restored_ms']), ('pfx-a-0002-hit', 4096, 9000, [8192], 1234, 'traced', 210.5))
        self.assertEqual((untagged['req'], untagged['tag'], untagged['row'], untagged['capture_ms']), (None, None, 0, 0))
        audit, = out['audits']
        self.assertEqual((audit['kv_range'], audit['kv_sha'], audit['slot_sha']), ('0:9000', 'aa', 'bb'))
        refused, = out['refused']
        self.assertEqual((refused['tag'], refused['start_pos'], refused['q']), ('pfx-a-0009-hit', 4096, 8192))
        self.assertEqual(out['capture_skipped'][0]['reason'], 'MemoryError()')
        self.assertEqual(len(out['kill_switch']), 1)
        self.assertEqual(out['stats'], dict(grants=3, pins=0))
        self.assertEqual(len(out['dram']), 1)
        reading, = out['dram_readings']
        self.assertEqual((reading['point'], reading['unavailable'], [chip['chip'] for chip in reading['chips']]),
                         (pm.DRAM_REGISTRY, None, [0, 1]))
        self.assertEqual([f['signature'] for f in out['failures']], [pm.WEDGE])

    def test_the_eager_warm_line_and_the_mmio_timeout(self):
        """G1 v47 (run 36246961161): the fixed model graft's eager warm, and the host's read of the hung device."""
        out = pm.scan([
            '2026-09-26T15:03:46.300000000Z (EngineCore pid=67) 2026-09-26 15:03:46.300 | INFO | '
            'models.demos.blackhole.qwen36.tt.qwen36_vllm:_qwen_prefix_warm_eager:505 - [PINDIAG] prefix: eager prefill '
            'warmed before the decode trace: page_table_blocks=4128 programs=115->554',
            '(EngineCore pid=67) ERROR 09-26 15:04:40 [core.py:1233] RuntimeError: MMIO per-op timeout: 4B load took '
            '49571 us (budget=2 ms), 4 of 4 bytes remaining.'])
        warm, = out['eager_warm']
        self.assertEqual((warm['page_table_blocks'], warm['programs'], warm['index'], warm['time']),
                         (4128, '115->554', 0, '2026-09-26T15:03:46.300000000Z'))
        failure, = out['failures']
        self.assertEqual((failure['signature'], failure['index']), (pm.MMIO_TIMEOUT, 1))
        self.assertEqual(pm.scan(self.log())['eager_warm'], [])

    def test_by_tag_groups_and_drops_untagged(self):
        out = pm.scan(self.log())
        grouped = pm.by_tag(out['rows'])
        self.assertEqual(list(grouped), ['pfx-a-0002-hit'])

    def test_digests_stay_text(self):
        """An all-digit hex digest must not parse as an integer (leading zeros would go)."""
        row = pm.model_row('[PREFIX] req=chatcmpl-pfx-a-0001-hit-12345678 Q=0 L=10 captured=[] programs=3 '
                           'slot_sha=0123456789 logits_sha=00ab')
        self.assertEqual((row['slot_sha'], row['logits_sha'], row['programs']), ('0123456789', '00ab', 3))
        self.assertIsNone(pm.model_row('[PREFIX] req=x-12345678 Q=0 L=10')['slot_sha'])
        audit = pm.audit_row('[PREFIX-AUDIT] req=x-12345678 Q=0 L=10 kv_range=0:10 kv_sha=0042 slot_sha=0007')
        self.assertEqual((audit['kv_range'], audit['kv_sha'], audit['slot_sha']), ('0:10', '0042', '0007'))

    def test_the_stats_the_lifecycle_gates_require(self):
        """Every required counter is one the registry keeps (pins is a snapshot gauge)."""
        missing = set(pm.REQUIRED_STATS) - set(graft.STAT_NAMES) - {'pins'}
        self.assertEqual(missing, set())
        self.assertIn('pins', graft.PrefixRegistry(budget_bytes=1).snapshot())

    def test_a_design_spelling_row_with_a_request_id(self):
        row = pm.model_row('[PREFIX] row=chatcmpl-pfx-x-0001-cold-12345678 Q=0 L=10 captured=[8192,10240] ms=5')
        self.assertEqual((row['req'], row['tag'], row['captured'], row['capture_ms']),
                         ('chatcmpl-pfx-x-0001-cold-12345678', 'pfx-x-0001-cold', [8192, 10240], 5))


class DramReadingTests(unittest.TestCase):
    """G2's reading: the model graft's '[PINDIAG] dram after registry|first capture' lines, and the fast
    path's same-text lines, parsed per chip; a line without figures is read as unavailable."""

    def test_the_model_grafts_points_and_their_figures(self):
        import qwen_prefix_model_patch as model_patch

        self.assertEqual((pm.DRAM_REGISTRY, pm.DRAM_FIRST_CAPTURE),
                         (model_patch.DRAM_REGISTRY, model_patch.DRAM_FIRST_CAPTURE))
        self.assertTrue(model_patch.MARKER_DRAM.startswith(pm.DRAM))
        line = ('2026-09-26T12:00:31.000000001Z (EngineCore pid=67) 2026-09-26 12:00:31.000 | INFO     | '
                'models.demos.blackhole.qwen36.tt.model:_qwen_prefix_dram:180 - ' + model_patch.MARKER_DRAM
                + model_patch.DRAM_FIRST_CAPTURE + ': chip0 allocated=26.58GB free=7.33GB largest_free=7188.4MB of '
                '33.91GB; chip1 allocated=26.60GB free=7.31GB largest_free=7100.0MB of 33.91GB')
        out = pm.scan([line])
        reading, = out['dram_readings']
        self.assertEqual((reading['point'], reading['unavailable'], reading['time'], reading['index']),
                         (pm.DRAM_FIRST_CAPTURE, None, '2026-09-26T12:00:31.000000001Z', 0))
        self.assertEqual(reading['chips'], [
            dict(chip=0, allocated_gb=26.58, free_gb=7.33, largest_free_mb=7188.4, total_gb=33.91),
            dict(chip=1, allocated_gb=26.60, free_gb=7.31, largest_free_mb=7100.0, total_gb=33.91)])
        self.assertEqual(out['dram'], [line[len('2026-09-26T12:00:31.000000001Z '):]])

    def test_an_unavailable_view_and_a_line_without_figures(self):
        refused = pm.dram_reading('[PINDIAG] dram after registry: unavailable (RuntimeError: no allocator on this device)')
        self.assertEqual((refused['point'], refused['chips'], refused['unavailable']),
                         ('registry', [], 'RuntimeError: no allocator on this device'))
        garbled = pm.dram_reading('[PINDIAG] dram after registry: 7 GB or so')
        self.assertEqual((garbled['chips'], garbled['unavailable']), ([], 'no per-chip figures: 7 GB or so'))
        self.assertIsNone(pm.dram_reading('[PINDIAG] prefix: stats {"bytes": 0}'))

    def test_the_fast_paths_lines_parse_the_same_way(self):
        engine = pm.dram_reading('(EngineCore pid=66) 2026-09-26 11:50:50.803 | INFO     | dflash_device:pindiag:30 - '
                                 '[PINDIAG] dram after engine chatcmpl-b5c7269b0809cd10-b2da3b07: chip0 allocated=26.58GB '
                                 'free=7.33GB largest_free=7188.4MB of 33.91GB')
        self.assertEqual((engine['point'], engine['chips'][0]['free_gb']), ('engine chatcmpl-b5c7269b0809cd10-b2da3b07', 7.33))


class ArgvTests(unittest.TestCase):
    def test_the_contracts_rewrite_of_a_prefix_profile_passes(self):
        """The platform's --no-enable-prefix-caching is dropped (the contract owns it) and the
        profile's flags land: the argv the gate expects is the one the contract launches."""
        argv = launched_argv(prefix_profile())
        self.assertNotIn('--no-enable-prefix-caching', argv)
        self.assertEqual(pm.prefix_argv_problems(argv), [])
        flags = pm.argv_flags(argv)
        self.assertEqual((flags['block-size'], flags['max-num-seqs'], flags['enable-chunked-prefill']), ('64', '4', True))

    def test_general_fails_every_prefix_check_it_should(self):
        problems = pm.prefix_argv_problems(launched_argv(PROFILES['profiles']['general']))
        self.assertEqual(len(problems), 2, problems)
        self.assertTrue(any('does not enable the prefix cache' in p for p in problems))
        self.assertTrue(any('still carries --no-enable-prefix-caching' in p for p in problems))

    def test_a_missing_argv_and_bad_block_size(self):
        self.assertEqual(len(pm.prefix_argv_problems(None)), 1)
        bad = ['--enable-prefix-caching', '--block-size=128']
        problems = pm.prefix_argv_problems(bad)
        self.assertTrue(any('block size' in p for p in problems))
        self.assertTrue(any('async' in p for p in problems))

    def test_argv_flags_reads_both_spellings(self):
        self.assertEqual(pm.argv_flags(['--a', '1', '--b', '--c=2', '--d']), dict(a='1', b=True, c='2', d=True))


class PrometheusTests(unittest.TestCase):
    def test_labels_are_summed_and_counters_lose_their_suffix(self):
        text = ('# HELP vllm:num_preemptions_total x\n# TYPE vllm:num_preemptions_total counter\n'
                'vllm:num_preemptions_total{engine="0",model_name="Q"} 3.0\n'
                'vllm:num_preemptions_total{engine="1",model_name="Q"} 2.0\n'
                'vllm:num_requests_waiting{engine="0"} 1.0\nbroken line\n')
        values = pm.parse_prometheus(text)
        self.assertEqual(values['vllm:num_preemptions'], 5.0)
        self.assertEqual(values['vllm:num_requests_waiting'], 1.0)
        self.assertEqual(pm.parse_prometheus(None), {})


if __name__ == '__main__':
    unittest.main()
