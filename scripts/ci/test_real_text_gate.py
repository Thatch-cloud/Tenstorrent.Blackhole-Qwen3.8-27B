"""The M3native gate's real-text mode: options, the stream payload, the run, and the v157-v160 tags.

The defaults must leave every existing arm exactly as it was: synthetic prompts, ignore_eos=True,
and stream_once called with no keywords at all (so its payload and its recorded fields are the
same bytes). Real text is opt-in, refuses the combinations the fast path cannot serve, builds its
prompts before the server starts, and records what the offline compare needs."""

import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
import re
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lever_n_m3native_gate as gate  # noqa: E402
import longctx_cycle_bench  # noqa: E402
import real_text_prompts  # noqa: E402
from test_real_text_prompts import FakeTokenizer, code_corpus, write_package  # noqa: E402

HERE = Path(__file__).resolve().parent
WORKFLOW = HERE.parents[1] / '.github' / 'workflows' / 'qwen-lever-n-m3native-gate.yml'
IMAGE_A5 = 'sha256:126b30dfa72b0e008884f3a8a1cfcb5b7f79eadcdeaeda1cd0f91350dde6ee73'


def parse(*argv):
    with redirect_stdout(io.StringIO()), mock.patch.object(sys, 'stderr', io.StringIO()):
        return gate.parse_options(list(argv))


class OptionTests(unittest.TestCase):
    def test_defaults_are_todays_behaviour(self):
        options = parse()
        self.assertEqual((options.prompt_source, options.eos), ('synthetic', 'ignore'))
        self.assertEqual(gate.stream_kwargs(options), {})
        self.assertEqual((options.users, options.context, options.prompt_tokens, options.max_tokens),
                         (4, 33024, 32768, 256))

    def test_real_text_needs_allow_missing_references(self):
        with self.assertRaises(SystemExit):
            parse('--prompt-source', 'real-text')
        options = parse('--prompt-source', 'real-text', '--allow-missing-references')
        self.assertEqual(options.eos, 'stop', 'EOS is honoured by default on real text')
        self.assertEqual(gate.stream_kwargs(options), dict(ignore_eos=False, detail=True))

    def test_real_text_refuses_ignore_eos(self):
        with self.assertRaises(SystemExit):
            parse('--prompt-source', 'real-text', '--allow-missing-references', '--eos', 'ignore')

    def test_synthetic_may_honour_eos(self):
        options = parse('--eos', 'stop')
        self.assertEqual(gate.stream_kwargs(options), dict(ignore_eos=False, detail=True))
        with self.assertRaises(SystemExit):
            parse('--prompt-source', 'prose')

    def test_the_target_leaves_room_for_the_budget_in_the_arms_context(self):
        self.assertEqual(gate.real_text_target(parse()), 32768)
        self.assertEqual(gate.real_text_target(parse('--context', '131328', '--prompt-tokens', '131072')), 131072)
        self.assertEqual(gate.real_text_target(parse('--context', '33024', '--max-tokens', '512')), 32512)
        self.assertEqual(gate.real_text_target(parse('--prompt-tokens', '2048')), 2048)
        with self.assertRaises(SystemExit):
            parse('--prompt-source', 'real-text', '--allow-missing-references', '--context', '256')

    def test_real_text_prompts_must_be_exactly_prompt_tokens(self):
        """The image pins each request's position to QWEN_DSPARK_REQUEST_CONTEXT (= --prompt-tokens):
        a budget that would shorten the prompt below it is refused up front, not on the rig."""
        real = ('--prompt-source', 'real-text', '--allow-missing-references')
        self.assertEqual(gate.real_text_target(parse(*real)), 32768)
        self.assertEqual(gate.real_text_target(parse(*real + ('--context', '131328', '--prompt-tokens', '131072'))),
                         131072)
        for argv in (('--max-tokens', '512'), ('--context', '32768')):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                parse(*real + argv)


class FakeResponse:
    def __init__(self, lines):
        self.lines = [line.encode() for line in lines]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self.lines)


def sse(*chunks):
    return ['data: %s\n' % json.dumps(chunk) for chunk in chunks] + ['data: [DONE]\n']


CHUNKS = sse(
    dict(id='cmpl-abc', choices=[dict(index=0, text='Hel', finish_reason=None)],
         usage=dict(prompt_tokens=10, completion_tokens=1, total_tokens=11)),
    dict(id='cmpl-abc', choices=[dict(index=0, text='lo wor', finish_reason=None)],
         usage=dict(prompt_tokens=10, completion_tokens=8, total_tokens=18)),
    dict(id='cmpl-abc', choices=[dict(index=0, text='', finish_reason='stop')],
         usage=dict(prompt_tokens=10, completion_tokens=9, total_tokens=19)),
    dict(id='cmpl-abc', choices=[], usage=dict(prompt_tokens=10, completion_tokens=9, total_tokens=19)))


class FakeWatch(object):
    """stream_once's side of a StreamWatch: records the calls, drops at `drop_at` chunks, and says it
    cancelled the stream when `cancelled` is set."""

    def __init__(self, drop_at=None, cancelled=None):
        self.drop_at, self.reason, self.calls = drop_at, cancelled, []

    def begin(self, user, when):
        self.calls.append(('begin', user))

    def opened(self, user, sock):
        self.calls.append(('opened', user, sock))

    def chunk(self, user, tokens, when):
        self.calls.append(('chunk', user, tokens))
        return 'after %d chunks' % tokens if self.drop_at is not None and tokens >= self.drop_at else None

    def end(self, user, when):
        self.calls.append(('end', user))

    def cancelled(self, user):
        return self.reason


class StreamOnceTests(unittest.TestCase):
    def run_stream(self, **kwargs):
        sent = []

        def urlopen(request, timeout):
            sent.append(request.data)
            return FakeResponse(CHUNKS)

        results = [None]
        with mock.patch.object(longctx_cycle_bench, 'urlopen', side_effect=urlopen):
            longctx_cycle_bench.stream_once(8000, [5, 6, 7], 256, results, 0, 600, **kwargs)
        return sent[0], results[0]

    def test_the_default_payload_and_record_are_byte_identical_to_before(self):
        payload, entry = self.run_stream()
        legacy = json.dumps(dict(model='qwen-longctx', prompt=[5, 6, 7], max_tokens=256, temperature=0.0,
                                 stream=True, stream_options=dict(include_usage=True), ignore_eos=True)).encode()
        self.assertEqual(payload, legacy)
        self.assertEqual(sorted(entry), ['completion_tokens', 'gaps_ms', 'prompt_tokens', 'text', 'tokens', 'ttft_s',
                                         'wall_s'])
        self.assertEqual((entry['text'], entry['tokens'], entry['completion_tokens']), ('Hello wor', 2, 9))

    def test_honouring_eos_with_detail_records_rounds_the_id_and_the_finish(self):
        payload, entry = self.run_stream(ignore_eos=False, detail=True)
        sent = json.loads(payload)
        self.assertIs(sent['ignore_eos'], False)
        self.assertEqual(sent['stream_options'], dict(include_usage=True, continuous_usage_stats=True))
        self.assertEqual(entry['chunk_tokens'], [1, 7, 1], 'the EOS-only chunk carries a token and no text')
        self.assertEqual(len(entry['chunk_s']), 3)
        self.assertEqual((entry['request_id'], entry['finish_reason']), ('cmpl-abc', 'stop'))
        self.assertEqual((entry['tokens'], entry['text']), (2, 'Hello wor'), 'text chunks are counted as before')
        self.assertIsInstance(entry['started_s'], float)

    def test_a_served_name_changes_only_the_model_field(self):
        payload, _ = self.run_stream(model='Qwen/Qwen3.8-27B')
        legacy, _ = self.run_stream()
        self.assertEqual(json.loads(payload), dict(json.loads(legacy), model='Qwen/Qwen3.8-27B'))

    def test_a_watch_sees_the_stream_and_a_drop_at_a_chunk_is_not_an_error(self):
        watch = FakeWatch(drop_at=1)
        _, entry = self.run_stream(ignore_eos=False, detail=True, watch=watch)
        self.assertEqual((entry['dropped'], entry['text'], entry['tokens']), ('after 1 chunks', 'Hel', 1))
        self.assertNotIn('error', entry)
        self.assertEqual([call[0] for call in watch.calls], ['begin', 'opened', 'chunk', 'end'])
        watch = FakeWatch(drop_at=5)
        _, entry = self.run_stream(ignore_eos=False, detail=True, watch=watch)
        self.assertNotIn('dropped', entry, 'fewer chunks than the drop: the stream ran to its end')
        self.assertEqual([call[0] for call in watch.calls], ['begin', 'opened', 'chunk', 'chunk', 'end'])

    def test_a_cancel_before_the_first_byte_is_a_drop_and_the_limit_stays_the_stream_timeout(self):
        """Review finding 14: the old @S cancel was the socket timeout, still in force after the first
        byte. The watch cancels by shutting the socket down; the timeout is stream_timeout throughout."""
        timeouts = []

        class Cancelled(FakeResponse):
            def __iter__(self):
                raise ConnectionResetError('shut down by the watch')

        def urlopen(request, timeout):
            timeouts.append(timeout)
            return Cancelled([])

        results = [None]
        watch = FakeWatch(cancelled='no byte within 2.5 s')
        with mock.patch.object(longctx_cycle_bench, 'urlopen', side_effect=urlopen):
            longctx_cycle_bench.stream_once(8000, [1], 8, results, 0, 600, ignore_eos=False, detail=True, watch=watch)
        self.assertEqual(timeouts, [600])
        self.assertEqual((results[0]['dropped'], results[0]['text']), ('no byte within 2.5 s', ''))
        self.assertNotIn('error', results[0])
        # A shut-down socket may also read as a clean end of the stream.
        with mock.patch.object(longctx_cycle_bench, 'urlopen', return_value=FakeResponse([])):
            longctx_cycle_bench.stream_once(8000, [1], 8, results, 0, 600, ignore_eos=False, detail=True, watch=watch)
        self.assertEqual(results[0]['dropped'], 'no byte within 2.5 s')
        # Any failure the watch did not cause is still an error.
        with mock.patch.object(longctx_cycle_bench, 'urlopen', side_effect=ConnectionResetError('reset')):
            longctx_cycle_bench.stream_once(8000, [1], 8, results, 0, 600, watch=FakeWatch())
        self.assertIn('ConnectionResetError', results[0]['error'])
        self.assertNotIn('dropped', results[0])

    def test_the_socket_under_a_response_is_found_for_the_watch(self):
        sock = object()
        response = mock.Mock(fp=mock.Mock(raw=mock.Mock(_sock=sock)))
        self.assertIs(longctx_cycle_bench.response_socket(response), sock)
        self.assertIsNone(longctx_cycle_bench.response_socket(FakeResponse([])))

    def test_a_failed_detail_stream_keeps_what_arrived(self):
        def urlopen(request, timeout):
            raise TimeoutError('timed out')
        results = [None]
        with mock.patch.object(longctx_cycle_bench, 'urlopen', side_effect=urlopen):
            longctx_cycle_bench.stream_once(8000, [1], 8, results, 0, 1, ignore_eos=False, detail=True)
        self.assertIn('TimeoutError', results[0]['error'])
        self.assertEqual(results[0]['chunk_tokens'], [])


class RunTests(unittest.TestCase):
    """main() end to end with the server, the streams and the tokenizer faked."""

    def run_gate(self, argv, environ=None, served=lambda index, length: length):
        calls = []

        def stream(port, prompt, max_tokens, results, index, timeout, **kwargs):
            calls.append(dict(index=index, prompt=list(prompt), kwargs=kwargs))
            results[index] = dict(text='answer %d' % index, finish_reason='length' if kwargs else None,
                                  tokens=3, completion_tokens=5, gaps_ms=[250.0, 250.0], ttft_s=1.0,
                                  prompt_tokens=served(index, len(prompt)))

        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory, 'results')
            package = write_package(Path(directory, 'site'), code_corpus(files=80))
            with mock.patch.object(gate, 'start_server', return_value=(None, None, results / 'server.log', ['x'])), \
                    mock.patch.object(gate, 'stop_server'), \
                    mock.patch.object(gate, 'stream_once', side_effect=stream), \
                    mock.patch.object(real_text_prompts, 'load_tokenizer', return_value=FakeTokenizer()), \
                    mock.patch.object(real_text_prompts, 'package_root', return_value=package), \
                    mock.patch.dict(os.environ, environ or {}), \
                    mock.patch.object(sys, 'argv', ['gate'] + argv + ['--results', str(results),
                                                                     '--references', str(results)]), \
                    redirect_stdout(io.StringIO()) as out:
                code = gate.main()
            text = out.getvalue()
            report = json.loads(text[text.index(gate.BEGIN) + len(gate.BEGIN):text.index(gate.END)])
            written = sorted(p.name for p in results.glob('*.json'))
            prompts = json.loads((results / gate.REAL_TEXT_PROMPTS).read_text(encoding='utf-8')) \
                if (results / gate.REAL_TEXT_PROMPTS).is_file() else None
        return code, calls, report, written, prompts, text

    def test_a_real_text_run_builds_serves_and_records_the_prompts(self):
        code, calls, report, written, prompts, text = self.run_gate(
            ['--prompt-source', 'real-text', '--allow-missing-references', '--prompt-tokens', '3000'],
            environ={'QWEN_FAST_GDN_PREFILL_CONV': '1', 'QWEN35_GDN_STATE_BF16': '1', 'TT_METAL_CACHE': '/k',
                     'OMP_NUM_THREADS': '8', 'THATCH_SERVING_PORT': '8000'})
        self.assertEqual(code, 1, 'no packed-round evidence in a faked run')
        self.assertEqual(sorted(c['index'] for c in calls), [0, 1, 2, 3])
        self.assertTrue(all(c['kwargs'] == dict(ignore_eos=False, detail=True) for c in calls))
        lengths = report['real_text']['prompt_lengths']
        self.assertEqual([len(c['prompt']) for c in sorted(calls, key=lambda c: c['index'])], lengths)
        self.assertEqual(lengths, [3000] * 4, 'every prompt is exactly the request context the image pins')
        self.assertEqual(report['real_text_target'], 3000)
        self.assertEqual((report['prompt_source'], report['eos'], report['references_loaded']), ('real-text', 'stop', []))
        self.assertEqual(report['max_tokens'], 256)
        self.assertEqual(report['qwen_configuration'].get('QWEN_FAST_GDN_PREFILL_CONV'), '1')
        # Review finding 13: QWEN<n>_*, TT_*, MESH_DEVICE and OMP_NUM_THREADS are arithmetic too.
        self.assertEqual(report['configuration_scope'], 'qwen-tt')
        for name, value in (('QWEN35_GDN_STATE_BF16', '1'), ('TT_METAL_CACHE', '/k'), ('OMP_NUM_THREADS', '8')):
            self.assertEqual(report['qwen_configuration'].get(name), value)
        self.assertNotIn('THATCH_SERVING_PORT', report['qwen_configuration'])
        self.assertTrue(all(gate.CONFIGURATION_PREFIX.match(name) or name in gate.CONFIGURATION_NAMES
                            for name in report['qwen_configuration']))
        self.assertEqual({(c['served_prompt_tokens'], c['completion_tokens'], c['max_tokens'])
                          for c in report['comparisons']}, {(3000, 5, 256)})
        self.assertNotIn('tokens', report['real_text']['users'][0])
        self.assertEqual([c['prompt_sha256'] for c in report['comparisons']],
                         [u['prompt_sha256'] for u in report['real_text']['users']])
        self.assertEqual({c['finish_reason'] for c in report['comparisons']}, {'length'})
        self.assertEqual(prompts['users'][2]['tokens'], sorted(calls, key=lambda c: c['index'])[2]['prompt'])
        self.assertEqual(written, [gate.REAL_TEXT_PROMPTS], 'no reference candidate from a concurrent run')
        self.assertEqual(report['real_text_stream_problems'], [])
        self.assertIn('acceptance', report)
        self.assertIn('decode_rate', report)
        self.assertIn('[ACCEPT] ', text)
        self.assertIn('[RATE] ', text)
        self.assertIn('[REALTEXT] 4 prompts of ', text)
        # The prefill-conv floor is the synthetic arm's: 4 users x ceil(3000 / 2048).
        self.assertEqual(report['flag_markers']['prefill_conv']['required_chunks'], 4 * 2)

    def test_a_prompt_the_server_counts_differently_fails_the_run(self):
        _, _, report, _, _, _ = self.run_gate(
            ['--prompt-source', 'real-text', '--allow-missing-references', '--prompt-tokens', '3000'],
            served=lambda index, length: length - 1 if index == 2 else length)
        self.assertEqual(report['real_text_stream_problems'],
                         ['user 2: the server reports usage.prompt_tokens 2999 for a 3000-token prompt'])
        self.assertIs(report['gate_passed'], False)

    def test_a_sequential_real_text_run_builds_the_same_prompts_and_names_its_candidates_by_hash(self):
        _, concurrent_calls, concurrent, _, _, _ = self.run_gate(
            ['--prompt-source', 'real-text', '--allow-missing-references', '--prompt-tokens', '3000'])
        _, calls, report, written, _, _ = self.run_gate(
            ['--users', '1', '--sequential-users', '4', '--prompt-source', 'real-text',
             '--allow-missing-references', '--prompt-tokens', '3000'])
        self.assertEqual([u['prompt_sha256'] for u in report['real_text']['users']],
                         [u['prompt_sha256'] for u in concurrent['real_text']['users']])
        self.assertEqual([c['index'] for c in calls], [0, 1, 2, 3])
        expected = sorted(['reference-candidate-realtext-%s-p%d.json' % (u['prompt_sha256'][:12], u['prompt_tokens'])
                           for u in report['real_text']['users']] + [gate.REAL_TEXT_PROMPTS])
        self.assertEqual(written, expected)
        self.assertFalse(any(gate.REFERENCE_NAME.match(name) for name in written))

    def test_the_default_run_passes_no_keywords_and_builds_no_prompts(self):
        code, calls, report, written, prompts, text = self.run_gate(['--prompt-tokens', '128'])
        self.assertTrue(all(c['kwargs'] == {} for c in calls))
        self.assertEqual([c['prompt'][:2] for c in sorted(calls, key=lambda c: c['index'])],
                         [[1000, 1001], [1001, 1002], [1002, 1003], [1003, 1004]])
        self.assertIsNone(prompts)
        # The default arm's report and stdout carry exactly today's keys and lines.
        for key in ('real_text', 'real_text_stream_problems', 'prompt_source', 'eos', 'max_tokens',
                    'qwen_configuration', 'acceptance', 'decode_rate'):
            self.assertNotIn(key, report)
        self.assertNotIn('[ACCEPT]', text)
        self.assertNotIn('[RATE]', text)

    def test_a_broken_stream_fails_a_real_text_run(self):
        self.assertEqual(gate.real_text_stream_problems([dict(text='x', finish_reason='length'),
                                                         dict(text='y', finish_reason='stop')]), [])
        problems = gate.real_text_stream_problems([dict(error='boom'), dict(text=''), dict(text='z'), None])
        self.assertEqual(len(problems), 4)
        served = [dict(text='x', finish_reason='length', prompt_tokens=10), dict(text='y', finish_reason='stop')]
        self.assertEqual(gate.real_text_stream_problems(served, [10, 10]),
                         ['user 1: the server reports usage.prompt_tokens None for a 10-token prompt'])

    def test_a_diagnostic_failure_never_reaches_the_verdict(self):
        report = dict(streams=[dict(text='x')], gate_passed=True)
        with mock.patch.dict(sys.modules, {'acceptance_report': None}), redirect_stdout(io.StringIO()):
            gate.add_run_diagnostics(report, Path('does-not-exist.log'), sequential=False)
        self.assertIn('error', report['acceptance'])
        self.assertIn('error', report['decode_rate'])
        self.assertIs(report['gate_passed'], True)

    def test_a_diagnostic_json_cannot_carry_is_caught_before_the_report_prints(self):
        import acceptance_report
        report = dict(streams=[dict(text='x')], gate_passed=True)
        broken = dict(summary_line='[ACCEPT] x', users=[], value=object())
        with mock.patch.object(acceptance_report, 'report', return_value=broken), redirect_stdout(io.StringIO()):
            gate.add_run_diagnostics(report, Path('does-not-exist.log'), sequential=True)
        self.assertIn('TypeError', report['acceptance']['error'])
        self.assertNotIn('error', report['decode_rate'])
        json.dumps(report)


V235 = HERE / 'references' / 'c2-serving' / 'v235-real-text-4x131072.json'
C2_ARGV_LINE = ('[QWEN-C2] profile exact: vLLM argv ["--served-model-name", "Qwen/Qwen3.8-27B", "--host", "0.0.0.0", '
                '"--port", "8000", "--max-model-len", "131328"]')
DRAM_LINES = (
    '(EngineCore pid=88) 2026-09-25 02:41:42.032 | INFO     | dflash_device:pindiag:30 - [PINDIAG] dram after attach: '
    'chip0 allocated=29.74GB free=4.17GB largest_free=4007.1MB of 33.91GB; chip1 allocated=29.74GB free=4.17GB '
    'largest_free=4007.1MB of 33.91GB\n'
    '(EngineCore pid=88) 2026-09-25 02:42:41.853 | INFO     | dflash_device:pindiag:30 - [PINDIAG] dram after engine '
    'cmpl-ba5c3c305bfa5ec4-0-8161b507: chip0 allocated=30.42GB free=3.49GB largest_free=3309.9MB of 33.91GB; chip1 '
    'allocated=30.42GB free=3.49GB largest_free=3309.9MB of 33.91GB\n'
    '(EngineCore pid=88) 2026-09-25 02:45:29.184 | INFO     | dflash_device:pindiag:30 - [PINDIAG] dram after engine '
    'cmpl-8ddc72492c897448-0-a8b5e0cc: chip0 allocated=32.59GB free=1.32GB largest_free=1123.8MB of 33.91GB; chip1 '
    'allocated=32.60GB free=1.31GB largest_free=1120.0MB of 33.91GB\n')


class PlatformArgvTests(unittest.TestCase):
    """--server-argv platform: the C2 serving image's contract serves, the gate reads back what it launched."""

    PLATFORM = ('--server-argv', 'platform')
    REAL = ('--prompt-source', 'real-text', '--allow-missing-references')

    def test_the_default_is_the_gate_argv_and_no_model_keyword(self):
        options = parse()
        self.assertEqual((options.server_argv, options.prompt_lengths, options.readiness_seconds), ('gate', None, 900))
        self.assertEqual(gate.stream_kwargs(options), {})
        self.assertEqual(options.snapshot, gate.MODEL)

    def test_platform_streams_name_the_platforms_model(self):
        self.assertEqual(gate.stream_kwargs(parse(*self.PLATFORM)), dict(model='Qwen/Qwen3.8-27B'))
        options = parse(*self.PLATFORM + self.REAL + ('--context', '131328', '--prompt-tokens', '131072'))
        self.assertEqual(gate.stream_kwargs(options), dict(ignore_eos=False, detail=True, model='Qwen/Qwen3.8-27B'))

    def test_the_platform_argv_is_the_smokes(self):
        argv = gate.platform_argv(8000)
        self.assertEqual(argv[1:3], ['-m', 'vllm.entrypoints.openai.api_server'])
        workflow = (HERE.parents[1] / '.github' / 'workflows' / 'qwen-c2-serving.yml').read_text(encoding='utf-8')
        smoke = workflow[workflow.index('- name: Smoke on cards M+A'):workflow.index('- name: Replay')]
        for flag in ('--reasoning-parser', '--tool-call-parser', '--enable-auto-tool-choice', '--max-model-len',
                     '--max-num-seqs', '--block-size', '--no-enable-prefix-caching', '--additional-config'):
            self.assertIn(flag, argv)
            self.assertIn(flag, smoke)
        self.assertIn("'%s'" % argv[argv.index('--additional-config') + 1], smoke)

    def test_the_exact_profile_rewrites_the_platform_argv_to_v235s_engine(self):
        """What the contract makes of the platform argv under 'exact' is v235's served engine: every
        flag and value of run 36087022223's command but the served name, host and port (the
        platform's) and the snapshot path (the agent mounts the hub at /models)."""
        import serving_c2_contract as contract
        profile = contract.load_profile(str(HERE / 'qwen_c2_profiles.json'), 'exact')
        snapshot = profile['snapshots'][0]
        platform = gate.platform_argv(8000)
        served = contract.rewrite_argv(['/opt/venv/lib/python3.10/site-packages/vllm/entrypoints/openai/api_server.py']
                                       + platform[3:], profile, snapshot)[1:]
        v235 = json.loads(V235.read_text(encoding='utf-8'))['command'][3:]
        platform_own = ('--served-model-name', '--host', '--port')

        def engine(argv, drop):
            pairs, index = {}, 0
            while index < len(argv):
                flag = argv[index]
                value = argv[index + 1] if index + 1 < len(argv) and not argv[index + 1].startswith('--') else None
                index += 2 if value is not None else 1
                if flag not in drop:
                    pairs[flag] = value
            return pairs

        mine = engine(served, platform_own + ('--reasoning-parser', '--tool-call-parser', '--enable-auto-tool-choice'))
        theirs = engine(v235, platform_own)
        self.assertEqual(mine.pop('--model'), snapshot)
        self.assertEqual(theirs.pop('--model'), snapshot.replace('/models/', '/models/hub/'))
        mine['--additional-config'] = json.loads(mine['--additional-config'].replace(snapshot, '@'))
        theirs['--additional-config'] = json.loads(theirs['--additional-config'].replace(
            snapshot.replace('/models/', '/models/hub/'), '@'))
        self.assertEqual(mine, theirs)
        # The bring-up's own check (c2_serving_gate.bringup_verdict) is this normalisation.
        import real_text_compare
        command = json.loads(V235.read_text(encoding='utf-8'))['command']
        self.assertEqual(real_text_compare.served_engine_diff(served, command), {})
        dropped = [token for token in served if token != '--no-enable-chunked-prefill']
        self.assertEqual(real_text_compare.served_engine_diff(dropped, command),
                         {'--no-enable-chunked-prefill': [None, True]}, 'a flag present on one side only differs')
        self.assertIsNone(real_text_compare.served_engine_diff('not a list', command))

    def test_prompt_lengths_are_real_text_only_one_per_stream_and_fit_the_context(self):
        options = parse(*self.PLATFORM + self.REAL + ('--context', '131328', '--users', '3',
                                                      '--prompt-lengths', '60,2048,123136', '--max-tokens', '16384'))
        self.assertEqual(options.prompt_lengths, [60, 2048, 123136])
        self.assertIsNone(gate.real_text_target(options))
        sequential = parse(*self.PLATFORM + self.REAL + ('--context', '131328', '--users', '1', '--sequential-users',
                                                         '2', '--prompt-lengths', '255,4096'))
        self.assertEqual(sequential.prompt_lengths, [255, 4096])
        for argv in (('--prompt-lengths', '60,255'),                                        # synthetic
                     self.REAL + ('--prompt-lengths', '60,255'),                            # 2 lengths, 4 users
                     self.REAL + ('--users', '2', '--prompt-lengths', '60,x'),
                     self.REAL + ('--users', '2', '--prompt-lengths', '60,33000'),          # past the gate context
                     self.PLATFORM + self.REAL + ('--users', '1', '--prompt-lengths', '33024'),
                     self.REAL + ('--users', '1', '--prompt-lengths', '32769'),             # gate: + max_tokens
                     ('--expect-profile', 'exact'),                                         # needs platform
                     ('--readiness-seconds', '0')):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                parse(*argv)
        self.assertEqual(parse(*self.PLATFORM + self.REAL + ('--users', '1', '--prompt-lengths', '33023'))
                         .prompt_lengths, [33023], 'the contract clamps max_tokens; one token of room is enough')

    def test_the_launched_argv_is_read_back_from_the_contracts_line(self):
        text = 'noise\n(APIServer pid=1) ' + C2_ARGV_LINE + '\n[QWEN-C2] request contract installed: greedy\n'
        report = gate.platform_report(text, 'exact')
        self.assertEqual(report['served_profile'], 'exact')
        self.assertEqual(report['served_argv'][-2:], ['--max-model-len', '131328'])
        self.assertEqual((report['problems'], report['contract_installed'], report['launches']), ([], True, 1))
        self.assertEqual(len(report['qwen_c2_lines']), 2)
        self.assertEqual(gate.platform_report(text, 'c2')['problems'], ["served profile 'exact', expected 'c2'"])
        missing = gate.platform_report('vLLM API server version 0.25.1\n', 'exact')
        self.assertEqual(len(missing['problems']), 1)
        self.assertIn('did not rewrite', missing['problems'][0])
        two = gate.platform_report(text + C2_ARGV_LINE.replace('exact', 'general') + '\n')
        self.assertIn('2 different launched argv lines', two['problems'][0])

    def test_dram_lines_are_parsed_per_chip_with_the_engine_floor(self):
        report = gate.dram_report('x\n' + DRAM_LINES)
        self.assertEqual([e['event'] for e in report['events']], ['attach', 'engine', 'engine'])
        self.assertEqual(report['events'][1]['request'], 'cmpl-ba5c3c305bfa5ec4-0-8161b507')
        self.assertEqual(report['events'][2]['chips'][1], dict(chip=1, allocated_gb=32.6, free_gb=1.31,
                                                               largest_free_mb=1120.0, total_gb=33.91))
        self.assertEqual((report['engines'], report['min_free_gb'], report['min_largest_free_mb']), (2, 1.31, 1120.0))
        self.assertEqual(gate.dram_report(''), dict(events=[], engines=0, min_free_gb=None, min_largest_free_mb=None))

    def run_platform(self, argv, log_text):
        calls = []

        def stream(port, prompt, max_tokens, results, index, timeout, **kwargs):
            calls.append(dict(index=index, prompt=list(prompt), kwargs=kwargs, max_tokens=max_tokens))
            results[index] = dict(text='answer %d' % index, finish_reason='stop', tokens=3, completion_tokens=5,
                                  gaps_ms=[250.0], ttft_s=1.0, prompt_tokens=len(prompt))

        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory, 'results')
            results.mkdir()
            (results / 'server.log').write_text(log_text, encoding='utf-8')
            package = write_package(Path(directory, 'site'), code_corpus(files=200))
            started = []

            def start(*args, **kwargs):
                started.append(kwargs)
                return None, None, results / 'server.log', kwargs.get('command') or ['gate-argv']

            with mock.patch.object(gate, 'start_server', side_effect=start), \
                    mock.patch.object(gate, 'stop_server'), \
                    mock.patch.object(gate, 'stream_once', side_effect=stream), \
                    mock.patch.object(real_text_prompts, 'load_tokenizer', return_value=FakeTokenizer()), \
                    mock.patch.object(real_text_prompts, 'package_root', return_value=package), \
                    mock.patch.object(sys, 'argv', ['gate'] + argv + ['--results', str(results),
                                                                     '--references', str(results)]), \
                    redirect_stdout(io.StringIO()) as out:
                code = gate.main()
            text = out.getvalue()
            report = json.loads(text[text.index(gate.BEGIN) + len(gate.BEGIN):text.index(gate.END)])
        return code, calls, report, started

    def test_a_platform_run_serves_the_platform_argv_and_records_what_the_contract_launched(self):
        argv = list(self.PLATFORM + self.REAL) + ['--context', '131328', '--users', '3', '--prompt-lengths',
                                                  '200,3000,2049', '--max-tokens', '4096', '--expect-profile', 'exact',
                                                  '--readiness-seconds', '1800', '--snapshot', '/models/snap']
        code, calls, report, started = self.run_platform(argv, C2_ARGV_LINE + '\n' + DRAM_LINES)
        self.assertEqual(started[0]['command'], gate.platform_argv(8000))
        self.assertEqual(started[0]['readiness_seconds'], 1800)
        # Review finding 9: 'command' is what served (the contract's rewrite), the platform's is kept beside it.
        self.assertEqual(report['command_requested'], gate.platform_argv(8000))
        self.assertEqual(report['command'], gate.platform_argv(8000)[:3] + report['platform']['served_argv'])
        self.assertEqual(report['command'][-2:], ['--max-model-len', '131328'])
        self.assertIsNotNone(report['ledger'])
        self.assertEqual(report['platform']['served_profile'], 'exact')
        self.assertEqual(report['platform']['problems'], [])
        self.assertEqual(report['dram']['engines'], 2)
        self.assertEqual(sorted(len(c['prompt']) for c in calls), [200, 2049, 3000])
        self.assertEqual(report['real_text']['prompt_lengths'], [200, 3000, 2049])
        self.assertEqual(report['prompt_lengths_requested'], [200, 3000, 2049])
        self.assertIsNone(report['real_text_target'])
        self.assertTrue(all(c['kwargs']['model'] == 'Qwen/Qwen3.8-27B' for c in calls))
        self.assertTrue(all(c['max_tokens'] == 4096 for c in calls))
        self.assertEqual(report['real_text_stream_problems'], [])
        self.assertEqual((report['server_argv'], report['snapshot']), ('platform', '/models/snap'))
        self.assertEqual(code, 1, 'no packed-round evidence in a faked run')

    def test_a_platform_run_without_the_contracts_line_fails_and_still_reports(self):
        argv = list(self.PLATFORM + self.REAL) + ['--context', '131328', '--prompt-tokens', '3000', '--users', '1',
                                                  '--sequential-users', '2']
        code, calls, report, _ = self.run_platform(argv, 'no contract here\n')
        self.assertEqual(code, 1)
        self.assertIs(report['gate_passed'], False)
        self.assertIn('did not rewrite', report['platform']['problems'][0])
        self.assertEqual([c['index'] for c in calls], [0, 1])

    def test_a_server_that_never_came_up_still_reports_its_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            (results / 'server.log').write_text(C2_ARGV_LINE + '\nTraceback: boom\n', encoding='utf-8')
            argv = ['gate', '--server-argv', 'platform', '--results', str(results), '--references', str(results)]
            with mock.patch.object(gate, 'start_server', side_effect=RuntimeError('server exited before readiness: 1')), \
                    mock.patch.object(gate, 'stop_server'), mock.patch.object(sys, 'argv', argv), \
                    redirect_stdout(io.StringIO()) as out:
                code = gate.main()
            text = out.getvalue()
        report = json.loads(text[text.index(gate.BEGIN) + len(gate.BEGIN):text.index(gate.END)])
        self.assertEqual(code, 1)
        self.assertIn('RuntimeError', report['fatal'])
        self.assertEqual(report['platform']['served_profile'], 'exact')


class LifecycleOptionTests(unittest.TestCase):
    """--drops, --user-max-tokens, --user-ignore-eos and --alive-check (the C2 serving gate's lifecycle arms)."""

    REAL = ('--prompt-source', 'real-text', '--allow-missing-references', '--server-argv', 'platform',
            '--context', '131328', '--users', '6', '--prompt-lengths', '60000,4096,8192,2049,16384,255')

    def test_events_parse_per_user(self):
        options = parse(*self.REAL + ('--drops', '0:build,1:live=4,2:40,3:@2.5,5:prefill+5', '--user-max-tokens', '4:1',
                                      '--user-ignore-eos', '5'))
        self.assertEqual(options.events, dict(drops={0: ('build', 0), 1: ('live', 4), 2: ('chunks', 40),
                                                     3: ('seconds', 2.5), 5: ('prefill', 5.0)},
                                              max_tokens={4: 1}, ignore_eos=[5]))
        base = gate.stream_kwargs(options)
        watch = object()
        self.assertEqual(gate.user_stream(options, 0, base), (256, base), 'no watch, no drop keywords')
        self.assertEqual(gate.user_stream(options, 0, base, watch), (256, dict(base, watch=watch)))
        self.assertEqual(gate.user_stream(options, 4, base, watch), (1, dict(base, watch=watch)))
        self.assertEqual(gate.user_stream(options, 5, base), (256, dict(base, ignore_eos=True)))
        self.assertEqual([gate.describe_drop(options.events['drops'][user]) for user in (0, 1, 2, 3, 5)],
                         ['build', 'live=4', 'chunks 40', 'seconds 2.5', 'prefill+5.0'])

    def test_a_multiple_drop_is_a_barrier_on_every_member(self):
        options = parse(*self.REAL + ('--drops', '0:prefill+5,1+2+3:60'))
        barrier = ('barrier', (60, (1, 2, 3)))
        self.assertEqual(options.events['drops'], {0: ('prefill', 5.0), 1: barrier, 2: barrier, 3: barrier})
        self.assertEqual(gate.describe_drop(barrier), 'barrier 1+2+3 at 60 chunks')

    def test_no_events_is_every_existing_call(self):
        options = parse()
        self.assertEqual(options.events, dict(drops={}, max_tokens={}, ignore_eos=[]))
        self.assertEqual(gate.user_stream(options, 3, {}), (256, {}))
        self.assertEqual((options.alive_check, options.alive_seconds), (0, gate.ALIVE_SECONDS))

    def test_bad_events_are_refused(self):
        for argv in (('--drops', '6:1'), ('--drops', '0:0'), ('--drops', '0:x'), ('--drops', '0:1,0:2'),
                     ('--drops', '0'), ('--user-max-tokens', '1:0'), ('--user-ignore-eos', '9'),
                     ('--user-ignore-eos', 'a'), ('--drops', '1+1:5'), ('--drops', '1+2:build'),
                     ('--drops', '1+2:5,2:3'), ('--drops', '0:live=0'), ('--drops', '0:prefill+x'),
                     ('--alive-check', '-1')):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                parse(*self.REAL + argv)
        with self.assertRaises(SystemExit):
            parse('--drops', '0:1')   # the default mode has no detail streams to record a drop

    def test_a_log_timed_drop_needs_a_prompt_length_no_other_user_has(self):
        """The server log names a request's user before its first byte only by its ledger prompt length."""
        shared = ('--prompt-source', 'real-text', '--allow-missing-references', '--server-argv', 'platform',
                  '--context', '131328', '--users', '3', '--prompt-lengths', '4096,4096,255')
        with self.assertRaises(SystemExit):
            parse(*shared + ('--drops', '0:build'))
        self.assertEqual(parse(*shared + ('--drops', '2:build')).events['drops'], {2: ('build', 0)})
        with self.assertRaises(SystemExit):
            parse('--prompt-source', 'real-text', '--allow-missing-references', '--context', '131328',
                  '--prompt-tokens', '131072', '--drops', '0:prefill+5')

    def test_a_dropped_stream_is_not_a_stream_problem(self):
        self.assertEqual(gate.real_text_stream_problems([dict(dropped='after 1 chunks', text='x'),
                                                         dict(dropped='no byte within 2 s', text=''),
                                                         dict(dropped=None, text='y', finish_reason='stop')]), [])
        self.assertEqual(len(gate.real_text_stream_problems([dict(dropped='after 1 chunks', error='boom')])), 1)

    def test_a_lifecycle_run_watches_its_drops_and_asks_every_seat_afterwards(self):
        calls = []

        def stream(port, prompt, max_tokens, results, index, timeout, **kwargs):
            calls.append(dict(index=index, max_tokens=max_tokens, kwargs=kwargs, length=len(prompt)))
            watch = kwargs.get('watch')
            entry = dict(text='answer %d' % index, finish_reason='stop', tokens=3, completion_tokens=5,
                         gaps_ms=[250.0], ttft_s=1.0, prompt_tokens=len(prompt), request_id='cmpl-u%d' % index)
            if watch is not None:
                watch.begin(index, time.perf_counter())
                for chunk in (1, 2, 3):
                    reason = watch.chunk(index, chunk, time.perf_counter())
                    if reason:
                        entry = dict(text='ans', dropped=reason, tokens=chunk, gaps_ms=[], ttft_s=1.0,
                                     request_id='cmpl-u%d' % index)
                        break
                watch.end(index, time.perf_counter())
            results[index] = entry

        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory, 'results')
            results.mkdir()
            (results / 'server.log').write_text(C2_ARGV_LINE + '\n' + LEDGER_LINES, encoding='utf-8')
            package = write_package(Path(directory, 'site'), code_corpus(files=200))
            argv = ['gate'] + list(self.REAL[:-1]) + ['200,3000,2049,300,500,700', '--drops', '0:1,1:2',
                                                        '--user-max-tokens', '2:1', '--user-ignore-eos', '3',
                                                        '--alive-check', '2', '--results', str(results)]
            with mock.patch.object(gate, 'start_server', return_value=(None, None, results / 'server.log', ['x'])), \
                    mock.patch.object(gate, 'stop_server'), \
                    mock.patch.object(gate, 'stream_once', side_effect=stream), \
                    mock.patch.object(real_text_prompts, 'load_tokenizer', return_value=FakeTokenizer()), \
                    mock.patch.object(real_text_prompts, 'package_root', return_value=package), \
                    mock.patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()) as out:
                gate.main()
            text = out.getvalue()
        report = json.loads(text[text.index(gate.BEGIN) + len(gate.BEGIN):text.index(gate.END)])
        self.assertEqual(len(calls), 8, 'six streams and one alive request per seat')
        alive = calls[6:]
        self.assertEqual([(c['max_tokens'], c['length']) for c in alive], [(8, 200), (8, 200)],
                         'the shortest prompt, 8 tokens, twice at once')
        self.assertTrue(all('watch' not in c['kwargs'] for c in alive))
        self.assertTrue(all('watch' in c['kwargs'] for c in calls[:6]), 'every stream reports to the watch')
        self.assertIs(report['alive'], True)
        self.assertEqual(len(report['alive_after']), 2)
        self.assertEqual(report['user_events'], dict(drops={'0': 'chunks 1', '1': 'chunks 2'},
                                                     max_tokens={'2': 1}, ignore_eos=[3]))
        events = report['lifecycle']['events']
        self.assertEqual({user: (e['fired'], e['phase'], e['reason']) for user, e in events.items()},
                         {'0': (True, 'decode', 'after 1 chunks'), '1': (True, 'decode', 'after 2 chunks')})
        comparisons = report['comparisons']
        self.assertEqual([c['max_tokens'] for c in comparisons], [256, 256, 1, 256, 256, 256])
        self.assertEqual([c['dropped'] for c in comparisons][:3], ['after 1 chunks', 'after 2 chunks', None])
        self.assertEqual([c['ignore_eos'] for c in comparisons], [False, False, False, True, False, False])
        self.assertEqual(report['real_text_stream_problems'], [])
        self.assertEqual(report['ledger']['residual_status'], 'passed')
        self.assertIn('[ALIVE] after the streams: True (2 at once)', text)

    def test_every_seat_must_answer_within_the_limit(self):
        """Review finding 7: one alive request found a leaked seat only after four had leaked."""
        release = threading.Event()

        def stream(port, prompt, max_tokens, results, index, timeout, **kwargs):
            if index == 3:
                release.wait(5)   # the fourth request queued behind a seat never given back
                return
            results[index] = dict(text='OK')

        try:
            with mock.patch.object(gate, 'stream_once', side_effect=stream):
                entries, ok = gate.alive_check(8000, [1, 2], 4, 60, 0.3, {})
        finally:
            release.set()
        self.assertFalse(ok)
        self.assertEqual([bool(entry.get('error')) for entry in entries], [False, False, False, True])
        self.assertIn('no answer within 0 s', entries[3]['error'])
        with mock.patch.object(gate, 'stream_once', side_effect=lambda *a, **k: a[3].__setitem__(a[4], dict(text='OK'))):
            self.assertTrue(gate.alive_check(8000, [1], 4, 60, 5, {})[1])


class FakeSocket(object):
    def __init__(self):
        self.shut = []

    def shutdown(self, how):
        self.shut.append(how)


class Clock(object):
    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now


def held(request_id, prompt):
    return ("(EngineCore pid=9) [PINDIAG] prefill gate: held='%s' was=None\n"
            "(EngineCore pid=9) [MEMLEDGER] phase=prefill point=before prompt=%d chip0 allocated=30.000GB free=3.900GB\n"
            % (request_id, prompt))


class StreamWatchTests(unittest.TestCase):
    """The lifecycle arms' clock: drops fired on the server log's admission markers and on the streams'
    chunks, each recorded against the phase it hit (review finding 6)."""

    LENGTHS = [60000, 4096, 8192, 2049, 16384, 255]

    def watch(self, drops, directory, clock):
        return gate.StreamWatch(drops, self.LENGTHS, Path(directory, 'server.log'), clock=clock, poll=0.01)

    def log(self, directory, text):
        with open(os.path.join(directory, 'server.log'), 'a', encoding='utf-8') as handle:
            handle.write(text)

    def test_a_cancel_inside_the_prefill_and_a_drop_during_the_build_are_timed_on_the_log(self):
        clock = Clock()
        with tempfile.TemporaryDirectory() as directory:
            self.log(directory, 'startup\n')
            watch = self.watch({0: ('prefill', 5.0), 5: ('build', 0)}, directory, clock)
            sockets = {0: FakeSocket(), 5: FakeSocket()}
            for user in (0, 5):
                watch.begin(user, clock.now)
                watch.opened(user, sockets[user])
            self.log(directory, held('cmpl-aaa-0-11111111', 60000))
            clock.now = 101.0
            watch.read_log()
            watch.check(clock.now)
            self.assertEqual(sockets[0].shut, [], '1 s into the prefill: not yet')
            clock.now = 106.5
            watch.check(clock.now)
            self.assertEqual(len(sockets[0].shut), 1, 'shut down 5 s into its prefill')
            self.assertEqual(watch.cancelled(0), '5.0 s into its prefill')
            # User 5 (255 tokens) is admitted next; its build begins when the ledger's 'after' line appears.
            self.log(directory, "[PINDIAG] prefill gate: held=None was='cmpl-aaa-0-11111111'\n" +
                     held('cmpl-bbb-0-22222222', 255))
            clock.now = 110.0
            watch.read_log()
            watch.check(clock.now)
            self.assertEqual(sockets[5].shut, [], 'still in its prefill')
            self.log(directory, '[MEMLEDGER] phase=prefill point=after req=bb-0-22222222 chip0 allocated=31.000GB\n')
            clock.now = 110.4
            watch.read_log()
            watch.check(clock.now)
            self.assertEqual(len(sockets[5].shut), 1)
            self.log(directory, '[PINDIAG] dram after engine cmpl-bbb-0-22222222: chip0 allocated=31.5GB\n')
            clock.now = 114.0
            watch.read_log()
            for user in (0, 5):
                watch.end(user, clock.now)
            report = watch.report([dict(request_id='cmpl-aaa'), None, None, None, None, None])
        events = report['events']
        self.assertEqual((events['0']['phase'], events['0']['fired_s'], events['0']['live']), ('prefill', 6.5, 0))
        self.assertEqual((events['5']['phase'], events['5']['reason']), ('build', 'its engine build began'))
        self.assertEqual(report['users']['5']['request_ids'], ['cmpl-bbb-0-22222222'])
        self.assertEqual((report['users']['5']['prefill_s'], report['users']['5']['build_s'],
                          report['users']['5']['engine_s']), (10.0, 10.4, 14.0))
        self.assertEqual(report['users']['0']['admitted_s'], 10.0)
        self.assertTrue(report['ledger_markers'])

    def test_a_cancel_due_before_the_socket_opened_is_made_as_it_opens(self):
        clock = Clock()
        with tempfile.TemporaryDirectory() as directory:
            watch = self.watch({3: ('seconds', 2.0)}, directory, clock)
            watch.begin(3, clock.now)
            clock.now = 102.5
            watch.check(clock.now)
            sock = FakeSocket()
            watch.opened(3, sock)
            self.assertEqual((len(sock.shut), watch.cancelled(3)), (1, 'no byte within 2.0 s'))
            self.assertEqual(watch.report()['events']['3']['phase'], 'queued')

    def test_a_cancel_that_cannot_reach_a_socket_is_never_reported_as_a_drop(self):
        """With no socket under the response the stream runs on: it must not read as dropped."""
        clock = Clock()
        with tempfile.TemporaryDirectory() as directory:
            watch = self.watch({3: ('seconds', 2.0)}, directory, clock)
            watch.begin(3, clock.now)
            watch.opened(3, None)
            clock.now = 102.5
            watch.check(clock.now)
            self.assertIsNone(watch.cancelled(3))
            event = watch.report()['events']['3']
        self.assertEqual((event['fired'], event['delivered']), (True, False))
        self.assertIn('no socket under the response', event['undelivered'])

    def test_a_first_byte_before_its_drop_was_due_is_a_miss(self):
        clock = Clock()
        with tempfile.TemporaryDirectory() as directory:
            watch = self.watch({3: ('seconds', 2.0)}, directory, clock)
            watch.begin(3, clock.now)
            clock.now = 101.0
            self.assertIsNone(watch.chunk(3, 1, clock.now))
            clock.now = 105.0
            watch.check(clock.now)
            self.assertIsNone(watch.cancelled(3))
            event = watch.report()['events']['3']
        self.assertEqual((event['fired'], event['missed']), (False, 'the first byte came before the drop was due'))

    def test_live_drops_fire_at_their_live_count(self):
        clock = Clock()
        with tempfile.TemporaryDirectory() as directory:
            watch = self.watch({1: ('live', 3), 2: ('chunks', 2)}, directory, clock)
            for user in (1, 2, 3):
                watch.begin(user, clock.now)
            self.assertIsNone(watch.chunk(1, 1, 101.0), 'one live stream')
            self.assertIsNone(watch.chunk(2, 1, 101.1))
            self.assertIsNone(watch.chunk(3, 1, 101.2))
            self.assertEqual(watch.chunk(1, 2, 101.3), 'at 3 live streams')
            watch.end(1, 101.3)
            self.assertEqual(watch.chunk(2, 2, 101.4), 'after 2 chunks')
            events = watch.report()['events']
        self.assertEqual((events['1']['live'], events['1']['phase']), (3, 'decode'))
        self.assertEqual((events['2']['live'], events['2']['phase']), (2, 'decode'))

    def test_a_multiple_drop_closes_every_member_at_once(self):
        with tempfile.TemporaryDirectory() as directory:
            watch = gate.StreamWatch({u: ('barrier', (3, (1, 2, 3))) for u in (1, 2, 3)}, self.LENGTHS,
                                     Path(directory, 'server.log'), poll=0.01)
            reasons = {}

            def user(index, delay):
                watch.begin(index, time.perf_counter())
                time.sleep(delay)
                for chunk in (1, 2, 3, 4):
                    reason = watch.chunk(index, chunk, time.perf_counter())
                    if reason:
                        reasons[index] = (reason, time.perf_counter())
                        break
                watch.end(index, time.perf_counter())

            threads = [threading.Thread(target=user, args=(index, delay)) for index, delay in ((1, 0.0), (2, 0.1), (3, 0.2))]
            [thread.start() for thread in threads]
            [thread.join(10) for thread in threads]
            report = watch.report()
        self.assertEqual(sorted(reasons), [1, 2, 3])
        self.assertEqual({reason for reason, _ in reasons.values()}, {'barrier 1+2+3 at 3 chunks'})
        times = [when for _, when in reasons.values()]
        self.assertLess(max(times) - min(times), 0.1, 'the early arrivals waited for the last')
        self.assertTrue(report['barriers'][0]['complete'])

    def test_a_multiple_drop_whose_member_ends_first_is_released_incomplete(self):
        clock = Clock()
        with tempfile.TemporaryDirectory() as directory:
            watch = self.watch({u: ('barrier', (60, (1, 2))) for u in (1, 2)}, directory, clock)
            watch.begin(2, clock.now)
            watch.end(2, clock.now)       # user 2 reached EOS before 60 chunks
            self.assertEqual(watch.chunk(1, 60, clock.now), 'barrier 1+2 at 60 chunks')
            report = watch.report()
        self.assertFalse(report['barriers'][0]['complete'])
        self.assertFalse(report['events']['2']['fired'])


LEDGER_LINES = (
    '[MEMLEDGER] phase=P7 point=after_attach check=residual status=passed limit=1.500GB chip0=0.300GB chip1=0.310GB\n'
    '[MEMLEDGER] phase=prefill point=before prompt=200 chip0 allocated=30.000GB free=3.900GB\n'
    '[MEMLEDGER] phase=prefill point=before prompt=200 chip1 allocated=30.100GB free=3.800GB\n'
    '[MEMLEDGER] phase=prefill point=before prompt=3000 chip0 allocated=31.000GB free=2.900GB\n'
    '[MEMLEDGER] phase=prefill point=before prompt=3000 chip1 allocated=31.100GB free=2.800GB\n'
    '[MEMLEDGER] phase=prefill point=before prompt=200 chip0 allocated=30.050GB free=3.850GB\n'
    '[MEMLEDGER] phase=prefill point=before prompt=200 chip1 allocated=30.100GB free=3.800GB\n'
    '[MEMLEDGER] phase=prefill point=before prompt=200 chip0 allocated=31.000GB free=2.900GB\n'
    '[MEMLEDGER] phase=prefill point=before prompt=200 chip1 allocated=31.100GB free=2.800GB\n')


class LedgerTests(unittest.TestCase):
    def test_idle_readings_group_per_prefill_and_drift_is_after_less_first(self):
        readings = gate.ledger_readings(LEDGER_LINES)
        self.assertEqual([r['prompt'] for r in readings], [200, 3000, 200, 200], 'same prompt twice: two readings')
        self.assertEqual(readings[1]['chips'], {'0': 31.0, '1': 31.1})
        report = gate.ledger_report(LEDGER_LINES, alive_index=2)
        self.assertEqual(report['residual_status'], 'passed', 'read on its own, not via flag_marker_report')
        self.assertEqual(report['idle_drift_gb'], {'0': 0.05, '1': 0.0})
        self.assertEqual(report['readings'], 4)
        self.assertIsNone(gate.ledger_report(LEDGER_LINES, alive_index=9)['idle_drift_gb'])
        self.assertEqual(gate.ledger_report(''), dict(residual_status=None, readings=0, first_idle=None,
                                                      idle_after_streams=None, idle_drift_gb=None))


class MarkerFloorTests(unittest.TestCase):
    def test_real_text_floors_are_the_synthetic_arms(self):
        options = parse('--prompt-source', 'real-text', '--allow-missing-references')
        report = dict(real_text=dict(prompt_lengths=[32768] * 4))
        self.assertEqual(gate.marker_prompt_tokens(options, report), 32768)
        self.assertEqual(gate.prefill_conv_required_chunks(4, 32768), 64)
        self.assertEqual(gate.marker_prompt_tokens(parse(), report), 32768, 'synthetic: --prompt-tokens')
        # Were a prompt set ever to differ per user, the shortest sets the floor, which only lowers it.
        self.assertEqual(gate.marker_prompt_tokens(options, dict(real_text=dict(prompt_lengths=[32768, 30000]))),
                         30000)
        pf = gate.sdpa_pf_markers({'QWEN_FAST_SDPA_PF': '1'}, 131072)
        self.assertIn('chains=16 members=96 ', pf[1])


class WorkflowTagTests(unittest.TestCase):
    """v157-v160 and any later real-text tag: the image case, the flags the gate insists on, and
    the image's 256-token output budget (serving_fast_policy.OUTPUT_BUDGET)."""

    @classmethod
    def setUpClass(cls):
        cls.text = WORKFLOW.read_text(encoding='utf-8').replace(chr(13) + chr(10), chr(10))
        cls.exports = {}
        for match in re.finditer(r'^\s*((?:\*-v[0-9]+\|?)+)\) export ([^\n]*?) ;;$', cls.text, re.M):
            for tag in re.findall(r'\*-(v[0-9]+)', match.group(1)):
                cls.exports.setdefault(tag, match.group(2).split())
        cls.images = {}
        graft = cls.text[cls.text.index('  graft:'):cls.text.index('  arm:')]
        for match in re.finditer(r'^\s*((?:\*-v[0-9]+\|?)+)\) image=(sha256:[0-9a-f]{64}) ;;$', graft, re.M):
            for tag in re.findall(r'\*-(v[0-9]+)', match.group(1)):
                cls.images.setdefault(tag, match.group(2))

    def flags(self, tag):
        return dict(item.split('=', 1) for item in self.exports[tag])

    def test_the_four_tags_exist_on_image_a5(self):
        for tag in ('v157', 'v158', 'v159', 'v160'):
            with self.subTest(tag=tag):
                flags = self.flags(tag)
                self.assertEqual(self.images[tag], IMAGE_A5)
                self.assertEqual(flags['M3NATIVE_IMAGE'], IMAGE_A5)
                self.assertEqual(flags['M3NATIVE_PROMPT_SOURCE'], 'real-text')

    def test_every_real_text_tag_carries_what_the_gate_requires(self):
        tags = [tag for tag in self.exports if self.flags(tag).get('M3NATIVE_PROMPT_SOURCE') == 'real-text']
        self.assertGreaterEqual(len(tags), 4)
        for tag in tags:
            with self.subTest(tag=tag):
                flags = self.flags(tag)
                self.assertEqual(flags.get('M3NATIVE_ALLOW_MISSING_REFERENCES'), '1')
                self.assertIn(flags.get('M3NATIVE_EOS', 'stop'), ('stop',))
                self.assertLessEqual(int(flags.get('M3NATIVE_MAX_TOKENS', '256')), 256)
                self.assertIn(tag, self.images, 'the graft job needs the tag in its image case')
                self.assertEqual(self.images[tag], flags.get('M3NATIVE_IMAGE'))

    def test_the_pairs_differ_only_where_they_should(self):
        base = dict(self.flags('v155'))
        v157, v159 = self.flags('v157'), self.flags('v159')
        real = dict(M3NATIVE_PROMPT_SOURCE='real-text', M3NATIVE_EOS='stop', M3NATIVE_ALLOW_MISSING_REFERENCES='1',
                    M3NATIVE_MAX_TOKENS='256')
        self.assertEqual(v159, dict(base, **real))
        self.assertEqual(v157, dict({k: v for k, v in base.items()
                                     if k not in ('M3NATIVE_CONTEXT', 'M3NATIVE_PROMPT_TOKENS')}, **real))
        self.assertEqual(self.flags('v158'), dict(self.flags('v149'), **real))
        self.assertEqual(self.flags('v160'), dict(self.flags('v150'), **real))


if __name__ == '__main__':
    unittest.main()
