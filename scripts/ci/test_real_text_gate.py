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
            environ={'QWEN_FAST_GDN_PREFILL_CONV': '1'})
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
        self.assertTrue(all(name.startswith('QWEN_') for name in report['qwen_configuration']))
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
        self.assertEqual(report['command'], gate.platform_argv(8000))
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
