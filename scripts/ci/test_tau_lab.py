"""The W-T1 tau lab (c2_tau_lab.py, tau_lab_report.py, the taulab job action, the job template), held on CPU with fakes.

Nothing here opens a device, docker or a socket: the driver runs against a FakeEngine that answers each request with rounds
of its own, writes the matching [PACKED] and fast_serving_phases lines into the server log the report reads, and the data is
synthetic with sentinel strings that must never come out. The Qwen repo is public: what the lab prints and what its public
summary holds are checked for every sentinel."""

import copy
import io
import json
import os
import random
import re
import shutil
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_prefix_gate as pg  # noqa: E402
import c2_serving_gate as gate  # noqa: E402
import c2_serving_job as job  # noqa: E402
import c2_tau_lab as lab  # noqa: E402
import tau_lab_report as rep  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WORKFLOW = os.path.join(ROOT, '.github', 'workflows', 'qwen-c2-serving.yml')
CPU_WORKFLOW = os.path.join(ROOT, '.github', 'workflows', 'qwen-integration-cpu.yml')
FOLDER = os.path.join(HERE, 'references', 'tp4-taulab-jobs')
with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as _handle:
    PROFILES = json.load(_handle)
NAMES = sorted(PROFILES['profiles'])
TEXT = 'SENTINELTEXT'
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|'
                    r'/dev/tenstorrent|home/|zot\.|@[A-Z0-9_]+@')


def read_text(path):
    with open(path, encoding='utf-8') as handle:
        return handle.read()


def read_json(path):
    return json.loads(read_text(path))


def write_jsonl(path, records):
    with open(path, 'w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record) + '\n')


def messages(tag, tokens=None):
    return [dict(role='system', content='%s system %s' % (TEXT, tag)), dict(role='user', content='%s user %s' % (TEXT, tag))]


def make_data(directory, swe=6, own=2, sessions=2, followups=2, a5=False, big=True):
    """A synthetic data directory: A1 (swe), A2 (own), A3 (prompt ids with reference tau), A4 (sessions with follow-ups)."""
    a1 = [dict(id='zqswe%02d' % n, cluster='zqconv%d' % (n // 2), set='swe', turn=n % 2, tokens=90000 if big and n == 0 else 9000 + n,
               messages=messages('swe%d' % n)) for n in range(swe)]
    write_jsonl(os.path.join(directory, 'A1.jsonl'), a1)
    write_jsonl(os.path.join(directory, 'A2.jsonl'), [dict(id='zqown%02d' % n, cluster='zqown%d' % n, set='own', tokens=30000,
                                                           messages=messages('own%d' % n)) for n in range(own)])
    write_jsonl(os.path.join(directory, 'A3.jsonl'), [dict(id='zqcal%d' % n, cluster='zqcal%d' % n, set='calib',
                                                           tokens=4096 if n < 4 else 32768, prompt_ids=[1, 2, 3, n], ref_tau=6.0)
                                                      for n in range(8)])
    write_jsonl(os.path.join(directory, 'A4.jsonl'), [dict(id='zqses%d' % n, cluster='zqses%d' % n, set='chained',
                                                           messages=messages('ses%d' % n),
                                                           followups=['%s follow %d' % (TEXT, k) for k in range(followups)])
                                                      for n in range(sessions)])
    if a5:
        write_jsonl(os.path.join(directory, 'A5.jsonl'), [dict(id='zqswe%02d' % n, cluster='zqconv%d' % (n // 2), set='swe',
                                                               messages=messages('swe%d' % n)) for n in range(2)])


class FakeEngine(object):
    """What the container would do: each request gets rounds, the server log gets their lines, the stream their counts."""

    def __init__(self, path=None, rounds=12, packed_only=True):
        self.path = path
        self.rounds = rounds
        self.lock = threading.Lock()
        self.count = 0
        self.requests = []
        self.lines = []

    def emit(self, line):
        with self.lock:
            self.lines.append(line)
            if self.path:
                with open(self.path, 'a', encoding='utf-8') as handle:
                    handle.write(line + '\n')

    def stream(self, path, body, timeout, keep):
        with self.lock:
            self.count += 1
            number = self.count
            self.requests.append((path, copy.deepcopy(body)))
        rng = random.Random(number)
        emitted = [rng.randint(2, 16) for _ in range(self.rounds)]
        request_id = 'chatcmpl-%032x' % number
        position = 1000
        for index, value in enumerate(emitted):
            self.emit('2026-10-01T00:00:00.000000000Z [PACKED] request=%s-0-ab12cd34 segment=%d position=%d prefix=%d emitted=%d '
                      'predictions=[1, 2, 3]' % (request_id, number % 4, position, value - 1, value))
            position += value
        self.emit(json.dumps(dict(stage='fast_serving_phases', blocks=[
            dict(position=1000, rows=16, committed=value, verifier=dict(packed=True, users=4)) for value in emitted])))
        total = 1 + sum(emitted)
        reasoning = 20 if path.endswith('chat/completions') and 'chat_template_kwargs' not in body else 0
        kinds = ''.join(['r' if offset < 1 + (reasoning and 3) else 'c' for offset in range(len(emitted) + 1)]) if reasoning \
            else 'c' * (len(emitted) + 1)
        usage = dict(prompt_tokens=body.get('_tokens', 0), completion_tokens=total)
        return dict(request_id=request_id, chunk_tokens=[1] + emitted, kinds=kinds,
                    think_tokens=(1 + sum(emitted[:2])) if reasoning else 0, tool_at=None, finish='length', error=None, usage=usage,
                    output_ids=list(range(total)), content='%s ANSWER %d' % (TEXT, number), ttft=0.5, status=200)


class FakeClient(object):
    def ready(self):
        return True

    def context_tokens(self):
        return 131328


class FakeContainer(object):
    def running(self):
        return True


class FakeLog(object):
    def __init__(self, engine, path):
        self.engine, self.path = engine, path

    def start(self, since=None):
        self.engine.path = self.path
        self.engine.emit('[QWEN-C2] profile c2-packed-tp4+taulab booted')

    def lines(self):
        return list(self.engine.lines)


def run_lab(testcase, arguments=None, engine=None, data=None, results=None, **options):
    """main() against the fakes -> (exit code, output lines, results dir, engine)."""
    root = tempfile.mkdtemp()
    testcase.addCleanup(shutil.rmtree, root, True)
    data = data or os.path.join(root, 'data')
    results = results or os.path.join(root, 'results')
    public = os.path.join(root, 'public')
    if not os.path.isdir(data):
        os.makedirs(data)
        make_data(data, **options)
    engine = engine or FakeEngine()
    calls, out = [], []
    argv = ['--image', 'registry.invalid/tt-vllm:qwen38-c2-tp4-serve-2', '--data', data, '--results', results, '--public', public,
            '--deadline-seconds', '100000', '--in-flight', '4', '--profiles', os.path.join(HERE, 'qwen_c2_profiles.json')]
    argv += list(arguments or [])
    with mock.patch.object(gate, 'platform_containers', return_value=[]):
        code = lab.main(argv, say=out.append, make_stream=lambda: engine.stream, make_client=FakeClient,
                        make_container=lambda name: FakeContainer(), make_log=lambda name, path: FakeLog(engine, path),
                        docker=lambda arguments, timeout=0: (calls.append(arguments) or (0, '')), devices=['d0', 'd1', 'd2', 'd3'],
                        sleep=lambda seconds: None)
    testcase.docker_calls = calls
    return code, out, results, public, engine, data


class ProfileTests(unittest.TestCase):
    def test_the_derived_profile_adds_exactly_the_two_log_flags_and_no_arithmetic(self):
        name, document = lab.derive_profile(PROFILES, 'c2-packed-tp4')
        self.assertEqual(name, 'c2-packed-tp4+taulab')
        derived = document['profiles'][name]
        base = PROFILES['profiles']['c2-packed-tp4']
        added = dict((key, value) for key, value in derived['env'].items() if base['env'].get(key) != value)
        self.assertEqual(added, dict(QWEN_FAST_PACKED_AUDIT='1', QWEN_FAST_PHASE_TIMING='1'))
        self.assertEqual(derived['env']['QWEN_FAST_VERIFY_T1_AUDIT'], '1', 'the production audits stay on')
        self.assertEqual(derived['engine'], base['engine'])
        self.assertEqual(lab.arithmetic_diff(PROFILES, 'c2-packed-tp4', document), [])
        self.assertNotIn('QWEN_FAST_PACKED_AUDIT', base['env'], 'the base is left alone')

    def test_the_contract_boots_the_derived_file_and_exports_the_flags(self):
        import serving_c2_contract as contract
        name, document = lab.derive_profile(PROFILES, 'c2-packed-tp4')
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, True)
        path = os.path.join(directory, 'profiles.json')
        with open(path, 'w') as handle:
            json.dump(document, handle)
        profile = contract.load_profile(path, name)
        self.assertEqual(profile['name'], name)
        environ = contract.apply_environment(profile, {})
        self.assertEqual((environ['QWEN_FAST_PACKED_AUDIT'], environ['QWEN_FAST_PHASE_TIMING']), ('1', '1'))
        self.assertEqual(environ['MESH_DEVICE'], 'P150x4')

    def test_what_is_refused(self):
        with self.assertRaises(lab.LabError):
            lab.derive_profile(PROFILES, 'general')                    # not the four-card mesh
        with self.assertRaises(lab.LabError):
            lab.derive_profile(PROFILES, 'no-such-profile')
        gated = copy.deepcopy(PROFILES)
        gated['profiles']['c2-packed-tp4']['gate_only'] = True
        with self.assertRaises(lab.LabError):
            lab.derive_profile(gated, 'c2-packed-tp4')
        clash = copy.deepcopy(PROFILES)
        clash['profiles']['c2-packed-tp4']['env']['QWEN_FAST_PACKED_AUDIT'] = '0'
        with self.assertRaises(lab.LabError):
            lab.derive_profile(clash, 'c2-packed-tp4')

    def test_an_arithmetic_change_is_caught(self):
        name, document = lab.derive_profile(PROFILES, 'c2-packed-tp4')
        document['profiles'][name]['env']['QWEN_FAST_QUAD_DRAFT'] = '1'
        self.assertEqual(lab.arithmetic_diff(PROFILES, 'c2-packed-tp4', document), ['QWEN_FAST_QUAD_DRAFT'])

    def test_the_production_label_follows_the_image_tag(self):
        self.assertTrue(lab.is_production_image('zot.invalid:5000/tt-vllm:qwen38-c2-tp4-serve-2'))
        self.assertFalse(lab.is_production_image('zot.invalid:5000/tt-vllm:qwen38-c2-tp4-next-1'))
        self.assertEqual(lab.image_tag('host:5000/repo:tag'), 'tag')
        self.assertEqual(lab.image_tag('host:5000/repo'), '')


class RequestTests(unittest.TestCase):
    def test_thinking_on_sends_no_template_kwargs_and_off_sends_enable_thinking_false(self):
        record = dict(id='x', messages=messages('a'), tools=[dict(type='function')])
        path, on = lab.request_body(record, True, 700, 'm')
        self.assertEqual(path, '/v1/chat/completions')
        self.assertNotIn('chat_template_kwargs', on)
        self.assertEqual((on['temperature'], on['top_p'], on['stream'], on['max_tokens']), (0.0, 1.0, True, 700))
        self.assertTrue(on['stream_options']['continuous_usage_stats'] and on['return_token_ids'])
        self.assertEqual(on['tool_choice'], 'auto')
        _, off = lab.request_body(record, False, 700, 'm')
        self.assertEqual(off['chat_template_kwargs'], dict(enable_thinking=False))
        _, baked = lab.request_body(record, None, 700, 'm')
        self.assertNotIn('chat_template_kwargs', baked)

    def test_a_token_id_prompt_goes_to_completions_and_nothink_ids_serve_the_off_arm(self):
        record = dict(id='x', prompt_ids=[1, 2], prompt_ids_nothink=[3, 4])
        path, body = lab.request_body(record, True, 10, 'm')
        self.assertEqual((path, body['prompt']), ('/v1/completions', [1, 2]))
        self.assertEqual(lab.request_body(record, False, 10, 'm')[1]['prompt'], [3, 4])
        self.assertEqual(lab.request_body(dict(id='x', prompt_ids=[1, 2]), None, 10, 'm')[1]['prompt'], [1, 2])

    def chunks(self, lines, **kwargs):
        state = lab.Chunks(**kwargs)
        for line in lines:
            if state.feed(line):
                break
        return state.result()

    def test_the_continuous_usage_stats_give_the_tokens_per_chunk(self):
        def chunk(usage, ids, **delta):
            return 'data: ' + json.dumps(dict(id='cmpl-1', choices=[dict(delta=delta, token_ids=ids, finish_reason=None)],
                                              usage=dict(completion_tokens=usage)))
        result = self.chunks([chunk(1, [5], reasoning='a'), chunk(8, [6] * 7, reasoning='b'), chunk(12, [7] * 4, content='c'),
                              chunk(15, [8] * 3, content='d'),
                              'data: ' + json.dumps(dict(id='cmpl-1', choices=[dict(delta={}, token_ids=[], finish_reason='stop')],
                                                         usage=dict(completion_tokens=15))),
                              'data: ' + json.dumps(dict(id='cmpl-1', choices=[], usage=dict(completion_tokens=15))), 'data: [DONE]'],
                             keep_content=True)
        self.assertEqual(result['chunk_tokens'], [1, 7, 4, 3])
        self.assertEqual(result['kinds'], 'rrcc')
        self.assertEqual(result['think_tokens'], 8)
        self.assertEqual(result['finish'], 'stop')
        self.assertEqual(result['request_id'], 'cmpl-1')
        self.assertEqual(result['content'], 'cd')
        self.assertEqual(len(result['output_ids']), 15)
        self.assertTrue(lab.turn_ok(result))

    def test_without_usage_stats_the_ids_count_and_a_tool_call_marks_its_offset(self):
        lines = ['data: ' + json.dumps(dict(id='c', choices=[dict(delta=dict(content='x'), token_ids=[1, 2])])),
                 'data: ' + json.dumps(dict(id='c', choices=[dict(delta=dict(tool_calls=[{}]), token_ids=[3], finish_reason='tool_calls')])),
                 'data: [DONE]']
        result = self.chunks(lines)
        self.assertEqual((result['chunk_tokens'], result['kinds'], result['tool_at']), ([2, 1], 'ct', 2))

    def test_an_error_chunk_and_an_empty_stream_are_not_ok(self):
        self.assertFalse(lab.turn_ok(self.chunks(['data: ' + json.dumps(dict(error=dict(message=TEXT)))])))
        self.assertNotIn(TEXT, json.dumps(self.chunks(['data: ' + json.dumps(dict(error=dict(message=TEXT)))])))
        self.assertFalse(lab.turn_ok(self.chunks(['data: [DONE]'])))


class DataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory, True)
        make_data(self.directory)

    def test_load_counts_and_defaults(self):
        data, manifest, files = lab.load_data(self.directory, list(lab.ARMS))
        self.assertEqual(lab.plan_counts(data, lab.ARMS), dict(A1=6, A2=2, A3=8, A4=2, A5=0))
        self.assertEqual(manifest, {})
        self.assertEqual(data['A1'][0]['set'], 'swe')
        self.assertEqual(data['A4'][0]['set'], 'chained')

    def test_a_malformed_file_is_refused_by_line_number_and_never_by_value(self):
        for bad, why in (('{not json', 'is not JSON'), (json.dumps(dict(id='a b', messages=[])), 'id must match'),
                         (json.dumps(dict(id='ok', messages=[], prompt_ids=[1])), 'exactly one'),
                         (json.dumps(dict(id='ok')), 'exactly one'),
                         (json.dumps(dict(id='ok', messages=[], set=TEXT)), 'set must be')):
            write_jsonl(os.path.join(self.directory, 'A2.jsonl'), [])
            with open(os.path.join(self.directory, 'A2.jsonl'), 'w') as handle:
                handle.write(bad + '\n')
            with self.assertRaises(lab.LabError) as caught:
                lab.load_data(self.directory, ['A2'])
            self.assertIn(why, str(caught.exception))
            self.assertIn('line 1', str(caught.exception))
            self.assertNotIn(TEXT, str(caught.exception))

    def test_a_repeated_id_and_an_escaping_file_name_are_refused(self):
        write_jsonl(os.path.join(self.directory, 'A2.jsonl'), [dict(id='a', messages=[]), dict(id='a', messages=[])])
        with self.assertRaises(lab.LabError):
            lab.load_data(self.directory, ['A2'])
        with open(os.path.join(self.directory, 'manifest.json'), 'w') as handle:
            json.dump(dict(format=1, arms=dict(A1=dict(file='../elsewhere.jsonl'))), handle)
        with self.assertRaises(lab.LabError):
            lab.load_data(self.directory, ['A1'])

    def test_a_missing_arm_file_is_an_empty_arm_and_a_wrong_format_is_refused(self):
        os.remove(os.path.join(self.directory, 'A3.jsonl'))
        self.assertEqual(lab.load_data(self.directory, ['A3'])[0]['A3'], [])
        with open(os.path.join(self.directory, 'manifest.json'), 'w') as handle:
            json.dump(dict(format=2), handle)
        with self.assertRaises(lab.LabError):
            lab.load_data(self.directory, ['A1'])

    def test_the_order_is_a_seeded_shuffle_of_the_ids_and_the_pair_is_the_first_hundred(self):
        data, _, _ = lab.load_data(self.directory, ['A1'])
        first = [record['id'] for record in lab.order_records(data['A1'], 7, 'A1')]
        self.assertEqual(first, [record['id'] for record in lab.order_records(list(reversed(data['A1'])), 7, 'A1')])
        self.assertNotEqual(first, [record['id'] for record in lab.order_records(data['A1'], 8, 'A1')])
        self.assertEqual(sorted(first), sorted(record['id'] for record in data['A1']))
        many = [dict(id='r%03d' % n) for n in range(150)]
        order = lab.order_records(many, 1, 'A1')
        self.assertEqual(lab.pair_records(order, []), order[:100])
        self.assertEqual(lab.pair_records(order, [dict(id='own')]), [dict(id='own')])

    def test_the_calibration_waves_are_fours_by_size_or_by_the_records_wave(self):
        records = [dict(id='c%d' % n, tokens=32768 if n % 2 else 4096) for n in range(8)]
        waves = lab.calibration_waves(records)
        self.assertEqual([len(wave) for wave in waves], [4, 4])
        self.assertTrue(all(record['tokens'] == 4096 for record in waves[0]))
        tagged = [dict(id='c%d' % n, wave=n % 2) for n in range(6)]
        self.assertEqual([len(wave) for wave in lab.calibration_waves(tagged)], [3, 3])

    def test_arms_are_parsed_in_their_own_order(self):
        self.assertEqual(lab.parse_arms('A5, A1 A3'), ['A1', 'A3', 'A5'])
        for bad in ('', 'A9', 'A1 A1'):
            with self.assertRaises(lab.LabError):
                lab.parse_arms(bad)


class SecretGuardTests(unittest.TestCase):
    """The lab's last guard over the data (the data agent scrubs first): fail closed, count only."""

    def leaks(self):
        # Built at run time so no literal secret sits in this public file.
        return dict(
            platform_token='thatch_' + 'sess_' + 'a1b2c3d4e5f6a7b8',
            api_key='sk' + '-' + 'q' * 24,
            github_token='gh' + 'p_' + 'Z' * 36,
            github_pat='github' + '_pat_' + 'Y' * 30,
            cloud_key='AK' + 'IA' + 'ABCDEFGHIJKLMNOP',
            chat_token='xo' + 'xb-' + '1234567890-abcdef',
            private_key='-----BEGIN ' + 'RSA PRIVATE KEY-----',
            bearer='Authorization: Bearer ' + 'abcdefghijklmnop1234',
            password_arg='plink -batch -p' + 'w hunter2 host',
            password_literal='password = "' + 'hunter22' + '"',
            ipv4='ssh 10.' + '20.30.40',
            email='me' + '@example' + '.org')

    def test_every_detector_fires_on_its_own_pattern_and_names_only_its_category(self):
        expected = dict(platform_token='platform_token', api_key='api_key', github_token='github_token',
                        github_pat='github_token', cloud_key='cloud_key', chat_token='chat_token', private_key='private_key',
                        bearer='bearer', password_arg='password_arg', password_literal='password_literal', ipv4='ipv4',
                        email='email')
        for name, text in self.leaks().items():
            with self.subTest(name=name):
                found = lab.secret_categories(dict(id='x', messages=[dict(content='before ' + text + ' after')]))
                self.assertIn(expected[name], found)
                self.assertTrue(all(isinstance(item, str) and text not in item for item in found))

    def test_ordinary_code_is_not_dropped(self):
        record = dict(id='x', messages=[dict(content='def f(password):\n    return token_ok(1.2.3, "v1.2")\n# see issue 1234')])
        self.assertEqual(lab.secret_categories(record), [])

    def test_a_dirty_record_is_dropped_and_counted_and_nothing_leaks_into_errors_or_output(self):
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, True)
        make_data(directory)
        leak = self.leaks()['platform_token']
        rows = [dict(id='zqclean', messages=messages('clean')),
                dict(id='zqdirty1', messages=[dict(role='user', content='run with ' + leak)]),
                dict(id='zqdirty2', messages=messages('x'), followups=['mail ' + self.leaks()['email']])]
        write_jsonl(os.path.join(directory, 'A4.jsonl'), rows)
        dropped = {}
        data, _, _ = lab.load_data(directory, ['A4'], dropped)
        self.assertEqual([record['id'] for record in data['A4']], ['zqclean'])
        self.assertEqual(dropped['A4'], dict(platform_token=1, email=1, records=2))
        self.assertNotIn(leak, json.dumps(dropped))

    def test_the_manifests_deny_patterns_join_the_guard(self):
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, True)
        make_data(directory)
        with open(os.path.join(directory, 'manifest.json'), 'w') as handle:
            json.dump(dict(format=1, deny=['zqrighost\\d+']), handle)
        write_jsonl(os.path.join(directory, 'A2.jsonl'), [dict(id='a', messages=[dict(content='ssh zqrighost7')]),
                                                           dict(id='b', messages=messages('ok'))])
        dropped = {}
        data, _, _ = lab.load_data(directory, ['A2'], dropped)
        self.assertEqual([record['id'] for record in data['A2']], ['b'])
        self.assertEqual(dropped['A2'], dict(deny=1, records=1))
        with open(os.path.join(directory, 'manifest.json'), 'w') as handle:
            json.dump(dict(format=1, deny=['(']), handle)
        with self.assertRaises(lab.LabError):
            lab.load_data(directory, ['A2'])

    def test_a_lab_run_prints_the_drop_counts_and_never_the_secret(self):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, True)
        data = os.path.join(root, 'data')
        os.makedirs(data)
        make_data(data)
        leak = self.leaks()['api_key']
        rows = [dict(id='zqswe%02d' % n, cluster='zqconv', set='swe', messages=messages('swe%d' % n)) for n in range(4)]
        rows.append(dict(id='zqswe99', set='swe', messages=[dict(role='user', content='key ' + leak)]))
        write_jsonl(os.path.join(data, 'A1.jsonl'), rows)
        code, out, results, public, engine, _ = run_lab(SimpleCase(), arguments=['--arms', 'A1'], data=data)
        self.assertIn('[TAULAB] secret guard dropped from A1: api_key=1 records=1', out)
        self.assertEqual(len(engine.requests), 4)
        self.assertNotIn(leak, '\n'.join(out))


class DockerTests(unittest.TestCase):
    def test_the_container_is_the_agents_shape_serving_the_derived_profile_and_never_sees_the_data(self):
        arguments = pg.server_run('img:tag', lab.CONTAINER, 'c2-packed-tp4+taulab', ['d0', 'd1', 'd2', 'd3'], lab.PORT, gate.HUB,
                                  '/results/profiles.json')
        self.assertIn('-d', arguments)
        self.assertNotIn('--rm', arguments)
        self.assertIn('QWEN_C2_PROFILE=c2-packed-tp4+taulab', arguments)
        self.assertEqual(arguments.count('--device'), 4)
        self.assertTrue(any(token.startswith('type=bind,src=/results/profiles.json') and token.endswith('readonly')
                            for token in arguments))
        self.assertIn('127.0.0.1:%d:8000' % lab.PORT, arguments)


class LabRunTests(unittest.TestCase):
    def test_a_whole_lab_runs_every_arm_and_reports_only_aggregates(self):
        code, out, results, public, engine, data = run_lab(self)
        self.assertEqual(code, 0, '\n'.join(out))
        turns = [json.loads(line) for line in read_text(os.path.join(results, 'turns.jsonl')).splitlines()]
        by_arm = {}
        for turn in turns:
            by_arm.setdefault(turn['arm'], []).append(turn)
        self.assertEqual(dict((arm, len(items)) for arm, items in by_arm.items()),
                         dict(A1=6, A2=2, A3=8, A4=6, A5=6))
        self.assertTrue(all(turn['status'] == 'ok' for turn in turns))
        summary = read_json(os.path.join(public, 'tau-lab-summary.json'))
        self.assertTrue(summary['complete'])
        self.assertEqual(summary['label'], 'production')
        self.assertEqual(summary['checks']['sources']['verdict'], 'PASS', summary['checks']['sources'])
        self.assertEqual(summary['checks']['sources']['stream_disagree'], 0)
        self.assertEqual(sorted(summary['arms']), ['A1', 'A2', 'A3', 'A4', 'A5'])
        for name in ('turns.jsonl', 'outputs.jsonl', 'server.log', 'run.json', 'profiles.json', 'docker-run.json', 'tapes.jsonl',
                     'report.private.json'):
            self.assertTrue(os.path.exists(os.path.join(results, name)), name)
        self.assertEqual(sorted(os.listdir(public)), ['tau-lab-summary.json'])
        # Counts and aggregates only: no sentinel, no id, no token from the data on stdout or in the artifact.
        shown = '\n'.join(out) + json.dumps(summary)
        for sentinel in (TEXT, 'zqswe', 'zqown', 'zqses', 'zqcal', 'zqconv', 'chatcmpl'):
            self.assertNotIn(sentinel, shown)
        self.assertIn('C2_TAULAB profile=c2-packed-tp4 complete=True', out)

    def test_the_arms_run_in_the_designs_order_with_the_designs_thinking(self):
        code, out, results, public, engine, data = run_lab(self)
        arms = [line.split()[2][:-1] for line in out if line.startswith('[TAULAB] arm ')]
        self.assertEqual(arms, ['A1', 'A2', 'A3', 'A4', 'A5'])
        by_text = {}
        for path, body in engine.requests:
            key = [message['content'] for message in body.get('messages', ())][-1] if body.get('messages') else None
            by_text.setdefault(key, []).append(body)
        thinking_off = [body for path, body in engine.requests if body.get('chat_template_kwargs')]
        self.assertEqual(len(thinking_off), 6, 'A5 only (the pair of A1)')
        for body in thinking_off:
            self.assertEqual(body['chat_template_kwargs'], dict(enable_thinking=False))
        completions = [body for path, body in engine.requests if path == '/v1/completions']
        self.assertEqual(len(completions), 8)
        self.assertTrue(all('chat_template_kwargs' not in body for body in completions), 'A3 is baked')
        turns = [json.loads(line) for line in read_text(os.path.join(results, 'turns.jsonl')).splitlines()]
        self.assertEqual(set(turn['thinking'] for turn in turns if turn['arm'] in ('A1', 'A2', 'A4')), {True})
        self.assertEqual(set(turn['thinking'] for turn in turns if turn['arm'] == 'A5'), {False})
        # A5 is the pair of A1: the same ids.
        self.assertEqual(sorted(turn['id'] for turn in turns if turn['arm'] == 'A5'),
                         sorted(turn['id'] for turn in turns if turn['arm'] == 'A1'))

    def test_a_chained_session_feeds_the_models_own_answers_back(self):
        code, out, results, public, engine, data = run_lab(self, arguments=['--arms', 'A4'])
        self.assertEqual(code, 0, '\n'.join(out))
        chains = [body['messages'] for path, body in engine.requests]
        self.assertEqual(len(chains), 6)
        longest = max(chains, key=len)
        self.assertEqual([message['role'] for message in longest], ['system', 'user', 'assistant', 'user', 'assistant', 'user'])
        answers = [message['content'] for message in longest if message['role'] == 'assistant']
        self.assertTrue(all(answer.startswith(TEXT + ' ANSWER') for answer in answers))
        self.assertEqual(len(set(answers)), 2)

    def test_a_killed_run_resumes_and_sends_nothing_twice(self):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, True)
        data, results = os.path.join(root, 'data'), os.path.join(root, 'results')
        os.makedirs(data)
        make_data(data)
        first = FakeEngine()
        run_lab(self, arguments=['--arms', 'A1 A4'], engine=first, data=data, results=results)
        sent = len(first.requests)
        self.assertEqual(sent, 12)
        # Drop the last lines of turns.jsonl: the run died before recording them.
        path = os.path.join(results, 'turns.jsonl')
        kept = read_text(path).splitlines()
        with open(path, 'w') as handle:
            handle.write('\n'.join(kept[:-3]) + '\n')
        second = FakeEngine()
        second.count = 1000
        code, out, _, public, _, _ = run_lab(self, arguments=['--arms', 'A1 A4'], engine=second, data=data, results=results)
        self.assertEqual(code, 0, '\n'.join(out))
        self.assertEqual(len(second.requests), 3)
        counts = read_json(os.path.join(results, 'run.json'))['counts']
        self.assertEqual(counts['A1']['resumed'] + counts['A4']['resumed'], 9)
        summary = read_json(os.path.join(public, 'tau-lab-summary.json'))
        self.assertEqual(summary['arms']['A1']['turns'] + summary['arms']['A4']['turns'], 12)

    def test_a_prompt_past_the_profiles_limit_is_refused_not_sent(self):
        code, out, results, public, engine, data = run_lab(self, arguments=['--arms', 'A1'])
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, True)
        big = os.path.join(root, 'data')
        os.makedirs(big)
        write_jsonl(os.path.join(big, 'A1.jsonl'), [dict(id='zq1', tokens=200000, messages=messages('big')),
                                                     dict(id='zq2', tokens=1000, messages=messages('ok'))])
        engine = FakeEngine()
        code, out, results, public, engine, _ = run_lab(self, arguments=['--arms', 'A1'], engine=engine, data=big)
        self.assertEqual(len(engine.requests), 1)
        self.assertEqual(read_json(os.path.join(results, 'run.json'))['counts']['A1']['refused'], 1)

    def test_the_deadline_stops_new_turns_and_the_report_says_incomplete(self):
        code, out, results, public, engine, data = run_lab(self, arguments=['--deadline-seconds', '900', '--arms', 'A1 A2'])
        # RESERVE_SECONDS is 900: no send time is left at all.
        self.assertEqual(len(engine.requests), 0)
        self.assertEqual(code, 1)
        counts = read_json(os.path.join(results, 'run.json'))['counts']
        self.assertEqual(counts['A1']['skipped'], 6)
        self.assertFalse(read_json(os.path.join(public, 'tau-lab-summary.json'))['complete'])

    def test_the_container_is_removed_however_the_run_ends_and_the_data_is_untouched(self):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, True)
        data = os.path.join(root, 'data')
        os.makedirs(data)
        make_data(data)
        before = dict((name, os.stat(os.path.join(data, name)).st_mtime_ns) for name in os.listdir(data))

        class Dying(FakeEngine):
            def stream(self, path, body, timeout, keep):
                raise RuntimeError('the engine died')
        code, out, results, public, engine, _ = run_lab(self, engine=Dying(), data=data)
        self.assertEqual(code, 1)
        stops = [call for call in self.docker_calls if call[:2] == ['docker', 'stop']]
        removes = [call for call in self.docker_calls if call[:3] == ['docker', 'rm', '-f']]
        self.assertEqual(len(stops), 1)
        self.assertGreaterEqual(len(removes), 2)
        self.assertEqual(dict((name, os.stat(os.path.join(data, name)).st_mtime_ns) for name in os.listdir(data)), before)

    def test_a_platform_container_refuses_before_anything_starts(self):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, True)
        data = os.path.join(root, 'data')
        os.makedirs(data)
        make_data(data)
        out, calls = [], []
        with mock.patch.object(gate, 'platform_containers', return_value=['thatch-inference-1']):
            code = lab.main(['--image', 'x/y:tp4-serve-2', '--data', data, '--results', os.path.join(root, 'r'),
                             '--profiles', os.path.join(HERE, 'qwen_c2_profiles.json')], say=out.append,
                            docker=lambda *a, **k: calls.append(a) or (0, ''), devices=['a', 'b', 'c', 'd'])
        self.assertEqual(code, 2)
        self.assertEqual(calls, [])

    def test_dry_run_prints_the_argv_and_the_counts_and_starts_nothing(self):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, True)
        data = os.path.join(root, 'data')
        os.makedirs(data)
        make_data(data)
        out = []
        code = lab.main(['--image', 'x/y:tp4-serve-2', '--data', data, '--results', os.path.join(root, 'r'), '--dry-run',
                         '--profiles', os.path.join(HERE, 'qwen_c2_profiles.json')], say=out.append)
        self.assertEqual(code, 0)
        self.assertIn('[TAULAB] data: A1=6 A2=2 A3=8 A4=2 A5=0', out)
        argv = json.loads(out[-1])
        self.assertIn('QWEN_C2_PROFILE=c2-packed-tp4+taulab', argv)
        self.assertFalse(os.path.exists(os.path.join(root, 'r')))
        self.assertNotIn(TEXT, '\n'.join(out))

    def test_analyze_only_reruns_the_report_over_a_finished_directory(self):
        code, out, results, public, engine, data = run_lab(self)
        os.remove(os.path.join(public, 'tau-lab-summary.json'))
        lines = []
        code = lab.main(['--data', data, '--results', results, '--public', public, '--analyze-only'], say=lines.append)
        self.assertEqual(code, 0)
        self.assertTrue(os.path.exists(os.path.join(public, 'tau-lab-summary.json')))
        self.assertTrue(any(line.startswith('[TAULAB-REPORT]') for line in lines))


# -- the report ------------------------------------------------------------------------------------------------------------

def p(request, segment, emitted, cap=None):
    text = '[PACKED] request=%s segment=%d position=10 prefix=%d emitted=%d predictions=[1]' % (request, segment, emitted - 1, emitted)
    return text + ('' if cap is None else ' cap=%d' % cap)


class ParseTests(unittest.TestCase):
    def test_rounds_are_read_by_name_per_request_in_log_order(self):
        log = '\n'.join(['2026-10-01T00:00:00Z ' + p('r1-0-aa', 2, 7), p('r1-0-aa', 2, 5, cap=3),
                         '[SEQUENTIAL] request=r1-0-aa position=50 prefix=3', '[SEQ-PUBLISH] request=r1-0-aa rows=4 prefix=2',
                         '[SEQ-PUBLISH] request=r1-0-aa stages=[1]', '[PACKED-COMMIT-HOST] round=3 adopt_ms=[1]'])
        rounds = rep.parse_rounds(log)['r1-0-aa']
        self.assertEqual([entry['kind'] for entry in rounds], ['P', 'P', 'S', 'S'])
        self.assertEqual([entry['emitted'] for entry in rounds[:2]], [7, 5])
        self.assertEqual((rounds[0]['segment'], rounds[1]['cap']), (2, 3))
        self.assertEqual(rounds[3]['rows'], 4)

    def test_the_engine_id_is_matched_to_its_stream_id_even_cut_at_48_characters(self):
        index = {'chatcmpl-' + 'a' * 32: 1}
        self.assertEqual(rep.owner_of('chatcmpl-' + 'a' * 32 + '-0-deadbeef', index), 'chatcmpl-' + 'a' * 32)
        self.assertEqual(rep.owner_of(('chatcmpl-' + 'a' * 32 + '-0-deadbeef')[:48], index), 'chatcmpl-' + 'a' * 32)
        self.assertEqual(rep.owner_of('chatcmpl-' + 'a' * 32, index), 'chatcmpl-' + 'a' * 32)
        self.assertIsNone(rep.owner_of('chatcmpl-' + 'b' * 32 + '-0-x', index))

    def test_buckets(self):
        self.assertEqual([rep.bucket_of(value) for value in (None, 100, 4096, 4097, 70000, 81920, 81921, 123000)],
                         [None, '4k', '4k', '8k', '80k', '80k', '80k+', '80k+'])


class TurnTests(unittest.TestCase):
    def rounds(self, values, tail=None, caps=()):
        out = [dict(kind='P', segment=index % 4, emitted=value, cap=caps[index] if index < len(caps) else None)
               for index, value in enumerate(values)]
        out += tail or []
        return out

    def test_the_terminal_round_and_the_sequential_steps_are_never_counted(self):
        stat = rep.analyze_turn(dict(prompt_tokens=9000), self.rounds([6, 7, 5, 16]))
        self.assertEqual((stat['rounds'], stat['emitted'], stat['tau']), (3, 18, 6.0))
        tail = [dict(kind='S', segment=None, emitted=None, rows=4, cap=None)]
        stat = rep.analyze_turn(dict(prompt_tokens=9000), self.rounds([6, 7, 5, 16], tail))
        self.assertEqual((stat['rounds'], stat['emitted']), (4, 34), 'a sequential step after them is the terminal round')
        self.assertEqual(rep.analyze_turn(dict(), [])['tau'], None)

    def test_a_capped_round_counts_as_served_and_not_in_the_uncapped_tau(self):
        stat = rep.analyze_turn(dict(), self.rounds([6, 7, 5, 16], caps=(None, 2, None, None)))
        self.assertEqual((stat['rounds'], stat['uncapped_rounds'], stat['uncapped_emitted']), (3, 2, 11))

    def test_the_output_region_follows_the_reasoning_and_tool_offsets(self):
        turn = dict(think_tokens=14, tool_at=30)
        stat = rep.analyze_turn(turn, self.rounds([6, 8, 8, 8, 8, 1]))
        self.assertEqual(stat['regions'], ['reasoning', 'reasoning', 'content', 'content', 'tool'])
        unknown = rep.analyze_turn(turn, self.rounds([6]) + [dict(kind='S', emitted=None, rows=4)] + self.rounds([6, 6, 6]))
        self.assertEqual(unknown['regions'][-1], None)


class StatisticsTests(unittest.TestCase):
    def test_quantiles_are_numpys_linear_ones(self):
        values = [1, 2, 3, 4, 10]
        self.assertEqual(rep.quantile(values, 0.5), 3)
        self.assertAlmostEqual(rep.quantile(values, 0.10), 1.4)
        self.assertEqual(rep.quantile([], 0.5), None)

    def test_the_pooled_tau_weights_the_sets_equally_and_the_interval_brackets_it(self):
        items = [('swe', 'c%d' % n, 6 * 40, 40) for n in range(10)] + [('own', 'o%d' % n, 4 * 40, 40) for n in range(10)]
        clusters = rep.cluster_tallies(items)
        tallies = dict((name, (sum(a for a, _ in pool), sum(b for _, b in pool))) for name, pool in clusters.items())
        self.assertAlmostEqual(rep.pooled_equal_weight(tallies), 5.0)
        interval = rep.bootstrap_ci(clusters, resamples=500, seed=1)
        self.assertEqual((interval['low'], interval['high']), (5.0, 5.0), 'identical clusters: no spread')
        spread = rep.cluster_tallies([('swe', 'a%d' % n, 40 * (4 + n % 5), 40) for n in range(20)])
        interval = rep.bootstrap_ci(spread, resamples=500, seed=1)
        self.assertLess(interval['low'], 6.0)
        self.assertGreater(interval['high'], 6.0)
        self.assertEqual(rep.bootstrap_ci(spread, resamples=200, seed=3), rep.bootstrap_ci(spread, resamples=200, seed=3))

    def test_worst_of_k_falls_as_k_grows_and_is_certain_for_a_flat_field(self):
        flat = dict(swe=[(8.0, 50)] * 5)
        self.assertEqual(rep.worst_of_k(flat, 9, (7.0, 9.0), draws=300), dict(bar_7=1.0, bar_9=0.0))
        mixed = dict(swe=[(4.0, 50), (9.0, 50)])
        low = rep.worst_of_k(mixed, 2, (5.0,), draws=2000, seed=1)['bar_5']
        high = rep.worst_of_k(mixed, 8, (5.0,), draws=2000, seed=1)['bar_5']
        self.assertLess(high, low)
        self.assertIsNone(rep.worst_of_k({}, 8, (5.0,)))

    def test_the_screens_pass_only_when_the_low_end_clears_and_fail_when_the_high_end_misses(self):
        self.assertEqual(rep.verdict(4.9, 5.8, 5.95), 'FAIL')
        self.assertEqual(rep.verdict(5.8, 6.9, 5.95), 'MAYBE')
        self.assertEqual(rep.verdict(6.0, 7.0, 5.95), 'PASS')
        entry = rep.screen(4.16, rep.K2_BARS)
        self.assertAlmostEqual(entry['projected_low'], 4.16 * 1.15 * 1.03, places=2)
        self.assertEqual(entry['bar_5_95'], 'FAIL')
        self.assertEqual(entry['bar_8_2'], 'FAIL')
        self.assertEqual(rep.screen(4.91, rep.K2_BARS)['bar_5_95'], 'MAYBE')
        self.assertIsNone(rep.screen(None, rep.K2_BARS))

    def test_the_positions_check_is_within_ten_percent_on_positions_one_to_three(self):
        good = rep.position_comparison([0.80, 0.70, 0.60], [0.82, 0.68, 0.62])
        self.assertEqual(good['verdict'], 'PASS')
        bad = rep.position_comparison([0.60, 0.70, 0.60], [0.82, 0.68, 0.62])
        self.assertEqual(bad['verdict'], 'FAIL')
        self.assertIsNone(rep.position_comparison(None, [1, 1, 1]))
        self.assertEqual(rep.production_positions(dict(accepted_per_pos=[80, 70, 60], num_drafts=100)), [0.8, 0.7, 0.6])
        self.assertEqual(rep.production_positions(dict(per_position=[0.5])), [0.5])
        self.assertIsNone(rep.production_positions(dict(other=1)))


class CountersTests(unittest.TestCase):
    SCRAPE = chr(10).join([
        '# HELP vllm:spec_decode_num_drafts_total Number of spec decoding drafts.',
        'vllm:spec_decode_num_drafts_total{engine="0",model_name="m"} 1000.0',
        'vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="0"} 800.0',
        'vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="1"} 700.0',
        'vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="2"} 600.0',
        'vllm:request_success_total{finished_reason="stop"} 5.0'])

    def test_aggregates_only_are_read_from_a_scrape(self):
        counters = rep.counters_from_metrics(self.SCRAPE)
        self.assertEqual(counters, dict(accepted_per_pos=[800.0, 700.0, 600.0], num_drafts=1000.0))
        self.assertEqual(rep.production_positions(counters), [0.8, 0.7, 0.6])
        self.assertIsNone(rep.counters_from_metrics('vllm:request_success_total 5.0'))

    def test_the_cli_writes_the_counters_and_the_report_uses_them(self):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, True)
        scrape, out = os.path.join(root, 'scrape.txt'), os.path.join(root, 'counters.json')
        with open(scrape, 'w') as handle:
            handle.write(self.SCRAPE)
        lines = []
        self.assertEqual(rep.main(['--metrics', scrape, '--counters-out', out], say=lines.append), 0)
        self.assertEqual(read_json(out)['num_drafts'], 1000.0)
        with open(scrape, 'w') as handle:
            handle.write('nothing')
        self.assertEqual(rep.main(['--metrics', scrape, '--counters-out', out], say=lines.append), 2)
        code, output, results, public, engine, data = run_lab(self, arguments=['--counters', out])
        self.assertEqual(code, 0)
        self.assertIn('positions_1_3', read_json(os.path.join(public, 'tau-lab-summary.json'))['checks'])


class PairWithoutItsSourceTests(unittest.TestCase):
    def test_a5_alone_still_re_sends_the_first_turns_of_a1s_order(self):
        code, out, results, public, engine, data = run_lab(self, arguments=['--arms', 'A5'])
        self.assertEqual(code, 0, chr(10).join(out))
        self.assertEqual(len(engine.requests), 6)
        self.assertTrue(all(body.get('chat_template_kwargs') == dict(enable_thinking=False) for _, body in engine.requests))
        self.assertIn('[TAULAB] data: A5=0', out)


class PrivacyGuardTests(unittest.TestCase):
    def test_aggregates_pass_and_everything_else_is_refused(self):
        rep.assert_public(dict(arms=dict(A1=dict(tau=5.5, turns=3, seats=dict(seat_0=5.0))), label='production',
                               image_tag='tp4-serve-2', bar_5_95='PASS', ok=True, none=None))
        for bad in (dict(text='a string of the data'), dict(ids=list(range(65))), dict(nested=[[1, 2]]), dict(x=float('nan')),
                    {'bad key with spaces': 1}, dict(image_tag='Has Spaces'), dict(obj=object()), dict(token='thatch_sess_abc')):
            with self.assertRaises(rep.PrivacyError, msg=str(bad)):
                rep.assert_public(bad)

    def test_the_public_summary_of_a_real_run_passes_the_guard_and_a_leak_is_refused_before_it_is_written(self):
        code, out, results, public, engine, data = run_lab(SimpleCase())
        summary = read_json(os.path.join(public, 'tau-lab-summary.json'))
        rep.assert_public(summary)
        summary['arms']['A1']['leak'] = 'zqswe00'
        with self.assertRaises(rep.PrivacyError):
            rep.assert_public(summary)


class SimpleCase(unittest.TestCase):
    """run_lab needs a TestCase for its cleanup and docker-call record; this one is never run as a test."""

    def runTest(self):
        pass


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)

    def run_report(self, manifest=None, counters=None, **options):
        code, out, results, public, engine, data = run_lab(self, **options)
        return rep.build(results, manifest=manifest, counters=counters, info=read_json(os.path.join(results, 'run.json')))

    def test_the_numbers_are_the_logs_numbers(self):
        public, private, tapes = self.run_report()
        expected = {}
        for tape in tapes:
            rounds = [entry for entry in tape['rounds']][:-1]
            expected.setdefault(tape['arm'], []).append((sum(entry[2] for entry in rounds), len(rounds)))
        for arm, pairs in expected.items():
            self.assertAlmostEqual(public['arms'][arm]['tau'], sum(a for a, _ in pairs) / float(sum(b for _, b in pairs)), places=3)
            self.assertEqual(public['arms'][arm]['rounds'], sum(b for _, b in pairs))
        self.assertEqual(public['arms']['A1']['turns'], 6)
        self.assertEqual(sorted(tape['arm'] for tape in tapes).count('A3'), 8)
        self.assertEqual(len(tapes), 6 + 2 + 8 + 6 + 6)

    def test_the_gate_inputs_are_present_and_the_arms_are_pooled_with_equal_set_weights(self):
        public, _, tapes = self.run_report()
        k = public['k_inputs']
        self.assertEqual(sorted(k['sets']), ['chained', 'own', 'swe'])
        taus = [k['sets'][name]['tau'] for name in ('swe', 'own', 'chained')]
        self.assertAlmostEqual(k['pooled_tau'], sum(taus) / 3.0, places=2)
        self.assertLessEqual(k['ci95']['low'], k['pooled_tau'])
        self.assertGreaterEqual(k['ci95']['high'], k['pooled_tau'])
        for key in ('worst_of_8', 'worst_of_9', 'buckets', 'regions', 'seats', 'k1', 'k2', 'per_turn', 'share_turns_under_4_2'):
            self.assertIn(key, k)
        self.assertIn('bar_10_35', k['worst_of_9'])
        self.assertIn('bar_5_95', k['worst_of_8'])
        self.assertIn('80k+', k['buckets'], 'the >80k bucket is separate')
        self.assertEqual(sorted(k['seats']), ['seat_0', 'seat_1', 'seat_2', 'seat_3'])
        self.assertEqual(k['k1']['bar'], 10.35)
        self.assertIn(k['k2']['p10']['bar_5_95'], ('PASS', 'FAIL', 'MAYBE'))
        self.assertIn('reasoning', k['regions'])
        self.assertIn('content', k['regions'])

    def test_thinking_on_against_off_is_paired_over_the_same_turns(self):
        public, _, _ = self.run_report()
        self.assertEqual(public['thinking']['pairs'], 6)
        self.assertIsNotNone(public['thinking']['tau_on'])
        self.assertAlmostEqual(public['thinking']['mean_difference'], public['thinking']['tau_off'] - public['thinking']['tau_on'],
                               places=2)

    def test_the_calibration_passes_within_seven_percent_and_fails_beyond(self):
        public, _, _ = self.run_report()
        lab_tau = public['calibration']['lab_tau']
        self.assertEqual(public['calibration']['verdict'], 'PASS' if abs(lab_tau - 6.0) / 6.0 <= 0.07 else 'FAIL')
        near = self.run_report(manifest=dict(a3_reference=dict(pooled_tau=lab_tau * 1.05)))[0]
        self.assertEqual(near['calibration']['verdict'], 'PASS')
        far = self.run_report(manifest=dict(a3_reference=dict(pooled_tau=lab_tau * 1.2)))[0]
        self.assertEqual(far['calibration']['verdict'], 'FAIL')
        self.assertEqual(far['checks']['calibration'], 'FAIL')

    def test_the_positions_check_needs_production_counters(self):
        public, _, _ = self.run_report()
        self.assertEqual(public['checks']['positions_1_3']['verdict'], 'NOT_ESTABLISHED')

    def test_a_disagreeing_stream_fails_the_sources_check(self):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, True)

        class Chunky(FakeEngine):
            def stream(self, path, body, timeout, keep):
                result = FakeEngine.stream(self, path, body, timeout, keep)
                result['chunk_tokens'] = [1] + [value + 1 for value in result['chunk_tokens'][1:]]
                return result
        public, _, _ = self.run_report(engine=Chunky())
        self.assertEqual(public['checks']['sources']['verdict'], 'FAIL')
        self.assertGreater(public['checks']['sources']['stream_disagree'], 0)

    def test_a_coalesced_stream_is_accepted(self):
        class Coalesced(FakeEngine):
            def stream(self, path, body, timeout, keep):
                result = FakeEngine.stream(self, path, body, timeout, keep)
                tokens = result['chunk_tokens']
                result['chunk_tokens'] = [tokens[0], tokens[1] + tokens[2]] + tokens[3:]
                return result
        public, _, _ = self.run_report(engine=Coalesced())
        self.assertEqual(public['checks']['sources']['stream_disagree'], 0)
        self.assertGreater(public['checks']['sources']['stream_coalesced'], 0)

    def test_a_non_production_image_is_labelled_so(self):
        code, out, results, public, engine, data = run_lab(self, arguments=['--image', 'registry.invalid/tt-vllm:qwen38-c2-tp4-next-1'])
        summary = read_json(os.path.join(public, 'tau-lab-summary.json'))
        self.assertEqual(summary['label'], 'non-production')
        self.assertEqual(summary['checks']['image'], 'non-production')

    def test_vllms_own_spec_decode_lines_give_the_positions(self):
        line = ('INFO SpecDecoding metrics: Mean acceptance length: 5.00, Accepted throughput: 1.00 tokens/s, Drafted throughput: '
                '1.00 tokens/s, Accepted: 400 tokens, Drafted: 1500 tokens, Per-position acceptance rate: 0.900, 0.800, 0.700, '
                '0.600, 0.500, 0.400, 0.300, 0.200, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100, Avg Draft acceptance rate: 26.7%')
        positions, source = rep.lab_positions(line)
        self.assertEqual(source, 'spec_lines')
        self.assertEqual(positions[:3], [0.9, 0.8, 0.7])


# -- the job action and the template -----------------------------------------------------------------------------------------

class JobTests(unittest.TestCase):
    def read(self, **values):
        base = dict(C2_IMAGE_TAG='tp4-serve-2', C2_CARDS='quad', C2_ACTIONS='reset taulab')
        base.update(values)
        return job.read_job(base, NAMES, meshes=job.profile_meshes())

    def test_the_defaults(self):
        outputs = self.read()
        self.assertEqual((outputs['taulab_profile'], outputs['taulab_arms'], outputs['taulab_deadline'], outputs['taulab_in_flight']),
                         ('c2-packed-tp4', 'A1 A2 A3 A4 A5', '270', '8'))
        self.assertEqual((outputs['taulab_data'], outputs['taulab_max_tokens'], outputs['taulab_counters']), ('', '', ''))

    def test_the_keys_are_only_read_when_the_action_runs(self):
        outputs = job.read_job(dict(C2_IMAGE_TAG='tp4-serve-2', C2_TAULAB_ARMS='nonsense'), NAMES)
        self.assertEqual(outputs['taulab_arms'], 'A1 A2 A3 A4 A5')

    def test_what_is_refused(self):
        for values in (dict(C2_CARDS='pair'), dict(C2_TAULAB_PROFILE='nope'), dict(C2_TAULAB_PROFILE='general'),
                       dict(C2_TAULAB_ARMS='A1 A9'), dict(C2_TAULAB_ARMS='A1 A1'), dict(C2_TAULAB_DEADLINE='0'),
                       dict(C2_TAULAB_DEADLINE='541'), dict(C2_TAULAB_IN_FLIGHT='17'), dict(C2_TAULAB_DATA='../etc'),
                       dict(C2_TAULAB_DATA='a/../b'), dict(C2_TAULAB_DATA='a b'), dict(C2_TAULAB_DATA='-rf'),
                       dict(C2_TAULAB_COUNTERS='$HOME/x'), dict(C2_ACTIONS='fabric taulab'), dict(C2_ACTIONS='taulab cardm')):
            with self.subTest(values=values), self.assertRaises(job.JobError):
                self.read(**values)

    def test_a_custom_job_parses(self):
        outputs = self.read(C2_TAULAB_ARMS='A3,A1', C2_TAULAB_DATA='kwork64/taulab/data', C2_TAULAB_MAX_TOKENS='512',
                            C2_TAULAB_DEADLINE='60', C2_TAULAB_COUNTERS='/srv/x/c.json')
        self.assertEqual((outputs['taulab_arms'], outputs['taulab_max_tokens'], outputs['taulab_deadline']), ('A3 A1', '512', '60'))

    def test_the_action_sits_between_prefix_and_replay(self):
        self.assertEqual(job.ACTIONS[job.ACTIONS.index('prefix') + 1], 'taulab')
        self.assertIn('taulab', job.QUAD_ACTIONS)


class TemplateTests(unittest.TestCase):
    def rows(self):
        with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
            return [line.split() for line in handle.read().splitlines() if line.strip() and not line.startswith('#')]

    def text(self):
        with open(os.path.join(FOLDER, 'W-T1-tau-lab.env'), encoding='utf-8') as handle:
            return handle.read()

    def test_the_template_is_the_one_job_the_order_names(self):
        self.assertEqual(self.rows(), [['W-T1-tau-lab', 'optional', 'tp4-serve-2', '275']])
        self.assertEqual(sorted(name for name in os.listdir(FOLDER) if name.endswith('.env')), ['W-T1-tau-lab.env'])

    def test_it_parses_to_the_production_image_the_production_profile_and_the_lab(self):
        outputs = job.read_job(job.parse_env(self.text()), NAMES, meshes=job.profile_meshes())
        self.assertEqual((outputs['cards'], outputs['tag'], outputs['taulab_profile']), ('quad', 'tp4-serve-2', 'c2-packed-tp4'))
        self.assertEqual(outputs['actions'], 'reset taulab')
        self.assertEqual(outputs['taulab_arms'], 'A1 A2 A3 A4 A5')
        self.assertEqual(outputs['taulab_data'], '', 'the data directory is the step\'s default, under the runner\'s home')
        self.assertLessEqual(int(outputs['taulab_deadline']), 270)

    def test_it_is_public_so_it_names_no_rig_card_address_registry_or_digest(self):
        for text in (self.text(), read_text(os.path.join(FOLDER, 'ORDER.txt'))):
            self.assertIsNone(BANNED.search(text), BANNED.search(text))


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        with open(WORKFLOW, encoding='utf-8') as handle:
            self.text = handle.read()
        start = self.text.index('      - name: Tau lab (W-T1) on the four-card set\n')
        end = self.text.index('\n      - name: ', start + 10)
        self.step = self.text[start:end]

    def test_the_step_is_quad_only_and_reads_only_outputs_the_parse_writes(self):
        self.assertIn("contains(steps.job.outputs.actions, 'taulab') && steps.job.outputs.cards == 'quad'", self.step)
        used = set(re.findall(r'steps\.job\.outputs\.([a-z_]+)', self.step))
        self.assertTrue(used >= {'taulab_profile', 'taulab_data', 'taulab_arms', 'taulab_deadline', 'taulab_in_flight', 'tag'})
        written = set(job.read_job(dict(C2_IMAGE_TAG='tp4-serve-2'), NAMES))
        self.assertEqual(sorted(used - written), [])

    def test_it_refuses_a_held_card_a_platform_container_and_a_missing_data_directory_before_the_driver(self):
        script = self.step
        driver = script.index('python3 scripts/ci/c2_tau_lab.py')
        for refusal in ("grep -q '^thatch-inference-'", 'card_set_unheld', 'the tau lab data directory is missing'):
            self.assertIn(refusal, script)
            self.assertLess(script.index(refusal), driver)
        self.assertIn('trap cleanup_taulab EXIT', script)

    def test_only_aggregates_reach_the_uploaded_artifact_and_the_privates_stay_under_kwork64_taulab(self):
        script = self.step
        self.assertIn('--public "$public"', script)
        self.assertIn('public="$RUNNER_TEMP/c2-results/taulab"', script)
        self.assertIn('run="$HOME/kwork64/taulab/results/run-$GITHUB_RUN_ID"', script)
        self.assertNotIn('c2-results/taulab/server', script)
        self.assertNotIn('cat "$run', script)
        self.assertNotIn('docker logs', script)
        self.assertIn('--data "$data"', script)
        self.assertNotIn('-v "$data', script)
        for write in re.findall(r'(?:mkdir -p|>|tee) +"?([^\s"|]+)', script):
            self.assertTrue(write.startswith(('"$run', '$run', '"$public', '$public', '"$RUNNER_TEMP', '$RUNNER_TEMP', '/dev/null')) or
                            write in ('$run', '$public'), write)

    def test_the_budget_is_the_smallest_of_the_step_the_job_and_the_job_file(self):
        self.assertIn('budget=$(( budget < box ? budget : box ))', self.step)
        self.assertIn('${C2_JOB_STARTED:?}', self.step)

    def test_the_header_documents_the_action(self):
        self.assertIn('#   taulab - the W-T1 tau lab', self.text)


class CpuWorkflowTests(unittest.TestCase):
    def test_the_tests_are_allowlisted_in_the_cpu_workflow(self):
        with open(CPU_WORKFLOW, encoding='utf-8') as handle:
            text = handle.read()
        self.assertRegex(text, r'python -B -m unittest [^\n]*\btest_tau_lab\b')

    def test_the_lab_modules_are_host_side_only(self):
        with open(os.path.join(ROOT, 'docker', 'qwen-c2-overlay.txt'), encoding='utf-8') as handle:
            overlay = handle.read()
        for name in ('c2_tau_lab', 'tau_lab_report'):
            self.assertNotIn(name, overlay, 'the driver runs on the host: the image never imports it')


if __name__ == '__main__':
    unittest.main()
