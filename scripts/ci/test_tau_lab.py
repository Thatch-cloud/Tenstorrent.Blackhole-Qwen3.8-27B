"""The W-T1 tau lab (c2_tau_lab.py, tau_lab_report.py, the taulab job action, the job template), held on CPU with fakes.

Nothing here opens a device, docker or a socket: the driver runs against a FakeEngine that answers each request with rounds
of its own, writes the matching [PACKED-PHASE] / [PACKED] (with cap=) and fast_serving_phases lines into the server log the
report reads, and the data is synthetic, in the REAL file layout of the data directory (FORMAT.md: MANIFEST.json, one
conversation per line plus gzip token-id files, a build spec, session seeds), with sentinel strings that must never come out.
The Qwen repo is public: what the lab prints and what its public summary holds are checked for every sentinel."""

import copy
import gzip
import hashlib
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

sys.dont_write_bytecode = True
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
REFERENCE = os.path.join(HERE, 'references', 'tau-lab', 'a3-reference.json')
with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as _handle:
    PROFILES = json.load(_handle)
NAMES = sorted(PROFILES['profiles'])
TEXT = 'SENTINELTEXT'
# What no file of this change may carry: a board id, a host or registry name, a digest, an address (loopback aside), a device node,
# a home directory. The guard reads the lab's own code, its reference and the workflow step, not only the template.
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|(?!127\.0\.0\.1)\b\d{1,3}(\.\d{1,3}){3}\b|sha256:[0-9a-f]{16}|'
                    r'[0-9a-f]{40,}|/dev/tenstorrent|home/|zot\.|@[A-Z0-9_]+@|\bzot:|registry\.[a-z]+\.[a-z]+')


def read_text(path):
    with open(path, encoding='utf-8') as handle:
        return handle.read()


def read_json(path):
    return json.loads(read_text(path))


def write_jsonl(path, records):
    with open(path, 'w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record) + '\n')


def write_gz(path, records):
    with gzip.open(path, 'wt', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record) + '\n')


def messages(tag):
    return [dict(role='system', content='%s system %s' % (TEXT, tag)), dict(role='user', content='%s user %s' % (TEXT, tag))]


def turn_of(turn_id, tokens, bucket='8-16k', eligible=4, chosen=2):
    return dict(turn_id=turn_id, upto=2, bucket=bucket, prompt_tokens_think_on=tokens, prompt_tokens_think_off=tokens - 3,
                recorded_output_tokens=50, eligible_per_bucket={bucket: eligible}, chosen_in_bucket={bucket: chosen})


def conversations(arm, count, per, tokens, big=False):
    """[conversation] in FORMAT.md's shape (a SWE-rebench-shaped record carries a git identity and an email in its text)."""
    out = []
    for n in range(count):
        cid = '%s-%04d' % (arm.lower(), n)
        turns = [turn_of('%s-t%d' % (cid, k), 124000 + k if (big and n == 0 and k == 0) else (85000 if (arm == 'A2' and n == 0 and k == 1) else tokens + 10 * n + k),
                         eligible=3 + k, chosen=1 + k % 2) for k in range(per)]
        out.append(dict(id=cid, arm=arm, source='zqsource', license='CC-BY-4.0' if arm == 'A1' else 'internal',
                        group='%012x' % (n // 2 + (0 if arm == 'A1' else 500)), tools=[dict(type='function')],
                        messages=messages(cid) + [dict(role='tool', content='git config user.email openhands' + '@' + 'all-hands.dev')],
                        turns=turns))
    return out


def ids_rows(convs, think):
    return [dict(turn_id=turn['turn_id'], think=think, n=turn['prompt_tokens_think_on' if think == 'on' else 'prompt_tokens_think_off'],
                 ids=list(range(turn['prompt_tokens_think_on' if think == 'on' else 'prompt_tokens_think_off'])))
            for conv in convs for turn in conv['turns']]


def sha256_of(path):
    with open(path, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def make_data(directory, swe=3, per=2, own=1, sessions=2, a5=3, big=False, manifest_arms=None, scrub_hits=None, seed_turns=3):
    """A synthetic data directory in the real layout (FORMAT.md): A1 (swe conversations x per turns), A2 (own), A3's build spec,
    A4's seeds, A5's pairs and OFF ids, scrub_report.json and MANIFEST.json (sizes, sha256, record counts, arm counts)."""
    a1 = conversations('A1', swe, per, 9000, big)
    a2 = conversations('A2', own, per, 30000)
    write_jsonl(os.path.join(directory, 'a1_swe_heldout.jsonl'), a1)
    write_gz(os.path.join(directory, 'a1_swe_heldout.ids.jsonl.gz'), ids_rows(a1, 'on'))
    write_jsonl(os.path.join(directory, 'a2_own_sessions.jsonl'), a2)
    write_gz(os.path.join(directory, 'a2_own_sessions.ids.jsonl.gz'), ids_rows(a2, 'on'))
    pair_ids = [turn['turn_id'] for conv in a1 for turn in conv['turns']][:a5]
    with open(os.path.join(directory, 'a5_pairs.json'), 'w') as handle:
        json.dump(dict(arm='A5', think='off', turn_ids=pair_ids, rule='zq'), handle)
    off = [row for row in ids_rows(a1, 'off') if row['turn_id'] in pair_ids]
    write_gz(os.path.join(directory, 'a5_think_off.ids.jsonl.gz'), off)
    with open(os.path.join(directory, 'a3_calibration.json'), 'w') as handle:
        json.dump(dict(arm='A3', sets=[dict(name='4x4k', users=4, target=4096), dict(name='4x32k', users=4, target=32768)],
                       request=dict(max_tokens=256), turns=8), handle)
    write_jsonl(os.path.join(directory, 'a4_seeds.jsonl'), [
        dict(id='a4-%02d' % n, name='zq-a4-%02d' % n, seed='zq-a4-%02d' % n, system='full', first_tokens=0, turns=seed_turns,
             task_template='Explain `{name}` in {path}.', prompt_targets=[1500 + 500 * k for k in range(seed_turns)], thinking='on')
        for n in range(sessions)])
    with open(os.path.join(directory, 'scrub_report.json'), 'w') as handle:
        json.dump(dict(conversations_in=own, conversations_kept=own, conversations_dropped=0,
                       final_detector_hits_on_written_file=scrub_hits or {}), handle)
    names = ['a1_swe_heldout.jsonl', 'a1_swe_heldout.ids.jsonl.gz', 'a2_own_sessions.jsonl', 'a2_own_sessions.ids.jsonl.gz',
             'a5_pairs.json', 'a5_think_off.ids.jsonl.gz', 'a3_calibration.json', 'a4_seeds.jsonl', 'scrub_report.json']
    files = dict((name, dict(bytes=os.path.getsize(os.path.join(directory, name)), sha256=sha256_of(os.path.join(directory, name))))
                 for name in names)
    arms = dict(A1=dict(conversations=swe, turns=swe * per), A2=dict(conversations=own, turns=own * per),
                A3=dict(turns=8, built_in_container=True), A4=dict(sessions=sessions, turns=sessions * seed_turns),
                A5=dict(turns=len(pair_ids)))
    arms.update(manifest_arms or {})
    with open(os.path.join(directory, 'MANIFEST.json'), 'w') as handle:
        json.dump(dict(written='2026-10-01', format='FORMAT.md', files=files, arms=arms), handle)


def packed_lines(request_id, emitted, number, cap_of=None, live=4):
    """The lines one request leaves in the container's log, in the REAL format (v162): a [PACKED-PHASE] line per round (live=),
    then one [PACKED] line per user carrying cap= (16 unless the round was cut)."""
    lines, position = [], 1000
    for index, value in enumerate(emitted):
        cap = (cap_of(index) if cap_of else 16)
        lines.append('(EngineCore pid=90) 2026-10-01 00:00:00.000 | INFO     | packed_verifier:diagnostic:205 - [PACKED-PHASE] '
                     'round=%d users=4 bind_ms=0.00 input_ms=1.63 trace_ms=58.78 sync_ms=0.04 readback_ms=0.61 live=%d idle=-'
                     % (index + 1, live))
        lines.append('(EngineCore pid=90) 2026-10-01 00:00:00.000 | INFO     | serving_packed_step:audit_log:134 - [PACKED] '
                     'request=%s-0-ab12cd34 segment=%d position=%d prefix=%d emitted=%d predictions=[1, 2, 3] cap=%d'
                     % (request_id, number % 4, position, value - 1, value, cap))
        position += value
    return lines


class FakeEngine(object):
    """What the container would do: each request gets rounds, the server log gets their lines, the stream their counts."""

    def __init__(self, path=None, rounds=12, think_at=3, tool_every=0, live=4, finish_chat='stop'):
        self.path = path
        self.rounds = rounds
        self.think_at, self.tool_every, self.live, self.finish_chat = think_at, tool_every, live, finish_chat
        self.lock = threading.Lock()
        self.count = 0
        self.requests = []
        self.lines = []

    def emit(self, lines):
        with self.lock:
            self.lines.extend(lines)
            if self.path:
                with open(self.path, 'a', encoding='utf-8') as handle:
                    handle.write('\n'.join(lines) + '\n')

    def stream(self, path, body, timeout, keep):
        with self.lock:
            self.count += 1
            number = self.count
            self.requests.append((path, copy.deepcopy(body)))
        rng = random.Random(number)
        emitted = [rng.randint(2, 15) for _ in range(self.rounds)]
        request_id = 'cmpl-%016x' % number
        lines = packed_lines(request_id, emitted, number, live=self.live)
        lines.append(json.dumps(dict(stage='fast_serving_phases', blocks=[
            dict(position=1000, rows=16, committed=value, verifier=dict(packed=True, users=4)) for value in emitted])))
        self.emit(lines)
        total = 1 + sum(emitted)
        chat = path.endswith('chat/completions')
        reasoning = chat and 'chat_template_kwargs' not in body
        kinds = ''.join(['r' if offset < 2 else 'c' for offset in range(len(emitted) + 1)]) if reasoning else 'c' * (len(emitted) + 1)
        ids = list(range(total))
        if self.think_at is not None and total > self.think_at:
            ids[self.think_at] = lab.THINK_END_TOKEN
        has_tool = bool(self.tool_every) and number % self.tool_every == 0
        if has_tool:
            ids[total // 2] = lab.TOOL_CALL_TOKEN
        usage = dict(prompt_tokens=len(json.dumps(body)) // 3 if chat else len(body.get('prompt') or ()), completion_tokens=total)
        calls = [dict(id='call_%d' % number, type='function', function=dict(name='Read', arguments='{"file_path": "x"}'))] \
            if has_tool and chat else []
        return dict(request_id=request_id, chunk_tokens=[1] + emitted, kinds=kinds,
                    think_tokens=(1 + sum(emitted[:2])) if reasoning else 0, tool_at=None,
                    finish=('stop' if has_tool else self.finish_chat) if chat else 'length', error=None, usage=usage,
                    output_ids=ids, content='%s ANSWER %d' % (TEXT, number), tool_calls=calls, ttft=0.5, status=200)


class FakeClient(object):
    def ready(self):
        return True

    def context_tokens(self):
        return 131328


class FakeContainer(object):
    def __init__(self, alive=lambda: True):
        self.alive = alive

    def running(self):
        return self.alive()


class FakeLog(object):
    def __init__(self, engine, path):
        self.engine, self.path = engine, path

    def start(self, since=None):
        self.engine.path = self.path
        self.engine.emit(['[QWEN-C2] profile c2-packed-tp4+taulab booted'])

    def lines(self):
        return list(self.engine.lines)


def fake_a3(image=None):
    """What build_calibration returns: 4 prompts of 4096 and 4 of 32768 token ids."""
    return dict(((name, [list(range(target))] * users) for name, users, target in (('4x4k', 4, 4096), ('4x32k', 4, 32768)))), \
        dict(files=1, characters=1)


class SmallCorpus(object):
    """The A4 corpus the tests use: two tiny real-shaped sources (the lab's own interface of prefix_agent_corpus.Corpus)."""

    @staticmethod
    def make():
        import prefix_agent_corpus as corpus_module
        text = '\n'.join('def function_number_%d(value):\n    return value + %d' % (n, n) for n in range(60))
        return corpus_module.Corpus([corpus_module.Source('pkg/mod_a.py', text), corpus_module.Source('pkg/mod_b.py', text),
                                     corpus_module.Source('README.md', 'readme ' * 100)])


def run_lab(testcase, arguments=None, engine=None, data=None, results=None, alive=None, **options):
    """main() against the fakes -> (exit code, output lines, results dir, public dir, engine, data dir)."""
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
    container = FakeContainer(alive) if alive else FakeContainer()
    with mock.patch.object(gate, 'platform_containers', return_value=[]):
        code = lab.main(argv, say=out.append, make_stream=lambda: engine.stream, make_client=FakeClient,
                        make_container=lambda name: container, make_log=lambda name, path: FakeLog(engine, path),
                        docker=lambda arguments, timeout=0: (calls.append(arguments) or (0, '')), devices=['d0', 'd1', 'd2', 'd3'],
                        sleep=lambda seconds: None, build_a3=fake_a3, corpus=SmallCorpus.make(),
                        make_canary_check=lambda log_path, results: (lambda: None))
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

    def test_the_audits_are_part_of_what_production_means(self):
        self.assertTrue(lab.audits_on(PROFILES, 'c2-packed-tp4'))
        self.assertFalse(lab.audits_on(PROFILES, 'no-such-profile'))

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

    def test_compact_id_arrays_are_sent_as_plain_json_lists(self):
        import array
        body = lab.request_body(dict(id='x', prompt_ids=array.array('i', [5, 6, 7])), True, 10, 'm')[1]
        self.assertEqual(json.loads(json.dumps(body))['prompt'], [5, 6, 7])

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
    """The reader loads the REAL layout of the data directory (a regression of the first version, which read other file names
    and loaded every arm empty): conversations plus gzip ids, pairs, seeds, a build spec, all against MANIFEST.json."""

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory, True)
        make_data(self.directory)

    def test_load_counts_ids_clusters_tokens_and_weights_from_the_real_layout(self):
        data, manifest, scrub = lab.load_data(self.directory, list(lab.ARMS))
        self.assertEqual(lab.plan_counts(data, lab.ARMS), dict(A1=6, A2=2, A3=8, A4=2, A5=3))
        self.assertEqual(manifest['format'], 'FORMAT.md')
        first = data['A1'][0]
        self.assertEqual(first['id'], 'a1-0000-t0')                       # id = turn_id
        self.assertEqual(first['cluster'], '%012x' % 0)                   # cluster = the conversation's group
        self.assertEqual((first['set'], first['turn'], first['tokens']), ('swe', 0, 9000))
        self.assertEqual(len(first['prompt_ids']), first['tokens'])       # the ids of the thinking-ON render
        self.assertEqual(first['weight'], 3.0 / 1)                        # eligible_per_bucket / chosen_in_bucket
        self.assertEqual(data['A1'][1]['weight'], 4.0 / 2)
        self.assertEqual(data['A2'][0]['set'], 'own')
        self.assertTrue(all(record['cluster'] in ('%012x' % n for n in (0, 1)) for record in data['A1']))
        self.assertEqual(data['A4'][0]['set'], 'chained')
        self.assertTrue(all(record['pending'] for record in data['A3']))
        self.assertEqual(sorted(set(record['wave'] for record in data['A3'])), [0, 1])

    def test_a5_is_exactly_the_pairs_turn_ids_under_a1s_ids_with_the_off_render(self):
        data, _, _ = lab.load_data(self.directory, ['A1', 'A5'])
        pairs = read_json(os.path.join(self.directory, 'a5_pairs.json'))['turn_ids']
        self.assertEqual([record['id'] for record in data['A5']], pairs)
        by_id = dict((record['id'], record) for record in data['A1'])
        for record in data['A5']:
            mine = by_id[record['id']]
            self.assertEqual((record['cluster'], record['weight']), (mine['cluster'], mine['weight']))
            self.assertEqual(len(record['prompt_ids_nothink']), mine['tokens'] - 3)
            self.assertEqual(len(record['prompt_ids']), mine['tokens'])
        alone, _, _ = lab.load_data(self.directory, ['A5'])               # A1's pair of files is read for it
        self.assertEqual([record['id'] for record in alone['A5']], pairs)

    def test_a_missing_arm_file_is_refused_never_an_empty_arm(self):
        os.remove(os.path.join(self.directory, 'a4_seeds.jsonl'))
        with self.assertRaises(lab.LabError) as caught:
            lab.load_data(self.directory, ['A4'])
        self.assertIn('a4_seeds.jsonl', str(caught.exception))

    def test_every_file_is_checked_against_the_manifest(self):
        with open(os.path.join(self.directory, 'a2_own_sessions.jsonl'), 'a') as handle:
            handle.write('\n')                                            # one byte more than the manifest says
        with self.assertRaises(lab.LabError) as caught:
            lab.load_data(self.directory, ['A2'])
        self.assertIn('does not match', str(caught.exception))
        make_data(self.directory)
        manifest = read_json(os.path.join(self.directory, 'MANIFEST.json'))
        manifest['format'] = 1
        with open(os.path.join(self.directory, 'MANIFEST.json'), 'w') as handle:
            json.dump(manifest, handle)
        with self.assertRaises(lab.LabError):
            lab.load_data(self.directory, ['A1'])
        os.remove(os.path.join(self.directory, 'MANIFEST.json'))
        with self.assertRaises(lab.LabError):
            lab.load_data(self.directory, ['A1'])

    def test_a_shortfall_against_the_manifests_arm_counts_is_refused(self):
        for arm, counts in (('A1', dict(turns=7)), ('A2', dict(turns=3)), ('A4', dict(sessions=3, turns=6)), ('A5', dict(turns=4)),
                            ('A3', dict(turns=9))):
            make_data(self.directory, manifest_arms={arm: counts})
            with self.assertRaises(lab.LabError, msg=arm) as caught:
                lab.load_data(self.directory, [arm])
            self.assertIn('manifest says', str(caught.exception))
        make_data(self.directory, manifest_arms=dict(A4=dict(sessions=2, turns=7)))
        with self.assertRaises(lab.LabError):
            lab.load_data(self.directory, ['A4'])

    def test_a_malformed_file_is_refused_by_line_number_and_never_by_value(self):
        for name, bad in (('a4_seeds.jsonl', '{not json'), ('a4_seeds.jsonl', json.dumps(dict(id=TEXT)))):
            make_data(self.directory)
            with open(os.path.join(self.directory, name), 'w') as handle:
                handle.write(bad + '\n')
            manifest = read_json(os.path.join(self.directory, 'MANIFEST.json'))
            manifest['files'][name] = dict(bytes=os.path.getsize(os.path.join(self.directory, name)),
                                           sha256=sha256_of(os.path.join(self.directory, name)))
            with open(os.path.join(self.directory, 'MANIFEST.json'), 'w') as handle:
                json.dump(manifest, handle)
            with self.assertRaises(lab.LabError) as caught:
                lab.load_data(self.directory, ['A4'])
            self.assertIn('line 1', str(caught.exception))
            self.assertNotIn(TEXT, str(caught.exception))

    def test_a_session_seed_lacking_a_field_is_refused(self):
        seeds = [json.loads(line) for line in read_text(os.path.join(self.directory, 'a4_seeds.jsonl')).splitlines()]
        for field in lab.SEED_FIELDS:
            broken = [dict((key, value) for key, value in seed.items() if key != field) for seed in seeds]
            path = os.path.join(self.directory, 'broken.jsonl')
            write_jsonl(path, broken)
            with self.assertRaises(lab.LabError, msg=field):
                lab.seed_records(path)

    def test_ids_that_disagree_with_their_turns_are_refused(self):
        rows = [json.loads(line) for line in gzip.open(os.path.join(self.directory, 'a1_swe_heldout.ids.jsonl.gz'), 'rt')]
        rows[0]['n'] += 1
        path = os.path.join(self.directory, 'a1_swe_heldout.ids.jsonl.gz')
        write_gz(path, rows)
        manifest = read_json(os.path.join(self.directory, 'MANIFEST.json'))
        manifest['files']['a1_swe_heldout.ids.jsonl.gz'] = dict(bytes=os.path.getsize(path), sha256=sha256_of(path))
        with open(os.path.join(self.directory, 'MANIFEST.json'), 'w') as handle:
            json.dump(manifest, handle)
        with self.assertRaises(lab.LabError):
            lab.load_data(self.directory, ['A1'])

    def test_the_order_is_a_seeded_shuffle_of_the_ids(self):
        data, _, _ = lab.load_data(self.directory, ['A1'])
        first = [record['id'] for record in lab.order_records(data['A1'], 7, 'A1')]
        self.assertEqual(first, [record['id'] for record in lab.order_records(list(reversed(data['A1'])), 7, 'A1')])
        self.assertNotEqual(first, [record['id'] for record in lab.order_records(data['A1'], 8, 'A1')])
        self.assertEqual(sorted(first), sorted(record['id'] for record in data['A1']))

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

    def test_the_real_shipped_layout_is_what_the_format_document_names(self):
        """The file names the reader asks for are the ones FORMAT.md of the data directory lists (kept in step by name)."""
        names = set(name for arm in lab.ARMS for name in lab.ARM_FILES[arm]) | set([lab.SCRUB_REPORT_NAME, lab.MANIFEST_NAME])
        self.assertEqual(names, set(['a1_swe_heldout.jsonl', 'a1_swe_heldout.ids.jsonl.gz', 'a2_own_sessions.jsonl',
                                     'a2_own_sessions.ids.jsonl.gz', 'a3_calibration.json', 'a4_seeds.jsonl', 'a5_pairs.json',
                                     'a5_think_off.ids.jsonl.gz', 'scrub_report.json', 'MANIFEST.json']))


class DataGateTests(unittest.TestCase):
    """What stands in for the text guard: the lab sends token ids (no text is loaded, so none can be printed), A2 is gated on the
    scrub report's final check, and SWE-rebench-shaped public text (an identity email in every trajectory) is not a problem."""

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory, True)

    def test_a_swe_rebench_shaped_record_loads_whole_and_sends_ids_not_text(self):
        make_data(self.directory)
        text = read_text(os.path.join(self.directory, 'a1_swe_heldout.jsonl'))
        self.assertIn('all-hands', text)                                 # the git identity is in every trajectory
        data, _, _ = lab.load_data(self.directory, ['A1', 'A2'])
        self.assertEqual((len(data['A1']), len(data['A2'])), (6, 2))     # none dropped for an email or a loopback address
        for record in data['A1'] + data['A2']:
            path, body = lab.request_body(record, True, 10, 'm')
            self.assertEqual(path, '/v1/completions')
            self.assertNotIn('messages', record)
            self.assertNotIn('messages', body)
            self.assertNotIn(TEXT, json.dumps(body))

    def test_residual_detector_hits_in_a_scrub_report_refuse_a2_before_anything_runs(self):
        make_data(self.directory, scrub_hits=dict(conv_blob=1))
        with self.assertRaises(lab.LabError) as caught:
            lab.load_data(self.directory, ['A2'])
        self.assertIn('residual', str(caught.exception))
        self.assertEqual(lab.load_data(self.directory, ['A1'])[2], {})   # A1 does not read the report
        make_data(self.directory)
        report = os.path.join(self.directory, 'scrub_report.json')
        with open(report, 'w') as handle:
            json.dump(dict(conversations_kept=1), handle)               # a report with no final check at all
        manifest = read_json(os.path.join(self.directory, 'MANIFEST.json'))
        manifest['files']['scrub_report.json'] = dict(bytes=os.path.getsize(report), sha256=sha256_of(report))
        with open(os.path.join(self.directory, 'MANIFEST.json'), 'w') as handle:
            json.dump(manifest, handle)
        with self.assertRaises(lab.LabError):
            lab.load_data(self.directory, ['A2'])

    def test_a_run_prints_counts_and_never_the_data(self):
        code, out, results, public, engine, _ = run_lab(self, arguments=['--arms', 'A1 A2'])
        shown = '\n'.join(out) + json.dumps(read_json(os.path.join(public, 'tau-lab-summary.json')))
        for sentinel in (TEXT, 'a1-0', 'a2-0', 'all-hands', 'zqsource'):
            self.assertNotIn(sentinel, shown)
        self.assertIn('[TAULAB] A2 scrub report: kept=1 dropped=0, final check clean', out)


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


class CalibrationTests(unittest.TestCase):
    def test_the_prompts_are_built_in_a_throwaway_container_from_the_image(self):
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, True)
        make_data(directory)
        calls = []

        def runner(arguments, **kwargs):
            calls.append(arguments)
            out = json.dumps(dict(prompts=dict(((name, [[1] * target] * users) for name, users, target in (
                ('4x4k', 4, 4096), ('4x32k', 4, 32768)))), files=3, characters=9))
            return mock.Mock(returncode=0, stdout=out.encode(), stderr=b'')
        built, info = lab.build_calibration('img:tag', os.path.join(directory, 'a3_calibration.json'), ROOT, runner=runner)
        argv = calls[0]
        self.assertEqual(argv[:6], ['docker', 'run', '--rm', '--network', 'none', '-e'])
        self.assertNotIn('--device', argv)
        self.assertIn('img:tag', argv)
        self.assertTrue(any(token.endswith('dst=/bench/real_text_prompts.py,readonly') for token in argv))
        self.assertEqual(sorted(built), ['4x32k', '4x4k'])
        self.assertEqual((len(built['4x4k']), len(built['4x32k'][0])), (4, 32768))
        self.assertEqual(info, dict(files=3, characters=9))
        records = lab.calibration_records(os.path.join(directory, 'a3_calibration.json'))
        lab.attach_calibration(records, built)
        self.assertTrue(all(not record['pending'] and len(record['prompt_ids']) == record['tokens'] for record in records))
        self.assertEqual(set(record['max_tokens'] for record in records), {2400})

    def test_a_failed_or_misshapen_build_is_a_refusal(self):
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, True)
        make_data(directory)
        spec = os.path.join(directory, 'a3_calibration.json')
        with self.assertRaises(lab.LabError):
            lab.build_calibration('i', spec, ROOT, runner=lambda *a, **k: mock.Mock(returncode=1, stdout=b'', stderr=b''))
        short = json.dumps(dict(prompts={'4x4k': [[1] * 10] * 4, '4x32k': [[1] * 32768] * 4}))
        with self.assertRaises(lab.LabError):
            lab.build_calibration('i', spec, ROOT, runner=lambda *a, **k: mock.Mock(returncode=0, stdout=short.encode(), stderr=b''))

    def test_the_reference_is_the_reports_estimator_over_the_lanes_logs_and_a_missing_one_refuses(self):
        reference = lab.read_a3_reference()
        self.assertEqual(sorted(reference['sets']), ['32k', '4k'])
        self.assertGreater(reference['pooled_tau'], 5.0)
        self.assertIn('four-live', reference['estimator'])
        self.assertTrue(reference['source'])
        self.assertIsNone(BANNED.search(read_text(REFERENCE)), BANNED.search(read_text(REFERENCE)))
        with self.assertRaises(lab.LabError):
            lab.read_a3_reference('/nonexistent/a3.json')
        path = os.path.join(tempfile.mkdtemp(), 'a3.json')
        self.addCleanup(shutil.rmtree, os.path.dirname(path), True)
        with open(path, 'w') as handle:
            json.dump(dict(pooled_tau=5.9), handle)
        with self.assertRaises(lab.LabError):
            lab.read_a3_reference(path)

    def test_the_calibration_is_sent_thinking_off_four_at_a_time_with_the_lanes_budget(self):
        code, out, results, public, engine, data = run_lab(self, arguments=['--arms', 'A3', '--max-tokens', '512'])
        self.assertEqual(code, 0, '\n'.join(out))
        self.assertEqual(len(engine.requests), 8)
        self.assertTrue(all(path == '/v1/completions' and body['max_tokens'] == 2400 for path, body in engine.requests),
                        'the global budget does not change the calibration: tau depends on the output position')
        turns = [json.loads(line) for line in read_text(os.path.join(results, 'turns.jsonl')).splitlines()]
        self.assertEqual(set(turn['thinking'] for turn in turns), {False})
        summary = read_json(os.path.join(public, 'tau-lab-summary.json'))
        self.assertEqual(summary['arms']['A3']['thinking'], 'off')
        self.assertIn(summary['calibration']['verdict'], ('PASS', 'FAIL'))
        self.assertEqual(summary['calibration']['reference_tau'], round(lab.read_a3_reference()['pooled_tau'], 3))

    def test_an_a3_run_without_its_reference_is_refused_before_anything_starts(self):
        out = []
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, True)
        make_data(directory)
        with mock.patch.object(lab, 'A3_REFERENCE', '/nonexistent/a3.json'):
            code = lab.main(['--image', 'x/y:tp4-serve-2', '--data', directory, '--results', os.path.join(directory, 'r'),
                             '--arms', 'A3', '--profiles', os.path.join(HERE, 'qwen_c2_profiles.json')], say=out.append,
                            docker=lambda *a, **k: (0, ''), devices=['a', 'b', 'c', 'd'])
        self.assertEqual(code, 2)
        self.assertTrue(any('A3 reference' in line for line in out))


class LabRunTests(unittest.TestCase):
    def test_a_whole_lab_runs_every_arm_and_reports_only_aggregates(self):
        code, out, results, public, engine, data = run_lab(self)
        self.assertEqual(code, 0, '\n'.join(out))
        turns = [json.loads(line) for line in read_text(os.path.join(results, 'turns.jsonl')).splitlines()]
        by_arm = {}
        for turn in turns:
            by_arm.setdefault(turn['arm'], []).append(turn)
        self.assertEqual(dict((arm, len(items)) for arm, items in by_arm.items()),
                         dict(A1=6, A2=2, A3=8, A4=6, A5=3))
        self.assertTrue(all(turn['status'] == 'ok' for turn in turns))
        summary = read_json(os.path.join(public, 'tau-lab-summary.json'))
        self.assertTrue(summary['complete'])
        self.assertFalse(summary['stopped'])
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
        for sentinel in (TEXT, 'a1-0', 'a2-0', 'a3-4x', 'a4-0', 'zq', 'cmpl-'):
            self.assertNotIn(sentinel, shown)
        self.assertIn('C2_TAULAB profile=c2-packed-tp4 complete=True', out)

    def test_the_arms_run_in_the_designs_order_with_the_designs_thinking(self):
        code, out, results, public, engine, data = run_lab(self)
        arms = [line.split()[2][:-1] for line in out if line.startswith('[TAULAB] arm ')]
        self.assertEqual(arms, ['A1', 'A2', 'A3', 'A4', 'A5'])
        thinking_off = [body for path, body in engine.requests if body.get('chat_template_kwargs')]
        self.assertEqual(thinking_off, [], 'the OFF render is in the ids: a completions turn has no template kwargs')
        completions = [body for path, body in engine.requests if path == '/v1/completions']
        self.assertEqual(len(completions), 6 + 2 + 8 + 3, 'A1, A2, A3 and A5 are token ids; only A4 is chat')
        chats = [body for path, body in engine.requests if path == '/v1/chat/completions']
        self.assertEqual(len(chats), 6)
        self.assertTrue(all('chat_template_kwargs' not in body for body in chats), 'A4 is thinking ON: the template default')
        turns = [json.loads(line) for line in read_text(os.path.join(results, 'turns.jsonl')).splitlines()]
        self.assertEqual(set(turn['thinking'] for turn in turns if turn['arm'] in ('A1', 'A2', 'A4')), {True})
        self.assertEqual(set(turn['thinking'] for turn in turns if turn['arm'] in ('A3', 'A5')), {False})
        # A5 is the pair of A1: the same ids, the OFF render (3 tokens shorter than A1's ON render in the fake data).
        a5 = [turn for turn in turns if turn['arm'] == 'A5']
        a1 = dict((turn['id'], turn) for turn in turns if turn['arm'] == 'A1')
        self.assertEqual(len(a5), 3)
        self.assertTrue(all(turn['id'] in a1 and turn['declared_tokens'] == a1[turn['id']]['declared_tokens'] - 3 for turn in a5))
        # Every turn carries its cluster, index and the data's inverse-probability weight.
        self.assertTrue(all(turn['weight'] > 0 and turn['cluster'] for turn in turns))
        self.assertEqual(sorted(set(turn['weight'] for turn in turns if turn['arm'] == 'A1')), [2.0, 3.0])

    def test_a_chained_session_grows_and_feeds_the_models_answers_and_tool_calls_back(self):
        code, out, results, public, engine, data = run_lab(self, arguments=['--arms', 'A4'], engine=FakeEngine(tool_every=2))
        self.assertEqual(code, 0, '\n'.join(out))
        chains = [body['messages'] for path, body in engine.requests]
        self.assertEqual(len(chains), 6)
        longest = max(chains, key=len)
        roles = [message['role'] for message in longest]
        self.assertEqual(roles[:3], ['system', 'user', 'assistant'])
        every = [message for chain in chains for message in chain]
        self.assertIn('tool', [message['role'] for message in every], 'a tool-call answer is answered with the tool result')
        called = [message for message in every if message['role'] == 'assistant' and message.get('tool_calls')]
        self.assertTrue(called and called[0]['tool_calls'][0]['function']['name'] == 'Read')
        answers = [message['content'] for message in longest if message['role'] == 'assistant']
        self.assertTrue(all(answer.startswith(TEXT + ' ANSWER') for answer in answers))
        self.assertTrue(all('reasoning' not in message for message in every), 'the reasoning is not sent back')
        # The first message is filled from the corpus (a real file and function), the tools are the agent's nine.
        self.assertIn('pkg/mod_', chains[0][1]['content'])
        self.assertIn('tools', engine.requests[0][1])
        self.assertEqual(read_json(os.path.join(results, 'run.json'))['counts']['A4']['fed_empty'], 0)

    def test_a_session_whose_answer_ended_inside_its_reasoning_is_counted_as_fed_empty(self):
        class Empty(FakeEngine):
            def stream(self, path, body, timeout, keep):
                result = FakeEngine.stream(self, path, body, timeout, keep)
                result['content'] = ''
                result['finish'] = 'length'
                return result
        code, out, results, public, engine, data = run_lab(self, arguments=['--arms', 'A4'], engine=Empty())
        counts = read_json(os.path.join(results, 'run.json'))['counts']['A4']
        self.assertEqual(counts['fed_empty'], 6)
        self.assertEqual(counts['ok'], 6)

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
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, True)
        big = os.path.join(root, 'data')
        os.makedirs(big)
        make_data(big, big=True)
        engine = FakeEngine()
        code, out, results, public, engine, _ = run_lab(self, arguments=['--arms', 'A1'], engine=engine, data=big)
        self.assertEqual(len(engine.requests), 5)
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

    def test_a_crash_mid_run_never_reads_as_a_complete_run(self):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, True)
        data = os.path.join(root, 'data')
        os.makedirs(data)
        make_data(data)
        with mock.patch.object(lab.Lab, 'run_calibration', side_effect=RuntimeError('boom')):
            code, out, results, public, engine, _ = run_lab(self, data=data, arguments=['--arms', 'A1 A3'])
        self.assertEqual(code, 1)
        info = read_json(os.path.join(results, 'run.json'))
        self.assertEqual(info['error'], 'RuntimeError')
        self.assertEqual(info['counts']['A1']['ok'], 6, 'the counts of whatever ran are saved, in the finally block')
        summary = read_json(os.path.join(public, 'tau-lab-summary.json'))
        self.assertFalse(summary['complete'])
        self.assertTrue(summary['stopped'])
        self.assertNotIn('boom', chr(10).join(out))

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
                            docker=lambda *a, **k: calls.append(a) or (0, ''), devices=['a', 'b', 'c', 'd'], build_a3=fake_a3,
                            corpus=SmallCorpus.make())
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
                         '--profiles', os.path.join(HERE, 'qwen_c2_profiles.json')], say=out.append, corpus=SmallCorpus.make())
        self.assertEqual(code, 0)
        self.assertIn('[TAULAB] data: A1=6 A2=2 A3=8 A4=2 A5=3', out)
        argv = json.loads(out[-1])
        self.assertIn('QWEN_C2_PROFILE=c2-packed-tp4+taulab', argv)
        self.assertFalse(os.path.exists(os.path.join(root, 'r')))
        self.assertNotIn(TEXT, '\n'.join(out))

    def test_dry_run_and_a_run_refuse_every_shortfall_before_any_container(self):
        """A regression: the first version loaded every arm empty, printed 'A1=0 ...' and exited 0."""
        for arms in (dict(A1=dict(turns=7)), dict(A2=dict(turns=3)), dict(A4=dict(sessions=21, turns=120)), dict(A5=dict(turns=101))):
            root = tempfile.mkdtemp()
            self.addCleanup(shutil.rmtree, root, True)
            data = os.path.join(root, 'data')
            os.makedirs(data)
            make_data(data, manifest_arms=arms)
            for extra in (['--dry-run'], []):
                out, calls = [], []
                code = lab.main(['--image', 'x/y:tp4-serve-2', '--data', data, '--results', os.path.join(root, 'r'),
                                 '--profiles', os.path.join(HERE, 'qwen_c2_profiles.json')] + extra, say=out.append,
                                docker=lambda *a, **k: calls.append(a) or (0, ''), devices=['a', 'b', 'c', 'd'], build_a3=fake_a3,
                                corpus=SmallCorpus.make())
                self.assertEqual(code, 2, (arms, extra, out))
                self.assertTrue(any(line.startswith('refused: arm') for line in out), out)
                self.assertEqual(calls, [])
        out = []
        code = lab.main(['--image', 'x/y:tp4-serve-2', '--data', '/nonexistent/data', '--results', '/tmp/r', '--dry-run',
                         '--profiles', os.path.join(HERE, 'qwen_c2_profiles.json')], say=out.append)
        self.assertEqual(code, 2)

    def test_the_card_set_and_image_errors_name_no_board_in_the_log(self):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, True)
        data = os.path.join(root, 'data')
        os.makedirs(data)
        make_data(data)
        board = 'blackhole-' + 'ABCDEF0123456789'
        out = []
        with mock.patch.object(gate, 'devices_for', side_effect=RuntimeError('card set: ' + board)), \
                mock.patch.object(gate, 'platform_containers', return_value=[]):
            code = lab.main(['--image', 'x/y:tp4-serve-2', '--data', data, '--results', os.path.join(root, 'r'),
                             '--profiles', os.path.join(HERE, 'qwen_c2_profiles.json')], say=out.append, corpus=SmallCorpus.make())
        self.assertEqual(code, 2)
        self.assertNotIn(board, '\n'.join(out))
        self.assertTrue(any('RuntimeError' in line for line in out))
        out = []
        with mock.patch.object(gate, 'image_profiles', side_effect=RuntimeError('image ' + board)):
            code = lab.main(['--image', 'x/y:tp4-serve-2', '--data', data, '--results', os.path.join(root, 'r')], say=out.append,
                            corpus=SmallCorpus.make())
        self.assertEqual(code, 2)
        self.assertNotIn(board, '\n'.join(out))

    def test_analyze_only_reruns_the_report_over_a_finished_directory(self):
        code, out, results, public, engine, data = run_lab(self)
        os.remove(os.path.join(public, 'tau-lab-summary.json'))
        lines = []
        code = lab.main(['--data', data, '--results', results, '--public', public, '--analyze-only'], say=lines.append)
        self.assertEqual(code, 0)
        self.assertTrue(os.path.exists(os.path.join(public, 'tau-lab-summary.json')))
        self.assertTrue(any(line.startswith('[TAULAB-REPORT]') for line in lines))

    def test_a_non_production_label_when_the_verify_audits_are_off(self):
        profiles = copy.deepcopy(PROFILES)
        del profiles['profiles']['c2-packed-tp4']['env']['QWEN_FAST_VERIFY_T2_AUDIT']
        self.assertFalse(lab.audits_on(profiles, 'c2-packed-tp4'))
        self.assertTrue(lab.audits_on(PROFILES, 'c2-packed-tp4'))
        path = os.path.join(tempfile.mkdtemp(), 'profiles.json')
        self.addCleanup(shutil.rmtree, os.path.dirname(path), True)
        with open(path, 'w') as handle:
            json.dump(profiles, handle)
        code, out, results, public, engine, data = run_lab(self, arguments=['--arms', 'A1', '--profiles', path])
        self.assertEqual(read_json(os.path.join(public, 'tau-lab-summary.json'))['label'], 'non-production')


class HangTests(unittest.TestCase):
    """The hang handling: a breaker on consecutive transport failures, a container that is gone, and the canary."""

    def hung(self, **kwargs):
        class Hung(FakeEngine):
            def stream(self, path, body, timeout, keep):
                with self.lock:
                    self.count += 1
                    self.requests.append((path, body))
                return dict(error='connection: timeout', chunk_tokens=[], output_ids=[], status=None)
        return Hung(**kwargs)

    def test_consecutive_transport_failures_stop_the_lab_and_the_report_still_comes(self):
        code, out, results, public, engine, data = run_lab(self, engine=self.hung(), arguments=['--in-flight', '1'])
        self.assertEqual(code, 1)
        self.assertEqual(len(engine.requests), lab.BREAKER_FAILURES, 'sends nothing past the breaker')
        info = read_json(os.path.join(results, 'run.json'))
        self.assertIn('breaker', info['tripped'])
        self.assertTrue(read_json(os.path.join(public, 'tau-lab-summary.json'))['stopped'])
        self.assertTrue(any('stopping the lab: breaker' in line for line in out))
        self.assertGreater(info['counts']['A1']['skipped'], 0)

    def test_a_refused_request_does_not_trip_the_breaker(self):
        self.assertFalse(lab.transport_failure(dict(status='error', http=400)))
        self.assertTrue(lab.transport_failure(dict(status='error', http=None)))
        self.assertTrue(lab.transport_failure(dict(status='error', http=503)))
        self.assertFalse(lab.transport_failure(dict(status='ok', http=200)))

    def test_a_container_that_is_gone_stops_the_lab(self):
        state = dict(up=True)

        class Mortal(FakeEngine):
            def stream(self, path, body, timeout, keep):
                if self.count >= 2:
                    state['up'] = False
                return FakeEngine.stream(self, path, body, timeout, keep)
        code, out, results, public, engine, data = run_lab(self, engine=Mortal(), alive=lambda: state['up'],
                                                           arguments=['--in-flight', '1', '--arms', 'A1'])
        self.assertEqual(code, 1)
        self.assertLessEqual(len(engine.requests), 3)
        self.assertEqual(read_json(os.path.join(results, 'run.json'))['tripped'], 'the container is not running')

    def test_the_canary_runs_once_after_the_first_ok_turns_and_stops_a_lab_without_its_log_flags(self):
        calls = []

        def canary():
            calls.append(1)
            return 'no [PACKED] rounds of the lab\'s turns in the log (the log flags are not in effect)'
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, True)
        data = os.path.join(root, 'data')
        os.makedirs(data)
        make_data(data)
        engine = FakeEngine()
        out = []
        with mock.patch.object(gate, 'platform_containers', return_value=[]):
            code = lab.main(['--image', 'x/y:tp4-serve-2', '--data', data, '--results', os.path.join(root, 'r'),
                             '--public', os.path.join(root, 'p'), '--deadline-seconds', '100000', '--in-flight', '1', '--arms', 'A1 A2',
                             '--profiles', os.path.join(HERE, 'qwen_c2_profiles.json')], say=out.append,
                            make_stream=lambda: engine.stream, make_client=FakeClient, make_container=lambda name: FakeContainer(),
                            make_log=lambda name, path: FakeLog(engine, path), docker=lambda *a, **k: (0, ''),
                            devices=['a', 'b', 'c', 'd'], sleep=lambda s: None, build_a3=fake_a3, corpus=SmallCorpus.make(),
                            make_canary_check=lambda log_path, results: canary)
        self.assertEqual(code, 1)
        self.assertEqual(calls, [1], 'once')
        self.assertEqual(len(engine.requests), lab.CANARY_TURNS)
        self.assertTrue(any('stopping the lab: canary' in line for line in out))

    def test_the_real_canary_reads_the_log_for_attributed_packed_rounds_and_phase_records(self):
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, True)
        log = os.path.join(directory, 'server.log')

        class Done(object):
            done = {('A1', 'x'): dict(request_id='cmpl-%016x' % 7)}
        clock = dict(now=0.0)
        check = lab.make_canary(log, Done(), clock=lambda: clock['now'], sleep=lambda s: clock.update(now=clock['now'] + s), wait=20)
        self.assertIn('no [PACKED]', check())                                   # no log at all
        with open(log, 'w') as handle:
            handle.write('\n'.join(packed_lines('cmpl-%016x' % 7, [5, 6, 7], 7)) + '\n')
        clock['now'] = 0.0
        self.assertIn('fast_serving_phases', check())                           # rounds but no phases record
        with open(log, 'a') as handle:
            handle.write(json.dumps(dict(stage='fast_serving_phases', blocks=[])) + '\n')
        clock['now'] = 0.0
        self.assertIsNone(check())
        with open(log, 'w') as handle:                                           # rounds of a request the lab never sent
            handle.write('\n'.join(packed_lines('cmpl-%016x' % 99, [5, 6, 7], 7)) + '\n' + '{"stage": "fast_serving_phases"}\n')
        clock['now'] = 0.0
        self.assertIn('no [PACKED] rounds', check())

    def test_the_read_timeout_is_ten_minutes_not_twenty_five(self):
        self.assertEqual(lab.REQUEST_READ_TIMEOUT, 600)
        self.assertLessEqual(lab.REQUEST_READ_TIMEOUT + lab.TURN_GRACE_SECONDS, lab.RESERVE_SECONDS)


class RegionTests(unittest.TestCase):
    def test_regions_come_from_the_output_token_ids_on_the_completions_path(self):
        ids = list(range(40))
        ids[9] = lab.THINK_END_TOKEN
        ids[30] = lab.TOOL_CALL_TOKEN
        self.assertEqual(lab.regions_of(ids, True, {}), (10, 30, True, True))
        self.assertEqual(lab.regions_of(list(range(10)), True, {}), (10, None, False, False))   # never left its reasoning
        self.assertEqual(lab.regions_of(ids, False, {}), (0, 30, True, True))                   # thinking OFF has no reasoning
        self.assertEqual(lab.regions_of([], True, dict(think_tokens=8, tool_at=None, kinds='rrcc')), (8, None, True, False))
        self.assertEqual(lab.regions_of([], True, dict(think_tokens=0, tool_at=5, kinds='rrt')), (0, 5, True, True))


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
    def test_a5_alone_sends_exactly_the_pairs_turns_thinking_off(self):
        code, out, results, public, engine, data = run_lab(self, arguments=['--arms', 'A5'])
        self.assertEqual(code, 0, chr(10).join(out))
        self.assertEqual(len(engine.requests), 3)
        self.assertTrue(all(path == '/v1/completions' for path, _ in engine.requests))
        self.assertIn('[TAULAB] data: A5=3', out)
        pairs = read_json(os.path.join(data, 'a5_pairs.json'))['turn_ids']
        turns = [json.loads(line) for line in read_text(os.path.join(results, 'turns.jsonl')).splitlines()]
        self.assertEqual(sorted(turn['id'] for turn in turns), sorted(pairs))
        sizes = sorted(len(body['prompt']) for _, body in engine.requests)
        self.assertEqual(sizes, [8997, 8998, 9007])         # the OFF renders, 3 shorter than the ON ones


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
        summary['arms']['A1']['leak'] = 'a1-0000-t0'
        with self.assertRaises(rep.PrivacyError):
            rep.assert_public(summary)


class SimpleCase(unittest.TestCase):
    """run_lab needs a TestCase for its cleanup and docker-call record; this one is never run as a test."""

    def runTest(self):
        pass


class ReportStatisticTests(unittest.TestCase):
    """The K statistic as the design has it: four-live 16-row rounds, caps read from cap=, weights, regions from token ids. Each
    test is built from a miss of the first review (the real log line formats are in packed_lines)."""

    def log(self, requests):
        lines = []
        for number, (emitted, live, cap_of) in enumerate(requests):
            lines += packed_lines('cmpl-%016x' % number, emitted, number, cap_of=cap_of, live=live)
        return '\n'.join(lines)

    def test_every_extent_path_line_carries_cap_and_cap_16_is_the_uncut_round(self):
        # Regression: `cap is None` excluded every round of the real log (931 lines at cap=16 of 981), so tau_uncapped was None.
        text = self.log([([6, 7, 5, 9, 4], 4, lambda index: 16)])
        rounds = rep.parse_rounds(text)['cmpl-%016x-0-ab12cd34' % 0]
        self.assertEqual([entry['cap'] for entry in rounds], [16] * 5)
        stat = rep.analyze_turn(dict(prompt_tokens=9000), rounds, 4)
        self.assertEqual((stat['rounds'], stat['uncapped_rounds'], stat['capped_rounds']), (4, 4, 0))
        self.assertAlmostEqual(stat['uncapped_emitted'] / float(stat['uncapped_rounds']), stat['tau'])

    def test_a_round_cut_below_16_is_capped_and_one_that_emitted_its_cap_is_censored(self):
        text = self.log([([6, 3, 2, 5, 9], 4, lambda index: 3 if index in (1, 2) else 16)])
        stat = rep.analyze_turn(dict(), rep.parse_rounds(text)['cmpl-%016x-0-ab12cd34' % 0], 4)
        self.assertEqual((stat['rounds'], stat['capped_rounds'], stat['censored_rounds']), (4, 2, 1))   # the 3 reached its cap

    def test_only_the_four_live_rounds_count_and_the_others_are_tallied_apart(self):
        # Regression: padded 3-live and 2-live rounds also log [PACKED]; the v162 ladder is 4p 5.89, 3p 5.08, 2p 4.55.
        lines = []
        for live, values in ((4, [8, 8, 8]), (3, [5, 5]), (2, [4, 4, 4]), (4, [8]), (4, [1])):
            lines += packed_lines('cmpl-%016x' % 0, values, 0, live=live)
        rounds = rep.parse_rounds(chr(10).join(lines))['cmpl-%016x-0-ab12cd34' % 0]
        self.assertEqual([entry['live'] for entry in rounds], [4, 4, 4, 3, 3, 2, 2, 2, 4, 4])
        stat = rep.analyze_turn(dict(), rounds, 4)
        self.assertEqual((stat['rounds'], stat['emitted']), (4, 32))      # the last round is the terminal one
        self.assertEqual(stat['other_live'], {3: [10, 2], 2: [12, 3]})
        everything = rep.analyze_turn(dict(), rounds)
        self.assertEqual(everything['rounds'], 9)                         # without a live rule every packed round counts
        self.assertTrue(rep.has_live(dict(r=rounds)))
        self.assertFalse(rep.has_live(dict(r=[dict(kind='P', live=None)])))

    def test_a_cancelled_commit_with_emitted_zero_is_not_a_round(self):
        rounds = [dict(kind='P', segment=0, emitted=6, cap=16, live=4), dict(kind='P', segment=0, emitted=0, cap=16, live=4),
                  dict(kind='P', segment=0, emitted=7, cap=16, live=4), dict(kind='P', segment=0, emitted=5, cap=16, live=4)]
        stat = rep.analyze_turn(dict(), rounds, 4)
        self.assertEqual((stat['rounds'], stat['emitted']), (2, 13))

    def test_a_golden_log_in_the_real_format_gives_the_estimate_by_hand(self):
        text = self.log([([5, 6, 7, 8, 9, 2], 4, None), ([10, 4, 4, 16], 4, None), ([3, 3, 3], 3, None)])
        found = rep.estimate_log(text)
        # request 0: 5+6+7+8+9 over 5 rounds (the terminal 2 is cut), request 1: 10+4+4 over 3; the 3-live request counts nowhere
        self.assertEqual((found['emitted'], found['rounds']), (5 + 6 + 7 + 8 + 9 + 10 + 4 + 4, 8))
        self.assertAlmostEqual(found['tau'], 53 / 8.0)

    @unittest.skipUnless(os.environ.get('TAULAB_GOLDEN_LOG'), 'set TAULAB_GOLDEN_LOG to a padded-4k server.log of a logged run')
    def test_a_real_padded_4k_log_gives_the_reference_tau(self):
        with open(os.environ['TAULAB_GOLDEN_LOG'], encoding='utf-8', errors='replace') as handle:
            found = rep.estimate_log(handle.read())
        self.assertAlmostEqual(found['tau'], float(os.environ.get('TAULAB_GOLDEN_TAU', '5.89')), delta=0.01)

    def test_the_inverse_probability_weights_reach_the_pooled_tau_the_p10_and_the_buckets(self):
        def turn(arm, ident, weight, rounds, tau_emitted, cluster, tokens=9000):
            return dict(arm=arm, id=ident, status='ok', set='swe', cluster=cluster, weight=weight, prompt_tokens=tokens,
                        chunk_tokens=[1], think_tokens=0, tool_at=None, finish='length', request_id='cmpl-%016x' % int(ident[1:]),
                        thinking=True), (tau_emitted, rounds)
        specs = [turn('A1', 'x1', 1.0, 10, 4, 'c1'), turn('A1', 'x2', 9.0, 10, 8, 'c2')]
        lines = []
        for record, (value, rounds) in specs:
            number = int(record['id'][1:])
            lines += packed_lines(record['request_id'], [value] * (rounds + 1), number)
        public, _, _ = rep.build(None, turns=[record for record, _ in specs], log_text='\n'.join(lines), info=dict(arms=['A1']))
        k = public['k_inputs']
        self.assertAlmostEqual(k['pooled_tau'], (1 * 4 * 10 + 9 * 8 * 10) / float(1 * 10 + 9 * 10), places=2)   # 7.6, not the sample's 6.0
        self.assertAlmostEqual(public['arms']['A1']['tau'], 7.6, places=2)
        self.assertAlmostEqual(public['arms']['A1']['tau_sample'], 6.0, places=2)
        self.assertGreater(k['per_turn']['p10'], 4.0)                    # the heavy turn pulls the weighted p10 up from the minimum
        self.assertEqual(k['per_turn']['min'], 4.0)
        self.assertAlmostEqual(k['buckets']['16k']['tau'], 7.6, places=2)

    def built(self, specs):
        """specs: [(arm, id, segment, emitted per round, rounds)] -> the public summary over turns and a log in the real format."""
        turns, lines = [], []
        for number, (arm, ident, segment, value, rounds) in enumerate(specs):
            request_id = 'cmpl-%016x' % number
            turns.append(dict(arm=arm, id=ident, status='ok', set='swe', cluster='c%d' % number, weight=1.0, prompt_tokens=9000,
                              chunk_tokens=[1], think_tokens=0, tool_at=None, finish='length', request_id=request_id,
                              thinking=(arm != 'A5')))
            lines += packed_lines(request_id, [value] * (rounds + 1), segment)
        return rep.build(None, turns=turns, log_text=chr(10).join(lines), info=dict(arms=sorted(set(spec[0] for spec in specs))))[0]

    def test_the_p10_the_p50_the_segment_minimum_and_the_pairing_are_each_their_own_number(self):
        # Three mutants of the first version survived: p10 read as p50, the worst segment as the best, A5 paired against the wrong arm.
        specs = [('A1', 'x%d' % n, n % 4, 4 + n, 10) for n in range(8)]                # taus 4..11 on segments 0..3
        public = self.built(specs)
        per_turn = public['arms']['A1']['tau_per_turn']
        self.assertAlmostEqual(per_turn['p10'], 4.7, places=2)
        self.assertAlmostEqual(per_turn['p50'], 7.5, places=2)
        self.assertNotEqual(per_turn['p10'], per_turn['p50'])
        self.assertEqual(public['arms']['A1']['segment_min'], min(public['arms']['A1']['seats'].values()))
        self.assertLess(public['arms']['A1']['segment_min'], max(public['arms']['A1']['seats'].values()))
        self.assertAlmostEqual(public['arms']['A1']['segment_min'], 4.0 + 4 / 2.0, places=2)    # segment 0: turns 4 and 8 -> (4+8)/2
        pair = self.built([('A1', 'x1', 0, 4, 10), ('A1', 'x2', 1, 8, 10), ('A5', 'x1', 2, 6, 10), ('A5', 'x3', 3, 9, 10)])
        self.assertEqual(pair['thinking']['pairs'], 1, 'only a turn present in both arms is a pair')
        self.assertEqual((pair['thinking']['tau_on'], pair['thinking']['tau_off']), (4.0, 6.0))
        self.assertEqual(pair['thinking']['mean_difference'], 2.0)

    def test_k2_projects_the_worst_of_eight_and_does_not_screen_hardware_slots(self):
        code, out, results, public, engine, data = run_lab(SimpleCase())
        summary = read_json(os.path.join(public, 'tau-lab-summary.json'))
        k2 = summary['k_inputs']['k2']
        self.assertIn('worst_of_8_projected', k2)
        self.assertEqual(sorted(k2['worst_of_8_projected']), ['high', 'low'])
        self.assertIn('bar_5_95', k2['worst_of_8_projected']['low'])
        self.assertNotIn('worst_seat', json.dumps(summary))
        for entry in summary['arms'].values():
            self.assertIn('segment_min', entry)                           # the segments are slots every turn rotates through

    def test_regions_come_from_the_lab_turns_token_id_markers(self):
        code, out, results, public, engine, data = run_lab(SimpleCase(), engine=FakeEngine(tool_every=1, rounds=14), arguments=['--arms', 'A1'])
        regions = read_json(os.path.join(public, 'tau-lab-summary.json'))['k_inputs']['regions']
        self.assertEqual(sorted(regions), ['content', 'reasoning', 'tool'])
        turns = [json.loads(line) for line in read_text(os.path.join(results, 'turns.jsonl')).splitlines()]
        self.assertTrue(all(turn['think_tokens'] == 4 and turn['tool_at'] is not None and turn['past_think'] for turn in turns))

    def test_coverage_says_how_far_each_arm_got(self):
        code, out, results, public, engine, data = run_lab(SimpleCase(), engine=FakeEngine(think_at=None, tool_every=2, rounds=14),
                                                           arguments=['--arms', 'A1 A5'])
        arms = read_json(os.path.join(public, 'tau-lab-summary.json'))['arms']
        cover = arms['A1']['coverage']
        self.assertEqual(cover['share_past_think'], 0.0, 'thinking ON, </think> never produced: every answer is a reasoning prefix')
        self.assertEqual(cover['share_finish_length'], 1.0)
        self.assertEqual(cover['max_tokens'], 2048)
        self.assertEqual(arms['A5']['coverage']['share_past_think'], 1.0, 'thinking OFF has no reasoning to leave')
        self.assertEqual((arms['A1']['thinking'], arms['A5']['thinking']), ('on', 'off'))

    def test_the_private_files_are_written_before_a_leak_is_refused(self):
        code, out, results, public, engine, data = run_lab(SimpleCase(), arguments=['--arms', 'A1'])
        os.remove(os.path.join(results, 'report.private.json'))
        os.remove(os.path.join(results, 'tapes.jsonl'))
        with mock.patch.object(rep, 'assert_public', side_effect=rep.PrivacyError('leak')):
            with self.assertRaises(rep.PrivacyError):
                rep.build_and_write(results, None, info=read_json(os.path.join(results, 'run.json')))
        self.assertTrue(os.path.exists(os.path.join(results, 'report.private.json')))
        self.assertTrue(os.path.exists(os.path.join(results, 'tapes.jsonl')))

    def test_the_a3_reference_builder_uses_the_reports_own_estimator(self):
        log = self.log([([5, 6, 7, 8], 4, None), ([9, 9, 9, 9], 4, None)])
        reference = rep.make_reference([('4k', 'va', log), ('32k', 'vb', log)])
        self.assertAlmostEqual(reference['sets']['4k'], (5 + 6 + 7 + 9 + 9 + 9) / 6.0, places=3)
        self.assertEqual(reference['rounds'], dict([('32k', 6), ('4k', 6)]))
        self.assertAlmostEqual(reference['pooled_tau'], reference['sets']['4k'], places=3)
        self.assertNotIn('thatch', json.dumps(reference))

    def test_a_run_that_stopped_or_lost_turns_is_never_complete(self):
        base = dict(arms=['A1'], counts=dict(A1=dict(planned=3, ok=3, error=0, skipped=0, resumed=0, refused=0)))
        specs = [dict(arm='A1', id='x%d' % n, status='ok', set='swe', cluster='c', request_id='cmpl-%016x' % n, prompt_tokens=9000,
                      chunk_tokens=[1], thinking=True, weight=1.0, think_tokens=0, tool_at=None, finish='length') for n in range(3)]
        text = '\n'.join(sum((packed_lines('cmpl-%016x' % n, [6] * 12, n) for n in range(3)), []))
        self.assertTrue(rep.build(None, turns=specs, log_text=text, info=base)[0]['complete'])
        for change in (dict(error='RuntimeError'), dict(tripped='breaker'),
                       dict(counts=dict(A1=dict(planned=4, ok=3, error=0, skipped=0, resumed=0, refused=0)))):
            public = rep.build(None, turns=specs, log_text=text, info=dict(base, **change))[0]
            self.assertFalse(public['complete'], change)
            self.assertEqual(public['stopped'], bool(change.get('error') or change.get('tripped')))


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
            self.assertAlmostEqual(public['arms'][arm]['tau_sample'], sum(a for a, _ in pairs) / float(sum(b for _, b in pairs)), places=3)
            self.assertEqual(public['arms'][arm]['rounds'], sum(b for _, b in pairs))
        self.assertEqual(public['arms']['A1']['turns'], 6)
        self.assertEqual(sorted(tape['arm'] for tape in tapes).count('A3'), 8)
        self.assertEqual(len(tapes), 6 + 2 + 8 + 6 + 3)

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
        self.assertEqual(public['thinking']['pairs'], 3)
        self.assertIsNotNone(public['thinking']['tau_on'])
        self.assertAlmostEqual(public['thinking']['mean_difference'], public['thinking']['tau_off'] - public['thinking']['tau_on'],
                               places=2)

    def test_the_calibration_passes_within_seven_percent_and_fails_beyond(self):
        reference = lab.read_a3_reference()
        public, _, _ = self.run_report(manifest=dict(a3_reference=reference))
        lab_tau = public['calibration']['lab_tau']
        self.assertEqual(public['calibration']['verdict'], 'PASS' if abs(lab_tau - reference['pooled_tau']) / reference['pooled_tau'] <= 0.07 else 'FAIL')
        self.assertEqual(sorted(public['calibration']['sets']), ['32k', '4k'])
        self.assertEqual(public['calibration']['thinking'], 'off')
        near = self.run_report(manifest=dict(a3_reference=dict(pooled_tau=lab_tau * 1.05)))[0]
        self.assertEqual(near['calibration']['verdict'], 'PASS')
        far = self.run_report(manifest=dict(a3_reference=dict(pooled_tau=lab_tau * 1.2)))[0]
        self.assertEqual(far['calibration']['verdict'], 'FAIL')
        self.assertEqual(far['checks']['calibration'], 'FAIL')

    def test_the_positions_check_needs_production_counters(self):
        public, _, _ = self.run_report()
        self.assertEqual(public['checks']['positions_1_3'], dict(verdict='NOT_ESTABLISHED', reason='no_production_counters'))

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

    def test_the_template_describes_the_lab_as_it_is(self):
        text = self.text()
        self.assertIn('thinking', text)
        self.assertIn('2048', text)
        self.assertIn('2400', text)


class PublicRepoTests(unittest.TestCase):
    """The Qwen repo is PUBLIC: no host, registry, board, address, digest or transcript text in anything this change adds."""

    FILES = ('c2_tau_lab.py', 'tau_lab_report.py', os.path.join('references', 'tau-lab', 'a3-reference.json'))

    def test_the_lab_code_and_the_reference_name_no_infrastructure(self):
        for name in self.FILES:
            text = read_text(os.path.join(HERE, name))
            self.assertIsNone(BANNED.search(text), (name, BANNED.search(text)))

    def test_the_new_workflow_step_adds_no_registry_or_board_name(self):
        step = WorkflowTests.step_of(read_text(WORKFLOW))
        self.assertIsNone(BANNED.search(step), BANNED.search(step))

    def test_the_banned_pattern_itself_catches_what_it_should(self):
        for text in ('image=zot.example.test:5000/tt-vllm:x', 'ssh 10.20.30.40', 'blackhole-' + 'ABCDEF0123456789', '/home/someone/x',
                     'sha256:' + '0123456789abcdef' * 2, '/dev/tenstorrent/3'):
            self.assertIsNotNone(BANNED.search(text), text)
        for text in ('host 127.0.0.1', 'tag qwen38-c2-tp4-serve-2', '<registry>/tt-vllm:qwen38-c2-<tag>'):
            self.assertIsNone(BANNED.search(text), text)

    def test_the_tests_and_the_scripts_carry_no_secret_literals(self):
        for name in ('c2_tau_lab.py', 'tau_lab_report.py', 'test_tau_lab.py'):
            text = read_text(os.path.join(HERE, name))
            self.assertIsNone(re.search(r'gh[pous]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|thatch_(?:sess|sk)_[A-Za-z0-9]{6,}', text), name)


class WorkflowTests(unittest.TestCase):
    @staticmethod
    def step_of(text):
        start = text.index('      - name: Tau lab (W-T1) on the four-card set\n')
        end = text.index('\n      - name: ', start + 10)
        return text[start:end]

    def setUp(self):
        with open(WORKFLOW, encoding='utf-8') as handle:
            self.text = handle.read()
        self.step = self.step_of(self.text)

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

    def test_the_image_is_found_among_the_local_images_so_the_step_names_no_registry(self):
        self.assertNotIn('zot.', self.step)
        self.assertIn('docker image ls', self.step)
        self.assertIn('qwen38-c2-$TAG', self.step)

    def test_the_card_set_is_resolved_in_the_current_shell_and_its_boards_never_reach_a_log(self):
        line = [row for row in self.step.splitlines() if 'card_set_nodes' in row][0]
        self.assertRegex(line.strip(), r'^card_set_nodes 60 >"[^"|]+" 2>&1$', 'a redirect (not a pipe: CARD_SET_NODES must survive)')
        self.assertIn('$taulab/', line)
        self.assertNotIn('c2-results', line)
        self.assertNotIn('/dev/null', line)

    def test_only_aggregates_reach_the_uploaded_artifact_and_the_privates_stay_under_kwork64_taulab(self):
        script = self.step
        code = chr(10).join(row for row in script.splitlines() if not row.strip().startswith('#'))
        self.assertIn('--public "$public"', script)
        self.assertIn('public="$RUNNER_TEMP/c2-results/taulab"', script)
        self.assertIn('taulab="$HOME/kwork64/taulab"', script)
        self.assertIn('run="$taulab/results/run-$GITHUB_RUN_ID"', script)
        # The uploaded directory is c2-results as a whole: the ONLY thing this step puts in it is the summary directory. The
        # driver's log and stderr, the holders and the card-set listings are rig-local.
        self.assertEqual(re.findall(r'c2-results\S*', code), ['c2-results/taulab"'])
        for private in ('taulab.log', 'taulab.stderr', 'holders-', 'card-set-'):
            rows = [row for row in code.splitlines() if private in row]
            self.assertTrue(rows, private)
            for row in rows:
                self.assertTrue('$run/' in row or '$taulab/' in row, row)
        self.assertNotIn('cat "$run', script)
        self.assertNotIn('docker logs', script)
        self.assertIn('--data "$data"', script)
        self.assertNotIn('-v "$data', script)
        self.assertNotIn('cat "$taulab', script)

    def test_the_driver_cannot_outlive_its_window_by_more_than_a_few_minutes(self):
        self.assertIn('timeout --signal=TERM --kill-after=', self.step)
        self.assertRegex(self.step, r'budget \+ \d+')
        self.assertLess(self.step.index('timeout --signal=TERM'), self.step.index('python3 scripts/ci/c2_tau_lab.py'))

    def test_the_comments_say_what_the_step_does_with_its_log(self):
        self.assertIn('taulab/tau-lab-summary.json', self.step)
        self.assertNotIn('uploads taulab.log', self.step)

    def test_the_budget_is_the_smallest_of_the_step_the_job_and_the_job_file(self):
        self.assertIn('budget=$(( budget < box ? budget : box ))', self.step)
        self.assertIn('${C2_JOB_STARTED:?}', self.step)

    def test_the_header_documents_the_action(self):
        self.assertIn('#   taulab - the W-T1 tau lab', self.text)
        self.assertNotIn('uploaded', self.text[self.text.index('#   taulab'):self.text.index('#   replay')].replace(
            'the artifact gets only taulab/', ''), 'the header says only the summary is uploaded')


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
