"""c2_platform_replay: the agent's container copy and the G7 traffic steps, held on CPU (no docker, no HTTP)."""

import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_platform_replay as replay  # noqa: E402

AGENT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'references', 'c2-serving',
                     'agent-container-36104200953.json')

INSPECT = dict(
    Config=dict(Env=['QWEN_C2_PROFILE=general', 'THATCH_VLLM_KWARGS={}', 'HF_HOME=/models']),
    HostConfig=dict(ReadonlyRootfs=True, Tmpfs={'/tmp': 'rw,size=512m', '/root/.cache/ttnn': ''},
                    Binds=['/home/thatch/hf-cache/hub:/models'], ShmSize=4294967296, Memory=85899345920,
                    NanoCpus=8000000000, CapAdd=['SYS_NICE'],
                    Devices=[dict(PathOnHost='/dev/tenstorrent/0'), dict(PathOnHost='/dev/hugepages-1G')]),
    Mounts=[dict(Type='bind', Source='/home/thatch/hf-cache/hub', Destination='/models', RW=True),
            dict(Type='bind', Source='/dev/hugepages-1G', Destination='/dev/hugepages-1G', RW=False)])


class Response(object):
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


def content(text, finish=None):
    return dict(choices=[dict(index=0, delta=dict(content=text), finish_reason=finish)])


STREAM = sse(dict(choices=[dict(index=0, delta=dict(role='assistant', content=''))]), content('Hel'), content('lo'),
             content('', 'stop'), dict(choices=[], usage=dict(prompt_tokens=12, completion_tokens=41)))


class ArgumentTests(unittest.TestCase):
    def test_the_copy_keeps_the_agents_shape_with_new_cards_name_and_port(self):
        arguments = replay.run_arguments(INSPECT, 'zot/thatch-serving-tt@sha256:x', 'copy', 8011,
                                         devices=['/dev/tenstorrent/3', '/dev/tenstorrent/1'])
        self.assertEqual(arguments[:6], ['docker', 'run', '-d', '--name', 'copy', '-p'])
        self.assertEqual(arguments[6], '127.0.0.1:8011:8000')
        self.assertIn('--read-only', arguments)
        self.assertEqual(arguments[-1], 'zot/thatch-serving-tt@sha256:x')
        devices = [arguments[i + 1] for i, token in enumerate(arguments) if token == '--device']
        self.assertEqual(devices, ['/dev/hugepages-1G', '/dev/tenstorrent/3', '/dev/tenstorrent/1'])
        self.assertIn('-v', arguments)
        self.assertIn('/dev/hugepages-1G:/dev/hugepages-1G:ro', arguments, 'a mount not among the binds is kept')
        self.assertEqual(arguments.count('/home/thatch/hf-cache/hub:/models'), 1)
        env = [arguments[i + 1] for i, token in enumerate(arguments) if token == '-e']
        self.assertEqual(env, INSPECT['Config']['Env'])
        self.assertEqual(arguments[arguments.index('--cpus') + 1], '8')

    def test_a_profile_replaces_the_agents(self):
        arguments = replay.run_arguments(INSPECT, 'img', 'copy', 8011, profile='c2', devices=[])
        env = [arguments[i + 1] for i, token in enumerate(arguments) if token == '-e']
        self.assertEqual(env, ['THATCH_VLLM_KWARGS={}', 'HF_HOME=/models', 'QWEN_C2_PROFILE=c2'])

    def test_a_saved_inspect_stands_in_for_a_container_that_is_gone(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in ([dict(INSPECT, Image='sha256:old')], dict(INSPECT, Image='sha256:old')):
                path = os.path.join(directory, 'inspect.json')
                with open(path, 'w', encoding='utf-8') as handle:
                    json.dump(value, handle)
                info, image = replay.inspect_source(path)
                self.assertEqual((info['Config'], image), (INSPECT['Config'], 'sha256:old'))

    def test_the_tracked_agent_record_replays_to_its_own_argv(self):
        """references/c2-serving/agent-container-36104200953.json is replay v11's docker-run.json: read
        back as a source, run_arguments rebuilds the very argv (the same cards, name, port and image)."""
        info, image = replay.inspect_source(AGENT)
        with open(AGENT, encoding='utf-8') as handle:
            recorded = json.load(handle)['argv']
        devices = [recorded[i + 1] for i, token in enumerate(recorded) if token == '--device']
        rebuilt = replay.run_arguments(info, recorded[-1], 'qwen-c2-platform', 8011, devices=devices)
        self.assertEqual(rebuilt, recorded)
        self.assertEqual(image, recorded[-1])

    def test_the_source_images_own_env_is_not_copied_over_the_new_images(self):
        image_env = ['HF_HOME=/models', 'QWEN_C2_PROFILE=general']
        arguments = replay.run_arguments(INSPECT, 'img', 'copy', 8011, devices=[], image_env=image_env)
        env = [arguments[i + 1] for i, token in enumerate(arguments) if token == '-e']
        self.assertEqual(env, ['THATCH_VLLM_KWARGS={}'])


class StreamTests(unittest.TestCase):
    def test_parse_sse_times_the_first_text_not_the_role_chunk(self):
        calls = []
        pieces, usage, finish, error = replay.parse_sse([line.encode() for line in STREAM], lambda: calls.append(1))
        self.assertEqual((pieces, finish, error, calls), (['Hel', 'lo'], 'stop', None, [1]))
        self.assertEqual(usage['completion_tokens'], 41)
        _, _, _, error = replay.parse_sse(sse(dict(error=dict(message='boom'))))
        self.assertIn('boom', error)

    def test_stream_chat_records_ttft_rate_and_finish(self):
        ticks = iter([100.0, 101.5, 105.5])
        sent = []

        def opener(request, timeout):
            sent.append(json.loads(request.data))
            return Response(STREAM)

        result = replay.stream_chat(8011, 'm', 'hello', 3000, clock=lambda: next(ticks), opener=opener)
        self.assertEqual((result['ok'], result['finish'], result['completion_tokens']), (True, 'stop', 41))
        self.assertEqual((result['ttft_s'], result['decode_tok_s'], result['wall_s']), (1.5, 10.0, 5.5))
        self.assertEqual((sent[0]['max_tokens'], sent[0]['stream']), (3000, True))
        self.assertEqual(sent[0]['stream_options'], {'include_usage': True})

    def test_a_refused_stream_is_a_failed_step_with_its_status(self):
        def opener(request, timeout):
            raise urllib.error.HTTPError('u', 400, 'Bad Request', {}, io.BytesIO(b'prompt too long'))
        result = replay.stream_chat(8011, 'm', 'x', 10, opener=opener)
        self.assertEqual((result['ok'], result['status'], result['body']), (False, 400, 'prompt too long'))


class TrafficTests(unittest.TestCase):
    def test_arrivals_are_seeded_sorted_and_inside_the_window(self):
        first = replay.arrival_offsets(7)
        self.assertEqual(first, replay.arrival_offsets(7))
        self.assertNotEqual(first, replay.arrival_offsets(8))
        self.assertEqual(first, sorted(first))
        self.assertEqual(len(first), 4)
        self.assertTrue(all(0.0 <= offset < replay.ARRIVAL_WINDOW_S for offset in first))

    def test_the_long_prompt_is_a_share_of_the_served_context(self):
        corpus = 'q' * 10 ** 6
        prompt = replay.long_prompt(131328, corpus=corpus)
        excerpt = int(131328 * replay.LONG_PROMPT_SHARE * replay.CHARS_PER_TOKEN)
        self.assertEqual(prompt.count('q'), excerpt)
        self.assertLess(excerpt / 4.0, 131328 - 16384, 'under the c2 prompt cap even at 4 characters per token')
        with tempfile.TemporaryDirectory() as directory:
            for name in ('b.py', 'a.py', 'notes.txt'):
                with open(os.path.join(directory, name), 'w', encoding='utf-8') as handle:
                    handle.write('%s = 1\n' % name[0] * 10)
            text = replay.source_corpus(10 ** 6, root=directory)
            self.assertLess(text.index('# File: a.py'), text.index('# File: b.py'))
            self.assertNotIn('notes', text)
            self.assertEqual(len(replay.source_corpus(15, root=directory)), 15)

    def run_steps(self, stream_result=None, ask_results=None):
        streamed, asked, steps = [], [], []

        def stream(port, model, content, max_tokens):
            streamed.append((content[:40], max_tokens))
            return dict(stream_result or dict(ok=True, finish='stop', completion_tokens=2500))

        answers = dict(ask_results or {})

        def ask(port, model, content, max_tokens, **extra):
            asked.append((content, max_tokens, extra))
            key = 'n2' if extra.get('n') == 2 else 'one' if max_tokens == 1 else 'alive'
            return dict(answers.get(key) or dict(ok=True, status=200, completion_tokens=max_tokens))

        def record(step, value):
            steps.append((step, value))
            return value.get('ok', True)

        with mock.patch.object(replay, 'served_context', return_value=131328):
            ok = replay.traffic_steps(8011, 'm', record, 3, stream=stream, ask=ask, sleep=lambda seconds: None)
        return ok, streamed, asked, steps

    def test_the_g7_steps_run_in_order_and_pass(self):
        ok, streamed, asked, steps = self.run_steps(ask_results=dict(n2=dict(ok=False, status=400)))
        self.assertTrue(ok)
        self.assertEqual([name for name, _ in steps], ['long_prompt', 'streamed_long_answer', 'arrivals4', 'refused_n2',
                                                       'alive_after_refusal', 'max_tokens_1', 'alive_after_max_tokens_1'])
        self.assertEqual(streamed[1][1], replay.LONG_ANSWER_TOKENS)
        self.assertTrue(dict(steps)['streamed_long_answer']['thousands'])
        self.assertEqual(len(dict(steps)['arrivals4']['users']), 4)
        self.assertEqual(dict(steps)['arrivals4']['offsets'], replay.arrival_offsets(3))
        self.assertEqual(dict(steps)['long_prompt']['served_context'], 131328)
        self.assertEqual([extra for _, _, extra in asked][0], dict(n=2))

    def test_an_engine_that_dies_after_a_refusal_or_one_token_fails_the_replay(self):
        ok, _, _, steps = self.run_steps(ask_results=dict(alive=dict(ok=False, status=None, body='connection refused')))
        self.assertFalse(ok)
        self.assertFalse(dict(steps)['alive_after_refusal']['ok'])
        ok, _, _, steps = self.run_steps(ask_results=dict(one=dict(ok=True, status=200, completion_tokens=2)))
        self.assertFalse(ok, 'max_tokens=1 must return exactly one token')
        ok, _, _, steps = self.run_steps(ask_results=dict(n2=dict(ok=False, status=500)))
        self.assertFalse(ok, 'a refusal is a 400 (contract) or a 200 (general), never a server error')


if __name__ == '__main__':
    unittest.main()
