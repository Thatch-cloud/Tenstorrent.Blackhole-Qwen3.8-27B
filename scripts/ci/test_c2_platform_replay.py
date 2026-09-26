"""c2_platform_replay: the agent's container copy and the G7 traffic steps, held on CPU (no docker, no HTTP)."""

import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from contextlib import ExitStack, redirect_stdout
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


TT = 'Qwen/Qwen3.8-27B:tt'
PLAIN = 'Qwen/Qwen3.8-27B'

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

    def run_steps(self, stream_result=None, ask_results=None, contract=True):
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
            ok = replay.traffic_steps(8011, 'm', record, 3, stream=stream, ask=ask, sleep=lambda seconds: None,
                                      contract=contract)
        return ok, streamed, asked, steps

    REFUSED = dict(ok=False, status=400, body='{"error": {"message": "n must be 1 on this model"}}')

    def test_the_g7_steps_run_in_order_and_pass(self):
        ok, streamed, asked, steps = self.run_steps(ask_results=dict(n2=self.REFUSED))
        self.assertTrue(ok)
        self.assertEqual([name for name, _ in steps], ['long_prompt', 'streamed_long_answer', 'arrivals4',
                                                       'edge_refusal_n2', 'alive_after_refusal', 'max_tokens_1',
                                                       'alive_after_max_tokens_1'])
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
        ok, _, _, steps = self.run_steps(ask_results=dict(n2=self.REFUSED, one=dict(ok=True, status=200,
                                                                                     completion_tokens=2)))
        self.assertFalse(ok, 'max_tokens=1 must return exactly one token')
        ok, _, _, steps = self.run_steps(ask_results=dict(n2=dict(ok=False, status=500)))
        self.assertFalse(ok, 'a refusal is a 400 (contract) or a 200 (general), never a server error')

    def test_n2_is_refused_at_the_edge_with_the_contracts_words(self):
        """Review finding 10: under a contract profile n=2 never reaches the engine; any other 400 (or a
        200) there is not the contract's refusal."""
        ok, _, _, steps = self.run_steps(ask_results=dict(n2=self.REFUSED))
        self.assertTrue(dict(steps)['edge_refusal_n2']['ok'])
        ok, _, _, steps = self.run_steps(ask_results=dict(n2=dict(ok=False, status=400, body='bad request')))
        self.assertFalse(ok)
        ok, _, _, steps = self.run_steps(ask_results=dict(n2=dict(ok=True, status=200)))
        self.assertFalse(ok, 'a contract profile that served n=2 has no contract')
        ok, _, _, steps = self.run_steps(ask_results=dict(n2=dict(ok=True, status=200)), contract=False)
        self.assertTrue(ok, 'general has no contract: the stock path answers for itself')

    def test_the_long_answer_must_be_thousands_or_say_the_ceiling_stopped_it(self):
        """Review finding 10: a 50-token EOS answer passed 'a streamed answer of thousands of tokens'."""
        ok, _, _, steps = self.run_steps(stream_result=dict(ok=True, finish='stop', completion_tokens=50),
                                         ask_results=dict(n2=self.REFUSED))
        self.assertFalse(ok)
        self.assertIn('short of 1000', dict(steps)['streamed_long_answer']['note'])
        ok, _, _, steps = self.run_steps(stream_result=dict(ok=True, finish='length', completion_tokens=256),
                                         ask_results=dict(n2=self.REFUSED))
        self.assertTrue(ok)
        self.assertIs(dict(steps)['streamed_long_answer']['exercised'], False)
        self.assertIn('not exercised', dict(steps)['streamed_long_answer']['note'])


class SourceTests(unittest.TestCase):
    """Review finding 8: the recorded Env copied whole onto another image served that image with the old
    image's kernel cache key and defaults, and the replay still passed."""

    def test_a_whole_env_is_never_copied_onto_another_image(self):
        self.assertIsNone(replay.source_problem('zot/old@sha256:a', 'zot/new@sha256:b', ['PATH=/x']),
                          'the source image\'s ENV is known: subtracted')
        self.assertIsNone(replay.source_problem('zot/old@sha256:a', 'zot/old@sha256:a', None),
                          'the same image: the copy is exact')
        problem = replay.source_problem('zot/old@sha256:a', 'zot/new@sha256:b', None)
        self.assertIn('Pull zot/old@sha256:a first', problem)
        self.assertIn('source names no image', replay.source_problem(None, 'zot/new@sha256:b', None))

    def test_the_replay_refuses_before_starting_anything(self):
        commands = []

        def run(command, timeout=None, check=True):
            commands.append(command)
            if command[:3] == ['docker', 'image', 'inspect']:
                return mock.Mock(returncode=1, stdout='', stderr='No such image')
            return mock.Mock(returncode=0, stdout='', stderr='')

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(replay, 'run', side_effect=run), \
                mock.patch.object(sys, 'argv', ['replay', '--source', AGENT, '--image', 'zot/new@sha256:b',
                                                '--results', directory]), \
                redirect_stdout(io.StringIO()) as out:
            code = replay.main()
            with open(os.path.join(directory, 'platform-replay.json'), encoding='utf-8') as handle:
                recorded = json.load(handle)
        self.assertEqual(code, 1)
        self.assertFalse(any(command[:2] == ['docker', 'run'] for command in commands))
        self.assertFalse(recorded['steps']['source']['ok'])
        self.assertIn('refused', out.getvalue())


class ServedNameTests(unittest.TestCase):
    """TT advertises ONLY Qwen/Qwen3.8-27B:tt (the thin layer's THATCH_SERVING_MODEL_ALIAS): the spine
    routes on the advertised names by exact match, so plain Qwen/Qwen3.8-27B traffic stays on the Spark."""

    def served(self, status, body, expected=TT, checkpoint=PLAIN):
        with mock.patch.object(replay, 'http', return_value=(status, body)) as http:
            result = replay.served_name(8011, checkpoint, expected)
        http.assert_called_once_with(8011, '/v1/models', timeout=30)
        return result

    def test_the_default_is_the_tagged_name_and_the_checkpoint_is_loaded(self):
        self.assertEqual((replay.SERVED_MODEL, replay.CHECKPOINT), (TT, PLAIN))

    def test_the_tagged_name_alone_is_advertised(self):
        result = self.served(200, dict(status='ok', model=TT, aliases=[TT], healthy=True))
        self.assertTrue(result['ok'])
        self.assertEqual((result['served'], result['advertised']), (TT, [TT, TT]))

    def test_an_image_without_the_alias_would_take_plain_traffic(self):
        result = self.served(200, dict(status='ok', model=PLAIN))
        self.assertFalse(result['ok'])
        self.assertEqual(result['served'], PLAIN)

    def test_the_plain_id_advertised_beside_the_tag_fails(self):
        self.assertFalse(self.served(200, dict(model=TT, aliases=[TT, PLAIN]))['ok'])

    def test_a_refused_or_unreachable_models_endpoint_fails(self):
        self.assertFalse(self.served(404, 'not found')['ok'])
        self.assertFalse(self.served(404, dict(model=TT, aliases=[TT]))['ok'])
        self.assertFalse(self.served(None, 'URLError(connection refused)')['ok'])

    def test_the_legacy_replay_of_an_image_without_the_alias(self):
        self.assertTrue(self.served(200, dict(model=PLAIN), expected=PLAIN)['ok'])
        self.assertFalse(self.served(200, dict(model=TT, aliases=[TT]), expected=PLAIN)['ok'])


class ReplayNameTests(unittest.TestCase):
    """main(): serving.manage loads the checkpoint, the agent's warmup names the checkpoint (answered,
    not advertised), and every later request names the served model."""

    def drive(self, models, *arguments):
        commands, chats, traffic = [], [], []

        def run(command, timeout=None, check=True):
            commands.append(command)
            if command[:3] == ['docker', 'image', 'inspect']:
                return mock.Mock(returncode=0, stdout=json.dumps([dict(Config=dict(Env=[]))]), stderr='')
            return mock.Mock(returncode=0, stdout='true', stderr='')

        def http(port, path, body=None, timeout=60):
            return 200, dict(models)

        def chat(port, model, content, max_tokens, timeout=900, **extra):
            chats.append((model, content))
            return dict(ok=True, status=200, completion_tokens=max_tokens, tool_calls=[dict(id='call-0')])

        def traffic_steps(port, model, record, seed, **options):
            traffic.append(model)
            return True

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            for name, fake in (('run', run), ('http', http), ('chat', chat), ('traffic_steps', traffic_steps)):
                stack.enter_context(mock.patch.object(replay, name, side_effect=fake))
            stack.enter_context(mock.patch.object(replay.subprocess, 'run'))
            stack.enter_context(mock.patch.object(sys, 'argv', ['replay', '--source', AGENT, '--image',
                                                                'zot/new@sha256:b', '--results', directory,
                                                                '--seed', '5'] + list(arguments)))
            stack.enter_context(redirect_stdout(io.StringIO()))
            code = replay.main()
            with open(os.path.join(directory, 'platform-replay.json'), encoding='utf-8') as handle:
                steps = json.load(handle)['steps']
        loads = [command[command.index('--model') + 1] for command in commands if 'serving.manage' in command]
        return code, steps, loads, chats, traffic

    def test_the_warmup_names_the_checkpoint_and_every_later_request_the_tag(self):
        code, steps, loads, chats, traffic = self.drive(dict(model=TT, aliases=[TT]))
        self.assertEqual(code, 0)
        self.assertEqual(loads, [PLAIN, PLAIN, PLAIN], 'load, reload and the load after the restart')
        self.assertEqual(chats[0], (PLAIN, 'warmup'))
        self.assertEqual(chats[1][0], TT)
        self.assertIn('merges two sorted lists', chats[1][1])
        self.assertEqual(len(chats), 9)
        self.assertEqual({model for model, _ in chats[1:]}, {TT})
        self.assertEqual(traffic, [TT])
        self.assertEqual(list(steps), ['seed', 'source', 'start', 'load', 'served_name', 'warmup', 'coding',
                                       'concurrent4', 'tool_call', 'reload', 'after_reload', 'restart',
                                       'load_after_restart', 'served_name_after_restart', 'after_restart'])
        self.assertTrue(steps['served_name']['ok'] and steps['served_name_after_restart']['ok'])

    def test_an_image_that_advertises_the_plain_id_fails_the_replay(self):
        for models in (dict(model=PLAIN), dict(model=TT, aliases=[TT, PLAIN])):
            code, steps, _, _, _ = self.drive(models)
            self.assertEqual(code, 1)
            self.assertFalse(steps['served_name']['ok'])
            self.assertFalse(steps['served_name_after_restart']['ok'])

    def test_an_image_without_the_alias_replays_under_its_plain_name(self):
        code, steps, loads, chats, traffic = self.drive(dict(model=PLAIN), '--served-model', PLAIN)
        self.assertEqual(code, 0)
        self.assertEqual({model for model, _ in chats} | set(traffic) | set(loads), {PLAIN})


if __name__ == '__main__':
    unittest.main()
