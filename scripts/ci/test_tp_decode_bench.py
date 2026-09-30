"""tp_decode_bench: shapes, prompts, the per-stream and steady-window arithmetic, on the CPU with injected streams."""

import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tp_decode_bench as bench  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def times(first, count, rate):
    return [first + index / float(rate) for index in range(count)]


class ShapeTests(unittest.TestCase):
    def test_parse_shape(self):
        self.assertEqual(bench.parse_shape('4x130000'), (4, 130000))
        for bad in ('4', 'x4', '0x4', '4x0', 'ax4', '4x-1'):
            with self.assertRaises(ValueError):
                bench.parse_shape(bad)

    def test_the_default_shapes_are_the_job_ladder(self):
        parsed = [bench.parse_shape(shape) for shape in bench.DEFAULT_SHAPES]
        self.assertEqual([p for p in parsed if p[0] == 1], [(1, 4096), (1, 32768), (1, 65536), (1, 130000)])
        self.assertEqual([p[1] for p in parsed if p[0] == 8], [4096, 32768, 65536])
        self.assertIn((4, 130000), parsed)

    def test_prompts_are_distinct_per_stream_and_reach_the_length(self):
        corpus = bench.read_corpus(ROOT)
        a, b = bench.prompt_for(corpus, 0, 4096), bench.prompt_for(corpus, 1, 4096)
        self.assertNotEqual(a, b)
        self.assertNotEqual(a[200:1200], b[200:1200], 'no shared prefix past the instruction')
        self.assertGreaterEqual(len(bench.prompt_for(corpus, 0, 130000)), int(130000 * bench.CHARS_PER_TOKEN))
        self.assertEqual(len(bench.prompt_for(corpus, 3, 1000)), len(bench.prompt_for(corpus, 2, 1000)))


class ArithmeticTests(unittest.TestCase):
    def test_a_streams_decode_rate_is_after_the_first_token(self):
        record = bench.summarise_stream(100.0, times(102.0, 101, 20), dict(completion_tokens=101, prompt_tokens=4000),
                                        'length')
        self.assertEqual(record['ttft_s'], 2.0)
        self.assertAlmostEqual(record['decode_tok_s'], 20.0, places=2)
        self.assertEqual(record['prompt_tokens'], 4000)

    def test_chunks_carrying_several_tokens_scale_by_the_server_count(self):
        chunks = times(0.0, 51, 10)                   # 51 chunks over 5 s, 102 tokens in all: 2 per chunk
        record = bench.summarise_stream(0.0, chunks, dict(completion_tokens=102), 'length')
        self.assertAlmostEqual(record['decode_tok_s'], 20.0, places=2)

    def test_steady_rate_uses_only_the_window_every_stream_decodes(self):
        # stream 0 decodes 0..10 s at 20/s, stream 1 starts at 5 s: the shared window is 5..10 s
        first, second = times(0.0, 201, 20), times(5.0, 101, 20)
        result = bench.steady([first, second], [201, 101])
        self.assertAlmostEqual(result['window_s'], 5.0, places=2)
        self.assertAlmostEqual(result['per_user_tok_s'], 20.0, places=1)
        self.assertAlmostEqual(result['aggregate_tok_s'], 40.0, places=1)

    def test_no_overlap_or_one_chunk_has_no_steady_rate(self):
        self.assertIsNone(bench.steady([times(0.0, 10, 10), times(50.0, 10, 10)], [10, 10]))
        self.assertIsNone(bench.steady([[1.0], times(0.0, 10, 10)], [1, 10]))


class RunTests(unittest.TestCase):
    def fake(self, rate=25.0, tokens=256, fail_on=None):
        def stream(base, model, message, max_tokens, timeout=0):
            if fail_on and len(message) > fail_on:
                raise urllib.error.HTTPError(base, 400, 'too long', None, io.BytesIO(b'context'))
            return times(1000.0, tokens, rate), dict(completion_tokens=tokens, prompt_tokens=len(message) // 4), 'length'

        return stream

    def test_a_shape_reports_median_steady_and_ttft(self):
        corpus = bench.read_corpus(ROOT)
        shape = bench.run_shape('http://x', 'm', corpus, 4, 4096, 256, self.fake())
        self.assertEqual(shape['name'], '4x4096')
        self.assertEqual(len(shape['streams_detail']), 4)
        self.assertAlmostEqual(shape['median_decode_tok_s'], 25.0, places=1)
        self.assertAlmostEqual(shape['steady']['per_user_tok_s'], 25.0, places=1)
        self.assertAlmostEqual(shape['steady']['aggregate_tok_s'], 100.0, places=0)

    def test_a_refused_shape_is_recorded_and_the_run_goes_on(self):
        with tempfile.TemporaryDirectory() as directory:
            out = os.path.join(directory, 'bench.json')
            status = bench.main(['http://x', 'm', out, '--root', ROOT, '--shapes', '1x4096,1x130000'],
                                stream_fn=self.fake(fail_on=40000), log=lambda line: None)
            with open(out) as handle:
                written = json.load(handle)
        self.assertEqual(status, 1)
        self.assertNotIn('error', written['shapes'][0])
        self.assertIn('HTTP 400', written['shapes'][1]['error'])

    def test_the_smoke_runs_the_bench_only_when_named(self):
        with open(os.path.join(HERE, 'c2_serving_smoke.py'), encoding='utf-8') as handle:
            text = handle.read()
        self.assertIn("if ONLY and 'bench' in ONLY:", text)


if __name__ == '__main__':
    unittest.main()
