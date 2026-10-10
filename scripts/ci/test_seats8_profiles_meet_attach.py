"""The two eight-seat lanes meet: lane 2's profiles set QWEN_FAST_M3_BLOCKS=2 and lane 1's attach reads it.

Each test composes the environment the container attach sees (the C2 image's ENV, then the profile's env over it) and
drives lane 1's checks with it: serving_runtime.m3_blocks_for / m3_shape, packed_any_admission.check_environment, and
the attach itself on the fakes test_serving_runtime uses. The four eight-seat profiles are ADMITTED and build two M3
blocks; the production profile c2-packed-tp4 builds exactly one. No device is needed."""

import json
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import packed_any_admission  # noqa: E402
import serving_runtime  # noqa: E402

ROOT = HERE.parent.parent
PROFILES = HERE / 'qwen_c2_profiles.json'
DOCKERFILE = ROOT / 'docker' / 'qwen-c2-serving.Dockerfile'

PRODUCTION = 'c2-packed-tp4'
EIGHT = ('c2-packed-tp4-8', 'c2-packed-tp4-8-gate', 'c2-packed-tp4-8-time-gate', 'c2-packed-tp4-8-diag-strace',
         'c2-packed-tp4-8-diag-strace-nowarm', 'c2-packed-tp4-8-diag-strace-rshard')
NOWARM = 'c2-packed-tp4-8-diag-strace-nowarm'


def profiles():
    return json.loads(PROFILES.read_text(encoding='utf-8'))['profiles']


def image_env():
    """The C2 image's own ENV (the first ENV of the Dockerfile's final stage), as a dict."""
    text = DOCKERFILE.read_text(encoding='utf-8')
    block = text[text.index('ENV QWEN_ATTN_PREP=1'):]
    lines = []
    for line in block.splitlines():
        lines.append(line.rstrip(chr(92)).strip())
        if not line.rstrip().endswith(chr(92)):
            break
    # a ${NAME} in the ENV is the Dockerfile's build arg (QWEN_FAST_RUNTIME_BINARY_SHA256=${GRAFT_SHA}): read with the arg's DEFAULT, what a build with no --build-arg bakes
    defaults = dict(re.findall(r'(?m)^ARG (\w+)=(\S+)\s*$', text))
    return {name: re.sub(r'\$\{(\w+)\}', lambda match: defaults.get(match.group(1), match.group(0)), value)
            for name, value in re.findall(r'(QWEN_[A-Z0-9_]+)=(\S+)', ' '.join(lines))}


def container_env(name):
    return dict(image_env(), **profiles()[name]['env'])


def attach_env(name):
    """What the profile adds or changes over the production profile (everything the production profile carries is
    the model's real configuration - contexts, TP, kernel keys, source pins - which the attach fakes stand in for). For
    production itself that is nothing: the attach on the fakes IS the production attach."""
    base = profiles()[PRODUCTION]['env']
    return {key: value for key, value in profiles()[name]['env'].items() if base.get(key) != value}


def seats(name):
    return int(profiles()[name]['engine']['max-num-seqs'])


class ProfilesAreAdmittedByTheAttachChecks(unittest.TestCase):
    def test_each_eight_seat_profile_is_two_blocks_at_eight_requests(self):
        for name in EIGHT:
            environ = container_env(name)
            with self.subTest(profile=name):
                self.assertEqual(seats(name), 8)
                self.assertEqual(serving_runtime.m3_blocks(environ), 2)
                self.assertEqual(serving_runtime.m3_blocks_for(dict(scheduler_requests=seats(name)), environ), 2)
                met, shape = serving_runtime.m3_shape(dict(scheduler_requests=seats(name)), environ)
                self.assertTrue(met, shape)
                self.assertIn('M3_BLOCKS=2', shape)

    def test_each_eight_seat_profile_passes_the_extent_admission_environment_over_the_image_env(self):
        for name in EIGHT:
            environ = container_env(name)
            m3 = serving_runtime.m3_shape(dict(scheduler_requests=seats(name)), environ)
            with self.subTest(profile=name):
                self.assertEqual(packed_any_admission.check_environment(environ, m3), [])
                self.assertEqual(packed_any_admission.m3_blocks(environ), 2)

    def test_the_same_environment_at_four_requests_is_refused_by_name(self):
        # the flag cannot ride a four-seat engine config: lane 1 refuses it rather than build two blocks for four users
        for name in EIGHT:
            with self.subTest(profile=name), self.assertRaisesRegex(ValueError, 'QWEN_FAST_M3_BLOCKS=2 builds two'):
                serving_runtime.m3_blocks_for(dict(scheduler_requests=4), container_env(name))

    def test_production_is_one_block_at_four_requests(self):
        environ = container_env(PRODUCTION)
        self.assertEqual(seats(PRODUCTION), 4)
        self.assertNotIn('QWEN_FAST_M3_BLOCKS', profiles()[PRODUCTION]['env'])
        self.assertNotIn('QWEN_FAST_M3_BLOCKS', image_env())
        self.assertEqual(serving_runtime.m3_blocks(environ), 1)
        self.assertEqual(serving_runtime.m3_blocks_for(dict(scheduler_requests=4), environ), 1)
        met, shape = serving_runtime.m3_shape(dict(scheduler_requests=4), environ)
        self.assertTrue(met, shape)
        self.assertNotIn('M3_BLOCKS', shape)
        self.assertEqual(packed_any_admission.check_environment(environ, (met, shape)), [])


class AttachOnFakes(unittest.TestCase):
    """lane 1's attach, driven over each profile's own env (the image ENV also names the runtime binary, which the fakes do not have)."""

    def harness(self):
        import test_serving_runtime

        return test_serving_runtime.RuntimeAttachmentTests('exercise')

    def test_each_eight_seat_profile_attaches_two_deferred_blocks_over_disjoint_slots(self):
        for name in EIGHT:
            harness = self.harness()
            with self.subTest(profile=name):
                # the request-width warm is device work (its own tests drive it on fakes); here it is only counted
                with patch('request_width_warm.warm_request_widths') as warm:
                    harness.exercise(packed=True, users=8, four_as_two=False, admission={}, m3_blocks=2,
                                     extra_env=attach_env(name))
                self.assertEqual(warm.call_count, 0 if name == NOWARM else 1,
                                 'QWEN_FAST_M3_REQUEST_WARM=1 runs the warm once at attach; =0 never')
                self.assertEqual([call.kwargs['pool_slots'] for call in harness.engine_calls],
                                 [(0, 1, 2, 3), (4, 5, 6, 7)])
                self.assertTrue(all(call.kwargs['defer_capture'] for call in harness.engine_calls))
                self.assertEqual(harness.pool_options['packed_replicas'], {(4, 16): 2})

    def test_production_attaches_exactly_one_block_built_in_one_phase(self):
        harness = self.harness()
        harness.exercise(packed=True, users=4, four_as_two=False, admission={}, extra_env=attach_env(PRODUCTION))
        self.assertEqual(len(harness.engine_calls), 1)
        self.assertNotIn('pool_slots', harness.engine_calls[0].kwargs)      # the one block binds the whole pool, as today
        self.assertFalse(harness.engine_calls[0].kwargs.get('defer_capture', False))
        self.assertNotIn('packed_replicas', harness.pool_options)


if __name__ == '__main__':
    unittest.main()
