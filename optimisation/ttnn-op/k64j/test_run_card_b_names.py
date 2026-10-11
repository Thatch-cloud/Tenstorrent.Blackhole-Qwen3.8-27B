"""CPU checks for the by-name options of run_card_b.sh: KOPGRAFT64_NAME (the graft by directory name under $HOME) and IMAGE_C2_TAG (the image by C2 tag).

The RQ jobs that re-record the packed-any evidence on the K64j-OQ binary (scripts/ci/references/fusion-jobs/WPP) are pushed in public tag commits, and C2_CARDM_ENV
values are plain words: a graft PATH (a host path) and an image REFERENCE (a registry) cannot be in them. The runner takes a directory name under $HOME and a C2 tag
instead. These tests run the option block of the real script in bash against a scratch $HOME and a stub `docker`: no device, no ttnn, no rig.

    py -3.11 -B -m unittest test_run_card_b_names      (from this directory; needs bash)
"""

import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
RUNNER = HERE / 'run_card_b.sh'
NL = chr(10)
DEFAULT_IMAGE = 'sha256:57cb699489436842d7e7bdd5ab917d509b4492fc95f6f083ef7d2865b488c2ef'


def bash():
    for candidate in (shutil.which('bash'), 'C:/Program Files/Git/bin/bash.exe', '/bin/bash', '/usr/bin/bash'):
        if candidate and Path(candidate).exists():
            return candidate
    return None


BASH = bash()


def option_block():
    """The text from the by-name block through the two assignments that follow it (G and IMAGE), as the runner has it."""
    text = RUNNER.read_text(encoding='utf-8')
    start = text.index('# >>> graft and image by name')
    end = text.index('IMAGE=${IMAGE:-', start)
    end = text.index(NL, end) + 1
    return text[start:end]


@unittest.skipIf(BASH is None, 'needs bash')
class ByNameTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='run-card-b-names-'))
        self.addCleanup(shutil.rmtree, str(self.tmp), True)
        (self.tmp / 'home').mkdir()
        (self.tmp / 'bin').mkdir()
        self.images = []
        self.docker_fails = False

    def stub_docker(self):
        listing = NL.join(self.images)
        script = ['#!/bin/sh', 'if [ "$1" = image ] && [ "$2" = ls ]; then']
        script += ['  exit 3'] if self.docker_fails else ["  cat <<'EOF'", listing, 'EOF']
        script += ['fi', 'exit 0', '']
        path = self.tmp / 'bin' / 'docker'
        path.write_text(NL.join(script), encoding='utf-8', newline=NL)
        path.chmod(0o755)

    def run_block(self, **environ):
        """Run the runner's by-name block under `set -euo pipefail` with HOME and a stub docker; -> (returncode, G, IMAGE, stderr)."""
        self.stub_docker()
        program = 'set -euo pipefail' + NL + option_block() + 'printf "%s\\n%s\\n" "$G" "$IMAGE"' + NL
        env = {'PATH': '%s:/usr/bin:/bin' % (self.tmp / 'bin'), 'HOME': str(self.tmp / 'home')}
        env.update(environ)
        done = subprocess.run([BASH, '-c', program], capture_output=True, text=True, env=env)
        lines = done.stdout.splitlines()
        return done.returncode, (lines[0] if len(lines) > 0 else None), (lines[1] if len(lines) > 1 else None), done.stderr

    # ---- the graft by name ----
    def test_the_default_is_the_k64j_graft_and_the_pinned_default_image_whatever_else_is_set_nowhere(self):
        code, graft, image, _err = self.run_block()
        self.assertEqual((code, graft, image), (0, str(self.tmp / 'home' / 'opgraft-K64j'), DEFAULT_IMAGE))

    def test_a_name_resolves_under_home(self):
        code, graft, _image, _err = self.run_block(KOPGRAFT64_NAME='opgraft-K64j-OQ')
        self.assertEqual((code, graft), (0, str(self.tmp / 'home' / 'opgraft-K64j-OQ')))
        for name in ('opgraft-K64j', 'a.b_c-d', 'x1'):
            self.assertEqual(self.run_block(KOPGRAFT64_NAME=name)[1], str(self.tmp / 'home' / name), name)

    def test_the_full_path_still_wins_when_it_is_set(self):
        code, graft, _image, _err = self.run_block(KOPGRAFT64='/somewhere/else', KOPGRAFT64_NAME='opgraft-K64j-OQ')
        self.assertEqual((code, graft), (0, '/somewhere/else'))
        code, graft, _image, _err = self.run_block(KOPGRAFT64='/somewhere/else', KOPGRAFT64_NAME='../not-even-looked-at')
        self.assertEqual((code, graft), (0, '/somewhere/else'), 'a set KOPGRAFT64 means the name is not read')

    def test_an_empty_name_is_not_a_name(self):
        code, graft, _image, _err = self.run_block(KOPGRAFT64_NAME='')
        self.assertEqual((code, graft), (0, str(self.tmp / 'home' / 'opgraft-K64j')))

    def test_a_name_that_is_not_a_plain_directory_name_is_refused_before_anything_is_read(self):
        for name in ('a/b', '/abs', '..', '.', '../x', 'x/..', 'x..y', '.hidden', '-flag', 'a b', 'a;b', '$(id)', 'a`b`', 'a*', 'a\\b', 'a:b', 'opgraft-K64j-OQ/'):
            with self.subTest(name=name):
                code, graft, image, err = self.run_block(KOPGRAFT64_NAME=name)
                self.assertEqual(code, 1)
                self.assertIsNone(graft)
                self.assertIn('refusing: KOPGRAFT64_NAME=', err)
                self.assertIn("no '/', no '..'", err)

    # ---- the image by tag ----
    def test_a_tag_resolves_to_the_local_image_whatever_registry_prefix_it_carries(self):
        self.images = ['reg.example:5000/tt-vllm:qwen38-c2-tp4-oneq-1', 'reg.example:5000/tt-vllm:qwen38-c2-tp4-oneq-10', 'other/tt-vllm:qwen38-c2-tp4-fusion-1']
        code, _graft, image, _err = self.run_block(IMAGE_C2_TAG='tp4-oneq-1')
        self.assertEqual((code, image), (0, 'reg.example:5000/tt-vllm:qwen38-c2-tp4-oneq-1'))
        self.assertEqual(self.run_block(IMAGE_C2_TAG='tp4-fusion-1')[2], 'other/tt-vllm:qwen38-c2-tp4-fusion-1')
        self.assertEqual(self.run_block(IMAGE_C2_TAG='tp4-oneq-10')[2], 'reg.example:5000/tt-vllm:qwen38-c2-tp4-oneq-10')

    def test_a_tag_is_the_whole_tag_not_a_prefix_or_a_regex(self):
        self.images = ['reg.example:5000/tt-vllm:qwen38-c2-tp4-oneq-10', 'reg.example:5000/tt-vllm:qwen38-c2-tp4xoneq-1', 'reg.example:5000/not-tt-vllm:qwen38-c2-tp4-oneq-1']
        for tag in ('tp4-oneq-1', 'tp4.oneq-1', 'tp4-oneq'):
            code, _graft, image, err = self.run_block(IMAGE_C2_TAG=tag)
            with self.subTest(tag=tag):
                self.assertEqual(code, 1)
                self.assertIsNone(image)
                self.assertIn('no local image tt-vllm:qwen38-c2-%s on this host' % tag, err)

    def test_the_first_match_wins_and_image_wins_when_set(self):
        self.images = ['a/tt-vllm:qwen38-c2-t', 'b/tt-vllm:qwen38-c2-t']
        self.assertEqual(self.run_block(IMAGE_C2_TAG='t')[2], 'a/tt-vllm:qwen38-c2-t')
        code, _graft, image, _err = self.run_block(IMAGE='sha256:abc', IMAGE_C2_TAG='t')
        self.assertEqual((code, image), (0, 'sha256:abc'))

    def test_a_tag_that_is_not_a_plain_tag_is_refused(self):
        self.images = ['reg/tt-vllm:qwen38-c2-x']
        for tag in ('a/b', 'a b', 'a;b', '$(id)', '.x', '-x', 'a*', 'a|b'):
            with self.subTest(tag=tag):
                code, _graft, _image, err = self.run_block(IMAGE_C2_TAG=tag)
                self.assertEqual(code, 1)
                self.assertIn('refusing: IMAGE_C2_TAG=', err)

    def test_a_docker_that_fails_is_a_refusal_with_a_message_not_a_silent_exit(self):
        self.docker_fails = True
        code, _graft, _image, err = self.run_block(IMAGE_C2_TAG='tp4-oneq-1')
        self.assertEqual(code, 1)
        self.assertIn('no local image tt-vllm:qwen38-c2-tp4-oneq-1 on this host', err)

    def test_the_two_options_are_independent(self):
        self.images = ['reg/tt-vllm:qwen38-c2-tp4-oneq-1']
        code, graft, image, _err = self.run_block(KOPGRAFT64_NAME='opgraft-K64j-OQ', IMAGE_C2_TAG='tp4-oneq-1')
        self.assertEqual((code, graft, image), (0, str(self.tmp / 'home' / 'opgraft-K64j-OQ'), 'reg/tt-vllm:qwen38-c2-tp4-oneq-1'))


class RunnerTextTests(unittest.TestCase):
    def test_the_block_sits_before_the_canonical_qual_card_block_and_the_g_assignment_and_the_file_stays_lf(self):
        text = RUNNER.read_text(encoding='utf-8')
        self.assertNotIn('\r', text)
        block = text.index('# >>> graft and image by name')
        self.assertLess(block, text.index('G=${KOPGRAFT64:-$HOME/opgraft-K64j}'))
        self.assertLess(text.index('# <<< graft and image by name'), text.index('G=${KOPGRAFT64:-$HOME/opgraft-K64j}'))
        self.assertLess(block, text.index('# >>> qual_card.sh'))
        self.assertEqual(text.count('# >>> graft and image by name'), 1)

    def test_the_header_documents_both_options_and_the_default_is_unchanged(self):
        text = RUNNER.read_text(encoding='utf-8')
        self.assertIn('KOPGRAFT64_NAME (a directory name under $HOME)', text)
        self.assertIn('IMAGE_C2_TAG', text)
        self.assertIn('G=${KOPGRAFT64:-$HOME/opgraft-K64j}', text)
        self.assertIn('IMAGE=${IMAGE:-%s}' % DEFAULT_IMAGE, text)

    @unittest.skipIf(BASH is None, 'needs bash')
    def test_the_whole_runner_is_valid_bash(self):
        self.assertEqual(subprocess.run([BASH, '-n', str(RUNNER)], capture_output=True).returncode, 0)


if __name__ == '__main__':
    unittest.main()
