"""Tests for sanitise_scan.py: every structural BLOCK category fires on a sample, the private deny file works, clean
text passes, and the three modes (directory, git diff range, commit tree) agree on what is a leak.

Every sample here is SYNTHETIC: documentation addresses, zeroed identifiers, invented names. None names a real host,
card, person, run or image. The samples are assembled from pieces only so that this file does not trip the scanner
it tests (the scanner scans itself and this file like any other)."""
import contextlib
import io
import os
import shutil
import subprocess
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import sanitise_scan as scan

HERE = Path(__file__).resolve().parent


def j(*parts):
    return ''.join(parts)


SAMPLES = {
    'registry-host': j('FROM registry.exam', 'ple.lo', 'cal:5000/image'),
    'private-ipv4': j('ssh host 192', '.168.', '0.1'),
    'card-serial': j('/dev/tenstorrent/by-id/black', 'hole-', '0123456789ABCDEF'),
    'home-path': j('cp -r /ho', 'me/alice/models .'),
    'credential': j('ssh-tool -batch -', 'pw example-secret user@host'),
    'image-digest': j('FROM base@sha', '256:', '0123456789abcdef'),
    'run-id': j('see run ', '30000000000 for the log'),
}
DENY_RULES = '\n'.join([
    '# synthetic private rules for the tests',
    j('BLOCK project-name Acme', 'Secret'),
    j('WARN codename blue', 'falcon'),
    '',
])
CLEAN = [
    'FROM localhost:5000/tt-bringup:v0.77.0-rc1',
    'ARG BASE=${REGISTRY}/tt-vllm:qwen38-k',
    'cp -a ops/attn_prep "$root/ttnn/cpp/ttnn/operations/transformer/attn_prep"',
    '# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.',
    'git clone https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B',
    'docker run --device /dev/tenstorrent/0 -v $HF_HUB_CACHE:/models',
    'serial: blackhole-<id>',
    '/home/user/example and /home/ubuntu/example are placeholders',
    'sha256sum -c MANIFEST.txt',
]


def git(repo, *args):
    subprocess.run(['git', '-C', str(repo)] + list(args), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


class PatternTests(unittest.TestCase):
    def tearDown(self):
        scan.DENY_BLOCK.clear()
        scan.DENY_WARN.clear()

    def test_every_block_category_has_a_sample(self):
        self.assertEqual(set(SAMPLES), set(scan.BLOCK))

    def test_each_sample_trips_its_own_category(self):
        for category, text in SAMPLES.items():
            hits = scan.scan_text('x.txt', text)
            blocked = {h[1] for h in hits if h[0] == 'BLOCK'}
            self.assertIn(category, blocked, '%s did not fire on %r (got %s)' % (category, text, sorted(blocked)))

    def test_clean_lines_have_no_block_hit(self):
        for text in CLEAN:
            self.assertEqual([h for h in scan.scan_text('x.txt', text) if h[0] == 'BLOCK'], [], text)

    def test_windows_user_profile_and_laptop_checkout(self):
        self.assertTrue(scan.scan_text('x', j('C:\\Us', 'ers\\someone\\repo')))
        self.assertTrue(scan.scan_text('x', j('D:/Prog', 'ramming/repo')))

    def test_bare_run_number_is_only_a_warning(self):
        hits = scan.scan_text('x', j('artifact 35', '000000000'))
        self.assertEqual({h[0] for h in hits}, {'WARN'})

    def test_decorators_and_short_tokens_are_not_addresses(self):
        for text in ('+@pytest.mark.parametrize("x", [1])', 'the k@h product and q@h product'):
            self.assertEqual(scan.scan_text('x', text), [], text)

    def test_hex64_is_a_warning_not_a_block(self):
        hits = scan.scan_text('x', 'a' * 64)
        self.assertEqual([(h[0], h[1]) for h in hits], [('WARN', 'hex64-pin')])

    def test_allow_pattern_suppresses_one_hit(self):
        text = SAMPLES['card-serial']
        key = r'x\.txt:1:card-serial:.*'
        self.assertFalse([h for h in scan.scan_text('x.txt', text, [key]) if h[0] == 'BLOCK'])

    def test_file_name_can_leak(self):
        hits = scan.scan_blob(j('logs/black', 'hole-', '0123456789ABCDEF.txt'), b'fine')
        self.assertEqual([h[1] for h in hits], ['card-serial'])

    def test_binary_content_is_blocked(self):
        hits = scan.scan_blob('lib.so', b'ELF\0\0\0')
        self.assertEqual([(h[0], h[1]) for h in hits], [('BLOCK', 'binary-file')])

    def test_the_scanner_and_its_test_are_not_exempt(self):
        for name in ('scripts/ci/sanitise_scan.py', 'scripts/ci/test_sanitise_scan.py'):
            hits = scan.scan_blob(name, SAMPLES['credential'].encode())
            self.assertEqual([h[1] for h in hits if h[0] == 'BLOCK'], ['credential'], name)

    def test_the_public_scanner_names_no_organisation(self):
        """The shipped source holds shapes only; names live in the private deny file."""
        for name in ('sanitise_scan.py', 'test_sanitise_scan.py'):
            hits = scan.scan_blob(name, (HERE / name).read_bytes())
            self.assertEqual([h for h in hits if h[0] == 'BLOCK'], [], name)


class DenyFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='sanitise-deny-'))
        self.addCleanup(shutil.rmtree, str(self.tmp), True)
        self.addCleanup(scan.DENY_BLOCK.clear)
        self.addCleanup(scan.DENY_WARN.clear)
        self.path = self.tmp / 'deny.txt'
        self.path.write_text(DENY_RULES, encoding='utf-8')

    def test_rules_load_and_fire_with_their_level(self):
        self.assertEqual(scan.load_deny(self.path), 2)
        block = scan.scan_text('x', j('the Acme', 'Secret gateway'))
        self.assertEqual([(h[0], h[1]) for h in block], [('BLOCK', 'deny:project-name')])
        warn = scan.scan_text('x', j('codename blue', 'falcon'))
        self.assertEqual([(h[0], h[1]) for h in warn], [('WARN', 'deny:codename')])

    def test_no_rules_no_hit(self):
        self.assertEqual(scan.scan_text('x', j('the Acme', 'Secret gateway')), [])

    def test_malformed_rule_is_an_error(self):
        self.path.write_text('MAYBE name regex\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            scan.load_deny(self.path)

    def test_main_reads_the_deny_file_and_says_so_without_one(self):
        (self.tmp / 'a.txt').write_text(j('Acme', 'Secret') + '\n', encoding='utf-8')
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(scan.main(['--dir', str(self.tmp), '--deny', str(self.path)]), 1)
        self.assertIn('2 private rules loaded', out.getvalue())
        scan.DENY_BLOCK.clear()
        scan.DENY_WARN.clear()
        out = io.StringIO()
        saved = os.environ.pop('SANITISE_SCAN_DENY', None)
        try:
            with contextlib.redirect_stdout(out):
                self.assertEqual(scan.main(['--dir', str(self.tmp)]), 0)
        finally:
            if saved is not None:
                os.environ['SANITISE_SCAN_DENY'] = saved
        self.assertIn('no private deny file', out.getvalue())


class ModeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='sanitise-scan-'))
        self.addCleanup(shutil.rmtree, str(self.tmp), True)
        git(self.tmp, 'init', '-q')
        git(self.tmp, 'config', 'user.email', j('test', '@exam', 'ple.com'))
        git(self.tmp, 'config', 'user.name', 'Test')
        git(self.tmp, 'config', 'core.autocrlf', 'false')
        (self.tmp / 'old.txt').write_text('an old line with ' + SAMPLES['card-serial'] + '\n')
        git(self.tmp, 'add', 'old.txt')
        git(self.tmp, 'commit', '-q', '-m', 'base')
        git(self.tmp, 'branch', '-M', 'base')
        git(self.tmp, 'checkout', '-q', '-b', 'topic')

    def commit(self, name, text):
        (self.tmp / name).write_text(text)
        git(self.tmp, 'add', name)
        git(self.tmp, 'commit', '-q', '-m', 'add ' + name)

    def test_diff_mode_reports_only_added_lines(self):
        self.commit('clean.txt', 'nothing to see\n')
        self.assertEqual(scan.scan_diff(self.tmp, 'base...topic'), [])
        self.commit('leak.txt', 'line one\n' + SAMPLES['private-ipv4'] + '\n')
        hits = scan.scan_diff(self.tmp, 'base...topic')
        self.assertEqual([(h[1], h[2], h[3]) for h in hits if h[0] == 'BLOCK'], [('private-ipv4', 'leak.txt', 2)])

    def test_diff_mode_ignores_leaks_already_in_the_base(self):
        self.commit('clean.txt', 'ok\n')
        self.assertEqual([h for h in scan.scan_diff(self.tmp, 'base...topic') if h[2] == 'old.txt'], [])

    def test_diff_mode_blocks_an_added_binary(self):
        (self.tmp / 'blob.bin').write_bytes(b'\0\1\2\3')
        git(self.tmp, 'add', 'blob.bin')
        git(self.tmp, 'commit', '-q', '-m', 'bin')
        hits = scan.scan_diff(self.tmp, 'base...topic')
        self.assertEqual([(h[1], h[2]) for h in hits], [('binary-file', 'blob.bin')])

    def test_tree_mode_sees_the_base_leak(self):
        hits = scan.scan_tree(self.tmp, 'topic')
        self.assertIn(('card-serial', 'old.txt'), [(h[1], h[2]) for h in hits])

    def test_dir_mode_matches_tree_mode(self):
        self.commit('leak.txt', SAMPLES['credential'] + '\n')
        as_set = lambda hits: {(h[1], h[2], h[3]) for h in hits if h[0] == 'BLOCK'}
        self.assertEqual(as_set(scan.scan_dir(self.tmp)), as_set(scan.scan_tree(self.tmp, 'topic')))

    def test_dir_mode_skips_pycache(self):
        cache = self.tmp / '__pycache__'
        cache.mkdir()
        (cache / 'x.cpython-311.pyc').write_bytes(b'\0' + SAMPLES['card-serial'].encode())
        self.assertFalse([h for h in scan.scan_dir(self.tmp) if h[2].startswith('__pycache__')])

    def test_dir_mode_fails_closed_on_an_unreadable_file(self):
        (self.tmp / 'locked.txt').write_text('ok\n')
        real_open = open

        def refuse(path, *args, **kwargs):
            if str(path).endswith('locked.txt'):
                raise PermissionError('denied')
            return real_open(path, *args, **kwargs)

        with unittest.mock.patch('builtins.open', refuse):
            hits = scan.scan_dir(self.tmp)
        self.assertIn(('BLOCK', 'unreadable-file', 'locked.txt'), [(h[0], h[1], h[2]) for h in hits])

    def test_dir_mode_reads_a_path_longer_than_260_characters(self):
        deep = self.tmp
        for _ in range(6):
            deep = deep / ('d' * 45)
        os.makedirs(scan._long(str(deep)))
        with open(scan._long(str(deep / 'leak.txt')), 'w') as handle:
            handle.write(SAMPLES['credential'] + '\n')
        self.assertGreater(len(str(deep / 'leak.txt')), 260)
        self.assertIn('credential', [h[1] for h in scan.scan_dir(self.tmp) if h[0] == 'BLOCK'])

    def test_main_exit_codes(self):
        self.commit('clean.txt', 'ok\n')
        argv = ['--repo', str(self.tmp), '--diff', 'base...topic']
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(scan.main(argv), 0)
            self.commit('leak.txt', SAMPLES['home-path'] + '\n')
            self.assertEqual(scan.main(argv), 1)


if __name__ == '__main__':
    os.chdir(str(HERE))
    unittest.main()
