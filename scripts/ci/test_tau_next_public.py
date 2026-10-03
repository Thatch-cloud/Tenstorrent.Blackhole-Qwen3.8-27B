"""The Qwen repository is PUBLIC: the files of the tau-next work (the A0 screen, the drafter fine-tune packages, the ttml probe) must
carry no host, address, serial, digest, registry name, local path, prompt or completion. A scan of every such file by name pattern."""
import glob
import os
import re
import sys
import unittest

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PATTERNS = (
    os.path.join(HERE, 'a0_*.py'), os.path.join(HERE, 'test_a0_*.py'), os.path.join(HERE, 'tf_pair_*.py'),
    os.path.join(HERE, 'test_tf_pair_*.py'), os.path.join(HERE, 'ft_*.py'), os.path.join(HERE, 'test_ft_*.py'),
    os.path.join(HERE, 'dflash2_torch.py'), os.path.join(HERE, 'test_dflash2_torch.py'), os.path.join(HERE, 'test_upstream_parity.py'), os.path.join(HERE, 'ttml_probe*.py'),
    os.path.join(HERE, 'test_ttml_probe*.py'), os.path.join(ROOT, 'docs', 'a0-*.md'), os.path.join(ROOT, 'docs', 'ft-*.md'),
    os.path.join(ROOT, 'docker', 'a0-spark', '*'), os.path.join(ROOT, 'docker', 'tt-train', '*'),
    os.path.join(ROOT, 'patches', 'tt-train', '*'),
    os.path.join(HERE, 'references', 'tau-next', '*'))
# the files that quote the patterns they police
EXEMPT = ('test_tau_next_public.py', 'test_a0_run.py')
BANNED = re.compile(
    r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|(?!127\.0\.0\.1)\b\d{1,3}(\.\d{1,3}){3}\b|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|'
    r'/dev/tenstorrent|\bhome/|zot\.|@[A-Z0-9_]+@|\bzot:|registry\.[a-z]+\.[a-z]+|spark-[0-9a-f]{4}|[Oo]rnith|liamb|'
    r'[A-Za-z]:[\\/]Users|[A-Za-z]:[\\/]Programming|\.worktrees|/home/|/Users/')


# what the platform's operation looks like from outside: outages, recovery, reservations, placement. Method and gates belong here; the
# operator's runbook for a named host does not (review finding 4: the production runbook had been pushed to this public repository).
OPERATIONAL = re.compile(
    r'power[- ]cycle|\bstarv|resident (production )?(model|service)|restart policy|\bplacement\b|\bdrift\b|production (model|host|platform)|'
    r'\b\d+(\.\d+)? ?(GiB|GB) (reserv|reload)|\breservation\b|\boutage|serves a production|heartbeat of the node', re.I)


def files():
    out = []
    for pattern in PATTERNS:
        out.extend(path for path in glob.glob(pattern) if os.path.isfile(path) and os.path.basename(path) not in EXEMPT)
    return sorted(set(out))


class PublicTests(unittest.TestCase):
    def test_the_scan_finds_the_work(self):
        names = [os.path.basename(path) for path in files()]
        for expected in ('a0_run.py', 'tf_pair_walk.py', 'dflash2_torch.py', 'a0-dspark-screen.md', 'Dockerfile'):
            self.assertIn(expected, names)

    def test_no_file_carries_a_host_address_digest_or_local_path(self):
        for path in files():
            with open(path, encoding='utf-8') as handle:
                for number, line in enumerate(handle, 1):
                    self.assertIsNone(BANNED.search(line), '%s:%d' % (os.path.relpath(path, ROOT), number))

    def test_no_file_describes_the_platforms_operation(self):
        for path in files():
            with open(path, encoding='utf-8') as handle:
                for number, line in enumerate(handle, 1):
                    self.assertIsNone(OPERATIONAL.search(line), '%s:%d' % (os.path.relpath(path, ROOT), number))

    def test_the_operational_rule_catches_what_it_is_for(self):
        for text in ('the host needs a power cycle', 'the host starved twice', 'the resident model must stop', 'a 91 GiB reservation',
                     'restart policy no', 'placement drift', 'serves a production model'):
            self.assertIsNotNone(OPERATIONAL.search(text), text)
        for text in ('MemAvailable below the floor', 'the abort rule kills the screen container', 'no other container may start'):
            self.assertIsNone(OPERATIONAL.search(text), text)


if __name__ == '__main__':
    unittest.main()
