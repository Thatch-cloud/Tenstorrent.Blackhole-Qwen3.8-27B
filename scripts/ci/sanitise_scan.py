#!/usr/bin/env python3
"""Pre-publication scan: report every string that must not reach the public repository.

Three ways to point it at content:

  sanitise_scan.py --dir PATH               every text file under PATH
  sanitise_scan.py --diff BASE...HEAD       only the lines a git range ADDS (and any added binary file)
  sanitise_scan.py --tree REF [PATHSPEC..]  every file of a commit's tree

Exit status 1 if any BLOCK pattern matches, else 0. WARN patterns are listed for a human to read; most are
legitimate (a 64-hex string that names a public upstream file, for instance) and are not fatal.

An allow file (--allow FILE) holds one regular expression per line, matched with re.fullmatch against
"path:line:category:matched-text". Use it, with a written reason in a comment above each line, for the rare BLOCK hit
that is deliberate. Do not use it to silence a real leak: fix the file.

Organisation-specific names are NOT listed here: this file is public. Keep them in a private deny file and pass it
with --deny FILE (or SANITISE_SCAN_DENY=FILE). One rule per line, "LEVEL name regex" (LEVEL is BLOCK or WARN, name is
the category shown in the report, regex is the rest of the line); blank lines and lines starting with # are ignored.
Without a deny file the scan says so, because it then checks shapes only.

The scanner and its test are scanned like every other file; their samples are synthetic.
"""
import argparse
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

# Category -> regular expression. A hit is reported with the category name.
# Only STRUCTURAL patterns live here (shapes that are never fine to publish, whoever you are). Names that are specific
# to one organisation - its hosts, repositories, runners, e-mail domains - belong in a private deny file that is kept
# OUTSIDE this repository and passed with --deny FILE (or SANITISE_SCAN_DENY=FILE); see load_deny().
BLOCK = {
    # any host on a private-network suffix, followed by a path or port
    'registry-host': r'[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:local|lan|internal|corp|home\.arpa)(?::\d+)?/',
    'private-ipv4': r'\b(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b',
    'card-serial': r'blackhole-[0-9A-Fa-f]{16}\b',
    # a developer's or runner's home directory, a Windows user profile, a laptop checkout
    'home-path': (r'/home/(?!user\b|ubuntu\b|runner\b)[a-z_][a-z0-9_-]*'
                  r'|[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s]+|[A-Za-z]:[\\/]+Programming[\\/]'),
    'credential': (r'(?:-pw|--password)\s+\S+|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,}'
                   r'|\bhf_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|xox[baprs]-[A-Za-z0-9-]+'),
    'image-digest': r'@sha256:[0-9a-f]{12,64}|\bsha256:[0-9a-f]{12,64}\b',
    # a CI run id named as one (a bare 11-digit number is only a WARN)
    'run-id': r'\b[Rr]uns?\s+(?:id\s+)?#?\d{10,12}\b',
}
WARN = {
    'hex64-pin': r'(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])',
    'actions-run-id': r'(?<!\d)3[4-7]\d{9}(?!\d)',
    'any-email': r'[A-Za-z0-9][A-Za-z0-9._%+-]*@[A-Za-z0-9.-]+\.[a-z]{2,}',
    'ssh-target': r'\b[a-z_][a-z0-9_-]*@[a-z0-9.-]{3,}\b(?=\s|"|$)',
}

SKIP_DIRS = {'__pycache__', '.git', 'build', 'build_Release', '.cpmcache'}
SKIP_SUFFIX = {'.pyc', '.png', '.jpg', '.jpeg', '.gif', '.ico'}


def _compile(table):
    return {name: re.compile(rx) for name, rx in table.items()}


BLOCK_RX = _compile(BLOCK)
WARN_RX = _compile(WARN)


DENY_BLOCK = {}
DENY_WARN = {}


def load_deny(path):
    """Read a private deny file into DENY_BLOCK / DENY_WARN (name -> compiled regex). Returns the rule count."""
    DENY_BLOCK.clear()
    DENY_WARN.clear()
    count = 0
    for lineno, raw in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split(None, 2)
        if len(parts) != 3 or parts[0] not in ('BLOCK', 'WARN'):
            raise ValueError('%s:%d: expected "BLOCK|WARN name regex"' % (path, lineno))
        table = DENY_BLOCK if parts[0] == 'BLOCK' else DENY_WARN
        table['deny:' + parts[1]] = re.compile(parts[2])
        count += 1
    return count


def scan_text(name, text, allow=()):
    """Return [(level, category, name, lineno, matched_text)] for one file's text."""
    hits = []
    for lineno, line in enumerate(text.splitlines(), 1):
        hits.extend(_scan_line(name, lineno, line, allow))
    return hits


def _scan_line(name, lineno, line, allow):
    for level, table in (('BLOCK', BLOCK_RX), ('BLOCK', DENY_BLOCK), ('WARN', WARN_RX), ('WARN', DENY_WARN)):
        for category, rx in table.items():
            for match in rx.finditer(line):
                key = '%s:%d:%s:%s' % (name, lineno, category, match.group(0))
                if any(re.fullmatch(pattern, key) for pattern in allow):
                    continue
                yield (level, category, name, lineno, match.group(0)[:80])


def scan_path_name(name, allow=()):
    """File names can leak too (a serial number in a file name)."""
    return [(lvl, cat, name, 0, txt) for (lvl, cat, _n, _l, txt) in _scan_line(name, 0, name, allow) if lvl == 'BLOCK']


def scan_blob(name, data, allow=()):
    hits = scan_path_name(name, allow)
    if b'\0' in data[:8192]:
        hits.append(('BLOCK', 'binary-file', name, 0, '(NUL byte: binary content is not published)'))
        return hits
    hits.extend(scan_text(name, data.decode('utf-8', 'replace'), allow))
    return hits


def skipped(name):
    parts = name.replace('\\', '/').split('/')
    return bool(set(parts) & SKIP_DIRS) or Path(name).suffix.lower() in SKIP_SUFFIX


def run_git(repo, *args):
    return subprocess.run(['git', '-C', str(repo)] + list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          check=True).stdout


def _long(path):
    """Windows refuses paths of 260 characters or more unless they carry the extended-length prefix."""
    path = os.path.abspath(path)
    prefix = chr(92) * 2 + '?' + chr(92)
    return prefix + path if os.name == 'nt' and not path.startswith((chr(92) * 2)) else path


def scan_dir(root, allow=()):
    """Fail closed: a file that cannot be read is a BLOCK, never a silent skip."""
    root = os.path.abspath(str(root))
    hits = []
    for dirpath, dirnames, filenames in os.walk(_long(root)):
        dirnames.sort()
        for fname in sorted(filenames):
            full = os.path.join(dirpath, fname)
            rel = os.path.relpath(full, _long(root)).replace(os.sep, '/')
            if skipped(rel):
                continue
            try:
                with open(full, 'rb') as handle:
                    data = handle.read()
            except OSError as exc:
                hits.append(('BLOCK', 'unreadable-file', rel, 0, '(%s)' % exc.__class__.__name__))
                continue
            hits.extend(scan_blob(rel, data, allow))
    return hits


def scan_tree(repo, ref, specs=(), allow=()):
    cmd = ['ls-tree', '-r', '--name-only', '-z', ref]
    if specs:
        cmd += ['--'] + list(specs)
    names = [n for n in run_git(repo, *cmd).decode('utf-8', 'replace').split('\0') if n]
    hits = []
    for name in names:
        if skipped(name):
            continue
        hits.extend(scan_blob(name, run_git(repo, 'show', '%s:%s' % (ref, name)), allow))
    return hits


def scan_diff(repo, rng, allow=()):
    """Scan only what the range adds: added lines of text files, and the names of added or renamed files."""
    hits = []
    raw = run_git(repo, 'diff', '--no-color', '--no-ext-diff', '-U0', '--no-renames', rng).decode('utf-8', 'replace')
    current = None
    lineno = 0
    for line in raw.splitlines():
        if line.startswith('diff --git '):
            current = None
            continue
        if line.startswith('+++ '):
            target = line[4:]
            current = None if target == '/dev/null' else target[2:] if target.startswith('b/') else target
            if current is not None and (skipped(current)):
                current = None
            elif current is not None:
                hits.extend(scan_path_name(current, allow))
            continue
        if line.startswith('Binary files ') or line.startswith('GIT binary patch'):
            name = line.split(' and ')[-1].replace(' differ', '')
            name = name[2:] if name.startswith('b/') else name
            if not skipped(name):
                hits.append(('BLOCK', 'binary-file', name, 0, '(binary content is not published)'))
            continue
        if line.startswith('@@'):
            m = re.search(r'\+(\d+)', line)
            lineno = int(m.group(1)) - 1 if m else 0
            continue
        if current is None:
            continue
        if line.startswith('+') and not line.startswith('+++'):
            lineno += 1
            hits.extend(_scan_line(current, lineno, line[1:], allow))
    return hits


def load_allow(path):
    if not path:
        return []
    return [ln.strip() for ln in Path(path).read_text(encoding='utf-8').splitlines()
            if ln.strip() and not ln.lstrip().startswith('#')]


def report(hits, summary):
    if summary:
        by_cat = Counter((h[0], h[1]) for h in hits)
        for (level, category), count in sorted(by_cat.items()):
            files = len({h[2] for h in hits if h[0] == level and h[1] == category})
            print('%-5s %-16s hits=%-6d files=%d' % (level, category, count, files))
    else:
        for h in hits:
            print('%-5s %-16s %s:%d %s' % (h[0], h[1], h[2], h[3], h[4]))
    blocks = sum(1 for h in hits if h[0] == 'BLOCK')
    warns = len(hits) - blocks
    print('sanitise_scan: %d BLOCK, %d WARN' % (blocks, warns))
    return 1 if blocks else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dir', help='scan every text file under this directory')
    mode.add_argument('--diff', metavar='RANGE', help='scan the lines added by a git range, e.g. origin/main...HEAD')
    mode.add_argument('--tree', nargs='+', metavar=('REF', 'PATHSPEC'), help='scan a commit tree')
    ap.add_argument('--repo', default='.', help='git repository for --diff and --tree (default: current directory)')
    ap.add_argument('--deny', default=os.environ.get('SANITISE_SCAN_DENY'),
                    help='private deny file kept outside the repository (default: $SANITISE_SCAN_DENY)')
    ap.add_argument('--allow', help='file of allow regexes, matched against path:line:category:text')
    ap.add_argument('--summary', action='store_true', help='print counts per category instead of every hit')
    ap.add_argument('--block-only', action='store_true', help='hide WARN hits')
    args = ap.parse_args(argv)
    allow = load_allow(args.allow)
    if args.deny:
        print('sanitise_scan: %d private rules loaded' % load_deny(args.deny))
    else:
        print('sanitise_scan: no private deny file (--deny); shapes only, organisation names are NOT checked')
    if args.dir:
        hits = scan_dir(args.dir, allow)
    elif args.diff:
        hits = scan_diff(args.repo, args.diff, allow)
    else:
        hits = scan_tree(args.repo, args.tree[0], args.tree[1:], allow)
    if args.block_only:
        hits = [h for h in hits if h[0] == 'BLOCK']
    return report(hits, args.summary)


if __name__ == '__main__':
    sys.exit(main())
