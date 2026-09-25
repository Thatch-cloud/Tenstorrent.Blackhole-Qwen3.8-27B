"""Refuse workflow YAML containing a literal backslash-n.

Three runs were lost to this in one session: fabric topology v2, the GDN conv
gate, and the CCL link gate. Each time a Python generator wrote a shell line
continuation, the two-character sequence survived into the YAML instead of a
real newline, and bash turned the unquoted backslash-n into a bare argument `n`.
Docker read it as an image name once; argparse read it as a positional once.

Only the WHITESPACE-SURROUNDED form is flagged. A backslash-n inside quotes is
ordinary and common here - printf format strings, and newline='\n' in embedded
Python - so flagging every occurrence produced 60 hits of which 59 were fine. The
bug always looks like an argument separator: space, backslash-n, space.
"""

import io
import sys
import glob

BAD = ' ' + chr(92) + 'n '   # space, backslash-n, space


def main():
    paths = sorted(glob.glob('.github/workflows/*.yml'))
    bad = []
    for path in paths:
        text = io.open(path, encoding='utf-8', errors='replace').read()
        for number, line in enumerate(text.split('\n'), 1):
            if BAD in line:
                bad.append((path, number, line.strip()[:110]))
    for path, number, line in bad:
        print('%s:%d  literal backslash-n: %s' % (path, number, line))
    print('%d workflow files scanned, %d offending lines' % (len(paths), len(bad)))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
