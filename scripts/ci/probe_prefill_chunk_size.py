"""Is the prefill chunk size on this endpoint a tuned value or an unknown-model fallback?

The four-user arm (run 35658854824) prefills 32,768 tokens in 13.1 s - 2,423 tok/s -
and does it four times in series, which is where the whole 79.4 s decode stall comes
from (docs/ttft-decode-stall-2026-09-22.md). Anything that shrinks 13.1 s shrinks the
stall proportionally, and unlike the Lever N scheduling work it needs no resumable
prefill, no capture surgery and no new chunk boundaries.

That server's own startup log says, twice:

    models.tt_transformers.tt.model_config:get_max_prefill_chunk_size:2478 -
      Unknown model Qwen3.8-27B on device P300, setting MAX_PREFILL_CHUNK_SIZE to 4
      for compatibility
    ...:2481 - Try setting MAX_PREFILL_CHUNK_SIZE to larger powers of 2 up to e.g. 128
      for faster performance (if you run out of L1 memory it was too high)

So the value in force is a compatibility fallback for a model the table does not know,
and the vendor's own next line invites raising it. Before changing anything, this
probe establishes what is actually there:

  1. the full text of get_max_prefill_chunk_size and whatever table it consults, so
     the fallback and the recognised entries can be read rather than guessed;
  2. what the returned value is measured in - tokens, tiles, or a multiplier - since
     "4" and "128" mean nothing until that is known;
  3. every call site, to see what the value actually bounds;
  4. whether Qwen36ModelArgs could override it cleanly (it subclasses ModelArgs), and
     whether any env var already influences it;
  5. what the nearest recognised model/device entries choose, as evidence for what is
     safe rather than what is merely permitted.

CPU only: no device, no weights, no mesh, no build. Reads files and prints.
"""

import io
import os
import re
import sys
from pathlib import Path

TT = Path('/opt/tt-metal')
BASE = TT / 'models/tt_transformers/tt/model_config.py'
QWEN36 = TT / 'models/demos/blackhole/qwen36/tt/model_config.py'
NAME = 'MAX_PREFILL_CHUNK_SIZE'
FUNCTION = 'get_max_prefill_chunk_size'


def show(label, value):
    print('%-46s %s' % (label, value))


def read(path):
    return io.open(str(path), encoding='utf-8', errors='replace').read()


def span(source, header):
    """The full body of a def, by indentation - no ast import needed for a dump."""
    start = source.find(header)
    if start < 0:
        return None
    lines = source[start:].split('\n')
    indent = len(lines[0]) - len(lines[0].lstrip())
    body = [lines[0]]
    for line in lines[1:]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        body.append(line)
    return '\n'.join(body)


def numbered(text, first):
    for offset, line in enumerate(text.split('\n')):
        print('%5d  %s' % (first + offset, line))


def line_of(source, needle):
    index = source.find(needle)
    return None if index < 0 else source[:index].count('\n') + 1


def main():
    if not BASE.is_file():
        show('MISSING', BASE)
        print()
        print('VERDICT')
        print('  tt_transformers model_config.py is not at the expected path; locate it with')
        print('  find /opt/tt-metal -name model_config.py and re-run.')
        return 0

    source = read(BASE)
    show('base model_config.py', '%s (%d lines)' % (BASE, source.count('\n') + 1))
    show('qwen36 subclass present', QWEN36.is_file())

    print()
    print('=' * 100)
    print('1. %s and its table' % FUNCTION)
    print('=' * 100)
    header = span(source, 'def %s' % FUNCTION)
    if header is None:
        show('NOT FOUND', 'def %s' % FUNCTION)
    else:
        numbered(header, line_of(source, 'def %s' % FUNCTION))

    print()
    print('--- every %s mention in the base file ---' % NAME)
    for number, line in enumerate(source.split('\n'), 1):
        if NAME in line:
            print('%5d  %s' % (number, line.rstrip()))

    print()
    print('=' * 100)
    print('2. What the value is measured in')
    print('=' * 100)
    # The unit is whatever the call sites multiply or compare it against.
    for number, line in enumerate(source.split('\n'), 1):
        if FUNCTION in line and 'def ' not in line:
            print('%5d  %s' % (number, line.rstrip()))
    for keyword in ('chunk_size', 'max_prefill_chunk'):
        print()
        print('--- assignments mentioning %r ---' % keyword)
        shown = 0
        for number, line in enumerate(source.split('\n'), 1):
            if keyword in line and '=' in line and 'def ' not in line and shown < 25:
                print('%5d  %s' % (number, line.rstrip()))
                shown += 1

    print()
    print('=' * 100)
    print('3. Call sites across the served tree')
    print('=' * 100)
    roots = [TT / 'models/tt_transformers', TT / 'models/demos/blackhole/qwen36']
    hits = 0
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob('*.py')):
            try:
                text = read(path)
            except OSError:
                continue
            for number, line in enumerate(text.split('\n'), 1):
                if FUNCTION in line or NAME in line:
                    print('%-72s %5d  %s' % (str(path.relative_to(TT)), number, line.strip()[:110]))
                    hits += 1
    show('total call sites / mentions', hits)

    print()
    print('=' * 100)
    print('4. Override surface and environment')
    print('=' * 100)
    if QWEN36.is_file():
        qwen = read(QWEN36)
        show('qwen36 overrides %s' % FUNCTION, ('def %s' % FUNCTION) in qwen)
        show('qwen36 mentions %s' % NAME, NAME in qwen)
        show('qwen36 subclasses ModelArgs', 'ModelArgs)' in qwen)
    environment = sorted({match for match in re.findall(r"environ(?:\.get)?\(\s*['\"]([A-Z0-9_]+)", source)
                          if 'PREFILL' in match or 'CHUNK' in match})
    show('env vars in base touching PREFILL/CHUNK', environment or 'none')
    show('MAX_PREFILL_CHUNK_SIZE set in this env', os.environ.get(NAME, '(unset)'))

    print()
    print('=' * 100)
    print('5. What recognised models choose')
    print('=' * 100)
    table = span(source, 'def %s' % FUNCTION) or ''
    values = sorted({int(v) for v in re.findall(r'\b(\d+)\b', table) if v.isdigit() and int(v) in
                     (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)})
    show('power-of-two literals in the function', values or 'none')

    print()
    print('VERDICT')
    if header is None:
        print('  %s not found in the base model_config; the log line names it at :2478,' % FUNCTION)
        print('  so the image differs from what this probe expects. Dump the file and re-read.')
    else:
        print('  The function above is what returned 4 for Qwen3.8-27B on P300. Read its table to')
        print('  see which (model, device) pairs are recognised and what they pick, and the call')
        print('  sites in section 3 for what the value bounds. Section 4 says whether')
        print('  Qwen36ModelArgs can override it without touching the base file - if it can, the')
        print('  experiment is a subclass method in the existing m3native graft, not a new mount.')
        print('  Nothing here changes any value: raising it risks L1 exhaustion, which is why the')
        print('  next step is one measured arm at one larger power of two, not a default change.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
