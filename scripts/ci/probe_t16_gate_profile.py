"""What request profile does THIS image's T16 gate actually demand?

Runs 35472072127 and 35472250194 both failed at 'Qualified 32K T16 replay with
four-row groups and native sampling required' - at ONE user, and with
QWEN_FROZEN_COMBINED_RUNTIME unset. So the frozen adapters were applied when the
image was built, not chosen at run time, and the gate in this image is fixed.

Meanwhile serving_runtime.bridge_factory builds a fixed (1, 68) page table, which
is 68 x 64 = 4352 tokens. If the gate demands 32768 those two can never agree, and
no amount of two-user work matters until one of them moves.

Reports the facts rather than another hardware slot's worth of guessing:

  - what request_context() returns in this image
  - the source of the gate the verifier engine imports
  - which positions that source will accept
  - the page-table width serving_runtime builds

CPU only: no device, no weights.
"""

import inspect
import io
import os
import re
import sys


NEXT_DEF = chr(10) + 'def '


def show(label, value):
    print('%-40s %s' % (label, value))


def main():
    show('QWEN_FROZEN_COMBINED_RUNTIME', os.environ.get('QWEN_FROZEN_COMBINED_RUNTIME'))
    show('QWEN_DSPARK_REQUEST_CONTEXT', os.environ.get('QWEN_DSPARK_REQUEST_CONTEXT'))
    try:
        from dspark_context_selection import request_context
        show('request_context()', request_context())
    except BaseException as error:
        show('request_context()', 'failed: %s' % error)

    # Read the IMAGE's copies by PATH, not by import. The probe lane mounts the
    # repo at /probe and puts it FIRST on PYTHONPATH, so importing measures the
    # repo's files and says nothing about what the server actually runs.
    source = ''
    for name in ('target_t16_attention_gate.py', 'dspark_context_selection.py'):
        path = '/experiment-scripts/ci/' + name
        try:
            text = io.open(path, encoding='utf-8').read()
        except BaseException as error:
            show('image %s' % name, 'unreadable: %s' % error)
            continue
        show('image %s' % name, '%d bytes, frozen-adapted=%s'
             % (len(text), 'frozen_combined_runtime' in text))
        if name.startswith('target_t16'):
            source = text
            start = text.find('def validate_request_option')
            print('----- image target_t16_attention_gate.validate_request_option -----')
            print(text[start:text.find(NEXT_DEF, start + 1)])
        else:
            start = text.find('def request_context')
            print('----- image request_context -----')
            print(text[start:text.find(NEXT_DEF, start + 1)])

    positions = sorted(set(int(value) for value in re.findall(r'position != (\d+)', source or '')))
    show('positions this gate accepts', positions)

    try:
        import serving_runtime
        runtime_source = inspect.getsource(serving_runtime)
        widths = sorted(set(int(value) for value in re.findall(r'torch\.full\(\(1, (\d+)\)', runtime_source)))
        show('page table widths built', widths)
        show('tokens those cover', [width * 64 for width in widths])
    except BaseException as error:
        show('serving_runtime', 'unavailable: %s' % error)

    print()
    print('VERDICT')
    if not positions:
        print('  The gate in this image names no fixed position, so the profile is not')
        print('  the blocker and the failure is something else.')
        return 0
    covered = [width * 64 for width in widths] if 'widths' in dir() else []
    if covered and all(position > max(covered) for position in positions):
        print('  The gate demands position %s and the page table covers at most %d'
              % (positions, max(covered)))
        print('  tokens. They cannot both be satisfied, so this image cannot decode at')
        print('  ANY context and no two-user work will change that. The image must be')
        print('  rebuilt without the frozen adapters, or the page table widened.')
    else:
        print('  Gate positions %s against page coverage %s: a consistent request exists.'
              % (positions, covered))
    return 0


if __name__ == '__main__':
    sys.exit(main())
