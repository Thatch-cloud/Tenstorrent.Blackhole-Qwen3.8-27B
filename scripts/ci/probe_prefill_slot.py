"""Which native slot does the plugin's prefill write, and how is it chosen?

Runs 35492676194 and 35493208438 showed the second concurrent user decoding
from the FIRST user's GDN state: the plugin logged 'Prefilling 1 user(s) into
slots [1]' for it, while the fast path's ActiveSnapshot allocates, saves and
restores native GDN state at index 0 unconditionally (gdn_snapshot.py:18, :31,
:43-45). The second user's initial snapshot and carry were therefore taken from
slot 0 - the first user's post-prefill state - and its own state, written to
slot 1, was never read.

The fix needs the slot at admission. This prints, from the IMAGE's model source
(the repo has no copy):

  - the signature of _forward_prefill_chunk_masked_tp, the boundary
    PrefillWindowCapture wraps (dflash_prefill_window.py:87-109), so the
    capture can record the slot if it is an argument
  - the signature and slot-selection lines of _prefill_forward_tp_batched,
    prefill_paged_slots and prefill_paged_slots_range
  - every line of the model file mentioning 'slot', bounded

CPU only: no device, no weights. Reads by path; imports nothing from the model.
"""

import io
import re
import sys

MODEL = '/opt/tt-metal/models/demos/blackhole/qwen36/tt/qwen36_vllm.py'
CANDIDATES = (MODEL, '/opt/tt-metal/models/demos/blackhole/qwen36/tt/model.py')
METHODS = ('_forward_prefill_chunk_masked_tp', '_prefill_forward_tp_batched', 'prefill_paged_slots_range',
           'prefill_paged_slots', '_prefill_forward_tp', 'prefill_forward')


def show(label, value):
    print('%-44s %s' % (label, value))


def method_block(text, name, limit=40):
    match = re.search(r'^([ \t]*)def %s\s*\(' % re.escape(name), text, re.M)
    if not match:
        return None
    indent = match.group(1)
    lines = text[match.start():].splitlines()
    block = [lines[0]]
    for line in lines[1:]:
        if line.strip() and not line.startswith(indent + ' ') and not line.startswith(indent + '\t'):
            break
        block.append(line)
    return block[:limit] + (['    ... (%d more lines)' % (len(block) - limit)] if len(block) > limit else [])


def main():
    found = 0
    for path in CANDIDATES:
        try:
            text = io.open(path, encoding='utf-8').read()
        except BaseException as error:
            show('image %s' % path, 'unreadable: %s' % error)
            continue
        show('image %s' % path, '%d lines' % len(text.splitlines()))
        for name in METHODS:
            block = method_block(text, name)
            if block is None:
                continue
            found += 1
            print('----- %s: %s -----' % (path.rsplit('/', 1)[-1], name))
            for line in block:
                print(line[:200])
        slot_lines = [(number, line.strip()) for number, line in enumerate(text.splitlines(), 1)
                      if 'slot' in line.lower() and not line.strip().startswith('#')]
        print('----- %s: lines mentioning slot (%d) -----' % (path.rsplit('/', 1)[-1], len(slot_lines)))
        for number, line in slot_lines[:80]:
            print('%5d: %s' % (number, line[:180]))
        if len(slot_lines) > 80:
            print('... %d more' % (len(slot_lines) - 80))

    print()
    print('VERDICT')
    if not found:
        print('  No prefill method found in the image model source at the expected paths;')
        print('  the slot must be found another way (grep the model tree in the image).')
        return 0
    print('  %d prefill methods printed. The slot the plugin writes is whatever the' % found)
    print('  batched prefill assigns from its running set; the capture boundary above')
    print('  shows whether the slot reaches _forward_prefill_chunk_masked_tp as an')
    print('  argument the fast path can record at admission.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
