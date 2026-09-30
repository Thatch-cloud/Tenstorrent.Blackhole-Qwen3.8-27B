"""Dump the plugin scheduler sources Lever N M2 needs: TTScheduler and LaneScheduler.

M2 is two changes, both in the vLLM TT plugin, not our overlay code
(docs/lever-n-plugin-contract-2026-09-19.md):

1. Alternation in LaneScheduler._negotiate_forced_mode: a partial prefill in
   flight should not force every step prefill-only when some lane has running
   decodes. Today `intent = max(self._local_prefill_intent(sched) for sched in
   self.lanes)` and `_local_prefill_intent` returns 1 whenever
   `has_partial_prefill`, so any lane's partial prefill wins for the whole step.
2. One-in-flight in TTScheduler._schedule_prefill_only: hide `waiting` and
   `skipped_waiting` when `partial_prefills` is non-empty.

Prior CPU probes already captured some of this verbatim from the image.
probe_scheduler_hook.py and probe_frontier_refusal.py (runs 35485312330 and
later, cpu-probe-v20 through v23) printed, from vllm_tt_plugin.scheduler:
TTScheduler.schedule in full (43 lines), TTScheduler._schedule_prefill_only in
full, and TTScheduler._has_pending_prefill in full. None of those captured
outputs contain _schedule_decode_only's body, set_forced_mode's body, the
TTSchedulingMode enum's definition, or anything belonging to LaneScheduler.

That absence is itself informative: probe_frontier_refusal.plugin_sources()
walks every name bound in the vllm_tt_plugin.scheduler module's own namespace
(vars(module)) looking for classes with a _negotiate_forced_mode or
_local_prefill_intent method, across four separate probe runs, and found
nothing. So LaneScheduler is neither defined in, nor imported at module scope
into, scheduler.py - its actual module is unknown from outside the image, and
grepping this repo, its evidence directories and prior job tmp dirs for
"LaneScheduler" or "_negotiate_forced_mode" turns up only the design docs'
quoted two- and three-line fragments, never the real source.

This probe closes that gap by reading files directly by path rather than
importing anything from the plugin, so it needs no vLLM install and runs the
same on a bare CPU runner as inside the serving image. It:

  - lists every .py file under the plugin package, so the listing itself
    answers whether the package is even mounted where expected
  - dumps scheduler.py in full, with line numbers and a sha256 - covering
    TTScheduler, _schedule_decode_only, set_forced_mode and TTSchedulingMode
    in whatever form they take today, not just the fragments already seen
  - walks every .py file under the package for the literal string
    "_negotiate_forced_mode" and dumps whichever file(s) contain it in full;
    that file is where LaneScheduler actually lives
  - greps each dumped file for the bookkeeping fields the design doc's
    PREFILL_ONLY -> DECODE_ONLY fallback carries across a discarded pass
    (finished_req_ids, free_encoder_mm_hashes, preempted_req_ids), so that
    fallback's exact shape is visible without hunting for it by eye

CPU only: no device, no weights, no import from vllm or vllm_tt_plugin.
"""

import hashlib
import io
import sys
from pathlib import Path

PLUGIN = Path('/opt/qwen-fast-plugin/src/vllm_tt_plugin')
SCHEDULER = PLUGIN / 'scheduler.py'

# Anchors worth surfacing in a grep index atop each full dump: the two methods
# M2 edits, the class names and enum M2's docs assume exist, the fallback this
# repo's other probes already found, and the three bookkeeping fields the
# contract doc says that fallback must not drop.
MARKERS = [
    '_negotiate_forced_mode',
    '_local_prefill_intent',
    'class TTSchedulingMode',
    'class LaneScheduler',
    'class TTScheduler',
    'def schedule',
    'def _schedule_prefill_only',
    'def _schedule_decode_only',
    'def set_forced_mode',
    'def _has_pending_prefill',
    'self.lanes',
    'finished_req_ids',
    'free_encoder_mm_hashes',
    'preempted_req_ids',
]


def show(label, value):
    print('%-44s %s' % (label, value))


def read(path):
    return io.open(path, encoding='utf-8', errors='replace').read()


def header(path, text):
    print()
    print('=' * 100)
    print('%s  sha256=%s  lines=%d' % (
        path, hashlib.sha256(text.encode('utf-8')).hexdigest()[:16], text.count('\n') + 1))
    print('=' * 100)


def dump(path, text):
    header(path, text)
    for i, line in enumerate(text.split('\n'), 1):
        print('%5d  %s' % (i, line))


def grep(path, text, patterns):
    header('%s :: grep %s' % (path, patterns), text)
    hits = 0
    for i, line in enumerate(text.split('\n'), 1):
        if any(p in line for p in patterns):
            print('%5d  %s' % (i, line.rstrip()))
            hits += 1
    if not hits:
        print('  (no marker matched)')


def find_files_containing(root, needle):
    hits = []
    for path in sorted(root.rglob('*.py')):
        try:
            text = read(path)
        except OSError as error:
            show('unreadable', '%s: %s' % (path, error))
            continue
        if needle in text:
            hits.append((path, text))
    return hits


def main():
    if not PLUGIN.is_dir():
        show('missing plugin package', PLUGIN)
        print()
        print('VERDICT')
        print('  %s does not exist on this host; nothing to dump. Run this' % PLUGIN)
        print('  inside the serving image (the CPU-probe CI lane), not locally.')
        return 0

    listing = sorted(str(p.relative_to(PLUGIN)) for p in PLUGIN.rglob('*.py'))
    show('vllm_tt_plugin .py files (%d)' % len(listing), ' '.join(listing) or 'NONE')

    scheduler_text = ''
    if SCHEDULER.is_file():
        scheduler_text = read(SCHEDULER)
        dump(SCHEDULER, scheduler_text)
        grep(SCHEDULER, scheduler_text, MARKERS)
    else:
        show('missing', SCHEDULER)

    # LaneScheduler is not in scheduler.py's namespace (four prior probes
    # confirmed), so search widening roots first-hit-wins: the plugin package,
    # then the whole plugin checkout, then the vLLM install. That covers the
    # case the design doc is wrong about its exact home without a second run.
    wider_roots = [PLUGIN, Path('/opt/qwen-fast-plugin')]
    site = Path('/opt/venv/lib/python3.10/site-packages/vllm')
    if site.is_dir():
        wider_roots.append(site)
    lane_hits, searched = [], None
    seen_roots = []
    for root in wider_roots:
        if not root.is_dir() or root in seen_roots:
            continue
        seen_roots.append(root)
        lane_hits = find_files_containing(root, '_negotiate_forced_mode')
        if lane_hits:
            searched = root
            break
    show('searched roots for _negotiate_forced_mode', ' '.join(str(r) for r in seen_roots))
    lane_paths = [str(path) for path, _ in lane_hits]
    show('files containing _negotiate_forced_mode', lane_paths or 'NONE FOUND')
    for path, text in lane_hits:
        if path == SCHEDULER:
            continue  # already dumped above
        dump(path, text)
        grep(path, text, MARKERS)

    print()
    print('VERDICT')
    if scheduler_text:
        show('  TTSchedulingMode in scheduler.py', 'class TTSchedulingMode' in scheduler_text)
        show('  set_forced_mode in scheduler.py', 'def set_forced_mode' in scheduler_text)
        show('  _schedule_decode_only in scheduler.py', 'def _schedule_decode_only' in scheduler_text)
    if lane_hits:
        print('  LaneScheduler source found in: %s' % ', '.join(lane_paths))
        print('  The full dumps above are what an M2 patch (alternation in')
        print('  _negotiate_forced_mode, one-in-flight in _schedule_prefill_only)')
        print('  must be staged against, function-span or AST-node scoped, in the')
        print('  style of lever_n_model_patch.py / serving_plugin_patch.py.')
    else:
        print('  No file under %s contains _negotiate_forced_mode.' % PLUGIN)
        print('  LaneScheduler is either named differently, generated at import')
        print('  time, or defined outside this package - widen the search root')
        print('  (e.g. site-packages) and re-run before writing the M2 patch.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
