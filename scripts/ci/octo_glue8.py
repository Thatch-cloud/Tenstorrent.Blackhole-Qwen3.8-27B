"""QWEN_FAST_OCTO_GLUE8 (default off, GATE ONLY): the verify-glue levers (V2, V1, T2 windows) at the octo block's EIGHT-row users.

The octo block (QWEN_FAST_OCTO, docs/tp4-octo.md) packs eight users of eight rows into the 64-row block. The glue levers of tp4/vglue move HALF tiles (a sixteen-row M3 user), so at
eight rows (a quarter tile) every one of them took the SERVED path and said so with tp4_vglue.EIGHT_ROW_SERVED. With this flag on each lever that is on runs natively at eight rows:

  site        lever (flag that turns it on)                          what runs natively at eight rows
  split       V2  QWEN_FAST_TP4_GDN_GLUE                              the block projection -> eight (1, 8, W) pieces, one quarter-tile launch (gdn_rows_dma8_tp)
  merge       V2  QWEN_FAST_TP4_GDN_GLUE                              the eight gated outputs -> the (1, 64, N) block, one launch
  block_conv  V1  QWEN_FAST_TP4_GDN_BLOCK_CONV (needs V2 and windows) canon block, window stack, ONE conv_gates at batch 64, unstack (gdn_block_conv8_tp)
  windows     T2  QWEN_FAST_VERIFY_T2 (cut 'windows')                 the eight users' 32 windows in one launch (gdn_conv_windows_packed8)

A lever that is off stays off; the flag adds nothing to it. (V2 and V1 are profile keys; the T2 windows cut is the serving image's ENV default, so a profile that does not name
QWEN_FAST_VERIFY_T2 has it on: sites_wanted reads an absent key as on.) The sites NOT made native (and why) are in docs/tp4-octo-glue8.md: K5-A (QWEN_FAST_GDN_SEQ_BLOCK) is a recurrence compute
kernel for sixteen-row users, so the octo block keeps the served batched launch.

The flag is strict ('0' or '1'; anything else raises), read at each call (so when the verify trace is captured), and any value but unset/'0' at two cards raises. Unset, every path is the
one that ran before this module existed: nothing here is imported by the M3 path, the octo block logs tp4_vglue.EIGHT_ROW_SERVED as it always did, and the launch sequences are the pinned
ones (test_octo_glue8_off). A native site that declines (a launch the kernel cannot carry, a placement it does not take) runs the served path and logs FALLBACK, which a gated arm fails on.

Markers (server log; glue8_problems is the smoke rule):
  [PINDIAG] tp4 octo glue8 engaged site=<split|merge|block_conv|windows> users=8 rows=8      once per process per site, at the first native launch
  [PINDIAG] tp4 octo glue8 fell back site=<site> reason=<why>                                a native site declined; its served path ran (fails a gated arm)
  [PINDIAG] tp4 vglue engaged site=packed_verify ... glue8_split=48 glue8_merge=48 glue8_block_conv=48 glue8_windows=48
                                                                                              the existing per-capture count line (tp4_vglue.note): one count per GDN layer

Stdlib only, py 3.7.
"""

import os
import re

import tp_shapes

FLAG = 'QWEN_FAST_OCTO_GLUE8'
OCTO_FLAG = 'QWEN_FAST_OCTO'
V2_FLAG = 'QWEN_FAST_TP4_GDN_GLUE'
V1_FLAG = 'QWEN_FAST_TP4_GDN_BLOCK_CONV'
T2_FLAG = 'QWEN_FAST_VERIFY_T2'
T2_SKIP_FLAG = 'QWEN_FAST_VERIFY_T2_SKIP'
GATE_PROFILE_ENV = 'QWEN_C2_GATE_PROFILE'
IMAGE_T2_DEFAULT = '1'

ENGAGED = '[PINDIAG] tp4 octo glue8 engaged'
FALLBACK = '[PINDIAG] tp4 octo glue8 fell back'
SERVED = '[PINDIAG] tp4 vglue octo 8-row served path'          # tp4_vglue.EIGHT_ROW_SERVED (a literal so this module imports nothing of the levers)
VGLUE_ENGAGED = '[PINDIAG] tp4 vglue engaged'
SITES = ('split', 'merge', 'block_conv', 'windows')
GDN_LAYERS = 48                                                # verify_trace_t2.GDN_LAYERS: one count per GDN layer per captured trace
COUNT_NAMES = dict((site, 'glue8_%s' % site) for site in SITES)

USERS = 8
ROWS = 8

# What each native site needs at run time, by basename. Both image copy lists and the CPU allowlist must name every entry (test_octo_glue8 holds the list; the closure test of the
# image reads the lists).
RUNTIME_FILES = ('octo_glue8.py', 'gdn_rows_dma8_tp.py', 'gdn_rows_dma8_tp.cpp', 'gdn_block_conv8_tp.py', 'gdn_conv_windows_packed8.py',
                 'gdn_conv_windows_packed8.cpp')


def _read(environ=None):
    value = (os.environ if environ is None else environ).get(FLAG)
    if value is None or value == '0':
        return False
    if value == '1':
        return True
    raise ValueError('%s must be 0 or 1, got %r' % (FLAG, value))


def enabled(environ=None):
    """QWEN_FAST_OCTO_GLUE8=1. Strict; raises at two cards."""
    on = _read(environ)
    if on and tp_shapes.chip_count(os.environ if environ is None else environ) == tp_shapes.PAIR:
        raise ValueError('%s is a TP4 lever: it needs QWEN_FAST_TP=4, this process serves the pair' % FLAG)
    return on


def is_octo_spans(spans):
    """Whether these packed segments are the octo block's: eight contiguous eight-row users from row 0."""
    try:
        spans = [tuple(span) for span in spans]
    except TypeError:
        return False
    return len(spans) == USERS and spans == [(user * ROWS, (user + 1) * ROWS) for user in range(USERS)]


def sites_wanted(environ=None):
    """The sites the flag makes native, given which levers are on: [] when the flag is off."""
    environ = os.environ if environ is None else environ
    if not enabled(environ):
        return []
    wanted = []
    if environ.get(V2_FLAG) == '1':
        wanted += ['split', 'merge']
    # QWEN_FAST_VERIFY_T2 is an IMAGE default (docker/qwen-c2-serving.Dockerfile ENV), not a profile key: a profile env that does not name it runs with it on, so absent means on here.
    t2 = environ.get(T2_FLAG, IMAGE_T2_DEFAULT) == '1' and 'windows' not in [name.strip() for name in environ.get(T2_SKIP_FLAG, '').split(',')]
    if environ.get(V1_FLAG) == '1' and environ.get(V2_FLAG) == '1' and t2:
        wanted.append('block_conv')
    if t2:
        wanted.append('windows')
    return [site for site in SITES if site in wanted]


def refusal(environ=None):
    """Why the flag cannot serve this process (a configuration reason), or None: the octo shape, a gate run of a gate-only profile and at least one lever to make native."""
    environ = os.environ if environ is None else environ
    if not enabled(environ):
        return None
    if environ.get(OCTO_FLAG, 'off') in ('', 'off'):
        return '%s needs %s=live|alternate (the octo block is the only eight-row block)' % (FLAG, OCTO_FLAG)
    if environ.get(GATE_PROFILE_ENV) != '1':
        return '%s is gate only: it needs a gate run of a gate-only profile (%s=1), as %s does' % (FLAG, GATE_PROFILE_ENV, OCTO_FLAG)
    if not sites_wanted(environ):
        return '%s needs at least one of %s=1, %s=1 beside %s=1 (nothing to make native)' % (FLAG, V2_FLAG, V1_FLAG, T2_FLAG)
    return None


def log_line(message):
    """One line into the server log: loguru where it exists, stdout otherwise. Never raises."""
    try:
        try:
            from loguru import logger
        except ImportError:
            print(message, flush=True)
        else:
            logger.info('{}', message)
    except BaseException:
        pass


_LOGGED = set()


def log_once(message, key):
    if key in _LOGGED:
        return False
    _LOGGED.add(key)
    log_line(message)
    return True


def engaged_line(site):
    return '%s site=%s users=%d rows=%d' % (ENGAGED, site, USERS, ROWS)


def fallback_line(site, reason):
    return '%s site=%s reason=%s' % (FALLBACK, site, str(reason).replace('\n', ' ')[:200])


def note_engaged(site):
    """A native launch of `site` ran (inside a capture or an eager forward): the count tp4_vglue's per-capture line reports (one per GDN layer) and the once-per-process marker."""
    import tp4_vglue

    if site not in SITES:
        raise ValueError('Unknown glue8 site %r' % (site,))
    tp4_vglue.note(COUNT_NAMES[site])
    log_once(engaged_line(site), ('engaged', site))


def note_fallback(site, reason):
    """A native site declined and the served path ran: the count tp4_vglue reports and a FALLBACK line (once per site and reason)."""
    import tp4_vglue

    tp4_vglue.note('glue8_%s_fallback' % site)
    log_once(fallback_line(site, reason), ('fallback', site, str(reason)))


_ENGAGED_PATTERN = re.compile(r'\[PINDIAG\] tp4 octo glue8 engaged site=(\w+) users=(\d+) rows=(\d+)')
_COUNT_PATTERN = re.compile(r'\bglue8_(split|merge|block_conv|windows)=(\d+)')


def glue8_problems(text, environ=None):
    """The smoke rule: why this server log is not a clean QWEN_FAST_OCTO_GLUE8 arm (or a clean flag-off one), as a list of strings.

    Flag off: not one glue8 line, and no glue8_* count in a vglue engaged line (the control arm carries nothing of the lever).
    Flag on, for every site sites_wanted() names (the levers that are on): an engaged line at users=8 rows=8; a vglue engaged line of the octo block's capture whose count for the
    site equals the GDN layer count (48) - a partial count is a layer that declined; no octo-glue8 'fell back' line; and no tp4_vglue.EIGHT_ROW_SERVED line for that site (the served
    path must not have run there). A site whose lever is off is not required and its served line is the known state."""
    text = text or ''
    lines = text.splitlines()
    environ = os.environ if environ is None else environ
    problems = []
    marked = [line.strip()[:200] for line in lines if 'tp4 octo glue8' in line]
    counted = [line for line in lines if VGLUE_ENGAGED in line and _COUNT_PATTERN.search(line)]
    if not enabled(environ):
        if marked:
            problems.append('%s is off but the log carries glue8 lines: %s' % (FLAG, marked[0]))
        if counted:
            problems.append('%s is off but a vglue engaged line carries glue8 counts: %s' % (FLAG, counted[0].strip()[:200]))
        return problems
    reason = refusal(environ)
    if reason is not None:
        problems.append(reason)
        return problems
    engaged = {}
    for line in lines:
        match = _ENGAGED_PATTERN.search(line)
        if match:
            engaged.setdefault(match.group(1), []).append((int(match.group(2)), int(match.group(3))))
    counts = {}
    for line in counted:
        for site, value in _COUNT_PATTERN.findall(line):
            counts[site] = max(counts.get(site, 0), int(value))
    for line in lines:
        if FALLBACK in line:
            problems.append('an octo glue8 site fell back (its served path ran, it saved nothing): %s' % line.strip()[:200])
    for site in sites_wanted(environ):
        if (USERS, ROWS) not in engaged.get(site, []):
            problems.append('no "%s site=%s users=%d rows=%d" line: the native %s launch never ran' % (ENGAGED, site, USERS, ROWS, site))
        if counts.get(site, 0) != GDN_LAYERS:
            problems.append('site=%s engaged on %d of %d GDN layers in the octo block capture (vglue engaged line glue8_%s=)' % (site, counts.get(site, 0), GDN_LAYERS, site))
        for line in lines:
            if SERVED in line and re.search(r'site=%s\b' % site, line):
                problems.append('the served 8-row path ran at site=%s although it is native under %s: %s' % (site, FLAG, line.strip()[:200]))
                break
    return problems
