"""Page-table widths the four-card ordered K/V writers admit: ordered_cache.page_width_admitted, plus 4,096 behind evidence.

ordered_cache.py (hash-pinned writer evidence, never edited) admits up to the original 1,024-entry envelope and exactly one
wide width, 2,052 (a 131,328-token window at 64-token pages). A 262,144-token window is 4,096 entries. The kernels have no
page-table limit but do no bounds check either, so 4,096 is admitted only on a one-card hardware record of its own (E1,
ordered_writer_tp4_card_test.py): both of the four-card writers (the packed block's chained launch and the engines' 32-row-tile
launch) writing the complete cache exactly at widths 2,052 (the control) and 4,096, eager and replayed, with the page table
rewritten in place between replays.

admitted(width) is therefore:
  - the pinned answer, always (so every width that was admitted stays admitted, byte for byte);
  - width 4,096 ONLY when this process serves four cards (QWEN_FAST_TP=4) AND ordered_writer_evidence_tp4.json is a PASS at
    ORDERED_WRITER_EVIDENCE_TP4_SHA256 (the recorder writes the file and the pin together) for the live ordered_cache.py bytes.
    4,100 and 4,104 exceed max_position_embeddings (262,144 tokens) and are never valid; no other width is ever admitted.
At the pair the pinned answer is the whole answer. The record ships as a PENDING skeleton, so 4,096 is refused until the card
window has run.

Stdlib only, importable on py 3.7. Nothing here is sha256-pinned.
"""

import hashlib
import json
import os
from pathlib import Path

import ordered_cache
import tp_shapes

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / 'ordered_writer_evidence_tp4.json'
EVIDENCE_SCHEMA = 'qwen-c2-ordered-writer-evidence/1'
# The sha256 of ordered_writer_evidence_tp4.json as reviewed (record_ordered_writer_evidence_tp4.py rewrites it with the file).
ORDERED_WRITER_EVIDENCE_TP4_SHA256 = 'eb8e3a699cbef31b11a4cc58fa9e31ae115e523cdf5bbcac05cde8f67f3e844c'
WIDE_WIDTH_TP4 = 4096
CONTROL_WIDTH = 2052
WIDTHS_RECORDED = (CONTROL_WIDTH, WIDE_WIDTH_TP4)
WRITERS = ('chained64', 'tiles32')
CHIPS = '1of4'
SEEDS = (0, 1, 2)
SECTIONS = ('eager', 'replay_changed', 'replay_unchanged', 'complete_cache')
ENTRIES = (0, 1, 1023, 1024, 2047, 2048, 2051, 2052, 3071, 4094, 4095)
ANCHOR_FIRST, ANCHOR_LAST = 262080, 262111
VERDICT_SCOPE = 'full'


def _hex64(value):
    return isinstance(value, str) and len(value) == 64 and all(char in '0123456789abcdef' for char in value)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def evidence_problems(evidence, sources_root=HERE):
    """Every reason `evidence` does not qualify width 4,096 for the live ordered_cache.py; [] when it does."""
    if not isinstance(evidence, dict) or evidence.get('schema') != EVIDENCE_SCHEMA:
        return ['the record is not a %s record' % EVIDENCE_SCHEMA]
    problems = []
    status = evidence.get('status')
    if status != 'PASS':
        return ['status %s, not PASS' % (status,)]
    if type(evidence.get('run')) is not int or evidence['run'] <= 0:
        problems.append('no run id')
    if evidence.get('failures') != 0:
        problems.append('failures %r, not 0' % (evidence.get('failures'),))
    if evidence.get('scope') != VERDICT_SCOPE:
        problems.append('scope %s, not full' % (evidence.get('scope'),))
    if evidence.get('chips') != CHIPS:
        problems.append('chips %s, not %s' % (evidence.get('chips'), CHIPS))
    if evidence.get('kv_heads') != 1:
        problems.append('kv_heads %r, not 1' % (evidence.get('kv_heads'),))
    for key, wanted in (('widths', WIDTHS_RECORDED), ('writers', WRITERS), ('sections', SECTIONS), ('seeds', SEEDS)):
        found = evidence.get(key)
        missing = [item for item in wanted if not isinstance(found, list) or item not in found]
        if missing:
            problems.append('%s lack %s' % (key, missing))
    entries = evidence.get('entries')
    missing = [item for item in ENTRIES if not isinstance(entries, list) or item not in entries]
    if missing:
        problems.append('page-table entries lack %s' % missing)
    anchors = evidence.get('anchor_positions')
    if anchors != [ANCHOR_FIRST, ANCHOR_LAST]:
        problems.append('anchor positions %r, not %d-%d' % (anchors, ANCHOR_FIRST, ANCHOR_LAST))
    counts = evidence.get('counts') or {}
    for key in ('checks', 'exact'):
        if type(counts.get(key)) is not int or counts[key] <= 0:
            problems.append('counts.%s %r is not a positive count' % (key, counts.get(key)))
    if counts.get('checks') != counts.get('exact'):
        problems.append('counts: %r of %r checks exact' % (counts.get('exact'), counts.get('checks')))
    recorded = (evidence.get('sources') or {}).get('ordered_cache.py')
    path = Path(sources_root) / 'ordered_cache.py'
    live = sha256_file(path) if path.is_file() else None
    if not _hex64(recorded):
        problems.append('sources: no sha256 recorded for ordered_cache.py')
    elif recorded != live:
        problems.append('sources: ordered_cache.py is %s, but the record qualified %s' % ((live or 'absent')[:16], recorded[:16]))
    return problems


def evidence_state(path=None, expected=None, sources_root=HERE):
    """(ok, problems): the record at its pin, parsed, with evidence_problems empty."""
    path = Path(EVIDENCE if path is None else path)
    expected = ORDERED_WRITER_EVIDENCE_TP4_SHA256 if expected is None else expected
    if not path.is_file():
        return False, ['%s is missing' % path.name]
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected:
        return False, ['%s is %s, not the reviewed %s' % (path.name, digest[:16], expected[:16])]
    try:
        evidence = json.loads(payload.decode('utf-8'))
    except ValueError as error:
        return False, ['%s does not parse: %s' % (path.name, str(error)[:80])]
    problems = evidence_problems(evidence, sources_root)
    return not problems, problems


# THE 262k EVIDENCE WAIVER (gate only). QWEN_FAST_262K_EVIDENCE_WAIVER=1 lets width 4,096 (here) and the admission at capacity 262,144
# (packed_any_admission) proceed without their evidence records, ONLY in a gate run of a gate-only profile: QWEN_C2_GATE=1 (the contract's
# boot condition for gate_only profiles) AND the profile's own marker QWEN_C2_GATE_PROFILE=1 (set by no traffic profile; the same pair
# packed_any_admission.unqualified_allowed reads). Anywhere else the flag is refused by name. Unset (or 0) every path is what it was.
WAIVER_ENV = 'QWEN_FAST_262K_EVIDENCE_WAIVER'
WAIVER_GATE_ENV = 'QWEN_C2_GATE'
WAIVER_GATE_PROFILE_ENV = 'QWEN_C2_GATE_PROFILE'
WAIVER_MARKER = '262k evidence WAIVED (gate-only)'
_WAIVER_LOGGED = []


class WaiverRefused(ValueError):
    """QWEN_FAST_262K_EVIDENCE_WAIVER is set in a process that is not a gate run of a gate-only profile (or to a value other than 0/1)."""


def waiver_active(environ=None):
    """False when the waiver is unset (or 0); True when it is 1 in a gate run of a gate-only profile; WaiverRefused otherwise."""
    environ = os.environ if environ is None else environ
    value = environ.get(WAIVER_ENV)
    if value in (None, '', '0'):
        return False
    if value != '1':
        raise WaiverRefused('%s=%r is not 0 or 1' % (WAIVER_ENV, value))
    if environ.get(WAIVER_GATE_ENV) != '1' or environ.get(WAIVER_GATE_PROFILE_ENV) != '1':
        raise WaiverRefused('%s=1 is refused: the 262k evidence waiver exists only in a gate run of a gate-only profile (%s=1 and %s=1), '
                            'never for traffic' % (WAIVER_ENV, WAIVER_GATE_ENV, WAIVER_GATE_PROFILE_ENV))
    return True


def _log(template, *values):
    text = template.format(*values)
    try:
        from loguru import logger
    except ImportError:
        print(text, flush=True)
        return
    logger.warning('{}', text)


def log_waiver_once(what, log=None):
    """The loud line, once per process: '262k evidence WAIVED (gate-only): <what>'. Returns whether it logged now."""
    if _WAIVER_LOGGED:
        return False
    _WAIVER_LOGGED.append(what)
    (_log if log is None else log)('{}: {}; every result of this process is UNQUALIFIED (no card evidence record stands behind width 4096 '
                                   'or capacity 262144)', WAIVER_MARKER, what)
    return True


def admitted(width, environ=None, *, path=None, expected=None, sources_root=HERE):
    """ordered_cache.page_width_admitted(width), or width 4,096 at four cards on a passing E1 record (or, in a gate run of a
    gate-only profile with QWEN_FAST_262K_EVIDENCE_WAIVER=1, without one: logged once, loudly)."""
    if ordered_cache.page_width_admitted(width):
        return True
    if type(width) is not int or width != WIDE_WIDTH_TP4 or width % 4:
        return False
    if tp_shapes.chip_count(environ) != 4:
        return False
    waived = waiver_active(environ)
    if evidence_state(path, expected, sources_root)[0]:
        return True
    if waived:
        log_waiver_once('page-table width %d admitted without ordered_writer_evidence_tp4.json (E1)' % WIDE_WIDTH_TP4)
        return True
    return False
