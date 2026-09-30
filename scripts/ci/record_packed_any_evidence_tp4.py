"""Record the four-card packed-any evidence: the reports of the one-card windows into packed_any_evidence_tp4.json.

    py -3.11 -B scripts/ci/record_packed_any_evidence_tp4.py \
        --cb1 <reports>/card-EV-F1.json --cb1-run <run id> --cb1-tag <experiment tag> \
            --cb1-watcher <reports>/card-EV-W1.json --cb1-watcher-run <run id> --cb1-watcher-tag <tag> \
        --cb2a <reports>/card-EV-F2.json --cb2a-run ... --cb2a-tag ... [--cb2a-watcher ...] \
        --cb2b <reports>/reader-EV-F3.json --cb2b-run ... --cb2b-tag ... --cb2b-commit <sha> --cb2b-image <tag name> \
            [--cb2b-watcher ...] \
        --extent-audit-run <run id> --extent-audit-tag <tag>

It is the four-card twin of the recorder the pair's CB2b was written with (packed_any_admission.py, RECORDING CB2b): every value
is taken from the report JSON the harness wrote (optimisation/ttnn-op/k64j/k64j_card_b.py for CB1 and CB2a, at --kv-heads 1;
extent_reader_card_b.py for CB2b, at width 4) and the verdict line inside it, and each report is checked against the admission's
own constants BEFORE anything is written: the K64j binary and kernels, the pass verdict and scope, no failures, every liveness
control moved, the one-KV-head shape (kv_heads=1, the 0x23 flags, chips=1of4), the reader's sha256 against the live file. A
report that does not qualify is an error, never a record: the section stays PENDING.

Writes, in one go: the evidence file (LF, one-space indent, the keys in the skeleton's order) and the EVIDENCE_TP4_SHA256 pin in
packed_any_admission.py, so the two cannot drift. Run the CPU tests and commit both files by explicit path (scripts/ci/__pycache__
is tracked: never `git add` a directory or -A).

PUBLIC-REPO HYGIENE: the record holds run ids, experiment tags, commit shas, report file names under cardm/ and content hashes.
Never an image digest or registry, a host path, an address or a board id: the record is checked for them before it is written, and
the image is recorded by its tag name only. The report itself stays with the run's artifacts.

Stdlib only, Python 3.7 syntax.
"""
import argparse
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import packed_any_admission as admission  # noqa: E402

READER_TP = 'extent_attention_replay_tp.py'
# The reader twin's pinned siblings, served from the image (provenance only: the admission does not read them).
PINNED_SIBLINGS = ('attention_fold_dma_tp.cpp', 'attention_fold_dma_tp.py', 'attention_mask_replay_tp.cpp',
                   'attention_mask_replay_tp.py')
CB1_SECTIONS = ('N', 'X', 'M', 'K', 'L', 'T')
CB2A_SECTIONS = ('K2', 'X7', 'Z')
# Report comparison kinds behind each count (k64j_card_b.verdict_line's own groups).
CB1_KINDS = (('extent', ('extent_vs_reference',)), ('mixed', ('mixed_vs_reference',)),
             ('share_slot0', ('share_slot0', 'share_skip_slot0_wins')), ('trace', ('trace_vs_eager', 'trace_vs_reference')),
             ('skip', ('skip_live', 'trace_skip_live')))
X7_KINDS = ('x7_narrow_vs_wide', 'x7_extent_vs_wide')
Z_KINDS = ('z_trace_vs_eager', 'z_trace_vs_reference')
SERVED_FLAGS = '0x23'
SCHEMA = admission.EVIDENCE_SCHEMA
# Order of the keys the file is written in (the skeleton's).
TOP_KEYS = ('schema', 'what', 'provenance', 'binary', 'kernels', 'sources', 'sections')
# What must never reach the record (the file is in a public repository).
HYGIENE = re.compile(r'(?:^|[\s=(\'"])/(?:home|root|Users|opt|dev|mnt|models|tmp|var|srv|etc)/|[A-Za-z]:\\|blackhole-[A-Za-z0-9]{8,}|'
                     r'thatch\.local|zot\.|sha256:[0-9a-f]{16}|@sha256|\b\d{1,3}(?:\.\d{1,3}){3}\b|/dev/tenstorrent|'
                     r'[0-9a-f]{64}@|[Pp]assword|token=')


class RecordError(ValueError):
    """A report that cannot be recorded: the message names every reason, one per line."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__('\n'.join(self.problems))


# ---------------------------------------------------------------------------------------------------------------------
# Reading a report.
# ---------------------------------------------------------------------------------------------------------------------

def read_report(path):
    """(report, sha256 of its bytes). The harness writes it with json.dumps(indent=2); a checkpoint (in_progress) is refused."""
    with open(path, 'rb') as handle:
        payload = handle.read()
    report = json.loads(payload.decode('utf-8'))
    if not isinstance(report, dict):
        raise RecordError(['%s is not a report object' % os.path.basename(path)])
    return report, hashlib.sha256(payload).hexdigest()


def line_words(line):
    """'K64J_CARD verdict=PASS extent=2100/2100 ...' -> {'verdict': 'PASS', 'extent': [2100, 2100], ...}; a word that is not
    key=value is skipped, key=a/b becomes [a, b]."""
    words = {}
    for word in str(line or '').split()[1:]:
        key, sep, value = word.partition('=')
        if not sep:
            continue
        match = re.fullmatch(r'(\d+)/(\d+)', value)
        words[key] = [int(match.group(1)), int(match.group(2))] if match else value
    return words


def tally_count(report, kinds):
    """[equal, runs] over the report's tally of comparison kinds (what verdict_line prints as a/b)."""
    tally = report.get('tally') or {}
    equal = sum(int((tally.get(kind) or {}).get('equal', 0)) for kind in kinds)
    runs = sum(int((tally.get(kind) or {}).get('runs', 0)) for kind in kinds)
    return [equal, runs]


def report_name(path):
    """cardm/<file name>: the run's artifact the record cites, never a host path."""
    return 'cardm/' + os.path.basename(path)


def common_problems(label, report, words, problems):
    """What every recordable report holds: the pass verdict, no failure, no error, nothing in progress."""
    decision = report.get('decision') or {}
    if report.get('in_progress'):
        problems.append('%s: a checkpoint (in progress %s), not the final report' % (label, report['in_progress']))
    if decision.get('verdict') != 'PASS' or words.get('verdict') != 'PASS':
        problems.append('%s: verdict %s (line %s), not PASS' % (label, decision.get('verdict'), words.get('verdict')))
    if report.get('passed') is not True:
        problems.append('%s: passed is %r' % (label, report.get('passed')))
    if report.get('failures'):
        problems.append('%s: %d failures (%s)' % (label, len(report['failures']), str(report['failures'][0])[:100]))
    if report.get('error'):
        problems.append('%s: error %s' % (label, str(report['error'])[:100]))
    if decision.get('decisive_differing'):
        problems.append('%s: %s decisive comparisons differ' % (label, decision['decisive_differing']))
    dead = [entry for entry in report.get('liveness') or () if not entry.get('live')]
    if dead:
        problems.append('%s: %d liveness controls did not move' % (label, len(dead)))
    if (report.get('deadline') or {}).get('skipped'):
        problems.append('%s: the deadline cut %d section runs' % (label, len(report['deadline']['skipped'])))


def binary_problems(label, report, problems, kernels=True):
    """The loaded _ttnncpp.so is the admission's K64j and the kernels the harness found are its four."""
    binary = report.get('binary') or {}
    if binary.get('sha256') != admission.K64J_TTNNCPP_SHA256:
        problems.append('%s: the loaded binary is %s, not K64j %s' % (label, str(binary.get('sha256'))[:16],
                                                                        admission.K64J_TTNNCPP_SHA256[:16]))
    if binary and binary.get('k64j') is False:
        problems.append('%s: the loaded binary is not K64j (stage %s)' % (label, binary.get('stage')))
    if kernels:
        found = (report.get('kernels') or {}).get('found') or {}
        for name, expected in admission.K64J_KERNELS.items():
            if found.get(name) != expected:
                problems.append('%s: kernel %s is %s, not K64j\'s %s' % (label, name, str(found.get(name))[:16], expected[:16]))


def kv_problems(label, report, words, problems):
    """One KV head per chip: the harness's own kv_heads word (E1's --kv-heads 1), never the pair's two."""
    heads = report.get('kv_heads', words.get('kv_heads'))
    if str(heads) != '1':
        problems.append('%s: kv_heads is %r, not 1 (the pair\'s report is not four-card evidence)' % (label, heads))


def watcher_entry(run, tag, path):
    """The watcher pass a full run cites: run, tag, status, the report's scope/verdict and its content hash."""
    if run is None:
        return None
    entry = dict(run=int(run), tag=tag, status='PASS')
    if path:
        report, digest = read_report(path)
        words = line_words(report.get('verdict_line'))
        problems = []
        common_problems('watcher report', report, words, problems)
        if problems:
            raise RecordError(problems)
        scope = (report.get('decision') or {}).get('scope')
        if scope:
            entry['scope'] = scope
        if (report.get('decision') or {}).get('k2') not in (None, 'not_run'):
            entry['k2_verdict'] = report['decision']['k2']
        entry['report_sha256'] = digest
    return entry


# ---------------------------------------------------------------------------------------------------------------------
# The three sections.
# ---------------------------------------------------------------------------------------------------------------------

def combos_of(report):
    """['G4B3:0x21', 'G4B3:0x23', 'G8B2:0x21', 'G8B2:0x23'] -> [{'geometry': 'G4B3', 'flags': ['0x21', '0x23']}, ...]."""
    grouped = []
    for name in report.get('combos') or ():
        geometry, _, flags = str(name).partition(':')
        for entry in grouped:
            if entry['geometry'] == geometry:
                entry['flags'].append(flags.lower())
                break
        else:
            grouped.append(dict(geometry=geometry, flags=[flags.lower()]))
    return grouped


def build_cb1(report, digest, path, run, tag, watcher=None):
    """sections.CB1 from the final report of the card harness (--kv-heads 1, the default sections), or RecordError."""
    problems = []
    words = line_words(report.get('verdict_line'))
    common_problems('CB1', report, words, problems)
    binary_problems('CB1', report, problems)
    kv_problems('CB1', report, words, problems)
    missing = [name for name in CB1_SECTIONS if name not in (report.get('sections') or ())]
    if missing:
        problems.append('CB1: sections %s did not run (asked %s)' % (','.join(missing), report.get('sections')))
    seeds = sorted(report.get('seeds') or ())
    if not set(admission.SEEDS) <= set(seeds):
        problems.append('CB1: seeds %s lack %s' % (seeds, sorted(set(admission.SEEDS) - set(seeds))))
    extents = sorted(report.get('extents') or ())
    if not set(admission.K1_EXTENTS) <= set(extents):
        problems.append('CB1: extents %s lack K1\'s %s' % (extents, sorted(set(admission.K1_EXTENTS) - set(extents))))
    combos = combos_of(report)
    if not any(entry['geometry'] == 'G8B2' and admission.CB1_FLAG_TP4 in entry['flags'] for entry in combos):
        problems.append('CB1: no G8B2 %s combo (combos %s)' % (admission.CB1_FLAG_TP4, report.get('combos')))
    if any(flag in ('0x27', '0x2f') for entry in combos for flag in entry['flags']):
        problems.append('CB1: a combo with the q-slice (0x27/0x2f), which needs a second KV head, is in a one-head record')
    counts = {}
    for key, kinds in CB1_KINDS:
        counts[key] = tally_count(report, kinds)
        if counts[key][1] == 0 or counts[key][0] != counts[key][1]:
            problems.append('CB1: %s %s is not a full pass' % (key, counts[key]))
        if key in words and words[key] != counts[key]:
            problems.append('CB1: %s is %s in the tally but %s on the verdict line' % (key, counts[key], words[key]))
    if run is None or not tag:
        problems.append('CB1: --cb1-run and --cb1-tag are required')
    if problems:
        raise RecordError(problems)
    section = dict(
        status='PASS',
        what='K64j K1/K3 at one KV head per chip (6 folded rows per token): the runtime extent (0x20) at cur_pos E - 1 equals the '
             'compile-time call at capacity E, the mixed, share-slot0, trace and skip families, and the refusal of the q-slice',
        run=int(run), tag=tag, card='M', kv_heads=1,
        harness='optimisation/ttnn-op/k64j/k64j_card_b.py via run_card_b.sh, --kv-heads 1 --seeds %s'
                % ','.join(str(seed) for seed in seeds),
        seeds=seeds, extents=extents, combos=combos,
        comparisons=len(report.get('comparisons') or ()), failures=0,
        families=int(report.get('trace_families_distinct', words.get('families', 0)) or 0), counts=counts)
    if report.get('skip_written'):
        section['skipped_rows'] = str(report['skip_written'])
    if watcher:
        section['watcher_pass'] = watcher
    section['report'] = report_name(path)
    section['report_sha256'] = digest
    return section


def build_cb2a(report, digest, path, run, tag, watcher=None):
    """sections.CB2a from the final report of the card harness (--sections K2,X7,Z at --kv-heads 1), or RecordError."""
    problems = []
    words = line_words(report.get('verdict_line'))
    decision = report.get('decision') or {}
    common_problems('CB2a', report, words, problems)
    binary_problems('CB2a', report, problems)
    kv_problems('CB2a', report, words, problems)
    missing = [name for name in CB2A_SECTIONS if name not in (report.get('sections') or ())]
    if missing:
        problems.append('CB2a: sections %s did not run' % ','.join(missing))
    seeds = sorted(report.get('seeds') or ())
    variants = sorted(report.get('variants') or ())
    if not set(admission.SEEDS) <= set(seeds):
        problems.append('CB2a: seeds %s lack %s' % (seeds, sorted(set(admission.SEEDS) - set(seeds))))
    if not set(admission.CB2A_VARIANTS) <= set(variants):
        problems.append('CB2a: variants %s lack %s' % (variants, sorted(set(admission.CB2A_VARIANTS) - set(variants))))
    coverage = decision.get('k2_coverage') or {}
    if decision.get('k2') != 'PASS' or not coverage.get('full'):
        problems.append('CB2a: K2 verdict %s, coverage %s (only a full PASS sets the exactness policy)' % (decision.get('k2'),
                                                                                                          coverage))
    tickets = [int(coverage.get('covered', 0)), int(coverage.get('design', 0))]
    if tickets != [admission.CB2A_K2_TICKETS] * 2:
        problems.append('CB2a: K2 tickets %s, not %d of %d' % (tickets, admission.CB2A_K2_TICKETS, admission.CB2A_K2_TICKETS))
    rows = report.get('k2_rows') or {}
    k2_rows = [int(rows.get('equal', 0)), int(rows.get('compared', 0))]
    if not k2_rows[1] or k2_rows[0] != k2_rows[1]:
        problems.append('CB2a: K2 rows %s are not all bitwise equal' % k2_rows)
    floor = [int(rows.get('floor_differing', 0)), int(rows.get('floor', 0))]
    x7 = tally_count(report, X7_KINDS)
    if x7[1] < admission.X7_FLOOR or x7[0] != x7[1]:
        problems.append('CB2a: X7 %s, not a full pass of at least %d' % (x7, admission.X7_FLOOR))
    z_passed = tally_count(report, Z_KINDS)
    families = sorted(report.get('z_families') or ())
    if z_passed[1] < admission.Z_FLOOR or z_passed[0] != z_passed[1]:
        problems.append('CB2a: Z %s, not a full pass of at least %d' % (z_passed, admission.Z_FLOOR))
    if set(admission.Z_FAMILIES) - set(families):
        problems.append('CB2a: Z families lack %s' % sorted(set(admission.Z_FAMILIES) - set(families)))
    for key, got in (('x7', x7), ('z', z_passed), ('k2_rows', k2_rows)):
        if key in words and words[key] != got:
            problems.append('CB2a: %s is %s in the report but %s on the verdict line' % (key, got, words[key]))
    liveness = report.get('liveness') or []
    live = [sum(1 for entry in liveness if entry.get('live')), len(liveness)]
    if run is None or not tag:
        problems.append('CB2a: --cb2a-run and --cb2a-tag are required')
    if problems:
        raise RecordError(problems)
    section = dict(
        status='PASS',
        what='K2 (native causal decode at 6 local heads on one KV head per chip == the extent path, bitwise; it sets the exactness '
             'policy), X7 (0x3 narrow == 0x3 wide with the real -inf tail) and Z (the 15 stale-writer families in one trace) at '
             'one KV head',
        run=int(run), tag=tag, card='M', kv_heads=1,
        harness='optimisation/ttnn-op/k64j/k64j_card_b.py --kv-heads 1 --sections K2,X7,Z --seeds %s --variants %s --no-timing '
                '(the cardm action)' % (','.join(str(seed) for seed in seeds), ','.join(variants)),
        seeds=seeds, variants=variants,
        k2=dict(verdict='PASS', tickets=tickets, rows=k2_rows, floor_rows_differing=floor),
        x7=x7, z=dict(passed=z_passed, families=families), liveness=live,
        comparisons=len(report.get('comparisons') or ()), failures=0)
    for key in ('local_heads', 'kv_heads_local'):
        if key in report:
            section[key] = report[key]
    if watcher:
        section['watcher_pass'] = watcher
    section['report'] = report_name(path)
    section['report_sha256'] = digest
    return section


def reader_sha(root=HERE):
    with open(os.path.join(root, READER_TP), 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def build_cb2b(report, digest, path, run, tag, commit, image, watcher=None, root=HERE):
    """sections.CB2b from the final report of the four-card extent reader (full scope, chips=1of4), or RecordError."""
    problems = []
    line = report.get('verdict_line') or ''
    words = line_words(line)
    decision = report.get('decision') or {}
    common_problems('CB2b', report, words, problems)
    # The reader harness checks the mounted kernels with the card harness's own check (k64j_card_b.check_kernels): the
    # reader's evidence is K64j's only if its kernels were the four the admission names, as CB1's and CB2a's are.
    binary_problems('CB2b', report, problems)
    if decision.get('scope') != admission.CB2B_SCOPE or words.get('scope') != admission.CB2B_SCOPE:
        problems.append('CB2b: scope %s (line %s), not full: %s' % (decision.get('scope'), words.get('scope'),
                                                                   decision.get('scope_short')))
    if words.get('chips') not in admission.CB2B_CHIPS_TP4:
        problems.append('CB2b: chips %s on the verdict line, not %s' % (words.get('chips'), '/'.join(admission.CB2B_CHIPS_TP4)))
    if report.get('capacity') != admission.CB2B_CAPACITY:
        problems.append('CB2b: capacity %s, not %d' % (report.get('capacity'), admission.CB2B_CAPACITY))
    served = report.get('served') or {}
    if served.get('flags') != SERVED_FLAGS:
        problems.append('CB2b: the reader served flags %s, not %s (one KV head)' % (served.get('flags'), SERVED_FLAGS))
    seeds = sorted(report.get('seeds_run') or ())
    variants = sorted(report.get('variants_run') or ())
    if not set(admission.CB2B_SEEDS) <= set(seeds):
        problems.append('CB2b: seeds run %s lack %s' % (seeds, sorted(set(admission.CB2B_SEEDS) - set(seeds))))
    if not {'normal', 'peaky'} <= set(variants):
        problems.append('CB2b: variants run %s lack normal or peaky' % variants)
    geometries = {name: sorted(words_) for name, words_ in (report.get('r1_run') or {}).items()}
    families = sorted(report.get('r2_families_replayed') or ())
    idle = sorted(report.get('idle_starts_run') or ())
    live_words = words.get('live')
    need = ('r1', 'r1_reader', 'staging', 'r2', 'r2_trace', 'r4', 'live')
    lacking = [key for key in need if not isinstance(words.get(key), list)]
    if lacking:
        problems.append('CB2b: the verdict line has no count for %s' % ','.join(lacking))
    sha = words.get('extent_sha256')
    live_sha = reader_sha(root)
    if sha != live_sha:
        problems.append('CB2b: the run loaded %s at %s, but %s is %s here (run CB2b on these bytes)'
                        % (READER_TP, str(sha)[:16], READER_TP, live_sha[:16]))
    shas = (report.get('modules') or {}).get('sha256') or {}
    if shas.get(READER_TP) not in (None, sha):
        problems.append('CB2b: the report\'s module sha256 of %s is not the verdict line\'s' % READER_TP)
    pinned = {name: shas.get(name) for name in PINNED_SIBLINGS}
    if any(value is None for value in pinned.values()):
        problems.append('CB2b: the report holds no sha256 of %s' % ','.join(name for name, value in pinned.items() if value is None))
    if not (isinstance(commit, str) and re.fullmatch(r'[0-9a-f]{40}', commit)):
        problems.append('CB2b: --cb2b-commit must be the 40-hex commit the run tested')
    if not (isinstance(image, str) and re.fullmatch(r'[0-9A-Za-z._-]{3,60}', image)):
        problems.append('CB2b: --cb2b-image is the image TAG NAME only (no registry, no digest)')
    if run is None or not tag:
        problems.append('CB2b: --cb2b-run and --cb2b-tag are required')
    if problems:
        raise RecordError(problems)
    r1 = [words['r1'][0] + words['r1_reader'][0], words['r1'][1] + words['r1_reader'][1]]
    r2 = [words['r2'][0] + words['r2_trace'][0], words['r2'][1] + words['r2_trace'][1]]
    counts = dict(R1=r1, S=words['staging'], R2=r2, R4=words['r4'], liveness=live_words)
    for key, got in counts.items():
        if got[0] != got[1] or not got[1]:
            problems.append('CB2b: %s %s is not a full pass' % (key, got))
    if problems:
        raise RecordError(problems)
    section = dict(
        status='PASS',
        what='W10b at four cards: the real extent reader twin over pool-like lent storage through the one-chip-of-four view - R1 (the '
             'pinned mask kernel at capacity 256 == narrow_mask_host, at G8B2, G4B3 and G4B1), S (construction and restage staging: '
             'word, cur_pos and tables read back), R2 (each segment == the 0x3 compile-time call with the wide -inf mask at '
             'capacity E, over more than 50 families), R4 (idle segments at 0/32 on the zero table)',
        run=int(run), tag=tag, commit=commit, card='M', image=image,
        harness='optimisation/ttnn-op/k64j/extent_reader_card_b.py through run_card_b.sh (K64J_HARNESS=extent_reader TP4_WIDTH=4), '
                '--sections R1,S,R2,R4 --seeds %s --variants %s (the cardm action)' % (
                    ','.join(str(seed) for seed in seeds), ','.join(variants)),
        report=report_name(path), report_sha256=digest, verdict_line=line.strip(),
        scope='full', chips=words['chips'], capacity=report['capacity'], seeds=seeds, variants=variants,
        r1_geometries=geometries, r2_families=families, idle_starts=idle, counts=counts,
        comparisons=len(report.get('comparisons') or ()), failures=0, warnings=len(report.get('warnings') or ()),
        phantom_programs=int(words.get('phantom', 0)),
        served=dict(flags=served['flags'], rows=served.get('rows'), batch=served.get('batch'),
                    segments=len(report.get('segments') or ())),
        pinned_modules=pinned, sources={READER_TP: sha})
    if watcher:
        section['watcher_pass'] = watcher
    return section, sha


# ---------------------------------------------------------------------------------------------------------------------
# The file and its pin.
# ---------------------------------------------------------------------------------------------------------------------

def hygiene_problems(value, where='record'):
    """Every string in the record that names a host path, registry, digest, address or board."""
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            found += hygiene_problems(item, '%s.%s' % (where, key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found += hygiene_problems(item, '%s[%d]' % (where, index))
    elif isinstance(value, str) and HYGIENE.search(value):
        found.append('%s: %r carries a host path, registry, digest, address or board id' % (where, value[:80]))
    return found


def dump(evidence):
    """The bytes of the file: LF, one-space indent, the skeleton's key order, a final newline."""
    ordered = {key: evidence[key] for key in TOP_KEYS if key in evidence}
    ordered.update({key: value for key, value in evidence.items() if key not in ordered})
    return (json.dumps(ordered, indent=1, ensure_ascii=False) + '\n').encode('utf-8')


def provenance_text(evidence, audit):
    sections = evidence['sections']
    parts = []
    for name in admission.SECTIONS:
        section = sections.get(name) or {}
        if section.get('status') == 'PASS':
            parts.append('%s from run %s (%s), report %s (sha256 %s)' % (name, section['run'], section['tag'], section['report'],
                                                                      section['report_sha256'][:16]))
    text = 'Recorded by scripts/ci/record_packed_any_evidence_tp4.py, which takes every value from the harness reports and checks ' \
           'each against the admission\'s K64j binary, kernels and reader sha256: ' + '; '.join(parts) + '.'
    for name in admission.SECTIONS:
        watcher = (sections.get(name) or {}).get('watcher_pass')
        if watcher:
            text += ' The watcher pass before %s is run %s (%s).' % (name, watcher['run'], watcher['tag'])
    if audit:
        text += (' The four-card extent audit (QWEN_FAST_EXTENT_AUDIT=1 on chips 0-3 in every S2 gate arm at four cards) ran in run %s (%s).'
                 % audit)
    text += (' The K64j binary and kernels are the pair\'s (K64j takes the head count and geometry as arguments). sources names '
             '%s as serving/tp4-s2 holds it: CB2b ran the checkout\'s copy at these bytes; the admission guard is in '
             'serving_buffer_pool, never in the reader.' % READER_TP)
    return text


WHAT_RECORDED = ('The four-card (QWEN_FAST_TP=4) record of packed_any_admission: the same three sections as packed_any_evidence.json '
                 'at one KV head per chip (K64j flags 0x23) and the reader twin extent_attention_replay_tp.py through the '
                 '1of4 chip view. packed_any_admission.check_evidence reads it at every four-card attach and its sha256 is pinned '
                 'there (EVIDENCE_TP4_SHA256), so a new record changes both in one commit.')


def record(evidence, sections, audit=None):
    """A new evidence dict: `sections` (name -> section dict) replaces the PENDING ones; the reader sha256 joins `sources`."""
    evidence = json.loads(json.dumps(evidence))
    evidence['sections'] = dict(evidence.get('sections') or {})
    for name, section in sections.items():
        evidence['sections'][name] = section
    if all((evidence['sections'].get(name) or {}).get('status') == 'PASS' for name in admission.SECTIONS):
        evidence['what'] = WHAT_RECORDED
    evidence['provenance'] = provenance_text(evidence, audit)
    return evidence


def repin(admission_path, digest):
    """packed_any_admission.py with EVIDENCE_TP4_SHA256 set to `digest`, the bytes otherwise untouched."""
    with open(admission_path, 'rb') as handle:
        text = handle.read().decode('utf-8')
    new, count = re.subn(r"^(EVIDENCE_TP4_SHA256 = )'[0-9a-f]{64}'", r"\1'%s'" % digest, text, flags=re.M)
    if count != 1:
        raise RecordError(['%s has %d EVIDENCE_TP4_SHA256 assignments, not 1' % (os.path.basename(admission_path), count)])
    return new.encode('utf-8')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--evidence', default=str(admission.EVIDENCE_TP4))
    parser.add_argument('--admission', default=os.path.join(HERE, 'packed_any_admission.py'))
    parser.add_argument('--sources-root', default=HERE, help='where extent_attention_replay_tp.py is read (default: here)')
    for name in ('cb1', 'cb2a', 'cb2b'):
        parser.add_argument('--' + name, help='the final report of the %s run' % name.upper())
        parser.add_argument('--%s-run' % name, type=int, help='the workflow run id of the %s job' % name.upper())
        parser.add_argument('--%s-tag' % name, help='the experiment tag the run was pushed under')
        parser.add_argument('--%s-watcher' % name, help='the watcher pass\'s report (optional)')
        parser.add_argument('--%s-watcher-run' % name, type=int)
        parser.add_argument('--%s-watcher-tag' % name)
    parser.add_argument('--cb2b-commit', help='the 40-hex commit the CB2b run tested')
    parser.add_argument('--cb2b-image', help='the image TAG NAME the pinned siblings were served from (no registry, no digest)')
    parser.add_argument('--extent-audit-run', type=int, help='the four-card gate run that audited the extent path on chips 0-3')
    parser.add_argument('--extent-audit-tag')
    parser.add_argument('--dry-run', action='store_true', help='check and print the result; write nothing')
    return parser.parse_args(argv)


def main(argv=None, out=print):
    args = parse_args(argv)
    if not (args.cb1 or args.cb2a or args.cb2b):
        out('nothing to record: name at least one of --cb1, --cb2a, --cb2b')
        return 2
    with open(args.evidence, 'rb') as handle:
        current = json.loads(handle.read().decode('utf-8'))
    sections, sources, problems = {}, {}, []
    for name, section_name, builder in (('cb1', 'CB1', build_cb1), ('cb2a', 'CB2a', build_cb2a)):
        path = getattr(args, name)
        if not path:
            continue
        try:
            report, digest = read_report(path)
            watcher = watcher_entry(getattr(args, name + '_watcher_run'), getattr(args, name + '_watcher_tag'),
                                    getattr(args, name + '_watcher'))
            sections[section_name] = builder(report, digest, path, getattr(args, name + '_run'), getattr(args, name + '_tag'),
                                             watcher)
        except RecordError as error:
            problems += error.problems
    if args.cb2b:
        try:
            report, digest = read_report(args.cb2b)
            watcher = watcher_entry(args.cb2b_watcher_run, args.cb2b_watcher_tag, args.cb2b_watcher)
            sections['CB2b'], sha = build_cb2b(report, digest, args.cb2b, args.cb2b_run, args.cb2b_tag, args.cb2b_commit,
                                               args.cb2b_image, watcher, args.sources_root)
            sources[READER_TP] = sha
        except RecordError as error:
            problems += error.problems
    if problems:
        for problem in problems:
            out('REFUSED: ' + problem)
        return 1
    evidence = record(current, sections, (args.extent_audit_run, args.extent_audit_tag)
                      if args.extent_audit_run and args.extent_audit_tag else None)
    evidence['sources'] = dict(current.get('sources') or {}, **sources)
    dirty = hygiene_problems(evidence)
    if dirty:
        for problem in dirty:
            out('REFUSED: ' + problem)
        return 1
    payload = dump(evidence)
    digest = hashlib.sha256(payload).hexdigest()
    remaining = admission.evidence_problems(evidence, args.sources_root, tp=4)
    out('evidence sha256 %s; sections recorded: %s; %d problems left for the admission%s' % (
        digest, ','.join(sorted(sections)), len(remaining), '' if not remaining else ':'))
    for problem in remaining:
        out('  - ' + problem)
    pinned = repin(args.admission, digest)        # both payloads are made before either file is opened for writing
    if args.dry_run:
        return 0
    with open(args.evidence, 'wb') as handle:
        handle.write(payload)
    with open(args.admission, 'wb') as handle:
        handle.write(pinned)
    out('wrote %s and re-pinned EVIDENCE_TP4_SHA256 in %s' % (args.evidence, args.admission))
    return 0


if __name__ == '__main__':
    sys.exit(main())
