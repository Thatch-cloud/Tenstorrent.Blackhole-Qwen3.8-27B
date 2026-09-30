"""Summarise a gate run's markers across EVERY log in its artifact, not one of them.

Four times in one session I grepped a single artifact file, got zero, and reported a
negative that was false. The worst was run 35688313093, where I reported that the
resumable prefill path had not run:

    m3native-gate-stdout.log   one-shot=0 resumable=0
    gate/server.log            one-shot=1 resumable=1   <- the markers live here

The gate copies a filtered tail into m3native-server-tail.log and the engine's own log
stays in gate/server.log, so which file holds a line depends on which process emitted
it and whether the filter kept it. Guessing that per grep is how the mistake happens.

Usage:
    python3 -B read_gate_markers.py <artifact-directory>

Prints, per marker, the count in each file that has it, then the verdict fields from
m3native-gate.json. Takes no position on what the numbers mean - it exists so the
numbers are the real ones.
"""

import json
import os
import re
import sys


# (label, regex). Ordered: prefill path first, because that is the control that decides
# whether anything downstream is worth reading.
MARKERS = (
    ('prefill: one-shot', r'\[M1\] prefill path: one-shot'),
    ('prefill: RESUMABLE', r'\[M1\] prefill path: resumable prefill_paged_slots_range'),
    ('  resumable detail', r'resumable prefill_paged_slots_range starts=\[[^\]]*\] ends=\[[^\]]*\]'),
    ('native_m3 engaged', r'native_m3 engaged'),
    ('native_attn engaged', r'native_attn engaged: one attn_decode_prep'),
    ('one-in-flight scheduler', r'one-in-flight scheduler installed'),
    # M2 one-in-flight fires per prefill step; m2 alternation fires only when a
    # partial prefill and a running decode coexist. Absent means the policy did
    # not run - v65 mounted it on a class the platform never constructed.
    ('m2 one-in-flight', r'm2 one-in-flight: \S'),
    ('m2 alternation', r'm2 alternation: \S'),
    # Names the request id at every prefill-slot transition. v69's mechanism was
    # ambiguous between a race at the START of a prefill and one at its END, and
    # the artifact could not separate them; this settles it by evidence.
    ('prefill gate', r'prefill gate: held=\S'),
    ('lifecycle delegate', r'lifecycle delegate: cached=\S'),
    ('runtime binary override', r'runtime binary pin overridden'),
    ('CCL links', r'\[CCLLINKS\]'),
    ('readiness timeout', r'readiness exceeded'),
    ('faulthandler dump', r'Timeout \(0:\d\d:\d\d\)!'),
    ('engine fatal', r'EngineCore encountered a fatal error'),
    # Anchored anywhere on the line, not at its start: every engine line is prefixed
    # with '(EngineCore pid=NN) ERROR ... [core.py:NNNN] ', so a start anchor finds
    # nothing. Validating this tool against run 35688313093 is what caught that - the
    # first version reported 0 ValueErrors for a run that died on one.
    ('ValueError raised', r'ValueError: \S'),
    ('ImportError raised', r'ImportError: \S'),
)

VERDICT_FIELDS = ('ready', 'gate_passed', 'fatal', 'users_checked',
                  'native_m3_marker_present', 'native_attn_engaged',
                  'retired_binder_calls_nonzero')


def logs(root):
    for base, _, names in os.walk(root):
        for name in sorted(names):
            if name.endswith('.log') or name.endswith('.txt'):
                yield os.path.join(base, name)


def main(root):
    files = list(logs(root))
    if not files:
        print('no .log or .txt files under %s' % root)
        return 1
    print('%d log files under %s' % (len(files), root))
    print()

    for label, pattern in MARKERS:
        regex = re.compile(pattern, re.MULTILINE)
        hits = []
        for path in files:
            try:
                with open(path, encoding='utf-8', errors='replace') as handle:
                    count = len(regex.findall(handle.read()))
            except OSError:
                continue
            if count:
                hits.append((os.path.basename(path), count))
        total = sum(count for _, count in hits)
        where = ', '.join('%s x%d' % (name, count) for name, count in hits) or '-'
        print('  %-24s %3d   %s' % (label, total, where))

    detail = re.compile(r'resumable prefill_paged_slots_range starts=\[[^\]]*\] ends=\[[^\]]*\]')
    seen = []
    for path in files:
        try:
            with open(path, encoding='utf-8', errors='replace') as handle:
                seen.extend(detail.findall(handle.read()))
        except OSError:
            continue
    if seen:
        print()
        print('  resumable calls, in file order:')
        for line in seen[:12]:
            print('    %s' % line)
        if len(seen) > 12:
            print('    ... and %d more' % (len(seen) - 12))

    for base, _, names in os.walk(root):
        if 'm3native-gate.json' in names:
            with open(os.path.join(base, 'm3native-gate.json'), encoding='utf-8') as handle:
                try:
                    report = json.load(handle)
                except ValueError:
                    print('\n  m3native-gate.json is not valid JSON')
                    return 0
            print()
            print('  verdict:')
            for key in VERDICT_FIELDS:
                if key in report:
                    print('    %-28s %s' % (key, json.dumps(report[key])[:160]))
            profile = report.get('ttft_profile')
            if isinstance(profile, dict):
                print('    %-28s users=%s recorded=%s incomplete=%s ttft=%s' % (
                    'ttft_profile', profile.get('users'), profile.get('ttfts_recorded'),
                    profile.get('incomplete'), profile.get('ttft_s')))
            break
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
