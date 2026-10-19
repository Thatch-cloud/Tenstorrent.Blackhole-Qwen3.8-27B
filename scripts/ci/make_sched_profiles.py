"""The tp4/sched-1 gate-only twins of the approved ship-4 profile: the scheduler-side levers of docs/tp4-sched-1.md, generated from the files they are made of (stdlib only).

    python3 scripts/ci/make_sched_profiles.py --write   regenerate the twins in scripts/ci/qwen_c2_profiles.json
    python3 scripts/ci/make_sched_profiles.py --check   exit 1 if the checked-in twins are not what their parents generate

NEVER hand-merge these entries into the profiles file. Run make_fusion_profiles.py --write, make_octo3_profiles.py --write and make_ship4_profiles.py --write FIRST (the audited twin
copies the ship-4 audited twin, the timed ones the ship candidate), then this. The twins sit right after the ship-4 twins (and tp4/np1-fastlane's, when merged).

WHAT THEY SERVE. Production (the approved ship-4 profile, `c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-fx-traffic-s4`) spends 58% of card wall time in prefill steps, and a returning turn's
engine build is 541 ms (p50) of host-side accounting around a 188 ms rebind. Four independent levers, each default off, each in a timed arm of its own so that the card ABAB names which
moved what; one arm with all of them; one audited arm with all of them on the ship-4 audited twin:

  -sch-lean      THE ADMISSION ARM: the ship candidate + QWEN_FAST_ADMISSION_LEAN=1 (the conv-slice readback after a slot's first adoption only, no 'dram after engine' line for a rebound
                 request, the post-prefill backstop reuses the admission's own DRAM reading when it clears the need by a wide margin, no trace-region view where no trace term applies)
  -sch-spill     THE TIER ARM: + QWEN_PREFIX_HOST_TIER_SPILL_MODE=complete (the host tier spills what a restore displaces instead of dropping all but 512 blocks a flush)
  -sch-pace      THE PACING ARM: + QWEN_FAST_LEVERN_OWED_CAP_MS=1500, QWEN_FAST_LEVERN_SHORT_SHARE=0.65, QWEN_FAST_LEVERN_SHORT_WAIT_S=5 (the decoders' share after a returning turn's step, the
                 cap in ms instead of eight rounds, and the guard that bounds the returning turn's wait)
  -sch-adaptive  THE FEW-DECODER ARM: + QWEN_FAST_LEVERN_ADAPTIVE=1 (a prefill share of 0.9 and a 120 s deadline while at most two decoders are live)
  -sch-all       the four together (the timed arm of the combined ABAB)
  -sch-all-audit THE AUDITED ATTACH ARM: the ship-4 audited twin (-fx-s4-audit: every audit of the candidate, the tier's instruments) + the four together, so every audit runs under the new
                 pacing, the lean admission and the complete spills

The timed arms are the candidate with the gate marker (QWEN_C2_GATE_PROFILE=1) in place of its owner waiver id, as -fx-s4-nogrid13 does, and gate_only; nothing about the engine, the
memory plan or any other flag differs from the parent. The approved traffic profile is NOT changed.
"""

import argparse
import copy
import json
from pathlib import Path
import sys

import make_octo_profiles
import make_ship4_profiles as ship4

PROFILES = Path(__file__).resolve().parent / 'qwen_c2_profiles.json'

NAMESPACE = 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-sch-'
# The twins sit right after the tp4/ship-4 twins and, when tp4/np1-fastlane has been merged, after its two (make_np1_profiles inserts its own right after the ship-4 anchor, so the
# order is ship-4, np1, sched whichever generator runs last and each --check reproduces it).
NP1_PREFIX = 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-fx-s4-np1'
SHIP = ship4.SHIP
AUDIT_PARENT = ship4.AUDIT
WAIVER_ENV = ship4.WAIVER_ENV
WAIVER_FIELD = ship4.WAIVER_FIELD
MARKER_ENV = dict(make_octo_profiles.MARKER_ENV)

LEAN = {'QWEN_FAST_ADMISSION_LEAN': '1'}
SPILL = {'QWEN_PREFIX_HOST_TIER_SPILL_MODE': 'complete'}
PACE = {'QWEN_FAST_LEVERN_OWED_CAP_MS': '1500', 'QWEN_FAST_LEVERN_SHORT_SHARE': '0.65', 'QWEN_FAST_LEVERN_SHORT_WAIT_S': '5'}
ADAPTIVE = {'QWEN_FAST_LEVERN_ADAPTIVE': '1'}
ALL = dict(LEAN, **dict(SPILL, **dict(PACE, **ADAPTIVE)))
# (suffix, flags, parent: 'ship' | 'audit', what it is)
ARMS = (('lean', LEAN, 'ship', 'THE ADMISSION ARM'), ('spill', SPILL, 'ship', 'THE TIER ARM'), ('pace', PACE, 'ship', 'THE PACING ARM'),
        ('adaptive', ADAPTIVE, 'ship', 'THE FEW-DECODER ARM'), ('all', ALL, 'ship', 'THE COMBINED TIMED ARM'), ('all-audit', ALL, 'audit', 'THE AUDITED ATTACH ARM'))
NAMES = tuple(NAMESPACE + suffix for suffix, _flags, _parent, _what in ARMS)
FLAGS = tuple(sorted(ALL))


def name_of(suffix):
    return NAMESPACE + suffix


def describe(suffix, flags, parent, what):
    parent_name = SHIP if parent == 'ship' else AUDIT_PARENT
    adds = ', '.join('%s=%s' % pair for pair in sorted(flags.items()))
    if parent == 'ship':
        marker = ('the gate marker %s instead of its owner waiver id %s (the gate run admits the octo shape, as -fx-s4-nogrid13 does) and gate_only'
                  % (', '.join('%s=%s' % pair for pair in sorted(MARKER_ENV.items())), WAIVER_ENV))
    else:
        marker = 'nothing else (the parent is already a gate-only profile with every audit on)'
    return ('GATE ONLY, UNQUALIFIED (tp4/sched-1, docs/tp4-sched-1.md; never traffic). %s of the scheduler-side levers. The parent is %s and this profile is that parent plus exactly %s and %s; its '
            'engine, memory plan and every other flag are the parent\'s byte for byte. %s Generated by scripts/ci/make_sched_profiles.py: regenerate, never hand-merge.'
            % (what, parent_name, adds, marker,
               'The paired jobs of references/fusion-jobs/SCH time it against the approved profile.' if parent == 'ship'
               else 'Every audit of the candidate and the tier\'s instruments run beside the new pacing, the lean admission and the complete spills; the read is the ship-4 audited attach\'s.'))


def specs():
    """[(profile name, parent name, what it is)] in the order they are written."""
    return [(name_of(suffix), SHIP if parent == 'ship' else AUDIT_PARENT, describe(suffix, flags, parent, what)) for suffix, flags, parent, what in ARMS]


def twin_names():
    """The generated twins' names (the profile-enumerating tests exempt them through profile_twins; test_sched_profiles holds each)."""
    return NAMES


def parent_problems(name, profile, audit):
    """Why `name` cannot be twinned, [] when it can."""
    problems = []
    env = profile.get('env') or {}
    for flag in FLAGS:
        if flag in env:
            problems.append('%s already names %s: a parent never carries the knob it is twinned with' % (name, flag))
    if audit:
        if profile.get('gate_only') is not True or env.get('QWEN_C2_GATE_PROFILE') != '1':
            problems.append('%s is not the gate-only audited twin of the ship candidate' % name)
    else:
        waiver = profile.get(WAIVER_FIELD)
        if profile.get('gate_only') is True or not isinstance(waiver, dict):
            problems.append('%s is not a traffic profile with an owner traffic waiver: the arm is that profile with the gate marker in its place' % name)
        elif env.get(WAIVER_ENV) != waiver.get('id'):
            problems.append('%s does not carry its waiver id in %s' % (name, WAIVER_ENV))
        for flag in MARKER_ENV:
            if flag in env:
                problems.append('%s already names %s: a traffic profile never carries the gate marker' % (name, flag))
    for needed in ('QWEN_FAST_LEVER_N', 'QWEN_PREFIX_REUSE', 'QWEN_FAST_LEVERN_PARK', 'QWEN_PREFIX_HOST_TIER_GIB', 'QWEN_FAST_EXTENT_REPLAY', 'QWEN_FAST_PARKED_ENGINES'):
        if needed not in env:
            problems.append('%s does not name %s, which the levers need' % (name, needed))
    return problems


def generate(data):
    """data's profiles with the twins inserted after the ship-4 (and np1) twins (every other entry unmoved); returns the new dict."""
    profiles = data['profiles']
    for name in (AUDIT_PARENT, SHIP):
        if name not in profiles:
            raise ValueError('the parent profile %s is not defined (make_fusion_profiles.py --write, make_octo3_profiles.py --write and make_ship4_profiles.py --write, first)' % name)
    problems = parent_problems(AUDIT_PARENT, profiles[AUDIT_PARENT], True) + parent_problems(SHIP, profiles[SHIP], False)
    if problems:
        raise ValueError('; '.join(problems))
    generated = {}
    for suffix, flags, parent, what in ARMS:
        if parent == 'ship':
            arm = copy.deepcopy(profiles[SHIP])
            arm['env'].pop(WAIVER_ENV)
            arm['env'].update(MARKER_ENV)
            arm.pop(WAIVER_FIELD, None)
        else:
            arm = copy.deepcopy(profiles[AUDIT_PARENT])
        arm['env'].update(flags)
        arm['description'] = describe(suffix, flags, parent, what)
        arm.pop('gate_only', None)
        arm['gate_only'] = True
        generated[name_of(suffix)] = arm
    remaining = [name for name in profiles if name not in generated]            # a previous generation is replaced below
    anchors = [name for name in remaining if name in ship4.twin_names() or name.startswith(NP1_PREFIX)]
    if not anchors:
        raise ValueError('the tp4/ship-4 twins are not in the profiles file (make_ship4_profiles.py --write first)')
    anchor = anchors[-1]
    result = {}
    for name in remaining:
        result[name] = profiles[name]
        if name == anchor:
            result.update(generated)
    out = dict(data)
    out['profiles'] = result
    return out


def render(data):
    return json.dumps(data, indent=2) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--path', default=str(PROFILES))
    arguments = parser.parse_args(argv)
    path = Path(arguments.path)
    current = path.read_text(encoding='utf-8')
    wanted = render(generate(json.loads(current)))
    if arguments.check:
        if wanted != current:
            sys.stderr.write('%s is not what make_sched_profiles.py generates from its parents: run it with --write\n' % path)
            return 1
        return 0
    if arguments.write:
        if wanted != current:
            with open(str(path), 'w', encoding='utf-8', newline='\n') as handle:
                handle.write(wanted)
        return 0
    parser.error('--write or --check')


if __name__ == '__main__':
    sys.exit(main())
