"""The round-host profile twins (tp4/round-host, docs/tp4-round-host.md), generated from their parent (stdlib only).

    python3 scripts/ci/make_round_host_profiles.py --write   regenerate the twins in scripts/ci/qwen_c2_profiles.json
    python3 scripts/ci/make_round_host_profiles.py --check   exit 1 if the checked-in twins are not what the parent generates

NEVER hand-merge these entries into the profiles file. After the branch is merged with another that moves the parent (the Lever N traffic profile),
regenerate: each twin is the parent's env, engine and every other field byte for byte plus exactly the env below, and gate_only.

The twins (suffixes of PARENT):
  -roundhost-log    the CONTROL of the paired timing: the per-round ledger line alone (QWEN_FAST_TP4_ROUND_HOST_LOG), no path changes
  -roundhost        the ARM: the ledger and the four levers (SELECT, READ, KEYED, LEAN)
  -roundhost-audit  the arm plus QWEN_FAST_TP4_ROUND_HOST_AUDIT: the reference runs beside each fast path on the live data and the two must agree

Every twin is gate_only: nothing serves traffic on the levers until the exactness and timing jobs of scripts/ci/references/tp4-round-host-jobs have run
and the owner decides. The production profile and every non-gate profile carry no QWEN_FAST_TP4_ROUND_HOST_ name (test_tp4_round_host holds it).
"""

import argparse
import copy
import json
from pathlib import Path
import sys

PROFILES = Path(__file__).resolve().parent / 'qwen_c2_profiles.json'

PARENT = 'c2-packed-tp4-8x262k-ship-prefix-levern-traffic'
LOG = 'QWEN_FAST_TP4_ROUND_HOST_LOG'
SELECT = 'QWEN_FAST_TP4_ROUND_HOST_SELECT'
READ = 'QWEN_FAST_TP4_ROUND_HOST_READ'
KEYED = 'QWEN_FAST_TP4_ROUND_HOST_KEYED'
LEAN = 'QWEN_FAST_TP4_ROUND_HOST_LEAN'
AUDIT = 'QWEN_FAST_TP4_ROUND_HOST_AUDIT'
FLAGS = (LOG, SELECT, READ, KEYED, LEAN, AUDIT)

COMMON = ('GATE ONLY (tp4/round-host, docs/tp4-round-host.md; the host work inside a verified round). The parent is %s, the profile measured at eight and four live '
          'users on the integration base: an eight-live round is 157 ms, two 49.7 ms verify traces, 7.2 ms of verify host work, the 42.9 ms early draft (two quads) and '
          '6.2 ms between. This profile is the parent plus the flags named below and nothing else; every flag is host only (no kernel, trace, allocation or device command) '
          'and with all of them at 0 the parent\'s bytes run. ' % PARENT)


def specs():
    """[(profile name, env additions, description)] in the order they are written."""
    return [
        (PARENT + '-roundhost-log', {LOG: '1'},
         COMMON + 'THIS ARM, the CONTROL of the paired timing: %s=1 alone, the per-round ledger line ([PACKED-ROUND-HOST]: the host time of every phase of the step) and nothing '
                  'that changes a path, so the arm that carries the levers is read against the same instrument phase by phase.' % LOG),
        (PARENT + '-roundhost', {LOG: '1', SELECT: '1', READ: '1', KEYED: '1', LEAN: '1'},
         COMMON + 'THIS ARM: the ledger (%s) and the four levers: %s (the FP64 draft selection with the codebook checks made once), %s (the quad readback\'s merge as one batch, the '
                  'replicated-feature guard on the first 64 reads and every 64th), %s (the verify-time diff keyed on the start and page table of each segment: the tokens buffer alone is '
                  'written when neither moved since the pre-stage) and %s (the per-phase lines nobody parses are not written). Every fast path declines to the reference whole on any '
                  'failed check. Untested on hardware: a declined or unequal line is a finding.' % (LOG, SELECT, READ, KEYED, LEAN)),
        (PARENT + '-roundhost-audit', {LOG: '1', SELECT: '1', READ: '1', KEYED: '1', LEAN: '1', AUDIT: '1'},
         COMMON + 'THIS ARM: the arm above plus %s=1, which runs the reference beside each fast path on the live data (the selection, the merge, the keyed round\'s full values against '
                  'the snapshot\'s) and logs [ROUND-HOST-AUDIT] kind= equal=; a difference uses the reference\'s bytes and fails the smoke check. Host work only, so it is not a timing '
                  'arm.' % AUDIT),
    ]


def twin_names():
    """The generated twins' names (the profile-enumerating tests of the other lever branches exempt them: test_tp4_round_host holds each)."""
    return tuple(name for name, _env, _description in specs())


def parent_problems(profile):
    """Why the parent cannot be twinned, [] when it can."""
    problems = []
    for flag in FLAGS:
        if flag in (profile.get('env') or {}):
            problems.append('%s already names %s: a parent never carries the lever it is twinned with' % (PARENT, flag))
    if profile.get('gate_only') is True:
        problems.append('%s is gate_only: the parent is the traffic profile' % PARENT)
    return problems


def generate(data):
    """data's profiles with the twins inserted right after their parent (every other entry unmoved); returns the new dict."""
    profiles = data['profiles']
    if PARENT not in profiles:
        raise ValueError('the parent profile %s is not defined' % PARENT)
    problems = parent_problems(profiles[PARENT])
    if problems:
        raise ValueError('; '.join(problems))
    generated = {}
    for name, env, description in specs():
        profile = copy.deepcopy(profiles[PARENT])
        profile['env'].update(env)
        profile['description'] = description
        profile.pop('gate_only', None)
        profile['gate_only'] = True
        generated[name] = profile
    result = {}
    for name, profile in profiles.items():
        if name in generated:
            continue                                    # a previous generation: replaced below
        result[name] = profile
        if name == PARENT:
            result.update(generated)
    out = dict(data)
    out['profiles'] = result
    return out


def render(data):
    return json.dumps(data, indent=2, ensure_ascii=False) + '\n'


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
            sys.stderr.write('%s is not what make_round_host_profiles.py generates from its parent: run it with --write\n' % path)
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
