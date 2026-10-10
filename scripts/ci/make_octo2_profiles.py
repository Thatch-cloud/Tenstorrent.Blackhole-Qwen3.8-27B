"""The tp4/octo-2 lever twins (docs/tp4-octo2.md), generated from their parents (stdlib only).

    python3 scripts/ci/make_octo2_profiles.py --write   regenerate the twins in scripts/ci/qwen_c2_profiles.json
    python3 scripts/ci/make_octo2_profiles.py --check   exit 1 if the checked-in twins are not what the parents generate

NEVER hand-merge these entries into the profiles file. After the branch is merged with another that moves a parent (make_octo_profiles.py's twins), regenerate: each twin is its
parent's env, engine and every other field byte for byte plus exactly the env below, and gate_only. The parents are the octo-T8 timed arm (-octo: QWEN_FAST_OCTO=alternate, so
octo and m3 rounds are timed in pairs on one boot) with the memory plan of the third block already in it; nothing here edits the pool or the trace region.

The twins (suffixes of the parent P = c2-packed-tp4-8x262k-ship-prefix-levern):
  P-octo-draft         QWEN_FAST_OCTO_DRAFT=1: octo rounds are drafted by ONE eight-seat, eight-row pass instead of the two quads (docs/tp4-octo-draft.md). THE ARM of Lever 1.
  P-octo-bundle        QWEN_FAST_OCTO_ATTN_BUNDLE=4: the octo block's attention in two launches per layer instead of eight (docs/tp4-octo-attn-bundle.md). THE ARM of Lever 3.
  P-octo-bundle-audit  the bundle arm plus QWEN_FAST_OCTO_ATTN_BUNDLE_AUDIT=1: the eight single launches run beside the bundled ones and are compared in-trace. Not a timing arm.
  P-octo-glue8         QWEN_FAST_OCTO_GLUE8=1: the GDN glue levers at the octo block's eight-row users (docs/tp4-octo-glue8.md). THE ARM of Lever 2.
  P-octo-levers        all three levers (draft, bundle, glue8): the arm whose gain is the sum the owner asked for, read after each lever has passed alone.

Every twin is gate_only and unqualified: nothing serves traffic on a lever until its exactness and paired-timing jobs (scripts/ci/references/tp4-octo2-jobs) have run and the owner decides.
"""

import argparse
import copy
import json
from pathlib import Path
import sys

import make_octo_profiles

PROFILES = Path(__file__).resolve().parent / 'qwen_c2_profiles.json'

BASE = 'c2-packed-tp4-8x262k-ship-prefix-levern'
PARENT = BASE + '-octo'
DRAFT = 'QWEN_FAST_OCTO_DRAFT'
BUNDLE = 'QWEN_FAST_OCTO_ATTN_BUNDLE'
BUNDLE_AUDIT = 'QWEN_FAST_OCTO_ATTN_BUNDLE_AUDIT'
GLUE8 = 'QWEN_FAST_OCTO_GLUE8'
FLAGS = (DRAFT, BUNDLE, BUNDLE_AUDIT, GLUE8)
BUNDLE_VALUE = '4'

COMMON = ('GATE ONLY, UNQUALIFIED (tp4/octo-2, docs/tp4-octo2.md; octo-T8 itself has run on cards once: the 8-live round is two ~50 ms verify traces or one 86.5 ms octo trace, '
          'a ~30 ms early draft and ~45 ms of gap, +3.5%% committed tokens per second per seat against the m3 rounds on the same boot). The parent is %s (QWEN_FAST_OCTO=alternate, the third '
          "block's memory plan in it) and this profile is that parent plus exactly the env named below and nothing else; every lever is default off and byte-identical off. " % PARENT)


def specs():
    """[(profile name, env additions, what it is)] in the order they are written."""
    return [
        (BASE + '-octo-draft', {DRAFT: '1'},
         'THIS ARM, Lever 1: %s=1. An octo round with all eight seats live is drafted by ONE eight-seat, eight-row, 64-row pass (octo_draft_tp) instead of the two four-seat quad passes. '
         'The drafts are T8 (eight live keys in the block, not sixteen), so the cost is tau, which the paired read prints beside the draft time. The m3 rounds of the same boot still '
         'draft with the quads.' % DRAFT),
        (BASE + '-octo-bundle', {BUNDLE: BUNDLE_VALUE},
         'THIS ARM, Lever 3: %s=%s. The octo block\'s attention is two K64j launches of four users per layer instead of eight launches of one (flags 0x21, 16 cores per entry, no kernel '
         'or factory change). Untested on a card.' % (BUNDLE, BUNDLE_VALUE)),
        (BASE + '-octo-bundle-audit', {BUNDLE: BUNDLE_VALUE, BUNDLE_AUDIT: '1'},
         'THIS ARM, Lever 3 audited: the bundle arm plus %s=1, which runs the eight single launches beside the bundled ones and compares the block outputs word for word in the trace. '
         'The exactness job; not a timing arm.' % BUNDLE_AUDIT),
        (BASE + '-octo-glue8', {GLUE8: '1'},
         'THIS ARM, Lever 2: %s=1. The GDN verify-glue levers (split, merge, block conv, packed windows) run natively at the octo block\'s eight-row users instead of the served path.' % GLUE8),
        (BASE + '-octo-levers', {DRAFT: '1', BUNDLE: BUNDLE_VALUE, GLUE8: '1'},
         'THIS ARM, all three levers: %s=1, %s=%s and %s=1 together; read after each has passed alone.' % (DRAFT, BUNDLE, BUNDLE_VALUE, GLUE8)),
    ]


def twin_names():
    """The generated twins' names (the profile-enumerating tests exempt them through profile_twins; test_octo2_profiles holds each)."""
    return tuple(name for name, _env, _why in specs())


def parent_problems(profile):
    """Why the parent cannot be twinned, [] when it can."""
    problems = []
    for flag in FLAGS:
        if flag in (profile.get('env') or {}):
            problems.append('%s already names %s: a parent never carries the lever it is twinned with' % (PARENT, flag))
    if profile.get('gate_only') is not True:
        problems.append('%s is not gate_only' % PARENT)
    if (profile.get('env') or {}).get('QWEN_FAST_OCTO') != 'alternate':
        problems.append('%s is not the QWEN_FAST_OCTO=alternate timed arm' % PARENT)
    return problems


def generate(data):
    """data's profiles with the twins inserted right after their parent (every other entry unmoved); returns the new dict."""
    profiles = data['profiles']
    if PARENT not in profiles:
        raise ValueError('the parent profile %s is not defined (make_octo_profiles.py --write first)' % PARENT)
    problems = parent_problems(profiles[PARENT])
    if problems:
        raise ValueError('; '.join(problems))
    generated = {}
    for name, env, why in specs():
        profile = copy.deepcopy(profiles[PARENT])
        profile['env'].update(env)
        profile['description'] = COMMON + why
        profile.pop('gate_only', None)
        profile['gate_only'] = True
        generated[name] = profile
    # After the last of make_octo_profiles' twins: that generator re-inserts its block after the engine-reuse family and leaves every other entry where it finds it, so the
    # two generators agree on the file only if these twins sit behind that block.
    anchors = [name for name in profiles if name in make_octo_profiles.twin_names()]
    anchor = anchors[-1] if anchors else PARENT
    result = {}
    for name, profile in profiles.items():
        if name in generated:
            continue                                    # a previous generation: replaced below
        result[name] = profile
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
            sys.stderr.write('%s is not what make_octo2_profiles.py generates from its parent: run it with --write\n' % path)
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
