"""The tp4/octo-3 profile twins: octo-T8 and its three levers on the FUSED stack, generated from their parents (stdlib only).

    python3 scripts/ci/make_octo3_profiles.py --write   regenerate the twins in scripts/ci/qwen_c2_profiles.json
    python3 scripts/ci/make_octo3_profiles.py --check   exit 1 if the checked-in twins are not what the parents generate

NEVER hand-merge these entries into the profiles file. After the branch is merged with another that moves a parent (the fusion generator's `-fx-all` family, make_octo_profiles' memory plan),
regenerate (make_fusion_profiles.py --write first): each twin is its parent's env, engine and every other field byte for byte plus exactly the env named below, the third block's memory plan
(make_octo_profiles.MEMORY: the KV pool and the trace region) and gate_only. A parent whose pool or trace region differs from what that plan was derived under is refused.

WHAT THE TWINS ARE. tp4/octo-2 built the octo-T8 shape (one 64-row verify pass of 8 seats x 8 rows instead of two M3 passes of 4 x 16) and three levers on it (docs/tp4-octo2.md) on the Lever N
octo arm, and measured on cards (8 live, ~3.5k-token coding prompts, QWEN_FAST_OCTO=alternate, paired per round against the arm's own m3 rounds): the octo draft +6.1%, the glue levers +28.6%,
the attention bundle +2.5%, all three +46.6%. The profiles here put the same shape and levers on the stack production runs: W2 (multi SDPA on the M3 blocks), engine reuse, the fused op set
(`-fx-all`), and on `-fx-all-oneq-grid13` (the prefill levers). docs/tp4-octo3.md has the lever-by-lever status.

  P-octo3              QWEN_FAST_OCTO=live (every eligible 8-live round is octo: the NON-alternating timing arm; the control is `P` itself, the fused stack's normal m3 rounds on the same image) +
                       QWEN_FAST_OCTO_DRAFT=1 + QWEN_FAST_OCTO_ATTN_BUNDLE=4 + QWEN_FAST_OCTO_GLUE8=1: all three levers.
  P-octo3-quaddraft    the same without the octo draft: the fused quad drafters draft the octo rounds (cut to seven rows on the host). The draft A/B: which drafter an octo round wants.
  P-octo3-audit        the audited twin (on P-audit): the three levers with their audits (the bundle's, the GDN glue's, the packed windows', F1's) and the combined arm's fusion audits, with
                       the host read-back audits SAMPLED (audit_sampling: the first round, then every 16th), so an audited attach completes. Not a timing arm.
  for P in (the combined arm `-fx-all`, and `-fx-all-oneq-grid13`, the combined prefill arm).

Every twin is gate_only and unqualified: nothing serves traffic on octo-T8 until its exactness, memory, hang and paired-timing gates have run and the owner decides. Each carries
QWEN_C2_GATE_PROFILE=1, the profile's own marker without which serving_octo refuses the flag.
"""

import argparse
import copy
import json
from pathlib import Path
import sys

import make_octo2_profiles
import make_octo_profiles

PROFILES = Path(__file__).resolve().parent / 'qwen_c2_profiles.json'

BASE = 'c2-packed-tp4-8x262k-ship-prefix-levern'
FX_ALL = BASE + '-w2-er-fx-all'
FX_ALL_AUDIT = FX_ALL + '-audit'
FX_ALL_PREFILL = BASE + '-w2-er-fx-all-oneq-grid13'
FX_ALL_PREFILL_AUDIT = FX_ALL_PREFILL + '-audit'

OCTO = 'QWEN_FAST_OCTO'
MIN_LIVE = 'QWEN_FAST_OCTO_MIN_LIVE'
DRAFT = make_octo2_profiles.DRAFT
BUNDLE = make_octo2_profiles.BUNDLE
BUNDLE_AUDIT = make_octo2_profiles.BUNDLE_AUDIT
GLUE8 = make_octo2_profiles.GLUE8
BUNDLE_VALUE = make_octo2_profiles.BUNDLE_VALUE
AUDIT_FIRST = 'QWEN_FAST_AUDIT_FIRST'
AUDIT_EVERY = 'QWEN_FAST_AUDIT_EVERY'
SAMPLE_FIRST, SAMPLE_EVERY = '1', '16'

MARKER_ENV = dict(make_octo_profiles.MARKER_ENV)
BASE_ENV = dict(MARKER_ENV, **{OCTO: 'live', MIN_LIVE: '6'})
LEVERS_ENV = {DRAFT: '1', BUNDLE: BUNDLE_VALUE, GLUE8: '1'}
QUAD_LEVERS_ENV = {BUNDLE: BUNDLE_VALUE, GLUE8: '1'}
# The audits of the octo levers (the vglue audit compares every native glue site of the octo block with the served path in the trace on every chip; the windows audit the 32 packed
# windows after the conv gates; F1's the spread launch the block conv uses at 64 rows; the bundle's the eight single launches beside the two bundled ones) and the sample that keeps their
# host read-backs affordable (audit_sampling.py).
AUDIT_ENV = {BUNDLE_AUDIT: '1', 'QWEN_FAST_TP4_VGLUE_AUDIT': '1', 'QWEN_FAST_VERIFY_T2_AUDIT': '1', 'QWEN_FAST_TP4_CONV_GATES_SPREAD_AUDIT': '1',
             AUDIT_FIRST: SAMPLE_FIRST, AUDIT_EVERY: SAMPLE_EVERY}
FLAGS = (OCTO, MIN_LIVE, DRAFT, BUNDLE, BUNDLE_AUDIT, GLUE8, AUDIT_FIRST, AUDIT_EVERY)

COMMON = ('GATE ONLY, UNQUALIFIED (tp4/octo-3, docs/tp4-octo3.md; octo-T8 and its three levers on the fused stack: the shape and the levers have run on cards on the Lever N stack only - '
          'octo draft +6.1%%, glue8 +28.6%%, attention bundle +2.5%%, all three +46.6%% committed tokens per second per seat on octo rounds against the m3 rounds of the same boot, '
          'text identical in every arm - and nothing here has run on a card). The parent is %s and this profile is that parent plus exactly the env named below, the gate marker '
          'QWEN_C2_GATE_PROFILE and the third block\'s memory plan (the KV pool lowered from %d to %d blocks, QWEN36_MAX_TOKENS_ALL_USERS %d, and the trace region raised from %d to %d MiB); '
          'every lever is default off and byte-identical off. ')


def specs():
    """[(profile name, parent name, env additions, what it is)] in the order they are written."""
    levers = ', '.join('%s=%s' % item for item in sorted(LEVERS_ENV.items()))
    quad_levers = ', '.join('%s=%s' % item for item in sorted(QUAD_LEVERS_ENV.items()))
    out = []
    for parent, audit_parent, prefill in ((FX_ALL, FX_ALL_AUDIT, False), (FX_ALL_PREFILL, FX_ALL_PREFILL_AUDIT, True)):
        where = ('on the combined prefill arm (the prefill SDPA with one q chunk per core and the 13x10 matmul grids): the prefill levers act on prefill, the octo levers on decode, so the pair '
                 'is the whole stack\'s candidate' if prefill else 'on the combined arm (every op-fusion lever on)')
        out.append((parent + '-octo3', parent, dict(BASE_ENV, **LEVERS_ENV),
                    'THE TIMING ARM, all three levers %s: QWEN_FAST_OCTO=live (every eligible 8-live round runs the octo shape, no alternation: the control is %s itself, the fused stack\'s normal '
                    'm3 rounds on the same image) and %s. Which drafter drafts an octo round is the A/B of the quaddraft twin (docs/tp4-octo3.md section 3).' % (where, parent, levers)))
        if not prefill:
            out.append((parent + '-octo3-quaddraft', parent, dict(BASE_ENV, **QUAD_LEVERS_ENV),
                        'THE DRAFT A/B ARM: %s without the octo draft (%s): the fused quad drafters draft the octo rounds, cut to seven rows on the host (docs/tp4-octo3.md section 3).' % (
                            where, quad_levers)))
        out.append((parent + '-octo3-audit', audit_parent, dict(BASE_ENV, **dict(LEVERS_ENV, **AUDIT_ENV)),
                    'THE AUDITED ATTACH ARM %s: the three levers with their audits (the bundle compares the eight single launches with the bundled ones word for word; the GDN glue audit '
                    'compares every native split, merge and block-conv entry of the octo block with the served path in the trace, on every chip; the packed-windows audit the 32 windows; '
                    'F1\'s audit the spread launch), plus the fusion audits of %s, and the host read-back audits SAMPLED (QWEN_FAST_AUDIT_FIRST=%s, QWEN_FAST_AUDIT_EVERY=%s: audit_sampling) '
                    'because the octo-2 audited glue8 attach (every round, two layers of 73 entries) warmed for 412 s and decoded at 3.4 tok/s a seat until its users timed out. '
                    'Not the Lever N audited parent (its prefix digests, singles audit and Lever N audit are 8-minute-a-request instruments). Not a timing arm.'
                    % (where, audit_parent, SAMPLE_FIRST, SAMPLE_EVERY)))
    return out


def twin_names():
    """The generated twins' names (the profile-enumerating tests exempt them through profile_twins; test_octo3_profiles holds each)."""
    return tuple(name for name, _parent, _env, _why in specs())


def parents():
    return (FX_ALL, FX_ALL_AUDIT, FX_ALL_PREFILL, FX_ALL_PREFILL_AUDIT)


def parent_problems(profile, name):
    """Why the parent cannot be twinned, [] when it can."""
    problems = list(make_octo_profiles.parent_problems(name, profile))
    for flag in FLAGS:
        if flag in (profile.get('env') or {}):
            problems.append('%s already names %s: a parent never carries the lever it is twinned with' % (name, flag))
    if (profile.get('env') or {}).get('QWEN_FAST_TP4_SDPA') != 'multi':
        problems.append('%s is not a W2 profile (QWEN_FAST_TP4_SDPA=multi): the octo3 twins are the fused stack\'s' % name)
    if (profile.get('env') or {}).get('QWEN_FAST_PARKED_ENGINES') != '1':
        problems.append('%s is not an engine-reuse profile (QWEN_FAST_PARKED_ENGINES=1): the octo3 twins are the fused stack\'s' % name)
    return problems


def generate(data):
    """data's profiles with the twins inserted right after the tp4/octo-2 twins (every other entry unmoved); returns the new dict."""
    profiles = data['profiles']
    problems = []
    for name in parents():
        if name not in profiles:
            raise ValueError('the parent profile %s is not defined (make_fusion_profiles.py --write first)' % name)
        problems.extend(parent_problems(profiles[name], name))
    if problems:
        raise ValueError('; '.join(problems))
    plan = make_octo_profiles.memory_plan()
    if plan['margin_bytes'] < 0 or plan['windows'] < 4:
        raise ValueError('the memory plan does not fit: %r' % (plan,))
    memory = make_octo_profiles.MEMORY
    generated = {}
    for name, parent, env, why in specs():
        profile = copy.deepcopy(profiles[parent])
        profile['env'].update(env)
        profile['engine']['num-gpu-blocks-override'] = memory['blocks']
        profile['engine']['additional-config']['tt']['trace_region_size'] = memory['trace_region']
        profile['env']['QWEN36_MAX_TOKENS_ALL_USERS'] = str(plan['pool_tokens'])
        profile['description'] = (COMMON % (parent, memory['parent_blocks'], memory['blocks'], plan['pool_tokens'], memory['parent_trace_region'] >> 20, memory['trace_region'] >> 20)) + why
        profile.pop('gate_only', None)
        profile['gate_only'] = True
        generated[name] = profile
    # After the last of make_octo2_profiles' twins: that generator re-inserts its block after the octo twins and leaves every other entry where it finds it, so the generators agree on the
    # file only if these twins sit behind that block (and ahead of the fusion block and the region-read twins that close the file, which their generators place themselves).
    anchors = [name for name in profiles if name in make_octo2_profiles.twin_names()]
    anchor = anchors[-1] if anchors else make_octo2_profiles.PARENT
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
            sys.stderr.write('%s is not what make_octo3_profiles.py generates from its parents: run it with --write\n' % path)
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
