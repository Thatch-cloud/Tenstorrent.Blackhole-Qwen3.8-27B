"""The engine-reuse (parked per-slot engines) profile twins, generated from their parents (stdlib only).

    python3 scripts/ci/make_parked_profiles.py --write   regenerate the twins in scripts/ci/qwen_c2_profiles.json
    python3 scripts/ci/make_parked_profiles.py --check   exit 1 if the checked-in twins are not what the parents generate

NEVER hand-merge these entries into the profiles file. After the branch is merged with another that moves a parent (the combined window branch moves
the Lever N and prefix profiles), regenerate: the twins are the parent's env, engine and every other field byte for byte plus exactly the env below.
A parent whose KV block count or trace region differs from what the memory margins of docs/tp4-engine-reuse.md were derived under is refused, so a
later change to either has to re-read those margins first (design section 12, K1).

Every twin is gate_only: nothing serves traffic on engine reuse until its exactness, lifecycle, memory and hang gates have run and the owner decides.
The production profile and every non-gate profile carry no QWEN_FAST_PARKED_ name (test_parked_tp4_profiles holds it).

The twins (suffixes of PARENT):
  -parked           the candidate: engine reuse and the bound drafter traces (2c), the governor charging the learned admission cost
  -parked-e1        engine reuse alone (the drafter traces retired at every park, the governor's fixed cost): isolates what 2c adds
  -parked-audit     the audited twin of -parked (the parent's audit twin plus the parked audit and the collective-handle guard)
  -audit-r2         the flag-off CONTROL of -parked-audit: the audited parent plus the replay ledger and the handle guard, no parked engines, so the
                    sequential step ordering of replays (R2) is measured where the control runs it
  -parked-neg-<n>   the negative controls (carry, pages, widths on -parked; drafter on -parked-audit): each must fail its own judge
  -parked-fault-<n> the injected faults (park, rebind)
  -parked-ballast-<MB>  the DRAM ballast at fixed levels, so the parked admission can be run at the boundary of its need without an image rebuild
"""

import argparse
import copy
import json
from pathlib import Path
import sys

PROFILES = Path(__file__).resolve().parent / 'qwen_c2_profiles.json'

PRODUCTION = 'c2-packed-tp4-8x262k-ship-prefix'
CANDIDATE_PARENT = PRODUCTION + '-levern'
AUDIT_PARENT = PRODUCTION + '-levern-audit'
# The memory margins of docs/tp4-engine-reuse.md (section 4) were derived under these; a parent that differs must be re-read before it is twinned.
KV_BLOCKS = 19968
TRACE_REGION = 536870912
SEATS = 8

ENGINES = {'QWEN_FAST_PARKED_ENGINES': '1'}
DRAFTS = {'QWEN_FAST_PARKED_DRAFTS': '1'}
LEARNED = {'QWEN_FAST_LEVERN_BUILD_MS': 'learned'}
AUDIT = {'QWEN_FAST_PARKED_AUDIT': '1', 'QWEN_FAST_CCL_HANDLE_GUARD': 'log'}
BALLAST_MB = (512, 1024, 1536, 2048)

NEGATIVES = {'carry': 'the rebind seeds no carry', 'pages': 'the page-table rewrite is skipped', 'widths': 'the cold-width restriction (R4) is off',
             'drafter': 'the drafter\'s slot is neither re-zeroed nor reseeded'}
FAULTS = {'park': 'the first park after attach is refused once', 'rebind': 'the first rebind is refused once, on the host, before any device write'}


def specs():
    """[(profile name, parent, env additions, what it is)] in the order they are written."""
    out = [
        (CANDIDATE_PARENT + '-parked', CANDIDATE_PARENT, dict(ENGINES, **DRAFTS, **LEARNED),
         'engine reuse (one engine per pool slot parked at attach and rebound to each request), the pair and quad drafter traces bound to the slots for the '
         'process, and the deadline governor charging each pending prefill the learned cost of the slot it will take. THE ARM'),
        (CANDIDATE_PARENT + '-parked-e1', CANDIDATE_PARENT, dict(ENGINES),
         'engine reuse alone: the drafter traces are retired at every park and recaptured at the next formation, the governor charges its fixed 2,500 ms. '
         'The arm that isolates what the bound traces add'),
        (CANDIDATE_PARENT + '-parked-audit', AUDIT_PARENT, dict(ENGINES, **DRAFTS, **LEARNED, **AUDIT),
         'THE EXACTNESS ARM: the audited parent plus engine reuse, the bound drafter traces, the parked audit (digests at every rebind, the R2 replay ledger) '
         'and the collective-handle guard'),
        (AUDIT_PARENT + '-r2', AUDIT_PARENT, dict(AUDIT),
         'THE FLAG-OFF CONTROL of -parked-audit: the audited parent plus the replay ledger and the handle guard and NO parked engines, so the sequential '
         'step ordering of replays is measured where the control runs it'),
    ]
    for name, why in NEGATIVES.items():
        env = dict(ENGINES, **DRAFTS, **LEARNED)
        if name == 'drafter':
            env.update(AUDIT)
            full, parent = CANDIDATE_PARENT + '-parked-audit-neg-drafter', AUDIT_PARENT
        else:
            full, parent = CANDIDATE_PARENT + '-parked-neg-' + name, CANDIDATE_PARENT
        env['QWEN_FAST_PARKED_NEGATIVE'] = name
        out.append((full, parent, env,
                    'NEGATIVE CONTROL (QWEN_FAST_PARKED_NEGATIVE=%s: %s): it must fail its own judge, never a traffic or timed arm' % (name, why)))
    for name, why in FAULTS.items():
        out.append((CANDIDATE_PARENT + '-parked-fault-' + name, CANDIDATE_PARENT,
                    dict(ENGINES, **DRAFTS, **LEARNED, QWEN_FAST_PARKED_FAULT=name),
                    'INJECTED FAULT (QWEN_FAST_PARKED_FAULT=%s: %s): the slot unparks, a per-request build serves it and it re-parks; never a timed arm' % (name, why)))
    for megabytes in BALLAST_MB:
        out.append((CANDIDATE_PARENT + '-parked-ballast-%d' % megabytes, CANDIDATE_PARENT,
                    dict(ENGINES, **DRAFTS, **LEARNED, QWEN_FAST_GATE_DRAM_BALLAST=str(megabytes * 10 ** 6)),
                    'DRAM BALLAST of %d MB per chip held unread from the end of the parked build (QWEN_FAST_GATE_DRAM_BALLAST), so the admissions run '
                    'at the boundary of the parked need; never a timed arm' % megabytes))
    return out


def parent_problems(name, profile):
    """Why this parent cannot be twinned, [] when it can."""
    problems = []
    engine = profile.get('engine') or {}
    if engine.get('num-gpu-blocks-override') != KV_BLOCKS:
        problems.append('%s: num-gpu-blocks-override %r, not %d: the memory margins were derived under that KV pool (re-read them first)'
                        % (name, engine.get('num-gpu-blocks-override'), KV_BLOCKS))
    region = (((engine.get('additional-config') or {}).get('tt') or {}).get('trace_region_size'))
    if region != TRACE_REGION:
        problems.append('%s: trace_region_size %r, not %d: the trace margins were derived under that region (re-read them first)'
                        % (name, region, TRACE_REGION))
    if engine.get('max-num-seqs') != SEATS:
        problems.append('%s: max-num-seqs %r, not %d seats' % (name, engine.get('max-num-seqs'), SEATS))
    for flag in ('QWEN_FAST_PARKED_ENGINES', 'QWEN_FAST_PARKED_DRAFTS', 'QWEN_FAST_PARKED_AUDIT', 'QWEN_FAST_LEVERN_BUILD_MS'):
        if flag in (profile.get('env') or {}):
            problems.append('%s already names %s: a parent never carries the lever it is twinned with' % (name, flag))
    if profile.get('gate_only') is not True:
        problems.append('%s is not gate_only' % name)
    return problems


def generate(data):
    """data's profiles with the twins inserted right after the last profile they derive from (every other entry unmoved); returns the new dict."""
    profiles = data['profiles']
    for parent_name in (CANDIDATE_PARENT, AUDIT_PARENT):
        if parent_name not in profiles:
            raise ValueError('the parent profile %s is not defined' % parent_name)
        problems = parent_problems(parent_name, profiles[parent_name])
        if problems:
            raise ValueError('; '.join(problems))
    generated = {}
    for name, parent_name, env, why in specs():
        profile = copy.deepcopy(profiles[parent_name])
        profile['env'].update(env)
        profile['description'] = ('GATE ONLY, UNQUALIFIED (the 262k evidence waiver, as its parent carries it; engine reuse, docs/tp4-engine-reuse.md): %s. '
                                  'The parent is %s and this profile is that parent plus exactly the QWEN_FAST_PARKED_ flags, '
                                  'QWEN_FAST_LEVERN_BUILD_MS and QWEN_FAST_GATE_DRAM_BALLAST in its env (generated by scripts/ci/make_parked_profiles.py: '
                                  'regenerate, never hand-merge).' % (why, parent_name))
        profile['gate_only'] = True
        generated[name] = profile
    result, placed = {}, False
    for name, profile in profiles.items():
        if name in generated or name.startswith(CANDIDATE_PARENT + '-parked'):
            continue                                    # a previous generation: replaced below
        result[name] = profile
        if name == AUDIT_PARENT:
            result.update(generated)
            placed = True
    if not placed:
        raise ValueError('the audit parent is not in the profiles file')
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
            sys.stderr.write('%s is not what make_parked_profiles.py generates from its parents: run it with --write\n' % path)
            return 1
        return 0
    if arguments.write:
        if wanted != current:
            path.write_text(wanted, encoding='utf-8', newline='\n')
        return 0
    parser.error('--write or --check')


if __name__ == '__main__':
    sys.exit(main())
