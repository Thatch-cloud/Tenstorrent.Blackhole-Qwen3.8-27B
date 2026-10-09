"""The octo-T8 profile twins, generated from their parents (stdlib only).

    py -3.11 scripts/ci/make_octo_profiles.py --write   regenerate the twins in scripts/ci/qwen_c2_profiles.json
    py -3.11 scripts/ci/make_octo_profiles.py --check   exit 1 if the checked-in twins are not what the parents generate

NEVER hand-merge these entries into the profiles file. After the branch is merged with another that moves a parent, regenerate (make_parked_profiles.py first, whose
twins the -octo-parked ones derive from): a twin is the parent's env, engine and every other field byte for byte plus exactly the env and the two memory edits below.
A parent whose KV block count or trace region differs from what the memory plan was derived under is refused (MEMORY, below), so a later change to either has to
re-read that plan first.

Every twin is gate_only: nothing serves traffic on octo-T8 until its exactness, memory, hang and paired-timing gates have run and the owner decides (docs/tp4-octo.md).
Each carries QWEN_C2_GATE_PROFILE=1, the profile's own marker without which serving_octo refuses the flag (a gate run of a gate-only profile).

The twins (suffixes of PARENT, c2-packed-tp4-8x262k-ship-prefix-levern):
  -octo               QWEN_FAST_OCTO=alternate: every eligible round switches shape (octo, m3, octo, ...), so both arms are timed in pairs on one boot. THE TIMED ARM.
  -octo-live          QWEN_FAST_OCTO=live: every eligible round runs octo. The boot to set against the flag-off parent for the aggregate, without a switch's restage.
  -octo-audit         the audited twin of -octo (the parent's audit twin: verify T1 and T2 audits, extent audit): THE EXACTNESS AND ATTACH ARM.
  -octo-parked        -octo on the engine-reuse parent (-parked): engine reuse ships next, and the third block's memory is taken from the same pool the parked engines sit in.
  -octo-parked-live   -octo-live on the engine-reuse parent.
  -solopacked         QWEN_FAST_SOLO_PACKED=1: a lone live user takes the padded 64-row T16 block instead of the per-request engines. No third block: the KV pool and the trace region are
                      the parent's. The flag refuses to engage until a block holds three idle segments (serving_octo.solo_packed_admission).
  -solopacked-parked  -solopacked on the engine-reuse parent.

MEMORY. The third block is one more 64-row block: about 224 MB per DRAM bank (the measured need of a packed block, test_c2_packed_tp4_profiles), eight banks a chip,
plus the trace region it captures into (the verify trace, 64 GDN commit traces, 8 projection and 64 slide traces; the region is raised from 512 to 640 MiB). The KV pool
is the one lever with a known conversion (557,056 bytes a block a chip, docs/tp4-drafter-bf16.md): 19,968 blocks become 16,500, which frees 1.93 GB, the block and the
extra trace region with 5 MB to spare, and still holds four full 262,144-token windows ((16,500 - 8) x 64 = 1,055,488 tokens). The alternative, REPLACING block B with the
octo block, costs no DRAM and is refused: block B is what every round at 4 or fewer live seats in slots 4..7 and every fall-back runs on, so the M3 path would
no longer be the control it is measured against. QWEN36_MAX_TOKENS_ALL_USERS follows the pool, as the contract demands (serving_c2_contract.kv_pool_problem).
"""

import argparse
import copy
import json
from pathlib import Path
import sys

PROFILES = Path(__file__).resolve().parent / 'qwen_c2_profiles.json'

PRODUCTION = 'c2-packed-tp4-8x262k-ship-prefix'
PARENT = PRODUCTION + '-levern'
AUDIT_PARENT = PARENT + '-audit'
PARKED_PARENT = PARENT + '-parked'
SEATS = 8
WINDOW = 262144
BLOCK_TOKENS = 64

# The memory plan. Parent figures the plan was derived under, and what the twins carry.
MEMORY = dict(
    parent_blocks=19968, parent_trace_region=536870912,
    blocks=16500, trace_region=671088640,
    kv_block_bytes=557056,                  # one 64-token block of the pooled KV, per chip (docs/tp4-drafter-bf16.md)
    block_bytes_per_bank=224 * 10 ** 6, banks=8, # a packed block's measured need (test_c2_packed_tp4_profiles: 224 MB per bank, 8 banks a chip)
)


def memory_plan():
    """The arithmetic behind MEMORY, for the tests and the doc: bytes the pool gives back, bytes the third block and the larger trace region take, the margin,
    the pool in tokens and the full windows it holds."""
    freed_blocks = MEMORY['parent_blocks'] - MEMORY['blocks']
    freed = freed_blocks * MEMORY['kv_block_bytes']
    block = MEMORY['block_bytes_per_bank'] * MEMORY['banks']
    trace = MEMORY['trace_region'] - MEMORY['parent_trace_region']
    tokens = (MEMORY['blocks'] - SEATS) * BLOCK_TOKENS
    return dict(freed_blocks=freed_blocks, freed_bytes=freed, third_block_bytes=block, trace_region_bytes=trace, margin_bytes=freed - block - trace,
                pool_tokens=tokens, windows=tokens // WINDOW)


MARKER_ENV = {'QWEN_C2_GATE_PROFILE': '1'}
LEVERS = ('QWEN_FAST_OCTO', 'QWEN_FAST_OCTO_MIN_LIVE', 'QWEN_FAST_SOLO_PACKED')


def specs():
    """[(profile name, parent, env additions, third block, what it is)] in the order they are written. `third block`: the twin carries the octo memory plan."""
    octo = dict(MARKER_ENV, QWEN_FAST_OCTO_MIN_LIVE='6')
    return [
        (PARENT + '-octo', PARENT, dict(octo, QWEN_FAST_OCTO='alternate'), True,
         'THE TIMED ARM: QWEN_FAST_OCTO=alternate, every eligible round (6 or more live seats) switches shape, octo then m3 then octo, so both arms are timed in pairs on one boot'),
        (PARENT + '-octo-live', PARENT, dict(octo, QWEN_FAST_OCTO='live'), True,
         'THE AGGREGATE ARM: QWEN_FAST_OCTO=live, every eligible round runs the octo shape; set against the flag-off parent it is the throughput without a switch\'s restage'),
        (PARENT + '-octo-audit', AUDIT_PARENT, dict(octo, QWEN_FAST_OCTO='alternate'), True,
         'THE EXACTNESS AND ATTACH ARM: QWEN_FAST_OCTO=alternate on the audited parent (verify T1 and T2 audits, the extent audit), so every audit runs on both shapes and a text '
         'that differs from its solo run is a shape that changed a token'),
        (PARENT + '-octo-parked', PARKED_PARENT, dict(octo, QWEN_FAST_OCTO='alternate'), True,
         'THE TIMED ARM ON ENGINE REUSE: QWEN_FAST_OCTO=alternate on the parked parent (one engine per slot parked at attach, the bound drafter traces, the learned governor)'),
        (PARENT + '-octo-parked-live', PARKED_PARENT, dict(octo, QWEN_FAST_OCTO='live'), True,
         'THE AGGREGATE ARM ON ENGINE REUSE: QWEN_FAST_OCTO=live on the parked parent'),
        (PARENT + '-solopacked', PARENT, dict(MARKER_ENV, QWEN_FAST_SOLO_PACKED='1'), False,
         'THE LONE-USER ARM: QWEN_FAST_SOLO_PACKED=1, a lone live user (and two or three in a block) take the padded 64-row T16 block instead of the per-request engines; the flag '
         'refuses to engage until a block holds three idle segments (serving_octo.solo_packed_admission)'),
        (PARENT + '-solopacked-parked', PARKED_PARENT, dict(MARKER_ENV, QWEN_FAST_SOLO_PACKED='1'), False,
         'THE LONE-USER ARM ON ENGINE REUSE: QWEN_FAST_SOLO_PACKED=1 on the parked parent'),
    ]


def twin_names():
    """The generated twins' names (the profile-enumerating tests of the other lever branches exempt them: test_octo_profiles holds each)."""
    return tuple(name for name, _parent, _env, _third, _why in specs())


def octo_twin_names():
    """The twins that carry QWEN_FAST_OCTO (and so the memory plan)."""
    return tuple(name for name, _parent, _env, third, _why in specs() if third)


def parent_problems(name, profile):
    """Why this parent cannot be twinned, [] when it can."""
    problems = []
    engine = profile.get('engine') or {}
    if engine.get('num-gpu-blocks-override') != MEMORY['parent_blocks']:
        problems.append('%s: num-gpu-blocks-override %r, not %d: the memory plan was derived under that KV pool (re-read it first)'
                        % (name, engine.get('num-gpu-blocks-override'), MEMORY['parent_blocks']))
    region = (((engine.get('additional-config') or {}).get('tt') or {}).get('trace_region_size'))
    if region != MEMORY['parent_trace_region']:
        problems.append('%s: trace_region_size %r, not %d: the memory plan was derived under that region (re-read it first)'
                        % (name, region, MEMORY['parent_trace_region']))
    if engine.get('max-num-seqs') != SEATS:
        problems.append('%s: max-num-seqs %r, not %d seats' % (name, engine.get('max-num-seqs'), SEATS))
    for flag in LEVERS:
        if flag in (profile.get('env') or {}):
            problems.append('%s already names %s: a parent never carries the lever it is twinned with' % (name, flag))
    if profile.get('gate_only') is not True:
        problems.append('%s is not gate_only' % name)
    return problems


def generate(data):
    """data's profiles with the twins inserted right after the last profile of the engine-reuse family (every other entry unmoved); returns the new dict."""
    profiles = data['profiles']
    for parent_name in (PARENT, AUDIT_PARENT, PARKED_PARENT):
        if parent_name not in profiles:
            raise ValueError('the parent profile %s is not defined (make_parked_profiles.py --write first)' % parent_name)
        problems = parent_problems(parent_name, profiles[parent_name])
        if problems:
            raise ValueError('; '.join(problems))
    plan = memory_plan()
    if plan['margin_bytes'] < 0 or plan['windows'] < 4:
        raise ValueError('the memory plan does not fit: %r' % (plan,))
    generated = {}
    for name, parent_name, env, third, why in specs():
        profile = copy.deepcopy(profiles[parent_name])
        profile['env'].update(env)
        if third:
            profile['engine']['num-gpu-blocks-override'] = MEMORY['blocks']
            profile['engine']['additional-config']['tt']['trace_region_size'] = MEMORY['trace_region']
            profile['env']['QWEN36_MAX_TOKENS_ALL_USERS'] = str(plan['pool_tokens'])
            edits = ('the memory the third block needs: the KV pool lowered from %d to %d blocks (QWEN36_MAX_TOKENS_ALL_USERS %d) and the trace region raised from %d to %d MiB'
                     % (MEMORY['parent_blocks'], MEMORY['blocks'], plan['pool_tokens'], MEMORY['parent_trace_region'] >> 20, MEMORY['trace_region'] >> 20))
        else:
            edits = 'nothing else (no third block: the KV pool and the trace region are the parent\'s)'
        profile['description'] = ('GATE ONLY, UNQUALIFIED (octo-T8 and the lone-user padded round, docs/tp4-octo.md; -octo twins: the device pieces are built on the host and nothing has run on a card (serving_octo.DEVICE_PIECES, UNQUALIFIED_ITEMS); '
                                  '-solopacked twins: the flag refuses to engage until a block holds three idle segments): %s. The parent is %s and this profile is that parent plus exactly %s and the gate marker QWEN_C2_GATE_PROFILE in its env, '
                                  'and %s (generated by scripts/ci/make_octo_profiles.py: regenerate, never hand-merge).'
                                  % (why, parent_name, ', '.join(sorted(name for name in env if name != 'QWEN_C2_GATE_PROFILE')), edits))
        profile['gate_only'] = True
        generated[name] = profile
    family = [index for index, name in enumerate(profiles) if name.startswith(PARKED_PARENT)]
    last = max(family + [list(profiles).index(PARKED_PARENT)])
    result = {}
    for index, (name, profile) in enumerate(profiles.items()):
        if name in generated:
            continue                                    # a previous generation: replaced below
        result[name] = profile
        if index == last:
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
            sys.stderr.write('%s is not what make_octo_profiles.py generates from its parents: run it with --write\n' % path)
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
