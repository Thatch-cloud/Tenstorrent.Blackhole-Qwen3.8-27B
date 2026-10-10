"""The host KV tier's profiles (docs/prefix-store-hygiene.md), generated from the production and the op-fusion ship profiles (stdlib only).

    python3 scripts/ci/make_prefix_tier_profiles.py --write   regenerate the three profiles in scripts/ci/qwen_c2_profiles.json
    python3 scripts/ci/make_prefix_tier_profiles.py --check   exit 1 if the checked-in profiles are not what the parents generate

NEVER hand-merge these entries into the profiles file. After the branch is merged with another that moves a parent (the production profile, or the op-fusion ship
profile make_fusion_profiles writes), regenerate: each profile is its parent's env, engine and every other field byte for byte plus exactly the env below.

  -tier-audit    GATE ONLY. The production stack (Lever N + prefix reuse + W2 + engine reuse) with the two prefix-store features, every instrument on: the host KV tier
                 (QWEN_PREFIX_HOST_TIER_GIB: evicted attention-KV pages go to host RAM and come back on a returning turn, byte for byte), the GDN checkpoints stored in
                 the device's own tile layout and dtype (QWEN_PREFIX_CKPT_PRECONVERTED: the restore is a straight upload), the read-back of every restored block compared
                 with the bytes written (QWEN_PREFIX_HOST_TIER_AUDIT), the digest of every block checked at restore (QWEN_PREFIX_HOST_TIER_VERIFY=all) and the
                 preconverted tensors compared with today's conversion on card (QWEN_PREFIX_CKPT_PRECONVERTED_AUDIT). The tier's own kill-switch file is a scratch path
                 inside the container (the gate writes it). QWEN_FAST_TP4_SDPA_AUDIT=1 rides along: the parent runs the multi-user SDPA launch (W2), which refuses the
                 extent audit the gate gives every audited arm unless it carries its own audit (c2_serving_gate.multi_audit_refusal). The TIMED tier arm of the
                 tiers plan is this profile derived by c2_prefix_gate (the `tiertime` kind: the same, with the audit instruments off).
  -tierfx-traffic  THE SHIP CANDIDATE, a TRAFFIC profile (not gate_only): the op-fusion ship profile (the fx-traffic profile make_fusion_profiles writes) plus the tier at
                 its production setting (QWEN_PREFIX_HOST_TIER_GIB=32 beside the 8 GiB checkpoint store, QWEN_PREFIX_HOST_TIER_VERIFY=sample, QWEN_PREFIX_HOST_TIER_AUDIT=0)
                 and the preconverted checkpoints (QWEN_PREFIX_CKPT_PRECONVERTED=1, its audit 0): no gate instrument, no kill-switch path (the tier's own file is the hub
                 mount's default, beside prefix-reuse.off). It carries its OWN owner traffic waiver, id and words and evidence below, whose decision sentence starts PENDING:
                 a PENDING profile boots in a gate or a smoke and logs the word, but no build bakes it (c2_serving_job.read_bake). The contract refuses the tier on a traffic
                 profile whose waiver does not name it (serving_c2_contract.prefix_policy_problems).
  -tierfx-audit  GATE ONLY. The audited twin of the above: the fusion programme's combined audit arm (fx-all-audit: every fusion lever plus its _AUDIT flag) plus the
                 -tier-audit env (every tier instrument on, and the multi launch's own audit).

To approve the ship candidate, replace SHIP_DECISION below with a sentence starting APPROVED, run --write and commit (the field that flips is owner_traffic_waiver.decision of
-tierfx-traffic, nothing else).
"""

import argparse
import copy
import json
from pathlib import Path
import sys

PROFILES = Path(__file__).resolve().parent / 'qwen_c2_profiles.json'

PARENT = 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-traffic'
TWIN = 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-tier-audit'
# The op-fusion programme's ship profile (a traffic profile make_fusion_profiles writes) and its combined audit arm: the parents of the ship candidate and its audit twin.
FX_PARENT = 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-fx-traffic'
FX_AUDIT_PARENT = 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-fx-all-audit'
SHIP = 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-tierfx-traffic'
SHIP_AUDIT = 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-tierfx-audit'
# The owner's defaults (32 GiB of host RAM for the whole prefix state, swap off): the checkpoint store keeps its 8 GiB, the KV pages get the rest.
TIER_GIB = '32'
OFF_PATH = '/tmp/qwen-prefix-host-tier.off'        # prefix_replay.TIER_DRILL_OFF_PATH
ENV = {'QWEN_PREFIX_HOST_TIER_GIB': TIER_GIB, 'QWEN_PREFIX_HOST_TIER_AUDIT': '1', 'QWEN_PREFIX_HOST_TIER_VERIFY': 'all',
       'QWEN_PREFIX_CKPT_PRECONVERTED': '1', 'QWEN_PREFIX_CKPT_PRECONVERTED_AUDIT': '1', 'QWEN_PREFIX_HOST_TIER_OFF_PATH': OFF_PATH,
       'QWEN_FAST_TP4_SDPA_AUDIT': '1'}
# The ship candidate's tier env: the production setting, every instrument off.
SHIP_ENV = {'QWEN_PREFIX_HOST_TIER_GIB': TIER_GIB, 'QWEN_PREFIX_HOST_TIER_AUDIT': '0', 'QWEN_PREFIX_HOST_TIER_VERIFY': 'sample',
            'QWEN_PREFIX_CKPT_PRECONVERTED': '1', 'QWEN_PREFIX_CKPT_PRECONVERTED_AUDIT': '0'}
WAIVER_FIELD = 'owner_traffic_waiver'
# The two levers the contract lets a waiver name beside the four of the earlier ship waivers (serving_c2_contract.TIER_WAIVABLE_LEVERS).
TIER_LEVERS = {'QWEN_PREFIX_HOST_TIER_GIB': TIER_GIB, 'QWEN_PREFIX_CKPT_PRECONVERTED': '1'}
SHIP_WAIVER_ID = 'w2-er-fx-tier-262k-2026-10-11'
SHIP_WAIVES = ('everything the parent profile waives (its waiver w2-er-fx-262k-2026-10-11: the W2 and engine-reuse qualification records and the op-fusion levers it names), and, '
               'for the two levers this profile adds, which have no pinned qualification record: the host KV tier (QWEN_PREFIX_HOST_TIER_GIB=32 beside the 8 GiB checkpoint store: '
               'evicted attention-KV pages are held in host RAM and written back to the device pool when their session returns, digest-checked at restore) and the GDN checkpoints stored '
               'in the device\'s tile layout and dtype (QWEN_PREFIX_CKPT_PRECONVERTED=1). Both are default-off and byte-identical off, and exact on cards on the gate-only audit twins '
               '-tier-audit and -tierfx-audit (the evidence below), but the tier has run on cards only beside no concurrent decode of other seats: the concurrent plans (TC1 to TC5: the '
               'decode stall of seven seats while a session is restored and another spills) and the ship gates (G1, GX, G2) have not run on this profile')
SHIP_EVIDENCE = (
    'Q1, the raw block ops on the served four-card mesh (the kvread probe, run 38082194508): raw=ok, bytes read, written into a second cache and read back equal, no program-cache growth, '
    '13.6 GB/s read and 22.8 GB/s write at 4,096 blocks',
    'A1, the audited attach smoke of -tier-audit (run 38082317009): 0 audit mismatches over 936 publish rounds',
    'T1, tier-attach (run 38082865675): three cold/hit pairs IDENTICAL, one restore from a preconverted checkpoint compared on the card, [PREFIX-AUDIT-CKPT] conversion_equal=1 readback_equal=1, '
    'program count unchanged',
    'T2, tier-returning (run 38083229148): 24 fresh-salt floods over a 1,277,952-token pool, three ~56k sessions restored from host RAM at the oracle\'s Q (5 pairs IDENTICAL), 2,435 blocks '
    'digest-checked and read back, zero failure counters, the tier\'s kill switch latched it and the next turn was exact',
    'T3, tier-timed with the audits off (run 38084230415): the restored answer IDENTICAL to its cold twin at L=250,919; first token 3.8 s, 4.1 s and 6.5 s after a session left the device '
    'pool against 8.3 s, 37.6 s and 96.0 s to prefill it again at 32k, 128k and 254k',
)
SHIP_DECISION = ('PENDING: not decided. The tier ships on this profile only after the concurrent plans TC1 to TC5 and the ship gates pass on it and the owner replaces this '
                 'sentence with one starting APPROVED (docs/prefix-store-hygiene.md)')
# Where the profiles sit: after the kill drill's twin of the production profile and after the engine-start upload family (...-w2-er-traffic-<lever>), whose generator
# inserts its block right after the kill twin (so the order here is the one both --check commands reproduce). NOT at the end of the file (the region-read twins, make_kvread_profiles,
# own it) and not in the op-fusion block (make_fusion_profiles owns its namespace).
AFTER = 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-kill'
NAMES = (TWIN, SHIP, SHIP_AUDIT)


def description():
    return ('GATE ONLY, UNQUALIFIED (the host KV tier and the preconverted checkpoints, docs/prefix-store-hygiene.md; never traffic, never a timed arm: the tier gate derives its '
            'timed arm). The parent is %s, the production profile (Lever N + prefix reuse + W2 + engine reuse), and this profile is that parent plus exactly %s and nothing '
            'else (QWEN_FAST_TP4_SDPA_AUDIT=1 is the multi-user SDPA launch\'s own audit, which the gate\'s extent audit needs beside W2), without its owner traffic waiver. Evicted attention-KV pages go to host RAM (%s GiB for the whole prefix state, the checkpoint store keeping its own '
            'share) and a returning turn restores them before vLLM looks for its hit; the GDN checkpoints are stored already converted to the device\'s tile layout and '
            'dtype. Every instrument is on: each restored block is read back and compared with the bytes written, each block\'s digest is checked at restore, and the '
            'preconverted tensors are compared with today\'s conversion. The tier\'s kill-switch file is %s inside the container. Needs the qwen_kv_read extension, '
            'version 2 (the raw block read and write). Generated by scripts/ci/make_prefix_tier_profiles.py: regenerate, never hand-merge.'
            % (PARENT, ', '.join('%s=%s' % pair for pair in sorted(ENV.items())), TIER_GIB, OFF_PATH))


def ship_description():
    return ('The op-fusion SHIP CANDIDATE with the host KV tier (docs/tp4-fusion.md, docs/prefix-store-hygiene.md), a TRAFFIC profile (NOT gate_only): %s plus exactly %s and nothing else; '
            'no gate instrument, no audit, no telemetry, no kill-switch path (the tier\'s own kill-switch file is the hub mount\'s default, beside prefix-reuse.off). The tier holds evicted '
            'attention-KV pages in host RAM (%s GiB for the whole prefix state, the 8 GiB checkpoint store keeping its share) and writes them back when their session returns; the GDN '
            'checkpoints are stored already converted to the device\'s tile layout and dtype. The tier and the preconverted checkpoints serve traffic only under the owner traffic '
            'waiver of this profile (id %s), whose decision sentence starts PENDING until the owner records the approval as a sentence starting APPROVED: a profile whose decision does not '
            'start with APPROVED boots in a gate or a smoke and logs the word, but no build bakes it. The control of the paired timing is %s itself. Needs the qwen_kv_read extension, '
            'version 2. Generated by scripts/ci/make_prefix_tier_profiles.py: regenerate, never hand-merge.'
            % (FX_PARENT, ', '.join('%s=%s' % pair for pair in sorted(SHIP_ENV.items())), TIER_GIB, SHIP_WAIVER_ID, FX_PARENT))


def ship_audit_description():
    return ('GATE ONLY, UNQUALIFIED (the host KV tier on the op-fusion ship stack, docs/prefix-store-hygiene.md; never traffic, never a timed arm). The parent is %s, the op-fusion '
            'programme\'s combined audit arm (every fusion lever of the ship candidate plus its _AUDIT flag), and this profile is that parent plus exactly %s and nothing else, '
            'without an owner traffic waiver: the audited twin of %s. Every tier instrument is on (each restored block is read back and compared, each block\'s digest is checked at restore, '
            'the preconverted tensors are compared with today\'s conversion) and the multi-user SDPA launch carries its own audit, which the gate\'s extent audit needs beside W2. The tier\'s '
            'kill-switch file is %s inside the container. Generated by scripts/ci/make_prefix_tier_profiles.py: regenerate, never hand-merge.'
            % (FX_AUDIT_PARENT, ', '.join('%s=%s' % pair for pair in sorted(ENV.items())), SHIP, OFF_PATH))


def twin_names():
    """The generated GATE-ONLY profiles' names (the profile-enumerating tests exempt them: test_prefix_tier_profiles holds them). The ship candidate is a traffic profile: ship_names."""
    return (TWIN, SHIP_AUDIT)


def ship_names():
    """The generated TRAFFIC profile's name (the ship candidate: not gate-only, so not a twin; profile_twins.generated_names)."""
    return (SHIP,)


def parent_problems(profile):
    """Why the production profile cannot be twinned, [] when it can."""
    problems = []
    env = profile.get('env') or {}
    for name in ENV:
        if name in env:
            problems.append('%s already names %s: a parent never carries the knob it is twinned with' % (PARENT, name))
    if profile.get('gate_only') is True:
        problems.append('%s is gate_only: the parent is the production traffic profile' % PARENT)
    if env.get('QWEN_FAST_TP4_SDPA') != 'multi':
        problems.append('%s does not run the multi-user SDPA launch (QWEN_FAST_TP4_SDPA=multi): the SDPA audit this twin adds would have nothing to audit' % PARENT)
    if env.get('QWEN_PREFIX_REUSE') != '1':
        problems.append('%s does not run prefix reuse (QWEN_PREFIX_REUSE=1): the tier has nothing to hold' % PARENT)
    if float(env.get('QWEN_PREFIX_STORE_GIB') or 8) >= float(TIER_GIB):
        problems.append('%s names QWEN_PREFIX_STORE_GIB=%s, not under the tier\'s %s GiB: nothing would be left for KV pages' % (
            PARENT, env.get('QWEN_PREFIX_STORE_GIB'), TIER_GIB))
    return problems


def ship_problems(name, profile, traffic):
    """Why the op-fusion profile `name` cannot be the parent of the ship candidate (traffic) or its audit twin (not), [] when it can."""
    problems = []
    env = profile.get('env') or {}
    for flag in list(ENV) + list(SHIP_ENV):
        if flag in env:
            problems.append('%s already names %s: a parent never carries the knob it is twinned with' % (name, flag))
    if traffic:
        waiver = profile.get(WAIVER_FIELD)
        if profile.get('gate_only') is True or not isinstance(waiver, dict) or not isinstance(waiver.get('levers'), dict):
            problems.append('%s is not a traffic profile with an owner traffic waiver naming levers: the ship candidate extends that waiver' % name)
    elif profile.get('gate_only') is not True:
        problems.append('%s is not gate_only: the audit twin\'s parent is the fusion programme\'s audit arm' % name)
    if env.get('QWEN_PREFIX_REUSE') != '1':
        problems.append('%s does not run prefix reuse (QWEN_PREFIX_REUSE=1): the tier has nothing to hold' % name)
    return problems


def anchor_of(names):
    """The name the block follows: the kill drill's twin when the file has it (else the parent), then every consecutive entry of the engine-start upload family
    (...-w2-er-traffic-<lever>[-audit]) that follows it, because make_upload_profiles inserts its block right after the kill twin."""
    index = names.index(AFTER) if AFTER in names else names.index(PARENT)
    while index + 1 < len(names) and names[index + 1].startswith(PARENT + '-'):
        index += 1
    return names[index]


def generate(data):
    """data's profiles with the three profiles inserted after anchor_of() (every other entry unmoved: the other generators own the order of their own blocks); returns the new dict."""
    profiles = data['profiles']
    for name in (PARENT, FX_PARENT, FX_AUDIT_PARENT):
        if name not in profiles:
            raise ValueError('the parent profile %s is not defined' % name)
    problems = parent_problems(profiles[PARENT]) + ship_problems(FX_PARENT, profiles[FX_PARENT], True) + ship_problems(FX_AUDIT_PARENT, profiles[FX_AUDIT_PARENT], False)
    if problems:
        raise ValueError('; '.join(problems))
    twin = copy.deepcopy(profiles[PARENT])
    twin['env'].update(ENV)
    twin['description'] = description()
    twin.pop(WAIVER_FIELD, None)
    twin.pop('gate_only', None)
    twin['gate_only'] = True
    ship = copy.deepcopy(profiles[FX_PARENT])
    ship['env'].update(SHIP_ENV)
    ship['description'] = ship_description()
    levers = dict(ship[WAIVER_FIELD]['levers'])
    levers.update(TIER_LEVERS)
    ship[WAIVER_FIELD] = dict(id=SHIP_WAIVER_ID, levers=levers, waives=SHIP_WAIVES, evidence=list(SHIP_EVIDENCE), decision=SHIP_DECISION)
    ship.pop('gate_only', None)
    audit = copy.deepcopy(profiles[FX_AUDIT_PARENT])
    audit['env'].update(ENV)
    audit['description'] = ship_audit_description()
    audit.pop(WAIVER_FIELD, None)
    audit.pop('gate_only', None)
    audit['gate_only'] = True
    remaining = [name for name in profiles if name not in NAMES]       # a previous generation is replaced below
    after = anchor_of(remaining)
    result = {}
    for name in remaining:
        result[name] = profiles[name]
        if name == after:
            result[TWIN], result[SHIP], result[SHIP_AUDIT] = twin, ship, audit
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
            sys.stderr.write('%s is not what make_prefix_tier_profiles.py generates from its parents: run it with --write\n' % path)
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
