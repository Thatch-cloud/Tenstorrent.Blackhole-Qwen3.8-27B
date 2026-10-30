"""The NVMe cold tier's gate-only twins of the ship-7 candidate (tp4/kv-nvme-2, docs/tp4-kv-nvme.md section 12), generated from the files they are made of (stdlib only).

    python3 scripts/ci/make_nvme_profiles.py --write   regenerate the twins in scripts/ci/qwen_c2_profiles.json
    python3 scripts/ci/make_nvme_profiles.py --check   exit 1 if the checked-in twins are not what their parents generate

NEVER hand-merge these entries into the profiles file. Run make_fusion_profiles.py --write FIRST (it writes the ship-7 candidate, `-fx-traffic-s7`), then the generators of the tp4/s7 chain
(make_octo3_profiles.py ... make_s6_profiles.py, make_sched2_profiles.py, make_pf3_profiles.py, make_pf3k_profiles.py, make_s7_profiles.py: the audited twin of the candidate is the audit
arm's parent), then this. The twins sit right after the ship-7 audited twin, so each generator's --check holds whichever of them ran last.

THE THREE TWINS (all gate_only, unqualified, none with an owner traffic waiver; the cards' first NVMe gates, docs/tp4-kv-nvme.md section 12):

  -nvme-s7         THE TIMED ARM:   the ship-7 candidate (`-fx-traffic-s7`, the profile the s7c image bakes as its default) + the NVMe tier. The candidate's owner waiver id
                   (QWEN_FAST_OCTO_WAIVER) is replaced by the gate marker QWEN_C2_GATE_PROFILE=1, as the sched-1 and sched-2 timed arms are: a gate-only profile admits the octo shape through
                   its marker and the contract refuses a waiver id on it. Nothing else of the candidate moves (the ledger trim stays: no audit reads the ledger here).
  -nvme-s7-audit   THE AUDITED ARM: the ship-7 audited twin (`-fx-s7-audit`, every audit of the shipped stack) + the NVMe tier + QWEN_PREFIX_NVME_AUDIT=1 (every object read back after
                   its write; the tier's digests all).
  -nvme-s7-fmt     THE FIRST GATE:  the audited arm + QWEN_PREFIX_NVME_FORMAT=1 (the gate-only force-format). The drive today holds only peer-to-peer test patterns, which the tier's default
                   `auto` format policy reads as not blank and not ours, so the first gate formats it once; every later gate uses the two arms above, which open the store's own superblock.

WHAT THE NVMe TIER ADDS to each (QWEN_PREFIX_NVME* names, qwen_kv_nvme.NAMES): QWEN_PREFIX_NVME=1; QWEN_PREFIX_NVME_DEVICE=/dev/c2-cache-drive (the fixed in-container path: the job runner
maps the drive it resolved from C2_NVME_DEVICE there, scripts/ci/c2_cache_drive.py IN_CONTAINER_DEVICE); the retention acknowledgement QWEN_PREFIX_NVME_RETENTION_ACK=1 (the contract and the
engine refuse the tier without it); the gate-only override QWEN_PREFIX_NVME_ALLOW_ON_ZERO_RETENTION=1 (the gate's host is a rig, not a zero-retention node, but may carry the signals).
QWEN_PREFIX_NVME_REQUIRE_SED stays UNSET, so the drive must be an Opal self-encrypting drive with locking enabled and unlocked (the status file the boot unit writes, mounted by the runner).

THE HOST TIER'S SHARE is cut: QWEN_PREFIX_HOST_TIER_GIB = the parent's QWEN_PREFIX_STORE_GIB + 4, so the host tier's KV pages (the GiB less the checkpoint store's budget,
qwen_prefix_registry.tier_config) are 4 GiB, smaller than a 254k session's 8.56 GB: every restore of the big sessions in the arm has to come from the NVMe. The control of every timed pair is
the ship-7 candidate itself (its 48 GiB host tier).
"""

import argparse
import copy
import json
from pathlib import Path
import sys

import make_s5_profiles as s5
import make_s7_profiles as s7

PROFILES = Path(__file__).resolve().parent / 'qwen_c2_profiles.json'

# Not under make_fusion_profiles.NAMESPACE (`...-er-fx-`), whose --write drops every name under it that it does not list as another generator's: the sched-1 and sched-2 twins are named the same way.
NAMESPACE = 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-nvme-s7'
SHIP = s7.SHIP                                       # the ship-7 candidate: the parent of the timed arm and the control of every pair
AUDIT_PARENT = s7.AUDIT                              # the ship-7 audited twin: the parent of the other two
TIMED = NAMESPACE
AUDIT = NAMESPACE + '-audit'
FORMAT = NAMESPACE + '-fmt'
NAMES = (TIMED, AUDIT, FORMAT)
WAIVER_ENV = s5.WAIVER_ENV                           # QWEN_FAST_OCTO_WAIVER: a traffic profile's, replaced by the gate marker in the timed arm
WAIVER_FIELD = s5.WAIVER_FIELD
MARKER_ENV = {'QWEN_C2_GATE_PROFILE': '1'}
MAX_NAME = 64

IN_CONTAINER_DEVICE = '/dev/c2-cache-drive'          # scripts/ci/c2_cache_drive.py IN_CONTAINER_DEVICE (test_nvme_profiles holds the two equal)
TIER_GIB = 'QWEN_PREFIX_HOST_TIER_GIB'
STORE_GIB = 'QWEN_PREFIX_STORE_GIB'
KV_SHARE_GIB = 4                                     # the host tier's KV pages: under a 254k session (8.56 GB), so every restore of it is the NVMe's
REQUIRE_SED = 'QWEN_PREFIX_NVME_REQUIRE_SED'

NVME_ENV = {
    'QWEN_PREFIX_NVME': '1',
    'QWEN_PREFIX_NVME_DEVICE': IN_CONTAINER_DEVICE,
    'QWEN_PREFIX_NVME_RETENTION_ACK': '1',
    'QWEN_PREFIX_NVME_ALLOW_ON_ZERO_RETENTION': '1',
}
AUDIT_ENV = {'QWEN_PREFIX_NVME_AUDIT': '1'}
FORMAT_ENV = {'QWEN_PREFIX_NVME_FORMAT': '1'}
# (name, parent kind: 'ship' | 'audit', env additions beyond the tier's own, what it is)
ARMS = (
    (TIMED, 'ship', {}, 'THE TIMED ARM'),
    (AUDIT, 'audit', AUDIT_ENV, 'THE AUDITED ARM'),
    (FORMAT, 'audit', dict(AUDIT_ENV, **FORMAT_ENV), 'THE FIRST GATE (THE ONE THAT FORMATS THE DRIVE)'),
)


def tier_share(profile):
    """The QWEN_PREFIX_HOST_TIER_GIB string for the parent: its checkpoint store's GiB plus KV_SHARE_GIB, whole numbers printed whole ('16' -> '20')."""
    store = float((profile.get('env') or {}).get(STORE_GIB))
    total = store + KV_SHARE_GIB
    return str(int(total)) if total == int(total) else repr(total)


def description(name, what, parent, env, share):
    shown = ', '.join('%s=%s' % pair for pair in sorted(dict(env, **{TIER_GIB: share}).items()))
    extra = ''
    if name == TIMED:
        extra = ('Its owner waiver id %s is replaced by the gate marker %s=1 (a gate-only profile admits the octo shape through its marker and the contract refuses a waiver id on it); nothing else of the '
                 'candidate moves, the memory-ledger trim included. ' % (WAIVER_ENV, ', '.join(sorted(MARKER_ENV))))
    return ('GATE ONLY, UNQUALIFIED (tp4/kv-nvme-2, docs/tp4-kv-nvme.md section 12; never traffic). %s of the NVMe cold tier\'s card gates. The parent is %s, and this profile is that parent plus exactly '
            '%s and nothing else, without an owner traffic waiver; the host tier\'s share is the checkpoint store\'s GiB plus %d (QWEN_PREFIX_HOST_TIER_GIB less QWEN_PREFIX_STORE_GIB = %d GiB of KV pages, '
            'under a 254k session\'s 8.56 GB), so every restore of a big session has to come from the NVMe. %sQWEN_PREFIX_NVME_REQUIRE_SED is unset: the drive must be an Opal self-encrypting drive, locking '
            'enabled and unlocked (the runner mounts the boot unit\'s status file). QWEN_PREFIX_NVME_DEVICE is the fixed in-container path the job runner maps the resolved drive to. '
            'Generated by scripts/ci/make_nvme_profiles.py: regenerate, never hand-merge.' % (what, parent, shown, KV_SHARE_GIB, KV_SHARE_GIB, extra))


def specs():
    """[(profile name, parent kind, env additions, what)] in the order they are written."""
    return [(name, kind, dict(NVME_ENV, **flags), what) for name, kind, flags, what in ARMS]


def twin_names():
    """The generated twins' names (the profile-enumerating tests exempt them through profile_twins; test_nvme_profiles holds them)."""
    return NAMES


def chain_names():
    """The names of the generators that run before this one: the twins this generator's own are placed after (the tp4/s7 chain, the audited twin of the candidate last)."""
    return tuple(s7.chain_names()) + tuple(s7.twin_names())


def parent_problems(profiles):
    """Why the twins cannot be generated, [] when they can."""
    import qwen_kv_nvme

    problems = []
    for name in (SHIP, AUDIT_PARENT):
        if name not in profiles:
            problems.append('the parent profile %s is not defined (make_fusion_profiles.py --write, then make_s7_profiles.py --write first)' % name)
    problems += ['%s is %d characters, over the %d a profile name may have' % (name, len(name), MAX_NAME) for name in NAMES if len(name) > MAX_NAME]
    if problems:
        return problems
    audit, ship = profiles[AUDIT_PARENT], profiles[SHIP]
    if audit.get('gate_only') is not True or (audit.get('env') or {}).get('QWEN_C2_GATE_PROFILE') != '1':
        problems.append('%s is not the gate-only audited twin of ship 7' % AUDIT_PARENT)
    waiver = ship.get(WAIVER_FIELD)
    if ship.get('gate_only') is True or not isinstance(waiver, dict):
        problems.append('%s is not a traffic profile with an owner traffic waiver' % SHIP)
    elif ship['env'].get(WAIVER_ENV) != waiver.get('id'):
        problems.append('%s does not carry its waiver id in %s' % (SHIP, WAIVER_ENV))
    for name in (SHIP, AUDIT_PARENT):
        env = profiles[name]['env']
        for flag in qwen_kv_nvme.NAMES:
            if flag in env:
                problems.append('%s already names %s: a parent never carries the tier it is twinned with' % (name, flag))
        if profiles[name].get('mesh_device') != 'P150x4' or env.get('QWEN_FAST_TP') != '4':
            problems.append('%s is not a four-card TP4 profile' % name)
        try:
            tier, store = float(env.get(TIER_GIB)), float(env.get(STORE_GIB))
        except (TypeError, ValueError):
            problems.append('%s names no numeric %s / %s: the host tier\'s share cannot be cut' % (name, TIER_GIB, STORE_GIB))
            continue
        if tier <= store + KV_SHARE_GIB:
            problems.append('%s: %s=%s is not above %s + %d: the share would not be a cut' % (name, TIER_GIB, env.get(TIER_GIB), STORE_GIB, KV_SHARE_GIB))
    return problems


def generate(data):
    """data's profiles with the twins inserted right after the last twin of the generators before this one (every other entry unmoved); returns the new dict."""
    profiles = data['profiles']
    problems = parent_problems(profiles)
    if problems:
        raise ValueError('; '.join(problems))
    generated = {}
    for name, kind, flags, what in specs():
        parent = SHIP if kind == 'ship' else AUDIT_PARENT
        twin = copy.deepcopy(profiles[parent])
        if kind == 'ship':
            twin['env'].pop(WAIVER_ENV)
            twin['env'].update(MARKER_ENV)
        twin['env'].update(flags)
        share = tier_share(profiles[parent])
        twin['env'][TIER_GIB] = share
        twin['description'] = description(name, what, parent, flags, share)
        twin.pop(WAIVER_FIELD, None)
        twin.pop('gate_only', None)
        twin['gate_only'] = True
        generated[name] = twin
    remaining = [name for name in profiles if name not in generated]            # a previous generation is replaced below
    anchors = [name for name in remaining if name in chain_names()]
    if not anchors:
        raise ValueError('the tp4/s7 twins are not in the profiles file (make_s7_profiles.py --write first)')
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
            sys.stderr.write('%s is not what make_nvme_profiles.py generates from its parents: run it with --write\n' % path)
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
