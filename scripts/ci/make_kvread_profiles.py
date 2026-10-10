"""The region-read audit twins (docs/prefix-audit-cost.md), generated from their parents (stdlib only).

    python3 scripts/ci/make_kvread_profiles.py --write   regenerate the twins in scripts/ci/qwen_c2_profiles.json
    python3 scripts/ci/make_kvread_profiles.py --check   exit 1 if the checked-in twins are not what the parents generate

NEVER hand-merge these entries into the profiles file. After the branch is merged with another that moves a parent, regenerate: each twin is its parent's env,
engine and every other field byte for byte plus exactly the env below (and gate_only).

Why: the Lever N audit (QWEN_FAST_LEVERN_AUDIT=1) digests the KV of every prefill of at most 32,785 tokens by reading EVERY cache tensor of the whole pool to the
host, about 8.4 minutes a request at eight seats x 262k, so an audited arm of nine requests is over an hour and a half of reads (the engine-reuse A0 and A1 jobs
timed out on it). QWEN_FAST_LEVERN_KV_READ=region reads only the blocks the row's page table names through the qwen_kv_read extension (the same extension the
prefix audit uses, optimisation/ttnn-op/kv_region_read), byte for byte the same values, and the digest is the same digest. The default (unset, full) is the
whole-cache read, so every existing audited profile is untouched; a twin is the parent plus the flag.

The twins (TWINS below; a suffix names the mode):
  -kvr  QWEN_FAST_LEVERN_KV_READ=region   the audited arm whose digest reads only the row's blocks (the arm to use once the region read is QUALIFIED)
  -kvx  QWEN_FAST_LEVERN_KV_READ=cross    the QUALIFICATION arm: the region read AND the whole-cache read of the first digest (QWEN_FAST_LEVERN_KV_CROSS_STEPS,
                                          default 1), compared byte for byte, and the whole-cache bytes feed the digest, so a wrong region read is counted
                                          ('[PINDIAG] lever N kv read cross ... mismatched=') and can never make two arms agree. It pays one whole-pool read
                                          (about 8.4 minutes) inside a job that already boots the engine.

Every twin is gate_only. The production profile and every non-gate profile carry no QWEN_FAST_LEVERN_KV_ name (serving_c2_contract.levern_problems holds it).
"""

import argparse
import copy
import json
from pathlib import Path
import sys

PROFILES = Path(__file__).resolve().parent / 'qwen_c2_profiles.json'

READ = 'QWEN_FAST_LEVERN_KV_READ'
CROSS_STEPS = 'QWEN_FAST_LEVERN_KV_CROSS_STEPS'
AUDIT = 'QWEN_FAST_LEVERN_AUDIT'

BASE = 'c2-packed-tp4-8x262k-ship-prefix-levern-'
# (parent, twin, mode): the audited Lever N profiles the Lever N + prefix and engine-reuse windows boot at eight seats x 262k (the octo twin's audit and the W2 ones
# are not twinned: octo's generator owns its block of the file and every W2 twin is one more carrier for the W2 lever census tests; add one here when its audited job
# runs, with its census exemptions). The engine-reuse twin is NOT named
# '<parent>-kvr': make_octo_profiles and make_parked_profiles anchor on (and regenerate) every name that starts with '...-levern-parked'.
TWINS = (
    (BASE + 'audit', BASE + 'audit-kvr', 'region'),
    (BASE + 'audit-r2', BASE + 'audit-r2-kvr', 'region'),
    (BASE + 'audit-r2', BASE + 'audit-r2-kvx', 'cross'),
    (BASE + 'parked-audit', BASE + 'kvr-parked-audit', 'region'),
)

COMMON = ('GATE ONLY, UNQUALIFIED (the parent\'s waiver and marker are carried unchanged; docs/prefix-audit-cost.md). The parent is %s and this profile is that parent '
          'plus %s and nothing else. The Lever N audit digests the KV of every prefill of at most 32,785 tokens; by default that reads every cache tensor of the whole pool to '
          'the host (about 8.4 minutes a request at eight seats x 262k). ')

REGION = ('THIS ARM: %s=region reads only the blocks the row\'s page table names, as raw page ranges through the qwen_kv_read extension (the image bakes it), into a '
          'host tensor of just those blocks that ttnn.to_torch unpacks as it unpacks a whole cache, so the digest is the same digest in a time that follows the '
          'prompt. An image without the extension refuses at the attach; a read that raises falls back to the whole-cache read and the smoke check fails the arm '
          '(c2_smoke_check.levern_kv_read_problems). Run it only after the region read is qualified (the cross arm and the kvread probe).' % READ)
CROSS = ('THIS ARM, the QUALIFICATION of the region read on a real prompt: %s=cross reads the blocks through the extension AND every cache whole for the first '
         'digest (%s, default 1), compares the two selections byte for byte and digests the WHOLE-cache bytes, so a wrong region read is counted '
         '("[PINDIAG] lever N kv read cross ... mismatched=") and cannot make two arms agree. It pays one whole-pool read inside a job that already boots the engine.'
         % (READ, CROSS_STEPS))


def specs():
    """[(profile name, parent, env additions, description)] in the order they are written."""
    return [(twin, parent, {READ: mode}, COMMON % (parent, '%s=%s' % (READ, mode)) + (REGION if mode == 'region' else CROSS))
            for parent, twin, mode in TWINS]


def twin_names():
    """The generated twins' names (the profile-enumerating tests of the other lever branches exempt them: test_kvread_profiles holds each)."""
    return tuple(name for name, _parent, _env, _description in specs())


def parent_problems(name, profile):
    """Why a parent cannot be twinned, [] when it can."""
    problems = []
    env = profile.get('env') or {}
    if profile.get('gate_only') is not True:
        problems.append('%s is not gate_only: the audit twins are gate arms' % name)
    if env.get(AUDIT) != '1':
        problems.append('%s does not carry %s=1: there is no audit digest to read' % (name, AUDIT))
    for flag in (READ, CROSS_STEPS):
        if flag in env:
            problems.append('%s already names %s: a parent never carries the knob it is twinned with' % (name, flag))
    return problems


def generate(data):
    """data's profiles with the twins appended at the end (every other entry unmoved: the other generators own the order of their own blocks, and the
    tests that hold it look for their twins right after a parent); returns the new dict."""
    profiles = data['profiles']
    generated = {}
    for name, parent, env, description in specs():
        if parent not in profiles:
            raise ValueError('the parent profile %s is not defined' % parent)
        problems = parent_problems(parent, profiles[parent])
        if problems:
            raise ValueError('; '.join(problems))
        profile = copy.deepcopy(profiles[parent])
        profile['env'].update(env)
        profile['description'] = description
        profile['gate_only'] = True
        generated[name] = profile
    result = {}
    for name, profile in profiles.items():
        if name in generated:
            continue                                    # a previous generation: replaced below
        result[name] = profile
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
            sys.stderr.write('%s is not what make_kvread_profiles.py generates from its parents: run it with --write\n' % path)
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
