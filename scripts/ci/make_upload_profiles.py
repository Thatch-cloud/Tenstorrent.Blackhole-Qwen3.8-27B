"""The engine-start upload twins (qwen_device_zeros, qwen_lazy_shard; docs/tp4-fabric-upload.md section 5, P0 and P1a), generated from the production profile (stdlib only).

    python3 scripts/ci/make_upload_profiles.py --write   regenerate the twins in scripts/ci/qwen_c2_profiles.json
    python3 scripts/ci/make_upload_profiles.py --check   exit 1 if the checked-in twins are not what the parent generates

NEVER hand-merge these entries into the profiles file. After the branch is merged with another that moves the parent (the ship profile), regenerate: each twin is the
parent's env, engine and every other field byte for byte, minus the owner traffic waiver (a gate-only profile carries none), plus exactly the env below, and gate_only.
The parent itself, unaudited and flag-off, is the CONTROL of every pair: the engine-start A/B (references/upload-p0-jobs) times the parent against a twin, and the
attach smoke compares the parent's answers with an audited twin's.

The twins (TWINS below; a suffix names the levers):
  -dzero         QWEN_FAST_DEVICE_ZEROS=1                                  the KV pool and the buffer pool's zero buffers filled on the card (P0), no audit
  -dzero-audit   QWEN_FAST_DEVICE_ZEROS=1 + _AUDIT=1                       the same, with the byte audit of the first tensors of each user against a host-built twin
  -lazy          QWEN_FAST_LAZY_SHARD_W=1                                  a weight-cache hit without the host transpose (P1a), no audit
  -lazy-audit    QWEN_FAST_LAZY_SHARD_W=1 + _AUDIT=1                       the same, with the first loads compared byte for byte with the stock conversion
  -upload        both levers                                               what production would run
  -upload-audit  both levers + both audits                                 the audited attach smoke
Every twin is gate_only. The production profile and every non-gate profile carry no QWEN_FAST_DEVICE_ZEROS / QWEN_FAST_LAZY_SHARD_W name until the audited arms have
passed on cards and the owner moves them (serving_c2_contract.upload_problems holds the audit flags to gate profiles).
"""

import argparse
import copy
import json
from pathlib import Path
import sys

PROFILES = Path(__file__).resolve().parent / 'qwen_c2_profiles.json'

PARENT = 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-traffic'
KILL_TWIN = 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-kill'      # make_w2_kill_profiles' twin of the same parent
WAIVER_FIELD = 'owner_traffic_waiver'

ZEROS = 'QWEN_FAST_DEVICE_ZEROS'
ZEROS_AUDIT = 'QWEN_FAST_DEVICE_ZEROS_AUDIT'
LAZY = 'QWEN_FAST_LAZY_SHARD_W'
LAZY_AUDIT = 'QWEN_FAST_LAZY_SHARD_W_AUDIT'
KNOBS = (ZEROS, ZEROS_AUDIT, LAZY, LAZY_AUDIT, 'QWEN_FAST_DEVICE_ZEROS_MIN_BYTES', 'QWEN_FAST_DEVICE_ZEROS_AUDIT_TENSORS',
         'QWEN_FAST_DEVICE_ZEROS_AUDIT_BLOCKS', 'QWEN_FAST_LAZY_SHARD_W_AUDIT_LOADS')

# (suffix, env additions): the order the twins are written in, appended at the end of the file.
TWINS = (
    ('dzero', {ZEROS: '1'}),
    ('dzero-audit', {ZEROS: '1', ZEROS_AUDIT: '1'}),
    ('lazy', {LAZY: '1'}),
    ('lazy-audit', {LAZY: '1', LAZY_AUDIT: '1'}),
    ('upload', {ZEROS: '1', LAZY: '1'}),
    ('upload-audit', {ZEROS: '1', ZEROS_AUDIT: '1', LAZY: '1', LAZY_AUDIT: '1'}),
)

WHAT = {
    ZEROS: ('QWEN_FAST_DEVICE_ZEROS=1 allocates the paged KV pool (32 tensors) and the buffer pool\'s big replicated tiled zeros on the card and fills them there '
            '(ttnn.empty + the in-place SFPU fill) instead of building them on the host and writing them over PCIe'),
    LAZY: ('QWEN_FAST_LAZY_SHARD_W=1 moves the host bf16 transpose of every tp_common.shard_w weight into as_tensor\'s preprocess callback, so a weight-cache '
           'hit (the cache file is the finished device tensor, its name unchanged) does no host copy'),
}
AUDITS = {
    ZEROS_AUDIT: ('QWEN_FAST_DEVICE_ZEROS_AUDIT=1 reads back a sample of pages of the first tensors from the device-filled tensor and from a host-built twin and '
                  'compares the packed bytes per chip; any difference latches the lever off and the host path builds the tensor (logs "[PINDIAG] tp4 device zeros '
                  'audit exact=True" or "audit mismatch")'),
    LAZY_AUDIT: ('QWEN_FAST_LAZY_SHARD_W_AUDIT=1 converts the first loaded weights on the host the stock way and compares the packed bytes per chip with what '
                 'as_tensor returned (logs "[PINDIAG] tp4 lazy shard audit exact=True" or "audit mismatch")'),
}


def twin_name(suffix):
    return '%s-%s' % (PARENT, suffix)


def description(suffix, env):
    parts = [WHAT[name] for name in (ZEROS, LAZY) if name in env] + [AUDITS[name] for name in (ZEROS_AUDIT, LAZY_AUDIT) if name in env]
    return ('GATE ONLY, UNQUALIFIED (the engine-start upload levers, docs/tp4-fabric-upload.md P0 and P1a; never traffic, never a speed claim beyond load time). '
            'The parent is %s, the production profile, and this profile is that parent plus exactly %s and nothing else, without its owner traffic waiver. %s. '
            'Generated by scripts/ci/make_upload_profiles.py: regenerate, never hand-merge.'
            % (PARENT, ' and '.join('%s=%s' % (name, env[name]) for name in sorted(env)), '. '.join(parts)))


def specs():
    """[(profile name, env additions, description)] in the order they are written."""
    return [(twin_name(suffix), env, description(suffix, env)) for suffix, env in TWINS]


def twin_names():
    """The generated twins' names (the profile-enumerating tests exempt them through profile_twins; test_upload_p0_profiles holds each)."""
    return tuple(name for name, _env, _description in specs())


def parent_problems(profile):
    """Why the parent cannot be twinned, [] when it can."""
    problems = []
    env = profile.get('env') or {}
    for name in KNOBS:
        if name in env:
            problems.append('%s already names %s: a parent never carries the lever it is twinned with' % (PARENT, name))
    if profile.get('gate_only') is True:
        problems.append('%s is gate_only: the parent is the production traffic profile' % PARENT)
    if profile.get('mesh_device') != 'P150x4':
        problems.append('%s does not open the four-card mesh: the upload levers are written for it' % PARENT)
    if env.get('QWEN_FAST_TP') != '4':
        problems.append('%s is not a TP4 profile (QWEN_FAST_TP=4)' % PARENT)
    return problems


def anchor(profiles):
    """The entry the twins follow: the W2 kill drill's twin when the file has it (make_w2_kill_profiles keeps that one right after the parent), else the parent. The file's LAST
    block belongs to make_kvread_profiles (its tests look for its twins at the end), so these are inserted in the middle."""
    return KILL_TWIN if KILL_TWIN in profiles else PARENT


def generate(data):
    """data's profiles with the twins inserted right after anchor() (every other entry unmoved: the other generators own the order of their own blocks); returns the new dict."""
    profiles = data['profiles']
    if PARENT not in profiles:
        raise ValueError('the parent profile %s is not defined' % PARENT)
    problems = parent_problems(profiles[PARENT])
    if problems:
        raise ValueError('; '.join(problems))
    generated = {}
    for name, env, text in specs():
        twin = copy.deepcopy(profiles[PARENT])
        twin['env'].update(env)
        twin['description'] = text
        twin.pop(WAIVER_FIELD, None)
        twin.pop('gate_only', None)
        twin['gate_only'] = True
        generated[name] = twin
    remaining = {name: profile for name, profile in profiles.items() if name not in generated}     # a previous generation is replaced below
    after = anchor(remaining)
    result = {}
    for name, profile in remaining.items():
        result[name] = profile
        if name == after:
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
            sys.stderr.write('%s is not what make_upload_profiles.py generates from its parent: run it with --write\n' % path)
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
