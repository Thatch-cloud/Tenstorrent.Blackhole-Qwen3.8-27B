"""G1 provenance for a freshly built C2 serving image: check it is what its context says.

Run by build-c2-serving-image.sh on the rig host right after `docker build`, before the image
gets its tag; a problem exits 1, so a failed image is never tagged, smoked or pushed.

(a) binaries: the installed _ttnncpp.so (both paths the K64i graft is copied to) and _ttnn.so
    are the graft's bytes and carry the graft's QWEN_ strings; a QWEN_ flag the image or a
    profile SETS that the P8 base's binaries read but the installed ones do not is a failure. A
    graft replaces the whole binary, so a patch the base carried and the graft's source tree
    lacked is silently dropped and its flag goes inert (memory graft-so-drops-image-patches:
    K64c lost QWEN_SDPA_TREE_SCRATCH_ROUNDS). Each op directory the JIT compiles from
    (attn_prep, nlp_concat_heads_decode, sdpa_decode, sdpa) must be the context's K64i copy,
    tree for tree: a wrong kernel directory is silent at run time.
(b) overlay: every destination docker/qwen-c2-overlay.txt names holds the sha256 of its source
    in the build context; the install record says which base files the overlay changed. The
    image's revision label and /opt/qwen-c2/source-revision name the commit the context was
    staged from, and /opt/qwen-c2/build-stamp says the build ran the context's own
    build-c2-serving-image.sh.
(c) boot: for the default profile and every named one, the `[QWEN-C2] profile <p>: vLLM argv
    [...]` line the image logs equals the argv the context's serving_c2_contract computes for
    the same platform argv; and exact's QWEN_ environment, as a process in the image sees it
    after the boot hook, is the v235 gate's (run 36087022223 m3native-gate.json
    qwen_configuration, docker/qwen-c2-v235-environment.json) - not the contract's own idea of
    it (memory read-the-launched-argv).
(d) layers: every file of the image's /experiment-scripts/ci and /speculative-decoding/harness
    that the layer model (layers.json, c2_image_layers.expected_tree) names holds its modelled
    version or HEAD's. Anything else - bytes neither version has, or a file the model says the
    image carries and it does not - means the closure verdicts of test_c2_image_overlay are
    wrong for that file, and fails unless ACCEPTED_LAYER_DRIFT reviewed those exact bytes.
(e) frozen pins: every source frozen_combined_runtime.qualify pins (the frozen evidence in
    /experiment-scripts/ci/frozen-evidence) holds its pinned bytes. Exact runs qualify on every
    request and a mismatch kills the engine in execute_model (image v42, run 35495227738).
(f) prefix reuse (the TT prefix-reuse design, G1): the stage record /opt/qwen-c2/prefix-stage.json
    (qwen_prefix_stage.py) says every target held its pinned original, the image still holds what the
    stage wrote, each stage module that ran is the context's overlaid source, and the stage table is
    this checkout's; and every profile's launched argv and booted environment say prefix reuse is on
    exactly where the profile turns it on: --enable-prefix-caching, --enable-chunked-prefill,
    --no-async-scheduling and QWEN_PREFIX_REUSE=1 under general-prefix, --no-enable-prefix-caching and
    no QWEN_PREFIX_REUSE under every other profile (memory read-the-launched-argv).
(i) with --checkout, every image tree file the checkout also has is compared with it and the
    differences are listed. Informational only: (d) is the gate.

--base-drift runs (a)'s strings, (d), (e) and (f)'s anchors (the prefix stage's pinned originals)
against the P8 base image alone, read-only and before any build, to show what the first build would
fail on.

Stdlib only, python >= 3.6. Every container runs with --network none.
"""

import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c2_overlay  # noqa: E402
import qwen_prefix_stage  # noqa: E402

# The K64i binaries and where docker/qwen-c2-serving.Dockerfile copies them (its first RUN).
GRAFT_BINARIES = (
    ('_ttnn.so', '/opt/tt-metal/ttnn/ttnn/_ttnn.so'),
    ('_ttnncpp.so', '/opt/tt-metal/build_Release/ttnn/_ttnncpp.so'),
    ('_ttnncpp.so', '/opt/tt-metal/build_Release/lib/_ttnncpp.so'),
)
# The K64i op directories and where the Dockerfile's second loop puts them.
GRAFT_OP_DIRS = (
    ('attn_prep', '/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/attn_prep'),
    ('nlp_concat_heads_decode', '/opt/tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode'),
    ('sdpa_decode', '/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode'),
    ('sdpa', '/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa'),
)
GRAFT_IN_IMAGE = '/opt/qwen-c2/opgraft-K64i'
# QWEN_ flags the base binaries read, the image sets, and the graft is KNOWN not to read, each
# with the reason that is acceptable (e.g. the gate that qualified the graft ran without it
# too). Empty until a build reports one; never add a flag without reading which patch it is.
ACCEPTED_BASE_ONLY = {}
# Image tree files (d) found at bytes neither the layer model nor HEAD has, reviewed:
# {image path: (the image's sha256, or None for a file the model expects and the image lacks,
# reason)}. Empty until a build (or --base-drift) reports one; each entry says why the file
# is harmless where the image holds it - an entry is a statement that the closure model is
# wrong there and nothing in C2 depends on it being right.
ACCEPTED_LAYER_DRIFT = {}
# Frozen-recipe pins (e) found broken in an image, reviewed: {image path: (the image's sha256, reason)}.
# Empty, and it should stay so: an entry is admissible only with evidence that exact never reaches
# that pin (e.g. --base-drift shows the P8 base holds the same bytes AND a v235-geometry gate passed
# on that base), never to get a build through.
ACCEPTED_PIN_MISMATCH = {}
INSTALL_RECORD = '/opt/qwen-c2/overlay-install.json'
# The prefix-reuse stage's record (qwen_prefix_stage.py apply, run by the Dockerfile).
PREFIX_RECORD = qwen_prefix_stage.RECORD
MANIFEST_IN_IMAGE = '/opt/qwen-c2/qwen-c2-overlay.txt'
REVISION_IN_IMAGE = '/opt/qwen-c2/source-revision'
STAMP_IN_IMAGE = '/opt/qwen-c2/build-stamp'
STAMP_FILE = 'build-stamp'
REVISION_LABEL = 'org.opencontainers.image.revision'
# P8's record of every file the bundle archive held (docker/qwen-fast-serving.Dockerfile COPYs it).
BUNDLE_RECORD = '/opt/qwen-serving/serving-bundle.json'
TREES = ('/experiment-scripts/ci', '/speculative-decoding/harness')
TREE_SUFFIXES = ('.py', '.cpp', '.h', '.hpp', '.json', '.sh')
CHECKOUT_ROOTS = {'/experiment-scripts/ci': 'scripts/ci', '/speculative-decoding/harness': 'speculative-decoding/harness'}
# The profile that must boot into the v235 gate's environment, and the contract's own switches
# (read by nothing but serving_c2_contract), which the gate never had.
PROFILE_OF_RECORD = 'exact'
CONTRACT_SWITCHES = ('QWEN_C2_SERVING', 'QWEN_C2_PROFILE', 'QWEN_C2_PROFILES')

# A platform-shaped argv, as the node agent passes it; the contract replaces what it owns.
BOOT_ARGS = ('--model', 'Qwen/Qwen3.8-27B', '--served-model-name', 'Qwen/Qwen3.8-27B',
             '--port', '8001', '--max-model-len', '65536', '--help')
API_SERVER = 'vllm.entrypoints.openai.api_server'
ARGV_LINE = re.compile(r'\[QWEN-C2\] profile (\S+): vLLM argv (\[.*\])\s*$')
QWEN_TOKEN = re.compile(rb'QWEN_[A-Z0-9_]+')

# Runs inside an image: sha256 (and QWEN_ strings) of named files, every file of the fast-path
# trees, a tree digest per named directory, the frozen pins, the bundle record and the overlay
# install record. One JSON line on stdout, marked. frozen_pins and tree_digest are
# c2_overlay's own functions, so the host and the image compute them with the same code.
PROBE_BODY = r'''
import hashlib, json, os, re, sys, sysconfig
request = json.loads(sys.argv[1])
purelib = sysconfig.get_paths()['purelib']
token = re.compile(rb'QWEN_[A-Z0-9_]+')
def digest(path, strings=False):
    with open(path, 'rb') as handle:
        if strings:
            data = handle.read()
            return hashlib.sha256(data).hexdigest(), sorted({m.group(0).decode() for m in token.finditer(data)})
        sha = hashlib.sha256()
        for block in iter(lambda: handle.read(1 << 24), b''):
            sha.update(block)
    return sha.hexdigest(), []
files = {}
for path, strings in request.get('files', []):
    real = purelib.rstrip('/') + '/' + path[len('@purelib/'):] if path.startswith('@purelib/') else path
    if not os.path.isfile(real):
        files[path] = dict(exists=False, path=real)
        continue
    sha, found = digest(real, strings)
    files[path] = dict(exists=True, path=real, realpath=os.path.realpath(real), sha256=sha)
    if strings:
        files[path]['qwen'] = found
trees = {}
for top in request.get('trees', []):
    for base, _, names in os.walk(top):
        if '__pycache__' in base.split('/'):
            continue
        for name in names:
            path = os.path.join(base, name)
            if name.endswith(tuple(request.get('suffixes', []))) and os.path.isfile(path):
                trees[path] = digest(path)[0]
dirs = {}
for top in request.get('dirs', []):
    real = os.path.realpath(top)
    dirs[top] = dict(realpath=real, sha256=tree_digest(real))
pins = frozen_pins(request['pins']) if request.get('pins') else None
def load(path):
    if path and os.path.isfile(path):
        with open(path, encoding='utf-8') as handle:
            return json.load(handle)
    return None
bundle = (load(request.get('bundle')) or {}).get('sources')
sys.stdout.write('C2PROBE ' + json.dumps(dict(purelib=purelib, files=files, trees=trees, dirs=dirs, pins=pins,
                                              bundle=bundle, record=load(request.get('record')),
                                              prefix=load(request.get('prefix_record')))) + '\n')
sys.stdout.flush()
'''
PROBE = '\n\n'.join(inspect.getsource(function) for function in (c2_overlay.frozen_pins, c2_overlay.tree_digest)) \
    + '\n' + PROBE_BODY

# Runs inside an image with the image's own environment: what a process there sees once the
# .pth boot hook (qwen_c2_boot -> serving_c2_contract.boot -> apply_environment) has run.
ENV_PROBE = r'''
import json, os, sys
sys.stdout.write('C2ENV ' + json.dumps({name: value for name, value in os.environ.items()
                                       if name.startswith(('QWEN', 'TT_', 'MESH_'))}, sort_keys=True) + '\n')
sys.stdout.flush()
'''


def qwen_strings(data):
    """QWEN_ tokens in a binary, as `strings | grep -oE 'QWEN_[A-Z0-9_]+' | sort -u` sees them."""
    return sorted({match.group(0).decode() for match in QWEN_TOKEN.finditer(data)})


def base_image(dockerfile_text):
    """The P8 base the C2 Dockerfile builds FROM (its ARG BASE default)."""
    match = re.search(r'^ARG BASE=(\S+)\s*$', dockerfile_text, re.MULTILINE)
    if not match:
        raise ValueError('no ARG BASE= line in the C2 Dockerfile')
    return match.group(1)


def graft_pairs(dockerfile_text):
    """The (binary, path) pairs the Dockerfile's K64i loop copies, to hold GRAFT_BINARIES to it."""
    match = re.search(r'for pair in ((?:[A-Za-z0-9_.]+\.so:\S+\s*\\?\s*)+); do', dockerfile_text)
    if not match:
        raise ValueError('no K64i binary loop in the C2 Dockerfile')
    return tuple(tuple(pair.split(':', 1)) for pair in match.group(1).replace('\\', ' ').split())


def graft_dirs(dockerfile_text):
    """The (op, path) pairs the Dockerfile's K64i directory loop replaces, to hold GRAFT_OP_DIRS to it."""
    match = re.search(r'for pair in ((?:[A-Za-z0-9_]+:/\S+\s*\\?\s*)+); do', dockerfile_text)
    if not match:
        raise ValueError('no K64i op directory loop in the C2 Dockerfile')
    return tuple(tuple(pair.split(':', 1)) for pair in match.group(1).replace('\\', ' ').split())


def dockerfile_env(dockerfile_text):
    """{name: value} every ENV instruction in a Dockerfile sets (KEY=VALUE form, unquoted)."""
    joined = dockerfile_text.replace('\r\n', '\n').replace('\\\n', ' ')
    env = {}
    for line in joined.split('\n'):
        if line.startswith('ENV '):
            for token in line[4:].split():
                name, separator, value = token.partition('=')
                if separator:
                    env[name] = value
    return env


def parse_marked(stdout, mark):
    for line in stdout.splitlines():
        if line.startswith(mark + ' '):
            return json.loads(line[len(mark) + 1:])
    raise ValueError('the probe printed no %s line' % mark)


def parse_probe(stdout):
    return parse_marked(stdout, 'C2PROBE')


def argv_line(log_text, profile):
    """(line, argv list) of the image's `profile <p>: vLLM argv` line, or (None, None)."""
    for line in log_text.splitlines():
        match = ARGV_LINE.search(line)
        if match and match.group(1) == profile:
            return line.strip(), json.loads(match.group(2))
    return None, None


def load_contract(path):
    spec = importlib.util.spec_from_file_location('c2_contract_under_check', str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_argv(contract, profiles_path, profile, snapshot):
    """What serving_c2_contract.boot logs for BOOT_ARGS: sys.argv[1:] after rewrite_argv.
    At interpreter start under `python3 -m` sys.argv[0] is '-m'; it is kept and not logged."""
    loaded = contract.load_profile(str(profiles_path), profile)
    return contract.rewrite_argv(['-m'] + list(BOOT_ARGS), loaded, snapshot)[1:]


def qwen_flags(env, profiles):
    """Every QWEN_ flag the image's ENV or a profile sets."""
    names = {name for name in env if name.startswith('QWEN_')}
    for profile in profiles['profiles'].values():
        names.update(name for name in profile.get('env', {}) if name.startswith('QWEN_'))
    return names


def binary_strings(files, paths):
    found = set()
    for path in paths:
        found.update(files.get(path, {}).get('qwen', ()))
    return found


def dropped_flags(base, installed, qwen_names):
    """(dropped, accepted): flags the base binaries read and the installed ones do not, which the
    image sets - failures unless ACCEPTED_BASE_ONLY names them."""
    lost = (set(base) - set(installed)) & set(qwen_names)
    return sorted(lost - set(ACCEPTED_BASE_ONLY)), sorted(lost & set(ACCEPTED_BASE_ONLY))


def check_binaries(host_graft, image_files, base_files, qwen_names):
    """Problems and report lines for (a)'s binaries. host_graft: {binary: sha256} of the context's
    graft; image_files/base_files: probe 'files' of the built and the base image; qwen_names:
    every QWEN_ flag the image env or a profile sets."""
    problems, lines = [], []
    for binary, path in GRAFT_BINARIES:
        found = image_files.get(path, {})
        copy = image_files.get(GRAFT_IN_IMAGE + '/' + binary, {})
        if not found.get('exists'):
            problems.append('(a) %s is missing from the image' % path)
            continue
        lines.append('(a) %s -> %s sha256 %s' % (path, found['realpath'], found['sha256']))
        if found['sha256'] != host_graft.get(binary):
            problems.append('(a) %s is %s, not the graft %s (%s)' % (path, found['sha256'], binary,
                                                                     host_graft.get(binary)))
        if copy.get('sha256') != host_graft.get(binary):
            problems.append('(a) the image\'s %s/%s is %s, not the context\'s %s' % (
                GRAFT_IN_IMAGE, binary, copy.get('sha256'), host_graft.get(binary)))
        installed = set(found.get('qwen', ()))
        graft = set(copy.get('qwen', ()))
        lines.append('(a) QWEN_ strings: installed %s %d, graft %s %d' % (path, len(installed), binary, len(graft)))
        if installed != graft:
            problems.append('(a) %s QWEN_ strings differ from the graft\'s: only installed %s; only graft %s' % (
                path, sorted(installed - graft), sorted(graft - installed)))
    base_paths = sorted({path for _, path in GRAFT_BINARIES})
    for path in base_paths:
        if not base_files.get(path, {}).get('exists'):
            problems.append('(a) the base image has no %s to compare against' % path)
    base = binary_strings(base_files, base_paths)
    installed = binary_strings(image_files, base_paths)
    dropped, accepted = dropped_flags(base, installed, qwen_names)
    lines.extend('(a) the installed binaries lack %s, accepted: %s' % (name, ACCEPTED_BASE_ONLY[name]) for name in accepted)
    if dropped:
        problems.append('(a) the installed binaries lack %s, which the base binaries read and the image sets: the '
                        'graft dropped that patch and the flag is inert (graft-so-drops-image-patches)' % dropped)
    lines.append('(a) QWEN_ strings only in the base binaries: %s' % (sorted(base - installed) or 'none'))
    lines.append('(a) QWEN_ strings only in the installed binaries: %s' % (sorted(installed - base) or 'none'))
    return problems, lines


def check_op_dirs(host_dirs, image_dirs):
    """Problems and report lines for (a)'s op directories: {op: tree digest} of the context's
    K64i copy against the probe's 'dirs' of the built image."""
    problems, lines = [], []
    for op, path in GRAFT_OP_DIRS:
        want = host_dirs.get(op)
        found = image_dirs.get(path, {})
        if want is None:
            problems.append('(a) the context has no opgraft-K64i/%s to compare %s with' % (op, path))
        elif found.get('sha256') is None:
            problems.append('(a) %s is not a directory in the image' % path)
        elif found['sha256'] != want:
            problems.append('(a) %s (%s) is tree %s, not the K64i %s tree %s: the JIT would compile other '
                            'kernels' % (path, found.get('realpath'), found['sha256'][:16], op, want[:16]))
        else:
            lines.append('(a) %s is the K64i %s tree %s' % (path, op, want[:16]))
    return problems, lines


def check_runtime_sha(env, image_files):
    """QWEN_FAST_RUNTIME_BINARY_SHA256, when the image sets it, must name the installed binary."""
    pinned = env.get('QWEN_FAST_RUNTIME_BINARY_SHA256')
    installed = image_files.get(GRAFT_BINARIES[-1][1], {}).get('sha256')
    if pinned and pinned != installed:
        return ['(a) QWEN_FAST_RUNTIME_BINARY_SHA256=%s but the installed binary is %s' % (pinned, installed)]
    return []


def check_overlay(entries, source_shas, image_files, record):
    """Problems and report lines for (b)."""
    problems, lines = [], []
    for entry in entries:
        for destination in entry.destinations:
            found = image_files.get(destination, {})
            want = source_shas.get(entry.source)
            if not found.get('exists'):
                problems.append('(b) %s -> %s is missing from the image' % (entry.source, destination))
            elif found['sha256'] != want:
                problems.append('(b) %s -> %s is %s in the image, not the source %s' % (
                    entry.source, found['path'], found['sha256'], want))
            else:
                lines.append('(b) %s -> %s sha256 %s' % (entry.source, found['path'], want))
    rows = (record or {}).get('files') or []
    if not rows:
        problems.append('(b) the image has no overlay install record at %s' % INSTALL_RECORD)
    for row in rows:
        lines.append('(b) install: %-7s %s%s' % (row['state'], row['path'],
                                                 ' (base %s)' % row['before'][:16] if row['state'] == 'changed' else ''))
    return problems, lines


def check_revision(context, labels, image_files):
    """Problems and report lines for (b)'s provenance: the revision label, source-revision and
    the build stamp (the sha256 of the build script that ran, which must be the context's)."""
    context = Path(context)
    problems, lines = [], []
    revision = (context / c2_overlay.REVISION_FILE).read_text(encoding='utf-8').strip()
    label = (labels or {}).get(REVISION_LABEL)
    if label != revision:
        problems.append('(b) the image\'s %s label is %s, not the context\'s commit %s' % (REVISION_LABEL, label, revision))
    else:
        lines.append('(b) %s %s' % (REVISION_LABEL, label))
    for name, path in ((c2_overlay.REVISION_FILE, REVISION_IN_IMAGE), (STAMP_FILE, STAMP_IN_IMAGE)):
        local = context / name
        found = image_files.get(path, {})
        if not local.is_file():
            problems.append('(b) the context has no %s' % name)
        elif found.get('sha256') != c2_overlay.sha256(local):
            problems.append('(b) the image\'s %s is not the context\'s %s' % (path, name))
    stamp = context / STAMP_FILE
    script = context / Path(c2_overlay.BUILD_SCRIPT).name
    if stamp.is_file() and script.is_file():
        ran = stamp.read_text(encoding='utf-8').strip()
        if ran != c2_overlay.sha256(script):
            problems.append('(b) the build ran a build script with sha256 %s, not the context\'s %s (%s)' % (
                ran, script.name, c2_overlay.sha256(script)))
        else:
            lines.append('(b) built by the context\'s %s (%s)' % (script.name, ran[:16]))
    return problems, lines


def check_argv(contract, profiles_path, profile, expected_name, log_text):
    """Problems and report lines for one boot (c). profile is the QWEN_C2_PROFILE passed (None
    for the image default); expected_name the profile that should answer."""
    problems, lines = [], []
    line, logged = argv_line(log_text, expected_name)
    lines.extend('(c) %s: %s' % (profile or 'default', text.strip()) for text in log_text.splitlines()
                 if '[QWEN-C2]' in text)
    if line is None:
        tail = '\n'.join(log_text.strip().splitlines()[-15:])
        problems.append('(c) %s: no "[QWEN-C2] profile %s: vLLM argv" line; the boot ended:\n%s' % (
            profile or 'default', expected_name, tail))
        return problems, lines
    snapshot = logged[logged.index('--model') + 1] if '--model' in logged[:-1] else None
    loaded = contract.load_profile(str(profiles_path), expected_name)
    if snapshot not in loaded['snapshots']:
        problems.append('(c) %s: --model %s is none of the profile\'s snapshots' % (expected_name, snapshot))
        return problems, lines
    want = expected_argv(contract, profiles_path, expected_name, snapshot)
    if logged != want:
        problems.append('(c) %s: the image launched %s; the context\'s contract gives %s' % (
            expected_name, json.dumps(logged), json.dumps(want)))
    else:
        lines.append('(c) %s: argv matches the context\'s contract and profile' % expected_name)
    return problems, lines


def qwen_environment(env):
    """The QWEN_ variables a fast-path process reads (the contract's switches left out)."""
    return {name: value for name, value in env.items() if name.startswith('QWEN_') and name not in CONTRACT_SWITCHES}


def environment_sha(env):
    return hashlib.sha256(json.dumps(sorted(env.items())).encode('utf-8')).hexdigest()


def environment_differences(reference, env):
    """(differences, equivalents) of a QWEN_ environment against the v235 gate's: each difference
    names a variable the gate ran otherwise; equivalents are variables the gate left unset whose
    value here is the code's own default for unset (reference['unset_equivalents'])."""
    gate = reference['qwen_configuration']
    unset = {name: record['value'] for name, record in reference.get('unset_equivalents', {}).items()}
    got = qwen_environment(env)
    differences, equivalents = [], []
    for name in sorted(gate):
        if got.get(name) != gate[name]:
            differences.append('%s=%s, the v235 gate ran %s' % (name, got.get(name, '(unset)'), gate[name]))
    for name in sorted(set(got) - set(gate)):
        if unset.get(name) == got[name]:
            equivalents.append('%s=%s is the gate\'s unset default' % (name, got[name]))
        else:
            differences.append('%s=%s, which the v235 gate never set' % (name, got[name]))
    return differences, equivalents


def check_environment(reference, environments):
    """Problems and report lines for (c)'s environment: {profile: env seen in the image}."""
    problems, lines = [], []
    exact = environments.get(PROFILE_OF_RECORD)
    if exact is None:
        return ['(c) no %s profile environment to compare with the v235 gate\'s' % PROFILE_OF_RECORD], lines
    differences, equivalents = environment_differences(reference, exact)
    problems.extend('(c) %s: %s' % (PROFILE_OF_RECORD, text) for text in differences)
    lines.extend('(c) %s: %s' % (PROFILE_OF_RECORD, text) for text in equivalents)
    lines.append('(c) %s: QWEN_ environment sha256 %s over %d variables; the v235 gate\'s %s over %d%s' % (
        PROFILE_OF_RECORD, environment_sha(qwen_environment(exact)), len(qwen_environment(exact)),
        environment_sha(reference['qwen_configuration']), len(reference['qwen_configuration']),
        '' if differences else ' (equal up to the unset defaults)'))
    baseline = qwen_environment(exact)
    for name in sorted(environments):
        if name == PROFILE_OF_RECORD:
            continue
        env = qwen_environment(environments[name])
        changed = sorted(key for key in set(env) | set(baseline) if env.get(key) != baseline.get(key))
        lines.append('(c) %s: QWEN_ environment sha256 %s; differs from %s in %s' % (
            name, environment_sha(env), PROFILE_OF_RECORD, ', '.join(
                '%s=%s' % (key, env.get(key, '(unset)')) for key in changed) or 'nothing'))
    return problems, lines


PREFIX_ON_FLAGS = ('--enable-prefix-caching', '--enable-chunked-prefill', '--no-async-scheduling')
PREFIX_OFF_FLAGS = ('--no-enable-prefix-caching',)


def check_prefix_launch(contract, profiles_path, name, logged, env):
    """Problems and report lines for (f)'s launch half: one profile's launched argv (the logged list,
    None when the boot logged none - (c) reports that) and its booted environment."""
    problems, lines = [], []
    loaded = contract.load_profile(str(profiles_path), name)
    on = contract.prefix_reuse(loaded)
    switch = (env or {}).get(contract.PREFIX_SWITCH)
    if on and switch != '1':
        problems.append('(f) %s: the booted environment has %s=%s, not 1' % (name, contract.PREFIX_SWITCH, switch))
    if not on and switch is not None:
        problems.append('(f) %s: the booted environment has %s=%s under a profile without prefix reuse' % (
            name, contract.PREFIX_SWITCH, switch))
    if logged is None:
        return problems, lines
    wanted, refused = (PREFIX_ON_FLAGS, PREFIX_OFF_FLAGS) if on else (PREFIX_OFF_FLAGS, ('--enable-prefix-caching',))
    missing = [flag for flag in wanted if flag not in logged]
    present = [flag for flag in refused if flag in logged]
    if missing or present:
        problems.append('(f) %s: prefix reuse is %s in the profile, but the launched argv lacks %s and has %s' % (
            name, 'on' if on else 'off', missing or 'nothing', present or 'nothing'))
    else:
        lines.append('(f) %s: prefix reuse %s in the launched argv (%s) and %s' % (
            name, 'ON' if on else 'off', ' '.join(wanted),
            '%s=1' % contract.PREFIX_SWITCH if on else 'no %s' % contract.PREFIX_SWITCH))
    return problems, lines


def prefix_stage_files():
    """[path, strings] probe rows for every file the prefix-reuse stage pins."""
    return [[path, False] for _, path, _, _ in qwen_prefix_stage.TARGETS]


def check_prefix_anchors(files):
    """(f) before a build: the stage refuses a base whose targets are not its pinned originals."""
    problems, lines = [], []
    for name, path, pin, source in qwen_prefix_stage.TARGETS:
        found = files.get(path) or {}
        if not found.get('exists'):
            problems.append('(f) %s: the base has no %s; the prefix stage refuses the build' % (name, path))
        elif found.get('sha256') != pin:
            problems.append('(f) %s: the base holds %s at %s, not the pinned %s (%s); the prefix stage refuses '
                            'the build' % (name, path, found.get('sha256'), pin, source))
        else:
            lines.append('(f) %s: the base holds the pinned %s' % (name, pin[:16]))
    return problems, lines


def check_prefix_stage(record, files, entries, source_shas):
    """(f) after a build: the stage record against the pins, the image's files and the overlay."""
    image_shas = {path: (files.get(path) or {}).get('sha256') for _, path, _, _ in qwen_prefix_stage.TARGETS}
    module_shas = {entry.source[len('scripts/ci/'):-len('.py')]: source_shas[entry.source] for entry in entries
                   if entry.source.startswith('scripts/ci/') and entry.source.endswith('.py')
                   and '/' not in entry.source[len('scripts/ci/'):]}
    return qwen_prefix_stage.record_problems(record, image_shas, module_shas)


def check_layers(layers, trees, bundle=None, replaced=(), accepted=None):
    """Problems, report lines and counts for (d). layers: layers.json; trees: the probe's
    {image path: sha256}; bundle: the image's bundle record sources (diagnostic); replaced:
    image paths a later step replaces (--base-drift: the C2 overlay's destinations). A test_*
    module off the model is counted, not failed: the image's own unittest RUN is the gate for the
    test modules it runs, and serving imports none."""
    accepted = ACCEPTED_LAYER_DRIFT if accepted is None else accepted
    problems, lines, notes = [], [], []
    counts = dict(model=0, head=0, accepted=0, drift=0, absent=0, replaced=0, tests=0, bundle=0, unbundled=0)
    commits = layers.get('commits', {})
    for path, want in sorted(layers['files'].items()):
        if path in replaced:
            counts['replaced'] += 1
            continue
        got = trees.get(path)
        if got is not None and got == want['sha256']:
            counts['model'] += 1
            continue
        # P8's bundle record (BUNDLE_RECORD) is the archive's own manifest, and it outranks a model
        # rebuilt from git at the bundle commit: the archive was a curated subset of scripts/ci (base
        # drift v20, run 36210232102: 572 files the git model expected were never in it) and some of
        # its files were staged rewrites (35 held the record's bytes, not git's). A file at the
        # record's bytes is as built; a file the record never listed was never in the base.
        if bundle:  # an empty or missing record says nothing: the git model stands
            recorded = bundle.get(path.lstrip('/'))
            if got is not None and recorded == got:
                counts['bundle'] += 1
                continue
            if got is None and recorded is None:
                counts['unbundled'] += 1
                continue
        if got is not None and got == want.get('head'):
            counts['head'] += 1
            notes.append('(d) %s holds HEAD\'s version; the model said %s\'s' % (path, want['layer'][:8]))
            continue
        if path in accepted and accepted[path][0] == got:
            counts['accepted'] += 1
            notes.append('(d) %s drift accepted: %s' % (path, accepted[path][1]))
            continue
        if path.rsplit('/', 1)[-1].startswith('test_'):
            counts['tests'] += 1
            continue
        hints = []
        if got is not None and got == want.get('read_combined'):
            hints.append('= read-combined %s' % (commits.get('read_combined') or '')[:8])
        if bundle is not None and got is not None and bundle.get(path.lstrip('/')) == got:
            hints.append('= the bundle record')
        if got is None:
            counts['absent'] += 1
            what = 'is absent'
        else:
            counts['drift'] += 1
            what = 'is %s' % got[:16]
        problems.append('(d) %s %s: the layer model says %s (%s at %s) and HEAD has %s%s' % (
            path, what, want['source'], (want['sha256'] or '-')[:16], want['layer'][:8], (want.get('head') or 'none')[:16],
            (' [%s]' % ', '.join(hints)) if hints else ''))
    unmodelled = sorted(path for path in trees if path not in layers['files'])
    lines.append('(d) layers (bundle %s, P8 %s, HEAD %s): %d files as modelled, %d at the bundle record\'s bytes, '
                 '%d never in the bundle, %d at HEAD\'s version, %d accepted, '
                 '%d at other bytes, %d absent, %d replaced later, %d test modules off the model (not gated); '
                 '%d image files the model does not name' % (
                     commits.get('bundle', '?')[:8], commits.get('p8', '?')[:8], commits.get('head', '?')[:8],
                     counts['model'], counts['bundle'], counts['unbundled'], counts['head'], counts['accepted'],
                     counts['drift'], counts['absent'], counts['replaced'], counts['tests'], len(unmodelled)))
    lines.extend(notes)
    return problems, lines, dict(counts, unmodelled=unmodelled)


def check_pins(pins, base_pins=None, overlaid=None):
    """Problems and report lines for (e). pins/base_pins: frozen_pins of the built image and of
    its P8 base; overlaid: {image path: source sha256} the C2 overlay writes."""
    problems, lines = [], []
    if not pins or not pins.get('evidence'):
        return ['(e) the image has no %s/frozen-evidence: the source pin exact\'s qualify reads on every request '
                'cannot be checked' % c2_overlay.PINNED_TREE], lines
    problems.extend('(e) %s' % text for text in pins['problems'])
    lines.append('(e) frozen evidence %s; %d pinned sources' % (
        ', '.join('%s %s' % (name, sha[:16]) for name, sha in sorted(pins['reports'].items())), len(pins['pins'])))
    base_live = (base_pins or {}).get('live', {})
    for path, pinned in sorted(pins['pins'].items()):
        live = pins['live'].get(path)
        if live == pinned:
            continue
        if path in ACCEPTED_PIN_MISMATCH and ACCEPTED_PIN_MISMATCH[path][0] == live:
            lines.append('(e) %s breaks its pin, accepted: %s' % (path, ACCEPTED_PIN_MISMATCH[path][1]))
            continue
        if base_pins is None:
            base = ''
        elif base_live.get(path) == live:
            base = '; the P8 base holds the same bytes'
        elif base_live.get(path) == pinned:
            base = '; the P8 base holds the pinned bytes, so C2 broke it'
        else:
            base = '; the P8 base holds %s' % (base_live.get(path) or 'nothing')[:16]
        problems.append('(e) %s is %s but the frozen recipe pins %s: exact would die at its first request '
                        '(frozen_combined_runtime.qualify)%s' % (path, (live or 'absent')[:16], pinned[:16], base))
    for path, digest in sorted((overlaid or {}).items()):
        if path in pins['pins']:
            lines.append('(e) the overlay writes pinned %s with %s bytes' % (
                path, 'its pinned' if digest == pins['pins'][path] else 'OTHER'))
    return problems, lines


def drift_lines(trees, checkout):
    """Informational (i): image files the checkout also has, compared by sha256."""
    same, differ = 0, []
    for path in sorted(trees):
        for top, relative in CHECKOUT_ROOTS.items():
            if path.startswith(top + '/'):
                local = Path(checkout) / relative / path[len(top) + 1:]
                if local.is_file():
                    if c2_overlay.sha256(local) == trees[path]:
                        same += 1
                    else:
                        differ.append(path)
    lines = ['(i) image vs checkout: %d identical, %d differ (informational; (d) is the gate)' % (same, len(differ))]
    lines.extend('(i) differs from the checkout: %s' % path for path in differ)
    return lines


class Docker(object):
    def __init__(self, docker='docker', timeout=600):
        self.docker, self.timeout = docker, timeout

    def run(self, arguments, timeout=None):
        result = subprocess.run([self.docker] + list(arguments), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                encoding='utf-8', errors='replace', timeout=timeout or self.timeout)
        return result.returncode, result.stdout, result.stderr

    def probe(self, image, request):
        code, out, err = self.run(['run', '--rm', '--network', 'none', '-e', 'QWEN_C2_SERVING=0',
                                   '--entrypoint', 'python3', image, '-c', PROBE, json.dumps(request)])
        if code != 0:
            raise RuntimeError('probe of %s exited %d: %s' % (image, code, err[-2000:]))
        return parse_probe(out)

    def config(self, image):
        """dict(env={name: value}, labels={name: value}) of an image."""
        code, out, err = self.run(['image', 'inspect', '--format', '{{json .Config}}', image])
        if code != 0:
            raise RuntimeError('docker image inspect %s: %s' % (image, err.strip()))
        config = json.loads(out)
        return dict(env=dict(item.partition('=')[::2] for item in config.get('Env') or []),
                    labels=dict(config.get('Labels') or {}))

    def environment(self, image, profile):
        arguments = ['run', '--rm', '--network', 'none']
        if profile:
            arguments += ['-e', 'QWEN_C2_PROFILE=' + profile]
        code, out, err = self.run(arguments + ['--entrypoint', 'python3', image, '-c', ENV_PROBE])
        if code != 0:
            raise RuntimeError('environment probe of %s (%s) exited %d: %s' % (image, profile, code, err[-2000:]))
        return parse_marked(out, 'C2ENV')

    def boot(self, image, models, profile):
        arguments = ['run', '--rm', '--network', 'none', '-v', '%s:/models:ro' % models]
        if profile:
            arguments += ['-e', 'QWEN_C2_PROFILE=' + profile]
        code, out, err = self.run(arguments + ['--entrypoint', 'python3', image, '-m', API_SERVER] + list(BOOT_ARGS))
        return err + '\n' + out


def read_json(path):
    with open(str(path), encoding='utf-8') as handle:
        return json.load(handle)


def overlay_destinations(entries, source_shas):
    """{image path: source sha256} the overlay writes into the fast-path trees (not @purelib)."""
    return {destination: source_shas[entry.source] for entry in entries for destination in entry.destinations
            if not destination.startswith(c2_overlay.PURELIB)}


def verify(image, context, models, checkout=None, docker=None, log=print):
    """Run (a)-(e) (and (i) with a checkout); return (problems, report)."""
    docker = docker or Docker()
    context = Path(context)
    entries = c2_overlay.read_manifest(context / 'qwen-c2-overlay.txt')
    dockerfile = (context / 'Dockerfile').read_text(encoding='utf-8')
    problems = []
    if tuple(graft_pairs(dockerfile)) != GRAFT_BINARIES:
        problems.append('(a) the Dockerfile copies %s, not GRAFT_BINARIES %s' % (graft_pairs(dockerfile), GRAFT_BINARIES))
    if tuple(graft_dirs(dockerfile)) != GRAFT_OP_DIRS:
        problems.append('(a) the Dockerfile replaces %s, not GRAFT_OP_DIRS %s' % (graft_dirs(dockerfile), GRAFT_OP_DIRS))
    base = base_image(dockerfile)
    graft = context / 'opgraft-K64i'
    host_graft = {binary: c2_overlay.sha256(graft / binary) for binary in sorted({binary for binary, _ in GRAFT_BINARIES})
                  if (graft / binary).is_file()}
    host_dirs = {op: c2_overlay.tree_digest(str(graft / op)) for op, _ in GRAFT_OP_DIRS}
    host_dirs = {op: digest for op, digest in host_dirs.items() if digest is not None}
    source_shas = {entry.source: c2_overlay.sha256(context / 'overlay' / entry.source) for entry in entries}
    files = [[path, True] for _, path in GRAFT_BINARIES]
    files += [[GRAFT_IN_IMAGE + '/' + binary, True] for binary in sorted(host_graft)]
    files += [[destination, False] for entry in entries for destination in entry.destinations]
    files += [[path, False] for path in (MANIFEST_IN_IMAGE, REVISION_IN_IMAGE, STAMP_IN_IMAGE)]
    files += prefix_stage_files()
    built = docker.probe(image, dict(files=files, record=INSTALL_RECORD, trees=list(TREES), suffixes=list(TREE_SUFFIXES),
                                     dirs=[path for _, path in GRAFT_OP_DIRS], pins=c2_overlay.PINNED_TREE,
                                     bundle=BUNDLE_RECORD, prefix_record=PREFIX_RECORD))
    based = docker.probe(base, dict(files=[[path, True] for path in sorted({path for _, path in GRAFT_BINARIES})],
                                    pins=c2_overlay.PINNED_TREE))
    config = docker.config(image)
    env = config['env']
    profiles_path = context / 'overlay' / 'scripts/ci/qwen_c2_profiles.json'
    profiles = read_json(profiles_path)
    qwen_names = qwen_flags(env, profiles)

    report = ['(image) %s, base %s' % (image, base)]
    found, lines = check_binaries(host_graft, built['files'], based['files'], qwen_names)
    problems += found + check_runtime_sha(env, built['files'])
    report += lines
    found, lines = check_op_dirs(host_dirs, built.get('dirs') or {})
    problems += found
    report += lines
    manifest = built['files'].get(MANIFEST_IN_IMAGE, {})
    if manifest.get('sha256') != c2_overlay.sha256(context / 'qwen-c2-overlay.txt'):
        problems.append('(b) the image\'s %s is not the context\'s manifest' % MANIFEST_IN_IMAGE)
    found, lines = check_overlay(entries, source_shas, built['files'], built.get('record'))
    problems += found
    report += lines
    found, lines = check_revision(context, config['labels'], built['files'])
    problems += found
    report += lines
    contract = load_contract(context / 'overlay' / 'scripts/ci/serving_c2_contract.py')
    boots = {}
    for profile in [None] + sorted(profiles['profiles']):
        boots[profile] = docker.boot(image, models, profile)
        found, lines = check_argv(contract, profiles_path, profile, profile or profiles['default'], boots[profile])
        problems += found
        report += lines
    environments = {name: docker.environment(image, name) for name in sorted(profiles['profiles'])}
    found, lines = check_environment(read_json(context / 'v235-environment.json'), environments)
    problems += found
    report += lines
    found, lines, layer_counts = check_layers(read_json(context / c2_overlay.LAYERS_FILE), built.get('trees') or {},
                                              built.get('bundle'))
    problems += found
    report += lines
    found, lines = check_pins(built.get('pins'), based.get('pins'), overlay_destinations(entries, source_shas))
    problems += found
    report += lines
    for name in sorted(profiles['profiles']):
        found, lines = check_prefix_launch(contract, profiles_path, name, argv_line(boots[name], name)[1],
                                           environments[name])
        problems += found
        report += lines
    found, lines = check_prefix_stage(built.get('prefix'), built['files'], entries, source_shas)
    problems += found
    report += lines
    if checkout:
        report += drift_lines(built.get('trees') or {}, checkout)
    for line in report:
        log('[G1] ' + line)
    for problem in problems:
        log('[G1] FAIL ' + problem)
    log('[G1] %s: %d problem(s)' % ('FAIL' if problems else 'PASS', len(problems)))
    return problems, dict(image=image, base=base, problems=problems, report=report, env_qwen=sorted(qwen_names),
                          environments=environments, layers=layer_counts, built=built, base_probe=based)


def base_drift(context, graft=None, docker=None, log=print):
    """Read-only, before any build: what the first build's (a) strings, (d), (e) and (f) anchors would fail on,
    measured on the P8 base image the context builds FROM. context needs the staged files
    (c2_overlay.py stage); graft, when given, is a K64i directory holding _ttnn.so and
    _ttnncpp.so. Returns (problems, report); nothing is built, tagged or written."""
    docker = docker or Docker()
    context = Path(context)
    entries = c2_overlay.read_manifest(context / 'qwen-c2-overlay.txt')
    dockerfile = (context / 'Dockerfile').read_text(encoding='utf-8')
    base = base_image(dockerfile)
    source_shas = {entry.source: c2_overlay.sha256(context / 'overlay' / entry.source) for entry in entries}
    overlaid = overlay_destinations(entries, source_shas)
    based = docker.probe(base, dict(files=[[path, True] for path in sorted({path for _, path in GRAFT_BINARIES})]
                                    + prefix_stage_files(),
                                    trees=list(TREES), suffixes=list(TREE_SUFFIXES), pins=c2_overlay.PINNED_TREE,
                                    bundle=BUNDLE_RECORD))
    problems, report = [], ['(base) %s, before any C2 build' % base]
    if graft:
        paths = sorted({path for _, path in GRAFT_BINARIES})
        base_strings = binary_strings(based['files'], paths)
        installed = set()
        for binary in sorted({binary for binary, _ in GRAFT_BINARIES}):
            installed.update(qwen_strings((Path(graft) / binary).read_bytes()))
        names = qwen_flags(dockerfile_env(dockerfile), read_json(context / 'overlay/scripts/ci/qwen_c2_profiles.json'))
        dropped, accepted = dropped_flags(base_strings, installed, names)
        if dropped:
            problems.append('(a) the graft binaries lack %s, which the base binaries read and the image sets' % dropped)
        report.append('(a) QWEN_ strings only in the base binaries: %s; accepted %s' % (
            sorted(base_strings - installed) or 'none', accepted or 'none'))
    found, lines, counts = check_layers(read_json(context / c2_overlay.LAYERS_FILE), based.get('trees') or {},
                                        based.get('bundle'), replaced=set(overlaid))
    problems += found
    report += lines
    pins = based.get('pins')
    if pins and pins.get('evidence'):
        # The built image's pins: the overlay's destinations take their new bytes. A destination the
        # overlay would write with other bytes than its pin is install's refusal, reported once below.
        pins = dict(pins, live=dict(pins['live']))
        for path, digest in sorted(overlaid.items()):
            pinned = pins['pins'].get(path)
            if pinned is None:
                continue
            pins['live'][path] = pinned
            if pinned != digest:
                problems.append('(e) the overlay would write %s over pinned %s: install refuses it' % (digest[:16], path))
    found, lines = check_pins(pins, None, overlaid)
    problems += found
    report += lines
    # The prefix stage patches files the C2 build leaves at the base's bytes (none is grafted or
    # overlaid), so the base shows what its anchor check will see.
    found, lines = check_prefix_anchors(based.get('files') or {})
    problems += found
    report += lines
    for line in report:
        log('[G1-BASE] ' + line)
    for problem in problems:
        log('[G1-BASE] WOULD FAIL ' + problem)
    log('[G1-BASE] %d problem(s) the first build would hit' % len(problems))
    return problems, dict(base=base, problems=problems, report=report, layers=counts, base_probe=based)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--image', help='the built image (id or name); not with --base-drift')
    parser.add_argument('--context', required=True, help='the build context it was built from')
    parser.add_argument('--models', help='host directory mounted as /models for the boot check')
    parser.add_argument('--checkout', help='a checkout to compare the image trees with (informational)')
    parser.add_argument('--report', help='write the full JSON report here')
    parser.add_argument('--base-drift', action='store_true',
                        help='read-only: check the P8 base the context names, before any build')
    parser.add_argument('--graft', help='with --base-drift: a K64i directory to diff the base binaries\' strings with')
    arguments = parser.parse_args(argv)
    if arguments.base_drift:
        if arguments.image:
            parser.error('--base-drift reads the base the context names; it takes no --image')
        problems, report = base_drift(arguments.context, arguments.graft)
    else:
        if not arguments.image or not arguments.models:
            parser.error('--image and --models are required')
        problems, report = verify(arguments.image, arguments.context, arguments.models, arguments.checkout)
    if arguments.report:
        Path(arguments.report).write_text(json.dumps(report, indent=1, sort_keys=True) + '\n', encoding='utf-8')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
