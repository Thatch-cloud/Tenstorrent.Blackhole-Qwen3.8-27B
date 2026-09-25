"""G1 provenance for a freshly built C2 serving image: check it is what its context says.

Run by build-c2-serving-image.sh on the rig host right after `docker build`, before the image
gets its tag; a problem exits 1, so a failed image is never tagged, smoked or pushed.

(a) binaries: the installed _ttnncpp.so (both paths the K64i graft is copied to) and _ttnn.so
    are the graft's bytes; the QWEN_ strings of the installed binary, the graft's and the P8
    base binary's are printed, and a QWEN_ flag the image or a profile SETS that the base binary
    reads but the installed one does not is a failure. A graft replaces the whole binary, so a
    patch the base carried and the graft's source tree lacked is silently dropped and its flag
    goes inert (memory graft-so-drops-image-patches: K64c lost QWEN_SDPA_TREE_SCRATCH_ROUNDS).
(b) overlay: every destination docker/qwen-c2-overlay.txt names holds the sha256 of its source
    in the build context; the install record says which base files the overlay changed.
(c) boot argv: for the default profile and every named one, the `[QWEN-C2] profile <p>: vLLM
    argv [...]` line the image logs equals the argv the context's serving_c2_contract computes
    for the same platform argv (memory read-the-launched-argv).
(d) with --checkout, every file under the image's /experiment-scripts/ci and
    /speculative-decoding/harness that the checkout also has is compared with it and the
    differences are listed. Informational only: the bundle's tree was staged from a workspace
    the repo does not pin, and test_c2_image_overlay is the gate for it.

Stdlib only, python >= 3.6. Every container runs with --network none.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c2_overlay  # noqa: E402

# The K64i binaries and where docker/qwen-c2-serving.Dockerfile copies them (its first RUN).
GRAFT_BINARIES = (
    ('_ttnn.so', '/opt/tt-metal/ttnn/ttnn/_ttnn.so'),
    ('_ttnncpp.so', '/opt/tt-metal/build_Release/ttnn/_ttnncpp.so'),
    ('_ttnncpp.so', '/opt/tt-metal/build_Release/lib/_ttnncpp.so'),
)
GRAFT_IN_IMAGE = '/opt/qwen-c2/opgraft-K64i'
# QWEN_ flags the base binary reads, the image sets, and the graft is KNOWN not to read, each
# with the reason that is acceptable (e.g. the gate that qualified the graft ran without it
# too). Empty until a build reports one; never add a flag without reading which patch it is.
ACCEPTED_BASE_ONLY = {}
INSTALL_RECORD = '/opt/qwen-c2/overlay-install.json'
MANIFEST_IN_IMAGE = '/opt/qwen-c2/qwen-c2-overlay.txt'
TREES = ('/experiment-scripts/ci', '/speculative-decoding/harness')
TREE_SUFFIXES = ('.py', '.cpp', '.h', '.hpp', '.json', '.sh')
CHECKOUT_ROOTS = {'/experiment-scripts/ci': 'scripts/ci', '/speculative-decoding/harness': 'speculative-decoding/harness'}

# A platform-shaped argv, as the node agent passes it; the contract replaces what it owns.
BOOT_ARGS = ('--model', 'Qwen/Qwen3.8-27B', '--served-model-name', 'Qwen/Qwen3.8-27B',
             '--port', '8001', '--max-model-len', '65536', '--help')
API_SERVER = 'vllm.entrypoints.openai.api_server'
ARGV_LINE = re.compile(r'\[QWEN-C2\] profile (\S+): vLLM argv (\[.*\])\s*$')
QWEN_TOKEN = re.compile(rb'QWEN_[A-Z0-9_]+')

# Runs inside an image: sha256 (and QWEN_ strings) of named files, optionally every file of the
# fast-path trees, and the overlay install record. One JSON line on stdout, marked.
PROBE = r'''
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
            if name.endswith(tuple(request.get('suffixes', []))):
                path = os.path.join(base, name)
                trees[path] = digest(path)[0]
record = None
if request.get('record') and os.path.isfile(request['record']):
    with open(request['record'], encoding='utf-8') as handle:
        record = json.load(handle)
sys.stdout.write('C2PROBE ' + json.dumps(dict(purelib=purelib, files=files, trees=trees, record=record)) + '\n')
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


def parse_probe(stdout):
    for line in stdout.splitlines():
        if line.startswith('C2PROBE '):
            return json.loads(line[len('C2PROBE '):])
    raise ValueError('the probe printed no C2PROBE line')


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


def check_binaries(host_graft, image_files, base_files, qwen_names):
    """Problems and report lines for (a). host_graft: {binary: sha256} of the context's graft;
    image_files/base_files: probe 'files' of the built and the base image; qwen_names: every
    QWEN_ flag the image env or a profile sets."""
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
    graft = set(image_files.get(GRAFT_IN_IMAGE + '/_ttnncpp.so', {}).get('qwen', ()))
    base_path = GRAFT_BINARIES[-1][1]
    base = set(base_files.get(base_path, {}).get('qwen', ()))
    if not base_files.get(base_path, {}).get('exists'):
        problems.append('(a) the base image has no %s to compare against' % base_path)
    for binary, path in GRAFT_BINARIES:
        if binary != '_ttnncpp.so' or not image_files.get(path, {}).get('exists'):
            continue
        installed = set(image_files[path].get('qwen', ()))
        lines.append('(a) QWEN_ strings: installed %s %d, graft %d, base %d' % (path, len(installed), len(graft),
                                                                              len(base)))
        if installed != graft:
            problems.append('(a) %s QWEN_ strings differ from the graft\'s: only installed %s; only graft %s' % (
                path, sorted(installed - graft), sorted(graft - installed)))
        dropped = sorted((base - installed) & set(qwen_names) - set(ACCEPTED_BASE_ONLY))
        accepted = sorted((base - installed) & set(qwen_names) & set(ACCEPTED_BASE_ONLY))
        lines.extend('(a) %s lacks %s, accepted: %s' % (path, name, ACCEPTED_BASE_ONLY[name]) for name in accepted)
        if dropped:
            problems.append('(a) %s lacks %s, which the base binary reads and the image sets: the graft dropped '
                            'that patch and the flag is inert (graft-so-drops-image-patches)' % (path, dropped))
    lines.append('(a) QWEN_ strings only in the base binary: %s' % (sorted(base - graft) or 'none'))
    lines.append('(a) QWEN_ strings only in the graft: %s' % (sorted(graft - base) or 'none'))
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


def drift_lines(trees, checkout):
    """Informational (d): image files the checkout also has, compared by sha256."""
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
    lines = ['(d) image vs checkout: %d identical, %d differ (informational)' % (same, len(differ))]
    lines.extend('(d) differs from the checkout: %s' % path for path in differ)
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

    def env(self, image):
        code, out, err = self.run(['image', 'inspect', '--format', '{{json .Config.Env}}', image])
        if code != 0:
            raise RuntimeError('docker image inspect %s: %s' % (image, err.strip()))
        return dict(item.partition('=')[::2] for item in json.loads(out))

    def boot(self, image, models, profile):
        arguments = ['run', '--rm', '--network', 'none', '-v', '%s:/models:ro' % models]
        if profile:
            arguments += ['-e', 'QWEN_C2_PROFILE=' + profile]
        code, out, err = self.run(arguments + ['--entrypoint', 'python3', image, '-m', API_SERVER] + list(BOOT_ARGS))
        return err + '\n' + out


def verify(image, context, models, checkout=None, docker=None, log=print):
    """Run (a)-(d); return (problems, report)."""
    docker = docker or Docker()
    context = Path(context)
    entries = c2_overlay.read_manifest(context / 'qwen-c2-overlay.txt')
    dockerfile = (context / 'Dockerfile').read_text(encoding='utf-8')
    problems = []
    if tuple(graft_pairs(dockerfile)) != GRAFT_BINARIES:
        problems.append('(a) the Dockerfile copies %s, not GRAFT_BINARIES %s' % (graft_pairs(dockerfile),
                                                                                GRAFT_BINARIES))
    base = base_image(dockerfile)
    host_graft = {binary: c2_overlay.sha256(context / 'opgraft-K64i' / binary)
                  for binary in sorted({binary for binary, _ in GRAFT_BINARIES})
                  if (context / 'opgraft-K64i' / binary).is_file()}
    source_shas = {entry.source: c2_overlay.sha256(context / 'overlay' / entry.source) for entry in entries}
    files = [[path, binary == '_ttnncpp.so'] for binary, path in GRAFT_BINARIES]
    files += [[GRAFT_IN_IMAGE + '/' + binary, binary == '_ttnncpp.so'] for binary in sorted(host_graft)]
    files += [[destination, False] for entry in entries for destination in entry.destinations]
    files += [[MANIFEST_IN_IMAGE, False]]
    request = dict(files=files, record=INSTALL_RECORD)
    if checkout:
        request.update(trees=list(TREES), suffixes=list(TREE_SUFFIXES))
    built = docker.probe(image, request)
    based = docker.probe(base, dict(files=[[GRAFT_BINARIES[-1][1], True]]))
    env = docker.env(image)
    profiles_path = context / 'overlay' / 'scripts/ci/qwen_c2_profiles.json'
    with open(str(profiles_path), encoding='utf-8') as handle:
        profiles = json.load(handle)
    qwen_names = {name for name in env if name.startswith('QWEN_')}
    for profile in profiles['profiles'].values():
        qwen_names.update(name for name in profile.get('env', {}) if name.startswith('QWEN_'))

    report = ['(image) %s, base %s' % (image, base)]
    found, lines = check_binaries(host_graft, built['files'], based['files'], qwen_names)
    problems += found + check_runtime_sha(env, built['files'])
    report += lines
    manifest = built['files'].get(MANIFEST_IN_IMAGE, {})
    if manifest.get('sha256') != c2_overlay.sha256(context / 'qwen-c2-overlay.txt'):
        problems.append('(b) the image\'s %s is not the context\'s manifest' % MANIFEST_IN_IMAGE)
    found, lines = check_overlay(entries, source_shas, built['files'], built.get('record'))
    problems += found
    report += lines
    contract = load_contract(context / 'overlay' / 'scripts/ci/serving_c2_contract.py')
    for profile in [None] + sorted(profiles['profiles']):
        found, lines = check_argv(contract, profiles_path, profile, profile or profiles['default'],
                                  docker.boot(image, models, profile))
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
                          built=built, base_probe=based)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--image', required=True, help='the built image (id or name)')
    parser.add_argument('--context', required=True, help='the build context it was built from')
    parser.add_argument('--models', required=True, help='host directory mounted as /models for the boot check')
    parser.add_argument('--checkout', help='a checkout to compare the image trees with (informational)')
    parser.add_argument('--report', help='write the full JSON report here')
    arguments = parser.parse_args(argv)
    problems, report = verify(arguments.image, arguments.context, arguments.models, arguments.checkout)
    if arguments.report:
        Path(arguments.report).write_text(json.dumps(report, indent=1, sort_keys=True) + '\n', encoding='utf-8')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
