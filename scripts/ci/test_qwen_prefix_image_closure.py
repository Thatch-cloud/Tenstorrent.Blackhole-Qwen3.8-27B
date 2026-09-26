"""A profile that turns prefix reuse on must ship in an image that carries every prefix-reuse stage.

The G1 modules reach the C2 serving image only through docker/qwen-c2-overlay.txt (the one list,
c2_overlay.py) and RUN steps in docker/qwen-c2-serving.Dockerfile. Neither names them yet, and
test_c2_image_overlay only flags files an older layer already holds - a new module in no layer is
invisible to it. So a profile that sets QWEN_PREFIX_REUSE=1 on today's image would boot with the
plugin's stock scheduler (no hook, no grants) beside a model whose hit path asserts on start_pos > 0
without a grant: fail-closed, but as a crash loop (memory serving-image-bundle-provenance).

Wherever reuse is on - a profile's env in qwen_c2_profiles.json, or the C2 Dockerfile's ENV - this
holds the image to:
1. the manifest lists qwen_prefix_registry.py and qwen_prefix_scheduler_patch.py at their
   /experiment-scripts/ci path AND at the TT plugin's own copy (/opt/qwen-fast-plugin/src/
   vllm_tt_plugin/<name>, which the scheduler stage copies beside the patched scheduler.py), so G1's
   provenance check (c2_image_provenance (b)) hashes the plugin copies against the same source as
   the tree copies; and qwen_prefix_runner_patch.py and qwen_prefix_model_patch.py at their
   /experiment-scripts/ci path;
2. the Dockerfile RUNs each stage with python3 from /experiment-scripts/ci after the RUN that does
   `c2_overlay.py install` (before it, the stages would run from the P8 tree's copies or none), the
   scheduler and runner stages on the plugin package, none of them with --check. The model stage
   ships with the flag (design section 2.0.1 item 4: with the flag on and no model graft, today's
   model ignores start_pos and silently rewrites shared blocks).
With the flag set nowhere it checks nothing about the image; its controls run either way.
"""

import json
import re
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import c2_overlay  # noqa: E402
import qwen_prefix_scheduler_patch  # noqa: E402

MANIFEST = ROOT / 'docker' / 'qwen-c2-overlay.txt'
DOCKERFILE = ROOT / 'docker' / 'qwen-c2-serving.Dockerfile'
PROFILES = HERE / 'qwen_c2_profiles.json'
FLAG = 'QWEN_PREFIX_REUSE'
TREE = '/experiment-scripts/ci/'
PLUGIN = c2_overlay.PLUGIN
PLUGIN_PACKAGE = PLUGIN.rstrip('/')
# source -> the destinations it must reach.
REQUIRED = {
    'scripts/ci/qwen_prefix_registry.py': (TREE + 'qwen_prefix_registry.py', PLUGIN + 'qwen_prefix_registry.py'),
    'scripts/ci/qwen_prefix_scheduler_patch.py': (TREE + 'qwen_prefix_scheduler_patch.py',
                                                  PLUGIN + 'qwen_prefix_scheduler_patch.py'),
    'scripts/ci/qwen_prefix_runner_patch.py': (TREE + 'qwen_prefix_runner_patch.py',),
    'scripts/ci/qwen_prefix_model_patch.py': (TREE + 'qwen_prefix_model_patch.py',),
}
# stage script -> the package it must be run on (None: the stage's own default target).
STAGES = (
    ('qwen_prefix_scheduler_patch.py', PLUGIN_PACKAGE),
    ('qwen_prefix_runner_patch.py', PLUGIN_PACKAGE),
    ('qwen_prefix_model_patch.py', None),
)
BACKSLASH = chr(92)


def read(path):
    return Path(path).read_text(encoding='utf-8').replace('\r\n', '\n')


def instructions(text):
    """The Dockerfile's instructions, continuation lines joined (comment lines inside dropped)."""
    steps, current = [], ''
    for line in text.replace('\r\n', '\n').split('\n'):
        stripped = line.strip()
        if stripped.startswith('#') or (not stripped and not current):
            continue
        if stripped.endswith(BACKSLASH):
            current += stripped[:-1].rstrip() + ' '
            continue
        current += stripped
        if current.strip():
            steps.append(current.strip())
        current = ''
    if current.strip():
        steps.append(current.strip())
    return steps


def reuse_enablers(profiles_text, dockerfile_text):
    """Where QWEN_PREFIX_REUSE=1 is set: 'profile <name>' or 'Dockerfile ENV'."""
    found = []
    for name, profile in sorted(json.loads(profiles_text)['profiles'].items()):
        if str((profile.get('env') or {}).get(FLAG)) == '1':
            found.append('profile %s' % name)
    for step in instructions(dockerfile_text):
        if step.startswith('ENV ') and re.search(r'(^|\s)%s=("?)1\2(\s|$)' % FLAG, step[4:]):
            found.append('Dockerfile ENV')
    return found


def commands(step):
    """A RUN step's shell commands (split on ; && ||)."""
    return [part.strip() for part in re.split(r';|&&|[|][|]', step[len('RUN '):]) if part.strip()]


def plumbing_problems(manifest_text, dockerfile_text):
    """What an image built from these two files lacks for prefix reuse ([] when nothing)."""
    try:
        entries = c2_overlay.parse_manifest(manifest_text)
    except c2_overlay.ManifestError as error:
        return ['docker/qwen-c2-overlay.txt does not parse: %s' % error]
    problems = []
    destinations = {entry.source: entry.destinations for entry in entries}
    for source, needed in sorted(REQUIRED.items()):
        if source not in destinations:
            problems.append('%s is not in docker/qwen-c2-overlay.txt' % source)
            continue
        missing = [path for path in needed if path not in destinations[source]]
        if missing:
            problems.append('%s does not land at %s' % (source, ', '.join(missing)))
    steps = instructions(dockerfile_text)
    installs = [index for index, step in enumerate(steps)
                if step.startswith('RUN ') and 'c2_overlay.py install' in step]
    if len(installs) != 1:
        problems.append('expected one RUN of c2_overlay.py install in the Dockerfile, found %d' % len(installs))
        return problems
    for script, package in STAGES:
        path = TREE + script
        runs = [(index, command) for index, step in enumerate(steps) if step.startswith('RUN ')
                for command in commands(step) if path in command]
        if not runs:
            problems.append('no RUN runs %s' % path)
            continue
        good = [index for index, command in runs
                if index > installs[0] and command.split()[0].startswith('python3') and '--check' not in command.split()
                and (package is None or package in command.split())]
        if good:
            continue
        index, command = runs[0]
        if index < installs[0]:
            problems.append('%s runs before the overlay install (step %d < %d)' % (path, index, installs[0]))
        elif '--check' in command.split():
            problems.append('%s runs with --check only: %s' % (path, command))
        elif package is not None and package not in command.split():
            problems.append('%s is not run on %s: %s' % (path, package, command))
        else:
            problems.append('%s is not run with python3: %s' % (path, command))
    return problems


COMPLETE_MANIFEST_LINES = (
    'scripts/ci/qwen_prefix_registry.py  /experiment-scripts/ci/qwen_prefix_registry.py  '
    + PLUGIN + 'qwen_prefix_registry.py',
    'scripts/ci/qwen_prefix_scheduler_patch.py  /experiment-scripts/ci/qwen_prefix_scheduler_patch.py  '
    + PLUGIN + 'qwen_prefix_scheduler_patch.py',
    'scripts/ci/qwen_prefix_runner_patch.py',
    'scripts/ci/qwen_prefix_model_patch.py',
)
STAGE_RUN = ('RUN set -eu; ' + BACKSLASH + '\n'
             '    python3 -B /experiment-scripts/ci/qwen_prefix_scheduler_patch.py ' + PLUGIN_PACKAGE + '; '
             + BACKSLASH + '\n'
             '    python3 -B /experiment-scripts/ci/qwen_prefix_runner_patch.py ' + PLUGIN_PACKAGE + '; '
             + BACKSLASH + '\n'
             '    python3 -B /experiment-scripts/ci/qwen_prefix_model_patch.py\n')


class PrefixImageClosureTests(unittest.TestCase):
    def test_wherever_reuse_is_on_the_image_carries_every_stage(self):
        enablers = reuse_enablers(read(PROFILES), read(DOCKERFILE))
        if not enablers:
            return  # nothing turns prefix reuse on; the controls below still run
        problems = plumbing_problems(read(MANIFEST), read(DOCKERFILE))
        self.assertEqual(problems, [], '%s set %s=1, but the C2 image would not carry prefix reuse:\n%s'
                         % (', '.join(enablers), FLAG, '\n'.join(problems)))

    def test_the_enablers_are_found_in_profiles_and_the_dockerfile_env(self):
        profiles = json.loads(read(PROFILES))
        name = sorted(profiles['profiles'])[0]
        profiles['profiles'][name].setdefault('env', {})[FLAG] = '1'
        self.assertEqual(reuse_enablers(json.dumps(profiles), ''), ['profile %s' % name])
        for env in ('ENV A=1 QWEN_PREFIX_REUSE=1', 'ENV QWEN_PREFIX_REUSE="1" ' + BACKSLASH + '\n    B=2'):
            self.assertEqual(reuse_enablers(json.dumps({'profiles': {}}), env + '\n'), ['Dockerfile ENV'], env)
        for env in ('ENV QWEN_PREFIX_REUSE=0', 'ENV QWEN_PREFIX_REUSE_X=1', '# ENV QWEN_PREFIX_REUSE=1'):
            self.assertEqual(reuse_enablers(json.dumps({'profiles': {}}), env + '\n'), [], env)

    def test_the_real_files_with_the_plumbing_added_pass(self):
        manifest = read(MANIFEST) + '\n' + '\n'.join(COMPLETE_MANIFEST_LINES) + '\n'
        dockerfile = read(DOCKERFILE) + '\n' + STAGE_RUN
        self.assertEqual(plumbing_problems(manifest, dockerfile), [])

    def test_the_check_names_what_is_missing(self):
        """Controls on the real files: each missing piece is named."""
        manifest = read(MANIFEST)
        dockerfile = read(DOCKERFILE)
        complete = manifest + '\n' + '\n'.join(COMPLETE_MANIFEST_LINES) + '\n'
        staged = dockerfile + '\n' + STAGE_RUN
        today = plumbing_problems(manifest, dockerfile)
        if any('qwen_prefix' in line for line in manifest.split('\n') if not line.lstrip().startswith('#')):
            self.skipTest('the manifest already names prefix modules; the today-control no longer applies')
        self.assertIn('scripts/ci/qwen_prefix_registry.py is not in docker/qwen-c2-overlay.txt', today)
        self.assertIn('no RUN runs /experiment-scripts/ci/qwen_prefix_scheduler_patch.py', today)
        without_plugin_copy = complete.replace('  ' + PLUGIN + 'qwen_prefix_registry.py', '')
        self.assertEqual(plumbing_problems(without_plugin_copy, staged),
                         ['scripts/ci/qwen_prefix_registry.py does not land at %sqwen_prefix_registry.py' % PLUGIN])
        without_model = complete.replace('scripts/ci/qwen_prefix_model_patch.py\n', '')
        self.assertEqual(plumbing_problems(without_model, staged),
                         ['scripts/ci/qwen_prefix_model_patch.py is not in docker/qwen-c2-overlay.txt'])
        marker = 'COPY qwen-c2-overlay.txt c2_overlay.py /opt/qwen-c2/\n'
        self.assertEqual(dockerfile.count(marker), 1)
        early = dockerfile.replace(marker, STAGE_RUN + marker)
        self.assertEqual([problem.split(' (')[0] for problem in plumbing_problems(complete, early)],
                         ['%s%s runs before the overlay install' % (TREE, script) for script, _ in STAGES])
        checked = dockerfile + '\n' + STAGE_RUN.replace('qwen_prefix_runner_patch.py ', 'qwen_prefix_runner_patch.py --check ')
        self.assertEqual(len(plumbing_problems(complete, checked)), 1)
        self.assertIn('--check only', plumbing_problems(complete, checked)[0])
        elsewhere = dockerfile + '\n' + STAGE_RUN.replace('qwen_prefix_scheduler_patch.py ' + PLUGIN_PACKAGE,
                                                           'qwen_prefix_scheduler_patch.py /tmp/plugin')
        self.assertIn('is not run on %s' % PLUGIN_PACKAGE, plumbing_problems(complete, elsewhere)[0])

    def test_the_plugin_copies_are_the_scheduler_stage_s_runtime_files(self):
        """What the stage copies beside scheduler.py is exactly what the manifest must also put
        there, so the provenance hash covers every plugin copy the hook imports."""
        plugin_copies = {Path(path).name for needed in REQUIRED.values() for path in needed if path.startswith(PLUGIN)}
        self.assertEqual(plugin_copies, set(qwen_prefix_scheduler_patch.RUNTIME_FILES))
        for script, _ in STAGES[:2]:
            self.assertTrue((HERE / script).is_file(), script)

    def test_instructions_join_continuations(self):
        text = 'FROM x\n# c\nRUN a; ' + BACKSLASH + '\n    # inner comment\n    b\nENV A=1\n'
        self.assertEqual(instructions(text), ['FROM x', 'RUN a; b', 'ENV A=1'])


if __name__ == '__main__':
    unittest.main()
