"""A profile that turns prefix reuse on must ship in an image that carries every prefix-reuse stage.

The G1 modules reach the C2 serving image only through docker/qwen-c2-overlay.txt (the one list,
c2_overlay.py) and the prefix stage RUN in docker/qwen-c2-serving.Dockerfile (qwen_prefix_stage.py
apply, which runs the stage table qwen_prefix_stage.STAGES from the overlaid tree). test_c2_image_overlay
only flags files an older layer already holds - a new module in no layer is invisible to it. So a
profile that sets QWEN_PREFIX_REUSE=1 on an image without them would boot with the plugin's stock
scheduler (no hook, no grants) beside a model whose hit path asserts on start_pos > 0 without a
grant: fail-closed, but as a crash loop (memory serving-image-bundle-provenance).

Wherever reuse is on - a profile's env in qwen_c2_profiles.json, or the C2 Dockerfile's ENV - this
holds the image to:
1. the manifest lists qwen_prefix_registry.py and qwen_prefix_scheduler_patch.py at their
   /experiment-scripts/ci path AND at the TT plugin's own copy (/opt/qwen-fast-plugin/src/
   vllm_tt_plugin/<name>: the patched TTScheduler.__init__ imports both from its own package, and
   the table-driven stage writes nothing but its targets), so G1's provenance check
   (c2_image_provenance (b)) hashes the plugin copies against the same source as the tree copies;
   and qwen_prefix_runner_patch.py and qwen_prefix_model_patch.py at their /experiment-scripts/ci path;
2. the Dockerfile RUNs `python3 ... qwen_prefix_stage.py apply --modules /experiment-scripts/ci` after
   the RUN that does `c2_overlay.py install` (before it, the stages would run from the P8 tree's
   copies or none), and not as `--check`/`anchors` only;
3. the stage table patches every target (qwen_prefix_stage.table_problems is empty) with the
   scheduler, runner and model stage modules. The model stage ships with the flag (design section
   2.0.1 item 4: with the flag on and no model graft, today's model ignores start_pos and silently
   rewrites shared blocks), which is why the table is all or nothing.
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
import qwen_prefix_stage  # noqa: E402

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
# The stage modules the table must run (each on the targets it owns).
STAGE_MODULES = {
    'qwen_prefix_scheduler_patch': ('plugin/scheduler.py',),
    'qwen_prefix_runner_patch': ('plugin/model_input.py', 'plugin/model_runner.py', 'plugin/worker.py'),
    'qwen_prefix_model_patch': ('model/model.py', 'model/qwen36_vllm.py'),
}
STAGE_TOOL = '/opt/qwen-c2/qwen_prefix_stage.py'
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


def table_problems(stages, targets=None):
    """What the stage table lacks for prefix reuse ([] when nothing)."""
    problems = ['the stage table: %s' % problem for problem in qwen_prefix_stage.table_problems(targets, stages)]
    if not stages:
        problems.append('qwen_prefix_stage.STAGES is empty: the build patches nothing')
        return problems
    for module, owned in sorted(STAGE_MODULES.items()):
        patched = sorted(target for target, name, _ in stages if name == module)
        if patched != sorted(owned):
            problems.append('the stage table runs %s on %s, not %s' % (module, patched, sorted(owned)))
    return problems


def plumbing_problems(manifest_text, dockerfile_text, stages=None):
    """What an image built from these two files and this stage table lacks for prefix reuse."""
    stages = qwen_prefix_stage.STAGES if stages is None else stages
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
    runs = [(index, command) for index, step in enumerate(steps) if step.startswith('RUN ')
            for command in commands(step) if STAGE_TOOL in command]
    applies = [(index, command) for index, command in runs if ' apply' in ' ' + command]
    if not applies:
        problems.append('no RUN runs %s apply' % STAGE_TOOL)
    else:
        good = [index for index, command in applies
                if index > installs[0] and 'python3' in command.split()[:4]
                and '--modules' in command.split()
                and command.split()[command.split().index('--modules') + 1] == TREE.rstrip('/')]
        if not good:
            index, command = applies[0]
            if index < installs[0]:
                problems.append('%s apply runs before the overlay install (step %d < %d)' % (STAGE_TOOL, index,
                                                                                           installs[0]))
            else:
                problems.append('%s apply does not run the overlaid tree with python3: %s' % (STAGE_TOOL, command))
    problems += table_problems(stages)
    return problems


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
        self.assertIn('profile %s' % name, reuse_enablers(json.dumps(profiles), ''))
        for env in ('ENV A=1 QWEN_PREFIX_REUSE=1', 'ENV QWEN_PREFIX_REUSE="1" ' + BACKSLASH + '\n    B=2'):
            self.assertEqual(reuse_enablers(json.dumps({'profiles': {}}), env + '\n'), ['Dockerfile ENV'], env)
        for env in ('ENV QWEN_PREFIX_REUSE=0', 'ENV QWEN_PREFIX_REUSE_X=1', '# ENV QWEN_PREFIX_REUSE=1'):
            self.assertEqual(reuse_enablers(json.dumps({'profiles': {}}), env + '\n'), [], env)

    def test_the_check_names_what_is_missing(self):
        """Controls on the real files: each piece taken out is named."""
        manifest, dockerfile = read(MANIFEST), read(DOCKERFILE)
        self.assertEqual(plumbing_problems(manifest, dockerfile), [])
        registry_line = [line for line in manifest.split('\n') if line.startswith('scripts/ci/qwen_prefix_registry.py')]
        self.assertEqual(len(registry_line), 1)
        without_plugin_copy = manifest.replace(registry_line[0], registry_line[0].replace(
            '  ' + PLUGIN + 'qwen_prefix_registry.py', ''))
        self.assertEqual(plumbing_problems(without_plugin_copy, dockerfile),
                         ['scripts/ci/qwen_prefix_registry.py does not land at %sqwen_prefix_registry.py' % PLUGIN])
        without_model = manifest.replace('\nscripts/ci/qwen_prefix_model_patch.py\n', '\n')
        self.assertEqual(plumbing_problems(without_model, dockerfile),
                         ['scripts/ci/qwen_prefix_model_patch.py is not in docker/qwen-c2-overlay.txt'])
        steps = [step for step in instructions(dockerfile) if STAGE_TOOL + ' apply' in step]
        self.assertEqual(len(steps), 1)
        no_apply = dockerfile.replace(STAGE_TOOL + ' apply', STAGE_TOOL + ' anchors')
        self.assertEqual(plumbing_problems(manifest, no_apply), ['no RUN runs %s apply' % STAGE_TOOL])
        marker = 'COPY qwen-c2-overlay.txt c2_overlay.py /opt/qwen-c2/\n'
        self.assertEqual(dockerfile.count(marker), 1)
        early = dockerfile.replace(marker, 'RUN python3 -B %s apply --modules /experiment-scripts/ci\n' % STAGE_TOOL
                                   + marker).replace(STAGE_TOOL + ' apply ' + BACKSLASH, STAGE_TOOL + ' anchors '
                                                     + BACKSLASH)
        self.assertEqual([problem.split(' (')[0] for problem in plumbing_problems(manifest, early)],
                         ['%s apply runs before the overlay install' % STAGE_TOOL])
        elsewhere = dockerfile.replace('--modules /experiment-scripts/ci', '--modules /tmp/ci')
        self.assertEqual(len(plumbing_problems(manifest, elsewhere)), 1)
        self.assertIn('does not run the overlaid tree', plumbing_problems(manifest, elsewhere)[0])

    def test_the_table_must_patch_every_target_with_the_three_stage_modules(self):
        self.assertEqual(table_problems(qwen_prefix_stage.STAGES), [])
        self.assertEqual(table_problems(()), ['qwen_prefix_stage.STAGES is empty: the build patches nothing'])
        no_worker = tuple(row for row in qwen_prefix_stage.STAGES if row[0] != 'plugin/worker.py')
        problems = table_problems(no_worker)
        self.assertEqual(len(problems), 2, problems)
        self.assertIn('prefix reuse is all or nothing', problems[0])
        self.assertIn('runs qwen_prefix_runner_patch on', problems[1])
        other = tuple((target, 'some_other_patch' if module == 'qwen_prefix_model_patch' else module, function)
                      for target, module, function in qwen_prefix_stage.STAGES)
        self.assertEqual(table_problems(other), ["the stage table runs qwen_prefix_model_patch on [], not "
                                                 "['model/model.py', 'model/qwen36_vllm.py']"])

    def test_every_stage_function_exists_and_is_the_module_s_own_edit(self):
        """The table's functions are the text-to-text edits each module's own stage() runs."""
        import qwen_prefix_model_patch
        import qwen_prefix_runner_patch
        modules = dict(qwen_prefix_scheduler_patch=qwen_prefix_scheduler_patch,
                       qwen_prefix_runner_patch=qwen_prefix_runner_patch,
                       qwen_prefix_model_patch=qwen_prefix_model_patch)
        for target, module, function in qwen_prefix_stage.STAGES:
            with self.subTest(target=target):
                self.assertTrue(callable(getattr(modules[module], function, None)), (module, function))
        runner_files = dict(qwen_prefix_runner_patch.FILES)
        for target, module, function in qwen_prefix_stage.STAGES:
            if module == 'qwen_prefix_runner_patch':
                self.assertIs(getattr(qwen_prefix_runner_patch, function), runner_files[target.split('/', 1)[1]])

    def test_the_plugin_copies_are_the_scheduler_stage_s_runtime_files(self):
        """What the CLI stage copies beside scheduler.py is exactly what the manifest must also put
        there, so the provenance hash covers every plugin copy the hook imports."""
        plugin_copies = {Path(path).name for needed in REQUIRED.values() for path in needed if path.startswith(PLUGIN)}
        self.assertEqual(plugin_copies, set(qwen_prefix_scheduler_patch.RUNTIME_FILES))
        for module in STAGE_MODULES:
            self.assertTrue((HERE / (module + '.py')).is_file(), module)

    def test_instructions_join_continuations(self):
        text = 'FROM x\n# c\nRUN a; ' + BACKSLASH + '\n    # inner comment\n    b\nENV A=1\n'
        self.assertEqual(instructions(text), ['FROM x', 'RUN a; b', 'ENV A=1'])


if __name__ == '__main__':
    unittest.main()
