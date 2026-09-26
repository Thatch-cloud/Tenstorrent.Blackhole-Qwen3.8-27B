"""G1 of the TT prefix-reuse design (section 2.2): apply the prefix-reuse AST stages to the image at build,
then exercise what they patched.

Conversation prefix reuse (the general-prefix profiles) needs code in two trees the C2 serving image
takes from its P8 base: the TT plugin (the scheduler graft's install point in TTScheduler.__init__, the
runner's request ids and the TTModelInput field that carries them, the worker's block-size assertion) and
the model tree (model.py and qwen36_vllm.py:
the capability flag, restore, resumable loops, captures). Neither tree is an overlay destination, so the
changes are AST stages - text-to-text functions in scripts/ci modules that docker/qwen-c2-overlay.txt lays
into /experiment-scripts/ci - and this tool runs them inside the image build
(docker/qwen-c2-serving.Dockerfile, after the overlay install).

`apply`:
  1. the table: STAGES names targets TARGETS knows, and patches all of them or none - an image with the
     model's capability flag but not the scheduler's trim would serve hits no checkpoint makes exact;
  2. which trees the image imports: vllm_tt_plugin must resolve to PLUGIN_ROOT (the copy P8 installs
     editable and serving_plugin_patch edits) and `models` to /opt/tt-metal, or a correct stage would
     patch a file nothing runs (memory graft-mounted-is-not-graft-executed; design R5);
  3. the anchors: every target must hold the pinned original bytes (the source.sha256 pattern of the
     v235 graft) - all of them are checked before anything is written, and a mismatch names the sha256
     the image holds;
  4. each stage runs from the overlaid tree (--modules), must change its target (a stage that changes
     nothing missed its own anchor) and its output must compile;
  5. every module the stages imported from --modules - the stage modules and whatever they import,
     directly or not (qwen_prefix_model_patch reuses lever_n_model_patch.patch_tp_replay) - is recorded
     with its sha256, and c2_image_provenance (f) requires each to be an overlay destination at the
     context's bytes: a module the manifest does not name ships at the bundle's or P8's version (memory
     serving-image-bundle-provenance; the bundle's patch_tp_replay patches the traced loop only, so the
     eager loop would replay chunk 0 on a hit);
  6. the targets are written, and the record (--record) keeps, per target, the pin, the bytes before and
     after, and which stage module (with its sha256) patched it; c2_image_provenance (f) holds the built
     image to that record, the pins and the overlay's sources.

Steps 2 and 3 are fatal only while STAGES patches something. The integrated G1 table (branch prefix/g1)
patches all six targets, so a base that moved off a pin, or a tree that resolves elsewhere, now fails the
build of EVERY C2 image, whatever its profile - the price of one image serving general-prefix. With the table
emptied nothing is written, and the build only logs, records (`warnings`) and reports (provenance (f)) such a
base. The first build reads the model pins no build has confirmed yet.

`check` (the same RUN, right after apply) exercises what apply patched: each patched target's module is
imported in a fresh interpreter with QWEN_PREFIX_REUSE unset and set to 1, and must load from the patched
file; then P8's installed-plugin tests (P8_INSTALLED_TESTS, what P8 qualified the plugin with) and the
overlay's in-image tests run again, with the switch unset, against the patched plugin and model. With
nothing patched there is nothing to exercise and it says so.

Every patched branch must stay off unless QWEN_PREFIX_REUSE=1, which only the general-prefix profiles set
(serving_c2_contract.prefix_reuse), so exact, c2, c2-gate, coding and general keep their behaviour. check
re-runs P8's tests under the switch unset; each stage module must also ship a CPU test that the patched
code with the switch off behaves as the original (test_qwen_prefix_image refuses a stage module without
an allowlisted test_<module>.py holding a *switch_off* test).

`bringup` holds a served engine's log to the record: the plugin the scheduler graft installed from
([PINDIAG] prefix: install ... plugin=) and the model entry the contract saw imported ([QWEN-C2] prefix:
model tree) must be the files apply patched - under the Thatch layer's PYTHONPATH, which the build's
resolution check (step 2) never sees - and the contract's salt policy must have installed.

Stdlib only; python >= 3.6: it runs in the image (3.10), in the CPU suite, and c2_image_provenance imports
its tables on the rig host.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys

PLUGIN_PACKAGE = 'vllm_tt_plugin'
PLUGIN_ROOT = '/opt/qwen-fast-plugin/src/vllm_tt_plugin'
MODEL_PACKAGE = 'models'
MODEL_TREE = '/opt/tt-metal'
MODEL_ROOT = MODEL_TREE + '/models/demos/blackhole/qwen36/tt'
MODULES = '/experiment-scripts/ci'
RECORD = '/opt/qwen-c2/prefix-stage.json'
SCHEMA = 1
MARK = '[PREFIX-STAGE] '
PREFIX_SWITCH = 'QWEN_PREFIX_REUSE'

# The files the prefix-reuse stages edit: (name, image path, sha256 of the original bytes the stages are
# cut against, where that pin comes from). Every pin is read on every build; it refuses one only while
# STAGES patches something.
TARGETS = (
    ('plugin/scheduler.py', PLUGIN_ROOT + '/scheduler.py',
     'a1bd6257d3a14c904b41b4795b8e8b4b1b132c70fc3a4db9d4340a128a100de4',
     'vllm-tt-plugin bf77cd63 src/vllm_tt_plugin/scheduler.py (git blob); P8 does not patch it'),
    ('plugin/model_input.py', PLUGIN_ROOT + '/model_input.py',
     '8adf4bac4daba576deb27111757bedad69c1d669eb5c15d0ad2d128af2040b54',
     'vllm-tt-plugin bf77cd63 src/vllm_tt_plugin/model_input.py (git blob); P8 does not patch it'),
    ('plugin/model_runner.py', PLUGIN_ROOT + '/model_runner.py',
     'eed4d0fbe0a41fcb18515ad72c0587a05ffa32e615032dd79771331226919530',
     'vllm-tt-plugin bf77cd63 src/vllm_tt_plugin/model_runner.py (git blob); P8 does not patch it'),
    ('plugin/worker.py', PLUGIN_ROOT + '/worker.py',
     '05b99d88a060ede31a686029c3e20a2abd34087dd4670e5fbfb955247406d696',
     'bf77cd63 worker.py after serving_plugin_patch.patch_worker as of P8 commit be9e184e (docker/'
     'qwen-fast-serving.Dockerfile RUNs it once)'),
    ('model/model.py', MODEL_ROOT + '/model.py',
     'c977f3808c39c9dacde5a62a1e30c09dbb55b27d272fecaa9ffea09991270391',
     'the design\'s IMG copy (md5 e4ba08d9), also pinned by dspark-target-hardware.py and full-prefix.py beside '
     'the graft originals; UNVERIFIED as the P8 base\'s bytes until a build or c2_image_provenance --base-drift '
     'reads them'),
    ('model/qwen36_vllm.py', MODEL_ROOT + '/qwen36_vllm.py',
     'cda38c3121b7a61417885469c224c0c69189fda899fbf8361565f4d93125c2fe',
     'the design\'s IMG copy (md5 b5230935), also pinned by dspark-target-hardware.py and full-prefix.py beside '
     'the graft originals; UNVERIFIED as the P8 base\'s bytes until a build or c2_image_provenance --base-drift '
     'reads them'),
)

# The module each target is when a served process imports it (check imports these).
TARGET_MODULES = {
    'plugin/scheduler.py': PLUGIN_PACKAGE + '.scheduler',
    'plugin/model_input.py': PLUGIN_PACKAGE + '.model_input',
    'plugin/model_runner.py': PLUGIN_PACKAGE + '.model_runner',
    'plugin/worker.py': PLUGIN_PACKAGE + '.worker',
    'model/model.py': MODEL_PACKAGE + '.demos.blackhole.qwen36.tt.model',
    'model/qwen36_vllm.py': MODEL_PACKAGE + '.demos.blackhole.qwen36.tt.qwen36_vllm',
}

# P8's installed-plugin tests: the last RUN of docker/qwen-fast-serving.Dockerfile at the P8 base's commit
# runs them, from /opt/tt-metal and under this environment, against the plugin the stages patch
# (test_qwen_prefix_image holds both to that Dockerfile).
P8_INSTALLED_TESTS = ('test_serving_scheduler', 'test_serving_vllm_installed', 'test_serving_dflash_registry_installed')
P8_TEST_ENV = (('VLLM_PLUGINS', ''), ('VLLM_USE_V2_MODEL_RUNNER', '0'), ('HF_HUB_OFFLINE', '1'),
               ('OMP_NUM_THREADS', '2'), ('MKL_NUM_THREADS', '2'))
IMPORT_MARK = 'PREFIXIMPORT '
IMPORT_PROBE = ('import importlib, os, sys\n'
                'module = importlib.import_module(sys.argv[1])\n'
                'print(%r + os.path.realpath(module.__file__))\n' % IMPORT_MARK)

# What a served engine logs about the trees it runs (qwen_prefix_scheduler_patch's install line, and the
# contract's post-import hook on the model entry under a prefix profile).
INSTALL_LINE = re.compile(r'\[PINDIAG\] prefix: install .*?\bplugin=(\S+)')
MODEL_LINE = re.compile(r'\[QWEN-C2\] prefix: model tree (\S+)')
SALT_LINE = re.compile(r'\[QWEN-C2\] prefix: cache_salt kept only when it verifies against the salt key \((\w+)')

# The AST stages, in the order they run: (target name, stage module, function). The module is a
# scripts/ci file the overlay manifest lays into /experiment-scripts/ci; the function takes the target's
# text and returns the patched text, and raises when its own anchor text is missing. A target may be
# patched by several stages, in table order.
#
# The integrated G1 table (branch prefix/g1): all six TARGETS or none (table_problems). Each function is the
# stage module's own text-to-text edit - the same one its CLI stage() runs - and refuses a text whose
# anchors it does not find; the model's two are held to its PATCHED_SHA256 as well (qwen_prefix_model_patch
# ._pinned_patch). The scheduler hook imports qwen_prefix_scheduler_patch and qwen_prefix_registry from the
# plugin package at runtime, and the stage writes no files beside its targets: the overlay manifest lays
# both at the plugin path as well as under /experiment-scripts/ci (test_qwen_prefix_image_closure).
# test_qwen_prefix_image refuses, for every module named here: one not in docker/qwen-c2-overlay.txt; one
# that imports (transitively) a scripts/ci module the manifest does not name; and one without an
# allowlisted scripts/ci/test_<module>.py holding a *switch_off* test. Every runtime module a patched
# target imports (the registry, the scheduler graft) goes in the manifest too.
STAGES = (
    ('plugin/scheduler.py', 'qwen_prefix_scheduler_patch', 'patch_scheduler'),
    ('plugin/model_input.py', 'qwen_prefix_runner_patch', 'patch_model_input'),
    ('plugin/model_runner.py', 'qwen_prefix_runner_patch', 'patch_model_runner'),
    ('plugin/worker.py', 'qwen_prefix_runner_patch', 'patch_worker'),
    ('model/model.py', 'qwen_prefix_model_patch', 'patch_model'),
    ('model/qwen36_vllm.py', 'qwen_prefix_model_patch', 'patch_vllm_entry'),
)


class StageError(RuntimeError):
    """The image is not one the prefix-reuse stages were cut against; the build must stop."""


def log(text):
    sys.stdout.write(MARK + text + '\n')
    sys.stdout.flush()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    with open(path, 'rb') as handle:
        return sha256_bytes(handle.read())


def short(value):
    return str(value)[:16] if value else '(missing)'


def target_names(targets=None):
    return [name for name, _, _, _ in (TARGETS if targets is None else targets)]


def table_problems(targets=None, stages=None):
    """Every reason the stage table cannot be applied as it stands."""
    targets = TARGETS if targets is None else targets
    stages = STAGES if stages is None else stages
    problems = []
    names = target_names(targets)
    if len(set(names)) != len(names):
        problems.append('a target is listed twice in TARGETS')
    seen = set()
    for row in stages:
        if len(row) != 3 or not all(isinstance(item, str) and item for item in row):
            problems.append('stage %r is not (target, module, function)' % (row,))
            continue
        target, module, function = row
        if target not in names:
            problems.append('stage %s.%s patches %s, which TARGETS does not pin' % (module, function, target))
        if row in seen:
            problems.append('stage %s.%s on %s is listed twice' % (module, function, target))
        seen.add(row)
        if '/' in module or module.endswith('.py'):
            problems.append('stage module %r is a module name, not a path' % module)
    if stages:
        patched = {row[0] for row in stages if len(row) == 3}
        missing = [name for name in names if name not in patched]
        if missing:
            problems.append('the stages patch some targets but not %s: prefix reuse is all or nothing (the '
                            'model\'s capability flag without the scheduler\'s trim serves inexact hits)'
                            % ', '.join(missing))
    return problems


def package_locations(name, find_spec=None):
    """The directories the package `name` imports from, in import order (several for a namespace
    package), without executing it; [] when it does not resolve."""
    find_spec = find_spec or importlib.util.find_spec
    try:
        spec = find_spec(name)
    except (ImportError, ValueError):
        return []
    if spec is None:
        return []
    locations = list(getattr(spec, 'submodule_search_locations', None) or ())
    if not locations and getattr(spec, 'origin', None):
        locations = [os.path.dirname(spec.origin)]
    return [os.path.realpath(location) for location in locations]


def package_directory(name, find_spec=None):
    """The first directory the package `name` imports from, or None."""
    locations = package_locations(name, find_spec)
    return locations[0] if locations else None


QWEN_ENTRY = 'demos/blackhole/qwen36/tt/qwen36_vllm.py'


def resolution_problems(find_spec=None, isfile=os.path.isfile):
    """(problems, found): the plugin and model trees this image's python imports must be the ones the
    targets name. For `models` the first location that holds the Qwen tree counts (a namespace
    package may span several)."""
    plugin = package_locations(PLUGIN_PACKAGE, find_spec)
    models = package_locations(MODEL_PACKAGE, find_spec)
    qwen = [location for location in models if isfile(os.path.join(location, QWEN_ENTRY))]
    found = dict(plugin=plugin[0] if plugin else None, models=qwen[0] if qwen else None, model_locations=models)
    problems = []
    if found['plugin'] != os.path.realpath(PLUGIN_ROOT):
        problems.append('%s imports from %s, not %s: the stages would patch a plugin nothing runs (design R5)'
                        % (PLUGIN_PACKAGE, found['plugin'], PLUGIN_ROOT))
    if found['models'] != os.path.realpath(MODEL_TREE + '/models'):
        problems.append('%s.%s imports from %s (models locations %s), not %s/models: the stages would patch a '
                        'model tree nothing runs' % (MODEL_PACKAGE, QWEN_ENTRY[:-3].replace('/', '.'),
                                                     found['models'], models, MODEL_TREE))
    return problems, found


def image_path(root, path):
    return os.path.join(root, path.lstrip('/')) if root else path


def anchor_problems(root='', targets=None):
    """(problems, {name: sha256 or None}) of every target against its pin."""
    targets = TARGETS if targets is None else targets
    problems, seen = [], {}
    for name, path, pin, source in targets:
        real = image_path(root, path)
        if not os.path.isfile(real):
            seen[name] = None
            problems.append('%s: %s is missing' % (name, path))
            continue
        seen[name] = sha256_file(real)
        if seen[name] != pin:
            problems.append('%s: %s is %s, not the pinned %s (%s)' % (name, path, seen[name], pin, source))
    return problems, seen


def load_stage_module(directory, name):
    """(module, path, sha256): a stage module loaded from `directory` by path, so no other tree shadows it."""
    path = os.path.join(directory, name + '.py')
    if not os.path.isfile(path):
        raise StageError('stage module %s is not in %s: add scripts/ci/%s.py to docker/qwen-c2-overlay.txt'
                         % (name, directory, name))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, path, sha256_file(path)


def modules_under(directory, modules=None):
    """[{module, path, sha256}] of every loaded module whose file lies under `directory` - after the stages
    ran, the stage modules and everything they imported from the overlaid tree, directly or not. `path`
    is the file under `directory` as given (the image path the overlay manifest names)."""
    modules = sys.modules if modules is None else modules
    root = os.path.realpath(directory)
    rows = {}
    for name, module in list(modules.items()):
        path = getattr(module, '__file__', None)
        if not isinstance(path, str):
            continue
        real = os.path.realpath(path)
        if not real.startswith(root + os.sep) or not os.path.isfile(real):
            continue
        relative = os.path.relpath(real, root).replace(os.sep, '/')
        rows[relative] = dict(module=name, path=directory.rstrip('/\\') + '/' + relative, sha256=sha256_file(real))
    return [rows[key] for key in sorted(rows)]


def run_stages(texts, paths, modules, stages=None):
    """Apply `stages` to {name: text}; return (texts, applied rows, imported rows). Raises StageError on any
    failure."""
    stages = STAGES if stages is None else stages
    texts = dict(texts)
    applied, loaded = [], {}
    if modules not in sys.path:
        sys.path.insert(0, modules)
    for target, name, function in stages:
        if name not in loaded:
            loaded[name] = load_stage_module(modules, name)
        module, path, digest = loaded[name]
        transform = getattr(module, function, None)
        if not callable(transform):
            raise StageError('%s has no function %s (stage for %s)' % (path, function, target))
        before = texts[target]
        try:
            after = transform(before)
        except Exception as error:
            raise StageError('%s.%s refused %s: %s: %s' % (name, function, target, type(error).__name__, error))
        if not isinstance(after, str):
            raise StageError('%s.%s returned %s for %s, not text' % (name, function, type(after).__name__, target))
        if after == before:
            raise StageError('%s.%s left %s unchanged: its anchor missed' % (name, function, target))
        try:
            compile(after, paths[target], 'exec')
        except SyntaxError as error:
            raise StageError('%s.%s made %s uncompilable: %s' % (name, function, target, error))
        texts[target] = after
        applied.append(dict(target=target, module=name, function=function, path=path, sha256=digest))
        log('%s.%s patched %s' % (name, function, target))
    imported = modules_under(modules) if stages else []
    for row in imported:
        log('the stages ran %s (%s) at %s' % (row['module'], row['path'], row['sha256'][:16]))
    return texts, applied, imported


def write_text(path, text):
    """Replace path's bytes with text (utf-8, newlines as given), keeping its mode."""
    temporary = path + '.qwen-prefix-stage'
    with open(temporary, 'w', encoding='utf-8', newline='') as handle:
        handle.write(text)
    shutil.copymode(path, temporary)
    os.replace(temporary, path)


def apply(modules=MODULES, record=RECORD, root='', targets=None, stages=None, find_spec=None,
          check_resolution=True):
    """The build step. Returns the record; raises StageError, writing nothing, on any problem before the
    write. root, when given, prefixes every image path (a staging tree standing in for the image).
    Resolution and anchor problems are fatal only when `stages` patches something (module docstring)."""
    targets = TARGETS if targets is None else targets
    stages = STAGES if stages is None else stages
    problems = table_problems(targets, stages)
    if problems:
        raise StageError('the stage table: ' + '; '.join(problems))
    strict = bool(stages)
    warnings, found = [], {}
    if check_resolution:
        problems, found = resolution_problems(find_spec)
        if problems and strict:
            raise StageError('; '.join(problems))
        warnings += problems
        log('%s imports from %s; %s from %s' % (PLUGIN_PACKAGE, found['plugin'], MODEL_PACKAGE, found['models']))
    problems, before = anchor_problems(root, targets)
    if problems and strict:
        raise StageError('the anchors (nothing was written): ' + '; '.join(problems))
    warnings += problems
    for warning in warnings:
        log('WARNING (not fatal: no stage patches anything, so nothing is written): ' + warning)
    paths = {name: image_path(root, path) for name, path, _, _ in targets}
    texts, applied, imported = {}, [], []
    if stages:
        for name, path in paths.items():
            with open(path, 'rb') as handle:
                data = handle.read()
            try:
                texts[name] = data.decode('utf-8')
            except UnicodeDecodeError as error:
                raise StageError('%s: %s is not utf-8: %s' % (name, path, error))
        texts, applied, imported = run_stages(texts, paths, modules, stages)
    rows = {}
    for name, path, pin, source in targets:
        if name in texts:
            data = texts[name].encode('utf-8')
            if sha256_bytes(data) != before[name]:
                write_text(paths[name], texts[name])
            after = sha256_file(paths[name])
            if after != sha256_bytes(data):
                raise StageError('%s: %s holds %s after the write' % (name, path, after))
        else:
            after = sha256_file(paths[name]) if os.path.isfile(paths[name]) else None
        rows[name] = dict(path=path, pin=pin, pin_source=source, before=before[name], after=after,
                          patched_by=['%s.%s' % (row['module'], row['function']) for row in applied
                                      if row['target'] == name])
        log('%-22s %s %s -> %s%s' % (name, path, short(before[name]), short(after),
                                     '' if rows[name]['patched_by'] else ' (no stage)'))
    complete = bool(stages) and all(rows[name]['patched_by'] for name in rows)
    result = dict(schema=SCHEMA, targets=rows, stages=applied, imported=imported, resolved=found,
                  complete=complete, warnings=warnings, python=sys.version.split()[0])
    log('%d stage(s) applied; prefix reuse %s' % (
        len(applied), 'grafted into every target' if complete else
        'NOT grafted (no stages): a general-prefix profile cannot serve from this image'))
    if record:
        with open(image_path(root, record) if root and record.startswith('/') else record, 'w',
                  encoding='utf-8') as handle:
            json.dump(result, handle, indent=1, sort_keys=True)
            handle.write('\n')
    return result


def load_record(path):
    try:
        with open(path, encoding='utf-8') as handle:
            return json.load(handle)
    except (OSError, ValueError) as error:
        raise StageError('no readable stage record at %s: %s: %s' % (path, type(error).__name__, error))


def run_process(argv, env, cwd):
    """(exit status, stdout and stderr) of one child."""
    result = subprocess.run(argv, env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return result.returncode, result.stdout.decode('utf-8', 'replace')


def tail(text, lines=12):
    return ' | '.join(text.strip().splitlines()[-lines:])


def check(record=RECORD, tests=(), runner=None, environ=None, python=None):
    """The build's exercise of what apply patched (module docstring). Returns the problems."""
    runner = runner or run_process
    python = python or sys.executable
    loaded = record if isinstance(record, dict) else load_record(record)
    rows = loaded.get('targets') or {}
    patched = [(name, rows[name]) for name in sorted(rows) if rows[name].get('patched_by')]
    if not patched:
        log('check: no stage patched a target, so the image runs the base\'s plugin and model files (the ones '
            'P8 tested); nothing to exercise')
        return []
    base = dict(os.environ if environ is None else environ)
    base.update(P8_TEST_ENV)
    off = dict((key, value) for key, value in base.items() if key != PREFIX_SWITCH)
    on = dict(off)
    on[PREFIX_SWITCH] = '1'
    problems = []
    for name, row in patched:
        module = TARGET_MODULES.get(name)
        if module is None:
            problems.append('%s: no module is known for it (TARGET_MODULES)' % name)
            continue
        wanted = os.path.realpath(row.get('path') or '')
        for label, env in (('unset', off), ('1', on)):
            code, out = runner([python, '-B', '-c', IMPORT_PROBE, module], env, MODEL_TREE)
            files = [line[len(IMPORT_MARK):].strip() for line in out.splitlines() if line.startswith(IMPORT_MARK)]
            if code != 0 or not files:
                problems.append('%s: importing %s with %s %s failed (exit %d): %s' % (
                    name, module, PREFIX_SWITCH, label, code, tail(out)))
            elif files[-1] != wanted:
                problems.append('%s: %s with %s %s imports %s, not the patched %s (memory graft-mounted-is-not-'
                                'graft-executed)' % (name, module, PREFIX_SWITCH, label, files[-1], wanted))
            else:
                log('check: %s imports the patched %s with %s %s' % (module, files[-1], PREFIX_SWITCH, label))
    for cwd, modules, what in ((MODEL_TREE, P8_INSTALLED_TESTS, 'P8\'s installed-plugin tests'),
                               (MODULES, tuple(tests), 'the overlay\'s in-image tests')):
        if not modules:
            continue
        code, out = runner([python, '-B', '-m', 'unittest'] + list(modules), off, cwd)
        sys.stdout.write(out)
        if code != 0:
            problems.append('%s (%s) fail against the patched files with %s unset (exit %d): %s' % (
                what, ' '.join(modules), PREFIX_SWITCH, code, tail(out, 4)))
        else:
            log('check: %s pass against the patched files with %s unset' % (what, PREFIX_SWITCH))
    return problems


def record_problems(record, image_shas, module_shas, overlaid=None, stages=None):
    """What c2_image_provenance (f) refuses in a built image.

    record: the image's RECORD (None when absent); image_shas: {image path: sha256 or None} of every
    target as the image holds it now; module_shas: {stage module name: sha256 of the overlay source the
    context carries}; overlaid: {image path: sha256} of every overlay destination the context lays.
    A target off its pin, or unresolved trees, are problems only while the stage table patches
    something (apply refuses those then); otherwise report lines. Returns (problems, report lines)."""
    stages = STAGES if stages is None else stages
    strict = bool(stages)
    overlaid = overlaid or {}
    if record is None:
        return ['(f) the image has no %s: the prefix-reuse stage never ran' % RECORD], []
    problems, lines = [], []
    if record.get('schema') != SCHEMA:
        problems.append('(f) %s has schema %r, not %d' % (RECORD, record.get('schema'), SCHEMA))
    rows = record.get('targets') or {}
    if sorted(rows) != sorted(target_names()):
        problems.append('(f) the record names targets %s, not %s' % (sorted(rows), sorted(target_names())))
    for name, path, pin, _ in TARGETS:
        row = rows.get(name)
        if row is None:
            continue
        if row.get('path') != path or row.get('pin') != pin:
            problems.append('(f) %s: the record has path %s, pin %s; this checkout pins %s at %s' % (
                name, row.get('path'), row.get('pin'), pin, path))
        if row.get('before') != pin:
            text = '(f) %s: the build found %s at %s, not the pinned %s' % (name, row.get('before'), path, pin)
            if strict:
                problems.append(text)
            else:
                lines.append(text + ' - not fatal while the stage table is empty; the stages would refuse this base')
        now = image_shas.get(path)
        if now != row.get('after'):
            problems.append('(f) %s: %s is %s in the image, not the %s the stage wrote' % (
                name, path, now, row.get('after')))
        wanted = ['%s.%s' % (module, function) for target, module, function in stages if target == name]
        if row.get('patched_by') != wanted:
            problems.append('(f) %s: patched by %s, the stage table says %s' % (name, row.get('patched_by'), wanted))
        if not row.get('patched_by') and row.get('after') != row.get('before'):
            problems.append('(f) %s: no stage patched it, yet the stage left %s where it found %s' % (
                name, row.get('after'), row.get('before')))
    for stage in record.get('stages') or ():
        overlaid_sha = module_shas.get(stage.get('module'))
        if overlaid_sha is None:
            problems.append('(f) stage module %s is not an overlay source of this context' % stage.get('module'))
        elif stage.get('sha256') != overlaid_sha:
            problems.append('(f) stage module %s ran at %s; the context overlays %s' % (
                stage.get('module'), stage.get('sha256'), overlaid_sha))
    imported = record.get('imported')
    if imported is None:
        problems.append('(f) the record does not list the modules the stages imported: an older qwen_prefix_stage '
                        'wrote it')
    for row in imported or ():
        wanted_sha = overlaid.get(row.get('path'))
        if wanted_sha is None:
            problems.append('(f) the stages imported %s from %s, which the overlay does not lay: the image holds the '
                            'bundle\'s or P8\'s version, not this commit\'s - name scripts/ci/%s in '
                            'docker/qwen-c2-overlay.txt' % (row.get('module'), row.get('path'),
                                                            posixpath.basename(str(row.get('path')))))
        elif row.get('sha256') != wanted_sha:
            problems.append('(f) the stages imported %s (%s) at %s; the context overlays %s' % (
                row.get('module'), row.get('path'), row.get('sha256'), wanted_sha))
    if bool(record.get('complete')) != bool(stages):
        problems.append('(f) the record says complete=%s; the stage table has %d stage(s)' % (
            record.get('complete'), len(stages)))
    resolved = record.get('resolved') or {}
    if not resolved.get('plugin') or not resolved.get('models'):
        text = '(f) the record does not say which plugin and model trees the build resolved: %s' % resolved
        if strict:
            problems.append(text)
        else:
            lines.append(text + ' - not fatal while the stage table is empty')
    for warning in record.get('warnings') or ():
        lines.append('(f) the prefix stage warned: %s' % warning)
    lines.append('(f) prefix-reuse stage: %d stage(s), %s; the build resolved %s at %s and %s at %s' % (
        len(record.get('stages') or ()), 'every target grafted' if record.get('complete') else
        'no target grafted: a general-prefix profile cannot serve from this image',
        PLUGIN_PACKAGE, resolved.get('plugin'), MODEL_PACKAGE, resolved.get('models')))
    for name in target_names():
        row = rows.get(name) or {}
        lines.append('(f) %s: %s -> %s by %s' % (name, str(row.get('before'))[:16], str(row.get('after'))[:16],
                                                 row.get('patched_by') or 'nothing'))
    for row in imported or ():
        lines.append('(f) the stages ran %s at %s' % (row.get('path'), str(row.get('sha256'))[:16]))
    return problems, lines


def bringup_problems(log_text, record):
    """The bring-up check (design 2.2): what a served general-prefix engine logged it runs, against the
    files apply patched. (problems, report lines)."""
    if not record:
        return ['bring-up: no prefix-stage record to hold the log to'], []
    rows = record.get('targets') or {}
    scheduler = (rows.get('plugin/scheduler.py') or {}).get('path')
    entry = (rows.get('model/qwen36_vllm.py') or {}).get('path')
    problems, lines = [], []
    if not record.get('complete'):
        problems.append('bring-up: the image\'s prefix stage grafted nothing (complete=%s)' % record.get('complete'))
    plugins = sorted(set(INSTALL_LINE.findall(log_text)))
    models = sorted(set(MODEL_LINE.findall(log_text)))
    if not plugins:
        problems.append('bring-up: no "[PINDIAG] prefix: install" line: the scheduler graft never installed')
    for found in plugins:
        if found != scheduler:
            problems.append('bring-up: the scheduler graft installed from %s, not the patched %s' % (found, scheduler))
        else:
            lines.append('bring-up: the scheduler graft installed from the patched %s' % found)
    if not models:
        problems.append('bring-up: no "[QWEN-C2] prefix: model tree" line: the contract never saw the model entry '
                        'imported')
    salts = SALT_LINE.findall(log_text)
    if not salts:
        problems.append('bring-up: no "[QWEN-C2] prefix: cache_salt kept only when..." line: the salt policy never '
                        'installed, so a client-chosen cache_salt would reach the engine')
    else:
        lines.append('bring-up: the salt policy installed, salt key %s' % ', '.join(sorted(set(salts))))
    for found in models:
        if found != entry:
            problems.append('bring-up: the engine imported the model entry from %s, not the patched %s' % (found, entry))
        else:
            lines.append('bring-up: the model entry came from the patched %s' % found)
    return problems, lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    commands = parser.add_subparsers(dest='command')
    command = commands.add_parser('apply', help='run the stages inside the image build')
    command.add_argument('--modules', default=MODULES)
    command.add_argument('--record', default=RECORD)
    command = commands.add_parser('check', help='exercise what apply patched, inside the image build')
    command.add_argument('--record', default=RECORD)
    command.add_argument('--tests', nargs='*', default=[], help='the overlay\'s in-image test modules')
    command = commands.add_parser('anchors', help='hash a tree laid out as the image (--root) against the pins')
    command.add_argument('--root', required=True)
    command = commands.add_parser('bringup', help='hold a served engine\'s log to the image\'s record')
    command.add_argument('--log', required=True)
    command.add_argument('--record', required=True, help='a copy of the image\'s %s' % RECORD)
    arguments = parser.parse_args(argv)
    if arguments.command == 'apply':
        try:
            apply(arguments.modules, arguments.record)
        except StageError as error:
            sys.stderr.write(MARK + 'REFUSED: %s\n' % error)
            return 1
    elif arguments.command == 'check':
        try:
            problems = check(arguments.record, arguments.tests)
        except StageError as error:
            problems = [str(error)]
        for problem in problems:
            sys.stderr.write(MARK + 'REFUSED: %s\n' % problem)
        return 1 if problems else 0
    elif arguments.command == 'anchors':
        problems, seen = anchor_problems(arguments.root)
        for name, path, pin, _ in TARGETS:
            print('%s  %s  %s' % (seen[name] or '(missing)', path, 'pinned' if seen[name] == pin else 'NOT the pin'))
        return 1 if problems else 0
    elif arguments.command == 'bringup':
        with open(arguments.log, encoding='utf-8', errors='replace') as handle:
            text = handle.read()
        try:
            record = load_record(arguments.record)
        except StageError as error:
            record = None
            sys.stderr.write(MARK + str(error) + '\n')
        problems, lines = bringup_problems(text, record)
        for line in lines:
            log(line)
        for problem in problems:
            sys.stderr.write(MARK + 'FAIL: %s\n' % problem)
        return 1 if problems else 0
    else:
        parser.print_help()
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
