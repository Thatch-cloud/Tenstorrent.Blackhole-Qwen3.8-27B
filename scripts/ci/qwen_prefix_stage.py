"""G1 of the TT prefix-reuse design (section 2.2): apply the prefix-reuse AST stages to the image at build.

Conversation prefix reuse (the general-prefix profiles) needs code in two trees the C2 serving image
takes from its P8 base: the TT plugin (the scheduler graft's install point in TTScheduler.__init__, the
runner's request ids, the worker's block-size assertion) and the model tree (model.py and qwen36_vllm.py:
the capability flag, restore, resumable loops, captures). Neither tree is an overlay destination, so the
changes are AST stages - text-to-text functions in scripts/ci modules that docker/qwen-c2-overlay.txt lays
into /experiment-scripts/ci - and this tool runs them, once, inside the image build
(docker/qwen-c2-serving.Dockerfile, after the overlay install):

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
  5. the targets are written, and the record (--record) keeps, per target, the pin, the bytes before and
     after, and which stage module (with its sha256) patched it. c2_image_provenance (f) holds the built
     image to that record, the pins and the overlay's sources.

Every patched branch must stay off unless QWEN_PREFIX_REUSE=1, which only the general-prefix profiles set
(serving_c2_contract.prefix_reuse), so exact, c2, c2-gate, coding and general keep their behaviour. That is
the stage modules' contract; this tool checks bytes, not behaviour.

Stdlib only; python >= 3.6: it runs in the image (3.10), in the CPU suite, and c2_image_provenance imports
its tables on the rig host.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
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

# The files the prefix-reuse stages edit: (name, image path, sha256 of the original bytes the stages are
# cut against, where that pin comes from). Every pin is checked on every build, stages or not.
TARGETS = (
    ('plugin/scheduler.py', PLUGIN_ROOT + '/scheduler.py',
     'a1bd6257d3a14c904b41b4795b8e8b4b1b132c70fc3a4db9d4340a128a100de4',
     'vllm-tt-plugin bf77cd63 src/vllm_tt_plugin/scheduler.py (git blob); P8 does not patch it'),
    ('plugin/model_runner.py', PLUGIN_ROOT + '/model_runner.py',
     'eed4d0fbe0a41fcb18515ad72c0587a05ffa32e615032dd79771331226919530',
     'vllm-tt-plugin bf77cd63 src/vllm_tt_plugin/model_runner.py (git blob); P8 does not patch it'),
    ('plugin/worker.py', PLUGIN_ROOT + '/worker.py',
     '05b99d88a060ede31a686029c3e20a2abd34087dd4670e5fbfb955247406d696',
     'bf77cd63 worker.py after serving_plugin_patch.patch_worker as of P8 commit be9e184e (docker/'
     'qwen-fast-serving.Dockerfile RUNs it once)'),
    ('model/model.py', MODEL_ROOT + '/model.py',
     'c977f3808c39c9dacde5a62a1e30c09dbb55b27d272fecaa9ffea09991270391',
     'the design\'s IMG copy (md5 e4ba08d9); UNVERIFIED as the P8 base\'s bytes until a build or '
     'c2_image_provenance --base-drift reads them'),
    ('model/qwen36_vllm.py', MODEL_ROOT + '/qwen36_vllm.py',
     'cda38c3121b7a61417885469c224c0c69189fda899fbf8361565f4d93125c2fe',
     'the design\'s IMG copy (md5 b5230935); UNVERIFIED as the P8 base\'s bytes until a build or '
     'c2_image_provenance --base-drift reads them'),
)

# The AST stages, in the order they run: (target name, stage module, function). The module is a
# scripts/ci file the overlay manifest lays into /experiment-scripts/ci; the function takes the target's
# text and returns the patched text, and raises when its own anchor text is missing. A target may be
# patched by several stages, in table order.
#
# EMPTY on branch prefix/g1-image: the G1 scheduler and model tracks deliver the stage modules, and the
# integration fills this table - all five TARGETS or none (table_problems). The design's names (section
# 2.2, "Code touched"):
#   ('plugin/scheduler.py',    'qwen_prefix_scheduler_patch', 'patch_scheduler'),
#   ('plugin/model_runner.py', 'qwen_prefix_runner_patch',    'patch_model_runner'),
#   ('plugin/worker.py',       'qwen_prefix_runner_patch',    'patch_worker'),
#   ('model/model.py',         'qwen_prefix_model_patch',     'patch_model'),
#   ('model/qwen36_vllm.py',   'qwen_prefix_model_patch',     'patch_vllm_entry'),
# Each stage module must also be named in docker/qwen-c2-overlay.txt (test_qwen_prefix_image fails
# otherwise), and so must every runtime module a patched target imports (the registry).
STAGES = ()


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


def run_stages(texts, paths, modules, stages=None):
    """Apply `stages` to {name: text}; return (texts, applied rows). Raises StageError on any failure."""
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
    return texts, applied


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
    write. root, when given, prefixes every image path (a staging tree standing in for the image)."""
    targets = TARGETS if targets is None else targets
    stages = STAGES if stages is None else stages
    problems = table_problems(targets, stages)
    if problems:
        raise StageError('the stage table: ' + '; '.join(problems))
    found = {}
    if check_resolution:
        problems, found = resolution_problems(find_spec)
        if problems:
            raise StageError('; '.join(problems))
        log('%s imports from %s; %s from %s' % (PLUGIN_PACKAGE, found['plugin'], MODEL_PACKAGE, found['models']))
    problems, before = anchor_problems(root, targets)
    if problems:
        raise StageError('the anchors (nothing was written): ' + '; '.join(problems))
    paths = {name: image_path(root, path) for name, path, _, _ in targets}
    texts = {}
    for name, path in paths.items():
        with open(path, 'rb') as handle:
            data = handle.read()
        try:
            texts[name] = data.decode('utf-8')
        except UnicodeDecodeError as error:
            raise StageError('%s: %s is not utf-8: %s' % (name, path, error))
    texts, applied = run_stages(texts, paths, modules, stages)
    rows = {}
    for name, path, pin, source in targets:
        data = texts[name].encode('utf-8')
        if sha256_bytes(data) != before[name]:
            write_text(paths[name], texts[name])
        after = sha256_file(paths[name])
        if after != sha256_bytes(data):
            raise StageError('%s: %s holds %s after the write' % (name, path, after))
        rows[name] = dict(path=path, pin=pin, pin_source=source, before=before[name], after=after,
                          patched_by=['%s.%s' % (row['module'], row['function']) for row in applied
                                      if row['target'] == name])
        log('%-22s %s %s -> %s%s' % (name, path, before[name][:16], after[:16],
                                     '' if rows[name]['patched_by'] else ' (no stage)'))
    complete = bool(stages) and all(rows[name]['patched_by'] for name in rows)
    result = dict(schema=SCHEMA, targets=rows, stages=applied, resolved=found, complete=complete,
                  python=sys.version.split()[0])
    log('%d stage(s) applied; prefix reuse %s' % (
        len(applied), 'grafted into every target' if complete else
        'NOT grafted (no stages): a general-prefix profile cannot serve from this image'))
    if record:
        with open(image_path(root, record) if root and record.startswith('/') else record, 'w',
                  encoding='utf-8') as handle:
            json.dump(result, handle, indent=1, sort_keys=True)
            handle.write('\n')
    return result


def record_problems(record, image_shas, module_shas):
    """What c2_image_provenance (f) refuses in a built image.

    record: the image's RECORD (None when absent); image_shas: {image path: sha256 or None} of every
    target as the image holds it now; module_shas: {stage module name: sha256 of the overlay source the
    context carries}. Returns (problems, report lines)."""
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
        if row.get('path') != path or row.get('pin') != pin or row.get('before') != pin:
            problems.append('(f) %s: the record has path %s, pin %s, before %s; this checkout pins %s at %s' % (
                name, row.get('path'), row.get('pin'), row.get('before'), pin, path))
        now = image_shas.get(path)
        if now != row.get('after'):
            problems.append('(f) %s: %s is %s in the image, not the %s the stage wrote' % (
                name, path, now, row.get('after')))
        wanted = ['%s.%s' % (module, function) for target, module, function in STAGES if target == name]
        if row.get('patched_by') != wanted:
            problems.append('(f) %s: patched by %s, the stage table says %s' % (name, row.get('patched_by'), wanted))
        if not row.get('patched_by') and row.get('after') != pin:
            problems.append('(f) %s: no stage patched it, yet the image holds %s' % (name, row.get('after')))
    for stage in record.get('stages') or ():
        overlaid = module_shas.get(stage.get('module'))
        if overlaid is None:
            problems.append('(f) stage module %s is not an overlay source of this context' % stage.get('module'))
        elif stage.get('sha256') != overlaid:
            problems.append('(f) stage module %s ran at %s; the context overlays %s' % (
                stage.get('module'), stage.get('sha256'), overlaid))
    if bool(record.get('complete')) != bool(STAGES):
        problems.append('(f) the record says complete=%s; the stage table has %d stage(s)' % (
            record.get('complete'), len(STAGES)))
    resolved = record.get('resolved') or {}
    if not resolved.get('plugin') or not resolved.get('models'):
        problems.append('(f) the record does not say which plugin and model trees the build resolved: %s' % resolved)
    lines.append('(f) prefix-reuse stage: %d stage(s), %s; the build resolved %s at %s and %s at %s' % (
        len(record.get('stages') or ()), 'every target grafted' if record.get('complete') else
        'no target grafted: a general-prefix profile cannot serve from this image',
        PLUGIN_PACKAGE, resolved.get('plugin'), MODEL_PACKAGE, resolved.get('models')))
    for name in target_names():
        row = rows.get(name) or {}
        lines.append('(f) %s: %s -> %s by %s' % (name, str(row.get('before'))[:16], str(row.get('after'))[:16],
                                                 row.get('patched_by') or 'nothing'))
    return problems, lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    commands = parser.add_subparsers(dest='command')
    command = commands.add_parser('apply', help='run the stages inside the image build')
    command.add_argument('--modules', default=MODULES)
    command.add_argument('--record', default=RECORD)
    command = commands.add_parser('anchors', help='hash a tree laid out as the image (--root) against the pins')
    command.add_argument('--root', required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == 'apply':
        try:
            apply(arguments.modules, arguments.record)
        except StageError as error:
            sys.stderr.write(MARK + 'REFUSED: %s\n' % error)
            return 1
    elif arguments.command == 'anchors':
        problems, seen = anchor_problems(arguments.root)
        for name, path, pin, _ in TARGETS:
            print('%s  %s  %s' % (seen[name] or '(missing)', path, 'pinned' if seen[name] == pin else 'NOT the pin'))
        return 1 if problems else 0
    else:
        parser.print_help()
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
