"""Serve ANOTHER checkpoint of the same architecture: strict, default off (stdlib only, Python 3.7 syntax).

    python3 scripts/ci/qwen_c2_checkpoint.py check --hub <hub dir> --repo <org/name> --revision <40 hex> [--hash-shards]
    python3 scripts/ci/qwen_c2_checkpoint.py cache-dir --base <TT_CACHE_PATH> --repo <org/name> --revision <40 hex>
    python3 scripts/ci/qwen_c2_checkpoint.py record --snapshot <snapshot dir>

The C2 serving stack serves ONE checkpoint, the base of qwen_c2_checkpoints.json: every profile of qwen_c2_profiles.json pins its snapshot
(profile["snapshots"], "@SNAPSHOT@" in the engine argv and in qwen_fast_runtime.target_snapshot) and the image's TT_CACHE_PATH names the directory
of the converted weights. That directory holds one file per tensor, named by the tensor and by nothing else (tensor_cache_bfp8/<tensor>.tensorbin):
a second checkpoint booted on it would load the FIRST checkpoint's converted weights without a word. This module is the one way to select another:

  QWEN_C2_MODEL_REPO + QWEN_C2_MODEL_REVISION   both, the 40-hex revision (never a branch or a tag), a pair the catalog lists. Unset (or empty), or the
                                                base's own pair: nothing below runs and the stack is byte-for-byte what it was (the base keeps its warm cache).

With a pair set, serving_c2_contract.boot (every python process of the image; the checks below only in the API server) does exactly this:

  1. the snapshot is the profile's own pinned location with the base's hub directory name replaced by the checkpoint's (so /models and /models/hub both
     work as before); a missing snapshot refuses the boot;
  2. API server only: every IDENTITY file of the snapshot (config.json, generation_config.json, tokenizer_config.json, chat_template.jinja, tokenizer.json, vocab.json,
     merges.txt, the two preprocessor configs) must hash to the BASE's digest, the safetensors index must carry the base's weight_map, every weight shard must be there with
     its pinned size, and no download may still be running (*.incomplete in the blob directory). Anything else refuses the boot (exit 78): this stack serves the same
     architecture with other weights, never another tokenizer or template;
  3. TT_CACHE_PATH becomes <TT_CACHE_PATH>-ckpt/<org>--<name>--<revision>, in every process (QWEN_C2_WEIGHT_CACHE_BASE remembers the base's so a child process, which
     inherits the keyed value, derives the same directory). The base's own tree is never read or written by the checkpoint. The API server writes CHECKPOINT.json
     beside the converted tensors (the pair and the weight_map digest) and refuses a directory whose marker names another pair;
  4. HF_MODEL and MODEL_WEIGHTS_DIR, which the model loader prefers over the engine's --model, must be unset or the snapshot (else the boot is refused: the loader would
     read the base through them);
  5. the engine's worker re-checks, once the model is built and before the fast path attaches (serving_startup.start -> startup_problems): the model's checkpoint
     directory is the snapshot and its weight cache directory is under the keyed one. A disagreement stops the worker (fail closed, after the load: the load itself is
     what the first four steps keep from reading the wrong bytes).

Log lines (the job's READ rule, c2_smoke_check --expect-checkpoint): "[QWEN-C2] checkpoint <repo>@<revision> ..." from the API server and
"[QWEN-C2] checkpoint bound ..." from the worker; an unselected run logs neither.
"""
import argparse
import hashlib
import json
import os
import re
import sys

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qwen_c2_checkpoints.json')
REPO_ENV = 'QWEN_C2_MODEL_REPO'
REVISION_ENV = 'QWEN_C2_MODEL_REVISION'
KEY_ENV = 'QWEN_C2_CHECKPOINT_KEY'
SNAPSHOT_ENV = 'QWEN_C2_CHECKPOINT_SNAPSHOT'
CACHE_BASE_ENV = 'QWEN_C2_WEIGHT_CACHE_BASE'
DERIVED_ENV = (KEY_ENV, SNAPSHOT_ENV, CACHE_BASE_ENV)
LOADER_ENV = ('HF_MODEL', 'MODEL_WEIGHTS_DIR')
CACHE_ENV = 'TT_CACHE_PATH'
CACHE_ROOT_SUFFIX = '-ckpt'
MARKER = 'CHECKPOINT.json'
LINE = 'checkpoint'
BOUND_LINE = 'checkpoint bound'
REPO = re.compile(r'[0-9A-Za-z._-]{1,96}/[0-9A-Za-z._-]{1,96}')
REVISION = re.compile(r'[0-9a-f]{40}')
SHARD = re.compile(r'model-[0-9]{5}-of-[0-9]{5}\.safetensors')


class CheckpointError(ValueError):
    """A selection, catalog, snapshot or cache the stack must not serve; the message says which."""


class Selection(object):
    """One catalog entry picked by the environment: repo, revision, the cache key, the entry and whether it is the base itself."""

    def __init__(self, repo, revision, entry, is_base):
        self.repo, self.revision, self.entry, self.is_base = repo, revision, entry, is_base
        self.key = cache_key(repo, revision)

    def __repr__(self):
        return 'Selection(%s@%s%s)' % (self.repo, self.revision, ', base' if self.is_base else '')


def hub_dirname(repo):
    """The hub cache's directory for a repo: org/name -> models--org--name."""
    if not REPO.fullmatch(repo):
        raise CheckpointError('%r is not an org/name repository id' % (repo,))
    return 'models--' + repo.replace('/', '--')


def cache_key(repo, revision):
    """The converted-weight directory's name: org--name--<full revision>, so two checkpoints (or two revisions of one) can never share it."""
    hub_dirname(repo)
    if not REVISION.fullmatch(revision):
        raise CheckpointError('%r is not a 40-hex revision (a branch or a tag could move under the cache)' % (revision,))
    return '%s--%s' % (repo.replace('/', '--'), revision)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def weight_map_digest(weight_map):
    return hashlib.sha256(json.dumps(weight_map, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def load_catalog(path=None):
    """The catalog, held to its own rule: every checkpoint equals the base in every identity file, the weight_map, the tensor count, the shard names and the shard sizes."""
    with open(path or DATA, encoding='utf-8') as handle:
        catalog = json.load(handle)
    base = catalog['base']
    names = catalog['identity_files']
    for entry in [base] + list(catalog['checkpoints']):
        hub_dirname(entry['repo'])
        if not REVISION.fullmatch(entry['revision']):
            raise CheckpointError('%s: the revision must be 40 hex' % entry['repo'])
        if sorted(entry['files']) != sorted(names):
            raise CheckpointError('%s: files must name exactly the identity files' % entry['repo'])
        if not entry['shards'] or not all(SHARD.fullmatch(name) for name in entry['shards']):
            raise CheckpointError('%s: shards must be model-NNNNN-of-NNNNN.safetensors names' % entry['repo'])
    seen = set()
    for entry in catalog['checkpoints']:
        pair = (entry['repo'], entry['revision'])
        if pair in seen or pair == (base['repo'], base['revision']):
            raise CheckpointError('%s@%s is listed twice' % pair)
        seen.add(pair)
        for name in names:
            if entry['files'][name] != base['files'][name]:
                raise CheckpointError('%s: %s differs from the base\'s: a checkpoint with another config, tokenizer or template is another model, not this stack\'s' % (entry['repo'], name))
        for field in ('weight_map_sha256', 'tensors', 'total_size'):
            if entry[field] != base[field]:
                raise CheckpointError('%s: %s differs from the base\'s (another architecture or shape)' % (entry['repo'], field))
        if sorted(entry['shards']) != sorted(base['shards']) or any(entry['shards'][n]['size'] != base['shards'][n]['size'] for n in base['shards']):
            raise CheckpointError('%s: the weight shards (names, sizes) differ from the base\'s' % entry['repo'])
    return catalog


def selection(environ, catalog=None):
    """None when no checkpoint is asked for (both names unset or empty); the Selection otherwise; CheckpointError for one name alone, a malformed or unlisted pair."""
    repo, revision = (environ.get(REPO_ENV) or '').strip(), (environ.get(REVISION_ENV) or '').strip()
    if not repo and not revision:
        return None
    if not repo or not revision:
        raise CheckpointError('%s and %s go together (got %s only)' % (REPO_ENV, REVISION_ENV, REPO_ENV if repo else REVISION_ENV))
    cache_key(repo, revision)
    catalog = catalog or load_catalog()
    base = catalog['base']
    if (repo, revision) == (base['repo'], base['revision']):
        return Selection(repo, revision, base, True)
    for entry in catalog['checkpoints']:
        if (repo, revision) == (entry['repo'], entry['revision']):
            return Selection(repo, revision, entry, False)
    raise CheckpointError('%s@%s is not a checkpoint of qwen_c2_checkpoints.json (listed: %s)' % (repo, revision, ', '.join(
        '%s@%s' % (item['repo'], item['revision'][:12]) for item in [base] + list(catalog['checkpoints']))))


def served_model(repo, revision, catalog=None):
    """The name the checkpoint is served under (the catalog's served_model; the base keeps the platform's own name)."""
    catalog = catalog or load_catalog()
    for entry in catalog['checkpoints']:
        if (repo, revision) == (entry['repo'], entry['revision']):
            return entry.get('served_model') or repo
    return ''


def base_location(catalog):
    base = catalog['base']
    return '%s/snapshots/%s' % (hub_dirname(base['repo']), base['revision'])


def snapshot_candidates(pinned, sel, catalog):
    """The profile's pinned snapshot paths with the base's hub directory replaced by the selection's: the same roots, in the same order."""
    where = base_location(catalog)
    candidates = []
    for path in pinned:
        if not path.endswith('/' + where):
            raise CheckpointError('the profile pins %s, which is not the base snapshot %s: a checkpoint cannot be derived from it' % (path, where))
        candidates.append('%s/%s/snapshots/%s' % (path[:-len(where) - 1], hub_dirname(sel.repo), sel.revision))
    return candidates


def resolve_snapshot(pinned, sel, catalog, exists=os.path.isdir):
    for candidate in snapshot_candidates(pinned, sel, catalog):
        if exists(candidate):
            return candidate
    raise CheckpointError('none of the snapshots of %s@%s is mounted: %s' % (sel.repo, sel.revision, ', '.join(snapshot_candidates(pinned, sel, catalog))))


def repo_of_path(path):
    """org/name of a hub snapshot path (.../models--org--name/snapshots/<revision>), or None for any other path."""
    parts = str(path).rstrip('/').split('/')
    if len(parts) >= 3 and parts[-2] == 'snapshots' and parts[-3].startswith('models--'):
        org, separator, name = parts[-3][len('models--'):].partition('--')
        if separator and org and name:
            return '%s/%s' % (org, name)
    return None


def platform_model_problems(asked, sel, catalog):
    """[problem] when the model id the platform asks for (vLLM's --model, before the contract rewrites it) is a listed checkpoint other than the one this image serves.

    The image serves the base unless QWEN_C2_MODEL_REPO/REVISION say otherwise, whatever the platform passes (the profile owns the engine's model): so a platform that asks for
    the Swift checkpoint of an image with no selection would be answered by the base under Swift's name, and one that asks for the base of an image set to Swift would be answered
    by Swift under the base's. Both are refused. An id the catalog does not list (a local path, a name of the platform's own) is not judged."""
    if not asked:
        return []
    listed = [catalog['base']['repo']] + [entry['repo'] for entry in catalog['checkpoints']]
    serves = sel.repo if sel is not None else catalog['base']['repo']
    named = asked if asked in listed else repo_of_path(asked)
    if named in listed and named != serves:
        return ['the platform asks for the model %s but this image serves %s (%s)' % (named, serves, 'QWEN_C2_MODEL_REPO/REVISION name it' if sel is not None and not sel.is_base else 'no checkpoint is selected: the base')]
    return []


def keyed_cache_dir(base_cache, sel):
    """The converted-weight directory of a selection. The base (or no selection) keeps `base_cache` itself: its warm cache is not touched."""
    if sel is None or sel.is_base:
        return base_cache
    if not base_cache or not base_cache.strip('/'):
        raise CheckpointError('%s is empty: the converted weights of %s@%s cannot be keyed' % (CACHE_ENV, sel.repo, sel.revision))
    return '%s%s/%s' % (base_cache.rstrip('/'), CACHE_ROOT_SUFFIX, sel.key)


def verify_snapshot(snapshot, sel, catalog, hash_shards=False, listing=os.listdir):
    """[problem] for a snapshot directory against the selection's entry and the base's identity: [] when it may be served."""
    base, entry = catalog['base'], sel.entry
    problems = []
    for name in catalog['identity_files']:
        path = os.path.join(snapshot, name)
        try:
            actual = sha256_file(path)
        except OSError:
            problems.append('%s is missing from %s' % (name, snapshot))
            continue
        if actual != base['files'][name]:
            problems.append('%s differs from the base\'s (%s, not %s): another config, tokenizer or template is another model' % (name, actual[:16], base['files'][name][:16]))
    index = os.path.join(snapshot, 'model.safetensors.index.json')
    try:
        with open(index, encoding='utf-8') as handle:
            document = json.load(handle)
        weight_map = document['weight_map']
        if weight_map_digest(weight_map) != entry['weight_map_sha256']:
            problems.append('the safetensors index names other tensors or shards than the base\'s (weight_map digest)')
        if len(weight_map) != entry['tensors']:
            problems.append('the safetensors index lists %d tensors, the base has %d' % (len(weight_map), entry['tensors']))
        if int(document['metadata']['total_size']) != entry['total_size']:
            problems.append('the safetensors index total_size differs from the base\'s')
    except (OSError, ValueError, KeyError, TypeError) as error:
        problems.append('model.safetensors.index.json unreadable: %s' % type(error).__name__)
    for name, pin in sorted(entry['shards'].items()):
        path = os.path.join(snapshot, name)
        try:
            size = os.stat(path).st_size
        except OSError:
            problems.append('%s is missing from %s' % (name, snapshot))
            continue
        if size != pin['size']:
            problems.append('%s is %d bytes, the pin says %d (a partial or another file)' % (name, size, pin['size']))
        elif hash_shards and sha256_file(path) != pin['sha256']:
            problems.append('%s does not hash to its pinned sha256' % name)
    blobs = os.path.join(os.path.dirname(os.path.dirname(snapshot.rstrip('/'))), 'blobs')
    try:
        running = [name for name in listing(blobs) if name.endswith('.incomplete')]
    except OSError:
        running = []
    if running:
        problems.append('a download is still running or was cut (%d *.incomplete in the blob directory)' % len(running))
    return problems


def within(path, root):
    path, root = os.path.normpath(str(path)), os.path.normpath(str(root))
    return path == root or path.startswith(root.rstrip('/') + '/')


def configure_environment(environ, sel, snapshot):
    """Set the process environment for a selection (idempotent: a child process re-derives the same values). -> the facts to log, or None when nothing is selected.

    The base, or no selection, removes the derived names and leaves TT_CACHE_PATH exactly as found."""
    for name in (KEY_ENV, SNAPSHOT_ENV):
        environ.pop(name, None)
    if sel is None or sel.is_base:
        environ.pop(CACHE_BASE_ENV, None)
        return None
    for name in LOADER_ENV:
        value = environ.get(name)
        if value and os.path.normpath(value) != os.path.normpath(snapshot):
            raise CheckpointError('%s=%s would make the model loader read another checkpoint than %s (unset it)' % (name, value, snapshot))
    current = environ.get(CACHE_ENV) or ''
    base = environ.get(CACHE_BASE_ENV) or current
    if environ.get(CACHE_BASE_ENV):
        if current not in (base, keyed_cache_dir(base, sel)):
            raise CheckpointError('%s=%s was moved after %s was set' % (CACHE_ENV, current, CACHE_BASE_ENV))
    elif CACHE_ROOT_SUFFIX + '/' in current + '/':
        raise CheckpointError('%s=%s is already a checkpoint-keyed directory' % (CACHE_ENV, current))
    keyed = keyed_cache_dir(base, sel)
    environ[CACHE_BASE_ENV] = base
    environ[CACHE_ENV] = keyed
    environ[KEY_ENV] = sel.key
    environ[SNAPSHOT_ENV] = snapshot
    return dict(repo=sel.repo, revision=sel.revision, key=sel.key, snapshot=snapshot, weight_cache=keyed, base_cache=base)


def claim_cache(environ, sel, makedirs=os.makedirs):
    """API server only: create the keyed directory and its marker, or confirm the marker already there names this pair. -> dict(created, first_boot_complete)."""
    directory = environ[CACHE_ENV]
    makedirs(directory, exist_ok=True)
    marker = os.path.join(directory, MARKER)
    wanted = dict(schema=1, repo=sel.repo, revision=sel.revision, key=sel.key, weight_map_sha256=sel.entry['weight_map_sha256'], first_boot_complete=False)
    try:
        descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        with open(marker, encoding='utf-8') as handle:
            found = json.load(handle)
        for field in ('repo', 'revision', 'key', 'weight_map_sha256'):
            if found.get(field) != wanted[field]:
                raise CheckpointError('%s names %s=%r, this boot is %r: the directory holds another checkpoint\'s converted weights' % (marker, field, found.get(field), wanted[field]))
        return dict(created=False, first_boot_complete=bool(found.get('first_boot_complete')))
    with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
        json.dump(wanted, handle, indent=1, sort_keys=True)
        handle.write('\n')
    return dict(created=True, first_boot_complete=False)


def mark_first_boot_complete(environ):
    """The worker, after the fast path attached: the converted weights of this boot are whole. -> True when the marker was updated. A no-op without a selection."""
    if not environ.get(KEY_ENV):
        return False
    marker = os.path.join(environ[CACHE_ENV], MARKER)
    try:
        with open(marker, encoding='utf-8') as handle:
            found = json.load(handle)
        if found.get('first_boot_complete') is True:
            return False
        found['first_boot_complete'] = True
        temporary = marker + '.tmp.%d' % os.getpid()
        with open(temporary, 'w', encoding='utf-8') as handle:
            json.dump(found, handle, indent=1, sort_keys=True)
            handle.write('\n')
        os.replace(temporary, marker)
        return True
    except (OSError, ValueError):
        return False


def startup_problems(args, environ):
    """[problem] the worker checks once the model exists: the model args name the selected snapshot and a weight cache under the keyed directory. [] without a selection."""
    key = environ.get(KEY_ENV)
    if not key:
        return []
    snapshot, cache = environ.get(SNAPSHOT_ENV), environ.get(CACHE_ENV)
    problems = []
    loaded = getattr(args, 'checkpoint_dir', None) or getattr(args, 'CKPT_DIR', None)
    if not loaded:
        problems.append('the model args hold no checkpoint directory (checkpoint_dir, CKPT_DIR): the loaded weights cannot be tied to %s' % snapshot)
    elif os.path.realpath(str(loaded)) != os.path.realpath(str(snapshot)) and os.path.normpath(str(loaded)) != os.path.normpath(str(snapshot)):
        problems.append('the model loaded %s but the selected snapshot is %s' % (loaded, snapshot))
    try:
        where = str(args.weight_cache_path())
    except Exception as error:   # noqa: BLE001 - any failure to name the cache fails the check closed
        problems.append('the model args cannot name their weight cache (%s)' % type(error).__name__)
        where = None
    if where is not None and not within(where, cache):
        problems.append('the model reads its converted weights from %s, outside the keyed directory %s' % (where, cache))
    return problems


def log_text(facts, verified=None):
    return '%s %s@%s key=%s snapshot=%s weight_cache=%s base_cache=%s (the base cache is not read or written)%s' % (
        LINE, facts['repo'], facts['revision'], facts['key'], facts['snapshot'], facts['weight_cache'], facts['base_cache'],
        '' if verified is None else ' identity=base files=%d shards=%d' % verified)


def container_log_problems(text, expected):
    """[problem] a container log against what the job asked: `expected` is None (no checkpoint asked: no checkpoint line may appear) or (repo, revision)."""
    lines = [line for line in text.splitlines() if '[QWEN-C2] ' + LINE in line]
    if expected is None:
        return ['the job asked for no other checkpoint but the log shows a checkpoint line: %s' % lines[0][-160:]] if lines else []
    repo, revision = expected
    head = '[QWEN-C2] %s %s@%s ' % (LINE, repo, revision)
    problems = []
    named = [line for line in lines if head in line]
    bound = [line for line in lines if '[QWEN-C2] %s ' % BOUND_LINE in line]
    key = cache_key(repo, revision)
    if not named:
        problems.append('the log never names the asked checkpoint (%s@%s): the selection did not cross into the container' % (repo, revision[:12]))
    elif not any(('key=%s ' % key) in line and 'weight_cache=' in line and (CACHE_ROOT_SUFFIX + '/' + key) in line for line in named):
        problems.append('the checkpoint line does not show the keyed weight cache directory (...%s/%s)' % (CACHE_ROOT_SUFFIX, key))
    if len(set(named)) > 1:
        problems.append('the log names the asked checkpoint with different values: %s' % ' | '.join(sorted(set(named))[:2]))
    if not bound:
        problems.append('the worker never logged "%s": the loaded weights were not tied to the snapshot and the keyed cache' % BOUND_LINE)
    elif not any(key in line for line in bound):
        problems.append('the worker bound another checkpoint than the asked one')
    others = [line for line in lines if line not in named and line not in bound]
    if others:
        problems.append('the log carries a checkpoint line for another selection: %s' % others[0][-160:])
    return problems


def record(snapshot):
    """The identity digests of a snapshot directory, as a catalog entry's fields (a new checkpoint's entry is made from this and the hub's shard sizes and sha256)."""
    catalog = load_catalog()
    names = catalog['identity_files']
    document = json.load(open(os.path.join(snapshot, 'model.safetensors.index.json'), encoding='utf-8'))
    return dict(files=dict((name, sha256_file(os.path.join(snapshot, name))) for name in names),
                weight_map_sha256=weight_map_digest(document['weight_map']), tensors=len(document['weight_map']),
                total_size=int(document['metadata']['total_size']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    commands = parser.add_subparsers(dest='command')
    check = commands.add_parser('check', help='verify a snapshot in a hub directory the way the API server will')
    check.add_argument('--hub', required=True)
    check.add_argument('--repo', required=True)
    check.add_argument('--revision', required=True)
    check.add_argument('--hash-shards', action='store_true', help='also hash every weight shard against its pinned sha256 (reads the whole checkpoint)')
    keyed = commands.add_parser('cache-dir', help='print the converted-weight directory of a checkpoint')
    keyed.add_argument('--base', required=True, help='the base TT_CACHE_PATH')
    keyed.add_argument('--repo', required=True)
    keyed.add_argument('--revision', required=True)
    recorded = commands.add_parser('record', help='print the identity digests of a snapshot directory')
    recorded.add_argument('--snapshot', required=True)
    options = parser.parse_args(argv)
    try:
        if options.command == 'record':
            print(json.dumps(record(options.snapshot), indent=1, sort_keys=True))
            return 0
        if options.command in ('check', 'cache-dir'):
            catalog = load_catalog()
            sel = selection({REPO_ENV: options.repo, REVISION_ENV: options.revision}, catalog)
            if options.command == 'cache-dir':
                print(keyed_cache_dir(options.base, sel))
                return 0
            snapshot = os.path.join(options.hub, hub_dirname(sel.repo), 'snapshots', sel.revision)
            problems = verify_snapshot(snapshot, sel, catalog, hash_shards=options.hash_shards)
            for problem in problems:
                print('CHECKPOINT FAILED: %s' % problem)
            print('CHECKPOINT %s@%s %s' % (sel.repo, sel.revision, 'REFUSED' if problems else 'ok (identity files equal the base\'s, shards present with their pinned sizes%s)' % (
                ', every shard hashed' if options.hash_shards else '')))
            return 1 if problems else 0
    except (CheckpointError, OSError, ValueError, KeyError) as error:
        print('CHECKPOINT refused: %s' % error, file=sys.stderr)
        return 1
    parser.print_usage(sys.stderr)
    return 2


if __name__ == '__main__':
    sys.exit(main())
