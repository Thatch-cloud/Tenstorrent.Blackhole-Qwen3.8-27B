"""The drafter checkpoint selector (docs/tp4-combined-window.md, drafter candidates).

The served DFlash drafter is one baked checkpoint (the image's /experiment-dflash-fixture weights and /draft-config). A candidate drafter is a
profile-selectable alternative, baked beside it, named by an id and PINNED: a revision, the sha256 of its config.json and of its fixture manifests, and
the geometry the rest of the stack was built around. Nothing here picks a checkpoint by itself: QWEN_FAST_DRAFTER_CHECKPOINT unset (or empty) means the
default, whose bytes and paths are today's, and the production profile is unchanged.

  - `selected(environ)`: the id the environment asks for (strict: an unknown id is an error, never the default).
  - `paths(id)`: (fixture directory, draft-config directory). The default maps to today's paths; a candidate to /experiment-dflash-fixtures/<id> and
    /draft-configs/<id>, which the image build stages from a pinned candidate list (build-c2-serving-image.sh, C2_DRAFTER_CANDIDATES).
  - `profile_problems(profile)`: what the contract checks at boot: the engine's speculative model and the fast path's fixture directory equal paths(id),
    and the candidate's geometry equals the default's. The geometry is load-bearing: Lever N keeps the final 2,048-token window inside its last step,
    the sticky plan boundary is floor2048(P) - 2048, the proposals are 15 (the scheduler graft's lookahead), a T16 block is 16 rows a user, the five
    target tap layers feed the draft and its hidden size and depth are fixed by the kernels.
  - `attach_check(environ, root)`: hash the baked config.json, the manifests AND every file of the candidate's fixture directory (the weights: one digest over the
    sorted relative paths and file hashes, pinned as weights_sha256; `python drafter_checkpoint.py --digest <fixture dir>` computes it) against the table and log
    `[PINDIAG] drafter checkpoint id=<id> revision=<12> verified=1 dtype=<bf8|bf16>`, or raise (the attach is refused). The DEFAULT carries no pins (its bytes are the
    image's own): it logs `verified=default`, which proves the selector path ran and nothing about the bytes.

The QWEN_FAST_DRAFTER_BF16 arm (draft_mlp_branch) is unchanged and independent: it changes the projection dtype of whatever checkpoint is served.

Stdlib only, py 3.7. The table is scripts/ci/drafter_checkpoints.json: no host, path or private repository name; a private candidate is an opaque id plus pins.
"""

import hashlib
import json
import os
import re
from pathlib import Path

FLAG = 'QWEN_FAST_DRAFTER_CHECKPOINT'
TABLE = Path(__file__).resolve().parent / 'drafter_checkpoints.json'
MARKER = '[PINDIAG] drafter checkpoint'
ID = re.compile(r'^[a-z0-9][a-z0-9.-]{2,63}$')
HEX40 = re.compile(r'^[0-9a-f]{40}$')
HEX64 = re.compile(r'^[0-9a-f]{64}$')
GEOMETRY_KEYS = ('window', 'num_speculative_tokens', 'rows_per_user', 'target_taps', 'hidden', 'layers')
DEFAULT_FIXTURE = '/experiment-dflash-fixture'
DEFAULT_CONFIG = '/draft-config'
CANDIDATE_FIXTURES = '/experiment-dflash-fixtures'
CANDIDATE_CONFIGS = '/draft-configs'
CONFIG_FILE = 'config.json'
# What a candidate's config.json must still say when it says it at all (the kernels and the target taps are built for these).
CONFIG_KEYS = {'hidden_size': 'hidden', 'num_hidden_layers': 'layers'}


def load(path=None):
    """The table, validated: -> dict(default, checkpoints). ValueError on anything malformed."""
    path = Path(path) if path else TABLE
    document = json.loads(path.read_text(encoding='utf-8'))
    problems = table_problems(document)
    if problems:
        raise ValueError('%s: %s' % (path.name, '; '.join(problems)))
    return document


def table_problems(document):
    problems = []
    if not isinstance(document, dict) or not isinstance(document.get('checkpoints'), dict):
        return ['the table needs a "checkpoints" object']
    default = document.get('default')
    checkpoints = document['checkpoints']
    if default not in checkpoints:
        return ['the default %r is not among the checkpoints' % (default,)]
    for name, entry in checkpoints.items():
        if not ID.match(name):
            problems.append('id %r is not a plain lowercase id' % (name,))
        if not isinstance(entry, dict):
            problems.append('%s: not an object' % name)
            continue
        if not HEX40.match(str(entry.get('revision', ''))):
            problems.append('%s: revision must be 40 hex digits' % name)
        if entry.get('dtype') not in ('bf8', 'bf16'):
            problems.append('%s: dtype must be bf8 or bf16' % name)
        geometry = entry.get('geometry')
        if not isinstance(geometry, dict) or sorted(geometry) != sorted(GEOMETRY_KEYS) or not all(type(geometry[key]) is int for key in geometry):
            problems.append('%s: geometry must be integers for exactly %s' % (name, ', '.join(GEOMETRY_KEYS)))
        if name != default:
            if not HEX64.match(str(entry.get('config_sha256', ''))):
                problems.append('%s: a candidate pins its config.json (config_sha256, 64 hex digits)' % name)
            manifests = entry.get('manifests')
            if not isinstance(manifests, dict) or not manifests or not all(HEX64.match(str(value)) and key and not key.startswith('/') and '..' not in key.split('/')
                                                                         for key, value in manifests.items()):
                problems.append('%s: a candidate pins its fixture manifests (relative path -> sha256)' % name)
            if not HEX64.match(str(entry.get('weights_sha256', ''))):
                problems.append('%s: a candidate pins its weights (weights_sha256)' % name)
    if not problems:
        wanted = checkpoints[default]['geometry']
        for name, entry in checkpoints.items():
            if entry['geometry'] != wanted:
                problems.append('%s: geometry %r differs from the default\'s %r (the window, proposals, rows, taps, hidden size and depth are load-bearing)'
                                % (name, entry['geometry'], wanted))
    return problems


def selected(environ=None, document=None):
    """The checkpoint id the environment names; the table's default when the flag is unset or empty. ValueError on an id the table does not hold."""
    environ = os.environ if environ is None else environ
    document = document or load()
    value = str(environ.get(FLAG, '')).strip()
    if not value:
        return document['default']
    if value not in document['checkpoints']:
        raise ValueError('%s=%r is not a pinned drafter checkpoint (%s)' % (FLAG, value, ', '.join(sorted(document['checkpoints']))))
    return value


def paths(name, document=None):
    """(fixture directory, draft-config directory) of a checkpoint id."""
    document = document or load()
    if name not in document['checkpoints']:
        raise ValueError('unknown drafter checkpoint %r' % (name,))
    if name == document['default']:
        return DEFAULT_FIXTURE, DEFAULT_CONFIG
    return '%s/%s' % (CANDIDATE_FIXTURES, name), '%s/%s' % (CANDIDATE_CONFIGS, name)


def profile_problems(profile, document=None):
    """Every way the profile's drafter selection breaks what the stack needs, [] when none. A profile that names no checkpoint must point at today's
    paths (the default's); one that names a candidate must point both the speculative model and the fast path's fixtures at the candidate's baked copy."""
    document = document or load()
    env = {key: str(value) for key, value in (profile.get('env') or {}).items()}
    try:
        name = selected(env, document)
    except ValueError as error:
        return [str(error)]
    engine = profile.get('engine') or {}
    runtime = ((engine.get('additional-config') or {}).get('qwen_fast_runtime')) or {}
    speculative = engine.get('speculative-config') or {}
    if not speculative and not runtime.get('fixtures'):
        return [] if FLAG not in env else ['%s needs a profile with the DFlash drafter (speculative-config and qwen_fast_runtime.fixtures)' % FLAG]
    fixture, config = paths(name, document)
    problems = []
    if runtime.get('fixtures') != fixture:
        problems.append('drafter checkpoint %s: qwen_fast_runtime.fixtures is %r, not %s' % (name, runtime.get('fixtures'), fixture))
    if speculative.get('model') != config:
        problems.append('drafter checkpoint %s: speculative-config.model is %r, not %s' % (name, speculative.get('model'), config))
    default = document['default']
    if document['checkpoints'][name]['geometry'] != document['checkpoints'][default]['geometry']:
        problems.append('drafter checkpoint %s: its geometry differs from the default\'s' % name)
    return problems


def sha256_of(path):
    digest = hashlib.sha256()
    with open(str(path), 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def combine(pairs):
    """The weights digest of (relative path, file sha256) pairs: sha256 over path, a NUL byte, the file hash and a newline, in sorted path order."""
    digest = hashlib.sha256()
    for relative, file_hash in sorted(pairs):
        digest.update(('%s' % relative + chr(0) + '%s' % file_hash + chr(10)).encode('utf-8'))
    return digest.hexdigest()


def weights_digest(directory):
    """The digest of every file under a fixture directory (the manifests and the weight files they list; a file added or removed changes it too)."""
    base = Path(directory)
    return combine((path.relative_to(base).as_posix(), sha256_of(path)) for path in sorted(base.rglob('*')) if path.is_file())


def attach_problems(name, document=None, root='/'):
    """[problem] for the baked bytes of checkpoint `name` (under `root`, so a test can lay them in a temporary directory). The default has no pins to
    check (its bytes are the image's own): only a candidate is hashed."""
    document = document or load()
    entry = document['checkpoints'][name]
    if name == document['default']:
        return []
    fixture, config = paths(name, document)
    base = Path(root)
    fixture_dir, config_dir = base / fixture.lstrip('/'), base / config.lstrip('/')
    problems = []
    config_file = config_dir / CONFIG_FILE
    if not config_file.is_file():
        problems.append('%s is missing' % config_file)
    else:
        if sha256_of(config_file) != entry['config_sha256']:
            problems.append('%s does not hash to the pinned config_sha256' % config_file)
        else:
            body = json.loads(config_file.read_text(encoding='utf-8'))
            for key, geometry_key in CONFIG_KEYS.items():
                if key in body and body[key] != entry['geometry'][geometry_key]:
                    problems.append('config.json %s=%r, the pinned geometry says %r' % (key, body[key], entry['geometry'][geometry_key]))
    for relative, want in sorted(entry['manifests'].items()):
        file = fixture_dir / relative
        if not file.is_file():
            problems.append('%s is missing' % file)
        elif sha256_of(file) != want:
            problems.append('%s does not hash to its pinned manifest sha256' % file)
    if not fixture_dir.is_dir():
        problems.append('%s is missing' % fixture_dir)
    elif weights_digest(fixture_dir) != entry['weights_sha256']:
        problems.append('the files under %s (the manifests and the weights they list) do not hash to the pinned weights_sha256' % fixture_dir)
    return problems


def attach_check(environ=None, document=None, root='/', log=print):
    """Verify the selected candidate's baked bytes and log the marker; raise ValueError (the attach is refused) on any mismatch. Returns the marker line, or None
    when the flag is unset (the production log is unchanged)."""
    environ = os.environ if environ is None else environ
    if not str(environ.get(FLAG, '')).strip():
        return None
    document = document or load()
    name = selected(environ, document)
    problems = attach_problems(name, document, root)
    if problems:
        raise ValueError('drafter checkpoint %s is not the pinned one: %s' % (name, '; '.join(problems)))
    entry = document['checkpoints'][name]
    verified = 'default' if name == document['default'] else '1'
    line = '%s id=%s revision=%s verified=%s dtype=%s' % (MARKER, name, entry['revision'][:12], verified, entry['dtype'])
    log(line)
    return line


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == '--digest':
        print(weights_digest(sys.argv[2]))
    else:
        print('usage: drafter_checkpoint.py --digest <fixture directory>: the weights_sha256 to pin for a candidate', file=sys.stderr)
        sys.exit(2)
