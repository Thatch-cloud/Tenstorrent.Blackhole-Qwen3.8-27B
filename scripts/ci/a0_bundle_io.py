"""Reading and verifying an A0 bundle: the light half of a0_bundle, with no import of the tau lab, so the GPU host's image needs only the
standard library for it. Python 3.7, stdlib."""
import gzip
import hashlib
import json
import os

FORMAT = 'a0-bundle-1'
BUNDLE_NAME = 'bundle.jsonl.gz'
META_NAME = 'meta.jsonl'
MANIFEST_NAME = 'MANIFEST.json'


class BundleError(ValueError):
    """Refused: the message names a file or a count, never a value of the data."""


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_bundle(directory, expect_turns=None):
    """The manifest of a bundle whose files match their recorded size and sha256, else BundleError."""
    path = os.path.join(directory, MANIFEST_NAME)
    if not os.path.isfile(path):
        raise BundleError('%s is missing' % MANIFEST_NAME)
    with open(path, encoding='utf-8') as handle:
        manifest = json.load(handle)
    if manifest.get('format') != FORMAT or not isinstance(manifest.get('files'), dict):
        raise BundleError('not an %s manifest' % FORMAT)
    for name in (BUNDLE_NAME, META_NAME):
        entry = manifest['files'].get(name)
        target = os.path.join(directory, name)
        if not entry or not os.path.isfile(target) or os.path.getsize(target) != entry['bytes'] \
                or sha256_file(target) != entry['sha256']:
            raise BundleError('%s does not match the manifest' % name)
    if expect_turns is not None and manifest['counts'].get('turns') != expect_turns:
        raise BundleError('the bundle holds %s turns, %d expected' % (manifest['counts'].get('turns'), expect_turns))
    return manifest


def read_bundle(directory):
    """The turn records of a verified bundle, one at a time."""
    verify_bundle(directory)
    with gzip.open(os.path.join(directory, BUNDLE_NAME), 'rt', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
