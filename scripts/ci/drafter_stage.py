"""Hash and stage a downloaded DFlash2-layout safetensors file; no checkpoint code runs, stdlib only.

  describe   reads the file's header and every tensor and prints a drafter manifest draft (the per-tensor sha256
             values a candidate is pinned to) for review and commit under references/drafter-manifests/;
  stage      splits the file into the fixture directories the image build copies (the layout and manifest.json
             format of draft_*_fixture.py), after checking every byte against a committed manifest.

Both read a local file, so the 3.85 GB download happens on the host that stages it (the rig), not on a laptop.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct

import drafter_manifest as manifests

SCOPE = 'Staged from a pinned local checkpoint file by drafter_stage.py; no checkpoint code executed'


def read_header(path):
    with open(str(path), 'rb') as handle:
        size = struct.unpack('<Q', handle.read(8))[0]
        raw = handle.read(size)
    if len(raw) != size or size > 1 << 20:
        raise ValueError('Safetensors header is truncated or implausible')
    return size, raw, json.loads(raw)


def checked_geometry(path):
    """(data start, header sha256, {name: (dtype, shape, first, last)}) of a file of the DFlash2 layout."""
    size, raw, header = read_header(path)
    total = os.path.getsize(str(path))
    header = {name: entry for name, entry in header.items() if name != '__metadata__'}
    if set(header) != manifests.expected_names():
        raise ValueError('The file does not hold exactly the 81 DFlash2 tensors')
    geometry = {}
    for name, entry in header.items():
        first, last = entry['data_offsets']
        length = 2
        for dimension in entry['shape']:
            length *= dimension
        if entry['dtype'] != 'BF16' or last - first != length or 8 + size + last > total:
            raise ValueError('Tensor geometry is not whole BF16: %s' % name)
        geometry[name] = ('BF16', list(entry['shape']), first, last)
    return 8 + size, hashlib.sha256(raw).hexdigest(), geometry, total


def tensor_bytes(handle, start, first, last, chunk=1 << 22):
    handle.seek(start + first)
    remaining = last - first
    while remaining:
        block = handle.read(min(chunk, remaining))
        if not block:
            raise ValueError('The file ends inside a tensor')
        remaining -= len(block)
        yield block


def describe(path, name, model, revision, config=None, **extra):
    start, header_sha, geometry, total = checked_geometry(path)
    tensors, first32 = {}, None
    with open(str(path), 'rb') as handle:
        for tensor, (dtype, shape, first, last) in sorted(geometry.items()):
            digest = hashlib.sha256()
            for index, block in enumerate(tensor_bytes(handle, start, first, last)):
                digest.update(block)
                if tensor == 'fc.weight' and index == 0:
                    first32 = hashlib.sha256(block[:32 * shape[1] * 2]).hexdigest()
            tensors[tensor] = dict(shape=shape, dtype=dtype, sha256=digest.hexdigest())
    manifest = dict(name=name, model=model, revision=revision, header_sha256=header_sha,
        checkpoint_bytes=total, weights_sha256=manifests.sha256_file(path),
        config_sha256=manifests.sha256_file(config) if config else None, trained_block_size=None,
        fc_first32_sha256=first32, **extra, tensors=tensors)
    return manifest


def directories(cache, revision):
    """Fixture directory of each component and layer, in the names build-c2-serving-image.sh copies."""
    cache = Path(cache)
    parts = {component: cache / f'dflash2-{component}-{revision}'
             for component in ('attention', 'convolution', 'mlp', 'projection', 'selector')}
    stack = cache / f'dflash2-stack-{revision}'
    return parts, {layer: stack / f'layer-{layer}' for layer in (1, 2, 3, 4)}


def stage(path, name, cache):
    manifest = manifests.load(name)
    start, header_sha, geometry, total = checked_geometry(path)
    if header_sha != manifest['header_sha256'] or total != manifest['checkpoint_bytes']:
        raise ValueError('The file is not the %s layout (header or size)' % name)
    for tensor, pin in manifest['tensors'].items():
        if geometry[tensor][1] != pin['shape']:
            raise ValueError('Shape differs from the manifest: %s' % tensor)
    parts, stack = directories(cache, manifest['revision'])
    plan = {parts['attention']: {manifests.layer_name(0, s): f for s, f in manifests.ATTENTION.items()},
            parts['convolution']: {manifests.layer_name(0, s): f for s, f in manifests.CONVOLUTION.items()},
            parts['mlp']: {manifests.layer_name(0, s): f for s, f in manifests.MLP.items()},
            parts['projection']: dict(manifests.PROJECTION), parts['selector']: dict(manifests.SELECTOR)}
    for layer, directory in stack.items():
        plan[directory] = {manifests.layer_name(layer, s): f for s, f in manifests.LAYER.items()}
    for directory in plan:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError('Empty fixture directory required; existing data is never overwritten: %s' % directory)
    written, created = [], []
    try:
        with open(str(path), 'rb') as handle:
            for directory, files in plan.items():
                if not directory.exists():
                    created.append(directory)
                directory.mkdir(parents=True, exist_ok=True)
                written.append(directory)
                entries = {}
                for tensor, filename in files.items():
                    _, shape, first, last = geometry[tensor]
                    digest = hashlib.sha256()
                    # The tensor lands under its final name only after its hash matches: a mismatch leaves no file a loader
                    # could pick up.
                    partial = directory / (filename + '.partial')
                    with open(str(partial), 'xb') as destination:
                        for block in tensor_bytes(handle, start, first, last):
                            digest.update(block)
                            destination.write(block)
                    if digest.hexdigest() != manifest['tensors'][tensor]['sha256']:
                        raise ValueError('Content differs from the manifest: %s' % tensor)
                    os.replace(str(partial), str(directory / filename))
                    entries[tensor] = dict(file=filename, bytes=last - first, sha256=digest.hexdigest(),
                                           shape=shape, dtype='BF16')
                document = dict(model=manifest['model'], revision=manifest['revision'],
                    header_sha256=header_sha, checkpoint_bytes=total, tensors=entries, scope=SCOPE)
                # manifest.json is last: a directory without it was never completed.
                with open(str(directory / 'manifest.json'), 'x') as destination:
                    json.dump(document, destination, indent=2)
    except BaseException:
        # Nothing this call wrote stays behind, so the same cache can be staged again.
        for directory in written:
            if directory in created:
                shutil.rmtree(str(directory), ignore_errors=True)
            else:
                for leftover in directory.iterdir():
                    leftover.unlink()
        for directory in created:
            if directory.parent.exists() and not any(directory.parent.iterdir()):
                directory.parent.rmdir()
        raise
    return [str(directory) for directory in written]


def main(argv=None):
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    describing = commands.add_parser('describe')
    describing.add_argument('--checkpoint', type=Path, required=True)
    describing.add_argument('--name', required=True)
    describing.add_argument('--model', required=True)
    describing.add_argument('--revision', required=True)
    describing.add_argument('--config', type=Path)
    describing.add_argument('--trained-block-size', type=int)
    describing.add_argument('--license', dest='license_id')
    describing.add_argument('--license-file', type=Path)
    staging = commands.add_parser('stage')
    staging.add_argument('--checkpoint', type=Path, required=True)
    staging.add_argument('--manifest', required=True)
    staging.add_argument('--cache', type=Path, required=True)
    options = parser.parse_args(argv)
    if options.command == 'describe':
        extra = {}
        if options.license_id:
            extra['license'] = options.license_id
        if options.license_file:
            extra['license_file_sha256'] = manifests.sha256_file(options.license_file)
        manifest = describe(options.checkpoint, options.name, options.model, options.revision, options.config, **extra)
        manifest['trained_block_size'] = options.trained_block_size
        print(json.dumps(manifest, indent=1))
    else:
        for directory in stage(options.checkpoint, options.manifest, options.cache):
            print('staged', directory)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
