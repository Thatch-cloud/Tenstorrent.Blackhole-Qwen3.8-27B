"""Loads the staged DFlash2 fixtures of the manifest in force, with that manifest's pins.

The default drafter goes through the unedited loaders (full_dflash_request.load_dflash_fixtures, handed in), so a
default build serves exactly the bytes and runs exactly the checks it did before drafter manifests existed: those
loaders, their pin tables and the frozen evidence that names them are not touched. A candidate manifest takes the
parallel path here, which repeats the same checks (manifest identity, per-file size, shape, dtype and sha256, a
finite-value check, the fc first-32-row slice) against the candidate's pins and returns the same four values.

Any mismatch raises: a candidate image whose fixtures are another revision's refuses at attach.
"""

import hashlib
import json
from pathlib import Path

import drafter_manifest as manifests


def load(root, default_loader, environ=None):
    """The (manifests, layers, projection, selector) tuple of the fixtures under root."""
    name = manifests.select(root, environ)
    if name == manifests.DEFAULT:
        return default_loader(root)
    manifest = manifests.load(name)
    found = load_candidate(root, manifest)
    # The marker the tau lab's launched-argv check and the smoke read; only a candidate prints it (the default's logs are as before).
    print('[DRAFTER_MANIFEST] %s in force: %s at %s, %d tensors verified' % (
        name, manifest['model'], manifest['revision'][:12], len(manifest['tensors'])), flush=True)
    return found


def _read_manifest(output, manifest):
    found = json.loads((Path(output) / 'manifest.json').read_text())
    if (found.get('model') != manifest['model'] or found.get('revision') != manifest['revision']
            or found.get('header_sha256') != manifest['header_sha256']
            or found.get('checkpoint_bytes') != manifest['checkpoint_bytes']):
        raise ValueError('Pinned %s manifest required' % manifest['name'])
    return found


def _verified_bytes(output, files, manifest):
    """files: tensor name -> file name. Returns (the directory's manifest, name -> bytes), every byte hashed."""
    output = Path(output)
    found = _read_manifest(output, manifest)
    data = {}
    for name, filename in files.items():
        pin = manifest['tensors'][name]
        entry = found['tensors'][name]
        length = 2
        for dimension in pin['shape']:
            length *= dimension
        if (entry['file'] != filename or entry['shape'] != pin['shape'] or entry['dtype'] != 'BF16'
                or entry['bytes'] != length or entry['sha256'] != pin['sha256']
                or (output / filename).stat().st_size != length):
            raise ValueError('Audited %s metadata required: %s' % (manifest['name'], name))
        payload = (output / filename).read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != pin['sha256']:
            raise ValueError('Audited %s content required: %s at %s; expected %s, read %s'
                             % (manifest['name'], name, output / filename, pin['sha256'], actual))
        data[name] = payload
    return found, data


def _tensors(output, files, manifest):
    from draft_convolution_fixture import verified_tensor

    found, data = _verified_bytes(output, files, manifest)
    return found, {name: verified_tensor(data[name], manifest['tensors'][name]['shape'],
                                         manifest['tensors'][name]['sha256'], name) for name in files}


def _layer_files(layer, suffixes):
    return {manifests.layer_name(layer, suffix): filename for suffix, filename in suffixes.items()}


def _projection(output, manifest):
    import torch

    output = Path(output)
    found = _read_manifest(output, manifest)
    tensors = {}
    for name, filename in manifests.PROJECTION.items():
        pin = manifest['tensors'][name]
        entry = found['tensors'][name]
        elements = 1
        for dimension in pin['shape']:
            elements *= dimension
        payload = (output / filename).read_bytes()
        if (entry['file'] != filename or entry['shape'] != pin['shape'] or entry['dtype'] != 'BF16'
                or entry['bytes'] != 2 * elements or len(payload) != 2 * elements
                or entry['sha256'] != pin['sha256'] or hashlib.sha256(payload).hexdigest() != pin['sha256']):
            raise ValueError('Complete hashed BF16 projection tensors required')
        if name == 'fc.weight' and (hashlib.sha256(payload[:32 * pin['shape'][1] * 2]).hexdigest()
                                    != manifest['fc_first32_sha256']):
            raise ValueError('Full projection differs from audited first32 outputs')
        tensors[name] = torch.frombuffer(bytearray(payload), dtype=torch.bfloat16).reshape(pin['shape'])
        if not torch.isfinite(tensors[name]).all():
            raise ValueError('Finite learned tensors required')
    return found, tensors['fc.weight'], tensors['hidden_norm.weight']


def load_candidate(root, manifest):
    root = Path(root)
    attention_manifest, attention = _tensors(root / 'attention', _layer_files(0, manifests.ATTENTION), manifest)
    convolution_manifest, convolution = _tensors(root / 'convolution', _layer_files(0, manifests.CONVOLUTION), manifest)
    mlp_manifest, mlp = _tensors(root / 'mlp', _layer_files(0, manifests.MLP), manifest)
    projection_manifest, projection, norm = _projection(root / 'projection', manifest)
    selector_manifest, selector = _tensors(root / 'selector', dict(manifests.SELECTOR), manifest)
    found = dict(attention=attention_manifest, convolution=convolution_manifest, mlp=mlp_manifest,
                 projection=projection_manifest, selector=selector_manifest, layers=[])
    layers = [(attention, convolution, mlp)]
    for layer in range(1, 5):
        layer_manifest, weights = _tensors(root / f'layer-{layer}', _layer_files(layer, manifests.LAYER), manifest)
        normalized = {name.replace(f'layers.{layer}.', 'layers.0.', 1): value for name, value in weights.items()}
        layers.append((normalized, normalized, normalized))
        found['layers'].append(layer_manifest)
    return found, layers, {'fc.weight': projection, 'hidden_norm.weight': norm}, selector
