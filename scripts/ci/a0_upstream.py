"""Pinned intake of the upstream code and weights the A0 screen runs: nothing is compiled, imported or executed before its sha256
and size have been checked against the pin.

    z-lab model.py            DFlash2DraftModel (the control)       pin: a pins file the lab keeps (not in this repository)
    DSpark dspark.py / dflash.py                                      pin: dspark_intake.FILES (reviewed, in this repository; the
                                                                      caller passes them, so this module stays free of repository imports)
    checkpoints                                                       pin: sha256 of the safetensors file(s)

`load_pinned_module(path, sha256, ...)` reads a bounded file, refuses a digest or size that differs, and only then executes it as a
module (the upstream files import transformers, so they have to run as modules: this is the pattern dspark_upstream_backbone uses
for the CPU reference, here for the whole reviewed file). The pins file is JSON {name: {"bytes": n, "sha256": hex}}; a name the
caller did not ask for, or a missing one, is refused.
"""
import hashlib
import json
import os
import types

MAX_SOURCE_BYTES = 1000000
BLOCK = 1 << 20


class IntakeError(ValueError):
    """Refused: the message names a file, never its contents."""


def sha256_file(path, block=BLOCK):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(block), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_pinned(path, expected_sha256, expected_bytes=None, max_bytes=MAX_SOURCE_BYTES):
    """The bytes of a source file whose size is bounded and whose sha256 equals the pin."""
    size = os.path.getsize(path)
    if size > max_bytes:
        raise IntakeError('%s is larger than the %d byte bound' % (os.path.basename(path), max_bytes))
    if expected_bytes is not None and size != expected_bytes:
        raise IntakeError('%s does not have the pinned size' % os.path.basename(path))
    with open(path, 'rb') as handle:
        data = handle.read(max_bytes + 1)
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise IntakeError('%s does not match its sha256 pin' % os.path.basename(path))
    return data


def load_pinned_module(path, expected_sha256, name, expected_bytes=None, namespace=None, max_bytes=MAX_SOURCE_BYTES):
    """Verify, then execute `path` as a module called `name`. `namespace`: names the module may start with."""
    data = read_pinned(path, expected_sha256, expected_bytes, max_bytes)
    module = types.ModuleType(name)
    module.__file__ = path
    if namespace:
        module.__dict__.update(namespace)
    exec(compile(data, path, 'exec'), module.__dict__)
    return module


def load_pins(path, wanted):
    """{name: (bytes, sha256)} from a pins file; exactly the `wanted` names must be present."""
    with open(path, encoding='utf-8') as handle:
        document = json.load(handle)
    if not isinstance(document, dict) or set(document) != set(wanted):
        raise IntakeError('the pins file must hold exactly: %s' % ', '.join(sorted(wanted)))
    out = {}
    for name in wanted:
        entry = document[name]
        if not isinstance(entry, dict) or not isinstance(entry.get('bytes'), int) or not isinstance(entry.get('sha256'), str) \
                or len(entry['sha256']) != 64:
            raise IntakeError('the pin of %s needs bytes and a 64 character sha256' % name)
        out[name] = (entry['bytes'], entry['sha256'])
    return out


def verify_checkpoint(directory, expected):
    """`expected`: {file name: sha256}. Every named file must be in `directory` with that digest."""
    for name, digest in sorted(expected.items()):
        target = os.path.join(directory, name)
        if not os.path.isfile(target) or sha256_file(target) != digest:
            raise IntakeError('checkpoint file %s is missing or does not match its pin' % name)
    return True


def draft_scalars(config):
    """{name: value} of the scalars upstream applies around a draft model that the A0 port and walkers do NOT: the input embedding scale
    (dflash_generate), the output multiplier and the final-logit softcap (compute_logits), and the tapped target layers. `config` is
    the checkpoint's config.json as a dict; z-lab's `_draft_value` reads `dflash_config` first, then the top level."""
    nested = config.get('dflash_config') if isinstance(config.get('dflash_config'), dict) else {}

    def value(name, default):
        return nested.get(name, config.get(name, default))
    return dict(input_embedding_scale=value('input_embedding_scale', 1.0), output_multiplier=value('output_multiplier', 1.0),
                final_logit_softcapping=value('final_logit_softcapping', None), target_layer_ids=value('target_layer_ids', None))


def check_draft_config(config, taps, ids=None, allow_unnamed=False):
    """{check name: bool}: both scalars neutral, no softcap, and the draft's tapped target layers equal to the layers the harness hooks.
    `ids`: the layers the loaded model itself resolved (z-lab computes them when the config names none); else the config's own.
    A checkpoint that names none is refused unless `allow_unnamed` (DSpark: its taps are pinned in the reviewed intake module)."""
    scalars = draft_scalars(config)
    ids = scalars['target_layer_ids'] if ids is None else ids
    softcap = scalars['final_logit_softcapping']
    if ids is None:
        taps_match = bool(allow_unnamed)
    else:
        taps_match = [int(i) for i in ids] == [int(i) for i in taps]
    return dict(embedding_scale_neutral=float(scalars['input_embedding_scale']) == 1.0,
                output_multiplier_neutral=float(scalars['output_multiplier']) == 1.0,
                no_softcap=softcap is None or float(softcap) <= 0,
                taps_match=taps_match)
