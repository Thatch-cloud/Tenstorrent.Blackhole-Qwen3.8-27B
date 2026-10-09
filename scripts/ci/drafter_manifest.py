"""Which DFlash2 drafter checkpoint a build or a gate serves, and every byte pin that goes with it.

The served drafter is a weights-only choice: every candidate that shares the DFlash2 layout (five sliding-window
layers, the same 81 BF16 tensors, byte-identical safetensors header) loads through the same code. What differs is the
identity the loaders pin: the model repository and revision, the per-tensor sha256 values and the one fc slice hash.
Those now live in manifests under references/drafter-manifests/<name>.json instead of in the loaders.

Selection, in order: the QWEN_DRAFTER_MANIFEST variable, then the DRAFTER_MANIFEST marker file the image build lays
beside the fixtures, then the default. The default manifest is today's served drafter and its file is pinned here by
sha256, so a changed default fails at load; the default loaders stay the unedited ones (drafter_fixtures.load), so a
default build serves the bytes it always did.

Stdlib only: the rig stages candidates with it (drafter_stage.py), the CPU suite and the image import it.
"""

import hashlib
import json
import os
from pathlib import Path
import re

DEFAULT = 'dedf8df6'
ENVIRONMENT = 'QWEN_DRAFTER_MANIFEST'
MARKER = 'DRAFTER_MANIFEST'
DIRECTORY = Path(__file__).resolve().with_name('references') / 'drafter-manifests'
# The default manifest's canonical content hash: the identity the production image carries today (test_drafter_manifest holds every
# value in it equal to the unedited loaders' tables).
DEFAULT_SHA256 = 'fd1c50b9d6dca5c1a2defaa6ac489ca46df39a77daab20faebc6ae3ebc94a54c'
HEADER_SHA256 = '0c2c70601b30f8d1ca7d5794b817779ba2dcf1956cfc7d4f83e87091e1ab7c8c'
CHECKPOINT_BYTES = 3848817896
NAME = re.compile(r'[a-z0-9][a-z0-9._-]{0,63}')
HEX64 = re.compile(r'[0-9a-f]{64}')
HEX40 = re.compile(r'[0-9a-f]{40}')

# Layer-zero tensor suffix -> fixture file name; layers 1-4 reuse the names inside layer-N/.
ATTENTION = {
    'self_attn.k_norm.weight': 'k-norm.bf16', 'self_attn.k_proj.weight': 'k-projection.bf16',
    'self_attn.o_proj.weight': 'o-projection.bf16', 'self_attn.q_norm.weight': 'q-norm.bf16',
    'self_attn.q_proj.weight': 'q-projection.bf16', 'self_attn.v_proj.weight': 'v-projection.bf16',
}
CONVOLUTION = {
    'attention_conv.base_kernel': 'attention-base.bf16',
    'attention_conv.kernel_projection.weight': 'attention-projection.bf16',
    'input_layernorm.weight': 'attention-norm.bf16', 'mlp_conv.base_kernel': 'mlp-base.bf16',
    'mlp_conv.kernel_projection.weight': 'mlp-projection.bf16',
    'post_attention_layernorm.weight': 'mlp-norm.bf16',
}
MLP = {
    'mlp.down_proj.weight': 'down-projection.bf16', 'mlp.gate_proj.weight': 'gate-projection.bf16',
    'mlp.up_proj.weight': 'up-projection.bf16',
}
PROJECTION = {'fc.weight': 'fc.bf16', 'hidden_norm.weight': 'hidden_norm.bf16'}
SELECTOR = {
    'candidate_selector.hidden_projection.weight': 'hidden-projection.bf16',
    'candidate_selector.predecessor_codebook': 'predecessor.bf16',
    'candidate_selector.successor_codebook': 'successor.bf16', 'norm.weight': 'norm.bf16',
}
LAYER = ATTENTION | CONVOLUTION | MLP
LAYERS = (0, 1, 2, 3, 4)


def layer_name(layer, suffix):
    return f'layers.{layer}.{suffix}'


def expected_names():
    """Every tensor name a DFlash2 checkpoint of this layout holds (81)."""
    names = [layer_name(layer, suffix) for layer in LAYERS for suffix in LAYER]
    return set(names) | set(PROJECTION) | set(SELECTOR)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(manifest):
    # Over the parsed content, so a checkout's line endings cannot move it.
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def names():
    return sorted(path.stem for path in DIRECTORY.glob('*.json'))


def path_of(name):
    if not isinstance(name, str) or NAME.fullmatch(name) is None:
        raise ValueError('A plain drafter manifest name is required')
    return DIRECTORY / (name + '.json')


def validate(manifest, name):
    """Refuses a manifest whose shape is not a DFlash2-layout checkpoint pinned to one revision."""
    required = ('name', 'model', 'revision', 'header_sha256', 'checkpoint_bytes', 'config_sha256',
                'fc_first32_sha256', 'trained_block_size', 'tensors')
    if any(key not in manifest for key in required) or manifest['name'] != name:
        raise ValueError('Drafter manifest %s is incomplete or names another manifest' % name)
    if (HEX40.fullmatch(str(manifest['revision'])) is None or not isinstance(manifest['model'], str)
            or manifest['model'].count('/') != 1):
        raise ValueError('Drafter manifest %s needs a full 40-character revision of an owner/name repository' % name)
    if (manifest['header_sha256'] != HEADER_SHA256 or manifest['checkpoint_bytes'] != CHECKPOINT_BYTES):
        raise ValueError('Drafter manifest %s: only the byte-identical DFlash2 layout loads (header and size)' % name)
    for key in ('config_sha256', 'fc_first32_sha256'):
        if HEX64.fullmatch(str(manifest[key])) is None:
            raise ValueError('Drafter manifest %s: %s is not a sha256' % (name, key))
    tensors = manifest['tensors']
    if set(tensors) != expected_names():
        raise ValueError('Drafter manifest %s must pin exactly the 81 DFlash2 tensors' % name)
    for tensor, entry in tensors.items():
        if (set(entry) != {'shape', 'dtype', 'sha256'} or entry['dtype'] != 'BF16'
                or HEX64.fullmatch(str(entry['sha256'])) is None
                or not entry['shape'] or any(type(value) is not int or value < 1 for value in entry['shape'])):
            raise ValueError('Drafter manifest %s: malformed pin for %s' % (name, tensor))
    return manifest


def load(name):
    path = path_of(name)
    manifest = json.loads(path.read_bytes().decode('utf-8'))
    if name == DEFAULT and canonical_sha256(manifest) != DEFAULT_SHA256:
        raise ValueError('The default drafter manifest differs from the production pins')
    return validate(manifest, name)


def select(root=None, environ=None):
    """The manifest name in force: the environment's, else the fixture root's marker, else the default."""
    environ = os.environ if environ is None else environ
    named = environ.get(ENVIRONMENT) or None
    marked = None
    if root is not None:
        marker = Path(root) / MARKER
        if marker.is_file():
            marked = marker.read_text(encoding='utf-8').strip() or None
    if named is not None and marked is not None and named != marked:
        raise ValueError('%s=%s but the fixtures were staged for %s' % (ENVIRONMENT, named, marked))
    name = named or marked or DEFAULT
    path_of(name)
    return name


def tensor_hashes(manifest, names_):
    return {name: manifest['tensors'][name]['sha256'] for name in names_}
