"""Reuse identical compiled factories without conflating admission and code generation."""

import json
from pathlib import Path

from dspark_runtime_cache import cache_key, inspect_entry, store_entry


CODEGEN_BUILDERS = ('dspark_splitk_fp32_factory.py', 'dspark_fp32_build.py', 'sdpa_graft_build.py')
SEED_KEY = '99347328e8af7cbe515775c64c0fa7099eba24469287cba169217d6aea329839'
SEED_BINARY = '475ab9189c458a707ee6aee4de7256f470acb5ad65506eefa9b1596c4eab9e49'


def compile_inputs(inputs):
    return dict(image=inputs['image'], backend=inputs['backend'], factory=inputs['factory'],
        registration=inputs['registration'],
        codegen_builders={name: inputs['builders'][name] for name in CODEGEN_BUILDERS},
        recipe=['ninja', '-C', '/opt/tt-metal/build_Release', '-j', '8', 'ttnncpp'])


def find_entry(cache, inputs):
    identity = compile_inputs(inputs)
    manifest = inspect_entry(cache, identity)
    if manifest is not None:
        return identity, manifest
    seed = Path(cache) / SEED_KEY
    if not seed.exists():
        return identity, None
    retained = json.loads((seed / 'manifest.json').read_text())
    if cache_key(retained['inputs']) != SEED_KEY:
        raise ValueError('Pinned compiled-factory cache identity changed')
    retained = inspect_entry(cache, retained['inputs'])
    if retained['binary_sha256'] != SEED_BINARY:
        raise ValueError('Pinned imported hardware binary changed')
    if compile_inputs(retained['inputs']) != identity:
        return identity, None
    return identity, store_entry(cache, identity, seed / '_ttnncpp.so')
