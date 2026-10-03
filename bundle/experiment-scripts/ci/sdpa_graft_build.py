"""Read-only audit for restoring existing transformer grafts to an isolated build."""

import hashlib
import json
from pathlib import Path


REGISTRATION_SHA256 = 'a2fb5b6ea5769c57f79353f2a8c3859f85d161f3a94ced173862d10cd5adca10'
COMPONENTS = ('attn_prep', 'gdn_conv_gates', 'gdn_norm_gate', 'gdn_decay', 'decode_gated_delta_rule')


def implementation_sources():
    return tuple(path for component in COMPONENTS for path in (
        f'{component}/{component}.cpp',
        f'{component}/device/{component}_device_operation.cpp',
        f'{component}/device/{component}_program_factory.cpp'))


def audit(root):
    transformer = Path(root) / 'ttnn/cpp/ttnn/operations/transformer'
    source = transformer / 'sources.cmake'
    if hashlib.sha256(source.read_bytes()).hexdigest() != REGISTRATION_SHA256:
        raise ValueError('Transformer source registrations differ from audited hardware34067681095')
    sources = {name: hashlib.sha256((transformer / name).read_bytes()).hexdigest() for name in implementation_sources()}
    return dict(registration_sha256=REGISTRATION_SHA256, implementation_sources=sources,
        scope='Restore existing on-disk implementations to CMake; no kernel or binding edits')


if __name__ == '__main__':
    report = audit('/opt/tt-metal')
    Path('/experiment/results/sdpa-graft-build.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
