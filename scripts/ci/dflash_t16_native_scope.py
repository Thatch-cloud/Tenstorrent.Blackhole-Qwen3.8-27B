"""Explicit source-bound admission for the approximate T16 proposal candidate."""

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
from pathlib import Path

from dflash_t16_native_attention_gate import SOURCES, NATIVE_SOURCES, ORIGINAL, hashes, native_hashes, qualify
from dflash_combined_sim_runtime import binary_hashes, BINARIES, BINARY_SHA256


REPORTS = {
    31: 'c67046eae2132c5cc3748a2b4ea1b91e06e3db48f71c93de8594f439bbde948f',
    2048: '06caf0d418bb94cdc592b49a0c0dd505f900b7c36b67f0f94c34ab92d2f8ce1f',
}
_ACTIVE = ContextVar('dflash_t16_native_admission', default=None)


def require_active():
    admission = _ACTIVE.get()
    if admission is None:
        raise ValueError('T16 native proposals require their own source-bound simulator admission')
    return admission


def validate_record(admission):
    if (not isinstance(admission, dict)
            or admission.get('reports') != {str(context): digest for context, digest in REPORTS.items()}
            or admission.get('sources') != hashes(Path(__file__).parent, SOURCES)
            or admission.get('approximate_proposals') is not True
            or admission.get('target_attention_changed') is not False
            or admission.get('runtime_binaries') != dict.fromkeys(BINARIES, BINARY_SHA256)
            or not isinstance(admission.get('native_sources'), dict)
            or set(admission['native_sources']) != set(NATIVE_SOURCES)
            or any(admission['native_sources'][name] != digest for name, digest in ORIGINAL.items())):
        raise ValueError('Source-bound T16 proposal admission must accompany the measured request')
    if any(not isinstance(digest, str) or len(digest) != 64
            or any(character not in '0123456789abcdef' for character in digest)
            for digest in admission['native_sources'].values()):
        raise ValueError('Complete native source fingerprints required')


def admit(evidence, sources, runtime):
    current_sources = hashes(sources, SOURCES)
    current_native = native_hashes(runtime)
    current_binaries = binary_hashes(runtime)
    for context, expected in REPORTS.items():
        data = (Path(evidence) / f'dflash-t16-{context}.json').read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Pinned T16 simulator report mismatch')
        report = json.loads(data)
        from dflash_t16_native_attention_gate import PACKER, SIMULATOR_PACKER

        expected_native = {**current_native, PACKER: SIMULATOR_PACKER}
        mismatches = [name for name in NATIVE_SOURCES
            if report.get('native_sources', {}).get(name) != expected_native[name]
            or report.get('native_sources_after', {}).get(name) != expected_native[name]]
        if mismatches:
            raise ValueError('Native T16 simulator/runtime source mismatch: ' + ', '.join(mismatches))
        qualify(report, context, current_sources, current_native)
    return dict(reports={str(context): digest for context, digest in REPORTS.items()},
        sources=current_sources, native_sources=current_native,
        approximate_proposals=True, target_attention_changed=False, runtime_binaries=current_binaries)


@contextmanager
def scoped_native_t16(evidence, sources, runtime):
    if _ACTIVE.get() is not None:
        raise ValueError('Nested T16 proposal admission is not supported')
    admission = admit(evidence, sources, runtime)
    token = _ACTIVE.set(admission)
    try:
        yield admission
    finally:
        _ACTIVE.reset(token)
        if admit(evidence, sources, runtime) != admission:
            raise ValueError('T16 proposal sources changed during the request')


if __name__ == '__main__':
    import os

    directory = Path(__file__).parent
    result = admit(directory / 'dflash-t16-native-evidence', directory, os.environ['TT_METAL_HOME'])
    print(json.dumps(dict(stage='dflash_native_preload_admission', passed=True, admission=result)))
