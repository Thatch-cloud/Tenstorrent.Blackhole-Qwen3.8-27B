"""Env-gated admission of one named, non-pinned combined-runtime binary for a
measurement gate only.

dflash_combined_sim_runtime.BINARY_SHA256 pins the exact combined hardware
runtime binary at both entries in BINARIES; binary_hashes() raises
'Exact combined hardware runtime binary required' whenever either path hashes
to anything else, and dflash_t16_native_scope.admit()/validate_record()
require admission['runtime_binaries'] == dict.fromkeys(BINARIES,
BINARY_SHA256). Run 35558196471 tried to attach with a rebuilt _ttnncpp.so (a
batch-64 kernel graft, ~/opgraft-K64c on the rig) mounted over the image, and
that pin refused it.

The refusal was correct: that first graft binary had been built over the
ORIGINAL SDPA factory (a263559f...) rather than the COMBINED one
(fd8c0676...) that dflash_combined_sim_runtime.factory_bytes() reconstructs
from it, so attaching it would silently have changed the qualified draft
fp32 SDPA intermediates (the fp32-vs-bf16 stats_df split in REPLACEMENT) out
from under the measurement, with no failure to say so.

This module does not touch the hash-pinned sources or lower the pin itself.
It admits exactly one substitute binary, named by the operator through
QWEN_FAST_RUNTIME_BINARY_SHA256, and only after independently confirming (a)
that binary is mounted at every path BINARIES names - not a partial or stale
mount - and (b) the runtime's SDPA factory still hashes to COMBINED_FACTORY,
i.e. the substitute sits over the same combined factory the pin was built
for. On success it rebinds BINARY_SHA256 on the simulator module and on each
named scope module for the life of the process, and returns a record of what
was measured (the pinned value, the override, the factory hash, and the
per-path binary hashes actually read) so the run's evidence names the exact
grafted binary rather than silently reporting the pinned one. Unset or empty,
it is inert.
"""

from pathlib import Path


ENV = 'QWEN_FAST_RUNTIME_BINARY_SHA256'


def requested(environ=None):
    """None when ENV is unset/empty; its lowercase value when it is 64 hex chars."""
    if environ is None:
        import os

        environ = os.environ
    value = environ.get(ENV)
    if not value:
        return None
    value = value.lower()
    if len(value) != 64 or any(character not in '0123456789abcdef' for character in value):
        raise ValueError('{} must be a 64-character hex sha256, got {!r}'.format(ENV, value))
    return value


def install(runtime_root, *, log, environ=None, sim=None, scopes=None):
    """Admit the requested binary for this measurement, or return None if unset.

    Raises ValueError if the requested binary is not mounted at every pinned
    path, or if it does not sit over the combined SDPA factory. On success,
    rebinds BINARY_SHA256 on `sim` and each of `scopes` and returns a record.
    """
    value = requested(environ)
    if value is None:
        return None

    if sim is None:
        import dflash_combined_sim_runtime as sim
    if scopes is None:
        import dflash_t16_native_scope

        scopes = [dflash_t16_native_scope]

    pinned = sim.BINARY_SHA256
    if value == pinned:
        log('[PINDIAG] {} names the pinned binary {}; nothing to override', ENV, value[:16])
        return None

    actual = {name: sim.digest(Path(runtime_root) / name) for name in sim.BINARIES}
    mismatched = {name: digest for name, digest in actual.items() if digest != value}
    if mismatched:
        raise ValueError('{} names {} but {} do not match: {}'.format(
            ENV, value[:16], ', '.join(sim.BINARIES),
            ', '.join('{}={}'.format(name, digest[:16]) for name, digest in mismatched.items())))

    factory = sim.digest(Path(runtime_root) / sim.FACTORY)
    if factory != sim.COMBINED_FACTORY:
        raise ValueError(
            '{} names {} but the runtime SDPA factory is {}, not the combined factory {}; '
            'a grafted binary is admitted only over the combined SDPA factory'.format(
                ENV, value[:16], factory[:16], sim.COMBINED_FACTORY[:16]))

    sim.BINARY_SHA256 = value
    for scope in scopes:
        scope.BINARY_SHA256 = value
    log('[PINDIAG] runtime binary pin overridden for this measurement: {} replaces {} at {} '
        '(SDPA factory {}, combined)', value[:16], pinned[:16], ', '.join(sim.BINARIES), factory[:16])
    return dict(pinned=pinned, override=value, factory=factory, binaries=actual)
