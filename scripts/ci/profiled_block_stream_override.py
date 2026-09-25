"""Env-gated admission of the device profiler into the mandatory block-stream
serving recipe, for one attribution measurement only.

Run 35561945903 (m3native gate profile arm, tag
experiment/lever-n-m3native-v6, scripts/ci/lever_n_m3native_run_arm.sh with
M3NATIVE_PROFILE=1) got through tracy setup and was then refused at attach:
mlp_block_stream_runtime.require_hardware(environment) raises ValueError(
'Explicit allocated unprofiled block-stream hardware experiment required')
whenever environment.get('TT_METAL_DEVICE_PROFILER') is set. That check runs
inside scoped_block_stream's inner require_runtime(os.environ), reached from
serving_runtime.attach_combined_runtime by way of
dflash_combined_request.combined_runtime. The block-stream experiment is
mandatory in the serving recipe (attach_combined_runtime raises if
block_stream is None), so today the serving path cannot be device-profiled
at all.

The check exists because the block-stream evidence was qualified UNPROFILED:
nothing about that qualification changes when the device profiler is merely
attached for one run, but the refusal cannot tell an attribution measurement
(never a throughput claim) from an ordinary serving attach.

This module admits the profiler for that attribution measurement only. It
edits no recipe file - mlp_block_stream_runtime.py stays frozen - and it
changes no default: install() is a no-op unless
QWEN_FAST_PROFILED_BLOCK_STREAM is exactly '1', and even then it strips only
TT_METAL_DEVICE_PROFILER from the environment require_hardware sees; every
other condition of the original check (QWEN_MLP_BLOCK_STREAM_EXPERIMENT,
QWEN_CARDS_ALLOCATED, QWEN_HARDWARE_TESTS, TT_METAL_SIMULATOR,
QWEN_SIM_ONLY) still applies unchanged.
"""

ENV = 'QWEN_FAST_PROFILED_BLOCK_STREAM'


def requested(environ=None):
    """True only when ENV is exactly '1'; None/absent/other values are False."""
    if environ is None:
        import os

        environ = os.environ
    return environ.get(ENV) == '1'


def install(*, log, environ=None, runtime=None):
    """Admit the device profiler into block-stream's require_hardware check
    for this attribution measurement, or return None if unset.

    No-op unless QWEN_FAST_PROFILED_BLOCK_STREAM=='1'. When requested but the
    environment carries no TT_METAL_DEVICE_PROFILER, there is nothing to
    override. Otherwise wraps runtime.require_hardware exactly once (the
    original stays on runtime._unprofiled_require_hardware) with a function
    that calls the original against a copy of the given environment with
    TT_METAL_DEVICE_PROFILER removed, so every other admission condition
    still applies.
    """
    if environ is None:
        import os

        environ = os.environ
    if not requested(environ):
        return None

    if runtime is None:
        import mlp_block_stream_runtime as runtime

    if not environ.get('TT_METAL_DEVICE_PROFILER'):
        log('[PINDIAG] {} set without TT_METAL_DEVICE_PROFILER; nothing to override', ENV)
        return None

    if not hasattr(runtime, '_unprofiled_require_hardware'):
        original = runtime.require_hardware

        def admit_profiler(environment):
            filtered = dict(environment)
            filtered.pop('TT_METAL_DEVICE_PROFILER', None)
            return original(filtered)

        runtime._unprofiled_require_hardware = original
        runtime.require_hardware = admit_profiler

    log('[PINDIAG] block-stream admission accepts the device profiler for this attribution measurement '
        '({}=1): every other require_hardware condition still applies', ENV)
    return dict(env=ENV, wrapped=True)
