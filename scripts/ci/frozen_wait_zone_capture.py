"""Explicit raw-marker capture adapter for future diagnostic-only hardware runs."""

from frozen_recipe_context import replace_once


def adapt_capture(source):
    source = replace_once(source, ' --disable-device-data-dump-to-files', '')
    source = replace_once(source,
        'for name in tracy_ops_data.csv cpp_device_perf_report.csv; do',
        'for name in tracy_ops_data.csv cpp_device_perf_report.csv profile_log_device.csv; do')
    source = replace_once(source,
        'export TTNN_OP_PROFILER=1 TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_TRACE_TRACKING=1',
        'unset TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES\n'
        'export TTNN_OP_PROFILER=1 TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_TRACE_TRACKING=1')
    source = replace_once(source,
        'preserve_metadata\npython3 /experiment-scripts/ci/request_verifier_profile_report.py',
        'preserve_metadata\n'
        'test -s "$output/metadata/profile_log_device.csv"\n'
        'python3 /experiment-scripts/ci/request_verifier_profile_report.py')
    return source
