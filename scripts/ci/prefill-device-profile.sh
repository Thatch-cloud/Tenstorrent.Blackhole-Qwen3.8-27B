#!/usr/bin/env bash
# Device-profile a prefill workload. Same tracy flags as
# dspark-combined-device-profile.sh, which is the combination that produced a complete
# C++ report: mid-run dumping keeps device markers from accumulating until final
# processing, which is what OOMed run 34296336943 at the 96 GiB container limit.
# --disable-device-data-push-to-tracy keeps the web GUI out of CI.
set -euo pipefail
output=/experiment/results/prefill-profile
mkdir -p "$output"
preserve_metadata() {
    mkdir -p "$output/metadata"
    for name in tracy_ops_data.csv cpp_device_perf_report.csv; do
        if [ -f "$output/.logs/$name" ]; then cp "$output/.logs/$name" "$output/metadata/$name"; fi
    done
}
trap preserve_metadata EXIT
export TTNN_OP_PROFILER=1 TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_TRACE_TRACKING=1
export TT_METAL_PROFILER_CPP_POST_PROCESS=1 TT_METAL_PROFILER_MID_RUN_DUMP=1
python3 -m tracy -p --check-exit-code --disable-device-data-dump-to-files \
    --disable-device-data-push-to-tracy --dump-device-data-mid-run --op-support-count 20000 \
    -o "$output" "$@" 2>&1 | tee "$output/console.log"
preserve_metadata
# The profiler's own default artifacts live under TT_METAL_HOME, while -o sets a
# separate tracy artifacts folder. Runs 35421182587 and 35422060812 produced neither a
# .logs directory under -o nor a report, so say plainly what exists anywhere it could be
# before failing, instead of inferring from absence.
echo '--- profiler artifacts anywhere they could be ---'
for root in "$output" /opt/tt-metal/generated/profiler /opt/tt-metal/generated; do
  if [ -d "$root" ]; then
    find "$root" -maxdepth 4 -type f \( -name '*.csv' -o -name '*.tracy' -o -name '*.log' \)       -printf '%10s  %p' -exec echo '' ';' 2>/dev/null | head -25
  else
    echo "absent: $root"
  fi
done
echo '--- end artifact scan ---'
# Salvage: if the report landed in the profiler's own tree rather than under -o, take it.
for candidate in /opt/tt-metal/generated/profiler/.logs/cpp_device_perf_report.csv                  /opt/tt-metal/generated/profiler/reports/cpp_device_perf_report.csv; do
  if [ -s "$candidate" ] && [ ! -s "$output/metadata/cpp_device_perf_report.csv" ]; then
    mkdir -p "$output/metadata"
    cp "$candidate" "$output/metadata/cpp_device_perf_report.csv"
    echo "salvaged report from $candidate"
  fi
done
test -s "$output/metadata/cpp_device_perf_report.csv"
