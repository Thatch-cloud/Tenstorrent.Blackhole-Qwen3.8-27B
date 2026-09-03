#!/bin/bash
set -u
IMG=zot.thatch.local:5000/tt-serving:v0.77.0-rc1-prstack
D1=$(readlink -f /dev/tenstorrent/by-id/blackhole-CEF5729692C19E6D)
echo "### dram read benchmark: k=5120 n=8704 (the gate/up shape), each data format"
for df in 2 0 1; do
  docker run --rm --device "$D1" -v /dev/hugepages-1G:/dev/hugepages-1G --cap-add SYS_NICE -w /opt/tt-metal -e TT_METAL_HOME=/opt/tt-metal \
    --entrypoint /opt/tt-metal/build_Release/test/tt_metal/perf_microbenchmark/8_dram_adjacent_core_read/test_dram_read "$IMG" \
    --k 5120 --n 8704 --num-blocks 8 --num-tests 5 --data-type $df --bypass-check 2>&1 | grep -vE "UMD|^ *$" | tail -8 | cut -c1-150 | sed "s/^/  df=$df  /"
done
echo "=== MMBENCH COMPLETE ==="
