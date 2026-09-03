#!/bin/bash
set -u
IMG=zot.thatch.local:5000/tt-serving:v0.77.0-rc1-prstack
D1=$(readlink -f /dev/tenstorrent/by-id/blackhole-CEF5729692C19E6D)
echo "### mmbench_d start=$(date -Is)"
docker rm -f mmbd >/dev/null 2>&1 || true
docker run --name mmbd --device "$D1" -v /dev/hugepages-1G:/dev/hugepages-1G --cap-add SYS_NICE \
  -v ~/ttcache:/ttcache -v ~/mmbench_d.py:/tmp/mmbench_d.py:ro \
  -e TT_METAL_HOME=/opt/tt-metal -e TT_METAL_CACHE=/ttcache -e TT_METAL_DEVICE_PROFILER=1 \
  --workdir /opt/tt-metal --entrypoint python3 "$IMG" -m tracy -r /tmp/mmbench_d.py > ~/mmbench_d.log 2>&1
echo "  rc=$?"; grep -E "^CFG|^PLAN" ~/mmbench_d.log
rm -rf ~/mmbd-out; mkdir -p ~/mmbd-out; docker cp mmbd:/opt/tt-metal/generated/profiler/reports/. ~/mmbd-out/ 2>/dev/null; docker rm -f mmbd >/dev/null
echo "=== MMBD COMPLETE ==="
