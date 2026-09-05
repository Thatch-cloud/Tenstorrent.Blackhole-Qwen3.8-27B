#!/usr/bin/env bash
set -euo pipefail
mkdir -p experiment-results
image=sha256:e0ce0246ae98ab105fcd4918ad0b40dbd5c7eb5bdf360d44060712704849a283
docker image inspect --format 'tested_image={{.Id}}' "$image" | tee experiment-results/hardware-image.log
probe_id=''
test_id=''
cleanup() {
    status=$?
    trap - EXIT
    if [ -n "$test_id" ]; then
        docker logs "$test_id" > experiment-results/hardware-container.log 2>&1 || true
        docker cp "$test_id:/experiment/results/." experiment-results/ || true
        docker rm -f "$test_id" >/dev/null || true
    fi
    if [ -n "$probe_id" ]; then docker rm -f "$probe_id" >/dev/null || true; fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
probe_id=$(docker create -i --network none --read-only --pid host --cap-drop ALL \
    --cap-add SYS_PTRACE --security-opt no-new-privileges --user 0:0 \
    --mount type=bind,src=/dev/tenstorrent,dst=/host-dev/tenstorrent,readonly \
    --label thatch.qwen.ownership-probe=true --entrypoint python3 "$image" -)
docker start -ai "$probe_id" < scripts/ci/device-owners.py | tee experiment-results/device-owners.log
probe_exit=$(docker inspect --format '{{.State.ExitCode}}' "$probe_id")
test "$probe_exit" = 0
docker rm "$probe_id" >/dev/null
probe_id=''
test_id=$(docker create --network none --cap-drop ALL --cap-add SYS_NICE \
    --security-opt no-new-privileges --pids-limit 2048 --memory 32g --cpus 12 \
    --device /dev/tenstorrent/0 --device /dev/tenstorrent/2 \
    --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
    --label thatch.qwen.correctness=true --workdir /opt/tt-metal \
    -e QWEN_HARDWARE_TESTS=1 -e TT_METAL_HOME=/opt/tt-metal -e TT_METAL_CACHE=/tmp/qwen-test-cache \
    -e TT_METAL_SIMULATOR= -e TT_METAL_SLOW_DISPATCH_MODE= \
    -e TT_METAL_MOCK_CLUSTER_DESC_PATH= -e PYTHONDONTWRITEBYTECODE=1 \
    -e HF_HUB_OFFLINE=1 --entrypoint /bin/bash "$image" /experiment-scripts/ci/hardware-suite.sh)
docker cp scripts "$test_id:/experiment-scripts"
docker cp optimisation "$test_id:/experiment-optimisation"
docker start -a "$test_id" | tee experiment-results/hardware-console.log
exit_code=$(docker inspect --format '{{.State.ExitCode}}' "$test_id")
test "$exit_code" = 0
