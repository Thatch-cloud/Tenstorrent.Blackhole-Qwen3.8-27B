#!/usr/bin/env bash
set -euo pipefail
mkdir -p experiment-results/gdn-source
image=sha256:f1e9b1a64b4f7aa04cd3d3b36fefed4d47320bfdd0f4d108d2ca85a932cf9465
probe_id=$(docker create --network none --read-only --cap-drop ALL \
    --user "$(id -u):$(id -g)" \
    --security-opt no-new-privileges --pids-limit 128 --memory 2g --cpus 2 \
    --mount "type=bind,src=$PWD/scripts/ci,dst=/audit,readonly" \
    --mount "type=bind,src=$PWD/experiment-results/gdn-source,dst=/results" \
    --entrypoint python3 "$image" -B /audit/gdn-source-audit.py)
trap 'docker rm -f "$probe_id" >/dev/null' EXIT
docker start -ai "$probe_id"
test "$(docker inspect --format '{{.State.ExitCode}}' "$probe_id")" = 0
