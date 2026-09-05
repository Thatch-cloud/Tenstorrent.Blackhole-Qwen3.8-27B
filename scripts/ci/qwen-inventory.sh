#!/usr/bin/env bash
set -euo pipefail
mkdir -p experiment-results
exec > >(tee experiment-results/inventory.log) 2>&1
printf 'utc=%s\nrunner=%s\ncommit=%s\n' "$(date -u +%FT%TZ)" "${RUNNER_NAME:-local}" "$(git rev-parse HEAD)"
uname -srvmo
id
free -h
df -h / /tmp
printf '\nDevice nodes\n'
find /dev/tenstorrent -maxdepth 1 -type c -printf '%f %M %u:%g\n'
printf '\nPCI inventory\n'
lspci -Dnn | grep -i -E 'tenstorrent|1e52' || true
printf '\nDevice users (read-only)\n'
if command -v fuser >/dev/null; then
    for device in /dev/tenstorrent/*; do fuser "$device" || true; done
fi
printf '\nDocker access\n'
docker version --format '{{.Server.Version}}'
docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'
printf '\nCandidate images\n'
docker images --format '{{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.Size}}' | grep -i -E 'qwen|tt-vllm|thatch-serving-tt' || true
printf '\nStopped Qwen containers and mount paths (no environment values)\n'
while IFS= read -r container; do
    test -n "$container" || continue
    docker inspect --format '{{.Name}} image={{.Config.Image}} status={{.State.Status}} image_id={{.Image}}' "$container"
    docker inspect --format '{{range .Mounts}}{{.Source}} -> {{.Destination}} ({{.Mode}}){{println}}{{end}}' "$container"
done < <(docker ps -a --filter name=Qwen --format '{{.ID}}')
printf '\nInventory only: no containers started/stopped, no reset or model execution.\n'
