#!/usr/bin/env bash
set -euo pipefail
mkdir -p experiment-results
exec > >(tee experiment-results/inventory.log) 2>&1
printf 'utc=%s\nrunner=%s\ncommit=%s\n' "$(date -u +%FT%TZ)" "${RUNNER_NAME:-local}" "$(git rev-parse HEAD)"
uname -srvmo
id
free -h
df -h / /tmp
printf '\nThe preceding system information is runner-visible, not proof of host device visibility.\n'
printf '\nDocker access\n'
docker version --format '{{.Server.Version}}'
docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'
printf '\nCandidate images\n'
docker images --format '{{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.Size}}' | grep -i -E 'qwen|tt-vllm|thatch-serving-tt' || true
printf '\nStopped Qwen containers and mount paths (no environment values)\n'
probe_image=''
containers=$(docker ps -a --filter name=Qwen --format '{{.ID}}')
while IFS= read -r container; do
    test -n "$container" || continue
    docker inspect --format '{{.Name}} image={{.Config.Image}} status={{.State.Status}} image_id={{.Image}}' "$container"
    docker inspect --format '{{range .Mounts}}{{.Source}} -> {{.Destination}} ({{.Mode}}){{println}}{{end}}' "$container"
    if [ -z "$probe_image" ]; then
        probe_image=$(docker inspect --format '{{.Image}}' "$container")
    fi
done <<< "$containers"
if [ -z "$probe_image" ]; then
    candidates=$(docker images --format '{{.Repository}} {{.ID}}' --no-trunc)
    while read -r repository image_id; do
        case "$repository" in
            */tt-vllm|*/thatch-serving-tt)
                probe_image=$image_id
                break
                ;;
        esac
    done <<< "$candidates"
fi
if [[ ! "$probe_image" =~ ^sha256:[a-f0-9]{64}$ ]]; then
    echo 'No local Qwen/TT serving image found; refusing to pull an unreviewed probe image.' >&2
    exit 1
fi
docker image inspect --format 'probe_image={{.Id}} os={{.Os}} arch={{.Architecture}}' "$probe_image"
printf '\nRunning-container device mappings (not a complete device-user check)\n'
running=$(docker ps -q)
while IFS= read -r container; do
    test -n "$container" || continue
    docker inspect --format '{{.Name}} privileged={{.HostConfig.Privileged}} devices={{json .HostConfig.Devices}}' "$container"
done <<< "$running"
probe_id=$(docker create -i --network none --read-only --cap-drop ALL \
    --security-opt no-new-privileges --user 65534:65534 --pids-limit 64 \
    --memory 128m --cpus 0.5 --label thatch.qwen.inventory=true \
    --mount type=bind,src=/dev/tenstorrent,dst=/host-dev/tenstorrent,readonly \
    --mount type=bind,src=/sys,dst=/host-sys,readonly \
    --entrypoint /bin/bash "$probe_image" -s)
trap 'docker rm -f "$probe_id" >/dev/null' EXIT
docker start -ai "$probe_id" <<'PROBE'
set -euo pipefail
printf '\nDocker-daemon host device nodes (metadata only)\n'
find /host-dev/tenstorrent -maxdepth 2 -printf '%P %y %M %u:%g %l\n'
printf '\nDocker-daemon host PCI inventory\n'
found=0
for device in /host-sys/bus/pci/devices/*; do
    test -r "$device/vendor" || continue
    read -r vendor < "$device/vendor"
    test "$vendor" = 0x1e52 || continue
    found=$((found + 1))
    read -r product < "$device/device"
    printf 'pci=%s vendor=%s device=%s\n' "${device##*/}" "$vendor" "$product"
    for field in current_link_speed current_link_width; do
        if test -r "$device/$field"; then
            printf '%s=' "$field"
            cat "$device/$field"
        fi
    done
done
printf 'tenstorrent_pci_devices=%s\n' "$found"
test "$found" -gt 0
printf '\nDevice idleness is UNVERIFIED: no host process scan or accelerator open performed.\n'
PROBE
printf '\nOnly the job-owned metadata probe was started. No serving changes, device resets or model execution.\n'
