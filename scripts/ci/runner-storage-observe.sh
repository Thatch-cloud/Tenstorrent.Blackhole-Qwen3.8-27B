#!/usr/bin/env bash
set -euo pipefail
test "$RUNNER_NAME" = thatch-build-amd64-02-cp-temp
output=${1:?output directory required}
mkdir -p "$output"

snapshot() {
    date -u +%FT%TZ
    for name in /proc/pressure/io /proc/pressure/memory /proc/pressure/cpu /proc/diskstats /proc/meminfo; do
        printf '\n%s\n' "$name"
        cat "$name"
    done
}

snapshot > "$output/storage-before.txt"
{
    date -u +%FT%TZ
    lsblk -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINTS
    df -hT
    df -i
    docker_root=$(timeout 10 docker info --format '{{.DockerRootDir}}')
    cache_root=$(timeout 10 docker volume inspect qwen-experiments-f1e9b1a64b4f --format '{{.Mountpoint}}')
    for path in "$docker_root" "$cache_root" \
        /home/thatch/hf-cache/hub/models--Qwen--Qwen3.8-27B \
        /home/thatch/.cache/qwen-experiments/dspark-b9a5dbdf03bc999c6c73c426b19c2d9041cea393; do
        printf '\nBacking filesystem: %s\n' "$path"
        findmnt -T "$path" -o TARGET,SOURCE,FSTYPE,OPTIONS || true
    done
} > "$output/storage-layout.txt" 2>&1
if command -v vmstat >/dev/null; then
    timeout 25 vmstat 1 16 > "$output/vmstat.txt" 2>&1
fi
if command -v iostat >/dev/null; then
    timeout 20 iostat -xz 1 10 > "$output/iostat.txt" 2>&1
fi
if command -v pidstat >/dev/null; then
    timeout 20 pidstat -d 1 10 > "$output/process-io.txt" 2>&1
fi
snapshot > "$output/storage-after.txt"
printf '%s\n' 'Read-only idle-host observation; no model load, device access, writes benchmarked, cleanup or process termination.' \
    > "$output/scope.txt"
