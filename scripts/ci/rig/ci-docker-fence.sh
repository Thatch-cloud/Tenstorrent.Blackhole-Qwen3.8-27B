#!/usr/bin/env bash
# ci-docker-fence: CPU placement for containers on the rig's host dockerd, applied as each starts
# (installed at /usr/local/bin/ci-docker-fence.sh, run by ci-docker-fence.service as root).
#
#   buildx_buildkit_*   CI image builders: cores 16-31,48-63 (the CI half), cpu-shares 205.
#   thatch-inference-*  the node agent's serving containers: cores 0-15,32-47 (the half no CI unit
#                       is pinned to), cpu-shares 262144 (cgroup v2 cpu.weight 10000, the maximum).
#
# Docker containers sit in system.slice, which the kernel weighs against kubepods.slice (the ARC
# runner pods and the control plane) at the slices' own cpu.weight. kubelet gives kubepods.slice
# 2500; system.slice defaulted to 100, so under CI load every docker container shared ~4% of the
# chip. A systemd drop-in (systemctl set-property system.slice CPUWeight=2500) puts the two slices
# level; within system.slice the serving container's weight then dwarfs the builders' and the
# runner services' (CPUWeight=20). The serving container's own --cpus quota still caps it.
set -u
CI_CPUS=16-31,48-63
SERVING_CPUS=0-15,32-47

fence() {
  local id=$1 name=$2
  case "$name" in
    buildx_buildkit_*)
      docker update --cpuset-cpus "$CI_CPUS" --cpu-shares 205 "$id" >/dev/null 2>&1 \
        && echo "fenced CI builder $name to $CI_CPUS" ;;
    thatch-inference-*)
      docker update --cpuset-cpus "$SERVING_CPUS" --cpu-shares 262144 "$id" >/dev/null 2>&1 \
        && echo "prioritised serving container $name: cores $SERVING_CPUS, weight 10000" ;;
  esac
}

docker ps --format '{{.ID}} {{.Names}}' | while read -r id name; do fence "$id" "$name"; done
docker events --filter type=container --filter event=start --format '{{.Actor.ID}} {{.Actor.Attributes.name}}' \
  | while read -r id name; do fence "$id" "$name"; done
