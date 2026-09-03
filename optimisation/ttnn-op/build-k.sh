#!/bin/bash
# Build the K ops inside the ttbuild container (tt-metal 9f9cd4fd, the rev both serving images
# carry), then assemble ~/opgraft-K: the rebuilt libraries plus every grafted op's JIT kernel
# sources and the wrapper, which the runners bind-mount over either image.
#   build-k.sh            incremental ninja build (~30 s) + graft assembly
set -u
OPS=/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer
G=$HOME/opgraft-K

echo "### build start $(date -Is)"
# Replace (not nest into) each op tree in the build container.
for op in gdn_conv_gates gdn_norm_gate decode_gated_delta_rule; do
  docker exec ttbuild rm -rf $OPS/$op
  docker cp "$HOME/kwork/$op" ttbuild:$OPS/$op
done
# decode_gated_delta_rule is upstream #53587's op plus patch_packed.py (K milestone 2); its CB
# namespace is renamed there too because the transformer ops are unity-built and per-file
# `namespace cb` blocks collide.
docker cp "$HOME/kwork/register_ops.py" ttbuild:/tmp/register_ops.py
docker exec ttbuild python3 /tmp/register_ops.py
docker exec ttbuild bash -c "cd /opt/tt-metal && set -o pipefail && ninja -C build_Release ttnn/_ttnncpp.so ttnn/_ttnn.so 2>&1 | tail -25"
rc=$?
echo "### ninja rc=$rc $(date -Is)"
if [ "$rc" != "0" ]; then echo "### build FAILED; graft not assembled"; exit 1; fi
mkdir -p "$G"
docker cp ttbuild:/opt/tt-metal/build_Release/ttnn/_ttnncpp.so "$G/_ttnncpp.so"
docker cp ttbuild:/opt/tt-metal/build_Release/ttnn/_ttnn.so "$G/_ttnn.so"   # the Python module (binding lives here)
for op in gdn_conv_gates gdn_norm_gate decode_gated_delta_rule; do
  rm -rf "$G/$op"; cp -r "$HOME/kwork/$op" "$G/$op"
done
rm -rf "$G/gdn_decay"; cp -r "$HOME/opgraft-53587/gdn_decay" "$G/"
cp "$HOME/kwork/ttnn_delta_rule_ops.py" "$G/"   # wrapper with the packed + in-place entries
ls -la "$G" | head -12; echo "### graft assembled $(date -Is)"
