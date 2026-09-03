#!/bin/bash
# Build the gdn_conv_gates op inside the ttbuild container (tt-metal 9f9cd4fd, the rev the
# serving image carries), then assemble ~/opgraft-K: the graft dir with the rebuilt .so
# files plus every JIT kernel source the runners mount.
#   build-k.sh            incremental ninja build + graft assembly
set -u
OPS=/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer
SRC=$HOME/kwork/gdn_conv_gates          # op source (synced from the workstation)
G=$HOME/opgraft-K

echo "### build start $(date -Is)"
docker exec ttbuild rm -rf $OPS/gdn_conv_gates
docker cp "$SRC" ttbuild:$OPS/gdn_conv_gates
docker exec ttbuild sh -c "cd /opt/tt-metal && python3 - <<'PY'
import re
p='ttnn/cpp/ttnn/operations/transformer/sources.cmake'; s=open(p).read()
if 'gdn_conv_gates' not in s:
    s=s.replace('    decode_gated_delta_rule/decode_gated_delta_rule.cpp\n',
      '    gdn_conv_gates/gdn_conv_gates.cpp\n    gdn_conv_gates/device/gdn_conv_gates_device_operation.cpp\n    gdn_conv_gates/device/gdn_conv_gates_program_factory.cpp\n    decode_gated_delta_rule/decode_gated_delta_rule.cpp\n',1)
    s=s.replace('    decode_gated_delta_rule/decode_gated_delta_rule_nanobind.cpp\n',
      '    decode_gated_delta_rule/decode_gated_delta_rule_nanobind.cpp\n    gdn_conv_gates/gdn_conv_gates_nanobind.cpp\n',1)
    open(p,'w').write(s); print('sources.cmake: gdn_conv_gates added')
p='ttnn/cpp/ttnn/operations/transformer/CMakeLists.txt'; s=open(p).read()
if 'gdn_conv_gates' not in s:
    s=s.replace('    gdn_decay/device/kernels/*.cpp\n','    gdn_decay/device/kernels/*.cpp\n    gdn_conv_gates/device/kernels/*.cpp\n',1)
    open(p,'w').write(s); print('CMakeLists: kernel glob added')
p='ttnn/cpp/ttnn/operations/transformer/transformer_nanobind.cpp'; s=open(p).read()
if 'gdn_conv_gates' not in s:
    s=s.replace('#include \"decode_gated_delta_rule/decode_gated_delta_rule_nanobind.hpp\"\n',
      '#include \"decode_gated_delta_rule/decode_gated_delta_rule_nanobind.hpp\"\n#include \"gdn_conv_gates/gdn_conv_gates_nanobind.hpp\"\n',1)
    s=s.replace('    bind_decode_gated_delta_rule(mod);\n','    bind_decode_gated_delta_rule(mod);\n    bind_gdn_conv_gates(mod);\n',1)
    open(p,'w').write(s); print('transformer_nanobind: bind added')
PY"
# incremental build of the two libraries only
# the grafted recurrence op's factory shares a unity TU with chunk_gated_delta_rule's: give its
# CB-index namespace a unique name (host code only; the JIT kernel sources are untouched)
docker exec ttbuild sh -c "cd /opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/decode_gated_delta_rule/device && grep -q 'namespace cbd ' decode_gated_delta_rule_program_factory.cpp || sed -i -e 's/^namespace cb {/namespace cbd {/' -e 's|^}  // namespace cb$|}  // namespace cbd|' -e 's/\bcb::/cbd::/g' decode_gated_delta_rule_program_factory.cpp && grep -c 'cbd::' decode_gated_delta_rule_program_factory.cpp"
docker exec ttbuild bash -c "cd /opt/tt-metal && set -o pipefail && ninja -C build_Release ttnn/_ttnncpp.so ttnn/_ttnn.so 2>&1 | tail -25"
rc=$?
echo "### ninja rc=$rc $(date -Is)"
if [ "$rc" != "0" ]; then echo "### build FAILED; graft not assembled"; exit 1; fi
mkdir -p "$G"
docker cp ttbuild:/opt/tt-metal/build_Release/ttnn/_ttnncpp.so "$G/_ttnncpp.so"
docker cp ttbuild:/opt/tt-metal/build_Release/ttnn/_ttnn.so "$G/_ttnn.so"
rm -rf "$G/gdn_conv_gates" "$G/decode_gated_delta_rule" "$G/gdn_decay"
docker cp ttbuild:$OPS/gdn_conv_gates "$G/gdn_conv_gates"
cp -r "$HOME/opgraft-53587/decode_gated_delta_rule" "$G/"
cp -r "$HOME/opgraft-53587/gdn_decay" "$G/"
cp "$HOME/opgraft-53587/ttnn_delta_rule_ops.py" "$G/"
ls -la "$G" | head; echo "### graft assembled $(date -Is)"
