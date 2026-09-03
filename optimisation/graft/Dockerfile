# Qwen3.8-27B serving image: base + the fused T=1 GDN decode from tt-metal #53482/#53587.
# Built against tt-metal 9f9cd4fd, the rev the base image carries -- the rebuilt .so is only ABI
# compatible with that libtt_metal.so, so this FROM line and the build tree must stay in step.
FROM zot.thatch.local:5000/tt-vllm:v0.77.0-rc1-prstack-plugin

# Rebuilt libraries carrying decode_gated_delta_rule (#53587) and gdn_recurrent_step (ours).
COPY _ttnn.so    /opt/tt-metal/ttnn/ttnn/_ttnn.so
COPY _ttnncpp.so /opt/tt-metal/build_Release/ttnn/_ttnncpp.so

# Device kernel sources: tt-metal JIT-compiles these at run time, so the .so alone is not enough.
COPY gdn_decay                /opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/gdn_decay
COPY decode_gated_delta_rule  /opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/decode_gated_delta_rule

# The integration that routes GDN decode through the fused op (strict superset of the base file).
COPY ttnn_delta_rule_ops.py      /opt/tt-metal/models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_ops.py

# On by default: without it the image behaves exactly like the base and the graft is pointless.
ENV QWEN_GDN_FUSED_DECODE=1
