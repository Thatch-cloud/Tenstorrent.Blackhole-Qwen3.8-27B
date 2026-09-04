# Qwen3.8-27B serving base with the full optimised decode stack baked in (docs/optimisation-plan.md
# §0 ship list): lever A (fused GDN decode, #53587) + C (bf16 GDN decode/state) + D (bf8 KV in
# SDPA) + K (conv+gates, packed q/k/v, direct projection read, norm+gate fold, the no-zeroing
# recurrence reader) + M (the attention decode prologue as one op).
#
# This is the BASE for Thatch.Server's thatch-serving-tt image (tt-serving-image.yml takes it as
# `base`; docker/tenstorrent-serving.Dockerfile layers py/serving on top), so it carries only what
# the model needs: the rebuilt ttnn libraries, the JIT kernel sources of every grafted op, the
# wired model files, and the engagement flags as ENV. Per-model vLLM kwargs (context length, the
# tt additional_config) stay in Thatch.Server's engine.py; nothing here sets THATCH_VLLM_KWARGS.
#
# Built against tt-metal 9f9cd4fd, the rev the base carries: the .so is ABI-bound to that
# libtt_metal.so, so this FROM line and the build tree (build-k.sh in the ttbuild container) must
# stay in step. Build context: a directory holding the graft (opgraft-K) and the wired files.
#
# The base is the lever-A graft image that thatch-serving-tt builds from today (the plain
# v0.77.0-rc1-prstack-plugin tag is not in the registry). Every COPY below is a strict overwrite
# of a file that image already carries, and the result is byte-for-byte what the benchmark
# runners produced by bind-mounting the same files over the same image.
#
# Deliberately NOT included: QWEN36_SHARD_GREEDY / sample_on_device_mode (endpoint row 3d). It
# is greedy-only -- temperature>0 requests silently get argmax -- so it is wrong for a general
# API until the sampler is request-aware.
FROM zot.thatch.local:5000/tt-vllm:qwen38-fused-decode

# Rebuilt libraries: decode_gated_delta_rule (#53587 + our packed/fold/reader patches),
# gdn_decay, gdn_conv_gates, gdn_norm_gate, attn_prep.
COPY opgraft-K/_ttnn.so    /opt/tt-metal/ttnn/ttnn/_ttnn.so
COPY opgraft-K/_ttnncpp.so /opt/tt-metal/build_Release/ttnn/_ttnncpp.so

# Device kernel sources: tt-metal JIT-compiles these at run time, so the .so alone is not enough.
COPY opgraft-K/gdn_decay                /opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/gdn_decay
COPY opgraft-K/decode_gated_delta_rule  /opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/decode_gated_delta_rule
COPY opgraft-K/gdn_conv_gates           /opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/gdn_conv_gates
COPY opgraft-K/gdn_norm_gate            /opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/gdn_norm_gate
COPY opgraft-K/attn_prep                /opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/attn_prep

# The wired model files (strict supersets of the base files, every change behind a flag).
COPY opgraft-K/ttnn_delta_rule_ops.py /opt/tt-metal/models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_ops.py
COPY wrap-K/tp.py           /opt/tt-metal/models/demos/blackhole/qwen36/tt/gdn/tp.py
COPY wrap-K/attention_tp.py /opt/tt-metal/models/demos/blackhole/qwen36/tt/attention/tp.py

# The ship stack, on by default. Any one of these off returns that path to the base behaviour.
ENV QWEN_GDN_FUSED_DECODE=1 \
    QWEN35_GDN_DECODE_BF16=1 \
    QWEN35_GDN_STATE_BF16=1 \
    QWEN_SDPA_BF8=1 \
    QWEN_GDN_CONV_GATES=1 \
    QWEN_GDN_PACKED_QKV=1 \
    QWEN_GDN_FUSED_INPLACE=1 \
    QWEN_GDN_PROJ_DIRECT=1 \
    QWEN_GDN_NORM_GATE=1 \
    QWEN_ATTN_PREP=1
