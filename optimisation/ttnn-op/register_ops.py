"""Register the K ops in the tt-metal build tree (run inside the ttbuild container).

Idempotent: each op is added to sources.cmake (host + nanobind sources), the transformer
CMakeLists kernel glob, and transformer_nanobind.cpp (include + bind call) only if absent.
"""
import os

ROOT = "/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer"
OPS = [
    ("gdn_conv_gates", "bind_gdn_conv_gates"),
    ("gdn_norm_gate", "bind_gdn_norm_gate"),
]


def rw(path, fn):
    s = open(path).read()
    s2 = fn(s)
    if s2 != s:
        open(path, "w").write(s2)
        return True
    return False


for op, bind in OPS:
    def sources(s, op=op):
        if op in s:
            return s
        anchor = "    decode_gated_delta_rule/decode_gated_delta_rule.cpp\n"
        assert anchor in s, "sources.cmake anchor"
        s = s.replace(
            anchor,
            f"    {op}/{op}.cpp\n    {op}/device/{op}_device_operation.cpp\n    {op}/device/{op}_program_factory.cpp\n" + anchor,
            1,
        )
        nb_anchor = "    decode_gated_delta_rule/decode_gated_delta_rule_nanobind.cpp\n"
        assert nb_anchor in s, "sources.cmake nanobind anchor"
        return s.replace(nb_anchor, nb_anchor + f"    {op}/{op}_nanobind.cpp\n", 1)

    def cmake(s, op=op):
        if op in s:
            return s
        anchor = "    gdn_decay/device/kernels/*.cpp\n"
        assert anchor in s, "CMakeLists glob anchor"
        return s.replace(anchor, anchor + f"    {op}/device/kernels/*.cpp\n", 1)

    def nanobind(s, op=op, bind=bind):
        if op in s:
            return s
        inc = '#include "decode_gated_delta_rule/decode_gated_delta_rule_nanobind.hpp"\n'
        call = "    bind_decode_gated_delta_rule(mod);\n"
        assert inc in s and call in s, "transformer_nanobind anchors"
        s = s.replace(inc, inc + f'#include "{op}/{op}_nanobind.hpp"\n', 1)
        return s.replace(call, call + f"    {bind}(mod);\n", 1)

    a = rw(os.path.join(ROOT, "sources.cmake"), sources)
    b = rw(os.path.join(ROOT, "CMakeLists.txt"), cmake)
    c = rw(os.path.join(ROOT, "transformer_nanobind.cpp"), nanobind)
    print(f"{op}: sources={'added' if a else 'present'} glob={'added' if b else 'present'} nanobind={'added' if c else 'present'}")
