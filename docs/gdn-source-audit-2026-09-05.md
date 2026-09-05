# GDN fusion source audit

## Evidence and scope

Read-only image audit [33960203974](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33960203974)
passed at `96fb9fc`. It exports the 12 native fused recurrent operator files,
the delta-rule Python wrapper and Qwen TP GDN implementation, with SHA256 hashes.
Image: `sha256:f1e9b1a64b4f7aa04cd3d3b36fefed4d47320bfdd0f4d108d2ca85a932cf9465`.
No devices, weights, network or production volumes are exposed to the audit container.
This is source evidence, not a reproducible binary-build attestation or a new benchmark.

Compared against ctxbot's current heads:

- [PR 53482](https://github.com/tenstorrent/tt-metal/pull/53482): `a7100468ec7a7df7b9c7699719a66779fc52d8d0`.
- [PR 53587](https://github.com/tenstorrent/tt-metal/pull/53587): `c802b8c54034c78e0c8f5aeeb0828e58a9238c3a`.

The local simulator checkout does not contain this recurrent kernel. Comparing
only its pinned base HEAD incorrectly shows the operator absent. The serving
image contains additional source, so the runner export is the relevant truth source.

## Findings

The image already contains the substantive fixes described by PR 53482:

| Upstream fix | Image evidence |
| --- | --- |
| Separate Q/K inverse-normalization factors and read-back barriers | `cb_sc2`, `cb_sc3`, dedicated `inv_rms` calls and waits in compute kernel |
| Correct full-page output writes | Unpacked writer uses output accessor with `page_id = bh` |
| Batch-aware TILE addressing | Unpacked reader uses `b * padH + h`; scalar addressing uses batch pages |
| Multiple head instances per core | Factory computes `per_core`, passes `n_inst`; kernels loop over assigned instances |
| FP32-safe copy width and direct DMA gathers | Reader uses `16 * elem / 4` and reads directly into target circular buffers |

The image is not byte-identical to either PR: it extends the native implementation
with packed QKV/GQA reading, fused output RMS normalization and SiLU gating,
and a two-broadcast rank-one state update instead of the original outer-product
matmul. That last change removes the reader's otherwise-required padding-zeroing
work. These are changes inside the reader/compute/writer kernels, not merely
Python graph fusion. Source comments' historical speed claims are not new evidence.

Full-model trace run 33957724963 separately recorded engagement of
`QWEN_GDN_PROJ_DIRECT`, `QWEN_GDN_CONV_GATES`, `QWEN_GDN_PACKED_QKV` and
`QWEN_GDN_NORM_GATE`; both chips executed the recurrent and conv/gates device
operations. Thus the additional packed/norm-gate paths are not only dormant source.

The newer PR's documentation correction is missing: the image's unpacked public
API still describes TILE output although its output spec is ROW_MAJOR. The packed
API correctly distinguishes ROW_MAJOR output from fused gated TILE output. This
is a documentation discrepancy, not a missing performance kernel.

Decision: do not replace the image kernel wholesale with either PR, which would
discard existing packed/norm-gate functionality. No missing core fusion or listed
hardening fix was identified in this comparison. This does not certify all shapes
or prove that every image modification is numerically equivalent to upstream.

## Next bounded experiment: active-prefix recurrent state

The served B1 path uses an eight-slot persistent state allocation. In-place decode
requires `B == Bmax` and buffer identity, so merely setting
`QWEN_GDN_FUSED_INPLACE=1` cannot enable it for B1/Bmax8. Current code slices active
state and writes the result back through `_write_recurrent_state_prefix`.

1. Attribute these slice/write-back operations before estimating removable latency.
2. Prototype an explicit active-prefix state contract in an isolated native kernel
   build: retain the eight-slot allocation, read/write only requested active slots.
   Do not relax validation and pass a larger state tensor without a defined contract.
3. Gate both chips against frozen control: exact outputs and all recurrent-state
   slots over repeated eager and traced steps; inactive slots remain bit-identical.
   Include B1/3/8, slot reuse and prefill-to-decode transitions, unchanged BF16 state.
4. Only after correctness, measure three ABBA blocks with the unchanged pool and
   precision. Require more than 2% layer-latency reduction in every block before
   a full-model exact-token/quality gate. A failed exact gate is not waived by PCC.

This is a proposed experiment, not implemented or measured. It is separate from
gate/up projection fusion and does not establish a path to 200 committed tok/s.

## Audit validation

Initial run 33960060823 failed on artifact-directory permissions; 33960122517
then failed on Git ownership validation after matching the container UID to the
runner. The final fix applies command-local trust only to `/opt/tt-metal` in the
pinned image. No global Git or host-access policy changes were made.
Final audit passed; shell syntax checked and 56 existing CI-helper tests passed.
