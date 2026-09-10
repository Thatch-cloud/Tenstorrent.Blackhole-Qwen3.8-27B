# Batched GDN publication experiment

Status: implementation and host checks only. No simulator or hardware pass yet.
The qualified publication path and serving defaults remain unchanged.

`scripts/ci/gdn-batched-publication-probe.py` now implements the first simulator
comparison: native and candidate against an independent host reference, two
distinct chip inputs, all 17 prefixes at one layer, and selected boundary prefixes
at 48 layers. It resets inputs and NaN-poisoned checkpoints before every eager
execution and changed-input captured replay. It checks all logical source, native
and checkpoint elements and stable device addresses. Checkpoints are poisoned
including physical tile padding. Readback expands the host tensor to its padded
shape, requiring checkpoint padding to become zero while source/native padding
canaries remain NaN. The host-only TTNN tiling/readback mechanism passed a local
check without opening a device; device execution is still unproven.
The previous full-chain simulator was deliberately stopped as incomplete and its
runtime restoration verified. One-layer execution is now running as
`qwen-batched-publication-1-20260911.service`, report
`20260910T224336Z-417-gdn-batched-publication-probe.json`. No pass is claimed yet.

`gdn_batched_publication_gate.py` independently requires the complete ordered
matrices for both layer counts, exact source hashes before/after, clean closure,
exit zero and all padding/poison checks. Its rejection tests pass; this does not
substitute for executing the simulator or the later hardware continuation gates.

After the existing owner exits and restores its runtime, use the bounded launcher:

```sh
QWEN_SIM_LAYER_PROBE=gdn-batched-publication-probe QWEN_SIM_BOUNDED_MEMORY=1 \
  KERNEL_TIMEOUT=1800 bash optimisation/sim/run-native-layer-dispatch-probe.sh --layers 1
```

Require complete passing evidence before repeating with `--layers 48`. Neither
report alone replaces the full accepted-prefix continuation hardware gate.

## Why test it

The matched score-layout request spends about 11.77 ms/block in selection and
commit, including more than this kernel. Source inspection shows the current
publication kernel waits after each recurrent tile and each convolution tile.
That is not evidence that all 11.77 ms is removable.

The isolated `gdn_commit_batched_dma` candidate keeps two workers per layer
(96 workers for 48 layers). Each worker batches eight independent transfers
before a read barrier and eight publications before a write barrier. Scratch
increases from 4 KiB to 32 KiB per worker. Convolution reads fetch only the two
32-byte face rows actually used, instead of each complete 2048-byte tile.
The arithmetic, selected prefix, checkpoint padding and inactive-slot writes
are unchanged by design; simulator evidence must verify that claim.

## Admission sequence

1. Compare native and candidate publication in TTsim on both chips, T16 prefixes
   0 through 16, with distinct recurrent/convolution histories and inactive slots.
2. Require bit-exact native and checkpoint outputs, zeroed checkpoint padding,
   unchanged entry/history inputs and all seven inactive native slots preserved.
3. Capture and replay changed histories, poison outputs, and check stable binding
   and clean release. Include a 48-layer case to exercise all 96 workers.
4. Only after simulator qualification, compare learned-state hardware publication
   and full accepted-prefix continuation. Keep native publication as control.
5. Run matched full requests with the repeat-confirmed score layout in both arms;
   compare committed TG and the complete selection/publication boundary, not
   kernel timing alone. Reject any token, state, acceptance or quality regression.

The current verifier profiler is independent and still needed: even removing
publication completely would not make the existing request reach 200 TG.
