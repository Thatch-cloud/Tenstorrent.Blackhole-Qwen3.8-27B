# Batching the four packed users' GDN launches (task #46)

Device profile, run 35573521447 (chip 0, no watcher), per 64-row four-user round,
48 GDN layers. The custom GDN kernels launch once per user per layer:

| family | launches | each | round |
| --- | ---: | ---: | ---: |
| recurrence, `gdn_vsplit` stage, 96 cores | 192 | 0.194 ms | 37.3 ms |
| norm_gate, `gdn_vsplit` stage, 24 cores | 192 | 0.053 ms | 10.2 ms |
| conv-gates, K-stack op, 81 cores | 192 | 0.044 ms | 8.4 ms |
| state copies + `build_windows`, 48 cores | 352 | 0.017 ms | 6.0 ms |

There are two ways to put four users in one program. **Token axis**: concatenate the four
segments on the row axis and run one T=64 recurrence, restarting the state chain every 16
rows. **Core axis**: give each user its own disjoint share of the worker grid and put four
kernel-descriptor triples into one `MeshProgramDescriptor`.

## Token axis is blocked by pinned files, not by the kernel

The transform is small and I worked it out in full. `gdn_multitoken.transform`
(scripts/ci/gdn_multitoken.py:92-97) already turns the native batch axis into a token axis
(`const uint32_t bh = bh_start + token * H;`), reads the initial state once under
`if (token == 0)` and carries the chain through CB 30 (gdn_multitoken.py:83-88). Four
users need three further transforms over the generated string, never the pinned file:
reader `if (token == 0)` becomes `token % 16 == 0`; `STATE_BASE` (gdn_vsplit.py:16) gains
a `(token / 16) * 384` term; compute's three `it == 0` tests become `it % 16 == 0` and its
CB 30 push guard becomes `(it + 1) % 16 != 0`. The reader's input gathers are already
tile-row aware (gdn_vsplit.py:87-88) and CB sizes do not depend on T (gdn_vsplit.py:65-81),
so the kernel side is done. What blocks it is everything around it:

1. It consumes a **64-row packed conv/beta/g**, which means `gdn_decode_conv_gates` at
   `batch=64` (its host validation is exercised only to 32 rows,
   optimisation/ttnn-op/test_gdn_conv_gates.py:137-142) over **64-row windows built from
   four separate histories**, which `gdn_conv_windows.cpp:13-15` cannot do (it reads row
   0 of one history per tap).
2. It produces `states` at `(64, 24, 128, 128)` and windows at `(1, 64, 5120)`. The commit
   path demands one 16-row history set per user: `gdn_records.validate_packed_record`
   (scripts/ci/gdn_records.py:41-48) checks `piece['states'].shape[0] == stop - start`,
   `gdn_commit_dma.validate_shapes` caps rows at 32 and wants `(1, rows, 5120)` windows,
   and `restore_prefix` (gdn_multitoken_conv.py:47-48) rejects 64 rows outright. **Two of
   those three files are hash-pinned**, so this route cannot be wired into
   `_decode_packed` without editing pinned sources. Slicing the 64-row state back into
   four 16-row tensors costs a 12.6 MB per-chip copy per user per layer, which is more
   than the launch saving.

## Core axis, which is what this change implements

The fused 24-core kernel `gdn_multitoken.load_kernels(root, True)` runs one whole head
per core (Vt=4, kv=16) and folds norm/gate into the recurrence, so it needs no FP32 DRAM
bridge at all. Four users at 24 cores each is 96 of the 110 cores on the 11x10 grid, so
the four users run **concurrently on disjoint core sets in one launch**:

* one `MeshProgramDescriptor` per chip, 12 `KernelDescriptor`s (4 users x reader/writer/
  compute), each with its own `core_ranges`, its own `TensorAccessorArgs` and its own
  per-core runtime args; CBs declared once over the union of the four ranges
* **no kernel transform at all** - the sources are `gdn_multitoken`'s verbatim, so
  `HASHES` and `HELPER_HASH` keep passing and there is no new pinned-source surface
* every tensor keeps its **per-user** shape: `(16, 24, 128, 128)` states, `(1, 16, 5120)`
  windows, `(1, 16, 3072)` output. `restore_prefix`, `copy_prefix`, `gdn_commit_dma` and
  `gdn_records` are untouched.

Per-core work across the round is unchanged (96 cores x 4 state tiles x 16 tokens x 4
launches today, against 24 cores x 16 tiles x 16 tokens x 1 launch). What goes is three
launch overheads, the four-way redundant q/k reads the value split makes, and the FP32
bridge round trip through DRAM between its two stages.

**Expected time.** The T8/T16 verifier profile (docs/verifier-overhead-2026-09-19.md:213,
6.29 ms at T8 and 9.53 ms at T16 over 48 layers) puts the recurrence at 63.5 us fixed plus
8.44 us per token, so a launch is not cheap. The 24-worker fused arm measures 0.5217 ms at
T16 against the 96-worker split arm's 0.3865 ms in one synthetic harness
(docs/experiment-execution.md:1367-1375) whose fixed cost is about 0.128 ms of blocking
dispatch the in-trace profile does not carry, putting the fused arm near 0.39 ms on device.

| per layer | today | batched |
| --- | ---: | ---: |
| recurrence + norm_gate launches | 8 | 1 |
| recurrence + norm_gate device time | 0.988 ms | ~0.39 ms |
| round (48 layers) | 47.4 ms | ~18.8 ms |

That is about 29 ms of the 172 ms verify trace, and it is the whole of the 37.3 + 10.2 ms
the two stages cost now.

**Bit-identity.** Against four sequential single-user *fused* launches the batched program
is identical by construction: same kernel sources, same compile args, same runtime args,
same inputs; only the physical core coordinates and the `head` runtime arg's core binding
differ, and arithmetic does not depend on either. Against the deployed 96-core value-split
arm the evidence is run 34050458771 (`7db0198`), which compared the two arms directly and
found every output row and recurrent prefix exact on both chips
(docs/experiment-execution.md:1362-1366; the comparison is the one
scripts/ci/gdn_vsplit_README.md:189-191 specifies). `gdn_user_batch_device_test.py`
re-proves both on one card.

## What the device test cannot prove, and the one real risk

**L1 pressure on 72 more cores.** The fused plan reserves 616 KB of circular buffers per
core (`gdn_multitoken.cb_plan(True)`) against the value split's 276 KB, plus 200 KB for its
norm stage. Per core that is what the single-user fused path already runs at, so no core is
asked for anything new - but 96 cores are now asked instead of 24, leaving about 725 KB of
1.5 MB claimed on 72 cores that held 276 KB, and every L1-resident decode activation is
spread over the same grid. The device test allocates only its own inputs and will not see
this; the first full-model run under the flag will, as an L1 allocation failure at program
build. That is why the flag defaults OFF.

**Concurrency, not just launch count.** The four users' kernels are independent and run on
disjoint cores, but share DRAM bandwidth and the NOC, and the estimate assumes they overlap
cleanly. The device test's timing arms measure that directly, so its number is the answer.

## The other two families, and the order

1. **recurrence + norm_gate** - implemented here. 47.4 ms, 8 launches per layer to 1.
2. **`build_windows` and the compact state copies** - repo-owned DMA kernels
   (gdn_conv_windows.cpp, gdn_state_copy.cpp) hardcoding 48 workers and a page loop
   strided by 48. Four users at 48 cores is 192, over the 110-core grid, so this needs the
   stride as a runtime arg and four descriptors at 24 cores - the same core-axis pattern,
   no new math. Worth about 2.4 ms of the 6.0 ms bucket, cheap and low risk.
3. **conv-gates** - a compiled TTNN op, one program per call, so it cannot be
   multi-instanced from the host at all. Batching it means `batch=64` over four-user
   windows, which is the token-axis route with its op-cap and commit-geometry problems.
   Last, and only alongside raising the op's host validation in the ttbuild container.
