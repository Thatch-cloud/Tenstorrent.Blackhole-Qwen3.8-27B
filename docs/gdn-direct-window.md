# Direct causal windows inside GDN convolution

Status: DRAM and exact L1 projection simulator checks pass. The request-scoped
hardware adapter is prepared. **No hardware measurement or serving change.**

L1 run **35271886537**, commit `c86d9dd2b27faf0fdd78453470f28c546d9795dd`,
passes in **59 seconds**. The strict validator independently confirms all 56
output/checkpoint comparisons, 88 immutable checks, seven script hashes and four
native pins. Exit and container cleanup succeed. Admission pins report SHA-256
`f1837c9621c53cdb69a8c9047e867acc843b65b3762ca99c5e48a3dcf5345835`.

The adapter replaces only the frozen batched path's window/convolution section.
Recurrence, normalization, prefix availability, ownership and deferred publication
remain byte-identical after that section. Both imported call bindings are scoped
and restored; T2/T4/T8 tails stay native. It rejects non-L1 or unexpected T16
geometry and requires all four deferred-checkpoint flags. Eight host tests pass,
including failure restoration and unchanged descriptor arithmetic. Next is the
matched complete-runtime hardware experiment, not an isolated speed claim.

Run **35271222958** at `16cae4e643f69ace42e9925d85be973de4723291`
passes in **57 seconds**. Independently checked: all 56 output/checkpoint
comparisons and 88 immutable-input checks, source hashes, clean closure, exit
zero and successful container cleanup. Report SHA-256:
`be5e28018639768781ed15ac5321ebb117aed64adf4272a4ebbb413f06f8588f`.
Generated reader SHA-256:
`83b7404e3d95291bb7f7b622dc4776800f6f3321559af3ad0b179f5f5f984098`.

`DeviceLoopState` projects into L1, whereas that first probe used DRAM. The
probe now explicitly records placement and defaults to L1; the strict report
validator requires this placement for runtime admission. Kernel code, arithmetic,
work partition and source pins are unchanged. This avoids adding a DRAM copy
to the runtime merely to fit the first test.

The retained combined trace assigns about 3.49 ms per T16 verifier block to
48 source-consistent window-builder calls. Overlapping the old builder's writes
did not improve combined TG. This candidate instead removes the materialized
input-window stage and feeds causal rows directly to native convolution math.

Read-only CI inventory **35270248051** exports 12 files from the pinned image
in **13 seconds**, without opening cards or loading weights. All downloaded
file hashes match the inventory. The native reader actually consumes shifted
states `[st1, st2, st3, x]`; its writer preserves that advanced shift register.
Therefore the candidate must still publish four complete per-prefix checkpoint
tensors. Removing those outputs would break speculative rollback.

`gdn_direct_window.py` changes only the native reader's window construction:
read the projected tile and three entry-history tiles into 8 KiB private L1
scratch, assemble causal rows into the existing window CB, and retain the
native tap/gate paths. Native compute, rounding and writer remain unchanged.
The intended descriptor uses immutable entry history as reader inputs and
distinct checkpoint buffers as writer outputs. T16, 5,120 channels and the
8,240-wide packed projection are deliberately bounded.

Host tests check every row and checkpoint against serial shift semantics and
reject unsupported coordinates/source boundaries. The actual pinned reader
passes the source transformation. The explicit descriptor retains native
81-core work partitioning, CB formats/counts, HiFi4 FP32 arithmetic and writer;
only an additional private 8 KiB scratch CB is introduced. Four native source
hashes, including the original program factory, are pinned before construction.
The simulator probe compares all seven outputs on both chips, poisons captured
outputs before changed-input replay, and checks all eleven inputs remain intact.
Fresh frozen staging and six host tests pass. Next: execute the simulator gate.
Only after that gate may it enter the matched combined hardware test. This
roughly 3.49-ms opportunity alone cannot close the full gap to 200 TG.
