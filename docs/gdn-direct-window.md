# Direct causal windows inside GDN convolution

Status: simulator and combined hardware correctness pass. A verifier reduction
is measured, but unstable control timing requires repetition before promotion.
**No serving change.**

## First combined hardware result

Attempt 2 of the same run also passes correctness, but fails repeatability:
native TG **78.35 / 34.37**, direct **85.06 / 129.41**. Mean selection/commit
times are 55.65 / 256.25 ms for native and 50.43 / 4.93 ms for direct.
Verifier means remain 67.54 / 67.43 ms versus 63.75 / 63.05 ms.
Do not promote the apparent aggregate +114.83% gain. The stalls recur across
both arms; their cause is not established. All samples remain included.
Attempt-2 report SHA-256:
`7a6f0b6a954a2bbb6ee5a59f4b2180e6b18380eff2e4c9e87b3d29d0aeb63614`.
The next experiment composes direct windows with compact draft selection;
it does not silently discard either failed-repeatability result.

Run **35273229698**, revision `b16f936`, passes in about seven minutes.
Both audited requests and four timed requests preserve exact tokens, target
state and inactive slots. Each candidate builds 96 direct-window calls, with
288 native shorter-tail calls and restored bindings. All 870 script, 1,520
native and 11 candidate fingerprints remain unchanged; device/checkpoint close
and process exit succeed. Report SHA-256:
`52e5d31229982dfa926d27b6bf0beb86a3b767f2e85957ec107da456af9d3e21`.

| Timed request order | Committed TG | Mean verifier ms | Mean selection/commit ms |
| --- | ---: | ---: | ---: |
| Native A1 | 25.14 | 66.82 | 387.17 |
| Direct B1 | 124.03 | 63.07 | 5.04 |
| Direct B2 | 128.99 | 63.13 | 5.11 |
| Native A2 | 122.15 | 66.51 | 5.28 |

The candidate aggregate is **126.46 TG**. Native A1 has two whole cycles of
2,582 and 1,316 ms, dominated by selection/commit; its prefill is also slow.
The cause is not established. Do not discard those samples or claim the
misleading aggregate +203% as a kernel speedup. Both candidate verifier means
are around 3.5 ms below controls, consistent with the window-builder target.
The cleaner second pair improves TG 5.60%, but one pair is insufficient.

The independent report now adds a repeatability check: neither arm may differ
by more than 10% between its two timed TG measurements. This is an engineering
screen, not statistical significance. This run fails that screen despite its
two positive paired changes. Next: repeat the same runtime, not a new kernel.

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
Fresh frozen combined staging and twelve host tests pass. The L1 simulator
gate above admits the candidate to the matched combined hardware test.
The new hardware workflow runs two complete correctness audits followed by
unchanged/direct/direct/unchanged timed requests, loading weights once.
Only window construction changes; shorter tails, native math, checkpoint
publication, drafting and the retained T16 recipe remain unchanged.
Both paired complete-cycle TG improvements must exceed 2% to pass the screen;
repeatability is additionally required after the first hardware result above.
This roughly 3.49-ms opportunity alone cannot close the full gap to 200 TG.
