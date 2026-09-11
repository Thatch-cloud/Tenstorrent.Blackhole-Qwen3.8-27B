# Read native GDN state during verification

Status: combined hardware correctness passed; no end-to-end throughput improvement. Not promoted.

## Combined hardware result

[Run 34552817569](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34552817569),
revision `f4db588c85d3a99d08c8d141c6bb50ec52f11c67`, passed. Independent analysis
validated 737 source files, exact target outputs/state/inactive slots, learned
feature and proposal audits, all 48 direct-state layers and paired publication
prefixes, restored hooks, and matched A/B/B/A request timings.

| One stream, merge-intervals prompt | PP tok/s | CTX | Committed TG tok/s | Mean setup-inclusive ms |
| --- | ---: | ---: | ---: | ---: |
| Combined control | 3299.53 | 4096 | 99.73 | 5915.29 |
| Direct native GDN state | 3317.76 | 4096 | 99.42 | 6991.65 |

Each arm committed 242 timed tokens, with identical 222/330 draft acceptance.
TG changed -0.312%; this is not an improvement. Candidate decode samples were
1216.01/1218.22 ms versus control 1227.58/1199.06 ms. No held-out quality
certification or default change is implied.

| Mean per block, 22 timed blocks per arm | Control ms | Direct ms |
| --- | ---: | ---: |
| Draft | 27.988 | 28.494 |
| Verify and readback | 69.231 | 68.494 |
| Select and commit | 11.912 | 12.364 |
| Complete cycle | 110.259 | 110.603 |

The measured verifier saving is about 0.737 ms/block, not tens of milliseconds.
Setup is worse because the candidate adds explicit zero-publication preparation.
Stop treating state-copy elimination as the main route to 200 TG. At 11 committed
tokens/block, the whole cycle needs to fall below 55 ms; projection/recurrence
work or accepted tokens per unit verification cost must change materially.

Hardware report SHA256:
`3b286eabd0668e40bdae7d22f808bb66fdea5a8ab14a8940cf1c36456c22db35`.

## Earlier simulator admission

The full projected GDN composition has now also passed TTsim: 84 output/history
comparisons and 144 input-preservation comparisons, including fresh eager controls
and three refreshed captured replays on both chips. Exit 0, clean close, and
unchanged current source hashes. This includes convolution/gates, recurrence,
normalization, and every retained prefix; it still uses synthetic inputs, not a
learned complete coding request.

Report `/opt/ttsim/results/20260911T013306Z-409-gdn-native-projected-probe.json`,
SHA256 `ebfb378a45327665675aee5aabef910c506156f83aed7d8eeb5d94e0bc4a5070`.
The projected-composition and zero-publication reports are retained under
`scripts/ci/gdn-native-{projected,zero}-simulator.json` for independent CI admission.
Next is the reversible verifier/publication scope and matched complete-request
hardware comparison, keeping all other combined-runtime choices identical.

## Simulator result

The source-corrected run completed cleanly on 11 September, exit status 0.
All 42 output comparisons and 56 input-preservation comparisons passed on both
chips. Captured compact and native-slot outputs were each compared with a fresh
eager control for three refreshed seeds. Checks include every retained prefix,
the FP32 bridge, and all eight native state slots, with seven inactive slots
poisoned. Current source hashes match the report before and after execution.

Report: `/opt/ttsim/results/20260911T010024Z-391-gdn-native-slot-probe.json`.
SHA256: `1457280642d502e75ad3ea2b633272bb55f9b2f05a8619d03033153137e61bac`.
The preceding attempt stopped before recurrence execution because the simulator
checkout lacked the pinned GDN source; the corrected run uses the hash-checked
source snapshot already used by earlier GDN simulator tests.

This validates the recurrence input-layout hypothesis, not complete GDN integration.
Convolution-window native-slot admission, retained publication, learned request
correctness, and combined runtime speed remain untested. Host readback dominates
the roughly fourteen-minute diagnostic; it is not an accelerator timing result.

The combined verifier still calls `active.save(entry)` in each GDN layer before
projection. The proposed replacement reads native slot zero directly and keeps
all candidate prefix states separate until the acceptance decision. This does not
change recurrence arithmetic, precision, or serving defaults.

`gdn_native_slot_recurrence.py` accepts only T16 on a two-chip mesh with native
BF16 recurrent state `[8,24,128,128]`. It reuses the existing 96-worker recurrence
and batched whole-head normalization kernel generators. The initial-state reader
addresses the first 384 tiles; those are slot zero in the native recurrent layout.
The output and retained prefix allocations remain independent. No input copy or
new device fence is inserted. These are source-level properties, not device proof.

## Required before combined testing

Native convolution windows also passed a separate real TTsim comparison: 64
window checks against independent CPU indexing and 72 immutable-input checks,
including all inactive rows and refreshed captured inputs on both chips.
Report `/opt/ttsim/results/20260911T011906Z-431-gdn-native-windows-probe.json`, SHA256
`d2b309cfa5e3374b4846b32cc62912b1f9c20dea26f30298c841e512d7a16776`, exit 0,
current sources unchanged.

The complete projected GDN composition now exists in
`gdn_native_slot_projected.py`, but is not selected by the verifier.
`gdn_native_slot_publication.py` preserves prefix-zero behavior by refreshing the
compact entry *inside the zero-prefix publication callback* before the original
DMA operation. Positive prefixes skip this refresh. Simply removing the entry
copy without this change would leave zero-prefix rollback reading stale storage.
Ten host tests cover geometry, composition, cleanup, and publication ordering;
the combined composition and publication still need device validation.

Zero-prefix publication subsequently passed 120 logical-tensor checks in TTsim,
one layer on both chips: stale entries, eager execution, and refreshed captured
replays. All histories, native state, refreshed entries, and checkpoints matched
the independent expectation. Exit 0 and source digests unchanged; padding was not
audited by this particular probe. Report
`/opt/ttsim/results/20260911T012509Z-409-gdn-native-zero-probe.json`, SHA256
`9c662740c8396a41ab1e25e0656bd9034c39ce4af9bd664c5c910aecbeb42660`.

`NativeSlotState` now provides an unselected T16 adapter. It refuses execution
without explicit paired-publication binding, skips the per-layer snapshot only
for commit-only T16, and keeps the existing adapter for other widths. This is not
yet wired into CI or serving. Complete composition device testing and matched
full-request audits remain necessary.

- TTsim: compare compact-copy control against native-slot candidate, including
  exact outputs, every retained prefix, and FP32 bridge on both chips.
- Poison all seven inactive slots and change active slot zero between captured
  replays. Verify full native state remains bit-exact and addresses stay stable.
- Retain native-state buffers through both graphs; verify cleanup never frees
  borrowed native storage, including failure paths.
- Extend convolution-window input admission to native `[1,8,5120]` history only
  after proving row-zero addressing and full inactive-row preservation.
- Integrate only into commit-only T16 verification, then repeat learned audits
  and matched combined PP / CTX / TG. Keep banked drafting disabled in both arms.

Three host geometry/admission tests pass. They do not run the recurrence kernel.
No throughput saving is claimed: the latest complete verifier remains about
69 ms/block, and the profile's generic labels do not identify the copy's exact cost.
