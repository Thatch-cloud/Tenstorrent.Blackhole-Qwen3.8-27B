# Read native GDN state during verification

Status: synthetic recurrence simulator comparison passed; no integrated or hardware qualification.

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
