# Read native GDN state during verification

Status: recurrence adapter prototype; not simulator- or hardware-qualified.

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
