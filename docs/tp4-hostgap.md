# tp4/hostgap: the eight-seat host gap (stage 0, 1a-lite, 1a, 1c, 1d)

Base: tp4/262k8, eight seats on two packed 64-row blocks, 262,144-token windows, every lever on. Every behaviour change below is behind a
default-off flag and is byte-identical off (`test_tp4_hostgap.FlagOffIdentityTests`, the existing H1a suite, and the flag-off identity of
the bridge, the page binding and the retained replay).

## What it addresses

The measured eight-live round at a 32k mean context is 222 ms against 178.7 ms of device work. About half of the 43 ms of device idle is the
full verify staging of both blocks at verify time: with two packed blocks the verify pre-stage is off (the fixture write epoch is one counter
per process), so 744 of 788 measured verifies took `path=full reason=no-snapshot`. The remedy moves that host work into the drafts' fence
window, where the host already waits for the two quads. No device command changes order. Every verify sees exactly the inputs it sees today.

## Flags

| Flag | Item | Effect |
|---|---|---|
| `QWEN_FAST_TP4_HOSTGAP_LOG=1` | stage 0 | New log lines only: `[PACKED-ENTRY]`, `[PACKED-HOSTGAP-VERIFY]`, `[PACKED-HOSTGAP-SELECT]`, `[PACKED-HOSTGAP-WINDOW]`, `[PINDIAG] gc` (collections over 5 ms). Thread CPU time sits beside wall time. |
| `QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE=1` | 1a-lite | The block that verifies first is pre-staged in the two-block window, under today's one epoch. Needs `QWEN_FAST_PRESTAGE=1` and two blocks. |
| `QWEN_FAST_TP4_PRESTAGE_BLOCK_EPOCHS=1` | 1a | Needs the flag above. Each block's own writes bump a per-fixture epoch, so both blocks keep a usable snapshot. |
| `QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE_AUDIT=1` | audit | After each verify-time diff write every destination is read back from every chip and compared with the full-stage value; the round is then staged in full anyway. |
| `QWEN_FAST_TP4_WINDOW_VALIDATE=1` | 1c | A verify that took a usable snapshot skips the retained block's second binding check right before its trace. Under the audit that check runs as a shadow. |
| `QWEN_FAST_TP4_ENTRY_DIET=1` | 1d | One storage check per distinct validator a step; incremental page-allocation validation. |

Not built: 1b (keyed diff), 1d's lazy sequential-table write (the review's no-go), 1e (`gc.freeze`), 1f (log diet), stages 2 and 3.

## Exactness, per item

**1a-lite.** Between the first block's pre-stage and its verify, the only epoch bumps are the other block's pre-stage (left out) and the first
block's own diff (which comes after the snapshot was consumed). The first block therefore sees a snapshot taken under an unmoved epoch: the
mechanism four seats already qualified. If the predicted first block falls to the sequential step at the step, the second block's full stage
bumps the epoch and the unused snapshot dies. The second block takes today's full stage.

**1a (block epochs).** A snapshot is usable only while the global epoch and its fixture's own epoch both stand. A block's own writes (its
pre-stage, its verify-time diff, its full `stage_packed`, so also the padded probe's, the extent audit's and the prestage audit's restage) bump
only its own epoch; every external writer (prefill, prefill chunk, admission, detach, bookkeeping, lane switch, a failed verify) still bumps
the global one and kills both. The mode needs the two blocks' staging destinations to be disjoint. That is checked in three places: the attach
(distinct fixtures, replay readers and extent storage), the first pre-stage of each block (chip-local addresses of every destination, before
any bump or copy), and the structure of the code (every destination is allocated before the first capture, `complete_blocks_two_phase`). An
overlap disengages the mode and the global bump returns.

**1c.** It removes a check, not a value. The window ran `validate_bindings` for the epoch, and the epoch vouches that nothing that can move a
native buffer (a prefill, an admission, an engine build) happened since. Under the audit the skipped check still runs and raises as the check
would.

**1d.** The storage validator is one bound method shared by every bridge, called per bridge with no state between the calls; one call per
distinct validator is the same check at the same point. The incremental page validation accepts only an extension of the allocation a full
validation already accepted, after checking the suffix (type, range, uniqueness within and against the held set, capacity); anything else
takes the full validation, so every refusal and its message are today's (`IncrementalPageValidationTests` runs 400 random allocations
against both routes).

## Reading the arms

- Audited attach: zero `[PACKED-PRESTAGE-FULLAUDIT]` mismatches, read apart by path. The audit also reads every destination back after a
  `path=full` stage, where the full stage is the reference itself: a mismatch there is an artifact of the comparator (layout, padding, dtype)
  and voids the diff-path reading; a mismatch on `path=diff` alone is the lever. A `path=diff` line for each pre-staging block, at least one
  `path=full` line, `buffers x chips` checked on every line, `[PACKED-PRESTAGE-SHADOW]` lines.
- At least 95% of the pre-staged blocks' 4-live verifies on `path=diff` (blocks A and B on the epochs arm, A on the lite arm), read from
  `[PACKED-HOSTGAP-VERIFY] block= path= reason=`. Only a full stage after an external writer (`epoch:admission`, `detach`, `prefill`,
  `prefill-chunk`, `bookkeeping`, `lane-switch`) is left out of the count; `no-snapshot`, `epoch:verify` and `destinations` are misses.
- The `[PINDIAG] gc` lines are written from the window and entry lines: the collector callback only records into a bounded list (a collection
  can start inside the logger's own emit).
- The per-block epochs are process-wide: they also make the solo lane's own writes bump only its own epoch, and the attach refusal does not
  look at the solo block. The lane-switch global bumps keep it safe; do not combine `PRESTAGE_BLOCK_EPOCHS` with the solo lane in an arm
  without a refusal for it.
- The arms log a few extra lines and thread-time reads per round that the control does not: a bias against the arms.
- Window overrun: `[PACKED-HOSTGAP-WINDOW]` `window_ms` against `fence_wait_ms`. A `fence_wait_ms` near 0 beside a `window_ms` past the quads'
  device time (38.8 ms; about 25 ms of host slack when the drafter lever D2a lands) means the pre-stages cost the lever device idle.
- Hang shapes: five consecutive completions (the audits-off hang is unexplained and sensitive to the trace shape; stage 1 removes about 20 ms
  of host time between traces).
- Timing, paired ABAB per round: the block-epochs arm -15 ms or better (kill -12), the lite arm -8 ms (kill -5).
- Stage 1 gains nothing during prefill, admission or lane switching: every one of those bumps the global epoch.
