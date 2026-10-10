# Host-gap levers (tp4/fx-wph, work package H of the op-fusion programme)

Three default-off, host-only levers and one instrument for the eight-live round. Nothing here touches a kernel, a trace or a token: the levers change which host bytes are
written when, which address reads are made and how many reads fetch the same bytes, and each has an `_AUDIT` twin that proves it on the device.

| Flag | What it does | Files |
| --- | --- | --- |
| `QWEN_FAST_PRESTAGE_DIFF` (`_AUDIT`) | the drafts' window pre-stages only the destinations whose bytes differ from the ones last written | `prestage_diff.py`; hooks in `verify_prestage.BlockPrestage` |
| `QWEN_FAST_WRITE_PACKED_LEAN` (`_AUDIT`) | `write_packed` without the per-value mapper and the before/after address reads: one mapper, addresses captured at first sight, checked once a round | `write_packed_lean.py`; `BlockPrestage.writer` |
| `QWEN_FAST_BATCHED_READS=1\|async` (`_AUDIT`) | the verify readback and the quad collect as one mesh-composed read a tensor (`1`) or non-blocking device-to-host copies and one fence (`async`) | `batched_reads_tp.py`; hooks in `packed_verifier.shard_predictions` and `quad_draft_tp.read_quad_outputs` |
| `QWEN_FAST_TP4_HOSTGAP_LOG` + `QWEN_FAST_TP4_HOSTGAP_PROBE` | spans with wall against thread CPU, context switches, collections and the host calls; a timed synchronize after the two quad launches every 16th eight-live round | `hostgap_instr.py`, `hostgap_read.py`; spans in `verify_prestage`, `packed_verifier`, the coordinator, `serving_worker_hook` |

The manifest is `scripts/ci/fusion-wp/WPH.json`; the jobs are `scripts/ci/references/fusion-jobs/WPH/` (`ORDER.txt`), the one-card read probe is
`optimisation/ttnn-op/batched_reads/`, and `hostgap_wph_smoke.py` is the smoke rule.

## Why: what was measured and what is estimated

From the host-gap analysis of the unprofiled eight-live runs (control, then the combined fusion arm "fx-all"; milliseconds, medians):

| Quantity | Control | fx-all | Source |
| --- | ---: | ---: | --- |
| device idle in a 150 ms round, seven host turnarounds | 16.5-18.6 | | measured |
| the drafts' window (two blocks' validate, `packed_values` and 148 writes each) | 17.96 | 18.14 | measured (`[PACKED-PRESTAGE-WINDOW]`) |
| window end to the return of the fence F9 | 9.40 | **0.39** | measured: with the fusion levers the window is the critical path |
| drafter saving the window eats | | 2.4-4.4 | integrator's estimate from the two rows above |
| verify readback, per block (8 blocking reads and the combine) | 0.85 | 0.87 | measured |
| collect (40 blocking reads and the merge, device idle) | 4.54 | 4.28 | measured |
| verify-time write of the tokens alone, per block | 0.54 | 0.61 | measured: a fixed per-call cost, about ten times a pre-stage destination's share |
| rounds with a quad launch of 5.8-6.4 ms instead of about 2, and writes 2-5 times slower | 22 % | 22 % | measured, cause unknown: the instrument is for this |

Only about 73 of the 148 pre-staged buffers change from one round to the next (the positions, the rotary tables, the 64 per-row positions, the tile positions and the readers'
words); the page-derived ones (the page table, the 64 per-row tables, the tile tables, the readers' per-bundle tables) change only at a page crossing.

| Lever | Estimate per 8-live round | Basis |
| --- | --- | --- |
| PRESTAGE_DIFF | window -3 to -4 ms a block, but only 2.4-4.4 ms comes back | 73 of 148 writes at about 45-60 us each; the round gains only what the window overshoots the quads by |
| WRITE_PACKED_LEAN | window -1 to -2.5 ms more, inside the same 2.4-4.4 ms | one address read and one mapper fewer per written destination, the verify-time write loses its address reads |
| BATCHED_READS | -1.0 to -1.2 (verify, 2 x 8 reads to 2 x 2) and -2.0 to -2.8 (collect, 40 reads to 10) | the fixed cost of a blocking read is about 80-100 us; both are device-idle time |
| together | about -5.5 to -8 ms (3.7-5.3 % of 150 ms) | **estimates, none measured on a card** |

These are estimates from the measured bases. The card-M probe (H1), the audited attaches (H2-H5) and the ABAB (H6-H9) replace them; the instrumented run (H10) says what the rest of the
gap is made of. A window that shrinks below the quads' device time stops paying: the first two levers together can shrink it by more than the 2.4-4.4 ms it overshoots by, and the
margin is then slack for the next drafter fusions.

## Exactness

* **PRESTAGE_DIFF.** After the pre-stage every destination holds `from_torch(value)` of the value the full pre-stage would have written: a written destination by the same call, a
  skipped one because its resident value is bit-equal (`same_bits`: dtype, shape and every bit, so +0 is not -0) and `from_torch` is a pure function of the host bits. The one invariant
  is that nothing wrote the destination in between, which the pre-stage already stands on: every other writer of a fixture's inputs moves the write epoch (global, and the fixture's own
  with the per-block epochs) and a resident whose epochs moved is not used; the block's own writes record themselves (the pre-stage its snapshot, the verify-time diff the snapshot with the
  changed destinations replaced, the keyed write the snapshot, a full `stage_packed` nothing). The device traces read the staging buffers and name none as an output: that claim is proved
  on the card, round by round, by the audit twin (every pre-staged destination read back from every chip behind the traces and the writes: a mismatch repairs the round and latches the
  lever off) and independently by the existing `QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE_AUDIT` read-back at the verify (zero mismatches on `path=diff`). It composes with KEYED (same staging path,
  the verify-time half), the lean writer and the per-block epochs (without them the other block's verify kills the record and every pre-stage is the full one).
* **WRITE_PACKED_LEAN.** The uploads, copies, order and fence are the original's argument for argument; what goes is a defensive check and a mapper object that is rebuilt (the mapper is a
  stateless placement description). The audit twin runs the original's before and after reads beside the book's and reads back a rotating eight written destinations.
* **BATCHED_READS.** Reading is not computing. The composed read is one `from_device` of the mesh tensor (the pinned build, tt-metal 9f9cd4fd) followed by a concatenation along dim 0 in
  chip order, so chip *c*'s piece is shard *c*'s tensor; `async` is `copy_device_to_host_tensor(blocking=False)` into host tensors allocated once, one `synchronize_device`, the same
  composition. The combine and merge code receives the per-chip tensors it received before. The audit twin reads the served way beside the batched way (the first 16 reads, then every 16th)
  and compares bit for bit; a difference returns the served bytes and latches the lever off.
* **Kill switches.** Every lever latches off for the process on a difference or an exception in its own bookkeeping and logs `... fell back`; the smoke rule fails an arm that did.

## The instrument

`hostgap_instr.py` writes, under `QWEN_FAST_TP4_HOSTGAP_LOG` (new lines only): one `[PACKED-HOSTGAP-ROUND]` a step and `[PACKED-HOSTGAP-SPAN]` lines for each block's pre-stage, the
window, the fence F9, the two quad launches, the collect, each verify's stage and readback, and the phases. Each carries the wall and thread CPU time, voluntary and involuntary context
switches, page faults, every garbage collection, and per host call (copy, upload, trace enqueue, blocking trace, fence, read) the count, total and **longest single** call: a copy that
blocked on a full command queue shows as one long call with the CPU idle; a descheduled thread shows involuntary switches. With `QWEN_FAST_TP4_HOSTGAP_PROBE=1` every 16th eight-live round
the coordinator does one timed `synchronize_device` right after the two quad launches (`[PACKED-HOSTGAP-PROBE]`: 2Q directly); that round runs its window after the quads and is not a round time
(`w2ln_timing_compare.timed_rounds` and `hostgap_read.py` drop it). `python3 scripts/ci/hostgap_read.py <container log>` prints the tables and classifies the slow spans by cause.

## Card plan

`ORDER.txt`: H1 the one-card read probe (decides `1` or `async` or neither), H10 the instrumented unprofiled run on fx-all, H2-H5 the audited attaches on fx-all plus each lever (and all three),
then the ABAB H6-H9 (control fx-all, lever fx-all plus the three), one tag at a time. The per-lever arms against the production profile are generated into the integrated pack
(`PRESTAGEDIFF*`, `LEAN*`, `READS*`).

## What the first instrumented run showed (H10: fx-all plus the instrument, ARC running, 4.0k prompts)

Read with `hostgap_read.py` (eight-live steps apart from the one-block rounds; the instrument adds about 1 us per wrapped host call and a log line per span, so absolute times are 1-2 ms a round high):

| Eight-live step (both blocks pre-staged) | p50 ms |
| --- | ---: |
| launch of the two quads | 4.33 |
| window (two pre-stages 10.0 + 9.5, T_proj staging about 1.3, glue) | 22.25 |
| fence after the window | 0.05 |
| the probe: first enqueue after the launch start, then 2Q | 2.25, 18.44 |
| host path (launch + window + fence) against device (first enqueue + 2Q) | 26.6 against 20.7 |
| **host time on the critical path** | **5.9** |
| collect (34 reads = 1.17 ms; the rest, 2.3 ms, is merge and selection on the CPU) | 3.47 |
| readback of one block (8 reads = 0.45 ms) | 0.83 |

* The 11.6 ms window median of the whole run is 78 % one-block steps (window 10.8 ms against a single quad). For eight-live steps the window is critical in 91 % of the rounds and exceeds the device's remaining
  time by about 6 ms: that, not 2.4-4.4 ms, is what PRESTAGE_DIFF and WRITE_PACKED_LEAN can return, minus the drain of the writes still queued behind the quads when the device finishes.
* A pre-stage is 10.0 ms wall and 9.1 ms CPU: only 3.2 ms of it is inside `copy_host_to_device_tensor` (148 x 11.6 us, one voluntary context switch each) and `from_torch` (148 x 10.4 us); about 6.7 ms
  is Python and binding work around them (the 296 address reads, 148 mappers, `packed_values`, the binding validation). With 73 of 148 destinations written the pre-stage falls to about 5.4 ms (the
  comparison itself costs 0.5 ms a block with memcmp), so the two window levers together remove about 7 ms a round, more than the 5.9 ms on the path.
* The read-backs are cheaper than estimated: 34-56 us a read, so BATCHED_READS is worth about 0.9-1.2 ms a round, not 3. The collect is CPU bound (3.1 of 3.5 ms).
* Scheduling: 42 % of the steps had more than 20 involuntary switches; on eight-live steps the contended ones cost +5.8 ms in the early draft (41.7 against 35.9) and +4.3 ms in the window at the median. The
  stalls are bimodal (0.08 ms or 43 ms a switch) and not aligned to a 100 ms period. The cause is not decided by that run; the next one records the cgroup throttle counters, a thread census and each
  blocked copy.
* The 491 "blocked-copy" fences of the first reader were fences: a fence waits for the device by design (the queue ahead of it holds the quad traces and, behind them, the window's writes). The
  reader now calls those `device-wait`. The copies that really block are single calls of 2.3-3.7 ms with the CPU idle (0.09 ms), 16 % of the T_proj staging spans, 24 % of the quad launches, 10 % of
  the pre-stages, 5 % of the verify-time writes; in most of them the thread was not descheduled.
* Placement: card jobs start the smoke container as `--cpus 8` with no pinning and docker's default weight, while the production container has no quota (unless THATCH_CONTAINER_CPUS is set) and is
  given cores 0-15,32-47 and the maximum weight by ci-docker-fence.service. `scripts/ci/fusion-wp/WPH-cpus.patch` (not applied) adds C2_CPUS / C2_CPUSET / C2_CPU_SHARES to the four-card smoke and the
  two jobs H11a (today's placement) and H11b (production-like), both with the instrument.
