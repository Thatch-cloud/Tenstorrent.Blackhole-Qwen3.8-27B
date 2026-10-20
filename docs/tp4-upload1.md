# tp4/upload-1: returning sessions back from the host tier faster

Status: **built and CPU-tested; no card has run it.** Branch `tp4/upload-1`, off `tp4/next-1` (the ship-5 stack) with `tp4/fabric-upload` merged in (its upload probes and measurements). Every lever is default off and
byte-for-byte today's behaviour when off. Labels: **M** measured (source named), **C** computed from code or config, **I** inferred, **ESTIMATE** a prediction nothing has measured.

## 0. What the ship-5 profile already carries (read this first)

The upload work the task list names is mostly shipped. These are set in the production profile `c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-fx-traffic-s5` (M, `scripts/ci/qwen_c2_profiles.json`):

| Lever | Switch in the profile | What it is |
| --- | --- | --- |
| KV pool and buffer pool zeros filled on the card (P0, `qwen_device_zeros`) | `QWEN_FAST_DEVICE_ZEROS=1` | about 14 GB a card no longer crosses PCIe at engine start |
| No host transpose on a weight-cache hit (P1a, `qwen_lazy_shard`) | `QWEN_FAST_LAZY_SHARD_W=1` | engine start |
| GDN checkpoints stored in the device's layout (P1b) | `QWEN_PREFIX_CKPT_PRECONVERTED=1` | a restore is 96 copies, no host conversion |
| Sliced host-tier restore (NP5) | `QWEN_PREFIX_TIER_RESTORE_SLICE=512`, `_MS=200` | a returning session's blocks come back a slice per engine step |
| Host tier 48 GiB beside the 16 GiB checkpoint store | `QWEN_PREFIX_HOST_TIER_GIB=48`, `QWEN_PREFIX_STORE_GIB=16` | 32 GiB of KV pages, 16 GiB of checkpoints |

Engagement on cards is **I**, not read from a log by me: `c2_smoke_check.upload_p0_problems` fails an arm that sets a lever and logs no engaged line, and the ship-5 gates passed. The audited twin
`...-fx-s5-audit` also sets `QWEN_FAST_DEVICE_ZEROS_AUDIT=1`. So this branch does **not** redo P0, P1a or P1b; it adds what the host tier's restore and spill still lack (section 2) and the card plan that
settles it (section 5).

**The "8 s to 0.4 s" estimate in the task brief does not describe the KV restore.** It divides a 254k session's 8.8 GB by the GDN checkpoint's old 1.15 GB/s, a rate set by single-threaded host conversion
(`docs/tp4-fabric-upload.md` 6.1), which P1b removed. The KV pages move as raw bytes through the `qwen_kv_read` extension (no conversion) and **T3 measured the 254k restore at 3,841 blocks, 8.56 GB in 0.96 s
(8.9 GB/s), plus 0.14 s of displaced blocks spilled** (M, `docs/prefix-store-hygiene.md`). The returning hit's first token was 6.49 s (device-resident hit 4.65 s, cold 96 s). The restore is about 1 s of that, so even a
perfect upload saves under 1 s of a 5.4 to 6.5 s first token (I); the rest is the tail prefill and the engine build (4.2 to 4.8 s, M, T3), which are not uploads.

## 1. Where the restore's second goes

Per block (2,228,224 B: 32 tensors x 4 chips x 17,408 B, C) the restore does, one after the other on one thread: a host copy from the tier's slab into a transfer buffer (numpy, one `[4, 17408]` copy per tensor and
block), one `ttnn.qwen_write_blocks_raw` per tensor and 512 blocks (8.9 MB a card, 22.8 GB/s at Q1, M), and for one block in sixteen a sha256 of the record (1.5 to 2.5 GB/s). The serial sum is what T3 saw, and the floor
is the device write plus the host cap: 8.84 GB at 22.8 GB/s is 0.39 s, at the four-card host aggregate of 23.9 GB/s 0.37 s (C). The spill is the same in reverse (device read at 13.6 GB/s, then a copy out).
Both raw ops release the GIL (`optimisation/ttnn-op/kv_region_read/qwen_kv_read.cpp`), and so do numpy's large copies and hashlib, so threads can overlap them.

## 2. What this branch adds (all default off, all need `QWEN_PREFIX_HOST_TIER_GIB`)

| Switch | Values | Effect |
| --- | --- | --- |
| `QWEN_PREFIX_TIER_IO_THREADS` | 1..16, 0 or unset: off | `qwen_tier_upload.ThreadedTierIO` wraps the model graft's tier IO (`PrefixRegistry.attach_tier_io`): the staging copy of the next device call runs on the workers while the current device call runs. Device calls stay one at a time, in the adapter's order, with buffers of the adapter's shape; payload bytes, errors and argument checks are the adapter's. Restore and spill both. |
| `QWEN_PREFIX_TIER_IO_CHUNK_BLOCKS` | 16..4096, needs the threads | blocks per device call (default the adapter's 512 = 8.9 MB a card). The knob for the host-cap sweep: F1 saw one x16 card fall from ~35 to 20.5 GB/s above 32 MiB a call. 2048 blocks is 35.6 MB a card. |
| `QWEN_PREFIX_TIER_VERIFY_THREADS` | 1..16, 0 or unset: off | the digest checks of a restore's records (`TierHooks.digest_failures`) on worker threads. `TierStore.verify` is a pure read, so the verdicts, the sampling count, the counters and the record dropped on a failure are the sequential loop's (held by tests). |

**Never worse than stock:** a failure inside the threaded path (a worker's error, a device refusal) latches it off for the process (`host tier upload latched off: <call>: <error>`, counter `tier_io_latched`), the same call is
repeated through the adapter's own serial code, and every later call is the adapter's. A device that really refuses fails the repeat too and the caller sees that error as before; the adapter's argument refusals
come first and never latch. The gate FAILS an arm that latched.

Strict parsing (digits only, in range), refused at start without the tier. The contract admits them on a gate-only profile or a traffic profile whose waiver names the tier (as the sliced restore), and drops an
inherited value under a profile that never named them. Off, `qwen_tier_upload` is not imported and the registry adds no counter. On, the registry adds `tier_io_{write,read}_{calls,blocks,ms,device_ms,stage_ms}`
and `tier_verify_{batches,blocks,ms}`: wall time against the time inside device calls and the staging work, so a log shows the overlap. Log lines: `host tier upload on io_threads=.. io_chunk_blocks=..
verify_threads=.. io_wrapped=..` and `host tier upload fell back: ..` (smoke rows in `fusion-wp/UP1.json`).

Memory: three transfer buffers of `chips x chunk x 17,408` B, kept between calls: 107 MB at the default call size, 427 MB at 2048 blocks (C). The checkpoint store and the KV share are unchanged.

### Checkpoint store budget (16 GiB stays; `qwen_tier_upload.checkpoint_bytes`, pinned by tests)

| Checkpoint | Bytes | Fit in 8 GiB | Fit in 16 GiB |
| --- | ---: | ---: | ---: |
| unconverted, bf16 state (before P1b) | 78,446,592 | 109 | 219 |
| preconverted, bf16 state (the s5 setting) | 106,954,752 | 80 | **160** |
| unconverted plus the exact-resume drafter window (up to 104,857,600 B) | 183,304,192 | 46 | 93 |
| preconverted plus the window (the s5 setting, worst case) | 211,812,352 | 40 | **81** |

The 28.5 MB of padding is the conv carry's 3 rows becoming 32 tile rows; the state (75.5 MB) is already whole tiles. The KV share of the 48 GiB tier is 32 GiB: 15,420 blocks, 987k tokens, 3.9 sessions of 254k (C).
Keeping the conv carry as a torch tensor (a "state only" preconversion) would put 219 checkpoints back in 16 GiB (and 93 with the window) and convert 48 padded carries at each restore: 31.5 MB at F1's 1.2 to 3 GB/s
conversion rate, **ESTIMATE +10 to +26 ms on a restore of about 6 ms**. It is **not built**: nothing shows the store's capacity binds (`qwen_prefix_class_ckpt_evicted_budget_total` and `lost_tokens_ckpt_evicted_total`
would), and a compact carry (3 rows of each tile's two top faces) needs ttnn host tensors built from raw tile bytes, which no CPU test can prove.

## 3. Effects

**Measured on the CPU host (M, `scripts/ci/bench_prefix_tier.py --blocks 1024 --threads N`)**: the adapter and `ThreadedTierIO` over the real staging copies, the device **simulated** as a sleep at Q1's rates (22.8 GB/s
write, 13.6 GB/s read); a shared 20-core host at load 8 to 9, three runs per row, so read these as ranges, not as the rig's numbers.

| Threads | write, slab to device (adapter 2.8 to 4.9 GB/s) | read, device to slab (adapter 4.4 to 6.9 GB/s) | digest check (sequential 1.5 to 2.2 GB/s) |
| ---: | --- | --- | --- |
| 2 | 5.8 to 14.4 GB/s (x2.0 to 3.0) | 11.8 to 12.2 GB/s (x1.8 to 2.5) | 3.3 to 4.5 GB/s (x2.0 to 2.9) |
| 4 | 8.9 to 10.0 GB/s (x2.9 to 3.4) | 9.6 to 10.6 GB/s (x1.5 to 2.4) | 5.8 to 7.4 GB/s (x3.7 to 4.5) |
| 8 | 6.4 to 10.3 GB/s (x1.6 to 2.8) | 7.5 to 9.0 GB/s (x1.6 to 1.9) | 11.7 to 13.0 GB/s (x6.9 to 8.5) |

Threads help the staging copy up to the host's memory bandwidth (here about 4); the digest scales to 8. The same table says where the time stays: with the copy parallel, the write is bounded by the staging copy on this
host, not by the simulated device.

**ESTIMATE for the cards (nothing has run):** the 254k restore falls from 0.96 s toward the device-write time, **0.4 to 0.6 s**; the spill of displaced blocks by a similar fraction; the sampled digest check (0.37 s of
a 254k restore at 1.5 GB/s, C) to under 0.1 s. The returning hit's first token falls by that much and no more: **at most about 0.5 s of 5.4 to 6.5 s (under 10%)**. With the sliced restore on, a slice is the time budget over the measured cost per block, at most 512 blocks: at the model's 0.53 ms a block for restore plus spill (`scripts/ci/tier_slice_model.py`) about 377 blocks, 10 slices
for 3,841 blocks; at roughly half that cost the 512-block cap binds (8 slices). A larger `QWEN_PREFIX_TIER_RESTORE_SLICE` is the owner's follow-up knob and is not in this pack.

## 4. The 23.9 GB/s cap

F1 (run 38036227448, M): four cards written together reach 23.9 GB/s against 68.6 GB/s for their alone rates added up; one x16 card falls from ~35 GB/s at 16 MiB a call to 20.5 above 32 MiB (the pinned zero-copy
path, with the IOMMU on); the x4 cards run at 88% of their link. Unknown, and what the re-run decides (arms already on the branch, `tp4_h2d_probe.py`): whether smaller calls lift the aggregate (`chunked/together`,
`chunked/alone`), whether one Python thread per card does (`threads`, `threads_busy`). The mesh command queue writes each card's shard from its own thread (`docs/tp4-fabric-upload.md` 1.3, read from the code; I for these ops), so the restore's calls are probably already one-thread-per-card
inside C++, and under 32 MiB a card; `QWEN_PREFIX_TIER_IO_CHUNK_BLOCKS` lets U7 (128 blocks) and U8 (2048, above the threshold, the negative control) test the same question on the real restore without a rebuild. A
C++ change (a per-chip write op) is **not** made: it would move the pinned extension hash and no compile is possible on a CPU host.

## 5. Card gates (`scripts/ci/references/upload-1-jobs`, one image `tp4-upload-1-1` as a placeholder tag, `ORDER.txt` has the rules)

The gate derives the levers onto the s5 profiles (`c2_prefix_gate.DERIVED`: `upload1`, `uptime`, `uptime8`, `uptimec128`, `uptimec2048`), no profile twin. The scenario is the tier-timed cycle at 64k, 128k, 254k:
cold prefill, device-resident hit, floods the size of the KV pool, the returning hit, its cold twin (`prefix_replay.scenario_tier_upload`).

| Job | Plan on profile | Question | Minutes (ESTIMATE) |
| --- | --- | --- | ---: |
| B0 | build, bakes the s5 profile | the image | 40 (no card) |
| X0 | status, rescan, reset | cards and grid | 20 |
| F1 | `fabric`, `C2_FABRIC_PROBE=h2d` | where the 23.9 GB/s cap is | 30 |
| **U1** (stop) | `tier-upload` on `...-fx-s5-audit` | byte-exact: 64k/128k/254k restored (every block read back and compared, every digest checked) against the cold twin, through the threaded IO | 100 |
| U2 to U5 | `tier-upload-timed-control`, `tier-upload-timed`, repeated | timed restore ABAB, ARC at zero | 45 each |
| U6 to U8 (optional) | `-8`, `-c128`, `-c2048` | thread count and call size | 45 each |
| Z | status, reset | end | 10 |

Total 380 minutes without the sweeps (6 h 20 min, X0 to Z about 340 of it on cards), 515 with. The cut line: X0 U1 U2 U3 (about 210 minutes) already answers both halves.

**Registered rule** (`upload_restore_compare.py <U2> <U3> <U4> <U5>`): GO needs, in both pairs, every returning answer IDENTICAL to the control's, the restore's ms at 128k and 254k lower by at least 25%, and the
returning hit's TTFT at no size above the control's by more than the larger of 5% and the control-to-control floor. NO-GO: a differing answer, an audit mismatch or digest failure, a fell-back line, a restore that
did not go through the threaded IO (`tier_io_write_blocks` 0), a restore not faster by the bar. U1 is the exactness gate; the timed pair cannot overrule it.

## 6. Risks

* Staging threads add host CPU during a restore; the seats' decode and the tokenizer share the host. The concurrent-seat stall (`tier-concurrent-timed-sliced`) is **not** in this pack; run it ABAB before shipping.
* The wrapper takes the raw ops from the namespace of the adapter's own methods (the model module's `ttnn`). If the threaded path ever fails it latches itself off and the adapter repeats the call (above), so the cost of
  a defect is one repeated call and a gate FAIL (`tier_io_latched`), not a failed restore.
* A worker error is seen only after every worker has stopped; no payload or buffer is touched after a call returns (held by tests).
* The levers are not in any profile: shipping is a ship-entry line (`fusion-wp/S5.json`'s successor) plus the waiver text naming them; the contract needs no change.
* CPU numbers are from a loaded shared host; the rig's memory bandwidth decides how far the threads go.
