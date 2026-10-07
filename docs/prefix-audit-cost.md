# The prefix audit's cost: reading the blocks a request names, not the pool

**Branch:** `tp4/audit-cost` (on `tp4/stage1-windows`). **Status:** the Python side is built and CPU-tested. The device-side region read
(`optimisation/sim/kv-region-read.patch`) is a DRAFT against the pinned tt-metal tree: it has been neither compiled nor run, and until it is, the
audit's cost is the old one. Nothing here ran on cards.

## What the finding was

`QWEN_PREFIX_AUDIT=1` (the exactness arms' instrument, `qwen_prefix_model_patch._qwen_prefix_audit`) compares, per 2,048-token window, the KV a prefix hit
reads with its salted cold twin's. W-0 measured the engine silent for about 8.4 minutes before every request, then serving it in seconds, so
`exactness-shared` (48 requests at eight seats) could not finish in its 10,800 s box and P1 never produced a verdict on cards. The 8/8 byte-exact hits read from
the log before the timeout (prompts of 4.6k to 6.4k tokens) are the only exactness evidence so far.

## Where the 8.4 minutes go (from code and sizes)

The audit called `ttnn.to_torch` on every full-attention layer's K and V paged cache, whole, one tensor at a time, and only then selected the row's blocks on the host.

| | Figure | Basis |
|---|---|---|
| Tensors read per request | 32 (16 full-attention layers x K and V) | `docs/tp4-drafter-bf16.md`: 557,056 B of KV per block per chip = 32 x 17,408 |
| One block, one chip, one tensor | 17,408 B (64 tokens x 256 head dim x 1 KV head at bfloat8_b, 1,088 B per 32x32 tile) | `docs/batch-spec-tasks-2026-09-19.md:2004` (34,816 B per page at two heads) |
| The pool at 8 x 262k | 19,968 blocks = 1.28 M tokens | `docs/tp4-combined-window.md` section 2 |
| Raw bytes moved off the four chips | 19,968 x 17,408 x 32 x 4 = 44.5 GB | |
| The same after the host unpacks to float32 | 19,968 x 4 heads x 64 x 256 x 4 B x 32 = 167 GB | what `to_torch` returns, then `ConcatMeshToTensor` copies and `index_select` reads |
| Measured | 504 s per request | W-0 P1a, 0.33 GB/s of float32 produced |

Nothing in that cost depends on the prompt: a 4.6k-token request read the same 167 GB as a 120k one, and a step with eight rows paid it eight times (one `_qwen_prefix_audit` call per row).
The split between the device read (44.5 GB packed) and the host conversion (167 GB unpacked, then copied) was not measured; at 0.33 GB/s overall the conversion and copies dominate unless the device read is slower than about 0.1 GB/s, which no figure here suggests.

## What was built

1. **A step is audited once** (`_qwen_prefix_audit_rows`). The prefill loop collects its rows and audits them together after the last row, so a whole-cache read happens once per cache per step, not once per row. `_qwen_prefix_audit` (the Lever N route's per-request call) keeps its signature and calls the same code.
2. **Only the blocks a row names are read** (`_qwen_prefix_audit_selections`, `_qwen_prefix_read_blocks`), through `ttnn.qwen_read_blocks`: the named blocks go from the device into a host tensor of just those blocks (`ttnn.allocate_tensor_on_host`, the cache's dtype and layout), and the same `ttnn.to_torch` the whole-cache read used unpacks it. The digests are therefore byte-identical to the old ones, window by window, and the read volume follows the rows' lengths. The read compiles nothing and allocates no device memory (a host tensor and DRAM reads only), so it is program-free after the traces are parked.
3. **Modes** (`QWEN_PREFIX_AUDIT_READ`): `auto` (default) uses the region read when the image has `ttnn.qwen_read_blocks` and reads whole caches otherwise; `region` refuses to run without the graft (an audit that silently reads whole caches costs 8 minutes a step); `full` forces the old read. A region read that raises falls back to the whole-cache read for the step and says so.
4. **One cost line per audited step**, `[PREFIX-AUDIT-COST] rows= reqs= tokens= mode= reads= blocks_read= read_ms= total_ms= [fallback=]`, so a window's log states what the audit read and for how long (it does not match the `[PREFIX-AUDIT] ` rows the judges parse).
5. **CPU tests** (`AuditCost` in `test_qwen_prefix_model_runtime`): the region digests equal the previous algorithm re-run on the same caches (window by window and as the chained `kv_sha`), region and full modes log the same audit lines, the blocks read per tensor equal the row's block count at pools of 4,096 and 8,192 blocks and no whole cache is read, a two-row step reads each cache once, a refused region read falls back with the same digests, `region` without the graft refuses, and the route's single-row entry still works.
6. **The graft** (draft): `optimisation/sim/kv-region-read.patch` adds `tt::tt_metal::enqueue_read_tensor_dim0_slices` (runs of consecutive block ids become `BufferRegion` reads, every device's shard, one wait) beside the pinned tree's `enqueue_read_tensor`, and the binding `ttnn.qwen_read_blocks(device_tensor, host_tensor, blocks)`. `optimisation/ttnn-op/kv_region_read/kv_region_read_card.py` is its card check (byte equality with the whole read for a single block, a run, scattered ids, a shuffled order and two runs; no program-cache growth; the cost per block). The pinned tree already has the pieces (`ShardDataTransfer::region`, `enqueue_read_shards`, `BufferRegion`); the draft only combines them for a multi-device mesh.

## What the graft still has to settle (it was written from the pinned headers, not built)

- It touches `libtt_metal.so` (the new function) and `_ttnn.so` (the binding). The image's graft mechanism (`docker/qwen-c2-serving.Dockerfile`, `optimisation/ttnn-op/k64j/build_k64j.sh`) replaces `_ttnn.so` and `_ttnncpp.so` only, and those hashes are pinned (`QWEN_FAST_RUNTIME_BINARY_SHA256`, the K64j checks). A build must start from the tree that built the served binaries, keep every `QWEN_` string of the served `_ttnncpp.so` (the lesson in `build_k64j.sh`), add `libtt_metal.so` to the copy list and re-pin the hashes, or the read can be moved into its own small extension module that links the same libraries and leaves every pinned binary alone.
- `allocate_tensor_on_host`'s shape is assumed to be the per-device shard shape (what the cache's `.shape` reports); the binding checks the host tensor against the device tensor with `TT_FATAL`, so a wrong assumption fails loudly, never as wrong bytes.
- Several `ShardDataTransfer`s for one shard in one `enqueue_read_shards` is assumed to be allowed; `kv_region_read_card.py` has a two-run case for it.
- A read bug would show as a hit/cold mismatch (a false FAIL), not a false PASS, so the card check must pass before P1ab-LN is read.

## The estimate, and its basis

Per audited request, with L its prompt length: `t(L) = L x (0.39 ms to unpack and copy + 0.09 to 0.26 ms to hash) + reads`. The 0.39 ms is W-0's own rate (504 s over the pool's 1.278 M tokens, every token of it unpacked); the hash term is 131,072 B of float32 per token at 0.5 to 1.5 GB/s of sha256; the reads are 32 tensors per request at an assumed 0.2 ms per block-run call in the worst (fully scattered) case. That gives about 3 s at 5k tokens, 20 s at 32k, 70 s at 120k and 2.5 minutes at 262k, against 504 s for every request before.

`exactness-shared` at eight seats runs 48 requests whose prompts total 608,400 tokens (eight agents, three rounds, each prompt audited as a cold twin and as a hit: 2 x 304,200 from `SHARED_AGENT_TARGETS_8`), so the audit adds about 6 to 8 minutes in all. The old figure of the arm, 150 minutes, priced the audit at about 2 minutes a request (95 of the 150); replacing that with 8 minutes leaves about 60 minutes, and the pack keeps 20 more for the read's cost, which no card has measured.

**P1ab-LN (W-1 pack): 190 -> 120 minutes, box 336 -> 240.** The box is 1.5 times a 160-minute high end instead of the gate's worst case; the prefix gate clips every arm's docker timeout to what the box has left, so a slower audit ends the job as a TIMEBOX, never past 240 (and not after 336 as before). At the central estimates `P1ab-LN` is admitted with 99 minutes to spare (it needed E <= 144, it needs E <= 240) and `E1-LN` with 66, so the W-1 plan is 401 minutes (6.7 h) instead of 471 and the carry-over (and with it a third window for the cutover) is the exception, not the likely outcome. All of this assumes the graft; without it the job does not fit any box and is not worth starting.

**A P1-CTL re-run** (exactness-shared on today's control, `P1a-CTL`, estimate 150, box 183): it can only use the fast audit on an image that carries it, so it is the window-image form (`P1a-CTL2` in the combined pack, Lever N flags off), no longer a check of the production bytes. Estimate 60 minutes; its box can drop to 120 (1.5 x an 80-minute high end) on the same clipping rule. `P1b-CTL` (lifecycle-evict, no audit) is unchanged at 75/153. These are not in the W-1 pack (which holds no P1-CTL); they are the numbers to use if the owner adds the re-run.

Without the graft, the new Python alone still saves what row-sharing saves: a step costs 8.4 minutes however many rows it holds. The arm's roughly 34 steps (16 sequential round-0 pairs, then 8 cold twins and one burst of hits per round for two rounds, if the burst is one step) would take about 4.8 hours: still past the 3 hour limit. A smaller pool for the audited arms (`QWEN36_MAX_TOKENS_ALL_USERS` with a matching `num-gpu-blocks-override`; the arm's peak is under 200k live tokens, so about 6,000 blocks would hold it, 3.3x smaller, about 1.5 hours) is the only graft-free route; it has not been built because the contract's pool arithmetic (`kv_pool_problem`, the reservation and the boot rule's DRAM band) was written for the production pool and a derived profile needs its own review.

## Lifecycle-evict's coverage rule (P1b)

P1b-CTL passed 19/19 concurrent pairs and 15/15 solo pairs and failed only `QWEN_FAST_TP4_RS_UNIT_MAJOR_AUDIT: shape 64x5120 replay audits came from 1 block owner(s), 2 blocks are served`. The rule (`c2_smoke_check.u1_audit_problems`) wants a replay audit (round >= 1) from every block owner (`QWEN_FAST_M3_BLOCKS=2`: seats 0-3 and 4-7). The lifecycle traffic never had more than four users decoding at once, and its abort-while-waiting case (`lifecycle_abort_waiting`) held four seats: at eight seats that leaves four free, so the queued turn is admitted instead of waiting and the second block never decodes. `scenario_lifecycle_evict` now passes `seats=driver.seats` (the served profile's `max-num-seqs`): eight long `ignore_eos` holders keep both packed blocks decoding for 1,500 tokens each, and the turn really queues. Four-seat profiles are unchanged. Test: `test_lifecycle_evict_holds_every_seat_of_an_eight_seat_profile`. The cost is one minute or two of extra decode in an arm that already runs for hours; whether both owners then log a replay audit is a card question.

## Does the baked image need the change?

Yes. The audit is code in the model graft that the image build stages (`qwen_prefix_stage` runs `qwen_prefix_model_patch` against the pinned originals and holds the result to `PATCHED_SHA256`, which changed for `model.py` on this branch). `qwen_prefix_model_patch.py` is already in both bundle copy lists (`docker/qwen-c2-overlay.txt` and the evidence tree), and no new `scripts/ci` file was added, so the provenance check needs no list change; but a `tp4-serve-11` built before this commit holds the old audit, so B0 must be rebuilt from a commit that contains it. The graft's binaries are a second artifact (above) and the one that matters for the cost. `prefix_replay.py` (the P1b fix) and the pack numbers are read from the checkout by the workflow, not from the image.
