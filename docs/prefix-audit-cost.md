# The prefix audit's cost: reading the blocks a request names, not the pool

**Branch:** `tp4/audit-cost` (on `tp4/stage1-windows`). **Status:** the Python side is built and CPU-tested, and the gate enforces what the W-1 numbers assume.
The device-side region read (`optimisation/sim/kv-region-read.patch`) is a DRAFT against the pinned tt-metal tree: it has been neither compiled nor run, and until it
is built into the image and its card check passes, the audit's cost is the old one. Nothing here ran on cards.

**What stops the pack running without the graft.** The W-1 numbers below (P1ab-LN 120/240) assume the graft. `c2_prefix_gate` now REFUSES an `exactness-shared` plan
before any container boots (exit 2, `refused: ...`) when the image's anchor probe (`docker run --network none`, no devices) finds no `ttnn.qwen_read_blocks`, cannot
import ttnn, or finds the served `model.py` off this checkout's `PATCHED_SHA256` (an image built before the narrowed audit; B0 skips a tag that exists).
`C2_PREFIX_ALLOW_FULL_AUDIT=1` overrides, for a run that wants the slow audit. The arms that audit a little (`exactness-audit`) are not gated. A missing graft thus costs
minutes and leaves P1ab-LN NOT exercised; it can no longer cost four hours of outage for a TIMEBOX. In the engine, `QWEN_PREFIX_AUDIT_READ=region` also refuses at attach
(`[PINDIAG] prefix: audit read mode=...` is logged at the model warm), not in the first audited prefill.

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
Under Lever N the route audits once per request (`levern_route`), so row-sharing saves nothing there; only the narrowed read does.
The split between the device read (44.5 GB packed) and the host conversion (167 GB unpacked, then copied) was not measured; at 0.33 GB/s overall the conversion and copies dominate unless the device read is slower than about 0.1 GB/s, which no figure here suggests. That is an inference from the rate: `kv_region_read_card.py` now times the two apart (`whole_device_read_s`, `whole_unpack_s`), the measurement the next card occasion should take first.

## What was built

1. **A step is audited once** (`_qwen_prefix_audit_rows`). The prefill loop collects its rows and audits them together after the last row, so a whole-cache read happens once per cache per step, not once per row. `_qwen_prefix_audit` (the Lever N route's per-request call) keeps its signature and calls the same code.
2. **Only the blocks a row names are read** (`_qwen_prefix_audit_selections`, `_qwen_prefix_read_blocks`), through `ttnn.qwen_read_blocks`: the named blocks go from the device into a host tensor of just those blocks (`ttnn.allocate_tensor_on_host`, the cache's dtype and layout), and the same `ttnn.to_torch` the whole-cache read used unpacks it. The digests are therefore byte-identical to the old ones, window by window, and the read volume follows the rows' lengths. The read compiles nothing and allocates no device memory (a host tensor and DRAM reads only), so it is program-free after the traces are parked.
3. **Safeguards on the region read**: the host tensor takes the cache's device topology (`update_tensor_topology`, when the tensor offers it), and the unpacked selection must have the shape the whole-cache read gives (blocks x heads-per-chip-times-chips x block x dim); a wrong shape raises, the step falls back to whole caches and the cost line says why. Without it a composer that saw fewer shards than chips would digest a subset of the heads, and hit and cold would agree on that subset (a false PASS on most of the KV). The audit is also measured against the program cache (`programs=a->b` on the cost line, the F3 warning and the `program_growth` counter on growth; the judges FAIL an arm whose audit compiled, and mark an audit arm NOT_EXERCISED when it logged no cost line at all). The audit runs outside every row's own program window, so this is the only place a compile in it would show.
3b. **Modes** (`QWEN_PREFIX_AUDIT_READ`): `auto` (default) uses the region read when the image has `ttnn.qwen_read_blocks` and reads whole caches otherwise; `region` refuses to run without the graft (an audit that silently reads whole caches costs 8 minutes a step); `full` forces the old read. A region read that raises falls back to the whole-cache read for the step and says so.
4. **One cost line per audited step**, `[PREFIX-AUDIT-COST] rows= reqs= tokens= mode= reads= blocks_read= read_ms= total_ms= [fallback=]`, so a window's log states what the audit read and for how long (it does not match the `[PREFIX-AUDIT] ` rows the judges parse).
5. **CPU tests** (`AuditCost` in `test_qwen_prefix_model_runtime`; the fake caches are per-chip replicated, as production allocates them): the region digests equal the previous algorithm re-run on the same caches (window by window and as the chained `kv_sha`), region and full modes log the same audit lines, the blocks read per tensor equal the row's block count at pools of 4,096 and 8,192 blocks and no whole cache is read, a two-row step reads each cache once, a refused region read falls back with the same digests, a region read that loses a chip shard falls back instead of digesting a subset, an audit that compiles a program is counted and warned, `region` without the graft refuses at attach, and the route's single-row entry still works. Gate and judge tests are in `test_c2_prefix_gate` (the cost line parsed, an absent line NOT_EXERCISED, audit growth FAIL, the refusal on four anchors and its override).
6. **The graft** (draft): `optimisation/sim/kv-region-read.patch` adds `tt::tt_metal::enqueue_read_tensor_dim0_slices` (runs of consecutive block ids become `BufferRegion` reads, every device's shard, one wait) beside the pinned tree's `enqueue_read_tensor`, and the binding `ttnn.qwen_read_blocks(device_tensor, host_tensor, blocks)`. `optimisation/ttnn-op/kv_region_read/kv_region_read_card.py` is its card check (byte equality with the whole read for a single block, a run, scattered ids, a shuffled order and two runs; no program-cache growth; the cost per block). The pinned tree already has the pieces (`ShardDataTransfer::region`, `enqueue_read_shards`, `BufferRegion`); the draft only combines them for a multi-device mesh.

## Delivery: how the graft reaches the window image (OPEN, no job exists yet)

The patch as drafted touches `libtt_metal.so` (the new function) and `_ttnn.so` (the binding). The image's graft mechanism (`docker/qwen-c2-serving.Dockerfile`,
`optimisation/ttnn-op/k64j/build_k64j.sh`) replaces `_ttnn.so` and `_ttnncpp.so` only, those hashes are pinned (`QWEN_FAST_RUNTIME_BINARY_SHA256`, the K64j checks), and rebuilding
them can silently drop image patches (a build must start from the tree that built the served binaries and keep every `QWEN_` string). **Recommended: a small standalone extension
module** (`qwen_kv_read.so`), compiled in the ttbuild container against the served tree, linking the image's `libtt_metal`/`_ttnncpp` without replacing them, imported by the audit
with `ttnn.qwen_read_blocks` as the fallback name. That leaves every pinned binary alone. It needs, before W-1 and not as part of B0:
1. the extension and its build (ttbuild, no card; 3 to 6 h of work and builds, an estimate, unmeasured);
2. the file in both bundle copy lists (`docker/qwen-c2-overlay.txt` and the evidence tree), the closure test, and the Dockerfile step that installs it;
3. `kv_region_read_card.py` PASSED inside the built `tp4-serve-11` on a card (about 30 minutes, an estimate), with its own tag and box;
4. the anchor probe's region check kept in step (for the extension route its line imports the module).
Nobody owns this yet: it is an open item for the owner to schedule, and W-1's P1ab-LN waits on it.

## What the graft still has to settle (it was written from the pinned headers, not built)

- Delivery: see the next section (an extension module, or a rebuild of the pinned binaries with their hashes re-pinned).
- `allocate_tensor_on_host`'s shape is assumed to be the per-device shard shape (what the cache's `.shape` reports); the binding checks the host tensor against the device tensor with `TT_FATAL`, so a wrong assumption fails loudly, never as wrong bytes.
- Several `ShardDataTransfer`s for one shard in one `enqueue_read_shards` is assumed to be allowed; `kv_region_read_card.py` has a two-run case for it.
- A wrong-bytes read bug shows as a hit/cold mismatch (a false FAIL); a lost chip shard is caught by the shape check above (the draft binding does not call `update_tensor_topology`; the Python does when the tensor offers it). Either way the card check must pass before P1ab-LN is read. The card check allocates the cache as production does (zeros of the per-chip shape through `ReplicateTensorToMesh`, then different values per chip) and compares shape and bytes.

## A graft-free route, not built: read packed, select on the host

If the card timing split shows the host unpack dominates (the likely case), a route with no C++ would read each cache packed once (`ttnn.from_device`, about 1.4 GB per cache across four chips),
select the named blocks' tiles on the host and unpack only those, or hash the packed tile bytes of the named blocks (the bytes every reader sees). It would ship with the image rebuild alone.
It is not built because the host-buffer layout of a mesh tensor (`Tensor.host_buffer`, `get_device_tensors`, `from_host_shards`, `from_buffer` are in the pinned ttnn) cannot be exercised on a
laptop, and an unexercised byte-layout assumption on the exactness path is what this audit must not carry. It would also still pay 44.5 GB of device reads per request, so it is a fallback; the
card timing split decides whether it is worth building.

## The estimate, and its basis

Per audited request, with L its prompt length: `t(L) = L x (0.39 ms to unpack and copy + 0.09 to 0.26 ms to hash) + reads`. The 0.39 ms is W-0's own rate (504 s over the pool's 1.278 M tokens, every token of it unpacked); the hash term is 131,072 B of float32 per token at 0.5 to 1.5 GB/s of sha256; the reads are 32 tensors per request at an assumed 0.2 ms per block-run call in the worst (fully scattered) case. That gives about 3 s at 5k tokens, 20 s at 32k, 70 s at 120k and 2.5 minutes at 262k, against 504 s for every request before.

`exactness-shared` at eight seats runs 48 requests whose prompts total 608,400 tokens (eight agents, three rounds, each prompt audited as a cold twin and as a hit: 2 x 304,200 from `SHARED_AGENT_TARGETS_8`), so the audit adds about 6 to 8 minutes in all. The old figure of the arm, 150 minutes, priced the audit at about 2 minutes a request (95 of the 150); replacing that with 8 minutes leaves about 60 minutes, and the pack keeps 20 more for the read's cost, which no card has measured.

**P1ab-LN (W-1 pack): 190 -> 120 minutes, box 336 -> 240.** The box is 1.5 times a 160-minute high end instead of the gate's worst case; the prefix gate clips every arm's docker timeout to what the box has left, so a slower audit ends the job as a TIMEBOX, never past 240 (and not after 336 as before). At the central estimates `P1ab-LN` is admitted with 99 minutes to spare (it needed E <= 144, it needs E <= 240) and `E1-LN` with 66, so the W-1 plan is 401 minutes (6.7 h) instead of 471 and the carry-over (and with it a third window for the cutover) is the exception, not the likely outcome. All of this assumes the graft; without it the job does not fit any box and is not worth starting.

**A P1-CTL re-run** (exactness-shared on today's control, `P1a-CTL`, estimate 150, box 183): it can only use the fast audit on an image that carries it, so it is the window-image form (`P1a-CTL2` in the combined pack, Lever N flags off), no longer a check of the production bytes. Estimate 60 minutes; its box can drop to 120 (1.5 x an 80-minute high end) on the same clipping rule. `P1b-CTL` (lifecycle-evict, no audit) is unchanged at 75/153. These are not in the W-1 pack (which holds no P1-CTL); they are the numbers to use if the owner adds the re-run.

**OPEN OWNER DECISION, W-2's precondition.** `tp4-w2-levern-jobs/ORDER.txt` requires W-0's P1a-CTL and P1b-CTL to be neither FAIL nor NO-VERDICT; W-0's P1a-CTL TIMEBOXed and its P1b-CTL FAILed on the coverage rule, so as packed W-2 and the cutover cannot start. Two ways out, both the owner's to take (each changes the outage, so neither was added to the pack unasked): (a) add to W-1 a `P1b-CTL-rerun` (lifecycle-evict on the production image, no audit, the fixed traffic: estimate 75, box 153; the anchor pin check runs only for the bring-up plan, so the re-pin does not block it) and a `P1a-CTL2` (the window image with Lever N flags off, audit on, graft required; estimate 60, box 120) with tags from W-1's reserve; that adds about 135 minutes to W-1's 401, so E1-LN moves to W-2 at the central estimates; (b) record that the controls are the window image's own S0-CTL plus the 8/8 byte-exact partial read of P1a, and re-scope W-2's precondition. Until one is chosen, W-2 stays blocked on W-0's two controls.

Without the graft, the new Python alone saves only what row-sharing saves: a step costs 8.4 minutes however many rows it holds. On the stock route the arm's roughly 34 steps (16 sequential round-0 pairs, then 8 cold twins and one burst of hits per round for two rounds, if the burst is one step) would take about 4.8 hours; under Lever N (P1ab-LN) the route audits per request, so the arm is 48 x 8.4 min = about 6.7 hours. Both are past any box. A smaller pool for the audited arms (`QWEN36_MAX_TOKENS_ALL_USERS` with a matching `num-gpu-blocks-override`; the arm's peak is under 200k live tokens, so about 6,000 blocks would hold it, 3.3x smaller, about 1.5 hours) is the only graft-free route; it has not been built because the contract's pool arithmetic (`kv_pool_problem`, the reservation and the boot rule's DRAM band) was written for the production pool and a derived profile needs its own review.

## Lifecycle-evict's coverage rule (P1b)

P1b-CTL passed 19/19 concurrent pairs and 15/15 solo pairs and failed only `QWEN_FAST_TP4_RS_UNIT_MAJOR_AUDIT: shape 64x5120 replay audits came from 1 block owner(s), 2 blocks are served`. The rule (`c2_smoke_check.u1_audit_problems`) wants a replay audit (round >= 1) from every block owner (`QWEN_FAST_M3_BLOCKS=2`: seats 0-3 and 4-7). The lifecycle traffic never had more than four users decoding at once, and its abort-while-waiting case (`lifecycle_abort_waiting`) held four seats: at eight seats that leaves four free, so the queued turn is admitted instead of waiting and the second block never decodes. `scenario_lifecycle_evict` now passes `seats=driver.seats` (the served profile's `max-num-seqs`): eight long `ignore_eos` holders keep both packed blocks decoding for 1,500 tokens each, and the turn really queues. Four-seat profiles are unchanged. Test: `test_lifecycle_evict_holds_every_seat_of_an_eight_seat_profile`. The cost is one minute or two of extra decode in an arm that already runs for hours; whether both owners then log a replay audit is a card question. Under Lever N (P1ab-LN) the waiting turn should still queue: the park path (`QWEN_FAST_LEVERN_PARK=host`) parks a long PREFILL's scratch so a short can run, and the holders here are decoding seats, not prefills, so a full batch queues the turn exactly as before; the event should read `phase='waiting'`. A `phase='unknown'` there is a finding to read against the park lines, not a rule failure (the P1ab-LN template says so).

## Does the baked image need the change?

Yes. The audit is code in the model graft that the image build stages (`qwen_prefix_stage` runs `qwen_prefix_model_patch` against the pinned originals and holds the result to `PATCHED_SHA256`, which changed for `model.py` on this branch). `qwen_prefix_model_patch.py` is already in both bundle copy lists (`docker/qwen-c2-overlay.txt` and the evidence tree), and no new `scripts/ci` file was added, so the provenance check needs no list change; but a `tp4-serve-11` built before this commit holds the old audit, so B0 must be rebuilt from a commit that contains it. The graft's binaries are a second artifact (above) and the one that matters for the cost. `prefix_replay.py` (the P1b fix) and the pack numbers are read from the checkout by the workflow, not from the image.
