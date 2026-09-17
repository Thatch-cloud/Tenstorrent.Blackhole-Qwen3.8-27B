# Exact Markov bias cache: feasibility only

The current 8K combined runtime reaches roughly 101 TG, not 200. Its draft
stage takes about 29.8 ms/block and verifier trace about 67.7 ms/block.

`dspark_markov_score_layout.py` repeats a learned rank-256 dot product at each
of 15 proposal steps. Its bias depends on the previous token and fixed learned
weights, not on the current base-logit row. An exact cached bias could therefore
be reused before adding the new base logits. Do not cache final scores or tokens.

## Measured reuse opportunity

Cold per-request LRU analysis of run 34730226400, without future tokens,
prompt prewarming or cross-request reuse. Each timed request has 165 actual
Markov predecessors, including rejected draft prefixes, with 63 distinct tokens.

| Cache slots | Hits / lookups | Hit rate | FP32 row payload per card |
| ---: | ---: | ---: | ---: |
| 16 | 73 / 165 | 44.24% | 15.16 MiB |
| 32 | 91 / 165 | 55.15% | 30.31 MiB |
| 64 | 102 / 165 | 61.82% | 60.63 MiB |
| 128 | 102 / 165 | 61.82% | 121.25 MiB |

These are host trace counts, not measured device hits or saved milliseconds.
Payload excludes tags, alignment and scratch space. One frozen coding task is
not evidence of general workload hit rates. The reproducible analyzer is
`scripts/ci/dspark_markov_cache_analysis.py`; three host tests pass.

## Required experiment

1. Prototype a bounded device-owned 64-slot cache, with validity and weight
   identity checks. Preserve the original native dot product for misses.
2. Prove cache-hit execution actually skips the dot product. A gather/where
   selection that still executes every matmul is not an optimization.
3. Synthetic simulator checks: exact FP32 bias bits, hits/misses, collision and
   eviction, weight invalidation, changed-input replay, poisoned cache rows,
   and teardown. No model weights in simulation.
4. Matched combined hardware requests: identical proposals, target tokens/state,
   cold/warm setup costs, actual hit counts and PP/CTX/TG. Include multiple coding
   tasks rather than tuning cache membership to this recorded completion.

Reject 128 slots unless new workloads demonstrate useful additional hits.
Do not change learned weights, shorten vocabulary, or move lookup decisions to
host-side per-token synchronization. Serving defaults remain unchanged.

This optimization cannot reach 200 TG alone: even deleting the entire draft
stage leaves the current verifier slower than the roughly 55 ms cycle budget
at the observed eleven committed tokens per block. Verifier and acceptance
improvements remain necessary. The full context ladder remains outstanding.

## Controller implementation status

`markov_cache_control.hpp` implements 64-slot LRU lookup and explicit commit
with 32-byte metadata records suitable for a device dataflow wrapper. A miss
does not become valid until commit; stale tickets, repeated commits, invalid
tokens, epoch rollback and counter overflow are rejected. Epochs must increase
when weights or cache ownership change; exhausted epochs require fresh state.

The C++ host test passes with `-Wall -Wextra -Werror`, covering hits, uncommitted
misses, eviction, invalidation and counter exhaustion. This is only controller
logic. A metadata-only RISCV dataflow wrapper and changed-input trace probe are
now simulator-qualified for metadata only. They use one worker per
chip, 2,176 bytes of persistent metadata and 4 KiB scratch per chip. Commands
and commit tickets remain device-resident; host reads in the probe are audits,
not a proposed serving path. Execution must be serialized by the cache owner.

The `markov-cache-control` suite in the CPU simulator workflow loads no model
weights, does no runtime-library rebuild, and has a ten-minute probe timeout.
It checks 148 commands on both chips, including uncommitted misses, LRU
eviction, generation changes, stale commits, overflow and changed-input replay.
This controller gate alone does not qualify a bias cache or speedup. The later
sparse-dot numerical gate and connected-pipeline work are recorded below.

| CI run | Result | Scope |
| --- | --- | --- |
| 34731557841 | Rejected immediately | DMA staging did not match DRAM address alignment; no completed checks |
| 34731657226 | Passed, 2m19s total CI | 296 exact checks, both chips, changed-input trace replay and clean teardown |

The corrected wrapper stages each transfer with matching low address bits,
then copies between aligned DMA staging and packed controller records.
This conservative metadata implementation is not a latency result. All four
probe/kernel source hashes match the run and remain unchanged after execution.
Report SHA256: `ff55bddd2dfbcce53601b1f56b0d220e57fb5b194cdc3637c673ee849a1bc3a5`.
Ten local Python tests and the warning-clean C++ controller test also pass.

## Next: skip the native dot, not just its result

Local native-source inspection found an existing `ttnn.sparse_matmul` path.
With `nnz` omitted, its reader sends a device-side validity flag to all three
compute threads, which skip the matmul loop for an inactive batch. This is a
candidate for avoiding a new custom conditional-matmul implementation, not
yet a qualified replacement for the current dense Markov dot.

The next tiny test must compare active sparse output bit-for-bit against the
current HiFi4/FP32 dense dot at rank 256, then alternate active/inactive inputs
through the same trace. Do not set a fixed `nnz`: changing hit/miss counts would
violate that kernel's protocol. The sparse operation zero-fills skipped output,
so cached bias needs a separate persistent payload and explicit selection.
Its FP32 intermediate-buffer configuration must also match dense arithmetic;
sharing the compute source alone does not prove numerical equivalence.

### Sparse-dot evidence

| CI run | Result | What it establishes |
| --- | --- | --- |
| 34731978621 | Rejected by shape validation | Default sparse semantics expand both batch dimensions; single-pair mode must be explicit |
| 34732171781 | Numerical failure, clean teardown | Four inactive chip/replay outputs exactly zero; all six active comparisons differ from dense FP32 bits |
| 34732469080 | Passed, 5m32s including factory build | All six active comparisons bit-exact to dense; all four inactive outputs zero under changed-input replay |

The active mismatch affects all 64 output values per comparison, with maximum
absolute differences of 0.00225–0.00310. No tolerance was relaxed. Native-source
inspection shows that dense matmul marks partials CB5 for FP32-to-destination
unpack, while the sparse factory omits this flag. The new factory trial adds
that flag only for the paired Markov geometry. The corrected simulator run
passes all ten comparisons with zero mismatches and clean teardown. The
original and corrected factories share the same native compute kernel.
Source and rebuilt-library hashes are retained and were reconciled locally.

Report SHA256: `24c67f8ddb688f7b3ac61ae4733275369515c9c0f7011011e0c555da4014ab37`.
Corrected factory SHA256: `5340ab7cfc29dc858f189bb83ee6aff71e79eab5fae8a0c661cbb94847cc0079`.
Both loaded library paths: `3f6bd505e052af45c07c7e48470f2b51075cd7e22baed9f23b4c166325e2bbc5`.

This gate uses rank 256 and 64 output columns. It does not qualify full-vocabulary
cache payloads, real-weight proposals, or runtime performance. Next connect the
controller's miss decision to a device-owned sparse mask and separate cached
FP32 bias payload, then check eviction/replay and the production per-worker tile
count before promoting a complete candidate to matched hardware PP/CTX/TG.

The inactive zero fill is an operation-level output initialization, not a
zeroing operation in the matmul writer. At full vocabulary this still writes
a padded FP32 tile row (about 30.31 MiB per chip). Even a numerically qualified
sparse path will need combined-runtime timing; skipped MACs alone are not proof
of a useful speedup.

## Connected cache pipeline

Run 34733011935 passed the connected device sequence: lookup, sparse mask,
native sparse dot, FP32 payload fill/read, then metadata commit. All decisions
execute inside the same replayed trace, without host hit/miss branching.

The synthetic matrix fills 64 slots, checks reuse and LRU eviction, changes
the weight epoch, and rejects a stale payload ticket without changing the
cache. Hits must return the exact dense bias while the sparse-dot output is
zero. Cache rows start poisoned. The dense dot in this test is an independent
audit, not intended for the optimized runtime. Real token-to-embedding feedback,
production per-worker tile counts and combined hardware PP/CTX/TG remain open.

The run completed in 6m18s including its disposable factory build. All 140
chip/request checks passed (including six chip-local cache hits), both stale
tickets produced poisoned output without changing cache bits, and teardown
completed cleanly. All seven pipeline/controller source hashes match the local
worktree and the before/after run records. The separate ten-comparison native
sparse/dense gate also runs before this pipeline test.

Report SHA256: `1e5380c8d640c89221962f40aa6a47e8c805f45ecb0de0b551d30dfc940cf351`.
Eleven local Python tests pass. This remains a 64-column synthetic gate, not a
full-vocabulary performance result. The next size gate must exercise the real
78 output tiles per worker, followed by complete-request hardware timing.

## Production-size worker gate

Run 34733393010 passed the 64-column protocol matrix plus two
larger synthetic widths, using the real rank-256 reduction and 78 tiles per
matmul worker. There is one factory build for the complete suite.

| Output columns | Workers | Active output tiles per worker | Purpose |
| ---: | ---: | --- | --- |
| 4992 | 2 | 78, 78 | Full production worker workload |
| 3712 | 2 | 78, 38 | Same final-worker padding as 248320 columns across 100 workers |

Each wider case checks misses, hits, weight invalidation and stale payload
rejection. This avoids loading the complete model into simulation; it does
not substitute for the eventual full-vocabulary hardware gate.

Runtime integration must reset metadata after proposal warmup/capture, so
timed requests start cold rather than inheriting cache entries from warmup.
Persistent cache buffers must outlive the prepared proposal trace and be
released only after that trace closes. Both comparison arms must retain the
qualified shared-Q/K, fused T16, four-link and 8K attention configuration.

All 164 chip/request checks and six stale-ticket checks passed with clean
teardown. All seven pipeline source hashes match the current worktree and
before/after records. Total CI time was 7m05s including the factory build.

| Columns | Exact comparisons | Report SHA256 |
| ---: | ---: | --- |
| 64 | 140 | `0934aee979f6a0d42abb930956db11a9e8cf10eee577ff365b6cc0a7b0ac346b` |
| 4992 | 12 | `66c18bd4c3e6c492af4ba30fc2519f26847a1de3acd6380f466b05de45ed30d9` |
| 3712 | 12 | `eb77b9c2cb1a93e0e8a4493407fda5000615d02d9cf729ee2df0e73677ef3ee9` |

These results qualify the tested worker tile counts, not 100-worker multicast
scaling or full-request speed. Live draft-token feedback and the complete
hardware request comparison are the next integration gate.

## Live token feedback

Run 34733995681 tested the new `dspark_cached_markov.py` request-owned
adapter. Each of its fifteen steps uses the device-selected previous token
for both embedding and cache lookup, then adds the current base logits and
selects the next token. The request metadata deliberately contains an invalid
token in the simulator, so lookup cannot accidentally use a host placeholder.

The gate repeats the controller/payload size matrix, then compares all scores
and tokens against the unchanged native fifteen-step chain. It includes a
zero-logit tie fixture, changed-input replay, immutable inputs/weights, output
poisoning, and cold resets that advance the epoch without changing addresses.
Sixteen local Python tests pass. Hardware integration is not enabled yet.

The combined-runtime reset belongs after `prepare_proposal_trace` completes
its extra warmup proposal, not immediately after trace capture: that later
warmup would otherwise repopulate the cache before timing starts. Its cost
must remain in request setup. The borrowing proposal trace must close before
the cache releases persistent buffers.

The live-anchor controller/payload tests passed at all three widths, but the
fifteen-step chain failed its first pattern/step/chip score comparison and
closed cleanly. This is not admitted to hardware. Run 34734406268 adds first-dot
shape, cache-state and dense-reference diagnostics, plus score/token mismatch
details. That diagnostic completed with the same failure and clean teardown.

It identifies a concrete geometry mismatch: native embedding/tilize returns
`[1, 1, 256]`, while the scoped FP32 reload guard accepted only `[1, 1, 1, 256]`.
The cache tags were correct on both chips, but the first sparse dot differed
from the dense dot by up to 0.000304. The guard therefore missed the native
feedback path even though the four-dimensional synthetic fixtures passed.

Run 34734814212 tests a guard accepting both exact single-row forms, with no
reshape, dtype conversion or added data movement. It repeats every size and
feedback gate, and now explicitly requires the diagnostic three-dimensional
first-dot comparisons to be bit-exact. Sixteen host tests pass; hardware
promotion remains disabled pending this result.

Run 34734814212 passed both first-dot comparisons and all 90 eager feedback
comparisons. Replay stopped before its first comparison because PyTorch CPU
does not implement `count_nonzero` for the returned UInt32 cache metadata.
Teardown was clean. The reset assertion now converts host metadata to int64
before counting; no device kernel or runtime precision changes. The host reset
test also exercises UInt32 metadata. All 16 host tests pass. A fresh synthetic
replay gate is required before hardware promotion; no new TG result is claimed.

Run 34735206013 (revision 8ef266f) passes: 90 eager and 120 changed-input
feedback comparisons, reset epochs 2 and 3, and clean teardown. All 15 local
feedback source hashes match both report snapshots. The 64-, 4992- and
3712-column pipeline reports also pass with clean teardown and all seven local
sources matching before/after snapshots. Feedback report SHA256:
`2084242137fd239a0a46e55810414cbb5cc8e411c2e01e212d025f6f295ac620`.

This admits work on the opt-in combined hardware integration, not a performance
claim: full-vocabulary qualification and performance qualification remain false.
The hardware build must include the scoped sparse FP32 factory change alongside
the existing four-link and 8K changes. Both comparison arms must retain shared
QK, fused T16, captured publication and score layout; only the candidate adds
the bias cache. Reset after the final proposal warmup and release borrowing
traces before cache buffers. Follow this comparison with the full-ladder
checkpoint in the programme, rather than another optimization cycle.

## Combined hardware outcome: rejected

Run 34736316338, revision efc847c, passes correctness and closes cleanly.
Both arms retain shared Q/K, fused T16, captured publication, score layout and
four-link sampling. At 8192 prompt tokens, control PP/TG is 3311.81/100.90;
cache PP/TG is 3306.93/93.91. TG regresses 6.9277%. Each arm has one audited
and two timed requests, 242 timed committed tokens, 330 proposed and 222
accepted (67.2727%). This is the 121-token natural-EOS fixture, not sustained
1024-token generation or held-out coding quality.

All 856 reported local source entries are unchanged across execution, as are
the native source snapshots. Report SHA256:
`196ce98ba43cef180d8b8bdb64569f794290a1f359f3fc1f337b83f573de219b`.
The preceding run 34735942778 built successfully but stopped before model
execution because the source gate expected the original sparse factory;
efc847c added exact transformed-source admission, not an arithmetic change.

Do not promote the cache. CI selection returns to uncached operation, preserving
the immutable cache experiment tags. Continue the full-context ladder on the
uncached baseline before further performance candidates. The regression cause
needs attribution; correctness and an offline predicted hit rate do not prove
that saved dot products outweigh cache traffic and dispatch costs.
