# 8K draft attention: 256-key component qualified

The 8K target-attention replay gate and the 256-key draft-attention component
now pass. The original 64-key draft path remains rejected at 8K. The first
combined-runtime hardware result passes request audits; repeat confirmation,
the remaining context ladder and broad coding-quality acceptance remain open.

## First combined 8K hardware result

Run **34729556803**, revision **8c102b2**, completes with exit 0. Independent
reconciliation verifies 828 source hashes and recomputes both arm summaries
from the six complete requests, all exact in target tokens, state and inactive
slots. Each arm has one audited and two timed requests, ending at EOS after
121 committed tokens per request (256 was the maximum allowance).

| One stream / batch 1 | PP tok/s | CTX | Committed TG tok/s | Draft ms/block | Verifier trace ms/block |
| --- | ---: | ---: | ---: | ---: | ---: |
| Combined control | 3323.93 | 8192 | 99.81 | 30.82 | 68.79 |
| Shared Q/K candidate | 3310.33 | 8192 | 101.27 | 30.22 | 67.73 |

Both arms use qualified 256-key draft attention and 8448-row fixed history.
Acceptance is 222/330 proposed tokens (67.27%) in each timed arm. The candidate
improves TG by 1.47%; it does not establish an isolated chunk-size speedup.
Its full-width cycle averages 108.57 ms, including 9.13 ms selection/publication.
These nested timing fields should not be added as independent profiled intervals.

The 200 TG target needs roughly 55 ms per cycle at this committed-token yield,
so the remaining gap is substantial. Report SHA256:
`ce36fa35d6c8b4acc1af8c581d6b99c1738f0a6a22c6d547d8398bc12f22b92a`.

Shutdown caveat: after the report marks completion, the log emits
`SubDeviceManagerTracker is not initialized on MeshDevice 0`. The same message
exists in earlier 4K runs 34702027898 and 34702526963. Exit status and request
audits pass, but do not describe the shutdown log as warning-free. No serving
or held-out coding-quality qualification follows from this result.

## Accepted synthetic component

Run **34728453080**, commit **d029589**, passes the unchanged FP32 tolerance,
changed-input trace replays and clean teardown. Independent gate verification
matches all **44 source hashes**, factory-builder hashes and complete coordinates.

| Evidence | Result |
| --- | --- |
| Eager comparisons, both chips / both fixtures | 4/4 pass; zero failed elements |
| Changed-input trace replays | 4/4 exact and numerically passing |
| Borrowed inputs / physical KV layouts | 48/48 exact; 16/16 exact |
| Oldest/proposal/gap/frontier controls | 8/8 detected |
| Stale replay controls | 2/2 detected |
| Case 0 maximum absolute errors | Chip 0: 0.40828; chip 1: 0.40487 |

Raw absolute maxima are evaluated with unchanged `rtol=0.01, atol=0.01`, not
an absolute-only threshold. No reference values or live operands were changed.
Report SHA256: `2baa47c721607fac512b72413024c15a510ef475954cab976ee6d59222ab61af`.
`dspark_attention_8k_gate.py` rejects changed sources, incomplete coordinate
matrices, failed comparisons and non-exact replay. This is **not** full-request,
coding-quality or performance qualification. Next integrate this exact candidate
and its FP32-statistics factory into the bounded combined-runtime hardware trial.

### Hardware integration status

`dspark_8k_admission.py` now requires the retained component gate, exact
8192/256 request geometry, the pinned factory transformation, successful build
evidence, and matching hashes at both native binary paths. Admission is scoped
and restores the default 8192-row limit even on failure. Three host tests pass.
This helper is not yet connected to request execution: it cannot currently
enable 8K serving or bypass existing guards.

The isolated four-link runtime cache now includes the qualified factory patch
and its provenance in the 8K cache key. Both cache hits and new builds install
matching source; successful import and matching binaries are required before
writing hardware-build admission evidence. CI downloads and hash-checks the
retained draft report. Seven build/cache tests and shell syntax validation pass.
No hardware build has run with this wiring yet.

Remaining integration work:
- Validate the newly wired entry and scoped adapter in the actual hardware container.
- Run the combined two-card request with token/state auditing and report PP/CTX/TG.

The hardware entry now admits only the explicit captured-publication 8K trial
after checking build evidence. Preflight checks the numerical report and exact
8192/256 geometry before building. The scoped adapter selects the retained
256-key attention function and 8448-row double-buffered history, restoring
bindings on exit. The native compatibility gate permits only the independently
verified factory hash to differ from its old reference; all other native source
checks remain in place. Tests pass: 20 full-request, eight admission/build/scope,
and four existing context-selection tests. Hardware acceptance remains pending.

| Draft configuration | CI run | First failing comparison |
| --- | --- | --- |
| 64-key chunks, precise exponential | 34705245937 | Chip 1: 256 values, heads 11/13 at query rows 13/11 |
| 32-key chunks, precise exponential | 34705535416 | Chip 0: 512 values, maximum absolute error 0.54798 |

Both use capacity 8448, logical positions 8192 and 8433, 15 live queries,
the same synthetic fixtures, and rtol/atol 0.01. Both stop at eager case 0;
neither has complete replay coverage. The 64-key diagnostic has finite output
near -46 versus FP32 reference near -45.46 in an affected row. These are live
queries, not padding. Fixture negative controls pass on both host shards.

## Precision experiments

| Configuration | CI run | Numerical outcome |
| --- | --- | --- |
| FP32 statistics and output intermediates | 34723541474 | Chip 0: 15,843 failures; max error 0.92117 |
| Rebuilt native control | 34723984465 | Exact original output hashes; same 256 failures |
| FP32 statistics only | 34724444926 | Chip 0: 30,720 failures; max error 17.85616 |
| Statistics plus explicit pack-format transition | 34724923953 | Chip 0 passes; chip 1 still has 256 failures |
| Above plus nonlegacy reciprocal | 34725385866 | Same 256 failures; max error 0.54706 |
| Pack fix plus accurate correction exponential | 34726258186 | Byte-identical to 34724923953 on both compared chips; same 256 failures |

These are failed numerical trials, not accepted optimizations. The rebuilt
control reproduces the prebuilt baseline exactly. Adding the missing explicit
pack-format transition removes the large mixed-format regression, but does not
resolve the original error. The reciprocal change is not a solution.

## CPU rounding control

`scripts/ci/dspark_attention_rounding_control.py` runs online softmax with 64-key
chunks against the same FP32 reference, two fixtures and both host shards.
It tests FP32 intermediates, BF16 partial/running output rounding, BF16
probability rounding, and both rounding options together. The expanded control
also follows native ordering (unscaled maximum, then scaled subtraction), BF16
maximum/correction storage, and truncation of the scale to its upper 16 bits.
All 32 comparisons pass the unchanged tolerance. On the failing simulator shard
(case 0, chip 1), even combined output/statistics rounding and truncated scaling
has maximum absolute error 0.38304, below that row's relative tolerance.

This CPU-only check takes seconds and loads no model weights or device. It
does **not** emulate Tensix instructions or qualify a kernel. The tested
rounding mechanisms alone do not reproduce the simulator failure. Next isolate
native score scaling, maximum subtraction and exponential arithmetic rather
than dispatch another reciprocal or chunk-size trial.

## Correction exponential trial

The native first-column correction uses the degree-four polynomial when
`EXP_APPROX_MODE` is false. The already-grafted probability-tile path instead
uses `_ckernel_sfpu_exp_accurate_` with FP32 destination accumulation. The next
synthetic simulator trial selects that accurate helper for the correction too,
only for the existing draft signature. Counterintuitively, `true` in
`exp_tile_first_column` selects this helper; it is not the fast approximate
probability-tile path. This difference is a hypothesis, not a demonstrated bug.

Keep FP32 statistics and explicit pack-format repair; restore the native
reciprocal so the comparator is run 34724923953, not the failed reciprocal
trial. Record both exponential headers in the runtime fingerprint. Host tests
verify scope restoration and the transformation matches pinned native sources;
only the unchanged simulator numerical/replay matrix can qualify the change.

Result: run 34726258186 failed eager case 0. Both output hashes exactly match
34724923953. Do not promote this correction swap or infer a speed improvement.

An additional CPU truncation hypothesis produces a different failure signature:
on case 0/chip 1, truncating partial/running BF16 output gives 30,417 failures
and maximum error 1.70295; also truncating statistics gives 30,617 failures and
1.77983. Neither matches the simulator's 256 failures. The diagnostic retains
these as hypotheses, not assertions about actual packer rounding. Distinguish
rounding/storage behavior with stage evidence before another precision change.

## Value-isolation evidence

Run 34726821443 retains identical baseline output hashes and the 256 failures.
Its three value-only diagnostics complete on both chips, with clean teardown.
They do not qualify attention merely because their absolute tolerance passes.

| Chip 1 diagnostic | Head 11, row 13 actual/reference | Head 13, row 11 actual/reference |
| --- | --- | --- |
| Constant one | 1.00000 / 1.00000 | 0.99609 / 1.00000 |
| Oldest-token indicator | 0.10352 / 0.10395 | 0.10352 / 0.10335 |
| Last-proposal indicator | 0.82031 / 0.81436 | 0.82422 / 0.81786 |

The last-proposal probability excess times its value of -64 accounts for about
-0.38 and -0.41 of the error in these rows. This is evidence of incorrect
effective weights, not proof of a denominator bug: numerator arithmetic can
also affect the indicator result. Constant-value maximum error is 0.01563.

Next isolate the final denominator row reduction: replace its matmul-by-identity
with the native SUM reduction API for the draft signature only. Preserve FP32
statistics, the pack fix, reciprocal, fixtures, and original tolerance. This
is an unqualified simulator experiment, not a runtime default change.

Run 34727344168 fails with both eager output hashes unchanged from 34726821443;
all six value-diagnostic maxima also remain unchanged. The SUM substitution is
not an accepted fix. Before drawing stronger conclusions from identical results,
the synthetic scope now adds compile-time assertions requiring the precise-draft
selector and exactly 8512 physical keys / 64-key chunks. Source hashes alone
prove installation, not that a conditional branch executed. Successful compilation
with these assertions will prove specialization selection for this probe only.

Run 34727823121 successfully compiles with both assertions and executes all
value diagnostics before reproducing the same eager output hashes and 256
failures. Specialization selection is therefore verified; a silently disabled
conditional does not explain these unchanged results. Teardown is clean.

The CPU control also tests BF16 score conversion before maximum/subtraction.
On case 0/chip 1, round-to-nearest produces 10,444 failures (max 1.20284), and
truncation produces 8,075 (max 1.05605), without output rounding. Neither matches
the native signature. These are additional excluded simple hypotheses, not
evidence that the native implementation actually converts scores to BF16.

## Bounded chunk-merging candidate

With correction, reciprocal, final reduction and selector hypotheses exhausted,
test fewer online merges directly: 256-key chunks give 34 iterations rather
than 133 at 64 keys. The earlier 32-key trial worsened errors; the value probes
now identify biased effective weights. This motivates a bounded comparison,
not an unrestricted chunk sweep or a claim that rounding is proven causal.

All original 8512 operand rows, absolute positions and 15 proposals are retained.
Add 192 rows of +8192 keys / -8192 values with negative-infinity mask entries,
giving 8704 physical keys. Host tests verify original operands and masks are
unchanged, added rows are poisoned/masked, dispatch requests 256 keys, and all
new tensors are retained for trace lifetime. The FP32-statistics predicate and
compile-time selector assertions match the new geometry. Remove the ineffective
SUM substitution; keep the pack-format fix. No serving default changes.

## Source-backed precision scope

The pinned SDPA factory uses FP32 QK and sum buffers when FP32 destination
accumulation is enabled, but keeps `im_df` and `stats_df` in BF16. Thus the
FP32 accumulation flag does not make every attention intermediate FP32.
This is a plausible contributor, not a proven explanation of the failures.

Local factory source matches the failed CI report exactly:
`a263559fe23cdf6fa8194604b238a939d299a356592eae1c7b2df11868383ebc`.
Relevant definitions are around lines 704-718 of
`ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp`.

Stop blind chunk-size changes. The isolated draft-only FP32 variants above
failed; do not promote them. Keep target attention and serving defaults unchanged. Require
the existing negative controls, full accuracy matrix, exact changed-input trace
replay and clean resource teardown before any full-request hardware trial.

32-key report SHA256:
`3cd1178e1888cf669a637c4af5f69aaa2f793ce225bb4e81fd3a32ae2f9e4331`.
