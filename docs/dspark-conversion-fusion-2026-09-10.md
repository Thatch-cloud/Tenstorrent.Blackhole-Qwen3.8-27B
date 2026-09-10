# Remove DSpark attention round trips

**Candidate, not a measured speedup.** The captured-proposal comparison is complete;
this attention replacement remains separate from the qualified request path.

## Fifteen-query simulator result

Run `20260910T055634Z-396-dspark-native-fixed-attention-probe` passes all82 checks
at capacity4384, positions4096/4109, and fifteen queries on both simulated chips:
4 eager, 4 exact replay, 48 unchanged inputs, 16 layouts, 8 fixture controls,
and 2 stale-input controls. The original CPU tolerance remains rtol/atol0.01.
Maximum absolute difference is0.452 on deliberately high-amplitude inputs;
CPU agreement is tolerance-based, while replay agreement is bit-exact.

The report and exit0 are pinned by `dspark_native_fixed_gate.py`. All recorded
sources match before/after and the checkout; the original packer is restored
and both ownership locks are absent. Existing decomposed-attention evidence
was not modified. Learned-layer comparison and full-request hardware checks
remain required before adoption. This result provides no TG measurement.

The full learned-layer simulator attempt `20260910T060817Z-381` timed out
after1800 seconds during baseline case1, before any candidate comparison.
Its retained report is explicitly incomplete (exit124), not a numerical rejection.
Baseline case0 preserved all checked inputs and weights. The timeout left the
child-owned SDPA graft installed; it was restored and both original source hashes
verified. The outer launcher now also restores this known child patch on timeout,
with tests rejecting unexpected external modifications.
Do not repeat this entire expensive baseline loop unchanged. The changed attention
kernel has already passed the82-check simulator gate; full learned-layer and
request comparison can next run on hardware without claiming the timeout passed.

## Existing cost sources

`dspark_full_attention.execute` explicitly widens query, keys, values and masks
to FP32, materializes score/softmax intermediates, then narrows its output to BF16.
It also repeats each KV head four times before widening. These are actual device
operations; trace capture removes host dispatch overhead, not their memory traffic.
Their individual wall-clock cost has not yet been isolated.

## Reuse the qualified native component first

The existing precise-native SDPA candidate with 64-key chunks passed 236 simulator
checks at historical lengths 31 and 4096, with seven queries and unchanged
0.01 relative/absolute tolerances. Its report is
`scripts/ci/dspark-attention-precise-chunk64-simulator.json`.
That evidence does not qualify the current 15-query fixed-history request layout.

1. Extend that candidate's simulator matrix to 15 live queries and fixed capacity
   4384, with query keys after capacity and actual positions 4096/4109.
2. Preserve the exact fixed-history mask: hide the unused history gap, retain
   noncausal proposal visibility, and keep padded query rows finite.
3. Check eager A/B and changing-input trace replay, both chips, poisoned padding,
   oldest-history dependency, and the original CPU attention tolerance.
4. Compare learned cached-layer outputs before changing full-request integration.
5. Run native versus decomposed attention in matched hardware requests; report
   PP / CTX / committed TG, acceptance, setup-inclusive time and target correctness.

Prefer BF16 inputs consumed directly by native SDPA with internal accumulation
over external FP32 expansion. The precise-native candidate requires its audited
runtime changes; it is not interchangeable with stock SDPA or a dtype flag.
Do not overwrite the shared runtime while another simulator owns it.

If native attention fails this expanded gate, retain the failure and investigate
the specific arithmetic/layout cause. A separate BF16-input SFPU path could remove
external widening and grouped-head replication without changing the arithmetic,
but should not replace this lower-effort reuse experiment preemptively.

No claim of reaching 200 committed tok/s follows from fewer operations alone.
