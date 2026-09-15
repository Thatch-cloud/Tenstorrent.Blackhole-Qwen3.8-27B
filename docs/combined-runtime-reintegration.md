# Restore one combined performance stack

The 4K/8K and 64K numbers are different configurations, not a context-scaling curve.
The target remains 200 committed tokens/s for one coding stream, with correctness
and coding quality preserved. Serving defaults stay unchanged.

| Optimisation | Current 64K evidence | Next gate |
| --- | --- | --- |
| Captured draft proposal and feature publication | Retained in full requests | Keep exact changed-input replay and publication audits |
| Commit-only GDN | Retained | Keep active/inactive state checks |
| Folded T16 target attention | Integrated in the 64K timed path | Retain exact target outputs and state |
| Fast draft attention | Combined 64K audit and two clean EOS requests pass; 30.23 TG | Repeat matched control/candidate and retain exact state checks |
| Fused T16 MLP | Explicit 64K audit and clean EOS timing pass; 31.40 TG | Pair control/candidate timing; do not attribute cross-run commit variation to MLP |
| Shared GDN Q/K preparation | Not enabled by the isolated 64K policy | Integrate after its fused-MLP/target dependencies; exact recurrence and full-request checks |
| Short-context score layout | Not enabled by the isolated 64K entry | Audit long-history capacity/layout assumptions before integration |

## Execution order

1. Repair partitioned draft attention without weakening numerical gates.
2. Validate the actual parallel configuration at 64K on hardware.
3. Integrate it into the existing full-request path and measure complete PP/CTX/TG.
4. Restore score-layout, fused-MLP and shared-Q/K dependencies as an explicit combined candidate, with source-verified admission and exact request/state checks.
5. Compare the same final configuration at 4K, 8K and 64K before extending the ladder to 16K, 32K, 131K and 262K.
6. Validate sustained coding output and held-out coding quality, then concurrency separately from single-stream TG.

Each hardware comparison must record enabled optimisations, source hashes,
precision, topology, prompt/output lengths, acceptance yield and complete draft,
verification and commit timings. Component results cannot fill TG cells.

The latest clean 64K result is 30.2348 TG (run 34923389309): approximately 86 ms
draft, 81 ms verification/readback and 54 ms selection/commit per block, averaging
6.75 committed tokens. The historical control was 14.3005 TG (34842710242),
not measured as a paired A/B in this run. Even eliminating
drafting would not establish 200 TG at the observed yield. Restoring and optimising
the verifier/commit stack is required, not optional work after a kernel-only win.

Evidence: [64K measurements and attribution](context-ladder-investigation.md),
[8K combined results](draft-8k-numerical-investigation.md),
[current split-K gates](splitk-draft-status.md).

## Split-K hardware handoff (15 September)

Simulator run **34918015135** passes the 256-key chunk gate. Hardware run
**34918350712** passes the full-history numerical/replay screen in 38 seconds.
`dspark_splitk_hardware_gate.py` verifies the retained report, source dependencies,
all numerical/replay checks and input/layout/frontier controls. Subsequently,
combined audit **34922265472** passed and clean timing **34923389309** completed
two exact EOS requests with 270 committed tokens total. PP is 2575.6711 tok/s,
CTX is 65536 and committed TG is 30.2348 tok/s. Both requests preserve exact
outputs, final state and inactive slots; the split-K source is restored and
devices close cleanly. Per-request decode times are 4.70258 and 4.22752 seconds.
The report SHA256 is
`78720d03941214b5cc389757592189a5684767ad6b94a7a618cf93d44f2ffc7b`.
This is a repeated short-fixture runtime measurement, not sustained/serving or
held-out coding acceptance. Next implementation work is step 4 above.

| Stage | Bound | Evidence required |
| --- | --- | --- |
| Rebuild or source-keyed cache restore | 360 seconds | Matching factory, registered operations and both loaded libraries |
| 64K hardware probe | 120 seconds | Four eager and four changed-input replay checks; poison/frontier controls; clean close |
| Container execution | 510 seconds | Terminal exit status and retained logs; only this run's container is cleaned up |

The combined runtime needs **both** factories in one rebuilt library: the existing
64K prefill factory (`dspark_64k_build.prepare`) and the qualified split-K decode
factory. Replacing the library after producing the old build report invalidates
that report; rebuild once, then issue and validate both manifests against the same
binary. Preserve the existing CCL/GDN registrations and source-keyed cache inputs.

`dspark_splitk_combined_build.py` now prepares both factory identities under the
existing single-library cache build. Its validation rejects a changed decode
factory even when the prefill manifest passes. Local gate/mutation tests pass;
the combined library and request integration have not yet run on hardware.

The combined request entry now installs the split-K adapter **after** the legacy
64K scope, checks the actual kernel hash, rejects zero split-K calls, and restores
the source on success or failure. It retains the bounded 17-token audited request,
captured publication, folded T16 verifier and target-state checks. This first CI
job is a correctness screen, not the full-response PP/CTX/TG measurement.
Host execution is capped at nine minutes; the request process at seven minutes.
The first run (34919330098) completed the cold combined build in 297 seconds,
then hit the host deadline during target loading. No request audit completed.
The built library is cached; the next run also confines the experimental precision
flag to draft calls, leaving native target decode untouched.

The request binding is `dspark_native_cached_layer.attend`, installed inside
`dspark_64k_scope.runtime_scope`. Install the admitted split-K adapter after that
scope enters, otherwise the baseline adapter overwrites it. Keep target attention,
history publication, request correctness and state audits unchanged. The retained
timing gate describes the old candidate: obtain a new combined request audit before
qualifying timings for the split-K candidate. Do not relabel its old report.
