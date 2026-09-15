# Restore one combined performance stack

The 4K/8K and 64K numbers are different configurations, not a context-scaling curve.
The target remains 200 committed tokens/s for one coding stream, with correctness
and coding quality preserved. Serving defaults stay unchanged.

| Optimisation | Current 64K evidence | Next gate |
| --- | --- | --- |
| Captured draft proposal and feature publication | Retained in full requests | Keep exact changed-input replay and publication audits |
| Commit-only GDN | Retained | Keep active/inactive state checks |
| Folded T16 target attention | Integrated in the 64K timed path | Retain exact target outputs and state |
| Fast draft attention | Long-history numerical repair path; split-K not admitted | Original-order multicore simulator, 64K hardware numerical gate, then combined requests |
| Fused T16 MLP | Not enabled by the isolated 64K entry | Port admission and dependencies; exact target outputs/state before matched timing |
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

The last clean 64K result is 14.3005 TG (run 34842710242): approximately 344 ms
draft, 81 ms verification and 45 ms selection/commit per T16 block. Even eliminating
drafting would not establish 200 TG at the observed yield. Restoring and optimising
the verifier/commit stack is required, not optional work after a kernel-only win.

Evidence: [64K measurements and attribution](context-ladder-investigation.md),
[8K combined results](draft-8k-numerical-investigation.md),
[current split-K gates](splitk-draft-status.md).

## Split-K hardware handoff (15 September)

Simulator run **34911012256** passes original-order, sixteen-worker split-K.
Hardware run **34912625173** is the weight-free 64K numerical/replay screen;
its dispatch is not hardware acceptance or a throughput result.

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

The request binding is `dspark_native_cached_layer.attend`, installed inside
`dspark_64k_scope.runtime_scope`. Install the admitted split-K adapter after that
scope enters, otherwise the baseline adapter overwrites it. Keep target attention,
history publication, request correctness and state audits unchanged. The retained
timing gate describes the old candidate: obtain a new combined request audit before
qualifying timings for the split-K candidate. Do not relabel its old report.
