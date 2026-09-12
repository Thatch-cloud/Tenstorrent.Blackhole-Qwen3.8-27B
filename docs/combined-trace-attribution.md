# Combined-runtime trace attribution

Status: experimental profiler route; no timing result yet.

The most recent GDN changes preserve exactness but do not reduce blocking trace
time. Before another fusion, measure the actual device operations in the current
combined runtime rather than reuse an older pre-fusion profile.

| Preserved feature | Configuration |
|---|---|
| Target | Qwen3.8-27B, two P150A, four fabric links |
| Workload | CTX 4096, one coding stream; output capped at 64 tokens for attribution |
| Drafter | Native DSpark attention, captured proposals, score layout |
| Verifier | T16 folded attention, commit-only GDN, fused T16 MLP |
| Publication | Captured feature projection; unused singleton position uploads skipped |
| Experimental GDN changes | Disabled; native recurrence reference |

Only one audited request runs. The existing request observer brackets real
verification calls and records their trace IDs; device profiler rows must match
those replay IDs on both chips. Feature, output and state audits remain enabled.
The report explicitly marks instrumented/correctness-only and leaves PP/TG null.
This is neither a new throughput measurement nor completed-function acceptance.

The prior profile flag rejects fusion and captured publication. A separate
`combined_profile` option retains those safeguards and admits only the complete,
audited combined configuration. The report validator additionally checks its
publication/fusion route. Serving defaults remain unchanged.
