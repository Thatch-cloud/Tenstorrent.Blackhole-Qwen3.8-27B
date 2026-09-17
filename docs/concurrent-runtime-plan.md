# Combined runtime: batching and concurrent users

This lane complements, **not replaces**, the 200 committed tok/s single-stream
target. Aggregate throughput across users cannot satisfy that target.

## Current boundary

The measured recipe is not yet a concurrent serving implementation:

| Source | Constraint |
| --- | --- |
| `dspark_request_experiment.py` | Native warmup uses `max_batch_size=1`; prefill uses `empty_slots=[0]` |
| `dspark_request_runtime.py` | One request frontier, drafter and verifier binding per runtime |
| `dspark_score_layout_scope.py` | Process-wide prepared Markov hook; nested owners rejected |
| `full_dspark_request.py` | Per-request capture, proposal history and verifier preparation |

T16 is sixteen speculative verifier positions for **one user**, not sixteen
users. Raising a batch flag does not provide independent recurrent state,
correct rollback or multi-user trace bindings. The old B8 endpoint measurements
do not qualify the new DSpark combined recipe.

## Ordered experiments

| Gate | Work | Required evidence |
| --- | --- | --- |
| C0: host protocol | Interleave two independent request bridges at equal and unequal positions | Prefixes 0–16, foreign request/ticket rejection, failure isolation |
| C1: two-slot simulator | Route explicit slots through KV, GDN convolution/recurrent state, feature capture and publication | Distinct prompts; alternate commits; compare each user against serial execution; poison inactive slots |
| C2: slot lifetime | Cancel/reject/reuse slots; bind traces and history to request generation | Stale tickets and old trace bindings rejected; no state leakage; resource release |
| C3: combined hardware | Two users at 4K on the same qualified kernels and immutable prompt corpus | Exact target tokens and active/inactive state; complete PP/CTX/TG; memory accounting |
| C4: concurrency ladder | 1, 2, 4, 8 users at 4K, 8K, 16K, 32K, 64K | Per-user and aggregate results; stop a geometry on OOM or correctness failure |
| C5: serving adapter | Disposable endpoint with continuous batching and streaming | Arrival scheduling, TTFT, token timestamps, cancellation, queueing and quality |

131K follows single-user qualification. Higher concurrency requires measured
memory headroom. The model's prompt-plus-generation position limit still applies.
No production serving flags or service lifecycle changes are authorized by this plan.

C0 tests use fake device operations and independent object graphs. They establish
only the request publication protocol, **not physical KV/GDN isolation**, a
scheduler, true batched execution or a concurrent throughput result.

The first C1 candidate is `gdn_slot_copy.py` / `.cpp`: an explicit-slot version
of the unchanged slot-zero state DMA. Recurrent state advances by 384 tiles per
slot; convolution state selects each slot's 32-byte row in both tile faces.
The candidate rejects hardware use until simulator qualification. The weight-free
`gdn-slot-copy` TT-Sim suite checks all eight slots, both copy directions, eager
execution and two changed-input replays on both chips, including whole logical
destination state and source preservation. Kernel correctness remains unproven
until its report passes; KV routing and combined multi-user integration remain separate gates.

Simulator attempts (17 September): run 35176302618 failed during checkout;
the workflow now uses a fresh per-attempt checkout directory. Run 35176440059
passed checkout but timed out after 30 seconds in Docker preflight creation,
before opening a mesh or executing the kernel. Retained host diagnostics show
59.34% full I/O stall over the recent 10-second window (41.78% over 60 seconds);
the named preflight container did not exist when inspected. This is a host setup
failure, not a numerical failure or simulator pass. Avoid increasing kernel
timeouts or changing the candidate to address disk contention.

## Measurement contract

- Keep model/image/kernel pins, prompt token IDs, sampler, output policy and warmup
  consistent with the one-user control. Record changes explicitly.
- Distinguish sequential scheduling of concurrent requests from actual batched
  device execution. Do not label duplicated model instances as shared-model batching.
- PP: report prompt tokens and measured prefill duration; label serialized versus
  batched prefill. TTFT includes admission and queueing.
- Per-user TG: committed output tokens divided by that user's measured generation
  interval. Aggregate TG uses total committed tokens divided by the shared
  measurement window, **not the sum of per-user rates**.
- Report median/P95 TTFT, E2E latency, per-user TG and fairness; P95 token latency
  requires token-level timestamps. SSE chunks containing multiple tokens do not
  establish per-token spacing; report chunk gaps separately when necessary.
- Include actual per-user CTX, accepted/proposed counts, active KV pages, recurrent
  state/history bytes, trace memory and peak allocation. Idle cache usage is not
  the occupancy metric for an active request.
- Test simultaneous arrivals, staggered arrivals and mixed contexts, with a long
  prefill arriving during active decode. This is where chunked prefill and
  prefill/decode scheduling must demonstrate responsiveness, not just throughput.

Reuse loaded weights and qualified builds within a fixture. Start with a bounded
two-user audit before expanding the matrix; do not load full model weights into
the simulator for protocol checks or run an hours-long matrix before the first
correctness gate passes.
