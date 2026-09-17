# DFlash2 versus the promoted combined runtime

Status: combined adapter and CI staging implemented; hardware qualification pending.
Control is the user-promoted T16/DSpark stack at `239c7c1`, not the older
DFlash2 74.27-TG recipe. No new throughput result is available.

## Comparison contract

- Start at CTX4096, one stream, the same coding fixture and output budget.
- Keep target weights, precision, four-link policy and T16 verification matched.
- Keep direct windows, wider MLP down, scatter normalization and register
  epilogue active in both arms, with per-request execution evidence.
- DSpark retains compact Markov selection. DFlash2 uses its own selector;
  absence of DSpark selection is intentional, not a missing target optimization.
- Preserve each drafter's checkpoint and native history policy. DFlash2's 2K
  draft history is not truncation of the target's 4K KV context. Report this
  difference rather than calling the comparison history-matched.
- Two fresh correctness audits followed by ABBA complete requests. Compare
  target outputs and state across arms, not draft proposals or acceptance paths.
- Report PP/CTX/committed TG, setup, draft, verification, publication and
  accepted tokens per block. Keep timing stalls, source hashes and raw reports.

## Implementation sequence

1. Separate drafter-specific selection from shared target scopes. Implemented
   with host tests for identity, route engagement and exception cleanup.
2. Prepare T16 masks, absolute rotary inputs and global vocabulary candidates.
   Host utilities now support 16 rows; old eight/32-row behavior is retained.
3. Device/cache execution now supports explicit T16 composed attention, with
   cached history and captured proposals. Capture is deferred until verifier
   persistent allocations exist. Native proposal attention and live-query QK
   remain T8-only; their gates are not widened. The DFlash2 request owns the
   same four target scopes directly, without claiming DSpark history/selector
   hooks executed. Full target-output, state, feature, cached-versus-uncached
   proposal and changed-input replay checks remain mandatory.
4. Qualify changed-input proposal replay and full-request target state on the
   combined hardware path before timed ABBA. Use tiny simulator cases only if
   kernel changes require them; do not load model weights in the simulator.
5. Compare DFlash2 T8 separately after matched T16; retain width/history labels.

The comparison driver runs both drafters in the same loaded target session.
The `experiment/cumulative-t16-full-v*-dflash` tag family selects the adapter
in `qwen-cumulative-t16.yml`; other cumulative tags retain the existing driver.
It uses cached pinned DFlash2 fixtures only (revision
`dedf8df68adfb1afeaf7b7480c0a0243108177b4`), fails if missing, and verifies tensors
with the existing fixture loaders. It does not start a multi-hour download.
The independent report validator checks both target component identities,
within-drafter proposal repeatability and exact shared target outputs, while
allowing different proposals between drafters.

Host tests are preparation evidence, not device numerical or performance
acceptance. Serving defaults and the immutable promoted DSpark recipe remain
unchanged. Source-qualified artifacts must be regenerated/reconciled where
the new host-source hashes differ; old admissions are not silently reused.

## First hardware launch: preflight failure

Run 35285659419 attempt 3 passed the disk gate but stopped before device
execution: the shared simulator-source gate detected modified `draft_attention.py`.
The integration had widened a host mask argument check in a file shared with
DSpark. This was a staging regression, not a kernel crash or numerical failure.
The follow-on Apport `FileNotFoundError` was only error-reporting noise.

The fix restores that file exactly to its simulator-pinned bytes and puts the
T16 host mask specialization in `dflash_attention_mask.py`, used only by DFlash2.
The comparison stage now runs the real shared-source preflight before and after
its overlay. Independent CPU visibility-formula tests cover T16 masks; the
original source-pin regression and existing widths remain checked. No source
hash, numerical threshold or simulator admission was bypassed.

Run 35287122216 subsequently reached both audit generation paths, but failed
at the DFlash2 summary before timed requests. The new driver omitted
`ended_with_eos`, `sampler_num_links` and `fabric_sources`, previously populated
by `full-prefix.py`. The retained DSpark audit also lacked these fields. This
is an integration-reporting failure, not an accepted benchmark or kernel-speed
result. DFlash2's request was not retained because validation preceded append.

The driver now derives completion from emitted/EOS IDs, reads the active sampler
link count, and checks fabric-source hashes against the runtime audit before
attaching them. It records completed requests before summary validation so a
future rejection retains its evidence. Source identities are rechecked after
the comparison; validation thresholds and exact-state gates are unchanged.
