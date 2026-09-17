# Integrated two-P150A candidate

Profile: `p150-pair-t8-norm-four-link-v1`. Experimental only; serving is unchanged.

On the allocated AMD runner, from this checkout:

```bash
QWEN_CARDS_ALLOCATED=1 bash scripts/ci/run-candidate.sh
```

This launches the pinned runtime image through the existing disposable-container
harness. It performs complete coding requests, comparing one-link control against
the packaged four-link candidate, with fresh native token/state oracles. It does
not reset cards. CI uses `full-norm-engine`, `coding_request=true`,
`fabric_link_probe=true`; other experiment toggles remain false.

| Included | Configuration |
| --- | --- |
| Target | Existing pinned Qwen3.8-27B weights and two-chip TP layout |
| GDN | Device-loop recurrence, batched convolution/norm, packed checkpoints |
| Verification | Captured T1/T2/T4/T8 buckets and reusable input buffers |
| Attention | Native attention with ordered cache handling; no experimental long-context replay |
| Selection | Force argmax, four-channel descriptor, sampler-scoped four-link override |
| Publication | Existing selected-prefix state publication and exact rollback checks |
| Drafting | Lookup only, capped at seven proposals plus the anchor |

`CandidateRuntime` owns verifier construction and the sampling override until its
traces are closed. Incompatible options fail rather than silently mixing experiments.
The implementation reuses the validated kernels; it does not fuse the whole model
into a new monolithic kernel or claim a new speedup merely from packaging.

Excluded: unvalidated DFlash2 integration, Ethernet dispatch, deferred convolution
publication, eager gate/up fusion regressions and unqualified precision changes.

The underlying combination measured 19.10 committed tok/s at CTX170/B1 in
[run 34185881285](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34185881285).
The packaged runtime then passed [hardware regression 34186950201](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34186950201):
19.09 committed tok/s versus 18.92 for the one-link control, with complete outputs
and exact native token/state checks. It is not a 200 tok/s result or a held-out coding-quality evaluation.
