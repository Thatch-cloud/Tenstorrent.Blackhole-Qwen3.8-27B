# T16 and T32 integration

## What is integrated

PR #7's T32 Markov, attention, prepared proposal, target, publication and
commit-only GDN paths are retained as working source and test modules.
T32 is **not** an alias for T16 and its code is not discarded by an ours-only merge.
Merge commits retain the full T16, T32 and main histories, plus PRs #1 and #2.

T16 remains the measured default. T32's simulator-only admission, source hashes,
numerical bounds and hardware rejection remain in place. No serving deployment
or new T32 performance claim follows from integrating the source.

## Validation gates

| Gate | Current status |
| --- | --- |
| T32 host tests: shapes, weights, proposal/state publication and all prefixes | 66 tests pass locally |
| Shared full-request, fusion, SDPA, profiling and routing regression tests | 47 tests; pass with one Linux-only skip on Windows |
| Frozen ladder guards, source pins, prompt, cache and audit ownership | 25 tests pass locally |
| Default generated SDPA replacements and signature vs T16 parent | Exact equality |
| Linux CPU CI | [35175404913](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/35175404913): all 138 tests pass, no skips |
| Frozen T16 source staging after merge | Recipe, draft-tail and wide-cache ladder stages pass against the exact frozen checkout and retained evidence |
| Fresh T32 changed-input simulator replay after integration | Pending |
| Matched combined T16/T32 hardware PP/CTX/TG and state audit | Pending; host backlog first |

Local checks use Python 3.11, CPU torch 2.14.0 and NumPy 2.4.6. They load small
test fixtures, not the 27B checkpoint, and do not establish hardware correctness.
Linux CI completes in under a minute; its 138 tests take about four seconds.
This admits integration of guarded experimental source, not T32 promotion into
the active hardware recipe. Fresh replay and combined hardware gates remain pending.

## T32 CI selection

Select a `t32-*` suite in `qwen-experiments.yml`, with
`simulator_only=true`, `learned_stack=true`, `cards_allocated=false`,
`simulator_fusion_t16=false`, `fabric_link_probe=false` and
`reset_authorized=false`. This reuses the existing suite input rather than
exceeding the workflow's 25-input limit.

Available suites: `t32-markov`, `t32-markov-learned`, `t32-attention`,
`t32-draft-attention`, `t32-commit`, `t32-combined`, `t32-publication`,
`t32-context-attention`. These are explicit diagnostic lanes, not an automatic
hours-long prerequisite for every context ladder run.

Before hardware promotion, qualify the combined proposal/replay and every-prefix
commit with fresh source manifests; then compare complete T16/T32 requests at
the same prompt, context and output policy. Keep failures and setup time.
Do not remove the T32 hardware guard merely to obtain a timing number.

Historical component evidence is in [T32 admission](t32-runtime-admission.md).
The frozen T16 ladder continues to use its pinned staging chain, not the merged
experimental modules by implication.
