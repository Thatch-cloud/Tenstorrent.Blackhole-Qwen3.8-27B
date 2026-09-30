# DFlash2 upstream implementation audit

Source review, 2026-09-18. No new hardware performance claim or serving change.

The [z-lab model card](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
identifies itself as a mirror of our `incoai` checkpoint source. Our fixture pin
remains `dedf8df68adfb1afeaf7b7480c0a0243108177b4`; mirror status is not a
byte-for-byte check of the latest weights.

Reviewed [SGLang model code](https://github.com/sgl-project/sglang/blob/69525389808719a9a2e4274142da7ff2f28f3eab/python/sglang/srt/models/dflash.py)
at revision `69525389808719a9a2e4274142da7ff2f28f3eab` against
`scripts/ci/draft_shared_head.py`, `dflash_device.py` and `draft_selector.py`.

| Area | Upstream | Current local path | Assessment |
|---|---|---|---|
| Candidate communication | Per-shard top-k, gather compact candidates | Four chunk top16 results per chip, compact host merge | Already avoids full-vocabulary readback; could reduce transfers further |
| Vocabulary selection | FlashInfer radix top-k when available | TT-NN topk on four 32K chunks | Different kernel; upstream CUDA speed claims do not establish TT gains |
| Learned selection | Device edge lattice and captured Triton path walk | Device hidden projection, CPU FP64 greedy path | Real placement difference; port requires numerical and replay qualification |
| Width | Published model-card evaluation uses T8 | Current matched winner is T16; T32 rejected | Compare T8 only on the same combined recipe, not historical runtimes |

The local selector already applies learned predecessor/successor interactions.
This is not a missing-model-feature fix. Upstream precomputes candidate-pair
edges; local greedy selection computes only the chosen predecessor's edges.
Moving the walk to device may remove host synchronization, but copying the full
lattice approach also increases arithmetic. Preserve deterministic tie handling
and compare complete proposal tapes before claiming equivalence.

## Next decision gates

1. Attribute the existing draft interval to head/top-k, readback and selector
   without global Tracy or changing proposals. Do not rerun whole-model simulation.
2. If material, qualify compact device candidate merging or selector execution
   with bounded fixtures and changed-input simulator replay, then combined ABBA.
3. Keep target-verifier optimization in parallel: the latest matched run spends
   about 60 ms in verification alone, above the 55 ms complete-cycle budget at
   11 committed tokens/block. Eliminating all drafter overhead cannot reach 200 TG
   at that acceptance rate. No selector-only experiment can establish the goal.

No new runtime candidate is qualified by this source review. The existing compact
CPU selector experiment must be checked before implementing a duplicate.

## Bounded host attribution helper

`scripts/ci/dflash_host_attribution.py` provides an opt-in context manager around
one already-prepared proposal capture. It records preparation, combined candidate
readback/merge/selection, and the remaining replay/bookkeeping interval. It adds
no device operations, fences, model loads or tensor copies. At most 64 proposal
records are retained; subsequent proposals execute normally without timing.

It rejects audit mode and nested scopes, checks operation coverage, and restores
instance methods even after failure. The remainder is host wall time around
blocking replay, not a device-kernel breakdown or a claimed optimization.
CPU tests cover timing accounting, unchanged outputs/order, bounds, restoration,
invalid clocks and missing stage coverage. Hardware staging is not wired yet;
this helper alone does not provide new measured attribution.
