# Loading diagnostic: 31-second CI result

Run **34932153411** (`fa63ce1`) reads the pinned image source and host/cache
metadata without loading tensors or opening devices. It completes in 31 seconds.

## Findings

- Both model storage and the experiment cache use the same ext4 filesystem,
  `/dev/md1p1`, with about 129 GiB available (86% used).
- Host I/O pressure is elevated: 10-second `some` 17.41%, `full` 12.80%.
  This later snapshot is not proof of the cause of the earlier decode slowdown.
- About 90 GiB of host memory is available at this snapshot.
- The pinned `initialize_vllm_model` unconditionally calls `init_vision_model`.
  That calls `reference_vision_model`, which loads a complete Hugging Face
  conditional-generation model and returns its visual submodule. The TT vision
  tower is then built even for this text-only coding fixture.
- The failed run `34931204191` confirms execution, not just dead code: its log
  contains cached visual-block and merger weight loads through 05:10:26 UTC.

## Next experiment

Prepare an explicit `QWEN_TEXT_ONLY_LOAD=1` constructor scope for the text coding
fixture. Skip only fresh default vision initialization, leave all text weights
and model layers unchanged, and restore the original method afterwards. Reject
explicit visual arguments. No serving default or multimodal capability claim.

The helper has local behavior tests and is wired into the explicit
`experiment/dspark-64k-score-text-only-v1` correctness run. It must pass the
combined text output/state audit before clean timing.
Measure loading and committed TG separately: removing unused vision construction
may improve setup and residency, but a decode improvement is not established.

Source evidence: `runner-evidence.local/34932153411/qwen-load-diagnostic-34932153411/image.json`.
Host snapshot: the adjacent `host.txt`. Complete native source SHA256 values are
retained in the image report; no native source was modified by the diagnostic.
