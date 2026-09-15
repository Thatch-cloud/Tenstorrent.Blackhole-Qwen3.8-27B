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

## Text-only attempt and bounded cache reads

Text-only audit `34932542190` times out before completing the request. Its
`load_target_once` to `upload_fc.weight` interval is 313.84 seconds. Removing
vision construction is not sufficient to cure the loading slowdown; the candidate
is not hardware-qualified and there is no new TG result.

Read-only diagnostic **34933388567** (`cb8deec`) completes in **14 seconds**:

| Cached input | Bytes sampled | Initial read + SHA256 | Repeat read + SHA256 |
|---|---:|---:|---:|
| Layer 0 input norm | 10,584 (whole file) | 0.189 ms | 0.028 ms |
| Layer 0 packed QKV projection | 67,108,864 of 89,825,728 | 33.246 ms | 29.707 ms |

The two hashes agree for each file. This is buffered read-plus-hash time, not
cold storage bandwidth, whole-checkpoint bandwidth, TT tensor deserialization,
host conversion, device upload, or a model-throughput result. No cache flush was
performed. It rules out a uniformly slow read of these sampled cached bytes at
the observation time, not intermittent contention during the failed model run.

The attempted no-device flatbuffer probe **34933749978** failed: passing
`device=None` still triggers chip discovery in this pinned runtime. It supplies
no deserialization timing. Do not repeat that probe without devices.

## Loading regression, not another throughput result

Paired run **34934197491** exhausts its outer 540-second budget before the first
control generates. Setup reaches the first request at 365 seconds. There is no
completed control/candidate comparison and no new TG result.

Read-only inventory **34935346770** completes in eight seconds and finds **zero
running containers using the pinned experiment image**. Leaked experiment
containers are not present at this observation. Host I/O pressure is still
visible (10-second `full` 10.49%), but this alone does not establish causation.

The next explicitly selected diagnostic samples the actual hardware Python
stack and process/cgroup I/O every ten seconds, with native stack dumps every
30 seconds. It runs the unchanged combined-runtime loading path with a
**120-second process limit and 240-second outer limit**, retaining logs even
on timeout. It does not qualify performance; an expected diagnostic timeout
is not converted into a green model-acceptance result. No kernel, numerical
gate, serving default, or model-loading behavior changes.

### Actual loader sampling: 34935654817

The job completes in **3m20s**, with the intended 120-second process timeout.
Samples repeatedly land in `tp_common.shard_w`'s BF16 transpose/copy and in
`mlp._build_gate_up`'s host packing. Other samples include native tensor loading
and attention packing. Physical process reads reach **49.35 GB at 110 seconds**.
Sparse samples identify executed work, not precise phase percentages or proof
that all read bytes belong to those functions. Loading reaches the vision tower
before termination; this is not a throughput or model-correctness result.

The pinned TTNN `as_tensor` has a `preprocess` callback invoked only when
conversion is needed (no cache, cache miss, or failed cache load). Both TP shard
loading and packed gate/up loading currently perform their host transformations
*before* calling it, even when the cached device tensor is successfully loaded.

The next opt-in candidate moves those exact transformations into `preprocess`.
Cache names, tensor dtypes/layouts, shard mapping, native tile packing and device
placement stay unchanged. Vision loading is unchanged to avoid combining two
unqualified loader changes. Cache-hit tests forbid touching the source tensor;
cache-miss tests check conversion order and exact small-tensor packing inputs.
The candidate must complete the existing combined output/state audit before
clean timing. No TG improvement is claimed from this setup optimization.
