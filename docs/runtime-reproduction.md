# Reproduce the combined runtime

## Choose the right recipe

The measured ladder uses the **T16 combined offline runtime**, not the historical
vLLM serving command and not the newly integrated T32 experiments. The unchanged
recipe varies `QWEN_DSPARK_REQUEST_CONTEXT`; each geometry still needs its own
fresh full-request correctness audit.

| Pin | Value |
| --- | --- |
| Frozen runtime checkout | `8c102b20df22329106955b4006bf4d650bb94e40` |
| Integration's T16 parent | `506af16f1b40554fbd6a6ce92aad5ce87da5f350` |
| tt-metal in experiment image | `9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9` |
| Local Docker image ID | `sha256:f1e9b1a64b4f7aa04cd3d3b36fefed4d47320bfdd0f4d108d2ca85a932cf9465` |
| Target | `Qwen/Qwen3.8-27B`; retain the audited snapshot/config/index fingerprints |
| Drafter | `RadixArk/Qwen3.8-27B-DSpark` |
| Drafter revision | `b9a5dbdf03bc999c6c73c426b19c2d9041cea393` |
| Drafter checkpoint SHA-256 | `2aff025f45823b40ebe726b9dfa40302f3512bd9a11c3a7347de32a567acd9a7` |

**Image portability gap:** the Docker ID is not a verified registry manifest
digest or a pull address. Reproduction currently depends on the runner's retained
image and audited model snapshots. A portable release must publish that image,
record its registry RepoDigest, and verify it against the runtime fingerprints.
Do not invent a registry tag or assume rebuilding a Dockerfile produces identical bits.

## Replay a measured row

Use GitHub CLI with repository access. These commands request real hardware:
coordinate card ownership and a quiet disk window first.

| CTX | Successful run | Job to rerun |
| ---: | ---: | ---: |
| 4,096 | 35167726511 | 105033990629 |
| 8,192 | 35167726511 | 105033990700 |
| 16,384 | 35165321998 | 105024997143 |
| 32,768 | 35165321998 | 105024997082 |
| 65,536 | 35172072077 | Rerun the single-context workflow |
| 131,072 | 35173979225, attempt 2 | 105090186482 |

```bash
REPO=Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B
gh run rerun 35167726511 --job 105033990629 --repo "$REPO"
gh run watch 35167726511 --repo "$REPO"
gh run download 35167726511 --repo "$REPO" --dir evidence/35167726511
```

Reruns use the run's original revision, not the latest branch. Retain the run
attempt and new artifacts separately; old checksums do not apply to a rerun.
If GitHub no longer permits a rerun or retained artifacts have expired, prepare
a new explicitly reviewed immutable trigger using the same pinned staging chain.
Do not move existing experiment tags or silently substitute missing evidence.

The workflow is [qwen-combined-ladder.yml](../.github/workflows/qwen-combined-ladder.yml).
It checks out the frozen runtime and stages context, draft-tail, evidence and
ladder adapters before execution. It does **not** benchmark whatever happens to
be at the branch tip. CPU tests of the merged tree are not requalification of
all staged kernels.

## Host prerequisites

- Runner: `thatch-build-amd64-02-cp-temp`; label `thatch-qwen-p150a-pair`.
- Both cards allocated; no other process holding them. Do not kill unrelated jobs.
- Devices `/dev/tenstorrent/0` and `/dev/tenstorrent/2`.
- `MESH_DEVICE=P300` selects the logical two-device model path; the physical
  descriptor must be `p150_x2_mesh_graph_descriptor.textproto`, not the P300 board descriptor.
- Four-link fabric discovery/overrides must pass the recorded guards.
- Retained image, read-only model snapshots, build cache and evidence artifacts.
- Existing runner authorization and registry credentials; never place secrets in this repo.

The exclusive hardware concurrency group prevents two experiment jobs using the
cards together. It does not prevent other CI jobs from saturating the host disk.
Earlier 131K retries were delayed by disk pressure; attempt 2 subsequently passed.

## Evidence and acceptance

Archive the entire run, including source manifests, image/runtime identity,
prompt/corpus digest, cache geometry, exit status and device/checkpoint closure.
Component evidence dependencies currently include runs 35060395395, 35062238298,
35068517785, 35024279412, 35086789628, 35158163367 and (wide cache) 35170767742.
GitHub artifact retention is not a long-term release archive.

Every row needs exact output, active/inactive target-state and feature checks,
then two timed complete requests. Recompute PP and committed TG from those
requests. A green simulator gate or kernel speedup is not a model TG result.
See [ladder results](combined-context-ladder.md) for report hashes and limitations.

131K passes with **2,124.78 PP tok/s and 53.69 committed TG tok/s**. The bounded
audit-memory fix reached hardware; the report closes cleanly with one audited
and two timed complete requests. Its SHA-256 is
`8ee07d6f794999d0a3684722b174bcae99abdaac84a0b45b75ef25809e768104`.
262,144 prompt tokens plus 256 generation tokens exceed the model limit;
the **261,888 + 256** full-window test is being qualified separately. It retains
the T16 recipe and adds target-only routing at the final positional boundary;
see [full-window qualification](full-window-262k.md). No 262K TG is qualified yet.

## Build versus replay

`scripts/tt-build-images.sh --prstack --vllm` builds the historical bring-up,
patched tt-metal, serving and plugin layers. See the Dockerfiles in `docker/`.
This alone does **not** recreate the later staged experimental runtime.
Use the replay pins above until a verified distributable image is published.
The [historical serving guide](serving-harness-history.md) documents tool-call
flags and terminal clients, with its outstanding accuracy/streaming caveats.
