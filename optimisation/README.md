# Decode optimisation: what shipped, what did not, and how to reproduce it

The record is **[docs/optimisation-plan.md](../docs/optimisation-plan.md)** — every
experiment, its arms, its numbers, and the decisions. This directory holds what the
plan's results were produced with, so a reader can re-run an arm rather than trust a
table.

Measured on the same two p150a as the README, tt-metal `v0.77.0-rc1` (`9f9cd4fd`),
2026-09-02/03. The endpoint numbers are **ITL at 8 concurrent streams with the first
token dropped**, never whole-request time (the plan's §2.4 explains why the latter
produced controls 41% apart).

## Result

| endpoint, 8 streams | ITL | per user | change |
| --- | ---: | ---: | --- |
| README serve command + lever A | 68.3 ms | 14.6 tok/s | |
| + C + D + shard-greedy (this stack) | 61.2 ms | 16.3 tok/s | −10.4% ITL, +11.6% per user |

Lever A alone was +15.5% per user over the README's composed decode, so the stack
compounds to roughly +29% per user. GSM8K for the stack is 57/60 against the README's
58/60 (one item, `60 → 40`, flips; the plan records it).

## Environment variables (the part that needs no files)

```
-e QWEN_GDN_FUSED_DECODE=1                               # A: fused T=1 GDN recurrence (tt-metal #53587)
-e QWEN35_GDN_DECODE_BF16=1 -e QWEN35_GDN_STATE_BF16=1   # C: bf16 GDN step + state (GSM8K 57/60 = fp32)
-e QWEN_SDPA_BF8=1                                       # D: bf8 KV (GSM8K 57/60; 262k output byte-identical)
-v ~/ttcache:/ttcache -e TT_METAL_CACHE=/ttcache         # kernel cache: readiness 510-615 s -> 165-435 s
```

A needs the image built by `graft/Dockerfile` (the fused op is upstream PR #53587,
grafted onto the base image; the rebuilt `_ttnn.so`/`_ttnncpp.so` are not in this repo —
they are 70 MB of build output tied to one tt-metal revision).

## Patches (`patches/`)

Unified diffs against the files in the `v0.77.0-rc1-prstack` images, applied by bind
mount in the runners. Each is named by its lever in the plan.

| patch | lever | what it does |
| --- | --- | --- |
| `A-graft-wrapper.patch` | A | routes `recurrent_gated_delta_rule_decode_ttnn` to the fused op behind `QWEN_GDN_FUSED_DECODE` (the wrapper baked into the graft image) |
| `3c-inplace-state-*.patch` | 3c | `QWEN_GDN_FUSED_INPLACE=1`: the op writes the new state into `rec_state` and the copy-back is skipped when `buffer_address()` matches (−0.42 ms B=1 / −3.26 ms B=8) |
| `3d-shard-greedy-endpoint.patch` | 3d | `QWEN36_SHARD_GREEDY=1`: greedy-only stand-in for the sampler so the endpoint argmaxes each device's own vocab shard inside the decode trace (−4.4 ms ITL at 8 streams; greedy traffic only, see the plan) |
| `B0-shard-mode-demo.patch` | B0 | the demo's `QWEN36_BATCHED_DECODE_MODE=shard` without the sampler assert (demo only) |
| `E-ccl-*.patch` | E | `QWEN_CCL_NUM_LINKS=4`: overrides the hardcoded link table (upstream issue #55125; −1.35 ms) plus the AG/RS link and grid env gates used for the sweeps |
| `L-proj-1d-flag.patch` | L | `QWEN_PROJ_1D=0` to A/B the DRAM-sharded in-projections (measured negative; kept for the record) |

Everything under `patches/` modifies tt-metal, which is Apache-2.0, and is offered
under **Apache-2.0**, as with `../patches/`. The graft's op source is upstream's, not
vendored here.

## Runners (`rig/`)

The scripts the plan's tables came from. They are rig-specific (device by-id paths, a
private registry, `$HOME` layout) and are here as the exact procedure, not as a portable
tool. All take the interleaved-A/B shape: the control is the *same mounts with the flag
off*, never a stock image.

| script | produces |
| --- | --- |
| `arun.sh` / `arun8.sh` | B=1 / B=8 demo arms (`text_demo.py` traced decode) with per-lever env passthroughs |
| `eprun.sh` + `bench_itl.py` | endpoint arm by ITL at 8 streams; `bench_itl.py` also hashes the 8×200-token outputs so arms can be checked for identical greedy text |
| `run3e.sh` / `runship.sh` | endpoint arms with the shard-greedy patch mounted; `runship.sh 1` is the full stack above |
| `serve262k.sh` | the 262k-context acceptance (TTFT, ITL at 262k, saved continuation) |
| `gsm-c.sh` / `gsm-d.sh` + `gsm_client.py` | the 60-item GSM8K drift gates |
| `run_spec_fused.sh` | the speculative-decoding harness on the fused baseline (§5) |
| `mmbench*.{sh,py}`, `drambench.sh`, `modprof.sh` | matmul microbench, raw DRAM read rate (~430 GB/s peak), per-module profile |

## Not shipped, and why

Recorded in the plan with numbers: H (more cores), I (packed gate|up), J (bf4 read
rate), L (DRAM-sharded in-projections), CPU offload, and speculative decoding on the
fused baseline (0.65–0.83×; the harness's rollback commit stalls with the fused op — a
harness interaction, not a serving one).

## Next: K, the conv + gates kernel

The remaining GDN cost is 68 small ops per layer (§3.3: 11.0 ms/step at B=1, 22.4 at
B=8). The first kernel of K replaces the conv shift-register, the four-tap FIR + SiLU
and the beta/g gates — 14 ops per layer — with one op, and is designed so the
recurrence op's reader can take its output layout directly, which retires the q/k/v
split and the GQA expansion as well. Work in `ttnn-op/` as it lands.

## Added 2026-09-04

- `ttnn-op/attn_prep/` — `ttnn.transformer.attn_decode_prep`: the attention layer's decode prologue as one op (head split from the fused projection output, QK RMS norm with the (1+w) weight, partial rotate-half RoPE, k/v padded and emitted in the KV update's height-sharded config). Wired by `ttnn-op/patch_attn_prep_wire.py` into `tt/attention/tp.py` behind `QWEN_ATTN_PREP=1` (engagement line `QWEN_ATTN_PREP engaged`). Test: `ttnn-op/test_attn_prep.py`.
- `ttnn-op/patch_reader_fast.py` — recurrence op reader without per-tile row zeroing (the rank-1 write is two broadcasts). Adopted, byte-identical.
- `ttnn-op/patch_state_fast.py`, `ttnn-op/patch_reader_2barrier.py` — measured negatives, kept as records (not applied).
- Ship stack env now also carries `QWEN_ATTN_PREP=1`; the runners take `ATTNPREP=1`.
- `graft/Dockerfile.k` — the full ship stack as a serving BASE image (`tt-vllm:qwen38-k`): FROM the lever-A graft image the production `thatch-serving-tt` builds from, every grafted op's kernels, the rebuilt libraries, the wired model files, and the engagement flags as ENV. Shard-greedy is deliberately left out (greedy-only). Hand it to Thatch.Server's `tt-serving-image.yml` as `base`; context and the tt config stay per-model in that repo's `engine.py`.
