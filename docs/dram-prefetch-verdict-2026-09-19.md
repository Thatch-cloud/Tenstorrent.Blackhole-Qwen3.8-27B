# DRAM-core prefetch: qualified, and it does not pay

**Verdict: do not pursue programmable-DRAM-core prefetch for Qwen's MLP.** The
architectural option opened on 19 September — firmware upgraded, harvesting clear,
capability native, kernel correct — and then measured slower than the path already
in place. This records what was established so the question does not get reopened
without new information.

Supersedes the open gates in [dram-prefetch-mlp-2026-09-09.md](dram-prefetch-mlp-2026-09-09.md).

## The four gates, resolved

| Gate | Result |
| --- | --- |
| Maintenance feasibility | Cleared. Both cards flashed 19.8.1.0 → **19.12.0.0**, tt-flash 3.11.0, run 35400761367 |
| Pair eligibility | Cleared. No DRAM harvesting; `is_tensor_prefetcher_supported() == true` natively, override unset (run 35401066774) |
| Kernel qualification | Cleared. Upstream `test_prefetcher_BH_validator` **30/30 passed** — byte-for-byte, both sender paths, both layouts |
| Combined value | **Failed.** 24.7% slower than the native 1D control at matched core count |

## The measurement

Same unit of work both sides — gate, up and their product, M=32, two p150a:

```
PREFETCH (padded 8960, ring=40, 40 receivers)   0.2890 ms
NATIVE   (unpadded 8704, 39 workers)            0.2317 ms
ratio 1.247  ->  prefetched is 24.7% SLOWER
```

Three repeats each, cleanly separated, no overlap: prefetch
`[0.2871, 0.2890, 0.2897]` vs native `[0.2308, 0.2317, 0.2325]`. Host io pressure
0.06. Control is `fused_1d.native_gate_up_control` — two `mcast_in0` 1D matmuls on
an (11,4) grid plus the multiply, occupying 39 of 44 cores.

M=32 is one tile row, so the matmul is already weight-bandwidth-bound — the
prefetcher's best case, and the regime the 200-TG decode goal lives in. It still
lost. Against the 91-worker fused recipe the gap would be wider, since the
39-worker native control is itself the slower mapping.

Correctness was established first: **PCC 0.978** against torch, which is the
expected cost of 4-bit mantissa weights rather than an error.

## Why 8704 is hostile

`8704 / 32 = 272 = 2^4 x 17`, and that prime 17 drives everything:

```
GCB per receiver = K_tiles * N_tiles * tile_bytes / ring_size
```

A larger ring is *cheaper* — pages shrink quadratically while their count grows
linearly, because the gather-in0 matmul does `wait_front(num_blocks)` and every
page must be resident.

| Case | Ring | GCB per receiver | Fits ~1.5 MB L1 |
| --- | ---: | ---: | --- |
| Qwen gate | 8 | 2.99 MB | no |
| Qwen gate, max legal | **16** | **1.49 MB** | **no** |
| Llama FF1 | 32 | 1.86 MB | no |
| Llama FF1, production | **64** | **0.93 MB** | **yes** |

At native width no legal ring fits L1 at all — the GCB factory refuses to build.
Padding N to 8960 (`280 = 2^3 x 5 x 7`, +2.94%) admits ring=40 at 0.615 MB, which
is how the measurement above was obtained. Upstream's production path depends on a
ring of 64 that Qwen's factorisation forbids.

**Splitting across more cards makes it worse.** Every power-of-two TP split leaves
the 17 in `per_core_N` and halves the achievable ring: TP1 → 32, TP2 → 16, TP4 → 8,
TP8 → 4. More hardware shrinks the ring.

## Pairing rules, learned the expensive way

Not obvious from the signatures, and each cost a hardware run:

- **`num_global_cb_receivers` must be set explicitly.** It defaults to 1, and the
  GCB factory reads receivers-per-bank from the *program config*, not from
  `bank_to_receivers`. Chasing the mapping through column-major, explicit and
  row-major forms could never have fixed it.
- **Receiver-contiguous weight ⇄ strided topology.** Row-major pairs with a
  width-sharded weight. Mixing them runs clean and silently returns **PCC 0.025**.
- **The GCB and ordinary matmuls cannot share receiver cores.** The GCB is resident
  L1; an unprefetched op on the same cores fails with *"statically allocated
  circular buffers clash with L1 buffers"*. This alone complicates any serving
  integration.
- **`prefetcher_common.bytes_per_tile` has no bfloat4_b entry**, so upstream never
  exercises the dtype gate/up actually use. bf4_b tile is 576 B (512 mantissa + 64
  exponent).

## Retracted

An earlier figure of **3.15x faster** was measured against
`dram_sharded_projection`, whose own module documents it as having no serving
integration and which runs four workers against forty. That was a core-count
artefact, not a prefetcher win. The 24.7%-slower figure above, at matched core
count against a real path, supersedes it.

## Reopen only if

- An intermediate size with more small factors than `2^4 x 17` makes a larger ring
  reachable, or
- `in0_block_w` becomes tunable on the DRISC path. It is forced to 1 there (the
  factory's kbw default) against the native control's 8, so a tensor becomes 160
  K-blocks instead of 20. That asymmetry is the leading hypothesis for the loss.

## Operational notes

- Repeatedly opening and closing the cards within one CI job wedged a board:
  `Read 0xffffffff over PCIe ID 2: the board should be reset`. Recovered with
  `tt-smi -r /dev/tenstorrent/2` (run 35404089371). **One device-open per job.**
- The p150a runner sits in org runner group 3 with `restricted_to_workflows: true`.
  An unlisted `workflow.yml@refs/tags/<tag>` **queues forever with no error**. Worse,
  a PATCH to that group silently wipes its five `selected_repositories` — always
  restore them afterwards.
- Card B (`f4:00.0`) is absent from PCI entirely, `PresDet-` at switch port
  `f2:01.0`. Not a software problem; needs on-site hands.
