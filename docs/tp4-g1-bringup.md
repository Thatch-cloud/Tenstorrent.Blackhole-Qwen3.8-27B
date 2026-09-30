# G1 at TP4 on four cards: what is built, what is unverified, and the hardware jobs

Status: built and CPU-tested on branch tp4/bringup; nothing here has run on hardware. Every number below is marked
(a) arithmetic from this repository's own records, (m) measured earlier on the pair, or (U) unverified until a job
in `scripts/ci/references/tp4-jobs/` runs.

## What TP4 is here

Four p150a boards opened as one (1, 4) mesh. Every board pair is cabled with two Ethernet links (a full mesh), the
model runs its own Ring collectives at 2 links (tt_ccl's P150x4 entry), and the profile family `general-tp4`,
`general-prefix-tp4` (and the `-131k` twins, and the gate-only `general-tp4-bench`) selects it through
`mesh_device: P150x4`. Pair profiles name no mesh and are byte-identical in behaviour.

| Piece | Where | Note |
|---|---|---|
| Descriptor | `scripts/ci/qwen_p150x4_ring_mesh_graph_descriptor.textproto`, `tp4_mesh.py` | 2x2 with 2 channels, no chip id: the plugin sets FABRIC_1D and a non-Galaxy fabric builds MESH connectivity, so a 1x4 ring descriptor would lose its wrap edge; a 2x2 makes every edge of the 4-cycle a mesh edge and `MeshShape(1, 4)` maps onto it ring-first (as upstream runs this model at P150x4) |
| Ring check | `tp4_mesh.check_ring`, installed by the contract | after the mesh opens, the device order is held against the cluster descriptor's trained links: a missing ring edge stops the engine, a single-link edge is logged |
| Contract | `serving_c2_contract.py` item 8 | sets MESH_DEVICE, refuses a mesh/descriptor mismatch, the fast path off the pair, and on-device sampling where the vocabulary shard exceeds 65,536 logits |
| Link policies | `mesh_link_policy.py` | the pinned pair policies (sha256-held by recorded evidence) are untouched; at (1, 2) this delegates to them, at (1, 4) the same scoped rules take 1 or 2 links |
| Sampling | profiles: `sample_on_device_mode: decode_only` | 248,320 / 4 = 62,080 <= 65,536 (a); host sampling per batch for what the device sampler cannot do; `QWEN_HOST_SAMPLING=1` or `/models/.qwen-c2/device-sampling.off` drops it |
| Model graft | `docker/qwen-c2-graft` | every per-device width is tp-generic (test_tp4_model_widths); `gdn_prefill_conv_exact` admits (1, 4); weight cache key `_mesh1x4` (a new ~27.5 GiB); the fused prefill out-projection is enabled at 4 devices, `QWEN_GDN_PREFILL_MMRS=0` takes it out without a rebuild |
| Card set in CI | `scripts/ci/card_set.sh`, `qwen-c2-serving.yml` `C2_CARDS=quad` | every Blackhole board present, by board id, resolved when used; one tt-smi reset call over all four with the heal; `tp4_fabric_probe.py` |

## Pool sizing (a)

KV per chip halves at TP4: one KV head per chip, 16 attention layers x (K, V) x 256 x 1.0625 B (bf8) = 8,704 B per
token per chip, against 17,408 at TP2. The serving pool of 524,288 tokens (8 x 65,536, or 4 x 131,072) is therefore
4.56 GB per chip, the same KV per chip as the pair's 4 x 65,536 G1. Weights are ~7.4 GB per chip (27.49 GiB / 4), the
non-weight non-KV working set is bounded by the pair's ~6.9 GB (assumed not to shrink), so ~14.3 GB of the ~33.9 GB
a chip allocates is spoken for before KV and ~15 GB per chip is free (U until G5 at TP4 measures it). The bench
profile, 8 x 131,072 = 1,048,576 tokens, is 9.13 GB per chip and gate only. Eight seats fit the GDN core waves: 8 x
12 value heads = 96 (user, head) pairs, one wave of 110 (the pair's 8 x 24 needs two).

## What speed to expect (a, from the pair's own decomposition)

G1 on the pair (m): 18.6-19.3 tok/s single stream, 12-13 tok/s each at four users. The 54 ms step splits into ~31
ms that scales with weights per chip (MLP, GDN and attention projections, LM head) and ~23 ms fixed per layer (small
op floors, launch, collectives). Halving the first and not the second gives ~38-40 ms, about 25 tok/s single stream
(+30%); doubling aggregate DRAM bandwidth halves the weight-streaming floor (`decode-payload-bound.md`: ~19.9 GB per
step) but not the fixed part, so a figure well above that means the fixed part shrank too. The larger gains are
capacity: eight seats at 64k, or four at 131k, on the same KV per chip. (U) until J3 measures it.

## Hardware jobs (`scripts/ci/references/tp4-jobs/`)

| Job | Question | Est. wall |
|---|---|---|
| J0 (J0b) | four boards enumerate, ring OK at 2 links, tt_ccl at 2 links, collectives exact, GB/s | 30-40 min |
| J1 | image build with the TP4 overlay (no card) | 60-125 min |
| J2r, J2 | pair reference and TP4 smoke: coherent text, token agreement (`tp_agreement.py`) | 45-75, 60-110 min |
| J3r, J3 | decode benchmark, pair and TP4 (`tp_decode_bench.py`): 1 stream 4k-130k, 4 and 8 streams, prefill 130k | 30-60, 80-150 min |
| J4 | prefix-reuse bringup gate at TP4 | 60-120 min |

TP4 reduces in a different order from TP2, so its greedy tokens are not bit-equal to the pair's: the agreement
report gives the common prefix before the first flip, coherence, and the ratio of the perplexity each side assigns
to its own text. Its thresholds are a first guess to be revised from J2's data.

## Open risks (U)

1. Whether `MeshShape(1, 4)` maps ring-first on the 2x2 descriptor on this fabric, and whether FABRIC_1D accepts it
   (upstream issue 49701 is how a 2x2 open on four p150a has failed elsewhere).
2. The fused matmul + reduce-scatter prefill out-projection at four devices (it deadlocked on the pair).
3. The GDN kernels were qualified at 24 value heads per chip and run 12 here; the exact prefill conv is held to the
   FIR on hardware by its audit switch.
4. The fast path (C2, S2 sticky sessions) is TP2 only: its kernels, per-chip literals and evidence are two-chip.
   TP4 serves the general profiles; porting the fast path is separate work and the contract refuses it on a ring mesh.
5. The 1x4 weight cache is a new ~28 GiB; check the disk before the first boot.
