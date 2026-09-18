# Fabric weight loading and card-1 tensix reclamation: implementation plan and experimental framework

Status: **plan, no measurements yet.** Serving defaults remain unchanged.
Everything here is opt-in, gated, and follows the repo's simulator-first,
matched-ABBA, exact-output audit conventions.

## 1. Problem statement and hypothesis

Current setup: two P150A cards. Card 0 attaches at PCIe **x16**; card 1 sits
behind a switch at PCIe **x4**. Weights and all host↔device traffic (upload,
readback, dispatch) traverse each card's own PCIe attachment. Card 1's narrow
x4 link is the bottleneck path for its weight load and any host-issued control
traffic, and the host-channel/dispatch core footprint reserved for that
attachment on card 1 is paid for whether or not the link is fast.

**Hypothesis.** If weight loading and host traffic for card 1 are re-routed so
that:

1. the host uploads the checkpoint once, to card 0, over its x16 attachment;
2. card 1 receives its weight copies **over the four-link QSFP ethernet
   fabric** (card 0 → card 1 relay); and
3. card 1's dispatch/host-I/O role is migrated to ethernet dispatch cores
   (building on the existing `optimisation/sim/eth-dispatch-*` core-descriptor
   work), so no tensix cores on card 1 are reserved for a PCIe-attached
   dispatcher role it no longer performs,

then (a) total model load time drops because the x4 path is removed from the
critical load sequence, and (b) the reclaimed tensix cores on card 1 can be
placed into the compute grid for the highest-cost kernels (draft score,
verifier MLP, GDN recurrence), improving TG.

**Non-goals.** This is not a claim that more cores alone raise TG —
[decode-payload-bound](decode-payload-bound.md) shows ~4 TB/s of combined
weight-read bandwidth is required for 200 single-token steps/s; the cores are
useful only where reads, not issue slots, dominate, or where multi-token
verification amortises reads. Nor does it change the serving image or enable
anything by default.

## 2. Current-state evidence (from repo docs, no new runs)

| Observation | Source |
| --- | --- |
| `load_target_once` → first weight upload measured **313.84 s**; text-only load skip did not fix it; host I/O pressure 10–17% | [loading-diagnostic-results](loading-diagnostic-results.md) |
| Card-1 x4 sits behind a switch; fabric is 2× QSFP-DD, 4 links | README |
| Target CCL link count (2 vs 4 links) is TG-neutral at 4K | [target-model-link-counts](target-model-link-counts-2026-09-09.md) |
| Ethernet-dispatch core-descriptor patches already exist (per-device `device_id` in `CoreDescriptor` key, eth-grid dispatch selection, availability filtering) | `optimisation/sim/eth-dispatch-harvesting.patch`, `eth-dispatch-common-pool` |
| Fabric tensix config is already a first-class field in the core descriptor | `tt_metal/llrt/core_descriptor.cpp` (pinned runtime) |
| Weight-load host path currently opens devices before any deserialization probe is possible | loading diagnostic, flatbuffer probe failure |
| Decode is payload-bound: 19.92 GB packed weight per complete step | [decode-payload-bound](decode-payload-bound.md) |

## 3. Phase 0 — measurement baselines (no runtime changes)

Every phase compares against a frozen matched control
([winning-runtime-controls](winning-runtime-controls.md)).

| # | Experiment | Method | Admits go/no-go on |
| --- | --- | --- | --- |
| P0.1 | Achieved PCIe bandwidth, per card | `tt-metal` microbench or `tt-smi` transfer loop, x16 vs x4, cold cache, ≥9 repeats ABBA | whether x4 is actually load-limiting |
| P0.2 | Achieved fabric bandwidth, 4 links | device-to-device NoC/fabric copy benchmark, unidirectional and bidirectional | whether fabric ≥ x4 (target: ≥ x4 sustained) |
| P0.3 | Per-card load timeline | instrument existing `load_target_once` path: file read, deserialize, convert, upload, per tensor family; both cards | where the 313.84 s actually goes |
| P0.4 | Card-1 core inventory | dump logical core maps under current dispatch config vs eth-dispatch candidate; count cores reserved for dispatch/host I/O on card 1 | how many cores are reclaimable, and for what shapes |

Decision gate: proceed only if **P0.2 ≥ P0.1(x4)** and P0.3 shows card 1's
upload is a material share of load time.

## 4. Phase 1 — fabric weight relay (load-time work only)

Change the *loading* path, not the serving loop. Opt-in flag, e.g.
`WEIGHT_LOAD_FABRIC_RELAY=1`.

1. Host reads each cached tensor once (bounded reads already proven fast:
   33.2 ms for 64 MiB — [loading-diagnostic-results](loading-diagnostic-results.md)).
2. Deserialize/convert on the host **once**; upload to card 0 over x16.
3. Card 0 pushes the card-1 shard to card 1 over the four-link fabric using a
   NoC/fabric unicast write (no host round-trip, no dispatch-core involvement
   on card 1's PCIe path).
4. Layout, sharding and addresses on card 1 must be bit-identical to the
   control: the gate compares full device-memory digests per tensor family on
   both cards against the PCIe-load control (the repo already does 120-exact
   weight-check audits).

**Admission gate:** complete native source digest reconciliation; both-chip
weight digest equality; clean close. Timing is reported separately and never
promotes on its own. Success criterion: card-1 load time improves by ≥ the
x16/x4 bandwidth ratio predicted in P0.1, with exact state equality.

Fallback if fabric copy cannot reach device DRAM directly: stage through card-0
DRAM then fabric-writes; measure the extra hop against P0.2 numbers.

## 5. Phase 2 — host-I/O role concentration on card 0

With weights arriving by fabric, card 1's remaining host traffic is control
plane and result readback. Two sub-experiments:

- **P2.a — ethernet dispatch on card 1.** Extend the existing
  `eth-dispatch-harvesting` descriptor change: card 1 runs dispatch on ETH
  cores; card 0 keeps its current config. Must pass the simulator trace-replay
  gate first (84-check harness, as the harvesting patch did), then hardware
  boundary + trace replay, then combined exact-output audit.
- **P2.b — readback path.** Verifier/draft readbacks on card 1 are relayed
  card1 → fabric → card0 → host. Measure added latency against TG; if the
  added hop exceeds ~1 ms/block on the 33.75 ms/block budget (from
  [two-card-experiment-programme](two-card-experiment-programme.md)), reject.

**Admission gate:** the existing three-stage ladder (simulator → hardware
replay → combined model with exact output/state and PP/CTX/TG) as used for
every promotion in this repo.

## 6. Phase 3 — reclaiming card-1 tensix for high-cost kernels

Only after Phase 2 passes: card 1's dispatch-host role is gone; enumerate the
freed logical tensix grid (P0.4 inventory) and re-place one high-cost kernel
at a time. Priorities from the current profile
([current-verifier-profile](current-verifier-profile-2026-09-09.md),
[dspark-draft-profile](dspark-draft-profile-2026-09-11.md)):

| Kernel | Cost today | Why wider grids may help |
| --- | ---: | --- |
| Verifier T16 readback/verification | 81–82 ms/block | read-amortised; more cores increase DRAM read concurrency |
| DSpark draft (score/conv) | 55–87 ms/block | recurrence+score are compute-latency bound, not payload bound |
| GDN norm/recurrence | in-draft | already worker-partitioned; wider grid is a single flag change in worker count |

Rules, per repo precedent:

- One kernel per experiment; ABBA matched blocks (nine-block pattern); native
  controls unchanged; the fabric relay stays opt-in throughout.
- A candidate that wins a layer timing but loses the complete combined loop is
  **not promoted** (precedent: worker-limit candidate, fixed-packet readers,
  DRAM-sharded MLP).
- Wider grids must show they are not read-bandwidth-starved
  ([weight-read-packets](weight-read-packets-2026-09-10.md): sixteen producers
  were *slower*; extra cores only help where the kernel is latency-bound).

## 7. Experimental framework summary

| Stage | Evidence required | Cost |
| --- | --- | --- |
| Baselines | P0.1–P0.4, recorded run IDs, host snapshots | cheap, no runtime change |
| Load relay | weight digest equality both chips + load timing | one build-cache hit + 2 hardware runs |
| Eth dispatch | 84-check sim replay → hardware replay → combined exact audit | existing ladder |
| Kernel placement | per-kernel ABBA → complete combined PP/CTX/TG → matched context ladder | full programme |

Every experiment lands as one doc in `docs/` with: purpose, matched-control
table, exact run IDs, result SHA256, and an explicit "do not promote" line when
it loses — matching the house style. CI stays green; serving defaults
unchanged.

## 8. Risks

- **Fabric bandwidth may be below x4 sustained** (4 links × ~line rate, minus
  CCL and verifier traffic sharing the same links during decode). Phase 0
  measures this first; if fabric ≈ x4, the relay only wins during load, and
  Phase 2/3 stand or fall on their own merits.
- **The 4 links are already used by the TP2 collectives** during execution;
  weight relay is load-time only, so no decode contention, but P2.b readback
  relay does contend — hence the measured gate.
- **Device renumbering after `tt-smi -r`** ([gotchas](gotchas.md)) — resolve
  `by-id` after any reset; cabling graph must be the known `(1,2)` pair.
- **Core-descriptor cache key changes** (the harvesting patch adds
  `device_id` to the hash) can silently change which config is reused; the
  sim trace-replay gate exists for exactly this.
