# Prefill/decode disaggregation on the four p150a cards: model and step-1 window (branch tp4/disagg)

Status: a design, a model (`docs/tp4-disagg-model.py`) and card-job templates (`scripts/ci/references/tp4-disagg-jobs`). There is no transfer code, and
nothing here has run on cards. Every speed below is a MODEL output. The inputs are marked measured (M) or estimated (E) in `--doc`.

## Verdict (model)

A static 2P+2D split (one TP2 prefill pair and one TP2 decode pair, llm-d style) does not beat today's TP4 pool on agent traffic. It wins only in one narrow regime:
- many busy agents;
- prompts short enough for 8 seats to fit in the decode pair's KV;
- a TP2 decode round costing at most about 1.6x a TP4 round.

Everywhere else it loses:
- at light load, and for a lone user;
- on long-context sessions, because the decode pair holds about a third of the TP4 pool's KV;
- on decode-heavy traffic.

It never raises the steady per-seat decode rate (the 8 x 75 target): a TP2 round is k times a TP4 round, and isolation only gives back what Lever N's interleave costs. It also cannot be byte-exact against today's TP4 baseline (section 5).

## 1. Layouts on four cards

- **TP3 is invalid.** None of these divide by 3: 4 KV heads, 16 GDN key heads, hidden 5120, intermediate 17408 (config.json).
  - So 1P+3D and 3P+1D would each need a TP3 side or a TP1 side.
  - A TP1 side is invalid too: about 28.6 GB of bf8 weights leaves no room on a 32 GB card for an engine, the scratch and a 32k KV (1.1 GB).
- **2P+2D (TP2 + TP2) is the only static split.**
  - Both sides shard KV and GDN state the same way, two KV heads and 24 GDN value heads per chip, so chip i of the prefill pair hands its shard to chip i of the decode pair over its own two-link cable. The cabling is a full mesh with two links per pair.
  - Only general-2link opens a pair on this cabling.
  - The fast path (qwen_fast_t16) is refused on a two-link pair (serving_c2_contract.mesh_problems), and no pair profile serves 8 seats.
- **Dynamic role switching:**
  - (a) One TP4 pool time-shared between the roles is what Lever N already does. Its prefill share f is the knob; the model's f = 0.3 and f = 0.7 rows show the trade.
  - (b) Two TP2 pools that swap roles become two colocated TP2 pools of 4 seats each, with work stealing (the model's "2 x TP2 colocated" row).

## 2. What moves per request, and where it lands

Per request: the bf8 KV of the 16 attention layers (34,816 B/token), plus the GDN recurrent state and conv carry of the 48 GDN layers (fp32, 153,944,064 B: qwen_prefix_registry.py), plus the DFlash2 taps for the 2048-token window (105 MB bf16). The taps can be 0 if the decode side re-prefills its last 2048 tokens.

| prompt | total | fabric 160 GB/s (ceiling) | fabric 40 GB/s (planning) | host-staged 2 GB/s | host via to_torch 0.09 GB/s | TP4 / TP2 prefill (model) |
|---|---|---|---|---|---|---|
| 4k | 0.40 GB | 3 ms | 10 ms | 0.20 s | 4.5 s | 0.6 / 1.0 s |
| 32k | 1.40 GB | 9 ms | 35 ms | 0.70 s | 16 s | 5.9 / 9.4 s |
| 128k | 4.82 GB | 30 ms | 0.12 s | 2.4 s | 54 s | 35 / 56 s |
| 262k | 9.39 GB | 59 ms | 0.23 s | 4.7 s | 104 s | 101 / 162 s |

Where the bandwidth figures come from:
- Fabric (M): a pair all-gather moves 84-90 GB/s on 2 links (50 GB/s each). There is no point-to-point or cross-mesh op anywhere in our code, and whether the pinned tt-metal has one is unverified.
- Host (M): checkpoint restore runs at 1.9-2.7 GB/s host-to-device and capture at 3.6-7.7 GB/s device-to-host. Two of the four cards sit on x4 PCIe links.
- No op writes host data into the KV pool. The only KV read paths are `ttnn.qwen_read_blocks` (unqualified) and the whole-pool `to_torch`.

Natural import points:
- **Engine reuse:** `serving_parked_engines.rebind_device` (it rezeroes, projects the taps and reseeds the drafter K/V), `VerifierEngineTP.rebind(session, pages)`, and `adopt_prefill_slot`. The contract is: KV in the request's blocks, the GDN state in native slot 0, the taps on device.
- **Prefix reuse:** `PrefixRegistry.put` plus vLLM's cached blocks. Pre-populating both lets an ordinary hit re-prefill [Q, P) locally. This is the lowest-friction route, and it stays exact by construction.
- **Lever N park and restore:** `_qwen_prefix_read_scratch` / `_qwen_prefix_restore`.

A vLLM KV connector is not an option without new work:
- the TT plugin has no connector hooks;
- the KV caches are ttnn tensors;
- vLLM never sees the GDN state;
- our grafts refuse a connector (qwen_prefix_scheduler_patch.py, serving_c2_contract.py).

## 3. Model

**Inputs.** The model replays the same per-seat turn scripts through two serving models (`--doc` lists every input):
- **The TP4 pool.** Lever N as served: f = 0.5, short lane at R = 1, solo steps of 16k, T* = 180 s, an 8 s floor, engine reuse, prefix reuse. Its inputs are the measured per-step and per-round medians of the production profile (run 37993848597).
- **2P+2D.** The prefill pair runs at r x the TP4 step time and the decode pair at k x the TP4 round time. State moves over a link, and the decode pair's KV pool is capped.

**Calibration.** parked_turns:
- makespan: model 228 s, measured 244.5-248.7 s;
- decode per turn: model 12.6 tok/s, measured 12.5-14.2;
- TTFT max: model 83 s, measured 74-84.

concurrent8_skew, slowest first token: model 252 s, measured 215-237 s.

**Results.** Means over 4 seeds, 8 seats, 60 turns per seat. Each cell gives committed tok/s / decode tok/s per seat / TTFT p50 / TTFT p99. The central case is r 1.6, k 1.8, fabric 40 GB/s, remote prefill of every tail, decode KV 400k. The range is the pessimistic arm (r 2.0, k 2.0, 10 GB/s, 300k) to the optimistic arm (r 1.5, k 1.6, 160 GB/s, 600k).

| traffic | TP4 pool today | 2P+2D central | 2P+2D committed, range |
|---|---|---|---|
| agents, 8 busy (gaps 2 s) | 47.3 / 12.0 / 3.7 / 93 | 51.2 / 11.7 / 5.2 / 50 | 45.3-55.4 |
| agents, gaps x10 | 25.2 / 27.3 / 2.7 / 36 | 23.9 / 20.1 / 3.8 / 28 | 23.1-24.5 |
| one agent alone | 15.9 / 29.2 / 2.1 / 9 | 10.3 / 16.2 / 3.2 / 14 | 9.3-11.2 |
| long sessions (60-240k) | 51.0 / 11.3 / 4.7 / 187 | 33.5 / 22.1 / 11.0 / 473 | 20.7-46.9 |
| cold 40k prompts, no reuse | 20.3 / 13.9 / 134 / 309 | 19.5 / 21.4 / 170 / 456 | 15.6-20.7 |
| 2k-token answers | 133.8 / 18.7 / 2.7 / 12 | 82.4 / 11.3 / 3.2 / 10 | 72.6-92.2 |

**Same busy-agent mix, cheaper alternatives:**
- Cutting the TP4 admission from 0.9 s to 0.3 s: 53.6 / 13.0 / 2.8 / 82.
- Two colocated TP2 pools: 52.8 / 13.2 / 4.1 / 65.
- An ideal disaggregation (free transfer, no cap, no admission cost) reaches 67.0. Most of that gap is the admission stall, which a TP4 lever can attack directly.

**What decides it.** The decode factor k decides it, then the decode pair's KV:
- **k:** on the busy mix, committed tokens per second run x1.07-1.23 at k 1.5-1.6, x1.04-1.11 at k 1.8, and x0.97-1.02 at k 2.0, for r anywhere in 1.4-2.0.
- **KV capacity:** long sessions drop to x0.66 at 400k and x0.36 at 250k.
- **Measured anchors for k:** k is 1.51-1.63 on the G1 path (TP2 vs TP4 on image tp4-1). No like-for-like fast-path figure exists. The pair-era S2 4 x 131k rounds were 162-168 ms, with none of the W1/W2 levers.

## 4. Step-1 window (no code, no build)

`scripts/ci/references/tp4-disagg-jobs` on image tp4-next2-1 (or tp4-octo2-1), run in this order:

1. **X0:** a four-card reset.
2. **D2:** the TP2 pair on general-2link.
   - Prefill at 4k/32k/60k.
   - Decode at 1/2/4 live.
   - Agreement.
3. **D1:** the TP4 G1 comparator, general-tp4-bench.
   - The same shapes and prompts, plus 1 x 130k and 8 live.
   - Agreement.
4. **D3:** the production profile.
   - Prefill at 4k/32k/128k/250k.
   - Decode at 1/2/4/8 live.
   - The decoders' rate inside a cold 120k prefill.
   - concurrent8_steady.
5. **Z:** a reset.

That is about 3.5 h of cards. Production must be down for it, and CI paused.

**Results to read off:**
- r = D2 / D1 time to first token.
- k = D1 / D2 steady rate.
- The TP4 fast-path absolutes come from D3.

**Pre-registered decision:**
- k >= 1.75 at 4 live closes 2P+2D.
- k <= 1.6 together with r <= 1.6 opens step 1b: a fast-path TP2 pair on the two-link cabling, to measure the fast path's own k before any transfer code.

**Not measurable on this cabling with existing profiles:** a TP2 prefill above 60k, 8 seats on a pair, and the fast path at TP2.

## 5. Step 2 (only if step 1b passes) and exactness

**Components:**

| # | work | effort (E) |
|---|---|---|
| (a) | The fast path at TP2 on two links: contract, link policy, evidence at C = 262,144, and the W2 / Lever N / engine-reuse stack at 2 chips | 15-25 d |
| (b) | An 8-seat TP2 decode pool and its memory ledger | 5-10 d |
| (c) | One process holding two (1, 2) submeshes of the ring, or two processes with host staging | 10-20 d |
| (d) | A KV block write op plus a GDN/conv/taps hand-off, fabric or host | 8-15 d |
| (e) | Router and scheduler: the session prefix lives on the decode pair, which the prefill pair pulls | 5-10 d |
| (f) | Exactness gates | 5-8 d plus cards |

In all, about 50-90 engineering days.

**Exactness.** TP2 and TP4 greedy outputs are not bit-equal in this stack, which docs/tp4-g1-bringup.md records ("TP4 reduces in a different order from TP2, so its greedy tokens are not bit-equal to the pair's"):
- J2 against J2r: the first divergence came at tokens 19-54, on near-ties, with a perplexity ratio of 0.988-1.009.
- docs/tp4-exact-ring-parity.md: the ring reduce-scatter sums four partials in a direction-dependent order, whereas at two chips the sum commutes.

So a TP2 + TP2 system can be exact only against a TP2 colocated cold run. The hand-off itself is a byte copy; the prefill must be chunked at the same 2048 boundaries, and the decode must start from the same state. It can never be exact against today's TP4 cold run, so adopting it re-bases every user's output.
