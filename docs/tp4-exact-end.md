# Exact-end resume (TP4): why a returning turn cannot yet prefill only its new tokens, and what to do instead

**Date:** 2026-10-19. **Branch:** `tp4/exact-end`, which is `tp4/swift` (27af6902, the ship-4 stack with the Swift checkpoint) merged with
`tp4/exact-resume` (4f141594, exact resume P0, built and never run on a card). The merge keeps P0's nine commits and its history intact (`tp4/exact-resume` was 67 commits behind `tp4/swift`; a rebase
stopped at its second commit on conflicts in the prefix modules, so the nine commits were merged instead, once: 13 files, all additions on both sides: the tier and quota plans beside the
exact-resume ones, the checkpoint class, the scheduler graft, the oracle's signature). **Status:** research and CPU work only. Nothing in this note has run on a
card; every number is labelled **measured** (a card artifact or production log), **modelled** (arithmetic on measured medians, `scripts/ci/exact_end_model.py`)
or **argued** (read from code, to be settled by the card job named next to it).

Companion notes: `docs/tp4-exact-resume.md` (the scoping, P0, the telemetry), `docs/tp4-packed-prefix.md` (sticky sessions), `docs/lever-n-tp4-design-2026-10-04.md`
(the chunk loop's exactness argument), `docs/real-text-2026-09-24.md` (what rounding-level differences do to real text).

## 0. Answers first

**The evidence.** On the ship-4 + Swift production stack 58% of card wall time is prefill. A returning turn grows its prompt by about 170 tokens
(164,887 -> 165,056 -> 165,225 -> 165,394) and resumes at the same checkpoint (161,792 = `floor2048(P_prev) - 2048`), so each turn re-prefills `P - Q` = 3,264 to 3,602
tokens, 3,095 to 3,433 of them old (2.95 million over 996 turns: 2,962 a turn; the model's expectation for a uniform `P_prev mod 2048` is 3,072).

| Question | Short answer |
| --- | --- |
| 1. Why is a non-2048 resume not bit-identical? | A program's arithmetic is chosen by its **bucket** (matmul program configs are a function of the row count, `graft/model_config.py:269`), and the GDN state is **stored** between programs in bf16 (`QWEN35_GDN_STATE_BF16=1`, the serving setting) where the cold prefill makes no store. A resume inside a 2048 cell runs another bucket on a state that was stored where the cold run does not store. Which of the two matters is **not settled by the one card probe that ran** (v839 mixes them); `tp4_split_probe.py --attribute` (job SP1, 30 minutes, no build) separates them. Section 2. |
| 2. Can the engine keep an end-of-turn state so that only the new tokens are prefilled, byte-exactly? | **No, not without a kernel or a numerics change, and nothing finer than P0 is worth building without one.** A byte-exact resume has to run the cold prefill's own programs on the cold prefill's own bytes, and a program costs its *bucket*, not its valid rows. The cold prefill's tail is one masked bucket (`bucket_for(P mod 2048)`); P0 already runs exactly that one program and nothing else. Resuming later inside the tail (a 128-aligned state, "intermediate per-chunk partial state") saves nothing unless a SMALLER bucket than the cold run's is numerically equal to it; the card probe that ran did not establish that (its arms mix two causes) and the code gives no reason to expect it (matmul configs are chosen by the bucket). Section 3. |
| 3. The numerically-equivalent option (resume at the decode-time state)? | Needs an owner decision **and kernel work** (the usable state is at a 128 boundary, not at `P_prev`; today's kernels give it only by an extra forward on turn N that cancels the gain). The most it can add over P0 is 12% of today's prefill time (about 7% of card wall time). The only past measurement of what rounding-level prefill differences do to real text is bad: 8 of 8 answers diverged within 256 tokens, 1 of 4 first tokens differed (`docs/real-text-2026-09-24.md` section 3). Section 4 has the facts and a gate. |
| 4. Interactions | P0 composes with the checkpoint store, the host tier, engine reuse, Lever N and the tile-aligned window, with **one defect found and fixed by this branch**: merging P0 onto the ship-4 stack had switched the tile-aligned drafter window off under a carry, which brings back the per-prompt-length concat compile on every carried turn. Section 5. |

**Recommendation (ranked, section 6).** (1) Take P0 to cards on the production stack and checkpoint: the `XE` pack, `scripts/ci/references/fusion-jobs/XE`
(modelled **-44% of returning-turn prefill time, -67% of re-prefilled old tokens, 2 forwards -> 1**). (2) Run SP1 first, in the same window, because it decides whether a
finer resume is a kernel job or a configuration job. (3) Do not pursue Option B (E1) now. (4) Spend the next effort on the fixed floor around the one remaining forward.

**Expected prefill reduction on the production sequence (modelled, section 1.3).** Per returning turn at 165k context: today 2 forwards (4,096 rows, 2.06 s) -> P0 1 forward (2,048 rows, 1.17 s).
Over random 170-token sessions (contexts 20k-250k): old tokens re-prefilled 3,068 -> 1,020 a turn (**-67%**), modelled prefill seconds 1.80 -> 1.01 (**-44%**). If all of the 58% is returning-turn prefill,
that is **-26% of card wall time** (an upper bound: the 58% includes cold first turns that P0 cannot touch); at a 40% / 30% share it is -18% / -13%.

**Owner decision needed (plainly, section 7).** One: allow a card window for the XE pack (about 9.5 hours of cards with everything, about 6 without the host-tier and eight-seat jobs; the cards serve production, as for every window) so that P0 is
proven byte-identical to cold prefills on real multi-turn text on Swift and timed against today. A yes only authorises the window: shipping P0 needs a traffic waiver for two switches that are gate-only
today, decided after the window reads. Two: nothing yet on Option B; the facts and a proposed gate are in section 4 for the day it is wanted.

## 1. How a prefill is built, and what the production sequence costs

### 1.1 The grid (read from the code)

| Fact | Where |
| --- | --- |
| A prefill of P tokens is a list of programs `(chunk_start, bucket, valid)`: every full 2048-row chunk is bucket 2048 with `valid_len == bucket == 2048` ("no padding/masking"), then the tail `t = P mod 2048` is ONE masked bucket, the smallest of (128, 256, 512, 1024, 2048) that holds it, with `valid_len = t`. | `scripts/ci/fixtures/qwen36_model.py:1943-1950` (`_PREFILL_MASK_BUCKETS`, `_mask_bucket_for`), `:2442-2490` (`_prefill_chunked_eager_tp`), `:2090-2160` (`prefill_masked_bucket`); the fast path's Lever N route calls `prefill_traced_chunked(tokens[:end], actual_len=end, start=...)` (`levern_route.py:732, 937, 1077`) |
| A full chunk and a masked tail go through the same function, `_forward_prefill_chunk_masked_tp`, and the same GDN code path (`valid_len` is an int in both; the native conv1d is for `valid_len is None` only). They differ in the bucket (M) and in the mask. | `qwen36_model.py:2010-2088`, `graft/gdn/tp.py:579-600` |
| **A program costs its bucket, not its valid rows**: the padded rows are computed. A tail of 1,047 rows and one of 2,047 both run the 2048 program (50% of tails do: `t > 1024`). | `qwen36_model.py:2090-2160`; measured per-forward medians below |
| Matmul program configs are a function of the row count: `prefill_progcfg = lambda seq_len, k, n: create_prefill_matmul_program_config(seq_len, k, n, ...)`. Another bucket is another config (grid, K blocking, per-core M). | `docker/qwen-c2-graft/graft/model_config.py:266-270` |
| The GDN scan runs 128-row sub-chunks counted from the program's first row; the recurrent state between programs is stored in bf16 in serving (`QWEN35_GDN_STATE_BF16=1`, the image's ENV; `docs/prefix-store-hygiene.md`: 78,446,592 bytes a checkpoint) and copied in place at the end of each program. | `graft/model_config.py:156`, `graft/gdn/tp.py:303-318, 660-665` |
| The flexible chunked SDPA runs `q_chunk = k_chunk = 128` and assumes `chunk_start % 128 == 0`; the KV fill writes whole 64-token blocks from `chunk_start // 64`. A start off 128 is wrong SDPA, off 64 silent KV corruption. | `graft/attention/tp.py:854-857`, `qwen36_model.py:2049-2053` |
| A resume starts only on a 2048 boundary: every layer refuses another. Lever N's final step starts at `S_last = floor2048(P) - 2048` when P has a tail, because the drafter window `[P - 2048, P)` must be computed by the step's own chunks; that is why today's `Q = floor2048(P_prev) - 2048`. | `levern_policy.py:406-415, 449-470`, `dflash_prefill_window.py:114-130` |
| P0 relaxes the last rule (window carry) and the undrop keeps vLLM's last matched block, so `Q = floor2048(P_prev)`. | `qwen_prefix_scheduler_patch.py:1492-1506`, `prefix_window_carry.py` |

`exact_end_model.py` encodes this grid; `test_exact_end` holds it against `prefix_judge.Oracle` (the gates' prediction of the graft's grants) and against the probe's arms.

### 1.2 Measured per-forward costs (docs/tp4-exact-resume.md 4.2, solo medians)

| context | full 2048 forward (not last) | final forward T=128 | final forward T=2048 |
| --- | --- | --- | --- |
| 32k | 0.42 s | 0.36 s | 0.53 s |
| 128k | 0.71 s | 0.49 s | 0.83 s |
| 254k | 1.20 s | 0.71 s | 1.34 s |

A small program is not cheap: at 128k a 128-row final forward costs 59% of the 2048-row one (eager dispatch of 64 layers is the floor). The model interpolates linearly between the
T=128 and T=2048 medians (the T=1024 medians, 0.42 s / 0.73 s at 32k / 128k, sit 0.01-0.08 s above the line) and adds the Lever N step setup (0.2 s, one final step per resumed prefill).

### 1.3 The production sequence, and the other options on it (modelled)

Prompts 164,887 / 165,056 / 165,225 / 165,394 (cell 80 = `[163,840, 165,888)`; each tail is a 2048-bucket program), then the same session carried on 8 turns to show the crossing of 165,888:

| turn P | today: Q, tail, old, programs | P0: Q, tail, old, programs | 128-aligned (Option A3/B): Q, programs | rows today / P0 / 128 | modelled s today / P0 / 128 |
| --- | --- | --- | --- | --- | --- |
| 165,056 | 161,792, 3,264, 3,095, chunk + (163,840, 2048, 1216) | **163,840**, 1,216, 1,047, (163,840, 2048, 1216) | 164,864, (164,864, 256, 192) | 4,096 / 2,048 / 256 | 2.06 / 1.17 / 0.78 |
| 165,225 | 161,792, 3,433, 3,264, 2 forwards | 163,840, 1,385, 1,216, 1 forward | 164,992, (164,992, 256, 233) | 4,096 / 2,048 / 256 | 2.06 / 1.17 / 0.78 |
| 165,394 | 161,792, 3,602, 3,433, 2 forwards | 163,840, 1,554, 1,385, 1 forward | 165,120, (165,120, 512, 274) | 4,096 / 2,048 / 512 | 2.06 / 1.17 / 0.84 |
| 165,563 | 161,792, 3,771 | 163,840, 1,723 | 165,376, (165,376, 256, 187) | 4,096 / 2,048 / 256 | 2.06 / 1.17 / 0.78 |
| 165,732 | 161,792, 3,940 | 163,840, 1,892 | 165,504, (165,504, 256, 228) | 4,096 / 2,048 / 256 | 2.06 / 1.17 / 0.78 |
| 165,901 (crosses 165,888) | 161,792, 4,109, 3 programs | 163,840, 2,061: (163,840, 2048, 2048) + (165,888, 128, 13), 2 forwards | 165,632, 2 programs | 4,224 / 2,176 / 384 | 2.54 / 1.65 / 1.33 |
| 166,070 (next cell) | 163,840, 2,230, 2 programs | **165,888**, 182, (165,888, 256, 182) | 165,888, same | 2,304 / 256 / 256 | 1.68 / 0.78 / 0.78 |
| 166,239 | 163,840, 2,399 | 165,888, 351, (165,888, 512, 351) | 166,016, (166,016, 256, 223) | 2,560 / 512 / 256 | 1.74 / 0.84 / 0.78 |

(`python3 -B scripts/ci/exact_end_model.py --start 164887 --delta 169 --turns 9`.) The first column reproduces the production evidence exactly (`Q = 161,792`, the same `prefix_judge.Oracle` rule); the crossing
turn is the one P0 cannot make a single forward (the cell's full chunk has not been computed by anyone yet), and the turn after it is the cheapest (a checkpoint at the new boundary, a small tail).

Over random sessions (start uniform in 20k-250k, 170 tokens a turn, 40 returning turns each, `exact_end_model.simulate`):

| policy | old tokens re-prefilled per turn | rows computed per turn | forwards per turn | modelled prefill s per turn | vs today |
| --- | --- | --- | --- | --- | --- |
| today (`floor2048(P_prev) - 2048`) | 3,068 | 3,582 | 2.08 | 1.80 | - |
| **P0** (`floor2048(P_prev)`, byte-exact) | **1,020** | 1,534 | 1.08 | **1.01** | **-44%** |
| 128-aligned resume (Option B, if its state existed for free) | 63 | 333 | 1.08 | 0.79 | -56% |
| prefix-stable tail grid (Option C, section 3.4) | 499 | 1,010 | 1.39 | 1.06 | -41% |

Sensitivity: the P0 saving per returning turn is 0.42 s at 33k, 0.55 s at 65k, 0.71 s at 129k, 0.89 s at 165k, 1.20 s at 250k (36% to 44% of the turn's prefill); with 60 / 400 / 900 new tokens a turn the
reduction is -45% / -42% / -38%. Of 996 turns with 2,962 old tokens each, P0 leaves about 990: **-1.97 million re-prefilled tokens**.

What these numbers do not contain: the engine rebind (0.5-1.3 s, bimodal), the Lever N step setup beyond the one step, the API render and tokenize (0.05-0.4 s growing with P), queueing behind decoders, and the cold
first turn of every conversation. They are modelled from solo medians; the ABAB in the XE pack (XC1 XL1 XC2 XL2) is the measurement.

## 2. Question 1: why a resume off the 2048 grid is not bit-identical

### 2.1 What ran, and what it showed

The split-invariance probe (`tp4_split_probe.py`, job SP0, v839, run 38109818878, 2026-10-11) builds the image's own model with its first real layers (up to and including the first full-attention layer: three
GDN layers and one attention layer), under the production profile, and compares bit for bit against the cold prefill A of a synthetic prompt: A2 (determinism control), C (split at a 2048 multiple, the harness
control), and four splits at 128-aligned positions inside a cell: B+128, B+1024, B+1920 and Bend (a decode-start case). Per B arm the state at R comes from a **separate prefill of `[0, R)`** (its tail in the
bucket the engine picks for `R - floor2048(R)`), then the head segment `[R, ceil2048(R))` runs `prefill_masked_bucket(chunk_start=R)` in the bucket for the head's length, then the grid resumes at `ceil2048(R)`.

Recorded result (the report file `split-probe.json` is not in the repository, its run is not visible to the author of this note, and the numbers below are those noted when it ran; **n is tiny and the model is a
four-layer stub**): controls A2 and C identical (2048-aligned resumes are bit-identical, as Lever N's design argues); every 128-aligned split differs: the GDN state differs in 11 of the 18 layer-states
compared, the largest absolute difference 0.125, the last-position logits differ, **the argmax is equal**. That is a measured DIFFERS verdict; it is not a measurement of how often text differs (a four-layer stub's
argmax says nothing about the 64-layer model's), and it does not say why.

### 2.2 Operation by operation

| Operation | Depends on the resume position? | Why (code) | Status |
| --- | --- | --- | --- |
| Embedding, RoPE | No | per-position cos/sin for `[chunk_start, chunk_start + bucket)` (`qwen36_model.py:2033`); text RoPE is a function of the position alone | argued |
| RMSNorm, SiLU, residual adds, the one-hot last-row select | No | row-wise; the select adds exact zeros (`_masked_bucket_logits_tp`, `:2163`) | argued |
| QKV / gate-up / down / out projections (prefill all-gather matmuls, including the 13x10 `QWEN_FAST_GRID13=prefill` grids) | **Yes, through the bucket** | `prefill_progcfg(seq_len, ...)` picks the program config by M; K blocking decides where the bf16 partial sums are rounded. Another bucket is another arithmetic. | argued from the code; settled by SP1 |
| Paged KV fill, bf8 KV (`QWEN_SDPA_BF8=1`) | Alignment only | 64-token blocks from `chunk_start // 64` (silent corruption off 64). bfloat8_b shares an exponent per 16 consecutive values of a tile-face row, so a row's rounding does not depend on the other rows written in the same program | argued from the format |
| Flexible chunked SDPA | Alignment only | `q_chunk = k_chunk = 128`, `chunk_start % 128 == 0`; each query row runs the online softmax over the same k-chunks in the same order whatever the bucket | argued |
| GDN conv | No | the exact prefill-conv op with a carry of the last 3 inputs (a data selection); card-proven exact on real text (`docs/real-text-2026-09-24.md`, v174 EXACT 4/4) | measured |
| GDN scan, state **hand-off** | **Yes** | the state is stored in bf16 at the end of every program (`graft/gdn/tp.py:303-318, 660-665`). A cold prefill passes every sub-chunk boundary inside one program with the op's internal state; a resume at R stores the state at R and loads it again. Whether the op's internal state at a sub-chunk boundary IS bf16 is in tt-metal's fused chunk kernel (`models/demos/blackhole/qwen36/tt/gdn/fused_chunk.py`), not in this repository | unknown; settled by SP1 |
| GDN scan, **sub-chunk phase and the masked end** | Yes, for an unaligned R | sub-chunks are 128 rows from the program's first row; a program whose `valid_len` ends mid-sub-chunk leaves the state of a partial one. The prompt end `P_prev` is at an arbitrary position, so it is never the cold run's own state for the longer prompt (the cold run makes no store there and its sub-chunk phase differs) | argued; the toy below shows it for every arithmetic |
| Drafter window (taps of `[P - 2048, P)`) | Only through the rows it needs | the taps exist for rows a prefill computes; P0 carries the older rows from the turn that computed them (bytes unchanged) | measured on CPU, to be proven on cards (XX) |
| Sampler (first token) | No | argmax of the last row's logits; one row | argued |

### 2.3 Is the divergence in the new tokens' prefill, or in the carried state?

The probe's arms cannot tell, by construction, **except for one region**. Segment 1 of an arm, `[floor2048(R), R)`, starts from the exact checkpoint at `floor2048(R)` (no hand-off in front of it) and runs the
tail in a bucket smaller than the cold run's 2048: any difference in the K/V rows of region `floor2048_R_to_R` is the **program** alone. The report carries that count per arm
(`kv.layers[].k.regions.floor2048_R_to_R`); whoever has the v839 file should read it first. Everything after R mixes the program of the head (another bucket) with the hand-off at R (a bf16 store the cold run
does not make), and the GDN-state differences, which persist in the recurrent state for the rest of the context, cannot be assigned.

SP1 removes the confound with two kinds of arm, in the same 30-minute job (`C2_FABRIC_PROBE=split-attribute`, `tp4_split_probe.py --attribute`, under the ship-4 candidate's own environment):

* **B arms** as before: each segment in the bucket the engine would pick.
* **F arms**: the same splits with BOTH segments forced into the 2048 bucket (`prefill_masked_bucket(bucket=2048)`): the cold run's own program, masked to the valid rows. What is left is the hand-off (and the mask).

| SP1 reads | Meaning | Consequence |
| --- | --- | --- |
| every B arm IDENTICAL (`verdict=IDENTICAL`) | the programs of the small buckets and the hand-off are both exact | a 128-aligned state is an E0 state. The remaining problem is getting it: section 3.3 (a background re-run in the tool gap, free only when the card is idle) |
| `cause=bucket` (B differs, F identical) | the store is exact; only the small buckets' programs differ | a resume is byte-exact in the cold run's own bucket, which saves nothing (section 3.2). Saving needs the small-bucket programs to be numerically equal to the 2048 one: pin the matmul K blocking across buckets in the graft (a config change that moves the cold baseline too, so it is a numerics change, not a free lever) |
| `cause=handoff` (F differs) | the store at R (the bf16 state and conv carry) or the mask is not the in-program state | the fused GDN op must emit the in-program state at a chosen sub-chunk (a kernel twin, section 3.3); or store the state in fp32 (`QWEN35_GDN_STATE_BF16=0`: checkpoints double to 154 MB and the cold baseline moves) |
| `cause=none` on some splits and not others | read the regions and the per-layer lists in `split-probe.json` | |

`test_exact_end.ToyNumericsTests` derives this table from a toy prefill (matmul blocking chosen by the bucket, a bf16 store between programs) and checks that the probe's `attribution()` prints the cause the
toy implies in each of the four cells. It is a model of the claim, not of the hardware: it shows what each answer would mean, and that a resume at the previous prompt's end (an arbitrary position) is
inexact in every cell.

## 3. Question 2: Option A, byte-exact

### 3.1 The variants, on the production sequence

| Variant | Resume | Programs run | Rows computed (P = 165,056) | Byte-exact? | Needs |
| --- | --- | --- | --- | --- | --- |
| today | `floor2048(P_prev) - 2048` = 161,792 | full chunk + the tail | 4,096 | yes | - |
| **A1 = P0** (undrop + window carry) | `floor2048(P_prev)` = 163,840 | the tail only | 2,048 | **yes (E0), by construction** | built; to be proven on cards (XX) |
| A2 "one checkpoint higher still" | `floor2048(P)` | the same tail | 2,048 | yes, and the same as A1 | impossible above A1: a state on the 2048 grid above `floor2048(P_prev)` exists only after a FULL chunk program, and turn N ran a tail there. On the crossing turn it would need the cell's full chunk, which nobody has computed |
| A3 128-aligned state, **cold bucket forced** | 164,864 | one program, bucket 2048, masked | **2,048** | yes if SP1's F arms are IDENTICAL | the state at R (section 3.3) |
| A4 128-aligned state, **small bucket** | 164,864 | one program, bucket 256 | **256** | only if SP1 reads `cause=none` (the small buckets' programs and the store are both exact; nothing says so today) | the state at R (section 3.3) |
| B (decode-time state, section 4) | 164,864 | bucket 256 | 256 | no (E1) | owner decision, kernel work |

### 3.2 Why P0 is the floor of anything byte-exact

Take the cold prefill of the new prompt P in the common case (P and P_prev in one cell, `c = floor2048(P)`). Its last program is `(c, bucket_for(t), t)`, one program over all of `[c, P)`.

1. Every row of that program is computed with that bucket's program config. A resume that reproduces the cold prefill's bytes for the rows `[R, P)` must run those rows in **a program with the same
   arithmetic**: the cold bucket (or, if SP1 proves it, a smaller one that is numerically equal, which no evidence supports today).
2. A program's cost is its bucket (section 1.1): running the cold bucket on the rows `[R, P)` costs the same as running it on `[c, P)`.
3. P0 runs exactly `(c, bucket_for(t), t)` and nothing else. It is the cheapest program list whose every program is a program the cold prefill runs.

Hence "keeping intermediate per-chunk partial state" (the state after sub-chunk j of the tail) cannot reduce the cost of an exact resume: it saves rows inside a program that is paid for by its size.
`test_exact_end.GridEquivalenceTests` states it as program lists: P0's list is a suffix of the cold list; a 128-aligned resume runs a program starting inside a cell that the cold list never contains.

The one place P0 is not yet minimal is the **cell-crossing turn** (8% of turns at 170 tokens a turn): it runs the old cell's full chunk and a small tail (2 forwards). Nothing above `floor2048(P_prev)` exists
on the grid, so no exact resume can avoid that chunk.

### 3.3 What a cheaper exact resume would need (kernel work, described precisely, and stopped)

Section 3.2 leaves exactly three routes below P0. None is possible without touching an op or the numerics; none was started.

1. **Bucket-invariant programs** (SP1 `cause=bucket`). Make the prefill matmul program configs of the buckets 128..1024 arithmetic-equal to the 2048 one: the same K blocking (`in0_block_w`), the same
   accumulation dtype and partial-sum spill for every M, with only `per_core_M` and the grid following M. That is a change in the tuning `tp_common.create_prefill_matmul_program_config` /
   `tp_common.prefill_tuning` hand to the prefill sites (the image's `models/demos/blackhole/qwen36/tt/tp_common.py`, called from `graft/model_config.py:266-270`; the file is sha-pinned on the serving
   path, so the change goes through its callers, the graft's `prefill_progcfg`) and re-qualification of every prefill: the cold prefill's own tail changes too, so it moves the baseline.
2. **An in-program state emit** (SP1 `cause=handoff`). A twin of the fused chunk GDN op (a JIT/programmable-op twin, no factory edit, like the op-fusion programme's) that also writes the recurrent state after
   a chosen 128-row sub-chunk, plus the conv carry rows `[R - 3, R)`, so that turn N's tail program hands over the cold run's own state at `R = floor128(P_prev)`. It must be qualified byte-for-byte against the op it twins.
3. **Where the state at R comes from without paying for it on turn N.** Splitting turn N's tail at R costs an extra forward on turn N (0.25-0.6 s, **estimated** from the T=128 medians) that cancels the
   saving of turn N+1. A background re-run of `[c, R)` in the tool gap is free only if the card is idle then; the card is shared by 8 seats and Lever N's decode-gap budget governs it. Only route 2 pays nothing.

So: with today's kernels the byte-exact answer for "prefill only the new tokens" is P0; the next step down is a **kernel/config project** with its own qualification and a card-time cost, and its prize
(Option B ideal) is -12% of today's prefill time beyond P0.

### 3.4 Option C, considered and not recommended: a prefix-stable tail grid

Cut the cold prefill's tail cell at FIXED offsets (a ladder of pieces such as 1024 / 512 / 256 / 128 / 128) instead of one masked bucket. Every completed piece's state is then a state any longer prompt of the same text
also passes through: intermediate per-chunk state becomes exact for the new baseline, and the cap rule could publish the KV up to the last completed piece. It needs no kernel. Under the measured per-forward floor it
is **no better than P0** (-41% against P0's -44%; ladders of 6 to 16 pieces were tried: 1.00 to 1.40 s per turn against P0's 1.01 s), because every extra piece pays the floor, and it makes every cold prompt pay more forwards. It also moves the
cold baseline (E2 against the old build). Dropped.

## 4. Question 3: Option B, numerically equivalent, not byte-identical (the facts for the owner)

**What B is.** Resume at `R = floor128(P_prev)` inside the cell, from a state that is not the cold prefill's own, with the head in a small bucket. Hit != miss for the first time: a hit would equal a *split twin*,
not the cold twin (exactness class E1, `docs/tp4-exact-resume.md` 5.1). The KV the split writes would be session-private (never published to the shared cache), which vLLM does not support natively.

**What it can add over P0 (modelled).** 0.22 s of 1.01 s per returning turn on average (the 128-aligned row of section 1.3): about 12% of today's prefill time, about 7% of card wall time at the 58% share.
It is available only with kernel work (section 3.3, route 2) or with the extra-forward trade that cancels it; with today's kernels B's net is about zero.

**Past evidence of divergence (the only kind this repository has).**

* The probe (above): GDN state 11 of 18 layer-states differ, largest absolute 0.125 (one bf16 ulp for values of 16 to 32), logits differ, argmax equal; a four-layer stub on synthetic tokens: weak evidence on text.
* Real text, whole-prefill rounding differences (`docs/real-text-2026-09-24.md` section 3, v157 against v158, reproduced by v178 = the plain packed path plus `SINGLE_GATEUP` alone): **all eight** streams
  (four users at 32k and four at 131k, 256 tokens each) diverged from the reference, the first divergent characters at 80 / 0 / 758 / 66 and 488 / 1 / 77 / 206; with `max_tokens=1` (v176) **one of four first tokens differed**
  (`Looking` against `## Review`). Every stream stayed coherent: near-tie tokens flip. These were differences in the arithmetic of the whole context; B changes the K/V of a few hundred rows and the recurrent
  state at one point (which then persists), so its rate should be lower, but the state perturbation never decays away entirely, and **nobody has measured it**.

**A gate for it (proposed; the thresholds are the owner's).**

1. *Teacher-forced paired logits* on real multi-turn text (`prefix_agent_corpus`, the agent replay): for N >= 1,000 returning turns, feed the cold twin's own answer to the hit and compare the logits at every
   answer position (about 170,000 positions): max |dlogit|, its 99.9th percentile, top-1 agreement, and the margin distribution (the fraction of positions whose top-2 margin is under 2 x the observed max |dlogit| is the
   predicted flip rate).
2. *Greedy turn identity*: the fraction of turns whose greedy answer is byte-identical to the cold twin's, with a Wilson interval, and the distribution of the first divergent token. Per-position flip probability p gives
   `(1 - p)^170` turns identical: 99% of turns identical needs `p <= 6e-5` per token, three orders of magnitude gentler than the whole-context differences of v178 (p about 2e-2 to 4e-2 by the first-divergence data). N turns with zero divergence
   bounds the divergent fraction below 3/N at 95% (0.3% at N = 1,000).
3. *Semantic read of the divergent turns* (tool name and arguments, or a judged equivalence), because a different but valid tool call is a different transcript, not a wrong one.
4. *Isolation*: split-twin blocks never reach an E0 consumer (fresh-salt and sibling probes), and a divergence below R gives a lower Q, never an inexact hit.

A proposal for the threshold is the owner's to set; a defensible default is "99% of turns identical with the 95% upper bound on divergent turns under 2%, and 100% of divergent turns semantically equivalent", which at
N = 300 turns needs at most one divergence.

## 5. Question 4: interactions

| Component | Interaction with P0 / exact end | Evidence |
| --- | --- | --- |
| **Prefix registry checkpoints** (`qwen_prefix_registry.py`) | P0 stores the same checkpoints (a chunk-program state at `floor2048(P)`), one per cell crossing, and attaches the window to the one it stores; a later turn at the same boundary reuses it (no new capture: `boundary > q` is false). The window is charged to the checkpoint's bytes and goes with it on eviction: with the 107 MB preconverted form plus about 52 MB of window on average (105 MB at most) the 8 GiB store holds about 50 checkpoints instead of about 80. `QWEN_PREFIX_SUPERSEDE` (off in every profile) keeps the newest two checkpoints of a chain: P0's newest is the one it needs. A cheap follow-up: drop the windows of superseded checkpoints. | `test_exact_resume.CarryTests`, `qwen_prefix_registry.py:1494-1525` |
| **Host KV tier** (32 GiB, preconverted checkpoints, in the production profile) | The tier moves KV *blocks* between the device pool and host RAM; the checkpoint (and its window) lives in the registry store and is not touched. The undrop wraps the coordinator's `find_longest_cache_hit`, which the tier's `recompute` also goes through, so a restored session's hit keeps its last block like any other. Not run on a card together: job **XT** (tier-returning plan on the audited XE twin) is the read. | `qwen_prefix_scheduler_patch.py:1492-1506, 1296-1305` |
| **Engine reuse / parked rebind** | The rebind projects the drafter window from the captured taps in independent 32-row chunks; P0 hands it a complete window of the usual shape (carried rows first), so the rebind is unchanged. **Defect found and fixed here:** on the ship-4 profile the window is built by the tile-aligned join (`QWEN_FAST_TAP_WINDOW_ALIGNED=1`, rebind-fast) so that no prompt length compiles a concat (91 of 217 rebinds were slow, +640-770 ms). P0 as merged turned that join off under a carry, so every carried turn would have compiled again (each turn has a new length). The carried rows are now the aligned join's first part: `Carried.align()` leads the upload with `start % 32` rows of padding that the join's one slice cuts away, the concat inputs are whole tiles, 2048 or 2080 rows tall, the slice sees one height. Flags off: unchanged. | `dflash_prefill_window.py` (`PrefillWindowCapture.__init__`, `aligned_outputs`), `prefix_window_carry.py` (`Carried.align`), `test_exact_end.AlignedCarryTests` (they fail on the merge-time behaviour) |
| **Drafter-window carry** | Byte-exact by construction: the carried rows are the taps the capturing turn computed with full-chunk programs at the same absolute positions the cold twin would, moved host to device unchanged. XX compares every pair's window digest with its cold twin's and the carry's audit reads the upload back. Cost: an export of up to 105 MB on the turn that stores a checkpoint (one per cell crossing) and an upload of the same size on each carried turn (modelled 0.04-0.1 s). | `prefix_window_carry.py`, `tap_window_aligned.py` |
| **Lever N chunking** | A resumed prefill is one final step (`step_budget`: at or past `S_last` the step takes everything left); with the carry `valid_step` accepts a start above `P - 2048` (`levern_policy.py:449-470`). The step count does not change; the step now holds one forward instead of two. The governor's `StepTimes` calibrates on the step time, which drops with it. `levern_policy.plan_from` (tests only) still refuses a start above `S_last`; the route does not call it. | `levern_policy.py`, `levern_route.py` |
| **Sticky sessions' ceiling and cap** | Unchanged: the cap still publishes only `[0, floor2048(P))`, so no tail-program block is ever shared; P0 only moves the resume up to the highest published boundary. | `qwen_prefix_scheduler_patch.py:1350` |
| **Memory on the device** | Nothing is held between turns: the carried rows are uploaded for one turn's window build, as the cold prefill's own capture holds them. | `docs/tp4-exact-resume.md` 5.7 |

## 6. Ranked recommendation

1. **P0 on the production stack, proven on cards (XX), timed (XC1 XL1 XC2 XL2), then a waiver.** Byte-exact, no kernel, no numerics change; modelled -44% of returning-turn prefill, -67% of re-prefilled old tokens, -26% of
   card wall time at most. This branch carries P0 onto the ship-4 + Swift stack (the merge), makes it work beside the tile-aligned window (section 5), serves Swift in the prefix gate, and ships the pack.
2. **SP1 in the same window (soft, 30 minutes, no build).** It decides, with one report, whether anything finer than P0 is a config project, a kernel project or a dead end (section 2.3).
3. **After P0, the fixed floor around the one forward**, not more rows: the engine rebind (0.5-1.3 s, bimodal), the 0.2 s step setup, the API render at long P. The modelled remaining prefill of a typical turn is 1.0-1.2 s; the TTFT floor is 1.8-2.5 s (`docs/tp4-exact-resume.md` 4.3).
4. **Option B only with an explicit owner decision AND SP1's answer AND kernel work.** Not now: with today's kernels its net gain is about zero, and its divergence risk is unmeasured with the only analogue being bad.
5. **Option C and P2 (end-of-turn decode state): no.** C is no better than P0; P2 needs the verify path touched and can never be E0.

## 7. What was built, and the decisions

### 7.1 Built (default off, CPU-tested; nothing has run on a card)

| Piece | Files |
| --- | --- |
| P0 merged onto the ship-4 + Swift stack, its generated parts placed (twins, overlay line, smoke row, CPU allowlist) | merge of `tp4/swift` into the branch; `fusion-wp/ER.json` placement; `test_exact_resume` compares the switches-off graft with the swift graft |
| P0 beside the tile-aligned window | `dflash_prefill_window.py`, `prefix_window_carry.py`; `test_exact_end.AlignedCarryTests` (flag-off identity, cold-window equality, bounded shapes, the audit) |
| SP1: the probe's attribution arms | `tp4_split_probe.py --attribute`, `C2_FABRIC_PROBE=split-attribute` (`c2_serving_job.py`, `qwen-c2-serving.yml`), `test_tp4_split_probe.AttributionTests` |
| The prefix gate serves Swift and reads its log | `c2_prefix_gate.py --model-repo/--model-revision`, `checkpoint_findings`; `c2_serving_job.py` (`prefix` forwards the pair); `test_exact_end.PrefixGateCheckpointTests`, `test_qwen_c2_checkpoint.JobKeyTests` |
| Gate-only twins of the ship-4 profile and the window pack | `make_xe_profiles.py` (`-fx-s4-xe`, `-fx-s4-xe-audit`), `make_xe_jobs.py`, `references/fusion-jobs/XE`, `fusion-wp/XE.json`, `test_xe_profiles`, `test_xe_pack` |
| The arithmetic and the chunk-grid proof | `exact_end_model.py`, `test_exact_end` (grid, policies, Oracle agreement, program-list suffix property, the toy) |

The switches stay what P0 made them: `QWEN_PREFIX_STICKY_UNDROP`, `QWEN_PREFIX_WINDOW_CARRY`, `QWEN_PREFIX_WINDOW_CARRY_AUDIT`, gate-only (`serving_c2_contract.exact_resume_problems`), refused on a
traffic profile and dropped when inherited. No traffic profile, image, default or ceiling changed.

### 7.2 The card plan (`references/fusion-jobs/XE`, ORDER.txt has the stop rules and read rules)

B0 build (K64j-OQ graft) 40 min; X0 status/rescan/reset 20; **SP1** attribution probe 30; **XX** exactness 120 (STOP: the exact-resume plan on the audited twin, Swift, real multi-turn text: every cold/hit pair
IDENTICAL, the oracle's Q strict, at least one carried admission, every window audit `exact=True`, every pair's window digests equal, the aligned-window lines with `parts=2` and no fell-back line, the checkpoint read);
**XT** host-tier beside the carry 120 (soft); **XC1 XL1 XC2 XL2** the timed ABAB 45 each; **XB1 XB2** the eight-seat replay 45 each (optional); Z 10. About 10 h 10 min with everything (B0 is a build and may run while production serves: 9 h 30 min of cards), 6 h 40 min without XT and XB.
NO-GO: any pair that differs, a window digest that differs, an audit line with `exact=False`, a Q above the oracle's, a program compiled after the warm, a missing checkpoint line.
The cards serve production: the owner stops and restores it around the window; the ARC runner sets go to zero from SP1 to XB2; one tag at a time.

### 7.3 Decisions

* **Window (needed).** A card window for the XE pack. A yes authorises measurement only.
* **Waiver (later, after XX and the ABAB read).** Putting `QWEN_PREFIX_STICKY_UNDROP=1` and `QWEN_PREFIX_WINDOW_CARRY=1` on the traffic profile needs an owner traffic waiver naming them: they are
  gate-only until a qualification record exists. P0 is exact by construction and by audit, so the waiver is of the same kind as the fusion levers'.
* **Option B (not now).** The facts are in section 4; the thresholds of its gate are the owner's. SP1 should be read first.

## 8. Not measured, and risks

* Every timing is modelled from solo medians on the contexts of the earlier runs; production queueing, the rebind mode and the busy case are the ABAB's (XC/XL, XB).
* The v839 report was not available; its region counts (section 2.3) are the first thing to read, and SP1 supersedes it.
* The GDN op's internal state precision is in tt-metal, not in this repository.
* The 58% prefill share includes cold first turns and new content that P0 cannot touch; the card-wall-time figures are upper bounds with the share as a parameter.
* P0's window export and upload (up to 105 MB) are estimated at 0.04-0.1 s; the store holds fewer checkpoints (about 50 against about 80). Both are visible in XX and XT.
* Reasoning-dropping clients and whitespace-edged tool arguments still make the next prompt diverge before `P_prev` (`docs/tp4-exact-resume.md` 2); P0 helps whenever the checkpoint's boundary lies below the divergence, which
  at `floor2048(P_prev)` it usually does. The turn telemetry (`qwen_prefix_turn_*`) says how often.

## 9. Reproduce

```
python3 -B scripts/ci/exact_end_model.py --start 164887 --delta 169 --turns 9     # section 1.3's table
python3 -B scripts/ci/exact_end_model.py --simulate 400 --delta 170                # the totals
cd scripts/ci && python3 -B -m unittest test_exact_end test_xe_profiles test_xe_pack test_exact_resume test_tp4_split_probe
python3 scripts/ci/make_xe_profiles.py --check; python3 scripts/ci/make_xe_jobs.py --check
for f in scripts/ci/references/fusion-jobs/XE/*.env; do python3 -s scripts/ci/c2_serving_job.py $f scripts/ci/qwen_c2_profiles.json >/dev/null || echo FAIL $f; done
```
