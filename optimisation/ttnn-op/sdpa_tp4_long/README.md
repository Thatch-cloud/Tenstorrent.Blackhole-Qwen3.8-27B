# sdpa_tp4_long: the long-context SDPA decode sweep at four cards

At a 262,144-token window the packed verify's attention is the largest term of the round: one K64j decode launch per user per attention
layer (G8B2, flags `0x23`: two entries of 8 tokens x 6 query heads on one KV head, 16 cores per entry, 32 of 110 cores active) reads K and V
at roughly 150 GB/s against about 405 GB/s of DRAM. This directory holds the one-card harness that times every candidate configuration
against the served call, byte-compares each with it, and reports GB/s, plus the serving flag that selects a configuration by name.

## What is here

| File | What |
|---|---|
| `sdpa_tp4_long.py` | the sweep (device part plus CPU-testable helpers) |
| `run_card_m.sh` | the card-M runner (embeds the canonical `qual_card.sh` block; a serving image, no weights, no graft mount) |
| `../../../scripts/ci/sdpa_long_tp.py` | the serving flag `QWEN_FAST_TP4_SDPA` and the table of configuration names both sides share |
| `../../../scripts/ci/references/tp4-sdpa-long-jobs/` | the job pack: `A0` (agentstop), `Q1` (this sweep, optional), `ORDER.txt` (the driver hands the cards back; no job of this pack does) |
| `../../../scripts/ci/test_sdpa_tp4_long.py`, `test_sdpa_long_tp.py`, `test_tp4_sdpa_long_window.py` | the CPU tests (on the CPU allowlist) |

## The configurations (arms)

The arm names are the values of `QWEN_FAST_TP4_SDPA` (`sdpa_long_tp.CONFIGS` is the one table; a test holds the sweep's arms equal to it).

| Arm | Call | Exact vs served | Servable today |
|---|---|---|---|
| `served` | G8B2 `0x23` per user, the mesh's own worker grid | reference | yes (explicit no-op) |
| `grid8x4`, `grid8x10`, `grid4x8` | the same call on another worker grid | by construction (the active cores move relative to the DRAM banks; the accumulation order does not); the sweep proves it per row | yes |
| `grid11x4` | the CONTROL for the grid arms: expected to place the 32 active cores where the served grid does, so only the idle-core and dispatch set differs | by construction; a win by it is noise or dispatch, not placement | yes |
| `multi` | ONE launch for all users, one G16 entry per user (96 rows, flags `0x21`, 16 cores per entry up to 6 users) | by construction; unproven until the sweep runs | no: needs a reader, a per-entry mask kernel and a pool-lent table |
| `rowsplit` | G4B4 per user (4 entries of 4 tokens, 64 cores, `0x23`) | by construction | no: slower than served until the reader sizes its KV barrier to the real reader count |
| `ra` | the served call with the KV read-ahead flag (`0x2B`) | by construction | no: the pinned extent reader admits `0x23` only |

Anything that changes the number of cores per entry, the 256-key chunk or the compute kernel config changes the online-softmax order and is
not an arm here (it would need a re-baselined reference).

## How to run

On the rig, as a `cardm` job (the cards are free after `A0`; one card, no reset needed before it):

```
C2_CARDS=pair
C2_ACTIONS=cardm
C2_IMAGE_TAG=tp4-next-3a
C2_CARDM_HARNESS=optimisation/ttnn-op/sdpa_tp4_long/run_card_m.sh
C2_CARDM_ARGS=--users 4
C2_CARDM_ENV=IMAGE_TAG=tp4-next-3a
```

That is `scripts/ci/references/tp4-sdpa-long-jobs/Q1-sdpa-long-sweep.env`; `scripts/ci/c2_serving_job.py <env>` parses it. By hand on the rig
(card named by `QUAL_CARD`, card B refused, the serving pair needs `ALLOW_SERVING_CARD=1`):

```
QUAL_CARD=<board id> ALLOW_SERVING_CARD=1 IMAGE_TAG=tp4-next-3a bash run_card_m.sh
CARD_B_ARGS="--arms served,multi --extents 262400 --users 4" ... bash run_card_m.sh
CARD_B_ARGS="--timing none" ... bash run_card_m.sh        # exactness only
```

**The image.** Any local `qwen38-c2-<tag>` serving image that carries the K64j op graft. The window's tag in the templates is a PLACEHOLDER the driver confirms BEFORE A0 takes production down: `IMAGE_CHECK_ONLY=1 IMAGE_TAG=<tag> bash run_card_m.sh` opens no device (it names the board like the run, `QUAL_CARD` and `ALLOW_SERVING_CARD=1`, because the canonical selection block runs first); it resolves the image and checks every `_ttnncpp.so` in it against the pinned graft (exit 0 only then).
The harness mounts no graft and no model: the sweep makes its own seeded K/V, and its five scripts come from the checkout, so no image build is
needed (the image need not contain `sdpa_long_tp.py`). At run time it records the sha256 of the `_ttnncpp.so` the process actually MAPPED (the image installs the graft at two paths) and holds it to the graft the
checkout's `build-c2-serving-image.sh` pins (`EXPECT_TTNNCPP_SHA256` overrides; empty skips with a warning), and every launched shape (flags, B, PNHt, St) must have as many `[QWEN-SDPA]` factory lines in the native log as arms launched it (a grid arm is a program of its own): an image with another graft, or a graft installed but not executed, gives `NO-DECISION`.

**Arguments** (`CARD_B_ARGS`, appended last): `--extents` (default 33024,65792,131328,262400), `--users` (1-6, default 4; 1 skips the multi-user
cases), `--arms` (default all; `served` always runs), `--starts` (default 240), `--seeds`, `--timing trace|eager|none`, `--rounds 5`,
`--iterations 10`, `--calls 8`, `--no-mixed`, `--deadline-s`, `--watchdog 300`, `--page-width` (pages per page-table row, default max(4100, longest case + one chunk): ONE width for every case, the served pool's, because the factory's St is compile-time). The device worker grid must be the serving mesh's 11x10, else `NO-DECISION`. The runner sets `--deadline-s` to the container timeout minus
600 s (`SWEEP_TIMEOUT_S`, default 3000).

## The cases and checks

Cases: `u1@E` (one user) and `uN@E` (N users, all at E) per extent, then `mixed` (262,400 / 131,328 / 65,792 / 33,024) and `skewed` (262,400 and
three at 4,352) at four users. Each user has its own seeded bf8 K/V in one shared cache, a disjoint shuffled page table poisoned past E
(K = 0, V = +16,384), the narrow tail mask, and `cur_pos = E - 1`.

Per case and arm: **exact** (every row of every user as int16 against the served arm in the same process; `differing_rows` 0), the served
arm's per-user **bit hashes** (compare two containers without sharing a process), **liveness** (`cur_pos` one chunk past E must change the
served output), **finite**, the **factory line**, and **timing**: `calls` steps in one trace replayed `iterations` times, `rounds` rounds with a
seeded arm order, the paired ratio arm / served per round, GB/s of K and V actually read, `trace_equal` (the replayed trace equals the eager
read). An eager burst is the fallback if a trace cannot be captured (`trace_error` says why).

## Reading the result

`sdpalong-<stamp>.json` in the results directory (set by the cardm step; by hand `$HOME/sdpalong/<card tag>`), and the last log line:

```
SDPA_TP4_LONG verdict=PASS|FAIL|NO-DECISION cases=N arms_ran=... arms_exact=... winner_u4=<arm> ratio=<mean paired ratio> gbps=<GB/s> ...
```

* `cases[].arms[<arm>]`: `status` (`ok`, `error`, `skipped`), `differing_rows`, `finite`, `timing` (`median_us`, `ratio_median/min/max`,
  `gbps_median`, `mode`, `busiest_chunks`), `trace_equal`, and the error text and traceback of an arm the factory refused.
* `fits`: per arm over the one-user cases, `t = fixed_us + per_chunk_us x busiest-core chunks` (the design's cost model is about 50 us + 14.5 us).
* `winners`: per case kind (`u1`, `uN`, `mixed`, `skewed`) the fastest arm that is exact on every case of that kind, by mean paired ratio, and only if it is faster than served in EVERY paired round of every case of the kind (`ratio_max` < 1) and by at least 2% on
  average (mean <= 0.98): a smaller margin is noise. An arm with a differing row, or an error in any case of the kind, is never a winner. **Nothing is applied by the sweep.**

PASS means the served arm ran in every case, its liveness control moved, every output was finite, every factory line was present and something
was timed. FAIL: a served output is not finite or a liveness control did not move. NO-DECISION: anything else (an error outside an arm, a wrong
binary, a missing factory line, nothing timed, a deadline that cut everything).

A hang: the Python watchdog (300 s) writes the partial report and exits 124 when its thread can run; a ttnn call that holds the GIL blocks that thread, so every step also arms `faulthandler.dump_traceback_later` (300 s + 30 s, exit 1, a C thread), whose stack dump in the log names the hung call (the runner reads it as a hang). The report on disk is the one written after the last finished arm. The runner's EXIT trap removes the container and prints the
reset line for the target card only (`qual_reset_hint`). `rowsplit` and `ra` run flag sets that have never run at one KV head, so they run in a SECOND pass over the cases, after every safe case (each risky case carries its own served arm; `phase` in the report says which pass). A hang there costs only the risky data. Never cancel the job while it runs (root-owned logs).

## The serving flag

`QWEN_FAST_TP4_SDPA=<name>` (unset, empty, `0` or `off` is off). Default off, byte-identical off: the hook is the constructor of
`extent_attention_fold_tp.PackedExtentReplayReader` (the unpinned subclass; the pinned `extent_attention_replay_tp.py` is untouched), and with the
flag off `sdpa_long_tp.apply` returns before it touches anything. `tp_addresses` binds that subclass when `QWEN_FAST_TP4_ATTN_FOLD` **or** this
flag is set. With a servable name each segment entry's `SDPAProgramConfig` is rebuilt after the pinned reader has qualified it, with the same
sentinel, chunk size and exp mode and only the worker grid changed, and `[PINDIAG] tp4 sdpa engaged config=<name> grid=XxY entries=N` is logged
(the smoke check fails a profile that sets the flag and logs no such line). A typo, a name that cannot be served yet (`multi`, `rowsplit`, `ra`),
a grid the mesh cannot hold, a flag set other than `0x23`, or the flag at the pair, raises.

The gate-only profile `c2-packed-tp4-best-sdpa` is `c2-packed-tp4-best-strace` plus exactly `QWEN_FAST_TP4_SDPA=served` (a placeholder: the
no-op control twin). After the sweep picks a winner, edit that one value to the winning servable name, build an image from the pushed branch,
and run the audited matrix, the 5/5 hang gates, then ABAB timed pairs against `c2-packed-tp4-best-strace`, judged paired per round.

Tests (py 3.11, from `scripts/ci`): `py -3.11 -B -m unittest test_sdpa_tp4_long test_sdpa_long_tp test_tp4_sdpa_long_window`.
