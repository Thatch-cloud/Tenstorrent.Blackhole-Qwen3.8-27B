# k64j_probe: can the decode SDPA take its extent at run time? (K64j P0, risks R1 and R2)

K64j (c2-serve-for-real-plan.md section 2.3) would let the packed 64-row block serve users at arbitrary, different
positions: the served non-causal replay call (tail mask, 256-key chunks) gets a runtime `cur_pos` word, `E - 1` with
`E = (start // 256 + 1) * 256`, instead of the compile-time `cur_pos_base = St * 32 - 1`. Two risks decide whether that
can be exact:

- **R1.** Does a runtime `cur_pos` reproduce the compile-time split? The split also takes `max_dynamic_chunk_size`,
  `St` still feeds three places, and the writer's `c_8` handling was unread.
- **R2.** The `UINT32_MAX` "skip this user" value exists only in the causal branch. Is it handled in compute and
  in the writer, and can it hang?

This directory answers both on paper, with a CPU test that makes the paper argument checkable (every family of the
served geometry, and the kernel's own C++ compiled against the Python mirror). It also holds the card-B harness that
measures the answer on the binary that serves today (K64i), with no build.

## The answer on paper

The sources read are tt-metal 9f9cd4fd, the same bytes the serving image and K64i carry. The runner re-checks
every hash in the graft on the host, and the probe re-checks them in the container.

| File | sha256 | Role |
|---|---|---|
| `reader_decode_all.cpp` | 49a05926 | the stock reader |
| `writer_decode_all.cpp` | 734c90c0 | the stock writer |
| `sdpa_flash_decode.cpp` | d24769bd | the stock compute kernel |
| `dataflow_common.hpp` | e4623a22 | shared dataflow helpers |
| `rt_args_common.hpp` | 1b52c60d | the split |
| factory | 3e0a69af | the program factory |

The first four are vendored under `fixtures/` next to `rt_args_common.hpp`, byte for byte (the tests check each
hash), from the tt-metal 9f9cd4fd sources the K64i build used.

The qwen kernels (`../sdpa_decode_qwen`, `../sdpa_decode_slice`) carry the stock `cur_pos` block byte for byte.
`ServedKernelTextTests` cuts the block out of the vendored stock reader, writer and compute kernel and compares each
served kernel's block with it as text: the readers' block hashes to bce6424f, the slice writer's to e741ca9c and the
qwen compute kernel's to a7a3964c. So the causal path the probe runs on hardware is the code a K64j flag would lift
out of `if constexpr (is_causal)`.

### R1: the split is a function of `nearest_n(cur_pos + 1, 256)` only

- **The chunk size never reads `max_dynamic_chunk_size` in the replay.** `get_dynamic_Sk_chunk_t<Sk_chunk_t, max>`
  returns the compile-time `Sk_chunk_t` whenever it is nonzero: the `if constexpr (Sk_chunk_t == 0)` branch that
  reads `max` is compiled out (`rt_args_common.hpp:95-108`). Every replay config
  passes `k_chunk_size=256` (`attention_replay.py:53-54`, `pooled_attention_replay.py:199-200`), so `Sk_chunk_t` is
  8 and `max_dynamic_chunk_size` is dead.
- **Its served value is 8.** It is `dst_size` (factory :388-389): 4 when `fp32_dest_acc_en` is set, 8 otherwise. No
  served call passes a compute config, and the op's default leaves `fp32_dest_acc_en` false
  (`sdpa_decode.cpp:77`, `init_device_compute_kernel_config(..., HiFi2, true, false, false)`).
- **It matters only to the native decode.** That path uses `k_chunk_size=0`
  (`docker/qwen-c2-graft/graft/attention/tp.py:737-742`). Its chunk is then the power of two of the tiles seen,
  capped at 8 tiles: 256 keys from position 225 on. Section D records whether it equals the fixed 256.
- **`get_workload_for_core` (`rt_args_common.hpp:35-93`) reads `cur_pos` only through
  `valid_seq_len = nearest_n(cur_pos + 1, 256)`.** Every `cur_pos` in `[E - 256, E - 1]` therefore gives the
  compile-time call's split at capacity E: the same chunk count, per-core ranges and active tree.
  `SplitModelTests` checks this for all 513 families of 131,328 keys, at 1, 2, 5, 13, 16 and 32 cores per head.
  It samples five `cur_pos` per family: both ends and three inside. That suffices because `nearest_n` is
  monotonic, so a value equal at both ends of the family is equal throughout.
- **`CompiledMirrorTests` shows `split_model.py` is the kernel code.** It compiles the vendored `rt_args_common.hpp`
  and `get_tree_reduction_params` with g++, then compares every workload, dynamic chunk and tree answer.
- **The cores per head depend on B** (factory :195-206): 16 for B <= 3, and 13 at B = 4. So a runtime-extent call
  equals the compile-time call only at the same B. Every served replay bundle has B <= 3, so 16 cores.
- **What `St` still feeds** (`reader_decode_qwen.cpp`):
  - The tail assert (:82) holds for any capacity.
  - The mask offsets (:293-297): tail mode reads the mask's LAST chunk (`mask_width_t - Sk_chunk_t`). A wide mask
    would therefore be read at `[C - 256, C)`, not `[E - 256, E)`. K64j must use the narrow 256-column tail
    (mask refresh v2, offset 0) or write the tail content into the wide mask's last chunk.
  - The non-paged offsets (:366-377) are unused: qwen mode refuses non-paged calls.
- **The writer computes its own split from its own `cur_pos`** (`writer_decode_all.cpp:118-138`, `:146-183`). It
  uses it for `has_local_data`, the active children (`child_id < k_num_chunks`) and the causal `generate_mask`.
  - A writer left on `St * 32 - 1` beside a reader and compute on a smaller runtime extent HANGS. On a core with a
    chunk at C and none at E, the writer waits for compute output that never comes (:347), and a parent waits for
    a child semaphore that is never raised (:270-279).
  - This happens exactly when `E / 256 < cores_per_head`, which is E < 4,096 at 16 cores
    (`split_model.stale_writer_hangs`; 15 families), and on every skip.
  - Above that, the stale writer's tree happens to be the live one. **So every 0x20 program needs a writer that
    reads the runtime word.** Today the factory picks the qwen-named slice writer under 0x4 only and the stock
    `writer_decode_all.cpp` otherwise (`apply_factory_slice.py` F17). K64j does not edit the stock writer: F21
    selects the slice writer, with W4, for every 0x20 program whatever its other flags (see "K64j never edits a
    stock kernel" below).
- **Compute** (`sdpa_flash_decode.cpp:139-177`, the same block in `sdpa_flash_decode_qwen.cpp`) reads `c_15` only
  under `is_causal`, so K64j must extend that branch in the qwen compute kernel (C3), never the stock one. The tail
  predicate `k_chunk == k_num_chunks - 1` then uses the runtime chunk count, which is right.

### R2: the skip is consistent across the three kernels, and the output rows stay unwritten

- **All three kernels read the same word.** The reader fills `c_8` and `c_15` and pushes both before it tests
  `UINT32_MAX` (`reader_decode_all.cpp:113-157`). The writer pops `c_8` and returns (`writer_decode_all.cpp:122-137`).
  The compute kernel pops `c_15` and returns (`sdpa_flash_decode.cpp:143-156`).
- **Nothing waits on a skipped batch entry.** Each entry has its own cores, reducer and output core, so the other
  entries are untouched.
- **The skipped entry's output rows are never written.** Inside a trace they hold the previous replay's output;
  in an eager call, whatever the DRAM held, NaN or Inf included (section K records which). Nothing downstream has
  been shown to ignore them: an idle row that reaches a KV-cache write, the GDN state or any cross-row op would
  leak silently. K64j must therefore either zero an idle segment's rows in the extent reader, or add a card test
  that pushes NaN-poisoned idle rows through the fold inverse, the gate, o_proj and the KV / GDN commit and shows
  the live rows and the committed state unchanged.
- **Under KV share (0x2) a mixed skip would hang.** Leader and twins run one READY/VALID round per chunk, so every
  entry of a share bundle must carry the same `cur_pos`: the same user, the same word. Either the extent reader
  writes one word per call and the K64j reader uses slot 0 for every entry, or the host guarantees equal slots.
  The first makes it true by construction.

### The flag bit

0x20 is free. The factory generators serve 0x1, 0x2, 0x4 and 0x8, and so do `pooled_attention_replay.py` and the
gate's `SDPA_MODE_FLAGS`. 0x10 is the card tests' unknown-flag control. Every factory refuses what it does not know
(`[QWEN-SDPA] unknown flags`). `FlagBitTests` checks all of this. Section N also checks it on the loaded binary: a
0x21 call must be refused as unknown.

## The probe (`probe_k64j_card_b.py`, `run_card_b.sh`)

The only kernel path that already takes its position at run time is the CAUSAL one. The probe runs it, from a page
table C / 64 wide (C = the served 131,328 keys) whose pages past each user's extent are POISONED (K = 0,
V = +16384: any read moves every row). It compares it byte for byte with the compile-time non-causal call at
capacity E.

Everything else that differs between the two programs is controlled:

- **The mask.** The causal writer generates the final chunk's mask; its `-inf` bits are 0xFF80. The provided masks
  hold the same bits at the same columns (the CPU test checks this against a mirror of `generate_mask`).
- **The Q tile.** A causal call with <= 16 bf16 heads uses a HALF-tile Q (factory :449). K64j stays non-causal, so
  the decisive shapes have 24, 48 or 96 rows; the 12-row shape is recorded.
- **The cores per head.** Every reference has the call's own B.

The sections run per seed in the order S, T, K, E, D, N, timing (T, D, N and the timing on the first seed only).
E comes last of the decisive ones because its G8B2 legacy program's L1 fit at 131,328 keys is unmeasured.

| Section | What | Verdict role |
|---|---|---|
| S | The plan's P0: 3 users per call at different positions p = E - 256 + s, where E is one of K1's six extents (2,304 ... 131,328) and s is +0, +7, +127, +240 or +255. Each user has its own table and variants normal / peaky, over seeds 0-2. The causal call is compared against the legacy call at each user's E. Only with `--served-pnht1` (off by default) do the qwen tail (0x1) and tail+share (0x3) also run on these 12- and 24-row shapes, which were never run on hardware in the qwen modes (K1 and card M ran G8B2, G4B3 and G4B1 only). | rows 2: DECISIVE; rows 1 (half tile): recorded; `--served-pnht1`: warnings only, a raise included |
| E | The replay's bundles, G4B3 and G8B2, causal at E - 1 on the poisoned table, against legacy with a zero mask at E. The served 0x1 and 0x3 run too, and 0x7 on G8 with K64i. | DECISIVE. A served mode that differs from legacy is a FAILURE at the capacities K64i was card-qualified at (2,304, 33,024, 131,328: the graft is not the qualified one) and a FINDING at 16,896, 65,792 and 98,560 (`served_unqualified_capacity`: evidence against K64j's "exact by construction" premise at that family, not a wrong mount) |
| liveness | `cur_pos = E` (one poisoned key into the next chunk) against `E - 1`, per entry. This is K3's "off by one chunk must change the output", and it proves the poison is live. | a dead control means NO-DECISION |
| T | One trace of the causal call on CLEAN tables, replayed 64 times. The `cur_pos` tensor is rewritten between replays (`copy_host_to_device_tensor`, as `attention_replay.stage` does), spanning more than 50 families. Each replay must equal eager at its positions, and entry 0 must equal the compile-time call at its E. The tables are clean, so this proves the replay re-reads the word and reproduces the split; a read past E would show only through rounding against legacy. Then the skip patterns replay in the same trace. Then the FENCE trace: the same program captured again on a table poisoned past each entry's own E (one family per entry, below C), replayed at +0, +1, +127, +128, +254 and +255 inside those families; each replay must equal eager on the clean table (a key read past E is a poisoned one), and a replay at `cur_pos = E` must differ (the poison is live in the trace). No new JIT build. | DECISIVE; the fence's `cur_pos = E` replay is a liveness control |
| K | R2, eager. `cur_pos = -1` on entries {0}, {1}, {2}, {0,2} and {0,1,2}, plus a B=1 call that skips its only user. The watchdog catches a hang. The live entries must equal the all-live call; {0,1,2} has no live entry and is recorded as returned, not compared. The output is poisoned first, so the skipped rows' state is recorded. | live entries: DECISIVE; skipped rows: recorded |
| D | The native dynamic chunk (`k_chunk_size` 0) against 256, rows 1 and 2. | recorded |
| N | 0x21 must be refused as unknown. A qwen sentinel on a causal call must be refused as non-causal: this is the refusal K64j relaxes. | recorded, warned |
| timing | Runtime at E - 1 against compile-time at E. The runtime extent should pay for E keys, not C. Also the saving from 0 to 3 skipped users. | recorded |

**Isolation and partial results.**

- A section that raises is a failure named `<section>/seed<n>`, and the run goes on with the next section. So a
  program that fails to build or fit costs only its own section's evidence.
- The JSON report is rewritten after every section, with an `in_progress` field.
- `--deadline-s` stops the run cleanly between device calls and lists the rest under `deadline`. The runner passes
  its container timeout less 600 s. A cut decisive section (S, T, K or E) means NO-DECISION.
- On SIGTERM (the container timeout, which `docker run` forwards to the probe) the probe writes the partial report at
  once, unwinds, closes the device and exits 143.

**The verdict line** is `K64J_P0 verdict=GO|NO-GO|NO-DECISION split=a/b extent=a/b trace=a/b fence=a/b skip=a/b served=a/b served_unqualified_cap=a/b served_pnht1=a/b live=a/b half_tile=a/b dynamic_chunk=a/b skipped_rows=... flag_0x20=... families=N binary_stage=4`, followed by `findings=N`, `sections_failed=...` and `deadline_skipped=N` when there are any.

- **GO:** every decisive comparison is byte-equal, every liveness control moved, nothing failed and the deadline cut
  no decisive section. The runtime position reproduces the compile-time split on the served binary, and the skip
  holds. A finding does not block GO, but it goes to the K64j design review.
- **NO-GO:** a decisive difference on a valid run. R1 or R2 is real, and the plan falls back to P1 (decision D-b).
- **NO-DECISION:** any of the following:
  - a failure: a wrong binary or wrong stock kernels, no compact scratch, a served mode that differs at a qualified
    capacity, a qwen program without its factory line, a section that raised, or the watchdog;
  - a dead liveness control;
  - a decisive section cut by the deadline;
  - SIGTERM.

### What P0 does not prove

- **The new non-causal branch.** K64j's own reader, compute and writer under 0x20 are unbuilt; K1-K4 prove them.
  P0 proves the shared machinery on hardware: the runtime word through `c_8`/`c_15`, the split, the tree, the
  trace re-reading the word, and the skip.
- **Share and slice with a runtime extent.** Qwen modes refuse `cur_pos` today.
- **The boundary cap.** Rows past E cannot see their own keys.
- **The narrow-mask content relative to E - 256.** This is mask refresh v2.
- **That idle rows are harmless downstream.** See R2 above.
- **K64i's tail at the families it was never card-qualified at.** Section E records it, but only at K1's six extents.

## Running it

**On CPU, from this directory** (Python 3.11; the bash runner tests need Git Bash on Windows). It takes 20-25 s wall
on the (loaded) development Windows box with `OMP_NUM_THREADS=2`, the CPU job's setting, and about 28 s of CPU. The
version before the review took 25-31 s wall and 43-46 s CPU there, back to back, and the review measured 47-52 s for
it elsewhere. Since then the dry runs are one full run plus single-section broken variants, and the fake converts the
K/V pool to float once.

```
py -3.11 -B -m unittest test_k64j_probe
```

`CompiledMirrorTests` needs a host g++, which the CI runner has; locally, set `QWEN_PF_GXX` or run it in WSL:
`python3 -B -m unittest test_k64j_probe.CompiledMirrorTests`. The CPU workflow runs this directory
(`qwen-integration-cpu.yml`), and `scripts/ci/test_qual_card.py` registers the runner.

**On card B, through the card-B runner** (`docs/card-b-runner.md`): put these lines in `.github/card-b-job.env`,
then push `experiment/card-b-vN`, first pass first:

```
CARD_B_HARNESS=optimisation/ttnn-op/k64j_probe/run_card_b.sh
CARD_B_ARGS=
CARD_B_ENV=WATCHER=1 KOPGRAFT64=/home/thatch/opgraft-K64i       # pass 1: watcher, 2 extents, 8 families, no timing
CARD_B_ENV=KOPGRAFT64=/home/thatch/opgraft-K64i                 # pass 2: everything, about 40-70 min
```

- **Pass 2 in halves (optional).** Pass 2 is about 125 JIT builds, 64 of them T's per-family legacy references,
  against a 90-minute container cap. The deadline makes an overrun clean but incomplete. To halve the risk, run it as
  two jobs, `CARD_B_ARGS=--sections S,T --no-timing` and then `CARD_B_ARGS=--sections K,E,D,N`. GO needs both.
- **`--served-pnht1` (optional, a separate job).** It adds about 24 JIT builds of qwen programs on shapes never run on
  hardware. Its results are warnings only, but a hang there still costs a card-B reset, so it never rides with pass 2.

- **Dry run first (optional, no device).** `CARD_B_ENV=K64J_DRY_RUN=1 KOPGRAFT64=/home/thatch/opgraft-K64i` checks
  the graft on the runner and prints the argv.
- **Results.** They come back in the `card-b-<run id>` artifact: `probe-<stamp>.json`, `.log` and
  `.json.native.log`. The runner echoes the `### K64J_P0 ...` line.
- **A hang.** Exit 3, 124 or 137, or `Timeout (` in the log, needs a person to reset card B by the printed hint.
  One 124 is not a hang: the log shows `ERROR terminated by signal`, which the probe prints only after it caught
  SIGTERM and closed the device. The runner then prints `TIMEOUT` instead of the hint; see the partial report.
  The runner never resets.

**By hand on the rig:** `bash run_card_b.sh` (with `WATCHER=1` first), from a checkout with `sdpa_decode_qwen/` beside
this directory. Results land in `~/kwork64/k64j/card-b/`.

## K64j itself: the edit list and the build (next step, not built here)

P0 needs no new kernel: it runs the stock causal and legacy programs of K64i. If it says GO, K64j is the following
edits over K64i's stage-4 factory `1634369677ae0247…` and kernels. Each is a sha-pinned generator step in the style of
`apply_factory_slice.py` and `make_slice_kernels.py`, and each inverts to stage 4.

**K64j never edits a stock kernel.** Every 0x20 program is built from qwen-named kernels only: the qwen or slice
reader, the qwen compute kernel and the slice writer (F21). The stock `reader_decode_all.cpp`,
`writer_decode_all.cpp` and `sdpa_flash_decode.cpp`, and the shared `dataflow_common.hpp` and `rt_args_common.hpp`,
stay byte for byte as vendored here. There are two reasons:

- **The exact profile.** The stock kernels also compile the native decode (the exact profile) and every legacy
  program. Editing them would change programs the exact profile runs.
- **The cache key.** The kernel-cache key hashes only the graft's `*qwen*.cpp` kernels
  (`scripts/ci/lever_n_m3native_run_arm.sh:279-284`, `scripts/ci/build-c2-serving-image.sh:33-38`). An edited stock
  kernel or header would leave the key unchanged, and a warm cache would serve the stale binaries.

**Factory**

| Edit | What |
|---|---|
| F19 | `kQwenRuntimeExtent = 0x20u`; admit it in the unknown-flag mask. |
| F20 | Under 0x20: require tail (0x1), a narrow mask (`qwen_mask_width_t == Sk_chunk_t`), an interleaved int32 row-major `cur_pos` tensor with B entries, and no causal flag. Relax `"modes are non-causal, full-window and take no cur_pos tensor"` to "no cur_pos tensor unless 0x20". |
| F21 | A reader suffix CTA `runtime_extent` (both qwen readers). Compute CTA 33. Select a writer carrying `runtime_extent` for EVERY 0x20 program: the slice writer gets W4 (its W1-W3 should reduce to the stock writer at `q_slice = 0`; the generator's source tests must prove it). `c_8`/`c_15` already exist whenever a `cur_pos` tensor is passed (factory :556-570). |
| F22 | One `[QWEN-SDPA] runtime-extent ...` log line per program, a new binary literal for the admission check. |

The `cb_bytes` of every 0x20 program grows by two sticks. The card tests' L1 table must be re-derived.

**Kernels**

| Edit | What |
|---|---|
| R10 | The qwen and slice readers take the causal block's `cur_pos` read under `is_causal \|\| runtime_extent`, with the same `UINT32_MAX` return before any Q, K or V read and before the share handshake. With share, every entry uses slot 0. `static_assert(!runtime_extent \|\| mask_tail)`. |
| C3 | Compute reads `c_15` under the same condition. `apply_mask_at_last_chunk` stays causal-only; the tail predicate is already right. |
| W4 | The writer reads `c_8` under the same condition, and never runs `generate_mask` under `runtime_extent`. |

**Python**

- `pooled_attention_replay`: a mode `extent`, flag 0x20, with its required binary literal.
- The gate: `SDPA_MODE_FLAGS['extent'] = 0x20`.
- The card tests: the unknown-flag control stays 0x10.

**Build** on the rig in the `ttbuild` container (memory tt-metal-op-build-workflow and the K64i recipe in
`../sdpa_decode_slice/README.md`):

1. **Ship.** Put the new directory and its siblings (`sdpa_decode_slice`, `sdpa_decode_qwen`,
   `sdpa_prefill_chain`) under `~/kwork64/k64j/`, as a base64 tar over plink stdin, LF only.
2. **Preflight.**
   - `docker ps` shows `ttbuild`.
   - `~/opgraft-K64i/MANIFEST.sha256` verifies, and its `_ttnncpp.so` is `cf54d716…`.
   - The saved factory bases exist: `~/kwork64/k64f/sdpa_decode_program_factory.cpp.3e0a69af…` and the prefill
     factory `fd8c0676`.
   - ttbuild's stock decode kernels are 49a05926 / 734c90c0 / d24769bd.
3. **Stage.**
   - Patch the saved 3e0a69af factory: stage 3, then stage 4 (`apply_factory_slice.py`), then F19-F22.
   - Stage the prefill factory as K64i does.
   - Copy the regenerated kernels into `$OPS/transformer/sdpa_decode/device/kernels/`
     (`rm -rf` before `docker cp`: it nests into an existing directory).
4. **Build.** `docker exec ttbuild ninja -C /opt/tt-metal/build_Release ttnn/_ttnncpp.so ttnn/_ttnn.so`, about 30 s.
   Only the factory changes, so there is no new unity-build namespace.
5. **Assemble** `~/opgraft-K64j.partial`: K64i's contents, the new `.so` files, and `sdpa_decode/` with the new
   kernels, its factory `.cpp` put back to the audited base.
6. **Verify.**
   - `strings _ttnncpp.so | grep -oE 'QWEN_[A-Z0-9_]+' | sort -u` is a superset of K64i's and the image's
     (memory graft-so-drops-image-patches).
   - The new F22 literal, `[QWEN-SDPA] q-slice rows_per_kv=` and `QWEN_SDPA_TREE_SCRATCH_ROUNDS` are all present.
   - The `qwen_draft_fp32_intermediates` count equals K64i's.
   - `diff -rq` against K64i's `sdpa_decode/` shows only the edited kernels.
   - Write `MANIFEST.sha256` and print `K64J_TTNNCPP_SHA256`.
7. **Restore ttbuild.** Put back factory 3e0a69af, remove the qwen kernels, rebuild, and check that no qwen literal
   remains. Use an EXIT trap.
8. **Card B.** Run K1-K4 with `KOPGRAFT64=~/opgraft-K64j EXPECT_TTNNCPP_SHA256=<K64J sha>`, `WATCHER=1` first.
9. **The arm.**
   - `QWEN_FAST_RUNTIME_BINARY_SHA256=<K64J sha>`.
   - The `.so` at both `build_Release/lib` and `build_Release/ttnn`, which the arm already does.
   - The kernel-cache key already covers every `*qwen*.cpp` under the graft's `sdpa_decode`. It takes the stage-3
     pair by content and every other qwen kernel by path and hash (`lever_n_m3native_run_arm.sh:279-284`,
     `build-c2-serving-image.sh:33-38`). So K64j's edited qwen kernels change the key by construction.
   - Any edited kernel that is not qwen-named, a header included, must enter the key first. Prefer not to edit
     one: K64j edits none (above).

## Files

| File | What |
|---|---|
| `split_model.py` | The pure-Python split, cores per head, tree and stale-writer hazard. |
| `probe_k64j_card_b.py` | The card-B harness. |
| `run_card_b.sh` | The `qual_card` runner: card B by board id; the serving pair only with `ALLOW_SERVING_CARD=1`; `K64J_DRY_RUN=1`; `WATCHER=1`. |
| `test_k64j_probe.py` | The CPU tests, including the fake-ttnn dry run and its broken variants. |
| `fixtures/rt_args_common.1b52c60d.hpp` | Vendored tt-metal 9f9cd4fd (Apache-2.0) for the compiled mirror test. |
| `fixtures/sdpa_decode_device_operation.ba2b3fc9.hpp` | Vendored tt-metal 9f9cd4fd (Apache-2.0) for the compiled mirror test. |
| `fixtures/reader_decode_all.49a05926.cpp`, `writer_decode_all.734c90c0.cpp`, `sdpa_flash_decode.d24769bd.cpp`, `dataflow_common.e4623a22.hpp` | Vendored tt-metal 9f9cd4fd (Apache-2.0): the stock `cur_pos` blocks the served kernels must carry, the `generate_mask` the mask test mirrors, and the stock sources of the runner tests' fake graft. |
