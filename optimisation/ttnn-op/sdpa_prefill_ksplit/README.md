# sdpa_prefill_ksplit: K-split of the chunked prefill SDPA for the tail passes (prefill lever PF3-K)

A tail pass of 1,024 or 512 rows keeps 48 or 24 of the grid's 130 cores busy under the oneq call (`../sdpa_prefill_oneq`, flag 0x8) and a q chunk's K stream is serial on its core. Flag 0x10 of the
per-call program word (`0x5EFA0000 | flags`) gives every q chunk S = min(4, 130 / busy) slices on the idle cores, streams the K blocks `[s B / E, (s+1) B / E)` on slice s, and folds the slices'
partial (normalised output, log-sum-exp) pairs in slice order with the pinned ring merge. The design, the estimate (about 1 percent of route time), the numerics waiver card and the card gates are in
`docs/tp4-prefill3-ksplit.md`. **Nothing here has been compiled by the device toolchain or run on a card.** The flag is default off, needs the oneq flag and the K64j-OQ-KS graft, and is gate-only.

| File | What |
|---|---|
| `ksplit_planner.py`, `ks_common.py` | the slice arithmetic (one C++ text the three kernels carry byte for byte), the slice-major plan, the time model |
| `apply_factory_pk.py` | factory patch PK0-PK13 over the oneq factory (sha-checked, anchored, inverted, inside `create_descriptor`) |
| `make_ks_reader.py`, `make_ks_writer.py`, `make_ks_compute.py` and their outputs `reader_interleaved_qwen_ksplit.cpp`, `writer_interleaved_qwen_ksplit.cpp`, `sdpa_qwen_ksplit.cpp` | the three kernels, generated from the committed chain reader and the pinned tt-metal sources in `fixtures/` |
| `ks_protocol_model.py` | S chains per q chunk plus the hand-off as a discrete-event model: termination, payloads, semaphores, the planted hang, four mutations |
| `host_exec_ksplit.py`, `stub_compile_ksplit.py`, `stubcheck/` | the patched factory's real decision C++ run with g++ and stubs; stub compiles of the touched regions, the reader and the writer |
| `ksplit_card_m.py`, `run_card_m_ks.sh` | the card-M qualification harness and its runner (template, not run) |
| `build_k64j_oq_ks.sh` | builds `opgraft-K64j-OQ-KS` beside K64j-OQ (never overwrites; six publish checks; dry run with `KS_BUILD_DRY_RUN=1`) |
| `ksplit_estimate.py`, `ksplit_emulation.py` | the ESTIMATE tables of the doc and the CPU emulation of the two arithmetics behind its numerics section |

Model side (`scripts/ci/`): `sdpa_pf_ksplit_tp.py` (the hook and the audit), `ksplit_numerics.py` (the statistic and the provisional bound), `ksplit_smoke.py` (the smoke rule),
`make_pf3k_profiles.py` (the three gate-only twins; never hand-edit `qwen_c2_profiles.json`).

Checks, all CPU (from this directory; the card and emulation tests need torch and skip without it):

    python3 -B -m unittest discover -p 'test_*.py'
    python3 -B ksplit_estimate.py --check
    python3 -B ksplit_estimate.py --doc ../../../docs/tp4-prefill3-ksplit.md        # the doc's tables are the script's output
    python3 -B make_ks_reader.py --check . && python3 -B make_ks_writer.py --check . && python3 -B make_ks_compute.py --check .

When `apply_factory_pk.py` or a generated kernel changes, re-record `PK_FACTORY` / `PK_SCRIPT` in `build_k64j_oq_ks.sh` and the kernel shas in `run_card_m_ks.sh` (tests hold them equal).
