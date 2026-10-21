# tp4/s7: the ship-7 candidate

Status: **CANDIDATE, never run on a card.** Built on `tp4/s6` (the ship-6 candidate, `docs/tp4-next2.md`, which passed its card gates on 2026-10-21) with four already-built, CPU-tested branches merged in, each lever behind its own
switch so that a failing one comes out by deleting one line. Every effect below is an **ESTIMATE**, none is a measurement. The owner approved the ship on 2026-10-21 **conditionally on the S7 window gates** ("Yes approve s7 if it
passes"); the decision sentence in `scripts/ci/fusion-wp/S7.json` records the conditions, and production switches to the image only after the gates are read.

## 1. What is in it

`...-fx-traffic-s7` is the ship-6 profile (`...-fx-traffic-s6`, unchanged, its waiver untouched) plus exactly these lines (and the waiver id, the profile's own name for its waiver):

| lever (branch) | line(s) | what it does | ESTIMATE |
| --- | --- | --- | --- |
| threaded host-tier upload (`tp4/upload-1`, `docs/tp4-upload1.md`) | `QWEN_PREFIX_TIER_IO_THREADS=4`, `QWEN_PREFIX_TIER_VERIFY_THREADS=4` | the staging copies of the tier's raw block IO run on workers beside the device calls, the sampled digest checks on workers; a failure latches it off and repeats the call serially | a 254k restore of about 1 s falls to 0.4-0.6 s: **under 0.5 s of a 5.4-6.5 s returning first token** (the brief's "8 s to 0.4 s" divided by a conversion rate that the preconverted checkpoints of ship 5 already removed) |
| admission pipelining (`tp4/sched-2`, `docs/tp4-sched2.md`) | `QWEN_FAST_ADMISSION_SEED_FIRST=1` | the seed token of a returning turn leaves before its engine is bound; the bind runs at the start of the next `schedule()` | returning turn first token p50 1.46 s to about 1.19 s |
| memory-ledger trim (`tp4/sched-2`) | `QWEN_FAST_ADMISSION_LEDGER_TRIM=1` | the memory ledger ends after the attach (no per-rebind bracket) | about 1.7 s of a 15.4 s burst into a fresh process, nothing warm |
| prompt lookup on the engine-reuse path (`tp4/stream-1`, `docs/tp4-stream-1.md`) | `QWEN_FAST_LOOKUP_DRAFT=n3m12` | a host-side lookup over the user's own tokens replaces the drafter's proposal rows where it matches; the target verifies every row | about +3 percent tokens a second, +8 percent on edits (short-prompt simulation) |
| traced masked tails (`tp4/prefill-3`, `docs/tp4-prefill3.md` PF3-T) | `QWEN_FAST_PREFILL_TAIL_TRACE=1` | the masked tail pass of a prompt below the 2,048-row bucket is a trace replay (buckets 128-1,024, arena 320 MB a chip) | 110-170 ms off a tail pass, 60 ms a prompt on the production population |

**Left out, default off, in no profile:** the hold reuse (`QWEN_FAST_ADMISSION_HOLD_REUSE`: its own doc wants a ballast-boundary run before any ship; the contract keeps it a gate's own), the K-split of the tail passes
(`QWEN_FAST_SDPA_PF_KSPLIT`, a tolerance-class numerics change that needs **its own owner waiver**; its code is in the tree because it came in the same branch, behind the flag and a contract rule that refuses it on a traffic profile
unless a waiver names it), the two-candidate tree verify (`tree_verify.py` is imported by nothing in serving) and the lookup shadow (`QWEN_FAST_LOOKUP_SHADOW`).

**The lookup's index cap.** `prompt_lookup.MAX_INDEXED_PROMPT = 32768` indexes only the last 32,768 prompt tokens. A later simulation (`docs/tp4-drafter-long.md` on `tp4/drafter-long`) finds the cap removes the lookup's
gain beyond 32k (n3m12 as built x1.014 [0.999, 1.041] against x1.046 [1.012, 1.089] with the whole prompt; at 128k x0.999 against x1.066). The cap is a module constant, not an environment setting, and the index is built on the
engine step thread at every rebind (about 0.07 s at the cap, 0.27 s for 123k tokens, 0.6 s for 254k, laptop figures, stalling every decoding seat), so raising it is a code change with a host-time cost that nothing has measured
on the rig. It is left as built here and is the follow-up if the lookup passes its gates.

## 2. The profiles and the pack

* `c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-fx-traffic-s7` (`make_fusion_profiles.py` from `scripts/ci/fusion-wp/S7.json`): the candidate. **Waiver `ship7-fx-262k-2026-10-21`**: the ship-6 text verbatim plus items (S) to (W),
  one per group. The contract admits seed first and the ledger trim on a traffic profile only under a waiver that names each at this value (`SCHED2_WAIVABLE_LEVERS`, added to `NEXT_WAIVABLE_LEVERS`); the upload levers ride the
  waiver's naming of the tier; the lookup and the traced tails have no contract mechanism (like PF2-H and the op-fusion levers), so the waiver names them in words.
* `...-fx-s7-audit` and `...-fx-s7-audit-mid` (`make_s7_profiles.py`): the audited attach arms, gate-only: the ship-6 audited twin plus every difference of the candidate **except the ledger trim** (the audit's judge reads the
  ledger's points) plus the traced tails' audit; the first for the buckets 128 and 1,024, the second for 256 and 512 (the audited stack leaves the trace region 114.8 MB free: two traces a twin).
* The control of every timed pair is the ship-6 profile itself. Generator order: `make_fusion_profiles.py`, `make_octo3_profiles.py`, `make_ship4_profiles.py`, `make_np1_profiles.py`, `make_xe_profiles.py`, `make_pf2_profiles.py`,
  `make_s5_profiles.py`, `make_s6_profiles.py`, `make_sched2_profiles.py`, `make_pf3_profiles.py`, `make_pf3k_profiles.py`, `make_s7_profiles.py`, then `make_fusion_jobs.py` and `make_s7_jobs.py`. The generators of this chain
  place their twins after the last twin of the generators before them, so each one's `--check` holds whichever ran last.
* The pack (`scripts/ci/references/fusion-jobs/S7`, `make_s7_jobs.py`), one image `tp4-s7-1` (a placeholder tag):

| job | mode | minutes (ESTIMATE) | what |
| --- | --- | ---: | --- |
| B0 | stop | 40 (build) | build the **ship image** from the pushed head with `...-fx-traffic-s7` baked as the serving default and the K64j-OQ graft (refused while the waiver is PENDING): the window's image is the ship image |
| X0 | stop | 1 | status, rescan, reset |
| A1 | stop | 61 | `-s7-audit` (buckets 128, 1,024), warmup, coding, np1_live2, np1_live4, prefill_few_decoders, concurrent8_drain, concurrent8_steady, agent_turns |
| A2 | soft | 41 | `-s7-audit-mid` (buckets 256, 512), warmup, coding, concurrent8_steady, agent_turns |
| T1A / T1B | soft | 18 each | ship-6 control / s7 candidate: warmup, np1_live1, np1_live2, np1_live4, prefill_few_decoders, concurrent8_steady, agent_turns |
| R1 / R2 | soft | 45 each | the restore pair (below) |
| T2A / T2B | opt | 15 each | agent_turns8 pair |
| Z | soft | 5 | status, reset |

Core (X0 A1 T1A T1B Z) 103 minutes of cards; with A2 144; with the restore pair 234; with the agent_turns8 pair 264. The minutes of X0, A1, T1 and T2 are the 2026-10-21 ship-6 window's; A2, R1 and R2 are the upload-1 and
prefill-3 packs' estimates. The audited A1 hit three agent-turn client timeouts from the audited profile's about 2.6 GB per chip DRAM admission hold (the owner accepted that for ships 5 and 6); the traced tails' 320 MB arena comes out of
the same headroom, so a hold line or a client timeout on A1 is the audit's weight, not the candidate's.

**The restore test.** No smoke test evicts a returning session to the host tier and restores it (the smoke tests' returning turns restore too, untimed and mixed with prefill). The prefix gate has exactly that cycle as a plan
(`tier-upload-timed` / `tier-upload-timed-control`: cold prefill, device hit, floods the size of the KV pool, the returning hit, its cold twin, at ~64k / 128k / 254k), so R1 and R2 are that plan on the control and on the candidate.
No new harness. `upload_restore_compare.py` wants an ABAB; the single pair is passed twice and its repeat floor reads 1.

## 3. Removing a lever

Delete its line from the ship entry of `scripts/ci/fusion-wp/S7.json` (for seed first and the ledger trim also from the waiver's `levers`), run the generators in the order above and rebuild: `QWEN_PREFIX_TIER_IO_THREADS` with
`QWEN_PREFIX_TIER_VERIFY_THREADS`, `QWEN_FAST_ADMISSION_SEED_FIRST`, `QWEN_FAST_ADMISSION_LEDGER_TRIM`, `QWEN_FAST_LOOKUP_DRAFT`, `QWEN_FAST_PREFILL_TAIL_TRACE` (or narrow `QWEN_FAST_PREFILL_TAIL_TRACE_BUCKETS`). The audited twins
read the candidate's differences from ship 6, so they follow.

## 4. Merge notes (tp4/s6 + four branches)

All four branches share the merge base `758264ac` with `tp4/s6`. The merges were clean except where each branch edited the census tests that exempt generated profile families (one-line unions with the s6 terms), `profile_twins.py`,
`qwen_c2_profiles.json` (kept s6's file and regenerated the twins) and, for `tp4/prefill-3`, `serving_c2_contract.py` (the K-split waivable-lever block beside the s6 governor entry) and the fusion-jobs ORDER header. A tracked
`.pyc` that `tp4/prefill-3` added is dropped. The sched-2, prefill-3 and prefill-3-split generators were re-anchored after the s6 audited twin so that all generators commute, and the tests that compared "every profile that existed
before this branch" with a fixed parent now subtract the other branches' generated twins.
