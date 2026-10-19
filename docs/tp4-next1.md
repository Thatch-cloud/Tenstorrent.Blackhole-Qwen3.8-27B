# tp4/next-1: the ship-5 candidate

Status: **integrated and tested on CPU; nothing here has run on a card together, and NP1, exact resume, the sliced restore and W3 WC have not run on a card at all.** Nothing in
production moved. The branch is `tp4/next-1`, cut from `tp4/swift` (the ship-4 stack plus the checkpoint selection, the production tree), and no experiment tag is pushed from it.

## 1. What is in it

| From | What | Switch | In the traffic candidate | In the audited twin |
| --- | --- | --- | --- | --- |
| `tp4/np1-fastlane` | the lone-user padded round (NP1): a lone live user decodes on the padded 64-row block, its third idle segment on a scratch KV block past vLLM's pool (`docs/tp4-np1-fastlane.md`) | `QWEN_FAST_SOLO_PACKED=1` | yes | yes |
| `tp4/np9-answer32k` | the per-request output ceiling 16,384 to 32,768 (`docs/tp4-np9-answer32k.md`) | the Swift catalog key `"output_budget": 32768` (data, not a profile line) | yes, for every Swift boot of the image | yes |
| `tp4/exact-end` (which contains `tp4/exact-resume`) | exact resume P0: the undrop and the drafter-window carry (`docs/tp4-exact-resume.md`, `docs/tp4-exact-end.md`) | `QWEN_PREFIX_STICKY_UNDROP=1`, `QWEN_PREFIX_WINDOW_CARRY=1` (audit: `..._CARRY_AUDIT=1`, gate only) | yes (not the audit) | yes, with the audit |
| `tp4/tier-slice` | the sliced host-tier restore (`docs/prefix-store-hygiene.md`) | `QWEN_PREFIX_TIER_RESTORE_SLICE=512`, `..._MS=200` | yes | yes |
| `tp4/w3-wc` | W3 WC: the GDN windows, conv and gates in one launch (`docs/tp4-w3-wc.md`) | `QWEN_FAST_TP4_GDN_WC=1` (audit: `..._WC_AUDIT=1`) | **no** | yes (and a fallback twin without it) |
| `tp4/prefill-2` | PF2-H: the final prefill step writes the decode slot from device-side clones of the GDN scratch instead of a host round trip (240 reads, 240 uploads; `docs/tp4-prefill-2.md`) | `QWEN_FAST_PREFILL_HANDOFF=1` (audit: `..._AUDIT=1`, gate only) | yes, **only if the audited run passes** | yes, with the audit |
| `tp4/sched-1` | the lean admission (`QWEN_FAST_ADMISSION_LEAN=1`), the complete spills of the host tier (`QWEN_PREFIX_HOST_TIER_SPILL_MODE=complete`, default knobs) and the existing `QWEN_FAST_ADMISSION_DIAG_TRIM=1` (`docs/tp4-sched-1.md`) | those three | yes | yes |
| `tp4/agent-turns` | the production-shaped multi-turn agent test (smoke names `agent_turns`, `agent_turns_long`) | opt-in smoke names | n/a | named in A1 and both timed legs |

The sched-1 governor knobs (`QWEN_FAST_LEVERN_ADAPTIVE`, `_OWED_CAP_MS`, `_SHORT_SHARE`, `_SHORT_WAIT_S`) are **not** in the candidate or in its twins: they trade decoder speed for first-token time and await the owner;
they stay available as sched-1's own gate-only arm `-sch-all` (no job of this pack runs it).

Also in the candidate: `QWEN_FAST_GRID13=1` (the decode verify sub-switch joins `prefill,drafter`; NP6, audited exact on cards in the ship-4 audit run) and the owner-approved host RAM budget
(`QWEN_PREFIX_HOST_TIER_GIB=48` beside `QWEN_PREFIX_STORE_GIB=16`: the tier's GiB is the whole prefix state, so a 32 GiB KV share and a 16 GiB checkpoint store).

## 2. What was left out, and why

* **W3 MR** (`tp4/w3-mr`: the verify readback as mesh-level reads, and the quad's top-16 outputs as one joined tensor). The shipped stack already carries the same lever twice over:
  `QWEN_FAST_BATCHED_READS=async` (WPH: the verify read-back as non-blocking copies and one fence, the quad collect as a few mesh reads) and `QWEN_FAST_TP4_ROUND_HOST_READ=1`. Merging MR
  would put two implementations of one lever on the verify path (it conflicts in `packed_verifier.shard_predictions` and `quad_draft_tp.read_quad_outputs`, both already rewritten around those),
  and MR's one piece the stack lacks (the joined quad outputs, 2a) is built on MR's own reader. Left out.
* **W3 keyed** (`tp4/w3-keyed`: the keyed staging diff). The shipped `QWEN_FAST_TP4_ROUND_HOST_KEYED=1` is the same idea (the staged values keyed on each segment's start and page table, the tokens
  buffer alone written); the two rewrite `verify_prestage.py` (seven conflicting hunks). Left out.
* W3 WC does merge, and does not duplicate a shipped fusion (it replaces G3 to G8 of a GDN layer, F1 inside). It takes **2 or 4 users**, so it declines the octo block (8 users of 8 rows) with one logged line;
  `packed_verifier` therefore skips WC's replay audit on the (8, 8) block, and the WC smoke rule accepts that one decline on a profile that names the octo shape (and nothing else).
  WC's kernels have **never been through a compiler**: the first audited attach is their first compile. That is why the audited twin has a fallback without WC (`-fx-s5-audit-nowc`, job `A1N`).

## 3. Conflicts and judgment calls (nothing dropped from either side)

* Exact resume came in twice: first as `tp4/exact-resume` (cut 67 commits before `tp4/swift`), then replaced by `tp4/exact-end`, which is P0 merged onto `tp4/swift` and carries a defect fix the first merge lacked:
  P0 had switched the tile-aligned drafter window (`QWEN_FAST_TAP_WINDOW_ALIGNED=1`, ON in the ship profile) off under a carry, bringing back a 640 to 770 ms compile on every carried turn;
  `prefix_window_carry.Carried.align()` keeps it on. (This branch's own first resolution of that join, a single exact-piece concat behind the carried rows, would have had the same compile: it is dropped, and
  `dflash_prefill_window` and `test_exact_resume` are `tp4/exact-end`'s.) Exact-end also gives the prefix gate the Swift selection (`C2_MODEL_REPO`/`C2_MODEL_REVISION` ride the `prefix` action), the gate-only
  `-fx-s4-xe` twins, the XE pack and `exact_end_model.py`.
* The prefix files (`qwen_prefix_registry`, `qwen_prefix_scheduler_patch`, `prefix_markers`, `prefix_judge`, `prefix_replay`, `c2_prefix_gate`, `c2_serving_job`, `qwen_prefix_metrics`, `serving_c2_contract`) are
  exact-end's beside the tier work, with the sliced restore's additions on top and every census exemption kept (NP1, XE and the ship-5 twins).
* The contract (`serving_c2_contract`, `serving_octo`): a **traffic** profile may now name `QWEN_FAST_SOLO_PACKED`, `QWEN_PREFIX_STICKY_UNDROP` and `QWEN_PREFIX_WINDOW_CARRY` **only under its owner traffic
  waiver naming each** (`NEXT_WAIVABLE_LEVERS`), and the tier at 48 GiB beside 32 (`TIER_WAIVABLE_ALSO`). The audits, the audit sampling, the quotas and every other gate name stay a gate's own. The
  lone-user admission is now `admitted_run` (a gate run or a waiver run), like octo-T8. This is the contract change `docs/tp4-np1-fastlane.md` section 9 said the integrator must make. The same list admits
  `QWEN_FAST_ADMISSION_LEAN=1` and `QWEN_PREFIX_HOST_TIER_SPILL_MODE=complete` (sched-1; the spills need the tier, whose waiver the candidate holds); the governor knobs stay refused on traffic.
* The memory ledger (`memory_ledger.py`) logs the trace region's headroom: at P7 (after the attach), P12 (after the first packed round) and P13 one line a chip,
  `[MEMLEDGER] phase=P7 chip<n> trace_total= trace_used= trace_free= trace_largest_free=` (or one `trace unavailable` line), and the same figures in its JSON report. It is a read of the allocator view the before/after points already
  read; A1 records it (the next prefill kernel decision, traced masked tails, needs the number). `test_memory_ledger.TraceHeadroomTests`.
* The placement chain of the generated twins is ship-4, np1, sched, exact-end, prefill-2, ship-5 (`make_xe_profiles` and `make_s5_profiles` anchor on the previous link, so every generator's `--check` holds whichever ran last).

## 4. The profiles (`scripts/ci/qwen_c2_profiles.json`, generated: never hand-merged)

* `...-fx-traffic-s5` (generated by `make_fusion_profiles.py` from `scripts/ci/fusion-wp/S5.json`): the approved ship-4 profile plus exactly the differences of section 1 and the waiver id. **Waiver
  `ship5-fx-262k-2026-10-19`, decision PENDING.** The ship-4 profile and its approved waiver are byte-identical (held against the parent commit by `test_s5_profiles`). The owner approves at ship
  time: `python3 scripts/ci/make_fusion_profiles.py --approve-ship "APPROVED ..." --ship traffic-s5` changes one sentence and regenerates; `test_s5_profiles` passes on either decision.
* `...-fx-s5-audit` (generated by `make_s5_profiles.py`): the ship-4 audited twin plus every difference of the candidate, the carry's audit and W3 WC with its audit, less F1's audit (`gdn_wc` refuses it beside WC).
  `...-fx-s5-audit-nowc`: the same without WC.
* No gate-only timed twin: the candidate is a traffic profile whose decision is PENDING, and such a profile boots in a gate or a smoke (only a build refuses to bake it), so the timed arm is the candidate itself.
* `...-fx-s4-np1`, `...-fx-s4-np1-audit` (NP1's gate-only twins) and `...-fx-all-er`, `...-fx-all-er-audit` (exact resume's) come with their branches.

## 5. The window packs (`scripts/ci/references/fusion-jobs/S5`, `make_s5_jobs.py`)

Templates only; image placeholder `tp4-next-1`; every smoke and prefix job carries the Swift keys (the replay and build steps cannot: the job reader refuses them beside those steps). Three ORDER files share the folder (the window driver reads a fixed `ORDER.txt`: run another window from a commit where its file is `ORDER.txt`).

* **Window 1** (`ORDER.txt`, 220 minutes of cards after the 40-minute build): B0 build (while production serves), X0 15, **A1** 110 the combined audited attach (stop; warmup, coding, np1_live2, concurrent5_split,
  concurrent8_drain, concurrent8_steady, agent_turns), **T1A** 45 control (the approved ship-4 profile) and **T1B** 45 candidate at one, two and four live seats with eight beside them and agent_turns on both legs, Z 5.
  P0's strict proof (the exact-resume prefix gate on real multi-turn text, 120 minutes) does not fit a 3 h window: it opens window 2. With it window 1 would be 340 minutes.
* **Window 2** (`ORDER-W2.txt`, 495 minutes with the optional A3): X0, **A2** 120 the exact-resume gate on `-fx-s5-audit` on Swift (stop: every cold/hit pair identical, the oracle's Q, the carry's audits,
  the aligned-window line on a carried turn), NP9's N1 75 (stop) N2 70 N3 80 on the candidate, A3 100 the audited sliced restore (optional), SP1 30 the split-attribution probe (soft, no build), Z.
* **Ship flow** (`ORDER-SHIP.txt`, after the owner's approval): B1 the ship build and SR the platform replay. The replay boots the profile by the image default, so both need the approval.

The agent test is named by `AGENT_TURNS_TEST` in `make_s5_jobs.py`; empty it and rerun `--write` to take it out of the pack.

## 6. Risks

* The WC kernels' first compile is A1 (fallback A1N). NP1's K1 (a block id past the pool in a page table) is also A1's first. WC does not touch the octo block, so it gives nothing at six or more live seats.
* NP9's catalog key reaches every Swift boot of an image built from this branch, window 1's included; deleting the one line restores 16,384.
* The tier at 48 GiB plus the carry's window rows (up to 105 MB a checkpoint) raise the host RAM the engine process holds; the smoke container runs under an 80 GiB limit.
* `real_text_compare` counts `QWEN_FAST_OUTPUT_BUDGET` as arithmetic: an exactness arm on the 32,768 ceiling held against a reference made at 16,384 reads NOT_COMPARABLE, never a false pass.
* Two lever pairs of the shipped stack are untouched by the candidate and unproven together with it: exact resume beside engine reuse's rebind, and the sliced restore beside octo-T8.
