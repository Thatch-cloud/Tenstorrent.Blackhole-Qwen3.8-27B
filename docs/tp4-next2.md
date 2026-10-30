# tp4/s6: the ship-6 candidate

Status: **CANDIDATE, never run on a card as a whole.** Built on `tp4/next-1` (the ship-5 tree, `docs/tp4-next1.md`) and unit-tested on the CPU. Every effect below is an **ESTIMATE**, none is a measurement.
The owner approves at ship time; until then the waiver decision of `...-fx-traffic-s6` is PENDING and no build bakes it.

## 1. What is in it

`...-fx-traffic-s6` is the ship-5 profile that serves production (`...-fx-traffic-s5`, unchanged, its waiver untouched) plus exactly two switches. Its environment differs from ship 5's in these lines and no others
(the third, the waiver id, is the profile's own name for its waiver):

| line | what | acts at | in the traffic candidate | in the audited twin |
| --- | --- | --- | --- | --- |
| `QWEN_FAST_TP4_GDN_WC=1` | W3 WC: the GDN layer's windows, convolution and gates in one launch (`docs/tp4-w3-wc.md`) | 2 or 4 live users of an M3 block; the octo block (8 users of 8 rows) declines it with one logged line (`[PINDIAG] tp4 gdn wc fell back reason=8 users (WC takes 2 or 4)`), which is expected | **yes** (new) | yes, with its audit (already in the ship-5 twin) |
| `QWEN_FAST_LEVERN_ADAPTIVE=1` | the adaptive Lever N governor, default knobs (`docs/lever-n-adaptive-governor.md`) | one or two live decoders while a long prompt prefills | **yes** (new) | yes (new) |
| `QWEN_FAST_OCTO_WAIVER=<id>` | the owner traffic waiver's id (`ship6-fx-262k-2026-10-21`) | | yes | no (a gate-only twin has no waiver) |

Everything else is ship 5: the lone-user padded round (NP1), exact resume P0, the sliced host-tier restore, the verify sub-switch of the 13x10 grids, PF2-H, the lean admission, the complete spills, the 48 GiB tier beside the
16 GiB checkpoint store, and the Swift catalog's 32,768-token output ceiling.

**WC.** It rode the audited twin `...-fx-s5-audit` in two card runs with 0 mismatches over about 12,800 audited rounds (the status line at the top of `docs/tp4-w3-wc.md`, "never compiled or run on a card", predates those runs).
It has not been in a timed run. It is not in the ship-5 traffic profile; this candidate adds it. It takes 2 or 4 users, so it gives nothing at six or more live seats (the octo block). The contract has no mechanism
for it (like PF2-H and the op-fusion levers), so the waiver names it in words and the candidate carries it as the owner's decision.

**The governor.** `tp4/prefill-1`'s commit `b06724f0` ("Adaptive Lever N governor ... gate only, default off") is **already in this tree**: `tp4/sched-1` ported it onto the ship-4 stack as `98c70a9b` (policy, scheduler install and
step-line fields, smoke rule, tests and the design note), and `tp4/next-1` carries it. Nothing of its code is re-ported here, so flag off it is byte for byte today's behaviour (a test compares the whole log of two schedules). What
this branch adds is the way in on a traffic profile: `serving_c2_contract.NEXT_WAIVABLE_LEVERS` gains `QWEN_FAST_LEVERN_ADAPTIVE: '1'`, so a traffic profile may set the switch only under an owner traffic waiver that names it at that
value. Its three knobs (`_DECODERS`, `_SHARE`, `_TTFT_S`) and the sched-1 pacing knobs (`_OWED_CAP_MS`, `_SHORT_SHARE`, `_SHORT_WAIT_S`) stay a gate's own, and the policy still REQUIRES the decode-gap floor and the merged route.
The smoke test that measures what the governor changes, `prefill_few_decoders` (a long cold prompt arriving while one or two decoders are live: the arrival's first-token time and the decoders' longest gaps and progress), is
in the tree already (from `tp4/prefill-1`, via `tp4/sched-1`); `scripts/ci/prefill_ladder_report.py` reads a pair of its logs.

### Expected effect (ESTIMATE)

* **Governor**, from `docs/lever-n-adaptive-governor.md` (the timing model of `levern_policy.StepTimes`, whose 254k seeds are not measured): a cold long prompt arriving beside one or two decoders reaches its first token **31 to 44
  percent sooner** (32k: 18.7-19.0 s to 12.0-13.0 s; 128k: 84-87 s to 49-53 s; 254k: 182-187 s to 114-119 s). The price is the decoders': during that window they get about a tenth of the wall time in decode rounds instead of
  half, so their inter-token rate falls about five-fold for a window about 40 percent shorter. At three or more live decoders (the production load) the pacing is today's, step for step: no effect expected there.
* **WC**, from `docs/tp4-w3-wc.md` (the fusion audit's figure with its review's haircut): **-4.5 to -5.5 ms a verify, about -9 to -11 ms a two-block round** at 2 to 4 live users, i.e. about 6 to 9 percent of a 125 to 150 ms round.
  Nothing at six or more live seats.
* Together: a gain at the low live counts and no change at eight. A timed run is the only way to know; the agent tests below are the judge.

## 2. What was left out

* The sched-1 production-shape pacing (`_OWED_CAP_MS`, `_SHORT_SHARE`, `_SHORT_WAIT_S`): untouched, a gate's own.
* The fallback twin without WC (`-fx-s5-audit-nowc` exists from the ship-5 pack; there is no `-s6-audit-nowc`): WC has compiled and audited exact on cards, so an audited attach that fails is a fault of the candidate.
* A gate-only timed twin: the candidate is a traffic profile whose waiver decision is PENDING, and such a profile boots in a gate or a smoke (only a build refuses to bake it), so the timed arm is the candidate itself and its
  control is the ship-5 profile.

## 3. The profiles (`scripts/ci/qwen_c2_profiles.json`, generated: never hand-merged)

* `c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-fx-traffic-s6` (`make_fusion_profiles.py` from `scripts/ci/fusion-wp/S6.json`): the candidate. **Waiver `ship6-fx-262k-2026-10-21`, decision PENDING.** The waiver text is the ship-5
  waiver verbatim plus items (Q) WC and (R) the governor; its levers are ship 5's plus `QWEN_FAST_LEVERN_ADAPTIVE`. The owner approves with
  `python3 scripts/ci/make_fusion_profiles.py --approve-ship "APPROVED ..." --ship traffic-s6`; `test_s6_profiles` passes on either decision.
* `c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-fx-s6-audit` (`make_s6_profiles.py`): the audited attach arm, gate-only: the ship-5 audited twin (every audit of the shipped stack, WC with its audit) plus the governor's switch.
* The control of every timed pair is `...-fx-traffic-s5` itself, unchanged.
* Generator order (each `--check` holds whichever ran last): `make_fusion_profiles.py`, `make_octo3_profiles.py`, `make_ship4_profiles.py`, `make_np1_profiles.py`, `make_xe_profiles.py`, `make_pf2_profiles.py`,
  `make_s5_profiles.py`, `make_s6_profiles.py`, then `make_fusion_jobs.py`, `make_s6_jobs.py`.

## 4. The judge: `agent_turns` and `agent_turns8`

Production traffic averages about 59k-token prompts and about 690-token answers; the `agent_turns` smoke (6 sessions at 64k, 128k and 200k, 12 returning turns each) is the main judge, and the ship-5 window's T1B figures are its
baseline (50.36 tok/s aggregate, returning-turn first token p50 1.46 s, p90 2.33 s). It is unchanged: a test pins its plan digest and settings, and `agent_turns_long` likewise.

`agent_turns8` is the new, optional variant: **eight** sessions, every seat busy, weighted like production instead of like `agent_turns`. `agent_turns`' mix (64k, 128k, 200k) averages 131k tokens and its answers 400; production
averages about 59k (p90 208k) and about 690. `agent_turns8` uses eight contexts (24k, 32k, 40k, 48k, 56k, 64k, 80k and 200k, jittered and rotated by the seed: about 550k tokens, mean ~69k) and answers capped at 300 to 1,080
tokens (mean 690); messages, gaps and the 12 turns are the same. Eight sessions of the `agent_turns` mix would be 1.06M tokens against the 1,055,488-token pool and evict each other; this one starts well inside it.
Its budget is 1,500 s (no new turn starts after 25 minutes), so it fits a 30-minute box; the plan estimate is **about 15.4 minutes** and the pack allows 20 with the boot's share. At eight live seats WC and the governor mostly
sit out, so this pair says whether anything is LOST at the production load. Compare it with its own control leg, never with `agent_turns`. The session count was an environment parameter already (`SMOKE_AGENT_SESSIONS`), but the
smoke workflow passes no `SMOKE_AGENT_*`, so a named variant was needed (and the salt-key case of the smoke step names it).

| test | estimated minutes inside a boot | what it judges |
| --- | --- | --- |
| `agent_turns` | 15 | the main judge, both profiles |
| `agent_turns8` | 20 | the production load, every seat busy (optional pair) |
| `prefill_few_decoders` | 12 | the governor's own shape: one decoder and a ~120k cold arrival, then two decoders and a ~254k one |
| `np1_live1`, `np1_live2`, `np1_live4` | 9 together | the live counts WC acts at (2 and 4) and the lone user |
| `concurrent8_steady` | 6 | eight users (the octo block); required by the smoke check on these profiles |

## 5. Window plan (template; the tags are placeholders)

The jobs are templates in `scripts/ci/references/fusion-jobs/S6` (generated by `make_s6_jobs.py`; image placeholder `tp4-s6-1`), listed in `ORDER.txt` with their stop rules, NEEDS lines, read rules and minutes (estimates). Nothing is
pushed from the branch. For each job: copy the template over `.github/c2-serving-job.env` on a throwaway commit of the pushed branch head, replace the image placeholder with the tag B0 built, push **one** tag `<tag-n>`, wait for
the run to finish, never cancel a running card job. The CPU suite must be green on that head first, and the ARC runner sets are at zero from A1 to the last timed job.

| order | template | tag | mode | what | minutes |
| --- | --- | --- | --- | --- | --- |
| 1 | `B0-build-window` | `<tag-1>` | stop | build the window image from the pushed head with the K64j-OQ graft, baking the ship-5 profile as the default (runs while production serves) | 40 (build) |
| 2 | `X0-status-rescan-reset` | `<tag-2>` | stop | card state | 15 |
| 3 | `A1-s6-audited-attach` | `<tag-3>` | stop | `-s6-audit`: warmup, coding, np1_live2, np1_live4, prefill_few_decoders, concurrent8_drain, concurrent8_steady, agent_turns; every audit exact, the governor's fields on every step line | 115 |
| 4 | `T1A-s5-control` | `<tag-4>` | soft | the ship-5 profile: warmup, np1_live1, np1_live2, np1_live4, prefill_few_decoders, concurrent8_steady, agent_turns | 60 |
| 5 | `T1B-s6-candidate` | `<tag-5>` | soft | the candidate, same tests | 60 |
| 6 | `T2A-s5-control-8` | `<tag-6>` | opt | the ship-5 profile: warmup, concurrent8_steady, agent_turns8 | 40 |
| 7 | `T2B-s6-candidate-8` | `<tag-7>` | opt | the candidate, same tests | 40 |
| 8 | `Z-reset` | `<tag-8>` | soft | status and reset | 5 |

The core (X0, A1, T1A, T1B, Z) is 255 minutes of cards; with the optional pair 335. After the owner's approval: `ORDER-SHIP.txt` (B1 the ship build, SR the platform replay), refused by the job reader while the decision is PENDING.

Reads: `c2_smoke_check` clean on every leg (its WC and governor rules read the container log); `np1_report.py` and `octo_compare.py` on T1; `prefill_ladder_report.py <T1A smoke> --second <T1B smoke>` for the governor (the cold
arrival's first-token ratio B over A, the decoders' gaps and progress, equal answer hashes); `agent_turns.py report` on both legs side by side. NO-GO for the candidate: a text difference against the control, an audit mismatch,
a fall-back line beyond the octo block's one, a decode gap over the 8 s floor, a step line without the governor's fields, a loss outside the boot-to-boot drift, an errored or timed-out agent request. A failing lever is
taken out by deleting its line from the ship entry of `scripts/ci/fusion-wp/S6.json` and regenerating (a new build).

## 6. Risks

* The audited profile holds about 2.6 GB more DRAM per chip than the unaudited one (the ship-5 window's A1 timed out three agent-turn reads on admission holds that T1B did not have). `prefill_few_decoders` puts a ~254k prompt
  through the audited twin; a hold line there is the audit's weight, not the candidate's.
* The governor and WC compile nothing new at run time (the governor is host-side scheduling; WC's kernels compiled in the ship-5 window), but the first TIMED WC run is T1B; a first-call JIT cost would show in the first
  rounds of `np1_live2`.
* The governor trades the decoders' speed for the arrival's first token; whether that is wanted at one or two live users is the owner's call, and it is the second switchable line of the ship entry.
* Neither lever has run in a timed run with the other, and the governor has never run on a card at all.
* `agent_turns8` is a new test with no past numbers: its first card run is T2A (the control), so read T2B against T2A only. Eight sessions plus their growth stay inside the pool at the start; later turns may still evict.
* The tau lab's `run.json` now records the `max_tokens` it sent per arm (a dict), where it recorded the unasked option (None) and the drafter pair report's `matched` gate could never pass; the pair report reads the value
  back from `turns.jsonl` for runs made before the change. This is lab and report code (host side, not in the image).
