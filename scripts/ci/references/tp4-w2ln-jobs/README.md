# tp4-w2ln-jobs: the combined development window's job pack

One window, one image (`tp4-w2ln-1`), built from the pushed `tp4/w2-levern-prefix` commit. It gates together: W2 (the multi-user SDPA launch and the conv-gates spread, "F1"), Lever N merged
with prefix reuse, the drafter hooks (the BF16 arm and the checkpoint selector), and a CONTROL arm of the production profile that runs the prefix exactness and lifecycle gate (P1) on the
production bytes. The design, the interaction analysis and the review are in `docs/tp4-combined-window.md`; the order, classes, minutes, NEEDS graph, swap rules, read rules and the cutover
rules are in `ORDER.txt`. Every file here is a template: nothing here pushes a tag, names a rig, a card, a host or a registry, or touches the node agent except A0 (stop) and Z (start).

## How to run a job

Copy the template over `.github/c2-serving-job.env` on a throwaway commit of `tp4/w2-levern-prefix` and push a tag `experiment/c2-serving-vN` from the hardware-allowlisted set (the k-th job of
`ORDER.txt` takes the k-th tag of its TAGLIST; the tags left are the swap re-run reserve). The CONTROL jobs (`P1a-CTL`, `P1b-CTL`, `L8-CTL`, `C16-CTL`, `E1-CTL`, `T0` and `G0`) name the production base image
`tp4-serve-10` and are driven from the SAME window commit: the engine bytes are production's, the host scripts (the smoke shapes including the skew, the gates with the raised exactness-shared box, the
kill-switch preflight, the load logger) are the window's. Read each job against its comment block and `ORDER.txt`; judge timing with `scripts/ci/w2ln_timing_compare.py judge` (one verdict per length: steady, 32k,
128k, skew), and calibrate its load maximum with `load-limit` on S0-CTL's `load.log`.

## The profiles

Every profile is gate only, derives from the production family (`c2-packed-tp4-8x262k-ship-prefix[-audit]`, `...-levern[-audit]`) and carries no waiver marker, no gate-profile marker and no
host-gap log. `test_tp4_w2ln_profiles` holds each row as "its parent plus exactly these keys". Audit twins keep the production audit set (the T1/T2 shard audits on) and add the SDPA and F1
audits; the committed fallbacks answer the fit questions the window cannot answer on a desk: `-lean` (drop the T1/T2 audits: L1 or trace region), `-pool` (19,200 blocks: DRAM), `-epochglobal` (the global
epoch scope) and the four slices `-sdpa`, `-f1`, `-ln` and `-w1` (one family of audits each, SMOKE jobs only; together exactly the whole audit set). Every profile a candidate job names has a committed target under
every swap (F1, LEAN, POOL, EPOCH) and every allowed pair of swaps (F1 with LEAN, POOL or EPOCH; EPOCH with LEAN or POOL), the audited `-nolna` twins the long prefix and gate jobs serve included:
`test_tp4_w2ln_window` derives each target and requires it, `test_tp4_w2ln_profiles` holds each as its parent plus its keys.

| Profile | Parent | Delta |
|---|---|---|
| `c2-packed-tp4-8x262k-ship-prefix-w2` | `c2-packed-tp4-8x262k-ship-prefix` | TP4_SDPA=multi, TP4_CONV_GATES_SPREAD=1 |
| `c2-packed-tp4-8x262k-ship-prefix-w2-audit` | `c2-packed-tp4-8x262k-ship-prefix-audit` | TP4_SDPA=multi, TP4_SDPA_AUDIT=1, TP4_CONV_GATES_SPREAD=1, TP4_CONV_GATES_SPREAD_AUDIT=1 |
| `c2-packed-tp4-8x262k-ship-prefix-w2-nof1` | `c2-packed-tp4-8x262k-ship-prefix-w2` | minus TP4_CONV_GATES_SPREAD |
| `c2-packed-tp4-8x262k-ship-prefix-w2-nof1-audit` | `c2-packed-tp4-8x262k-ship-prefix-w2-audit` | minus TP4_CONV_GATES_SPREAD, minus TP4_CONV_GATES_SPREAD_AUDIT |
| `c2-packed-tp4-8x262k-ship-prefix-levern-w2` | `c2-packed-tp4-8x262k-ship-prefix-levern` | TP4_SDPA=multi, TP4_CONV_GATES_SPREAD=1 |
| `c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit` | `c2-packed-tp4-8x262k-ship-prefix-levern-audit` | TP4_SDPA=multi, TP4_SDPA_AUDIT=1, TP4_CONV_GATES_SPREAD=1, TP4_CONV_GATES_SPREAD_AUDIT=1 |
| `c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1` | `c2-packed-tp4-8x262k-ship-prefix-levern-w2` | minus TP4_CONV_GATES_SPREAD |
| `c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit` | `c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit` | minus TP4_CONV_GATES_SPREAD, minus TP4_CONV_GATES_SPREAD_AUDIT |
| `c2-packed-tp4-8x262k-ship-prefix-w2-audit-lean` | `c2-packed-tp4-8x262k-ship-prefix-w2-audit` | VERIFY_T1_AUDIT=0, VERIFY_T2_AUDIT=0 |
| `c2-packed-tp4-8x262k-ship-prefix-w2-nof1-audit-lean` | `c2-packed-tp4-8x262k-ship-prefix-w2-nof1-audit` | VERIFY_T1_AUDIT=0, VERIFY_T2_AUDIT=0 |
| `c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-lean` | `c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit` | VERIFY_T1_AUDIT=0, VERIFY_T2_AUDIT=0 |
| `c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-lean` | `c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit` | VERIFY_T1_AUDIT=0, VERIFY_T2_AUDIT=0 |
| `c2-packed-tp4-8x262k-ship-prefix-w2-audit-pool` | `c2-packed-tp4-8x262k-ship-prefix-w2-audit` | QWEN36_MAX_TOKENS_ALL_USERS=1228288, engine num-gpu-blocks-override=19200 |
| `c2-packed-tp4-8x262k-ship-prefix-w2-nof1-audit-pool` | `c2-packed-tp4-8x262k-ship-prefix-w2-nof1-audit` | QWEN36_MAX_TOKENS_ALL_USERS=1228288, engine num-gpu-blocks-override=19200 |
| `c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-pool` | `c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit` | QWEN36_MAX_TOKENS_ALL_USERS=1228288, engine num-gpu-blocks-override=19200 |
| `c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-pool` | `c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit` | QWEN36_MAX_TOKENS_ALL_USERS=1228288, engine num-gpu-blocks-override=19200 |
| `c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna` | `c2-packed-tp4-8x262k-ship-prefix-levern-audit` | minus LEVERN_AUDIT |
| `c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna` | `c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit` | minus LEVERN_AUDIT |
| `c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-nolna` | `c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit` | minus LEVERN_AUDIT |
| `c2-packed-tp4-8x262k-ship-prefix-levern-epochglobal` | `c2-packed-tp4-8x262k-ship-prefix-levern` | LEVERN_EPOCH_SCOPE=global |
| `c2-packed-tp4-8x262k-ship-prefix-levern-audit-epochglobal` | `c2-packed-tp4-8x262k-ship-prefix-levern-audit` | LEVERN_EPOCH_SCOPE=global |
| `c2-packed-tp4-8x262k-ship-prefix-levern-w2-epochglobal` | `c2-packed-tp4-8x262k-ship-prefix-levern-w2` | LEVERN_EPOCH_SCOPE=global |
| `c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-epochglobal` | `c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit` | LEVERN_EPOCH_SCOPE=global |
| `c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-sdpa` | `c2-packed-tp4-8x262k-ship-prefix-levern-w2` | TP4_SDPA_AUDIT=1 |
| `c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-f1` | `c2-packed-tp4-8x262k-ship-prefix-levern-w2` | TP4_CONV_GATES_SPREAD_AUDIT=1, TP4_VGLUE_AUDIT=1 |
| `c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-ln` | `c2-packed-tp4-8x262k-ship-prefix-levern-w2` | LEVERN_AUDIT=1, QWEN_PREFIX_DIGESTS=1, TP4_TWO_BLOCK_PRESTAGE_AUDIT=1 |
| `c2-packed-tp4-8x262k-ship-prefix-pool` | `c2-packed-tp4-8x262k-ship-prefix` | QWEN36_MAX_TOKENS_ALL_USERS=1228288, engine num-gpu-blocks-override=19200 |
| `c2-packed-tp4-8x262k-ship-prefix-dbf16` | `c2-packed-tp4-8x262k-ship-prefix-pool` | DRAFTER_BF16=1 |
| `c2-packed-tp4-8x262k-ship-prefix-audit-digests` | `c2-packed-tp4-8x262k-ship-prefix-audit` | QWEN_PREFIX_DIGESTS=1 |
| `c2-packed-tp4-8x262k-ship-prefix-dckdefault` | `c2-packed-tp4-8x262k-ship-prefix` | DRAFTER_CHECKPOINT=dflash2-dedf8df6 |
| `c2-packed-tp4-8x262k-ship-prefix-levern-audit-lean` | `c2-packed-tp4-8x262k-ship-prefix-levern-audit` | VERIFY_T1_AUDIT=0, VERIFY_T2_AUDIT=0 |
| `c2-packed-tp4-8x262k-ship-prefix-levern-audit-pool` | `c2-packed-tp4-8x262k-ship-prefix-levern-audit` | QWEN36_MAX_TOKENS_ALL_USERS=1228288, engine num-gpu-blocks-override=19200 |
| `c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-w1` | `c2-packed-tp4-8x262k-ship-prefix-levern-w2` | VERIFY_T1_AUDIT=1, VERIFY_T2_AUDIT=1, TP4_DRAFT_CONV_AUDIT=1, TP4_DRAFT_HEADS_AUDIT=1, TP4_RS_UNIT_MAJOR_AUDIT=1, FUSED_COMMIT_AUDIT=1, DRAFT_SINGLES_AUDIT=all |
| `c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna-lean` | `c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna` | VERIFY_T1_AUDIT=0, VERIFY_T2_AUDIT=0 |
| `c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna-pool` | `c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna` | QWEN36_MAX_TOKENS_ALL_USERS=1228288, engine num-gpu-blocks-override=19200 |
| `c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna-lean` | `c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna` | VERIFY_T1_AUDIT=0, VERIFY_T2_AUDIT=0 |
| `c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna-pool` | `c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna` | QWEN36_MAX_TOKENS_ALL_USERS=1228288, engine num-gpu-blocks-override=19200 |
| `c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-epochglobal` | `c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1` | LEVERN_EPOCH_SCOPE=global |
| `c2-packed-tp4-8x262k-ship-prefix-levern-audit-lean-epochglobal` | `c2-packed-tp4-8x262k-ship-prefix-levern-audit-lean` | LEVERN_EPOCH_SCOPE=global |
| `c2-packed-tp4-8x262k-ship-prefix-levern-audit-pool-epochglobal` | `c2-packed-tp4-8x262k-ship-prefix-levern-audit-pool` | LEVERN_EPOCH_SCOPE=global |
| `c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna-epochglobal` | `c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna` | LEVERN_EPOCH_SCOPE=global |
| `c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-nolna-lean` | `c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-nolna` | VERIFY_T1_AUDIT=0, VERIFY_T2_AUDIT=0 |
| `c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-nolna-pool` | `c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-nolna` | QWEN36_MAX_TOKENS_ALL_USERS=1228288, engine num-gpu-blocks-override=19200 |
| `c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-lean-epochglobal` | `c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-lean` | LEVERN_EPOCH_SCOPE=global |
| `c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-pool-epochglobal` | `c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-pool` | LEVERN_EPOCH_SCOPE=global |
| `c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-epochglobal` | `c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit` | LEVERN_EPOCH_SCOPE=global |
| `c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna-epochglobal` | `c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna` | LEVERN_EPOCH_SCOPE=global |
| `c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna-lean-epochglobal` | `c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna-lean` | LEVERN_EPOCH_SCOPE=global |
| `c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna-pool-epochglobal` | `c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna-pool` | LEVERN_EPOCH_SCOPE=global |
| `c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna-lean-epochglobal` | `c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna-lean` | LEVERN_EPOCH_SCOPE=global |
| `c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna-pool-epochglobal` | `c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna-pool` | LEVERN_EPOCH_SCOPE=global |
| `c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-nolna-epochglobal` | `c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-nolna` | LEVERN_EPOCH_SCOPE=global |

Short names used in `ORDER.txt`: CONTROL = `ship-prefix`, CONTROL-A = `ship-prefix-audit`, LN = `ship-prefix-levern`, W2 = `ship-prefix-w2`, C (COMBINED) = `ship-prefix-levern-w2`, and an `-audit` suffix
for the audited twin.

## Drafter candidates

No candidate checkpoint id exists yet. When one is verified, its pins (revision, config and manifest hashes, the weights hash from `python scripts/ci/drafter_checkpoint.py --digest <fixture dir>`: one digest over every file
of the fixture, so a changed weight file, a file added or one removed is refused at the boot, geometry) are committed to `scripts/ci/drafter_checkpoints.json`, its bytes are staged by naming the id in the B0 job file
(`C2_DRAFTER_CANDIDATES=<id>`: the job parser accepts only pinned candidate ids and exports the list to the Build step), a profile `ship-prefix-drafter-<id>` (the production profile plus `QWEN_FAST_DRAFTER_CHECKPOINT=<id>` and the draft paths of its baked copy) is added, and a `Dk-<id>`
pair is made from `D1` and `D2` with that profile as B, against the pool-matched control for a bf16 candidate and against `ship-prefix` for a bf8 one. `DK0` runs first: it proves the selector's plumbing
changes no byte (the default checkpoint logs `verified=default`: it carries no pins, so the line proves the selector path ran, not that bytes were hashed; a candidate logs `verified=1` only after its config, manifests and weights hashed to the pins). A candidate that is ready after B0 cannot join this window (its bytes are baked).
