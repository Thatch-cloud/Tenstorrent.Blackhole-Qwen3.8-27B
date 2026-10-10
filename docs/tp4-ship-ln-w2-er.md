# TP4 ship candidate: Lever N + W2 + engine reuse at eight seats x 262k

**Branch:** `tp4/ship-ln-w2-er` (from `tp4/er-drafters`). **Profile:** `c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-traffic`. **Pack:**
`scripts/ci/references/tp4-ship-ln-w2-er-jobs`. **Test:** `scripts/ci/test_ship_ln_w2_er.py`.

## 1. The profile

`c2-packed-tp4-8x262k-ship-prefix-levern-traffic` (Lever N with prefix reuse, the 8 s decode floor, eight seats, 262,144 tokens) plus exactly:

| Lever | Keys | Card evidence |
|---|---|---|
| W2 (docs/tp4-w2.md) | `QWEN_FAST_TP4_SDPA=multi`, `QWEN_FAST_TP4_CONV_GATES_SPREAD=1` | exactness v568 (3,433 SDPA audits) and v595 (3,478 F1 audits); timed v596-v602; hang shapes v603; paired skew v592/v542 |
| Engine reuse (docs/tp4-engine-reuse.md) | `QWEN_FAST_PARKED_ENGINES=1`, `QWEN_FAST_PARKED_DRAFTS=1`, `QWEN_FAST_LEVERN_BUILD_MS=learned` | exactness v545/v546 and v609; hang shapes v610; paired turns v611-v614 |

It is the env of the gate arms `-levern-w2` and `-levern-parked` together, with no gate instrument, no 262k evidence waiver and no gate marker. The KV pool
(19,968 blocks) and the trace region (512 MiB) are the ones the engine-reuse memory margins were derived under. The two lever sets have never run
together in one process on a card: the pack's G1 is that run.

## 2. The owner traffic waiver

`serving_c2_contract` refuses the W2 levers and engine reuse on a traffic profile because neither has a pinned qualification record: the multi launch
(G16, flags 0x21) is outside `packed_any_evidence_tp4_262144.json`, F1 has no record, and engine reuse has not run its memory ballast ladder, kill-switch
drill, fault or negative-control jobs. Recording that evidence is card work that does not exist yet. The only other way through is the owner's decision,
which the profile carries as a field, `owner_traffic_waiver`:

| Field | Meaning |
|---|---|
| `id` | a name for the decision (`w2-er-262k-2026-10-10`) |
| `levers` | `{flag: value}` of every waived lever; only `QWEN_FAST_TP4_SDPA=multi`, `QWEN_FAST_TP4_CONV_GATES_SPREAD=1`, `QWEN_FAST_PARKED_ENGINES=1` and `QWEN_FAST_PARKED_DRAFTS=1` are waivable |
| `waives` | in words, what is waived |
| `evidence` | the card runs the decision rests on |
| `decision` | a sentence starting `PENDING` (a request) or `APPROVED` (the owner's decision) |

Rules (`traffic_waiver_problems`, `waived_levers`, `multi_problems`, `parked_problems`):

- a lever set in the env and not named stays refused; a lever named and not set (a stale waiver) is refused; a wrong value, an unknown field, a missing
  field, an empty evidence list or a decision that starts with neither word makes the waiver malformed, and a malformed waiver waives nothing;
- a gate instrument (the SDPA and F1 audits, the parked audit, faults, negative controls, ballast and kill-switch trigger) is never waivable, and the
  process-environment refusals of the gate instruments still apply;
- a gate-only profile carries no waiver;
- the boot logs `[QWEN-C2] profile <name>: OWNER TRAFFIC WAIVER <id>: <flag=value ...> serve traffic with no pinned qualification record (...)
  [decision: ...]`. The multi attach still logs its own `[PINDIAG] tp4 sdpa multi UNQUALIFIED capacity=262144` line, which stays true;
- **no build bakes a profile whose waiver decision is not `APPROVED`**: `c2_serving_job.read_bake` refuses the job and `build-c2-serving-image.sh`
  refuses the build. So the order is: the owner approves, the approval is committed (the decision sentence starts with `APPROVED`), the CPU suite runs
  on that commit, and B0 is tagged on it. The image then records the decision it was built under.

## 3. Kill switches

Lever N (`levern.off`), prefix reuse (`prefix-reuse.off`) and engine reuse (`parked.off`) each stop with a file on the hub mount and no restart. W2 has
a switch too (`w2.off`, `docs/tp4-w2-kill-switch.md`), of a different kind: the multi programs and the F1 spread are captured in the packed blocks' traces, so
the file does not edit a trace. Within about one round it sends every packed round to the exact sequential step (slower, byte-identical greedy output), and
a restart with the file still present attaches the packed blocks without W2, at full packed speed. An image rollback is no longer the only way off W2.

## 4. The pack

B0 builds `tp4-ship-1` with the profile baked as the image default (refused until approved). G1 gates it unaudited on the cards (an audited
eight-seat 262k profile costs about 8.4 minutes per request): warmup, coding, eight-seat steady and skew, the five Lever N hang shapes, the drain,
`parked_abort_reuse` and `parked_turns`, with the CI paused. G2 runs the prefix gate's `levern-hit` plan (a salted hit beside decoders). SR replays the
platform's serving sequence on the production layer built on that image, booting the profile by the image default. Every job is a stop job; the times
are in `ORDER.txt`.
