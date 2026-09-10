# DSpark: capture drafting, then remove redundant target-state writes

**Implementation ready for qualification; no new hardware throughput result.**
The repaired control remains **PP 3,350.93 / CTX 4,096 / TG 65.16**, one stream.
The repeat-confirmed DFlash2 lead remains 74.27 TG at the same context.

## What this experiment changes

| Arm | Proposal execution | Target GDN state writes |
| --- | --- | --- |
| A: eager | Existing full-history proposer | Existing verifier and accepted-prefix publication |
| B: trace | One captured proposal; persistent inputs | Same as A |
| C: trace + commit-only | Same captured proposal as B | Publish only the accepted prefix |

The control spends 86.49 ms/block drafting, 83.44 ms verifying and 14.07 ms
publishing. Proposal capture targets host dispatch overhead. Commit-only GDN
removes redundant state copies, not recurrent math, precision or rollback.
It is already used by the faster DFlash2 path, but this T16 combination needs
its own width-specific check.

## Keep the repaired cache safe

- Allocate proposal inputs and scratch ownership before target trace capture.
- Copy the current committed K/V bank into stable proposal input buffers.
- Store queries after the fixed 4,384-row capacity; mask the unused history gap.
- Use the actual logical position for RoPE, never the physical capacity.
- Keep every captured intermediate alive; release publication temporaries before replay.
- Read all 15 proposed token IDs together, once per chip.

The instrumented request compares complete normalized outputs, logits and token
IDs against fixed-layout eager execution on both chips at every proposal.
Existing full-cache, committed-feature, native-token/state and inactive-slot
checks remain required. Numerical tolerances are not widened.

## Qualification status

| Gate | Current status |
| --- | --- |
| Host tests | 1,317 pass; Python 3.10 and shell syntax pass; 61 simulator-harness tests pass |
| Fixed-capacity attention | TTsim `20260910T024802Z-399` passes all 82 checks; independent audit passes |
| T16 commit-only GDN | TTsim `20260910T031906Z-400` running: every prefix 0–16 and a real T2 continuation |
| Matched full requests | Blocked until both new simulator reports pass independent audit |
| Held-out coding / serving | Not qualified; defaults unchanged |

The fixed-attention probe checks real native append/pad/attention operations,
changing frontiers 4,096/4,109, both simulated chips, exact replay and poisoned
unused storage. It is a correctness test, not a hardware timing estimate.

The independent audit regenerates full FP32 reference and physical-layout hashes,
checks every matrix coordinate, and reconciles 38 sources against revision
`62082ec` plus 1,517 unchanged native fingerprints. Peak memory is 1.79 GB, no
swap or OOM, clean exit 0. The 0.01 relative/absolute tolerance is unchanged.
Report SHA256: `5fbabcecf3eb6988bb451ca0387a58861e0a0710bac47e998c1e885cff29c63a`.

The old bank probe recorded `full_dspark_request.py` as metadata but did not
import or execute it. Preflight now states that boundary explicitly: the bank
implementation must remain hash-identical, while changed request wiring is
recorded and must pass hardware integration. No executed simulator helper is
exempted from its source check.

## One loaded hardware session

Opt-in CI suite: `dspark-request-variants`. Load the target and drafter once.
Run one audited complete request per arm, then timed **A / B / C / C / B / A**.
Each arm must preserve native output and state through EOS. Audit requests do
not enter TG; every timed stall does. Fresh prefill and setup remain reported.

Report PP / actual CTX / committed TG per arm, draft acceptance, complete cycle
cost and setup-inclusive latency. Compare B against A, C against B and C
against A; do not attribute a combined improvement to either change alone.

At the measured 12.1 committed tokens/block, 200 TG needs a **60.5-ms total
cycle**. Capturing drafting alone cannot meet that budget. These experiments
measure how much remains before testing wider useful proposal windows.
