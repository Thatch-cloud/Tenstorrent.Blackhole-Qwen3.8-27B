# tp4/w1: the built levers gated together

Wave 1 of the fusion audit's plan, on the eight-seat 262k stack. Nothing new is built here: tp4/262k8-x (nosamp, D2a and D2c with the
warm-pass fix), tp4/hostgap (stage 1) and tp4/u1 (the unit-major reduce-scatter) are merged, and two gate-only profiles switch the levers
on together. Every flag stays default off.

| Profile | Definition |
|---|---|
| `c2-packed-tp4-8x262k-w1` | `c2-packed-tp4-8x262k-best-time-gate` minus `QWEN_FAST_PACKED_SAMPLER_IN_TRACE`, plus `QWEN_FAST_TP4_DRAFT_CONV`, `QWEN_FAST_TP4_DRAFT_HEADS`, the five host-gap flags of `hostgap-2` (`HOSTGAP_LOG`, `TWO_BLOCK_PRESTAGE`, `WINDOW_VALIDATE`, `PRESTAGE_BLOCK_EPOCHS`, `ENTRY_DIET`) and `QWEN_FAST_TP4_RS_UNIT_MAJOR` |
| `c2-packed-tp4-8x262k-w1-audit` | the above plus `QWEN_FAST_TP4_DRAFT_CONV_AUDIT`, `QWEN_FAST_TP4_DRAFT_HEADS_AUDIT`, `QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE_AUDIT`, `QWEN_FAST_TP4_RS_UNIT_MAJOR_AUDIT`, `QWEN_FAST_FUSED_COMMIT_AUDIT`, `QWEN_FAST_TP4_VGLUE_AUDIT`, `QWEN_FAST_DRAFT_SINGLES_AUDIT=all` |

## Left out

- S1 (`QWEN_FAST_TP4_SHARD_ARGMAX`): no measured gain.
- D1 (`QWEN_FAST_CCL_TOPOLOGY=ring` for the drafter's gathers): the flag exists and only the drafter's modules read it, but the ring
  arms also set the engine block's `fabric_config` to `FABRIC_1D_RING`. The fabric is a property of the whole process, so that arm also
  changes the routing of the verify's collectives. It keeps its own arm and its own window.

## Interactions found in code

1. **nosamp and the audits.** The in-trace sampler is already disengaged under the verify T1 audit (`shard_audit`), so an arm with that
   audit on never exercises nosamp. `w1-audit` keeps the verify T1 and T2 audits at 0 (as the host-gap audited arms do), so it carries
   the trace the timed arm carries. The request-width warm (`QWEN_FAST_M3_REQUEST_WARM=1`) is the hang fix that stays.
2. **Audit clones and capture order.** The U1 audit's clones are allocated inside each block's verify capture and can sit in holes the
   first block's capture freed; its audit is read only right after its own block's replay (`audit_replayed`, `audit_round`). The D2
   audits' clones are allocated inside the drafter bucket's capture and compared right after that bucket's replay
   (`compare_draft_audit`), before any verify or commit replay. Neither reads after another trace could overwrite it; the host-gap
   pre-stage writes only fixture inputs and reorders no replay, so it does not break either rule. A `read after another block
   replayed` error in the audited attach is a capture-order finding, not a lever mismatch.
3. **Host-gap pre-stage and U1 on the packed verify.** They touch different places: the pre-stage and the window validate change what is
   written to the fixture before the replay (and one binding check skipped after a usable diff snapshot), U1 changes only the
   all-reduces inside the trace, and only for 64-row blocks (any other height falls back by name). `packed_verifier.py` merged without a
   textual conflict and the two edits are in different regions of `step`.
4. **The D2 warm-pass fix and the audits.** With the D2 audits on, the warm pass runs the served op and one DRAM clone per output
   before the drafter capture (v403: `Cannot load new binaries during trace capture`). The U1 audit compares the block's warm forward as its
   round 0 (`audit_claim`, `audit_round`), so its programs are compiled before the capture too.
5. **Smoke rules.** Each lever's rule reads its own flags from the profile's env and logs its own lines; the three rule sets do not
   share a marker. The union is checked in `test_tp4_w1` against one composed log.
6. **Image.** No new served module: every file the levers touch is already in both copy lists and the overlay.

## Job pack

`scripts/ci/references/tp4-w1-jobs` (image `tp4-w1-1`): X0 status rescan reset, B0 build, S0c control attach smoke, A1 audited attach
on `-w1-audit`, H1-H5 hang shapes on `-w1`, T1-T4 ABAB timed control against `-w1` at 32k and 128k, P1 eight-user device profile, Z.
The projected 32k round is about 182 ms against 222 ms today; the delivered gain of each lever is read from P1, not from the ABAB pair.
