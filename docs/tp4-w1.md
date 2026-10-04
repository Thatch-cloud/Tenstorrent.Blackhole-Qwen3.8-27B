# tp4/w1: the built levers gated together

Wave 1 of the fusion audit's plan, on the eight-seat 262k stack. Nothing new is built here: tp4/262k8-x (nosamp, D2a and D2c with the
warm-pass fix), tp4/hostgap (stage 1) and tp4/u1 (the unit-major reduce-scatter) are merged, and two gate-only profiles switch the levers
on together. Every flag stays default off.

| Profile | Definition |
|---|---|
| `c2-packed-tp4-8x262k-w1` | `c2-packed-tp4-8x262k-best-time-gate` minus `QWEN_FAST_PACKED_SAMPLER_IN_TRACE`, plus `QWEN_FAST_TP4_DRAFT_CONV`, `QWEN_FAST_TP4_DRAFT_HEADS`, the five host-gap flags of `hostgap-2` (`HOSTGAP_LOG`, `TWO_BLOCK_PRESTAGE`, `WINDOW_VALIDATE`, `PRESTAGE_BLOCK_EPOCHS`, `ENTRY_DIET`) and `QWEN_FAST_TP4_RS_UNIT_MAJOR` and `QWEN_FAST_CCL_TOPOLOGY=ring` (D1) |
| `c2-packed-tp4-8x262k-w1-audit` | the above plus `QWEN_FAST_TP4_DRAFT_CONV_AUDIT`, `QWEN_FAST_TP4_DRAFT_HEADS_AUDIT`, `QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE_AUDIT`, `QWEN_FAST_TP4_RS_UNIT_MAJOR_AUDIT`, `QWEN_FAST_FUSED_COMMIT_AUDIT`, `QWEN_FAST_TP4_VGLUE_AUDIT`, `QWEN_FAST_DRAFT_SINGLES_AUDIT=all` |
| `c2-packed-tp4-8x262k-w1-lite` | `-w1` minus `QWEN_FAST_TP4_PRESTAGE_BLOCK_EPOCHS` (the host-gap fallback, pack job T7) |
| `c2-packed-tp4-8x262k-w1-nod1`, `c2-packed-tp4-8x262k-w1-audit-nod1` | `-w1` and `-w1-audit` minus D1 (to separate D1 from the other levers if the audited attach fails in a drafter gather) |

## D1 is in, S1 is out

- S1 (`QWEN_FAST_TP4_SHARD_ARGMAX`): no measured gain, out.
- D1 (`QWEN_FAST_CCL_TOPOLOGY=ring`): in. Only the drafter-side collectives read the flag (`dflash_device`, `quad_draft`, `feature_collective_tp`
  through `mesh_link_policy.fast_ccl_topology`: the feature projection and the drafter gathers). Nothing couples it to the engine block's
  `fabric_config`: the pairing with `FABRIC_1D_RING` exists only in the older ring arms, which are other profiles. Here the fabric stays
  `FABRIC_1D` as in the control, and the verify's own collectives already run `Topology.Ring` on it, so the flag does not move them. What is
  new is the first serving of Ring at the drafter's shapes on `FABRIC_1D` (the CX1 shape probe, Linear against Ring, has not run): the
  audited attach's draft-singles and D2 audits judge it, and the `-nod1` profiles separate it from the other levers if it fails.

## Interactions found in code

1. **nosamp and the audits.** The in-trace sampler is already disengaged under the verify T1 audit (`shard_audit`), so an arm with that
   audit on never exercises nosamp. `w1-audit` keeps the verify T1 and T2 audits at 0 (as the host-gap audited arms do), so it carries
   the timed arm's sampler state (none in the trace), but not its whole trace: U1's audited calls (3 reduce-scatters and 2 clones of each
   audited call, served from the split), the verify-glue clones and the D2 clones change it. A1 must show no `packed sampler arm` line. The request-width warm (`QWEN_FAST_M3_REQUEST_WARM=1`) is the hang fix that stays.
2. **Audit clones and capture order.** The U1 audit's clones are allocated inside each block's verify capture and can sit in holes the
   first block's capture freed; its audit is read only right after its own block's replay (`audit_replayed`, `audit_round`). The D2
   audits' clones are allocated inside the drafter bucket's capture; with two quads the compare for quad A runs at collect
   (`compare_draft_audit`), after quad B's replay, so it is safe only if A's bucket was captured first, and unlike U1 it has no
   replayed-owner guard: a capture-order artifact would read as a D2 mismatch. U1 reads only right after its own block's replay; the host-gap
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

## Host-gap overrun and the audit coverage

- `docs/tp4-hostgap.md` puts the quads' device time at 38.8 ms with about 25 ms of host slack once D2a lands, and W1 is the first arm that pairs
  D2a with both blocks pre-staging. P0 therefore needs window 14a to have selected the epochs arm, B's `[PACKED-HOSTGAP-WINDOW]` lines are read
  for an overrun (`fence_wait_ms` about 0 while `window_ms` exceeds the quad time), T7 runs the lite arm, and the pack states a kill line.
- The U1 audit covers the first 32 of the 128 calls of a forward; the other 96 rest on the answer hashes. The answer hashes cannot see a
  drafter-side (D2) error under lossless greedy verification, so the accepted tokens and verify rounds are recorded against S0c.
- Pre-existing in tp4/hostgap, not changed here: with the image's `ROUND_FENCES=1`, block B's only binding validation in a round is the
  window's, which runs before A's verify and commits, while `validated_this_round = self.round_fences` still tells B's commits the replay
  validated. A1's shadow covers only replay time; nothing W1 adds moves a native buffer in that span.

## Job pack

`scripts/ci/references/tp4-w1-jobs` (image `tp4-w1-1`): X0 status rescan reset, B0 build, S0c control attach smoke, U1a U1 alone audited attach
(soft), A1 audited attach on `-w1-audit`, H1-H5 hang shapes on `-w1`, H6 cold 262k arrival (soft), T1-T6 ABABAB timed control against `-w1` at 32k
and 128k, T7 the lite arm, P1 eight-user device profile of `-w1`, P1c the same on the control in this image, Z.
The projected 32k round is about 182 ms against 222 ms today; the delivered gain of each lever is P1 against P1c, not the ABAB pair.
