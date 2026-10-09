# tp4/w3 MR and 2a: the verify readback as mesh-level reads, and the quad's top-16 outputs as one joined tensor

Status: **built behind two default-off flags, CPU-tested, never run on a card.** Both move bytes and compute nothing, so exactness is "the same bytes, read differently"; the audits below hold it on a card.
ESTIMATED saving (the fusion audit's, not measured): MR plus 2a together -3 to -5.7 ms a round at eight live seats, -4.5 to -7 ms at 128k (item 15 of the fusion plan, `MR + 2a`).

## Flags

| Flag | Lever | Effect |
|---|---|---|
| `QWEN_FAST_TP4_VERIFY_MESH_READ=1` | MR | `PackedVerifierEngine.shard_predictions` reads the sampler's ids and values with ONE mesh-level read each, split back per chip. Also used for the quad readback below. |
| `QWEN_FAST_TP4_VERIFY_MESH_READ_AUDIT=1` | MR audit | Needs the lever. Every verify also runs the per-chip loop and compares both, bit for bit. |
| `QWEN_FAST_TP4_QUAD_OUTPUTS_JOINED=1` | 2a | The quad's pooled output set gains one joined values tensor and one joined indices tensor; the traced pass ends by joining its chunks into them; the readback reads those. |
| `QWEN_FAST_TP4_QUAD_OUTPUTS_JOINED_AUDIT=1` | 2a audit | Needs the lever. Every quad readback also reads the chunk tensors the old way (and all four chips' features) and compares. |

Each is strict 0/1 (anything else raises), TP4 only (raises at the pair), and an audit without its lever raises. With every flag unset nothing is imported by the readback paths, the pool spec is today's, the pool report is today's, and no log line changes (`test_tp4_w3_mr`, "off" tests).

## MR

`tp4_mesh_read.chip_parts` reads `ttnn.to_torch(tensor, mesh_composer=ConcatMeshToTensor(mesh, dim=0))` and returns rows `[c*per, (c+1)*per)` for chip `c`. Every chip's part of these tensors has the same shape, so that is exactly the host tensor `to_torch(part)` returns for chip `c`; the downstream `combine_shards`, the sampler audits and the `reads_ms` measurement are untouched. Eight blocking reads a verify (two blocks: sixteen a round) become two.

The `word` arm of the MR probe (one joined `(column << 16) | bits` uint32 tensor) is **not built**: S1's kernels write separate ids and values pages, so a joined word needs a change to the traced pass, and whether it beats two mesh reads is the MR4 job's verdict (`tp4_mr_probe`, verdicts GO / MESH-ONLY / NO-GO), not this branch's. If MR4 reads NO-GO, leave `QWEN_FAST_TP4_VERIFY_MESH_READ` unset.

Markers (new text only with the flag on): `[PINDIAG] tp4 verify mesh read engaged chips=4` once per captured verify trace (one a block); `mr=1` appended to `[PACKED-HOSTGAP-VERIFY]`; `[PINDIAG] tp4 verify mesh read fell back reason=...` when the verify does not read back through the shard argmax (then the flag does nothing and says so); under the audit `[PACKED-MESH-READ-AUDIT] round=N chips=4 equal=1`, or `equal=0 differing=ids.2,...` and an `AssertionError` (a round never uses unequal data).

## 2a

At TP4 the quad's head leaves per chip two 32,768-column chunks of the 62,080 shard, each a top-16 values (bf16 TILE) and indices (uint16 TILE) tensor of `(1, 1, 64, 16)`: today's readback is `2 chunks x 4 chips x 2 = 16` reads plus the replicated selector features from each of the four chips, **20 reads a quad, 40 a round** at eight seats (the audit's "18 to 6" counts the pair's two chips and four chunks).

With the flag the pool (`serving_buffer_pool`, `draft_outputs=`) gives the quad groups (`(0, 1, 2, 3)` and the quad blocks' groups, and no other group) an optional `joined` shape `(1, 1, 64, 32)`: one values and one indices buffer, allocated with the pool before any trace, right after the group's chunk and projection buffers. `dflash_proposal_trace.publish_outputs` (recorded in the trace) ends the pass by `concat`-ing the chunks along the last dimension and `copy`-ing the result into them. The reader takes each chunk's columns `[16c, 16c + 16)` of every chip's joined part, in the loop's own chunk-major, chip-minor order, into the same `merge_chunk_candidates`.

| Route | Reads a quad |
|---|---|
| today | 20 |
| joined | 9 (8 + one chip's features) |
| joined + `QWEN_FAST_TP4_VERIFY_MESH_READ` | 3 (two mesh reads + one chip's features) |
| `QWEN_FAST_TP4_VERIFY_MESH_READ` alone | 5 (four chunk tensors + the features, each one mesh read) |

**The features guard.** The selector features are replicated by construction; today's readback reads all four chips and compares. Under the joined lever without its audit chip 0's part alone is read (the design's 2a). The audit reads all four and the unchanged guard raises on a difference. This is a deliberate weakening of a defensive check outside the audit; say so before shipping it.

**The join is tried once, eagerly.** `tp4_mesh_read.settle` runs the concat and the copies over the warm-up's outputs (which compiles them before the capture, and is where a refusal shows). A uint16 TILE concat at a 16-wide last dimension is the unverified op: on any refusal this build loses the joined set only, logs `[PINDIAG] tp4 quad outputs joined fell back slots=[...] reason=...`, and the quad reads its chunks as before (the S2 pooling of the chunks is untouched). The same line names a pool without draft output sets (the sets exist only under `QWEN_FAST_EXTENT_REPLAY=1`). A success logs `[PINDIAG] tp4 quad outputs joined engaged quads=N slots=[0,1,2,3] chunks=2 width=32`.

**Capture accounting.** The joined buffers are 2 x 4 KiB a chip, pooled; the concat's two temporaries (4 KiB each a chip) are allocated inside the capture and kept alive on the recorded outputs namespace for the trace's life (never freed inside the capture). About 16 KiB a chip a quad against `QUAD_CAPTURE_BYTES_EST` (450 MiB): the estimate is left as is; the `draft_output_bytes` of the pool report grows by the joined payload only for a pool that holds one. The capture's allocation pattern does change (two more intermediates), so the hang shapes are the gate (they are in the pack).

Markers: the two above, and under the audit `[QUAD-JOINED-AUDIT] round=N chips=4 chunks=2 equal=1`, or `equal=0 differing=chunk=A-B.chip=C,...` and an `AssertionError`.

## Exactness

Nothing is computed. MR reads the same device buffers through one call instead of eight; 2a copies the same elements to a wider buffer on the device and reads them. The host tensors reaching the unchanged merge and selector are element-for-element and dtype-for-dtype the loop's, which `test_tp4_w3_mr` holds on the CPU with distinct per-chip data, negative-zero and NaN-payload bit patterns, and six random passes (`tp4_mesh_read.identical` compares dtype, shape and bytes). What a CPU test cannot show is that the runtime's mesh composer and the TILE concat move those bytes: that is the two audits, and the singles and batched-selection audits (`QWEN_FAST_DRAFT_SINGLES_AUDIT=all`, the round-B1 audit) which read the joined path through `quad_draft.read_quad_outputs` and hold each user's rows against the singles' and the batched selection against the per-user one.

## What only a card can show

- The MR4 verdict (`tp4_mr_probe`, four chips): whether two mesh reads beat the per-chip loop at all (the one-card probe can only read INCONCLUSIVE-SINGLE-CHIP). Pick the flags per arm from it; the pack's paired timing reads `reads_ms` in `[PACKED-HOSTGAP-VERIFY]` and `read_ms` in `[PACKED-SELECT]`.
- Whether the concat of uint16 TILE tensors at width 16 is accepted by this build, and that its bytes equal the chunks' (the joined audit).
- The mesh composer's chip order and dtype handling on a (1, 4) mesh against `get_device_tensors` order (the MR audit).
- The trace-region and hang effect of two more intermediates in the quad's capture (hang shapes 5 of 5 with the audits off).
- The saving: the audit's -3 to -5.7 ms a round assumes about 0.12 ms a read; it is an estimate until `reads_ms` / `read_ms` say.

## Files

`scripts/ci/tp4_mesh_read.py` (flags, reads, join, settle, audits; stdlib only at import), `scripts/ci/w3_mr_smoke.py` (the smoke rule, host side), `scripts/ci/test_tp4_w3_mr.py`; edits in `packed_verifier.py` (constructor flags, `shard_predictions`, two marker spots), `quad_draft_tp.py` (`read_quad_outputs`, the bucket's `settle`), `dflash_proposal_trace.py` (`publish_outputs`), `serving_buffer_pool.py` (the optional `joined` key), `dflash_packed_proposal_coordinator.py` (`pooled_draft_output_shapes`). `docker/qwen-c2-overlay.txt` carries `tp4_mesh_read.py`.
