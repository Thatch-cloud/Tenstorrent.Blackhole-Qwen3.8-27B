# tp4/w2: wave 2 of the fusion plan on the eight-seat 262k stack

Wave 2 is wave 1 (`tp4/w1`: nosamp, the drafter ops D2a and D2c, the host-gap stage 1, U1, the drafter ring gathers) plus the multi-user SDPA launch
(`tp4/sdpa-multi`, `QWEN_FAST_TP4_SDPA=multi`) plus F1, the conv-gates spread (`QWEN_FAST_TP4_CONV_GATES_SPREAD`). Every flag is default off and byte-identical off.
Two gate-only profiles switch the stack on (and two fallback twins without F1); a job pack drives the window.

## What is merged

| Piece | Where | State |
|---|---|---|
| Wave 1 | `origin/tp4/w1` | the base |
| Multi-user SDPA | `origin/tp4/sdpa-multi` | a real merge; 18 files conflicted, every conflict additive |
| V5 recurrence split | `origin/tp4/v5split` | already in wave 1 (it came in through `tp4/262k8-x`); not merged again; its flag still refuses |
| F1, the conv-gates spread | this branch | new |

The merge is reconciled at profile level: wave 1's profiles are untouched, and the three sdpa-multi profiles (`c2-packed-tp4-best-sdpa`,
`c2-packed-tp4-8x262k-best-sdpamulti` and its audit) are added as they were. The two flagged-twin rows of `tp_addresses` both survive (the attention twin binds
under `QWEN_FAST_TP4_ATTN_FOLD or QWEN_FAST_TP4_SDPA`, the K5-A launch under `QWEN_FAST_GDN_SPLIT_V`).

## The profiles

| Profile | Is | Exact delta |
|---|---|---|
| `c2-packed-tp4-8x262k-w2` | `-w1` | `QWEN_FAST_TP4_SDPA=multi`, `QWEN_FAST_TP4_CONV_GATES_SPREAD=1` |
| `c2-packed-tp4-8x262k-w2-audit` | `-w1-audit` | `QWEN_FAST_TP4_SDPA=multi`, `QWEN_FAST_TP4_SDPA_AUDIT=1`, `QWEN_FAST_TP4_CONV_GATES_SPREAD=1`, `QWEN_FAST_TP4_CONV_GATES_SPREAD_AUDIT=1` |

| `c2-packed-tp4-8x262k-w2-nof1` | `-w1` | `QWEN_FAST_TP4_SDPA=multi` only (the fallback when the F1 card-M job fails) |
| `c2-packed-tp4-8x262k-w2-nof1-audit` | `-w1-audit` | `QWEN_FAST_TP4_SDPA=multi`, `QWEN_FAST_TP4_SDPA_AUDIT=1` |

All four are gate only and carry the 262k evidence waiver, as the control does. The V5 flag is in none. The two `-nof1` profiles exist so that a compile error or a
differing byte in F1's never-compiled kernels does not end the window with wave 1 plus multi never run: the pack swaps the profile name in the later jobs (no new commit, no new image).

## F1: the conv-gates spread

**The cost.** The served `gdn_decode_conv_gates` factory puts every gate tile on one extra core. At a 64-row block that core runs two gate tiles one after the other, and its
reader gathers `a` and `b` one bfloat16 at a time with a barrier per page. The launch takes 42.3 us against 20.6 us for the same launch with one gate tile (the 4-row trace's):
48 launches a verify, two blocks a round, so about 2 to 3 ms a round.

**The change.** A K-jit `generic_op` twin of the block launch (`gdn_conv_gates_spread.py`), bound where the V1 stage makes its one call (`gdn_block_conv_tp.stage`):

- the served compute kernel's source runs unchanged: it is read from the image tree and accepted only when all four served conv-gates sources match their sha256 pins (it is JIT-compiled here as `SOURCE_CODE`
  with the served factory's compile arguments and compute config; the served `.so`'s binary is not byte-compared, the plan's U4 being open);
- two new data-movement kernels, generated from the served reader and writer by two changes only: a gate-start runtime word (the served gate loop starts at tile 0, which is why the served
  reader and writer cannot be reused), and a word gather of `a` and `b` behind one barrier (TP4: `a` at columns 0-11 and `b` at columns 12-23 of one tile, so six words a row);
- gate tile `k` sits on core `conv_cores + k`, alone. The conv instances keep the served partition (at 64 rows: 160 instances, 80 cores of two) whenever the gate cores fit beside it; otherwise
  instances per core grow by one at most, and a grid that cannot hold the gate cores falls back.

**No graft rebuild.** The kernels are JIT sources shipped as files (`gdn_conv_gates_spread_reader.cpp`, `_writer.cpp`) in both image copy lists and the overlay, the served `.so` is untouched,
and nothing here edits a pinned file. The plan's "F1a with the three served kernel files unchanged" was not possible (the served reader and writer have no gate start); the plan's review
said so, and this is that variant: the served compute file plus a reader and writer pair.

**Exactness.** By construction: the instance loops are the served ones verbatim (a CPU test holds this, the pins, the circular-buffer plan and the compute config against the four served sources vendored under
`scripts/ci/fixtures/gdn_conv_gates_served`), the compute source is the served file's, and the new code only copies the same bytes to the same positions of the same zeroed tile. Neither change touches arithmetic or rounding. On the card the audit proves it:

- `QWEN_FAST_TP4_CONV_GATES_SPREAD_AUDIT=1` (needs the lever and `QWEN_FAST_TP4_VGLUE_AUDIT=1`) runs the served op on one shared set of scratch windows (a copy of the block windows, made in the trace)
  beside the launch and holds the launch's conv, beta and g against it (three entries a launch). The replay audit compares them on every chip as int16 bit patterns (-0 and +0 differ). An audited layer
  without its entries fails. The four advanced windows are not held here: the verify-glue audit this one rides compares every user's advanced windows (unstacked from the launch's block windows) with the
  per-user served path's. That keeps the audit's resident memory at about 65 MB a chip at 48 layers and two blocks; holding a clone of each window per launch would have cost about 315 MB.
- the engaged marker `[PINDIAG] tp4 conv gates spread engaged rows=64 conv_cores=80 gate_cores=2 per_core=2 served_per_core=2 audit=<0|1>`, the fall-back marker (`... fell back reason=...`),
  and the audit markers (`... audit <n> exact=True layers=... entries=...`, `... audit mismatch ...`) are read by `c2_smoke_check.spread_problems`.

**Trace.** Without the audit no buffer is allocated before the first capture for this launch: its outputs are allocated where the served op's are, inside the capture, and it needs no scratch. With the audit
the four scratch windows are allocated inside the first capture that reaches the launch and kept for the process's life. Runtime-argument lengths are fixed per role (reader 17, writer 11, compute 2), because
`generic_op` does not hash them. The module needs no syntax later than Python 3.7 (the audit's `zip(strict=True)` is gone).

**What a CPU test cannot show** is that the compiled kernels move those bytes. That is A1's audit. The C++ was not compiled in this session (no toolchain): its first compile is the F1 card-M job of the pack (first of the card-M jobs).
A compile or JIT error in an F1 kernel surfaces as an attach failure, not as a fall-back.

## The job pack (`scripts/ci/references/tp4-w2-jobs`, image `tp4-w2-1`)

B0 first (it opens no card and the card-M jobs run in its image), then card M (F1, the one-card byte compare and timing of the spread launch against the served op at 16, 64 and 128 rows, each arm under a captured trace
so the host's descriptor build is not measured, which is also the first compile of its kernels; V1a and V1b, the V5 byte gate; MR, the one-card mesh-read probe, which cannot decide; U2, the canonical-rule probe), then X0 (status, rescan, reset), MR4
(the four-chip mesh-read probe, which decides MR and 2a), S0c (the control attach), A1 (the audited attach of the stack), H1-H5 (five hang shapes), T1-T8 in the order A B C A B C A B (control, stack, `-w1` alone: three control-against-stack pairs and two
interleaved wave-1 runs, each right after a stack run, at 32k, 128k and on the skewed eight), P1 (the eight-user device profile), Z. `ORDER.txt` carries the dependencies, the F1 fallback and the read rules. The projection is the plan's wave 2:
a 32k round of about 159 ms (27.6 tok/s per seat) and about 192 ms at 128k, against 222 and 279 ms today and 182 and 236 ms for wave 1. The ladder's wave 2 includes V5, which this arm leaves out.

**The skewed eight.** `concurrent8_skew` (new in `c2_serving_smoke.py`): two real-code prompts of about 253,920 tokens (users 0 and 4, one in each four-seat block) and six of about 4,096, as the server counts them. A multi-user SDPA launch on a
fixed block geometry is paid at its longest user (the plan's row 7s: up to 9 ms a round at heavy skew), so this is the shape where it can lose; the equal-length arms cannot show it. It runs in S0c, A1 and every timed job.

**The F1 fallback.** F1 is the one lever never compiled before it runs. If its card-M job reads FAIL or NOT-RUN, the later jobs that name `-w2` or `-w2-audit` (A1, H1-H5, T2, T5, T8, P1) run on `-w2-nof1` and `-w2-nof1-audit` instead
(a one-line `C2_PROFILE` swap in each template; the profiles are in the same image).

**U2 and MR4.** `optimisation/ttnn-op/canon_probe` puts every one of the 65,536 bfloat16 patterns (64 tiles) through the row mover's canonical modes 3 and 4 and through the served untilize / slice / tilize round trip and the served tile slice,
as int16 bit patterns, with the CPU model of the rule held against the served path and the raw modes as a control; patterns the runtime's upload alters are reported as not exercised (PASS-PARTIAL). `scripts/ci/tp4_mr_probe.py` runs the MR probe's arms on the served (1, 4)
mesh through a new `mr` mode of the quad `fabric` action (`C2_FABRIC_PROBE=mr`); its `reads_ms` counterpart is the stack's own `[PACKED-HOSTGAP-VERIFY]` line (the w2 profiles read back on the shard path).

**Also changed since the first push of this branch.** The multi launch refuses a scope whose segments were rebound to other positions, table or cur_pos buffers after attach (its gather and mask programs bake the attach-time ones, and the timed arms run no SDPA
audit). Not done here: the plan's D2b tau A/B arm and the CCL sweep winners (neither was in the brief), and the registry host and base-image digest on the first line of `docker/qwen-fast-serving.Dockerfile`, which is on `origin/main` and breaks the
public-repo rule on its own (track it separately).
