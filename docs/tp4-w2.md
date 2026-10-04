# tp4/w2: wave 2 of the fusion plan on the eight-seat 262k stack

Wave 2 is wave 1 (`tp4/w1`: nosamp, the drafter ops D2a and D2c, the host-gap stage 1, U1, the drafter ring gathers) plus the multi-user SDPA launch
(`tp4/sdpa-multi`, `QWEN_FAST_TP4_SDPA=multi`) plus F1, the conv-gates spread (`QWEN_FAST_TP4_CONV_GATES_SPREAD`). Every flag is default off and byte-identical off.
Two gate-only profiles switch the stack on; a job pack drives the window.

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

Both are gate only and carry the 262k evidence waiver, as the control does. The V5 flag is in neither.

## F1: the conv-gates spread

**The cost.** The served `gdn_decode_conv_gates` factory puts every gate tile on one extra core. At a 64-row block that core runs two gate tiles one after the other, and its
reader gathers `a` and `b` one bfloat16 at a time with a barrier per page. The launch takes 42.3 us against 20.6 us for the same launch with one gate tile (the 4-row trace's):
48 launches a verify, two blocks a round, so about 2 to 3 ms a round.

**The change.** A K-jit `generic_op` twin of the block launch (`gdn_conv_gates_spread.py`), bound where the V1 stage makes its one call (`gdn_block_conv_tp.stage`):

- the served compute kernel runs unchanged: it is read from the image tree and accepted only when all four served conv-gates sources match their sha256 pins;
- two new data-movement kernels, generated from the served reader and writer by two changes only: a gate-start runtime word (the served gate loop starts at tile 0, which is why the served
  reader and writer cannot be reused), and a word gather of `a` and `b` behind one barrier (TP4: `a` at columns 0-11 and `b` at columns 12-23 of one tile, so six words a row);
- gate tile `k` sits on core `conv_cores + k`, alone. The conv instances keep the served partition (at 64 rows: 160 instances, 80 cores of two) whenever the gate cores fit beside it; otherwise
  instances per core grow by one at most, and a grid that cannot hold the gate cores falls back.

**No graft rebuild.** The kernels are JIT sources shipped as files (`gdn_conv_gates_spread_reader.cpp`, `_writer.cpp`) in both image copy lists and the overlay, the served `.so` is untouched,
and nothing here edits a pinned file. The plan's "F1a with the three served kernel files unchanged" was not possible (the served reader and writer have no gate start); the plan's review
said so, and this is that variant: the served compute file plus a reader and writer pair.

**Exactness.** By construction: the instance loops are the served ones verbatim (a test holds this against the served sources when a tree is at hand), the compute binary is the served one,
and the new code only copies the same bytes to the same positions of the same zeroed tile. Neither change touches arithmetic or rounding. On the card the audit proves it:

- `QWEN_FAST_TP4_CONV_GATES_SPREAD_AUDIT=1` (needs the lever and `QWEN_FAST_TP4_VGLUE_AUDIT=1`) runs the served op on cloned block windows beside the launch and holds the launch's conv, beta, g and four
  advanced windows against it (seven entries a launch). The replay audit compares them on every chip as int16 bit patterns (-0 and +0 differ). An audited layer without its seven entries fails.
- the engaged marker `[PINDIAG] tp4 conv gates spread engaged rows=64 conv_cores=80 gate_cores=2 per_core=2 served_per_core=2 audit=<0|1>`, the fall-back marker (`... fell back reason=...`),
  and the audit markers (`... audit <n> exact=True layers=... entries=...`, `... audit mismatch ...`) are read by `c2_smoke_check.spread_problems`.

**Trace.** No buffer is allocated before the first capture for this launch: its outputs are allocated where the served op's are, inside the capture, and it needs no scratch. Runtime-argument
lengths are fixed per role (reader 17, writer 11, compute 2), because `generic_op` does not hash them.

**What a CPU test cannot show** is that the compiled kernels move those bytes. That is A1's audit. The C++ was not compiled in this session (no toolchain): its first compile is the F1 card-M job of the pack (first of the card-M jobs).
A compile or JIT error in an F1 kernel surfaces as an attach failure, not as a fall-back.

## The job pack (`scripts/ci/references/tp4-w2-jobs`, image `tp4-w2-1`)

B0 first (it opens no card and the card-M jobs run in its image), then card M (F1, the one-card byte compare and timing of the spread launch against the served op at 16, 64 and 128 rows, which is also the first compile of its kernels; V1a and V1b, the V5 byte gate; MR, the mesh-read probe), then X0 (status, rescan, reset), S0c (the control attach),
A1 (the audited attach of the stack), H1-H5 (five hang shapes), T1-T6 (three ABAB pairs, control against `-w2`, 32k and 128k), T7 (`-w1` alone, in the same image: wave 2's own increment), P1 (the eight-user
device profile), Z. `ORDER.txt` carries the dependencies and the read rules. The projection is the plan's wave 2: a 32k round of about 159 ms (27.6 tok/s per seat) and about 192 ms at 128k, against 222 and 279 ms
today and 182 and 236 ms for wave 1. The ladder's wave 2 includes V5, which this arm leaves out.

U2, the 65,536-pattern canonical-rule probe, has no harness anywhere in the tree and is not in the pack (ORDER.txt says so). MR's probe (`optimisation/ttnn-op/mr_probe`) times the verify readback per chip, as two mesh-level
reads, as one joined word tensor and non-blocking; on one card its verdict is inconclusive by construction (one chip), and the four-chip answer is read from the `reads_ms` of the stack's own host-gap lines.
