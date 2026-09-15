# 32K matched-context numerical blocker

## Resolved for this fixture: FP32 local maxima

Hardware run **35034936748** passes all 72 numerical/input/layout/replay checks
and eight fixture controls with clean shutdown. The isolated factory change
promotes local maxima buffers c27/c28 to FP32; tolerances and full history are
unchanged. First-case chip-1 maximum absolute error falls from 0.468784 to
0.279079, with zero out-of-tolerance values across both cases and both chips.
Report SHA256: `f9d1bc607a699490c4a4d87a3b0311bd24e7c2d9d54f4730c8cd624e900e3670`.

Simulator admission is run **35034190179**, report SHA256
`24a5659adba79424b6dee5106257b44469c22a8b8fa7341e2d788b1359b54d2e`.
Other contexts, full-model correctness and throughput remain unqualified for
this candidate. It is not enabled in serving or the combined runtime yet.

The 200 committed tok/s objective remains open. This is a draft-attention
correctness blocker, not a new model-throughput result or a serving change.

## Reproduced evidence

| Run | Result |
| --- | --- |
| 35029267653 | 8K and 16K jobs green; 32K fails; larger contexts cancelled |
| 35029765022 | Same 32K failure reproduced with value diagnostics; clean device shutdown |

Both failures occur in the first eager case on chip 1. Four elements in head 15,
query row 5 (columns 8, 38, 54, 77) return -45.0 instead of FP32 references
between -45.467598 and -45.468784. The nearest BF16 reference is -45.5.
Shapes match and outputs are finite. Error is 0.467598–0.468784 against the
unchanged allowed error of about 0.46468 (`atol=0.01`, `rtol=0.01`).

This is not merely unavoidable final BF16 rounding. It also does not establish
which intermediate operation causes the discrepancy. The eager record's
`exact_replay=true` compares the eager checksum to itself; replay was never
reached and is **not qualified** by this failed report.

## Avoid repeating rejected experiments

`splitk-draft-status.md` records the existing numerical investigation:

- Explicit BF16 correction-factor rounding previously left hardware output
  unchanged (34916226351); it is already in the active composed transform.
- Fused FP32 numerator recurrence worsened the 64K result (34915054290).
- FP32 local/tree denominators improved error but were insufficient alone.
- Increasing key chunks from 32 to 128 to 256 reduced repeated recurrence
  error; 256 passed the historical 64K fixture, not every context or input.

Do not loosen tolerances, retry unchanged rounding, or treat passing 64K as
proof of 32K numerical correctness. The context changes fixture values as well
as worker partition lengths, so context length alone is not the isolated cause.

## Next discriminating experiment

Use the failing head/query and the same input bits to compare the dominant
oldest-key and last-proposal contributions, local worker numerator/denominator,
and final normalization. Existing DPRINT snapshots select only the first four
rows of tile zero; folded query row 5/head-within-group 3 is row 23, so those
snapshots do not observe this failure. A targeted diagnostic must select that
row and its lane before attributing the error to reduction or normalization.

Any changed kernel or chunk configuration needs a bounded simulator gate
before the same 32K hardware fixture. Only after numerical and changed-input
replay gates pass should this configuration enter the combined context ladder.
Full-model weights are unnecessary for isolating this failure.

## Hardware contribution isolation

Run **35033433891** completed the failure diagnostics in a 14-second hardware
step and closed cleanly. It retains the original numerical failure; this is not
an acceptance pass. Q/K/mask and kernel math were unchanged while values were
replaced with oldest-key-only, last-proposal-only, or constant-one inputs.
All four failing columns produced the same diagnostic values:

| Value probe | Device BF16 output | CPU FP32 reference |
| --- | ---: | ---: |
| Oldest key = 1, others = 0 | 0.103515625 | 0.101952322 |
| Last proposal = 1, others = 0 | 0.808593750 | 0.812258601 |
| All values = 1 | 0.996093750 | 0.999999821 |

The constant result shows normalization/output-path error. The oldest-key
weight moves in the opposite direction to the last-proposal weight, so a
uniform output rescale cannot repair both. These BF16 outputs do not expose
unrounded internal probabilities; they cannot uniquely identify the defective
operation. Next inspect local maximum/correction precision and the final
normalization path, keeping the already-rejected fused numerator experiment
out of the candidate. The CPU attribution script reproduces the reference at
the four failing coordinates to within 0.00004.

Broad and root-only simulator DPRINT attempts repeatedly failed fabric startup
before attention execution. They do not invalidate hardware numerical evidence,
but no intermediate values were obtained. Do not repeat those runs unchanged.
