# Streamed target MLP projections

**Correct, but not faster.** Full projections and pooled MLP pass simulation;
the [complete hardware MLP](tensix-pooled-mlp-2026-09-09.md) takes 0.744 ms versus
0.335 ms native and fails promotion. This targets the roughly 62-ms verifier,
not another drafter-only improvement. No serving defaults change.

## What changes

| Part | Existing lead | Experimental projection |
| --- | --- | --- |
| Gate/up compute | 39 active workers | 68 workers |
| Down compute | 32 active workers | 80 workers |
| Weight loading | Each compute worker reads DRAM | Eight disjoint Tensix producers read and stream weights |
| Weight consumption | Local matmul buffer | Zero-copy local CB1 alias of remote CB31 |
| Arithmetic | Native TT-Metal matmul | Same native compute source and precision |
| Activation loading | Multicast | Multicast, with idle rectangle cores draining their buffers |

Producer reader/writer RISCs overlap DRAM reads with remote writes. The consumer
retains one K block of lookahead and returns remote credit only after the unpack
engine has drained the old local block. No intermediate weight tensor is copied
between the remote FIFO and matmul.

The prototype combines these kernels through `ttnn.generic_op`. It does not
enable worker senders in the native DRISC-only matmul entry, bypass its validator,
change firmware, or pretend programmable DRAM cores are supported.

## What stays fixed

- T8, physical 32-row tiles, eight-tile K blocks and full local N width.
- BF4 gate/up, BF8 down, BF16 activations/output, LoFi, FP32 destination and
  FP32 partial reload. Native approximate-math mode and packer L1 accumulation.
- Native SiLU only on gate. Up/down have no added activation.
- The current small-grid lead remains a comparator; beating a slower expanded
  grid alone is not enough to promote the candidate.

## Gates

| Gate | Current state |
| --- | --- |
| BF4/BF8 transport, odd five-block FIFO wrap | Pass: both chips, independent inputs and two live traces |
| Full BF4 gate transport, 5,120 x 8,704 per chip | Pass: 25,067,520 compressed bytes per chip |
| Full BF8 down transport, 8,704 x 5,120 per chip | Kernel/cleanup audits pass; outer wrapper fails, not a clean suite pass |
| BF4/BF8 native projections, five K blocks | Pass: native controls, all physical rows and two live traces |
| Full native gate/up/down projections | All pass: native controls, all physical rows, changed inputs and clean wrapper exits |
| Complete MLP, shared FIFOs/workspace and DRAM input boundary | [Exact simulator pass](tensix-pooled-mlp-2026-09-09.md) |
| Real-weight complete MLP and four-link reduction | Not qualified |
| Complete request PP / CTX / committed TG | Not qualified |

The projection harness first compares the native lead and expanded grid. It
then checks eight candidate eager comparisons, twelve changed-input trace
comparisons and four stale-input controls. Raw BF16 words cover **all 32 physical
rows**, not just eight live rows. Raw packed weights and input words must remain
unchanged; all buffers are allocated before capture and both traces stay live.

Initial arithmetic testing uses five K blocks before full-size gate/up/down.
Its report explicitly distinguishes partial and full projections. Fifteen
experiment sources and fourteen native files bind the independent evidence gate.
The known simulator-only packer compatibility graft is explicit and must be
restored afterward; it is not applied to hardware.

## First arithmetic result

`20260909T063913Z-408` passes all four native-control, eight candidate eager,
twelve changed-input replay and four stale-input checks on the simulated pair.
Every physical output word matches the native small-grid lead and expanded grid.
Both input and packed weight words remain unchanged. Cleanup and wrapper exit
status are clean. This is the five-block BF4 gate, not the full projection.

Checked-in report: `scripts/ci/tensix-projection-simulator-gate-5.json`, SHA256
`6db16e102f4a12f3dde271c9e6ba64b06ddf13c19c37c2876055e0b75dba85d1`.
Full-size tests and BF8 use the same source. No hardware timing is reported.

The matching five-block BF8 down test `20260909T064528Z-395` also passes the
complete control/eager/replay/stale matrix, input immutability and clean wrapper
exit. Report `scripts/ci/tensix-projection-simulator-down-5.json` has SHA256
`245cc47837828761535d61ec1f93652f1eed6e66e1f2dee65d083eae69daf069`.
Full gate, up and down tests pass on the same projection source, with all control,
eager, replay, stale-input and raw-word audits. Each recorded outer exit is zero.

| Full projection | Local K x N | Report SHA256 |
| --- | --- | --- |
| Gate, 20 blocks | 5,120 x 8,704 | `490e347f42b3bf4d78087e965b6e385954dfd46126db8601b738157760153d96` |
| Up, 20 blocks | 5,120 x 8,704 | `93c958a3dfd252be44400a6104271ed67375daa088fab261378058167423ea34` |
| Down, 34 blocks | 8,704 x 5,120 | `dc67b26a7ff67250d5808825b10eb7f727cd8bd1e5ae9ebf0b2bad2ed2edff1e` |

Reports are `scripts/ci/tensix-projection-simulator-{gate-20,up-20,down-34}.json`.
The earlier standalone BF8 transport wrapper failure remains a failure; the
new complete down-projection pass does not rewrite its history.

## Wrapper failure is retained

Full BF8 transport `20260909T061515Z-1075` completes all kernel checks and clean
mesh teardown, but its outer wrapper exits 1 without an exit-status artifact.
Editing that shell file while its Python child ran caused Bash to resume parsing
the modified tail at an obsolete offset. The JSON body passes its independent
audit; that does not turn the overall launch into a clean pass.

The terminal commands now form one compound block, parsed before the long-running
child starts. Two host regression tests reproduce the old failure and confirm
status recording survives the same edit with the fix. This does not authorize
editing kernel source during a test. Failure logs and JSON remain under
`hardware-evidence.local/tensix-weight-stream/wrapper-failure-down34/`.

## Performance budget

At the measured 4K acceptance rate, 200 TG requires **35.59 ms for the whole
proposal/verify/commit cycle**. The current verifier alone takes about 62 ms.
The unchanged MLP reads 97,484,800 weight bytes per chip. At the card's specified
[512 GB/s peak bandwidth](https://docs.tenstorrent.com/aibs/blackhole/index.html#card-comparison-table),
the ideal one-pass read floor is about 0.1904 ms per MLP, before compute, NoC,
other traffic and collectives. This is a calculated bound, not a benchmark.
Do not assume spare-core streaming alone delivers 200 TG; use complete-MLP and
request measurements to decide whether it warrants integration.
