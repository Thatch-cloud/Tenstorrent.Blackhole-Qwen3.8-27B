# Fused DFlash K/V history publication

**Simulator and combined correctness passed; no attributable speedup. Not promoted.**

## Combined hardware result

Run **35318783547**, source `feaa1d2`, passed at CTX4096, one stream, two cards,
four links, unchanged T16 native-DFlash and serial packed-weight reader.
Two audits and ABBA requests preserve proposals, accepted output, target state
and feature checks. Both timed arms commit 242 tokens in 22 blocks, with identical
222/330 proposal acceptance. All publication scopes restore and the pool releases.

| Metric | Original publication | Fused publication |
|---|---:|---:|
| PP tokens/s | 3,348.23 | 3,384.72 |
| Committed TG tokens/s | 112.98 | 115.44 |
| Per-request TG | 112.73 / 113.22 | 112.12 / 118.97 |
| Draft ms/block | 21.58 | 20.36 |
| Input ms/block | 2.69 | 1.62 |
| Verify/readback ms/block | 60.37 | 60.26 |
| Select/commit ms/block | 12.29 | 12.65 |
| Complete cycle ms/block | 97.25 | 95.18 |

The aggregate +2.18% TG is not attributed to this change: the modified
publication interval gets 0.36 ms slower, while savings occur mainly in drafting
and input work. One candidate request is slower than both controls. Keep the
original publication default; retain the exact fused implementation as a qualified
component for a structurally different combined design, not as a proven speedup.
Do not add this percentage to other experiments or rerun unchanged automatically.

Report SHA256:
`a3a184728660c1fdd631cb1a9c4b9c4b8363af85f7343a032d71752adf52ecac`.

## Simulator evidence

Run **35317724110** passed in **3m38s**: all 120 bitwise comparisons, five boundary
cases, both chips, eager and changed-input replay 0/1/0. Exit zero and clean close.
Report SHA256: `7eb9560c661f77bc422d9551d513bf0e264f88eb2a0c7ab0ac0ce6a7a8099a38`.
The previous run stopped at its 180-second probe deadline during the final case;
the successful retry used a 240-second inner cap, with the same kernel bytes.

`DraftKVHistory.prepare` currently performs historical slice, accepted-prefix
slice, concatenation, tail slice, padding and copy for each of five layers' K/V
tensors. `draft_kv_slide.cpp` replaces that transport chain with one dataflow
operation per tensor. It does not change projection arithmetic or quantization.

| Property | Candidate |
|---|---|
| Storage | BF16, four heads, 2,048 history rows, head dimension 128 |
| Accepted update | 1-32 rows; rolling window drops only the oldest overflow |
| Workers | 16 data-movement workers, one per head/32-column tile |
| Scratch | 8 KiB per worker; no matrix/SFPU compute kernel |
| Publication | Write spare storage; active history and accepted input remain unchanged |
| Partial history | Zero every row after the valid frontier |

This is dataflow fusion, not zero-copy: at a full window it still writes 2 MiB
per tensor per card. Across five K/V pairs that is 20 MiB per card per update.
It removes intermediate tensors and dispatches, not the necessary rolling-window
movement. Arbitrary accepted lengths require tile-face-aware row assembly on the
data-movement processor, alongside asynchronous NoC transfers.

## Gates

1. CPU geometry checks cover every prefix 1-32 across tile and capacity boundaries.
2. Weight-free simulator checks five boundary cases, both chips, eager execution
   and changed-input replay 0/1/0. All 120 output/input comparisons must be bitwise
   exact, source hashes must match, and cleanup must succeed.
3. Add an explicitly admitted request scope replacing only the publication copy
   chain. Preserve prepare/commit/discard ownership and the existing synchronization.
4. Compare serial versus fused publication in the complete T16/native-DFlash
   runtime: same proposals, target tokens, state, acceptance, weights and precision.
   Promote only a repeatable complete-cycle improvement.

The dedicated simulator workflow reuses the weight-free `history-append`
container route by staging the new probe under that route's filename. Its output
remains `history-append.json`, with explicit sliding-copy scope and independently
checked source hashes; this is not reuse of the old append kernel's qualification.

## Other fusion boundaries

### Direct face-segment DMA candidate

The first fused writer still assembles every output row with scalar 32-bit loads
and stores on its data-movement processor. `draft_kv_slide_direct.cpp` instead
issues face-segment NoC reads into the output tile, splitting only at
source/destination face boundaries and the historical/accepted frontier. Reads
complete before the output write; writes complete before reusing scratch.
Only invalid tail rows use scalar zeroing. The original qualified kernel remains
unchanged, and the direct-DMA candidate requires its own simulator evidence.

Simulator run 35319847641 rejected the initial direct path: DRAM source
`0x594e80` and L1 destination `0x1d160` differ modulo 64. This was a real
alignment bug, not a timeout. Blackhole's pinned `noc_parameters.h` specifies
64-byte DRAM-read and 16-byte L1-read alignment. The revised candidate uses
direct reads only when source/destination offsets match modulo 64. Otherwise,
it reads into aligned scratch, waits, performs local L1 DMA into the output,
and waits before scratch reuse. No scalar valid-row copies are introduced.
Host tests cover all face segments, 16-byte-aligned buffer offsets, scratch
bounds and the exact failing addresses. Simulator run **35320765158** passed
all 120 exact comparisons and clean shutdown on source `9cf68f7`. Its report
SHA256 is `221d4e5fc003f55a690cd583e3a4ea55d5269bec611b47dc21fb2fbfcf087c6e`;
the candidate kernel hash is `1679bbd779add56b4bd445a6b4c51bd3e49c39a8a520dfaddfc3bd9f36667d47`.
An explicit `--direct-dma` stage selects this kernel and separately pinned report
for the same combined publication comparison. The original scalar-copy route is
unchanged. Hardware performance remains unqualified: extra transfers and barriers
may negate any benefit.

### Combined staged-DMA result

Hardware run **35321449903**, source `01ab58b`, completed both feature audits and
ABBA requests with exact matching proposals, target tokens/state and clean close.
The workflow failed only in its final external validator: it inspected the
orchestrator's original kernel rather than the staged DMA kernel. Recovery run
**35322243498** passed the unchanged complete report validator with the exact
executed kernel; no hardware work was repeated. Raw report SHA256:
`0be812cceaf6f91686570bb02a9991cc5a58d8aaa95e05d0633388802799a2bc`.

| 4K, one stream, native DFlash T16 | Control | Staged DMA |
|---|---:|---:|
| PP tokens/s | 3314.98 | 3309.40 |
| Committed TG tokens/s | 111.94 | 121.12 |
| Individual request TG | 107.89 / 116.30 | 119.09 / 123.23 |
| Draft ms/block | 22.12 | 19.98 |
| Verify/readback ms/block | 60.46 | 60.24 |
| Select/commit ms/block | 12.71 | 8.77 |
| Complete cycle ms/block | 98.15 | 90.70 |

Both arms committed 242 timed tokens over 22 blocks. Aggregate TG improved 8.21%
in this run and publication-containing select/commit fell 3.94 ms/block; drafter
and input differences account for part of the total improvement. This is one
matched run, not a repeatability or held-out coding-quality qualification.
No promotion or serving change. At 11 committed tokens/block, the 200-TG budget
is 55 ms, still below the roughly 60-ms verifier alone.

This is a different mechanism, not an unchanged retry of the first fusion.
More small NoC transfers could offset the saved scalar copies, so no benefit is
assumed. CPU segment tests cover all accepted lengths 1-32, all 64 tiles and
history/face boundaries. The dedicated simulator selects it only with a
`-direct-dma` tag, retaining the same two-chip 120-comparison matrix.

Next review the copy from committed K/V banks into captured proposal buffers.
The banks swap on commit while trace addresses are fixed, so removing this copy
requires explicit bank-specific trace binding or stable-address publication.
Simply pointing the trace at whichever bank is currently active is unsafe.

Target MLP producer/consumer fusion remains a separate higher-impact investigation;
see `target-fusion-worker-audit.md`. Neither publication fusion nor batching alone
is assumed to close the single-stream 200-TG gap.
