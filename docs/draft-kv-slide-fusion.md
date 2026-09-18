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
issues 32-byte-aligned NoC reads directly into the output tile, splitting only at
source/destination face boundaries and the historical/accepted frontier. Reads
complete before the output write; writes complete before reusing scratch.
Only invalid tail rows use scalar zeroing. The original qualified kernel remains
unchanged, and the direct-DMA candidate requires its own simulator evidence.

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
