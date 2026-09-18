# Fused DFlash K/V history publication

**Simulator passed; no combined hardware speedup is qualified yet.**

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

Next review the copy from committed K/V banks into captured proposal buffers.
The banks swap on commit while trace addresses are fixed, so removing this copy
requires explicit bank-specific trace binding or stable-address publication.
Simply pointing the trace at whichever bank is currently active is unsafe.

Target MLP producer/consumer fusion remains a separate higher-impact investigation;
see `target-fusion-worker-audit.md`. Neither publication fusion nor batching alone
is assumed to close the single-stream 200-TG gap.
