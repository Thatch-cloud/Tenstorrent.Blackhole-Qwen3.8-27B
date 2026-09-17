# Direct causal windows inside GDN convolution

Status: reader transformation and host indexing tests only. **No simulator or
hardware qualification yet; not installed into the runtime.**

The retained combined trace assigns about 3.49 ms per T16 verifier block to
48 source-consistent window-builder calls. Overlapping the old builder's writes
did not improve combined TG. This candidate instead removes the materialized
input-window stage and feeds causal rows directly to native convolution math.

Read-only CI inventory **35270248051** exports 12 files from the pinned image
in **13 seconds**, without opening cards or loading weights. All downloaded
file hashes match the inventory. The native reader actually consumes shifted
states `[st1, st2, st3, x]`; its writer preserves that advanced shift register.
Therefore the candidate must still publish four complete per-prefix checkpoint
tensors. Removing those outputs would break speculative rollback.

`gdn_direct_window.py` changes only the native reader's window construction:
read the projected tile and three entry-history tiles into 8 KiB private L1
scratch, assemble causal rows into the existing window CB, and retain the
native tap/gate paths. Native compute, rounding and writer remain unchanged.
The intended descriptor uses immutable entry history as reader inputs and
distinct checkpoint buffers as writer outputs. T16, 5,120 channels and the
8,240-wide packed projection are deliberately bounded.

Host tests check every row and checkpoint against serial shift semantics and
reject unsupported coordinates/source boundaries. The actual pinned reader
passes the source transformation. Next: build the explicit descriptor, then
compare outputs, all prefixes and changed-input trace replay in the simulator.
Only after that gate may it enter the matched combined hardware test. This
roughly 3.49-ms opportunity alone cannot close the full gap to 200 TG.
