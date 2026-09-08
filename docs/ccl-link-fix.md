# Explicit CCL links: two separate fixes

| Problem | Change |
| --- | --- |
| Native warning despite explicit `num_links` | Replace eager `optional.value_or(discover())` with lazy conditional selection at seven audited all-gather/reduce-scatter sites. |
| Learned projection calls hard-coded to one link | Hardware experiment launcher selects the audited P150A-pair descriptor and requests four links for attention, MLP and feature projection. |
| Silent configuration drift | Multi-link policy requires allocation, exact descriptor hash and no simulator/mock/slow-dispatch configuration. |

The warning alone does **not** prove an explicit count was replaced: C++ evaluates
`value_or` arguments even when the optional is populated. The old native code can
print “falling back to 1 link” and still select the explicit four-link value.
The projection path nevertheless really did request one link before this change.

The isolated native rebuild preserves existing transformer registrations, audits
source hashes and records the rebuilt library hash. It does not replace the
serving image. Select `suite=learned-attention, fabric_link_probe=true`:

- First use `simulator_only=true`: build and exact one-link simulated collective
  checks at 1/8/32 rows on both chips; fail if explicit calls still log discovery fallback.
- After that passes, use `simulator_only=false, cards_allocated=true` for the
  four-link hardware gate. No reset is requested.

Host and standalone C++ tests are not hardware certification. Existing running
jobs retain their original code. Other model collectives with explicit two-link
settings remain unchanged; this is not a claim of runtime-wide four-link adoption
or repaired automatic topology discovery.
