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

For this host-only selection fix, the standalone compiled C++ test verifies
lazy evaluation without a full TTsim run; no device kernel math changes.
The hardware probe now measures three ABBA blocks of 1/4/4/1 links at each
1/8/32-row shape, checks exact outputs on both chips for every timed operation,
and reports per-arm milliseconds. These are collective costs, not model TG.

Host and standalone C++ tests are not hardware certification. Existing running
jobs retain their original code. Other model collectives with explicit two-link
settings remain unchanged; this is not a claim of runtime-wide four-link adoption
or repaired automatic topology discovery.

## Hardware result: 34189506734

The isolated rebuild and 36 timed ABBA collectives passed on both cards, with
exact outputs and no fallback-discovery warning. Rebuilt `_ttnncpp.so` SHA256:
`ae3d9a5f84a51249a319c7bf91083e741226fc2f76c9b748850d71f4dbd62d37`.

| Rows | 1-link median ms | 4-link median ms | Interpretation |
| --- | ---: | ---: | --- |
| 1 | 1.1592 | 0.1343 | Startup/timing transition contaminates comparison; not a credible 8.6x gain |
| 8 | 0.1271 | 0.1391 | No measured improvement from four links |
| 32 | 0.1229 | 0.1255 | Essentially similar small-payload latency |

Raw T1 samples began near 2 ms in both arms and later settled near 0.13 ms.
This certifies the explicit-link correctness path, not a throughput improvement.
Do not turn the first-row median ratio into a decode speed claim. Continue
learned drafting and full-request measurement rather than more link microbenchmarks.
