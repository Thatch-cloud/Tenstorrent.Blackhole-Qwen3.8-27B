# T16 weight-read issue order

**Candidate only: no simulator or hardware acceptance yet.**

Combined run 35210079674 found larger weight-read completion waits than the
isolated fixture. This experiment changes request ordering, not buffer capacity,
weight precision/layout, arithmetic, core count or NoC selection.

The 91-worker fused projection reads six columns per worker in the same order.
With eight-way interleaving, each synchronized issue slot initially addresses
four banks, with up to 23 requests to one bank. Rotating the six-column order
after each four-worker group spreads the idealized issue slot over all eight
banks, with at most 12 requests per bank. Workers are not actually synchronized
at each instruction: this is a scheduling hypothesis, not measured bank traffic
or a predicted throughput gain.

Each read keeps its original destination tile. The full block read barrier stays
before the same CB push. Host tests enumerate every worker and K block: all
87,040 packed pages and all padded destination tiles remain unchanged, exactly
once. A reverse source transformation reproduces the original reader verbatim.

Qualification order:
1. Bounded T16 simulator eager/replay and byte-exact packed-weight checks.
2. Same winning combined runtime, native output/state/feature audit and complete
   uninstrumented request timing against the unchanged reader.
3. Retain only a repeatable combined TG improvement; do not promote an isolated
   projection speedup or the analytical bank model.

The simulator fixture has a 510-second process cap and 12-minute whole-job cap;
it does not load the full target model. Serving defaults remain unchanged.
