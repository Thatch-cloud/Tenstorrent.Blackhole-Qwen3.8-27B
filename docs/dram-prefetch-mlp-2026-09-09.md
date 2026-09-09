# Next verifier experiment: DRAM-core weight prefetch

**Not implemented or performance-qualified.** The current request verifier is
about 62 ms; draft-only improvements cannot deliver the 200-TG target at the
measured 4K acceptance rate. Projection execution and weight movement are next.

## Proposed comparison

| Arm | Gate/up workers | Down workers | Weight delivery |
| --- | ---: | ---: | --- |
| Existing lead | 39 active, 44 requested | 32 active, 33 requested | Existing interleaved DRAM |
| Expanded control | 68 | 80 | Interleaved DRAM, same native math |
| Prefetch candidate | 68 | 80 | Receiver-contiguous DRAM shards into L1 circular buffers |

Compare the candidate against the existing lead, not just a slower expanded
control. Keep T8, native 32-row tiles, BF4 gate/up, BF8 down, LoFi, FP32
accumulation, SiLU and the same four-link reduction. Exactness is still unproven.

With K-block width 8, the proposed minimum two-page receiver buffers are 36,864
bytes for gate/up and 34,816 bytes for down. The full MLP weights occupy
97,484,800 bytes per chip; a shallow stream need not keep all of them in L1.
These are layout calculations, not measured capacity or bandwidth.

## Native support and limits

The pinned source already has `ttnn.experimental.tensor_prefetcher_matmul` and
`prefetch_and_linear`. This uses programmable DRAM cores and on-chip NoC, not
PCIe loading, QSFP weight loading or spare Tensix producer kernels.

- Use the paired helper: it captures the prefetch request with its consumer.
- Global-buffer receivers must exactly equal the active matmul workers.
- Receiver-contiguous shards use three DRISC staging slots, not the legacy
  K-row-major two-half schedule. Unequal receiver counts per bank are supported.
- The runtime enables programmable DRAM cores only with sufficient firmware
  (at least 19.12.0.0) and either no harvested DRAM channels or a single device.
  Do not assume this is available on our pair or bypass its capability check.
- Local simulator capability is false; no prefetch kernel has run there.
  The next request CI run records the real pair's capability without starting it.

Next gates: verify hardware capability, simulate unchanged projection math and
weight layout, then qualify the supported hardware producer/consumer protocol,
changed-input replay and complete-MLP ABBA. Any simulator protocol limitation must
remain explicit. No firmware, native source or serving-default change is authorized
by this experiment description.
