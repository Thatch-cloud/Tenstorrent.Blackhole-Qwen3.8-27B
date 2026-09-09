# Next verifier experiment: streamed MLP weights

**Transport prototype only; matmul and performance unqualified.** The current request verifier is
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
  Hardware run34314276820 also reports false with the API present. No DRISC
  kernel ran on the pair, and the specific firmware/harvesting cause is unproven.

The programmable-DRAM route stops at its capability gate. The worker alternative
below must separately qualify transport, unchanged projection math, complete-MLP
ABBA and request throughput. No firmware or serving-default change is authorized
by this experiment description.

## Tensix alternative

Do not spoof DRISC support. A new simulator-first transport experiment uses eight
actual Tensix producer cores on the last worker row, with 68 gate/up or 80 down
receivers on disjoint cores. Double-buffered staging and remote circular-buffer
credits allow weight delivery to overlap consumption. The first sink copies raw
compressed words back to DRAM; it is not a matmul or bandwidth result.

The native mcast consumer uses the same remote-CB protocol but explicitly admits
only DRAM senders today. The new [projection prototype](tensix-streamed-projection-2026-09-09.md)
instead combines the worker producer with unchanged native matmul compute through
`ttnn.generic_op`. It does not relax that native validator. Actual arithmetic,
complete-MLP integration and hardware performance remain unqualified.

## First transport result

`20260909T054848Z-409` passes the BF4 gate geometry with five K blocks
(1,280 x 8,704 local weights, 6,266,880 compressed bytes per chip). This is only
one quarter of the full gate matrix. Eight sender cores feed all 68 receivers,
including unequal receiver counts per sender and repeated two-page FIFO wrap.

- Eight eager comparisons, twelve changed-input trace comparisons and four
  stale-input controls pass; two independent GCBs and traces remain live.
- Both chips use different synthetic data, with two input patterns. Every raw
  packed word matches an independent direct-DRAM copy; weights stay unchanged.
- Trace input/output addresses remain stable. Trace release and mesh close pass.
- The runtime is unmodified, including the original packer. No DRISC kernel runs.
- 924 CI tests and 55 simulator-harness tests pass. These are host test counts,
  not additional hardware or kernel correctness results.

The checked-in report is `scripts/ci/tensix-stream-simulator-gate-5.json`, SHA256
`0b321c801717777c63c33d36f3ee045f7b55242cbc1fa10d50b1200dfa8e9226`.
`tensix_stream_gate.py` independently checks source hashes, geometry, every audit
coordinate and clean termination. Logs and exit status are archived under
`hardware-evidence.local/tensix-weight-stream/`.

The BF8 five-block test also passes the same complete audit matrix and clean
closure, using 80 receivers and 6,963,200 compressed bytes per chip. Report
`scripts/ci/tensix-stream-simulator-down-5.json` has SHA256
`6513206df406f181db8f1defde193040378c4d5cc5e47a40fc30a91f52f9f662`.
This is still only 1,280 of the required 8,704 K rows.

Full BF4 gate transport now passes as well: 5,120 x 8,704 local weights,
25,067,520 bytes per chip, with the same complete audit matrix and clean closure.
Report `scripts/ci/tensix-stream-simulator-gate-20.json` has SHA256
`8e885087945452eca5726849b3a14d49692c74c4078680405ec00717f7ec3d05`.
Full BF8 down transport completes its kernel and cleanup checks but the outer
wrapper fails after being edited during execution. It is not recorded as a clean
suite pass. The [failure and wrapper regression fix](tensix-streamed-projection-2026-09-09.md#wrapper-failure-is-retained)
are retained. The new zero-copy projection passes its first BF4 arithmetic/trace
gate, and full projection tests are running sequentially with the repaired wrapper.
Transport alone does not qualify matmul, useful overlap,
bandwidth, fabric weight loading or a new TG rate.
