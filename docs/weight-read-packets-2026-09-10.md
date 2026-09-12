# Fixed-packet weight reads

**Purpose: reduce reader issue overhead, not add another core-count sweep.**
The previous sixteen-producer MLP is correct but takes 0.455749 ms versus native
0.334690 ms. It is not promoted. This experiment changes only how those same
sixteen producers issue their compressed-weight reads.

**Hardware verdict: correct, but still slower than native. Do not promote.**

| Complete Layer-0 T8 MLP | Matched mean latency |
| --- | ---: |
| Native control | 0.334514 ms |
| Fixed-packet sixteen-producer candidate | 0.424554 ms |
| Candidate penalty | **26.92% slower** |

All nine ABBA blocks lose; all 118 correctness checks and independent
source/result reconciliation pass. Timing includes input staging and the same
four-link TP2 collective. This is not a PP / CTX / TG measurement.
The earlier generic-reader prototype measured 0.455749 ms in a separate run:
the lower current figure does not establish a matched reader-only speedup.

| Item | Control | Candidate |
| --- | --- | --- |
| Read API | Generic tile/page read | Compile-time-sized single packet |
| Packed tile bytes | BF4: 576; BF8: 1,088 | Identical |
| Mapping, FIFO depth, transfer order, barriers | Existing sixteen-producer layout | Identical |
| Native compute, precision, activation, reduction | Retained | Identical |
| Serving defaults | Unchanged | Opt-in experiment only |

`tensix_weight_packet_reader.cpp` differs from the pinned reader only in the
read call and two static size guards. The opt-in descriptor adapter preserves
all compile/runtime arguments, core ranges and NoC configuration. It records
each replacement; the MLP gate rejects missing or mismatched engagements.

## Compiler evidence, not a speed result

Both generated readers have smaller `_start`/`.text` sections. Each pair has
identical generated data-format descriptors. This confirms the specialization
is not optimized away; it does **not** establish a latency or bandwidth gain.

| Reader code | Generic | Fixed packet |
| --- | ---: | ---: |
| BF4 | 1,156 bytes | 832 bytes |
| BF8 | 1,180 bytes | 848 bytes |

| BF4 code artifact | SHA256 |
| --- | --- |
| Control ELF | `d86fe385bce4b412f78071edd17be76f8c9ed6c777e35a7be201c4d5e37450f2` |
| Candidate ELF | `68157708f960a111ec45a24f7a986b1e7ed091f6098a6000bbb7c046a2e65792` |
| Control `.text` | `46e84f0345e3f8ee14bc4234ab186acbc8a511ee70559cf0575653aa1dd69f0a` |
| Candidate `.text` | `5f940d64559ba8396f88e158ea5e9b8f27efd1b460bd24301a2bfe74eb77def1` |

BF8 ELF hashes: control
`9bd50e038921e8cd6ba909d9e655fb621d0fc39f53479e84bcea7dd7a4bf8743`,
candidate `888e08c9f41efa04271b0d46b2f9dfd29f7f97d7215c1a97244f93055c4adc17`.

The ELF and disassembly evidence is under `/opt/ttsim/kernel-cache/` and
`/opt/ttsim/results/weight-packet-gate-*`. No native source or packer patch is
needed for the transport-only experiment.

## Gates

1. Full-size BF4 and BF8 transport: independent packed-word oracle, two distinct
   fixtures/chips, both arms, unchanged inputs, changed-input traces and stale controls.
   BF4 run `20260909T141933Z-380` and BF8 run `20260909T143454Z-1338` each pass
   all 24 checks, clean exit 0 and independent source/geometry qualification.
   Each run retains the full projection extent; no packer graft is used.
2. Full T8 MLP: native arithmetic control, two complete weight fixtures, gate/up/
   product/down outputs, all 32 physical rows, pooled FIFO reuse and exact replay.
   Run `20260909T145410Z-414` passes all 188 checks with
   `--single-packet --producers 16`, clean teardown and outer exit 0.
   It uses the same isolated, reviewed simulator packer compatibility patch as
   the previous MLP qualification; this patch is never installed on hardware.
   The original packer and both native binaries are verified restored/unchanged;
   independent MLP and native-weight-view gates pass against the restored runtime.
3. Real Layer-0 hardware ABBA: same input staging and four-link reduction, all
   samples retained. Require a greater-than-2% win in every block before integration.
4. Only then measure a full request in PP / CTX / TG against the retained control.

CI uses suite `tensix-stream-mlp-16` with `weight_packets=true`. It requires the
separate `tensix-mlp-simulator-16-packet.json` and successful outer exit file;
generic-reader evidence cannot qualify this candidate. The input defaults false,
and incompatible suites/profiling are rejected. Hardware comparison
[34371489865](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34371489865)
completed successfully from immutable tag `ci-qwen-hardware-a32cf2a`, but its
performance gate rejects promotion. All 388 existing runner
workflow grants are retained; only this exact tagged workflow is added.

The BF4 report is `scripts/ci/weight-packet-gate-simulator.json`, SHA256
`04e0877c40970158954cc9e656dc22b8b70beb4af01371bdb9324d8761af7aa9`.
The BF8 report is `scripts/ci/weight-packet-down-simulator.json`, SHA256
`a1b88912e8d79dd90347738b5fc6923fc2d0dc83535631d17fe344261395579a`.
Both outer exit files are retained beside their reports. These are transport
passes, not an MLP pass.

The complete MLP evidence is `scripts/ci/tensix-mlp-simulator-16-packet.json`,
SHA256 `1107687be4f1ee302a8259bd2a6c84849bafb3573ffd44b78c96942be3165ced`,
with its successful outer exit file. It contains 32 native-control, 32 eager,
48 replay, 70 raw-input/weight, four stale-input and two distinct-fixture checks.
All 32 physical output rows are compared; all twelve candidate reader descriptors
are engaged. This qualifies hardware correctness/timing tests, not a speed gain.

Hardware report: `runner-evidence.local/34371489865/tensix-mlp.json`, SHA256
`9304b0ab7e7891d376c69b5d56a5f2cfe8cc14d1087c54837a8546d5640d3516`.
The downloaded CI artifact retains all 36 timing samples, original input/weight
audits, source fingerprints and clean device teardown. Its independently
recomputed `eligible_for_full_model_gate` is false. Fixed-packet dispatch alone
does not make this within-layer streaming design competitive; the native path
remains the control, with no serving-default change.

The current branch passes 1,123 CI tests, 58 simulator-harness tests, workflow YAML
parsing and shell syntax. These checks are not device evidence.

The goal remains 200 committed TG for one coding stream. Neither compiler size,
transport correctness nor a future single-layer timing result satisfies it.
