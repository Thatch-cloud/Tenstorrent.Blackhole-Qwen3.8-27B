# Fabric-relay weight loading: repeatable tuning procedure

Turns [the plan](../../docs/fabric-weight-loading-plan.md) into a repeatable
procedure. Each phase is a registered experiment spec, run through the same
four steps: measure → gate → report → promote/reject decision recorded in
`docs/`. No serving defaults change; everything is opt-in.

## Layout

| Path | Purpose |
| --- | --- |
| `spec/phases.yaml` | Machine-readable registry of every phase-0..3 experiment: variables, gates, metrics, rejection conditions |
| `spec/validate_spec.py` | Validates the registry against the plan's invariants (host tests, no hardware) |
| `fabric_bench.py` | P0.1/P0.2: PCIe and fabric bandwidth microbenchmarks (runs on the rig host) |
| `core_inventory.py` | P0.4: per-card logical core inventory under a dispatch config; counts reclaimable card-1 cores |
| `load_timeline.py` | P0.3: parses loader instrumentation events into the per-stage load timeline table |
| `abba.py` | Matched ABBA block scheduler/evaluator for layer-level candidates (control vs candidate) |
| `gates.py` | Admission gate checks: weight digest equality, complete-loop dominance, variability repeat |
| `report_gen.py` | Emits a `docs/tuning-experiment-template.md` record from spec + measured evidence |
| `test_*.py` | Host-only tests for every module, matching `optimisation/rig` style |
| `workflows/fabric-tuning-cycle.workflow.js` | Multi-agent orchestration: run one full phase cycle end to end |

## Procedure (one cycle)

1. Pick the next open phase from `spec/phases.yaml` (lowest numbered without a
   recorded decision).
2. Run its measurement tool (`fabric_bench`, `core_inventory`,
   `load_timeline`, or the runtime harness for phases 1–3) on the rig; keep
   run IDs and artifact SHA256s.
3. Evaluate gates with `gates.py`; every gate is pass/fail with evidence — no
   gate is ever silently skipped.
4. Generate the record with `report_gen.py` into `docs/`; commit it whether
   the candidate won or lost. Rejections are results too (see
   `weight-read-packets`, `dram-projection-reload`).
5. Only after the phase's promotion gate passes may the next phase start.

## Non-negotiables (inherited)

- Matched ABBA, nine blocks, frozen winning controls; never compare against a
  degraded baseline.
- Complete-loop PP/CTX/TG decides TG claims; layer timings never do.
- Exact output/state audits on both chips for any change touching load or
  dispatch.
- Simulator/trace-replay first, then hardware replay, then combined model.
