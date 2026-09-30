# Skill: fabric-relay-tuning

Run one repeatable tuning cycle of the fabric-relay weight-loading experiment
in this repo. Use this skill whenever the user asks to "run the next
fabric-relay phase", "start a fabric tuning cycle", or to evaluate evidence
against the fabric-relay gates.

## Context

- Plan: `docs/fabric-weight-loading-plan.md`
- Registry: `optimisation/fabric-relay/spec/phases.yaml` (single source of truth
  for phases, gates and metrics; validate with
  `python optimisation/fabric-relay/spec/validate_spec.py`)
- Procedure: `optimisation/fabric-relay/README.md`

## Steps

1. **Pick the phase.** The next open phase is the lowest-numbered registry
   entry without a committed record in `docs/`. Phase 3 experiments require
   P2.a and P2.b records first.
2. **Validate the registry.** Run
   `python optimisation/fabric-relay/spec/validate_spec.py`; stop and fix if it
   does not print `OK`.
3. **Prepare the cycle.** Submit
   `optimisation/fabric-relay/workflows/fabric-tuning-cycle.workflow.js` to the
   workflow tool with `{"phase": "<id>"}`. It produces the run plan, gate
   checklist, rig steps and a draft record via parallel subagents.
4. **Audit before hardware.** The workflow's audit step must end
   `VERDICT: CLEAN`. With VIOLATIONS, fix the plan and re-run step 3; never run
   hardware against a violating plan.
5. **Execute on the rig.** Hardware runs happen on the two-card rig, through
   the repo's normal CI/qualifier paths — never from this session. Use the
   plan's rig_steps; resolve devices by
   `/dev/tenstorrent/by-id/blackhole-<serial>` **after** any `tt-smi -r`
   (see `docs/gotchas.md`).
6. **Evaluate gates.** Collect evidence into JSON and run
   `python - <<'EOF'` style gate evaluation via `gates.evaluate_phase`, or use
   `report_gen.py --phase <id> --evidence evidence.json`. A gate that cannot be
   evaluated is a FAIL. Every gate result must appear in the record.
7. **Record.** Commit the generated record into `docs/` named
   `fabric-relay-<phase>-<short-desc>.md`, whether the candidate won or lost.
   Rejections are results (precedent: `docs/weight-read-packets-2026-09-10.md`,
   `docs/dram-projection-reload-2026-09-10.md`).

## Hard rules

- No serving-default changes, no opt-in flag promotion without the full gate
  ladder.
- Candidate phase timing: matched nine-block ABBA, control first, against the
  frozen winning controls; never against a degraded baseline.
- A layer/kernel win that loses the complete combined PP/CTX/TG loop is not
  promoted.
- Exact output and state digests on both chips are required for any change
  touching weight load or dispatch.
- Never claim TG from kernel-only or simulator-only timing.
