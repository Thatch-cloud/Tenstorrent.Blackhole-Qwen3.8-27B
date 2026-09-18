// Fabric-relay tuning cycle: prepares and audits one phase of the
// fabric-relay procedure using parallel subagents, then merges a draft
// experiment record for docs/. Hardware runs are NOT started here; the
// workflow produces the exact run plan, gate checklist and record skeleton
// a human or rig agent executes on the two-card rig.
//
// Input: { phase: "P0.1" | ... | "P3", evidencePath?: string }

phase("prepare");
log(`Preparing tuning cycle for phase ${args.phase}`);

const schema = {
  type: "object",
  properties: {
    phase: { type: "string" },
    tool_command: { type: "string" },
    gates: {
      type: "array",
      items: {
        type: "object",
        properties: {
          id: { type: "string" },
          rule: { type: "string" },
          required_evidence: { type: "string" },
          pass_condition: { type: "string" },
        },
        required: ["id", "rule", "required_evidence", "pass_condition"],
      },
    },
    rig_steps: { type: "array", items: { type: "string" } },
    run_id_fields: { type: "array", items: { type: "string" } },
    rejection_conditions: { type: "array", items: { type: "string" } },
  },
  required: ["phase", "tool_command", "gates", "rig_steps", "run_id_fields", "rejection_conditions"],
};

const plan = await agent(
  `You are preparing one phase of the fabric-relay tuning procedure in the repo ` +
  `Tenstorrent.Blackhole-Qwen3.8-27B. Read optimisation/fabric-relay/README.md, ` +
  `optimisation/fabric-relay/spec/phases.yaml and ` +
  `docs/fabric-weight-loading-plan.md. For phase "${args.phase}" produce: the exact ` +
  `tool command, the per-gate evidence requirements and pass conditions, ordered ` +
  `rig execution steps (respecting the ABBA nine-block convention where the phase ` +
  `is a candidate), the run-ID/artifact fields the record must cite, and explicit ` +
  `rejection conditions. Never claim serving-default changes are allowed. ` +
  `Return only the structured result.`,
  { label: "phase-plan", schema },
);
if (!plan) throw new Error("phase-plan agent failed");

phase("audit");
const audit = await agent(
  `Audit this prepared plan for the fabric-relay procedure against the repo's ` +
  `admission rules (docs/tuning-experiment-template.md acceptance checklist, ` +
  `docs/two-card-experiment-programme.md gates, docs/gotchas.md device-renumbering ` +
  `cabling rules). Plan JSON:\n${JSON.stringify(plan, null, 2)}\n` +
  `List every violation with the doc it violates and a corrected wording. If a ` +
  `step is fine, do not mention it. End with VERDICT: CLEAN or VERDICT: VIOLATIONS.`,
  { label: "rules-audit" },
);

phase("record");
const record = await agent(
  `Write a draft experiment record for phase ${args.phase} in the exact format of ` +
  `docs/tuning-experiment-template.md, filling every field from this plan and ` +
  `audit, using "Not measured" for anything not yet run and never omitting a ` +
  `gate. Plan:\n${JSON.stringify(plan, null, 2)}\nAudit findings:\n${audit}\n` +
  `Return only the markdown record.`,
  { label: "record-draft" },
);

return {
  phase: args.phase,
  plan,
  audit_verdict: audit,
  record_draft: record,
};
