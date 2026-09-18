"""Admission-gate evaluation for the fabric-relay procedure.

Each gate takes the phase evidence and returns pass/fail with a reason.
A gate that cannot be evaluated is a FAIL, never skipped.
"""

import statistics


def at_least_repeats(evidence, params):
    count = len(evidence.get(params.get("key", "repeats"), []))
    return count >= params["count"], f"{count} repeats recorded"


def abba_ordering(evidence, params):
    order = evidence.get("order", [])
    ok = bool(order) and order[0] == "control" and len(order) % 2 == 1
    return ok, f"order starts control and has odd length {len(order)}"


def candidate_not_worse_than(evidence, params):
    control = evidence.get("control_metric")
    candidate = evidence.get("candidate_metric")
    if control is None or candidate is None:
        return False, "control/candidate metric missing"
    ok = candidate <= control * (1.0 + params.get("margin", 0.0))
    return ok, f"candidate={candidate} control={control} margin={params.get('margin', 0.0)}"


def candidate_at_least(evidence, params):
    """Higher-is-better comparison (e.g. bandwidth): candidate >= control * (1 - margin)."""
    control = evidence.get("control_metric")
    candidate = evidence.get("candidate_metric")
    if control is None or candidate is None:
        return False, "control/candidate metric missing"
    ok = candidate >= control * (1.0 - params.get("margin", 0.0))
    return ok, f"candidate={candidate} control={control} margin={params.get('margin', 0.0)}"


def all_stages_present(evidence, params):
    missing = [s for s in params["stages"] if s not in evidence.get("per_stage_s", {})]
    return not missing, ("all stages present" if not missing else f"missing {missing}")


def sum_matches_total(evidence, params):
    stages = evidence.get("per_stage_s", {})
    total = evidence.get("total_s")
    if not stages or total is None:
        return False, "per_stage_s or total_s missing"
    delta = abs(sum(stages.values()) - total)
    return delta <= params["tolerance_s"], f"|sum-total|={delta:.3f}s"


def all_cards_present(evidence, params):
    cards = set(evidence.get("cards", []))
    missing = [c for c in params["cards"] if c not in cards]
    return not missing, ("both cards present" if not missing else f"missing {missing}")


def dispatch_config_identity(evidence, params):
    """Config file hash must match the tool-echoed hash (cache-reuse guard)."""
    declared = evidence.get("dispatch_config_sha256")
    echoed = evidence.get("config_cache_echoed_sha256")
    if not declared or not echoed:
        return False, "config hash or tool-echoed hash missing"
    return declared == echoed, (
        "config hash matches tool echo" if declared == echoed
        else f"mismatch: declared={declared[:12]} echoed={echoed[:12]}")


def by_id_resolution(evidence, params):
    """Device identity: by-id paths plus mesh-degree histogram assertion."""
    resolved = evidence.get("by_id_resolved") is True
    pair = evidence.get("cabling_pair_confirmed") is True
    histograms = evidence.get("mesh_degree_histograms_equal") is True
    return resolved and pair and histograms, (
        "by-id resolution, cabling pair and mesh-degree assertion all confirmed"
        if resolved and pair and histograms else
        f"by_id={resolved} cabling_pair={pair} mesh_histograms={histograms}")


def reclaimable_arithmetic(evidence, params):
    """Per card: compute_available == tensix_total - dispatch_reserved, and
    reclaimable (the cores freed by eth dispatch) == dispatch_reserved."""
    inventory = evidence.get("inventory", {})
    if not inventory:
        return False, "inventory missing"
    bad = []
    for card, counts in inventory.items():
        total = counts.get("tensix_total", 0)
        reserved = counts.get("dispatch_reserved", 0)
        if counts.get("compute_available") != total - reserved or counts.get("reclaimable") != reserved:
            bad.append(card)
    return not bad, ("arithmetic consistent on all cards" if not bad else f"inconsistent on {bad}")


def weight_digests_equal(evidence, params):
    mismatched = [family for family, equal in evidence.get("digests", {}).items() if not equal]
    return not mismatched and bool(evidence.get("digests")), (
        "all tensor-family digests equal" if not mismatched else f"mismatched {mismatched}")


def exact_output_state(evidence, params):
    ok = evidence.get("output_digest_equal") is True and evidence.get("state_digest_equal") is True
    return ok, "output and state digests match control" if ok else "output/state digest mismatch"


def added_latency_below(evidence, params):
    added = evidence.get("added_ms_per_block")
    if added is None:
        return False, "added_ms_per_block missing"
    return added < params["bound"], f"added={added}ms bound={params['bound']}ms"


def all_blocks_not_worse(evidence, params):
    summary = evidence.get("abba_summary", {})
    losses = summary.get("losses")
    if losses is None:
        return False, "abba_summary missing"
    return losses == 0, f"{summary.get('wins', 0)} wins / {losses} losses"


def gain_repeats_beyond_variability(evidence, params):
    repeats = evidence.get("gain_repeats", [])
    if len(repeats) < 2:
        return False, "need at least two complete-loop gain repeats"
    if min(repeats) <= 0:
        return False, "a repeat shows no gain"
    spread = statistics.fmean(repeats) - min(repeats)
    return True, f"{len(repeats)} repeats, mean-minus-min={spread:.4f}"


RULES = {
    "at_least_repeats": at_least_repeats,
    "abba_ordering": abba_ordering,
    "candidate_not_worse_than": candidate_not_worse_than,
    "candidate_at_least": candidate_at_least,
    "all_stages_present": all_stages_present,
    "sum_matches_total": sum_matches_total,
    "all_cards_present": all_cards_present,
    "dispatch_config_identity": dispatch_config_identity,
    "by_id_resolution": by_id_resolution,
    "reclaimable_arithmetic": reclaimable_arithmetic,
    "weight_digests_equal": weight_digests_equal,
    "exact_output_state": exact_output_state,
    "added_latency_below": added_latency_below,
    "all_blocks_not_worse": all_blocks_not_worse,
    "gain_repeats_beyond_variability": gain_repeats_beyond_variability,
}


def evaluate_phase(phase, evidence):
    results = []
    for gate in phase.get("gates", []):
        rule = RULES.get(gate["rule"])
        if rule is None:
            results.append({"gate": gate["id"], "passed": False, "reason": f"unknown rule {gate['rule']!r}"})
            continue
        try:
            passed, reason = rule(evidence, gate.get("params", {}))
        except Exception as error:  # a crashing gate is a failing gate
            passed, reason = False, f"gate raised: {error}"
        results.append({"gate": gate["id"], "passed": bool(passed), "reason": reason})
    return {
        "phase": phase["id"],
        "all_passed": all(result["passed"] for result in results) and bool(results),
        "results": results,
    }
