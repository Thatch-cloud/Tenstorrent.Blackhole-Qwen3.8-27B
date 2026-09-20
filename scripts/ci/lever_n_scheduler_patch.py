"""Lever N M2: stage the alternation + one-in-flight edits against the pinned plugin
scheduler sources.

Both real sources were unknown to the repo until CPU probe run 35535280599
(scripts/ci/probe_plugin_scheduler_sources.py) dumped, from
/opt/qwen-fast-plugin/src/vllm_tt_plugin: scheduler.py in full (TTScheduler,
TTSchedulingMode) and lane_scheduler.py in full (TTLaneCoordinator - the class the
design docs call "LaneScheduler"; that name does not exist in the plugin, the real
class is TTLaneCoordinator and this module patches it under its real name). Every
edit here is scoped to a named method's line range and asserts it matched exactly
once, so a source change fails the patch loudly instead of silently editing the
wrong method - the same discipline lever_n_model_patch.py uses against model.py's
ambiguous anchors.

What the edits do, per docs/lever-n-plugin-contract-2026-09-19.md's M2 scoping:

- patch_negotiate_forced_mode (TTLaneCoordinator._negotiate_forced_mode):
  ALTERNATION. Today `intent = max(self._local_prefill_intent(sched) for sched in
  self.lanes)` and _local_prefill_intent returns 1 whenever a lane has a partial
  prefill in flight, so any lane's continuation forces PREFILL_ONLY for the whole
  step until the prompt finishes - chunked prefill on its own does not stop decode
  from starving, it just starves it in smaller increments. This patch forces
  DECODE_ONLY on every other qualifying step instead: a step qualifies only when
  the vote is live because of a partial prefill (not a fresh admission) AND some
  lane also has a running non-chunk decode waiting on progress. It changes nothing
  else: _local_prefill_intent, _schedule_all_lanes and TTLaneCoordinator.schedule
  (the PREFILL_ONLY -> DECODE_ONLY fallback that carries finished_req_ids,
  free_encoder_mm_hashes and preempted_req_ids across a discarded pass) are
  untouched, so a step alternation forces to DECODE_ONLY is indistinguishable, to
  every downstream consumer, from one _negotiate_forced_mode would have returned
  on its own - the fallback's bookkeeping keeps working because its input is the
  only thing that changed, not its own logic.

- patch_one_in_flight (TTScheduler._schedule_prefill_only): ONE IN FLIGHT. Hides
  self.waiting and self.skipped_waiting while a partial prefill is in flight, so a
  fresh prompt cannot be admitted between two continuation steps of one already in
  progress and corrupt the model's single-occupancy GDN scratch and host RoPE
  table (lever_n_model_patch.SLOTS_RANGE's docstring already assumes exactly this
  is enforced upstream: "the scheduler enforces that; this method assumes it").
  Copies _schedule_decode_only's own save/hide/restore idiom verbatim
  (create_request_queue(self.policy) to build the empty stand-in,
  prepend_requests to merge anything a mid-schedule preemption pushed into it back
  onto the original queue) rather than swapping in a raw list or None:
  self.waiting is a RequestQueue built by create_request_queue, not a list, and
  _schedule_decode_only already shows the only mechanism in this module that
  empties and restores one without breaking that type.

Applying nothing on import: stage() is explicit, mirroring lever_n_model_patch and
serving_plugin_patch.
"""

import argparse
import ast
from pathlib import Path

NEGOTIATE_FUNCTION = '_negotiate_forced_mode'
PREFILL_ONLY_FUNCTION = '_schedule_prefill_only'

# 1-in-2: every other qualifying step is forced to decode. This is the tightest
# ratio that bounds the worst-case decode gap by exactly one prefill-chunk step's
# duration, independent of how many chunks the prompt has left - the M1 gate
# measured chunking a 5,918-token prompt at 4.43 s across three 2048-token chunk
# steps (~1.5 s/step), so 1-in-2 alternation bounds a decode's worst gap near that
# one-step figure instead of near the whole remaining prefill (which, absent M2,
# is what "chunked prefill alone still starves decode" in the docs means - it
# would have been ~3 s here and grows without bound for a longer prompt). A
# sparser ratio (1-in-3, 1-in-4, ...) only widens that bound for no clear benefit
# at this chunk size; there is no ratio denser than 1-in-2. Kept as a single named
# constant so the M2 gate's ITL measurement can justify retuning it later without
# touching the scoping logic around it.
ALTERNATION_PERIOD = 2


def function_span(source, name):
    """Line span [start, end) of a method, by AST sibling order.

    Mirrors lever_n_model_patch.function_span: end_lineno needs Python 3.8, and
    this has to run under whatever interpreter the graft step and the image agree
    on, so the end is taken as the next sibling definition's start (or the end of
    the class body) rather than from the node.
    """
    tree = ast.parse(source)
    total = len(source.splitlines())
    for body in tree.body:
        if not isinstance(body, ast.ClassDef):
            continue
        members = [node for node in body.body
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for index, node in enumerate(members):
            if node.name != name:
                continue
            if node.decorator_list:
                start = min(d.lineno for d in node.decorator_list) - 1
            else:
                start = node.lineno - 1
            if index + 1 < len(members):
                following = members[index + 1]
                end = (min(d.lineno for d in following.decorator_list)
                       if following.decorator_list else following.lineno) - 1
            else:
                end = total
            return start, end
    raise ValueError('no method named %s' % name)


def replace_once(lines, span, old, new, what):
    """Replace old with new inside a line span, requiring exactly one occurrence."""
    start, end = span
    region = ''.join(lines[start:end])
    if region.count(old) != 1:
        raise ValueError('%s: expected one occurrence of %r in %s, found %d'
                         % (what, old[:60], span, region.count(old)))
    lines[start:end] = (region.replace(old, new, 1)).splitlines(keepends=True)
    return lines


# ---------------------------------------------------------------------------
# 1a. lane_scheduler.py :: TTLaneCoordinator._negotiate_forced_mode -- alternation
# ---------------------------------------------------------------------------

ALTERNATION_OLD = '''        intent = max(self._local_prefill_intent(sched) for sched in self.lanes)
        return TTSchedulingMode.from_prefill_intent(intent)
'''

ALTERNATION_NEW = f'''        intent = max(self._local_prefill_intent(sched) for sched in self.lanes)
        if intent == 1:
            # Lever N M2: a partial prefill in flight always votes prefill
            # (_local_prefill_intent), and any lane voting prefill wins the
            # whole step, so a chunked prefill alone still starves every other
            # lane's decode until it finishes (docs/lever-n-plugin-contract).
            # Yield a decode-only step between chunks, but only when that is
            # actually the risk: some lane has a partial prefill in flight AND
            # some lane has a running decode waiting on progress. A vote of 1
            # from queued-but-not-partial work (has_waiting and capacity - the
            # one-shot fast-path admission case) is left alone: alternating
            # there would only delay admission, not rescue a decode already
            # in flight, since there is no in-flight continuation to interrupt.
            has_partial_prefill = any(
                any(r.is_prefill_chunk for r in sched.running)
                for sched in self.lanes
            )
            has_running_decode = any(
                any(not r.is_prefill_chunk for r in sched.running)
                for sched in self.lanes
            )
            if has_partial_prefill and has_running_decode:
                step = getattr(self, "_m2_alternation_step", 0)
                self._m2_alternation_step = step + 1
                if step % {ALTERNATION_PERIOD} == {ALTERNATION_PERIOD - 1}:
                    logger.info(
                        "[M2] alternation: forcing decode-only step "
                        "during a partial prefill"
                    )
                    return TTSchedulingMode.DECODE_ONLY
        return TTSchedulingMode.from_prefill_intent(intent)
'''


def patch_negotiate_forced_mode(source):
    """TTLaneCoordinator._negotiate_forced_mode: alternate between chunk steps.

    Reads only self.lanes and each lane's .running - the same surface
    _local_prefill_intent already reads - so it needs no new coordinator state
    beyond one counter, read defensively (getattr(..., 0)) so __init__ needs no
    edit. Does not touch _schedule_all_lanes or schedule(): the PREFILL_ONLY ->
    DECODE_ONLY fallback and its finished_req_ids / free_encoder_mm_hashes /
    preempted_req_ids carry are the negotiation's caller, not its concern, and
    stay exactly as pinned.
    """
    lines = source.splitlines(keepends=True)
    span = function_span(source, NEGOTIATE_FUNCTION)
    lines = replace_once(lines, span, ALTERNATION_OLD, ALTERNATION_NEW,
                         'negotiate_forced_mode alternation')
    result = ''.join(lines)
    ast.parse(result)
    return result


# ---------------------------------------------------------------------------
# 1b. scheduler.py :: TTScheduler._schedule_prefill_only -- one prefill in flight
# ---------------------------------------------------------------------------

ONE_IN_FLIGHT_OLD = '''        self.max_num_running_reqs = max(0, saved_max - len(pure_decodes))
        try:
            result = super().schedule()
        finally:
            self.running.extend(pure_decodes)
            self.max_num_running_reqs = saved_max
        return result
'''

ONE_IN_FLIGHT_NEW = '''        self.max_num_running_reqs = max(0, saved_max - len(pure_decodes))
        # Lever N M2: one prefill in flight per lane. A partial prefill already
        # occupies the model's single-occupancy GDN scratch and host RoPE table
        # (lever_n_model_patch.SLOTS_RANGE assumes exactly this is enforced
        # here); admitting a fresh prompt's first chunk between two of its
        # continuation steps would corrupt both. Hide the waiting queues too,
        # only while a continuation is actually in flight, with the same
        # save/hide/restore idiom _schedule_decode_only already uses for them:
        # self.waiting is a RequestQueue built by create_request_queue, not a
        # list, so that factory is the correct empty value, not [] or None.
        saved_waiting = self.waiting
        saved_skipped = getattr(self, "skipped_waiting", None)
        if partial_prefills:
            self.waiting = create_request_queue(self.policy)
            if saved_skipped is not None:
                self.skipped_waiting = create_request_queue(self.policy)
        try:
            result = super().schedule()
        finally:
            self.running.extend(pure_decodes)
            self.max_num_running_reqs = saved_max
            if partial_prefills:
                if self.waiting:
                    saved_waiting.prepend_requests(self.waiting)
                if saved_skipped is not None:
                    if self.skipped_waiting:
                        saved_skipped.prepend_requests(self.skipped_waiting)
                    self.skipped_waiting = saved_skipped
                self.waiting = saved_waiting
        return result
'''


def _imports_name(source, name):
    """Whether an import statement (not merely a usage) binds this name."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if (alias.asname or alias.name) == name:
                    return True
    return False


def patch_one_in_flight(source):
    """TTScheduler._schedule_prefill_only: hide waiting/skipped_waiting mid-continuation.

    create_request_queue is already imported at module level in scheduler.py (it
    is what _schedule_decode_only itself calls), so this patch adds no import;
    the check below just makes that assumption loud instead of a silent NameError
    on the grafted image if the pinned source ever drops it. Checked via the AST
    import graph, not a text search, so a usage elsewhere in the file (e.g. inside
    _schedule_decode_only, which this patch does not touch) cannot stand in for
    the import actually being present.
    """
    if not _imports_name(source, 'create_request_queue'):
        raise ValueError('scheduler.py must already import create_request_queue '
                         '(used by _schedule_decode_only) for this patch to compile')
    lines = source.splitlines(keepends=True)
    span = function_span(source, PREFILL_ONLY_FUNCTION)
    lines = replace_once(lines, span, ONE_IN_FLIGHT_OLD, ONE_IN_FLIGHT_NEW,
                         'one-in-flight hide waiting')
    result = ''.join(lines)
    ast.parse(result)
    return result


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------

def patch_lane_scheduler(source):
    return patch_negotiate_forced_mode(source)


def patch_scheduler(source):
    return patch_one_in_flight(source)


def stage(lane_scheduler_path, scheduler_path,
          lane_scheduler_output=None, scheduler_output=None):
    lane_scheduler_path = Path(lane_scheduler_path)
    scheduler_path = Path(scheduler_path)
    lane_source = lane_scheduler_path.read_text(encoding='utf-8')
    sched_source = scheduler_path.read_text(encoding='utf-8')
    lane_patched = patch_lane_scheduler(lane_source)
    sched_patched = patch_scheduler(sched_source)
    lane_target = Path(lane_scheduler_output) if lane_scheduler_output else lane_scheduler_path
    sched_target = Path(scheduler_output) if scheduler_output else scheduler_path
    lane_target.write_text(lane_patched, encoding='utf-8', newline='\n')
    sched_target.write_text(sched_patched, encoding='utf-8', newline='\n')
    return lane_target, sched_target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('lane_scheduler_path')
    parser.add_argument('scheduler_path')
    parser.add_argument('--lane-scheduler-output')
    parser.add_argument('--scheduler-output')
    options = parser.parse_args()
    lane_target, sched_target = stage(options.lane_scheduler_path, options.scheduler_path,
                                      options.lane_scheduler_output, options.scheduler_output)
    print('wrote %s' % lane_target)
    print('wrote %s' % sched_target)
