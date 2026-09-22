"""Lever N M1 step 2: stage the resumable-prefill edits against the pinned model source.

model.py is 174 KB and the anchors this needs are ambiguous globally - there are four
calls to _reset_gdn_state_for_new_sequence and three "for c in range(num_full)" loops.
Every edit here is therefore scoped to a named function's line range and asserts it
matched exactly once, so a source change fails the patch loudly instead of silently
editing the wrong loop.

What the edits do, per docs/lever-N-prefill-decode-interleave.md section 3.1:

- _prefill_traced_chunked_tp gains chunk_from / chunk_to / do_reset / do_tail. The
  GDN reset happens only on the first step, the replay covers a chunk range rather
  than always range(num_full), and the tail plus logits happen only on the last.
- prefill_traced_chunked gains start and derives that range.
- prefill_paged_slots_range drives one step across N requests, writing a decode slot
  only for the requests finishing on this step.

Applying nothing on import: stage() is explicit, mirroring serving_plugin_patch.
"""

import argparse
import ast
from pathlib import Path

TP_FUNCTION = '_prefill_traced_chunked_tp'
# The one serving actually runs. prefill_traced_chunked falls back to it whenever no
# chunk trace is captured, which is every serving run (trace_mode='decode_only').
EAGER_FUNCTION = '_prefill_chunked_eager_tp'
CHUNKED_FUNCTION = 'prefill_traced_chunked'
SLOTS_FUNCTION = 'prefill_paged_slots'


def function_span(source, name):
    """Line span [start, end) of a method, by AST sibling order.

    end_lineno needs Python 3.8; this has to run under 3.7 locally as well as 3.10 in
    the image, so the end is taken as the next sibling definition's start (or the end
    of the class body) rather than from the node.
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


def _patch_replay_loop(source, function, signature, reset_comment, label):
    """Make one chunk loop resumable: a chunk range, a conditional reset and tail.

    Shared by both members of the family. _prefill_traced_chunked_tp replays a captured
    trace; _prefill_chunked_eager_tp is the fallback the entry takes when there is none,
    which is every serving run. Patching one and not the other is how run 35693338281
    replayed chunk zero on a continuation.
    """
    lines = source.splitlines(keepends=True)
    span = function_span(source, function)
    lines = replace_once(
        lines, span, signature,
        signature + ',\n        chunk_from=0, chunk_to=None, do_reset=True, do_tail=True',
        '%s signature' % label)

    span = function_span(''.join(lines), function)
    lines = replace_once(
        lines, span,
        reset_comment + '\n        self._reset_gdn_state_for_new_sequence()',
        '        # Re-zero GDN on the first step only; state carries across steps, replays and tail.\n'
        '        chunk_to = num_full if chunk_to is None else int(chunk_to)\n'
        '        if do_reset:\n'
        '            self._reset_gdn_state_for_new_sequence()',
        '%s reset' % label)

    span = function_span(''.join(lines), function)
    lines = replace_once(lines, span, '        for c in range(num_full):',
                         '        for c in range(chunk_from, chunk_to):', '%s chunk loop' % label)

    span = function_span(''.join(lines), function)
    lines = replace_once(lines, span, '        if tail_real > 0:',
                         '        if do_tail and tail_real > 0:', '%s tail' % label)
    return ''.join(lines)


def patch_tp_replay(source):
    """Both chunk loops, the traced one and the eager fallback serving runs."""
    source = _patch_replay_loop(
        source, TP_FUNCTION,
        'self, token_ids, page_table, actual_len, num_full, chunk_size, tail_real, vision_tokens=None',
        '        # Re-zero GDN once; carries across replays + tail (chunk_start>0 skips reset).',
        'tp')
    return _patch_replay_loop(
        source, EAGER_FUNCTION,
        'self, token_ids, page_table, actual_len, num_full, chunk_size, tail_real, flex_sdpa=True, vision_tokens=None',
        '        # Re-zero GDN at sequence start; tail (chunk_start>0) keeps carried state.',
        'eager')


def patch_chunked_entry(source):
    """Thread start through the chunked entry point."""
    lines = source.splitlines(keepends=True)
    span = function_span(source, CHUNKED_FUNCTION)
    lines = replace_once(
        lines, span,
        'def prefill_traced_chunked(self, token_ids, page_table, actual_len, vision_tokens=None):',
        'def prefill_traced_chunked(self, token_ids, page_table, actual_len, vision_tokens=None,\n'
        '                               start=0):',
        'chunked signature')

    span = function_span(''.join(lines), CHUNKED_FUNCTION)
    lines = replace_once(
        lines, span,
        '        self._build_request_rope(token_ids[:, :actual_len], vision_tokens)',
        '        # Section 3.1: the RoPE table is staged once, on the first step, for the whole\n'
        '        # prompt; later steps slice the same sequence-indexed table by chunk position.\n'
        '        if start == 0:\n'
        '            self._build_request_rope(token_ids[:, :actual_len], vision_tokens)',
        'chunked rope')

    span = function_span(''.join(lines), CHUNKED_FUNCTION)
    lines = replace_once(
        lines, span,
        '                return self._prefill_traced_chunked_tp(\n'
        '                    token_ids, page_table, actual_len, num_full, chunk_size, tail_real, vision_tokens=vision_tokens\n'
        '                )',
        '                assert start % chunk_size == 0, (\n'
        '                    "resumable prefill must continue on a chunk boundary: "\n'
        '                    f"start={start} chunk_size={chunk_size}"\n'
        '                )\n'
        '                return self._prefill_traced_chunked_tp(\n'
        '                    token_ids, page_table, actual_len, num_full, chunk_size, tail_real,\n'
        '                    vision_tokens=vision_tokens, chunk_from=start // chunk_size,\n'
        '                    chunk_to=num_full, do_reset=(start == 0),\n'
        '                )',
        'chunked tp call')

    # The eager fallback, which is the branch every serving run takes: the traced call
    # above needs a captured chunk trace and trace_mode is 'decode_only'. Run
    # 35693338281 threaded the range into the traced call only, so the continuation
    # reached this one and replayed chunk zero.
    span = function_span(''.join(lines), CHUNKED_FUNCTION)
    lines = replace_once(
        lines, span,
        '            return self._prefill_chunked_eager_tp(\n'
        '                token_ids,\n'
        '                page_table,\n'
        '                actual_len,\n'
        '                num_full,\n'
        '                chunk_size,\n'
        '                tail_real,\n'
        '                flex_sdpa=True,\n'
        '                vision_tokens=vision_tokens,\n'
        '            )',
        '            assert start % chunk_size == 0, (\n'
        '                "resumable prefill must continue on a chunk boundary: "\n'
        '                f"start={start} chunk_size={chunk_size}"\n'
        '            )\n'
        '            return self._prefill_chunked_eager_tp(\n'
        '                token_ids,\n'
        '                page_table,\n'
        '                actual_len,\n'
        '                num_full,\n'
        '                chunk_size,\n'
        '                tail_real,\n'
        '                flex_sdpa=True,\n'
        '                vision_tokens=vision_tokens,\n'
        '                chunk_from=start // chunk_size,\n'
        '                chunk_to=num_full,\n'
        '                do_reset=(start == 0),\n'
        '            )',
        'chunked eager call')
    return ''.join(lines)


SLOTS_RANGE = '''
    def prefill_paged_slots_range(self, token_ids_list, page_table, empty_slots, starts, ends,
                                  valid_lens=None):
        """One resumable prefill step across N requests (Lever N M1, design section 3.1).

        The per-step analogue of prefill_paged_slots. Each request advances through its
        own [start, end) token window, where end is the chunk end the runner scheduled.

        No is_last: design section 3.1 assumed the runner would pass one, and it does not.
        It is not needed. The tail runs iff tail_real > 0, which is only true on a final
        chunk that does not land on a chunk boundary, and the runner already discards the
        logits of a mid-prompt row itself - model_runner zeroes next_token_ids when
        intermediate_prefill_mask covers the rows. So every step returns its logits and
        writes its slot, and the final step's values are the ones that survive.

        One in-flight prefill per lane in v1: the GDN scratch and the host RoPE table are
        single-occupancy, so a second request's prefill between two chunk steps of the
        first would corrupt it. The scheduler is meant to enforce that. This method no
        longer merely assumes it - see the two checks below and what they do NOT cover.
        """
        assert self.num_devices > 1, "prefill_paged_slots_range is the TP (num_devices>1) path"
        N = len(token_ids_list)
        assert len(empty_slots) == N and len(starts) == N and len(ends) == N, (
            "one slot, start and end per request"
        )
        # The scratch is single-occupancy and nothing downstream notices a violation.
        # docs/lever-n-build-plan-2026-09-22.md section 3: two of the three guards the
        # design cited are not guards (one is a constructor check that reads no address),
        # and the corruption mode this method introduces is structurally invisible to all
        # of them. A wrong resumption therefore produces wrong tokens, not an error. So
        # the invariant is asserted here, at the point of use.
        resumed = [u for u in range(N) if int(starts[u]) > 0]
        if resumed and N > 1:
            # Fatal within one call: every request runs through the same scratch in the
            # loop below, and a sibling either re-zeroes it (do_reset on its start == 0)
            # or advances it with its own tokens. The resumed request would then continue
            # from another prompt's recurrence.
            raise ValueError(
                'Lever N: a resumed prefill cannot share a call with another request; '
                'starts=%r with N=%d' % ([int(s) for s in starts], N))
        # Chunk sequencing for the in-flight prompt. A continuation must arrive at exactly
        # the offset the previous step left off; a fresh prompt (start == 0) restarts it.
        # This catches a wrong-offset resume and an out-of-order chunk. It does NOT catch
        # a fresh prompt admitted BETWEEN two continuations of another - that resets the
        # cursor legitimately, and only the scheduler can prevent it (build plan step 8,
        # capping prefill capacity at partials rather than partials + 1).
        expected = getattr(self, '_qwen_lever_n_next_start', None)
        if resumed:
            start_u = int(starts[0])
            if expected is None or start_u != expected:
                raise ValueError(
                    'Lever N: prefill resumed at %d but the scratch was left at %r; '
                    'a chunk was skipped, replayed, or belongs to another prompt'
                    % (start_u, expected))
        pt = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        assert pt.shape[0] == N, "page_table must have one row per request"
        comp = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        dn_states = [layer.attention for layer in self.layers if not layer.is_full_attention]

        prev = self._bind_gdn_prefill_scratch()
        host_logits = [None] * N
        finished = []
        try:
            for u in range(N):
                toks = token_ids_list[u]
                assert toks.shape[0] == 1, f"request {u}: token_ids must be [1, T_u]"
                actual = int(valid_lens[u]) if valid_lens is not None else toks.shape[1]
                assert actual >= 1, f"request {u}: empty prompt (actual_len={actual})"
                start = int(starts[u])
                lg = self.prefill_traced_chunked(
                    toks[:, :actual], pt[u : u + 1], actual_len=actual, start=start
                )
                host_logits[u] = (
                    ttnn.to_torch(lg, mesh_composer=comp)
                    .reshape(-1, self.args.vocab_size)[:1]
                    .float()
                    .view(1, 1, -1)
                )
                ttnn.deallocate(lg)
                # actual is valid_lens[u], which is the ABSOLUTE end of the range
                # just covered - not the chunk length. Adding start double-counts:
                # run 35694645353 covered [2048, 4096) and left the cursor at
                # 2048 + 4096 = 6144, so the next chunk at 4096 was refused. The
                # one-shot branch already writes plens[0], the same absolute end,
                # and these two have to agree.
                self._qwen_lever_n_next_start = actual
                finished.append(
                    (
                        u,
                        [ttnn.to_torch(dn.rec_state, mesh_composer=comp) for dn in dn_states],
                        [[ttnn.to_torch(c, mesh_composer=comp) for c in dn.conv_states] for dn in dn_states],
                    )
                )
        finally:
            self._unbind_gdn_prefill_scratch(prev)

        for u, rec_snap, conv_snap in finished:
            self._write_gdn_slot(int(empty_slots[u]), rec_snap, conv_snap)
        return host_logits
'''


def add_slots_range(source):
    """Insert prefill_paged_slots_range immediately after prefill_paged_slots."""
    if 'def prefill_paged_slots_range(' in source:
        raise ValueError('prefill_paged_slots_range already present')
    start, end = function_span(source, SLOTS_FUNCTION)
    lines = source.splitlines(keepends=True)
    lines[end:end] = [SLOTS_RANGE]
    return ''.join(lines)


VLLM_ENTRY = 'prefill_forward'
VLLM_BATCHED = '_prefill_forward_tp_batched'


def patch_vllm_entry(source):
    """Thread start_pos from the runner down to the batched prefill.

    vLLM hands the model a chunk window per step once chunked prefill is on. The entry
    passes it through unchanged; the batched path turns it into per-request starts and
    ends and calls prefill_paged_slots_range instead of prefill_paged_slots.

    Both branches log which path ran. If the plugin never supplies start_pos the else
    branch is dead, the resumable arm silently serves the one-shot path, and an equality
    gate comparing the two arms passes while testing nothing. The marker is the positive
    control that rules that out.
    """
    lines = source.splitlines(keepends=True)
    span = function_span(source, VLLM_ENTRY)
    lines = replace_once(
        lines, span,
        'return self._prefill_forward_tp_batched(model, tokens, page_table, prompt_lens, kwargs.get("empty_slots"))',
        'return self._prefill_forward_tp_batched(\n'
        '                model, tokens, page_table, prompt_lens, kwargs.get("empty_slots"),\n'
        '                start_pos=kwargs.get("start_pos"),\n'
        '            )',
        'vllm entry call')

    span = function_span(''.join(lines), VLLM_BATCHED)
    lines = replace_once(
        lines, span,
        'def _prefill_forward_tp_batched(self, model, tokens, page_table, prompt_lens, empty_slots):',
        'def _prefill_forward_tp_batched(self, model, tokens, page_table, prompt_lens, empty_slots,\n'
        '                                    start_pos=None):',
        'batched signature')

    span = function_span(''.join(lines), VLLM_BATCHED)
    lines = replace_once(
        lines, span,
        '        host_logits = model.prefill_paged_slots(token_ids_list, pt, empty_slots, valid_lens=plens)',
        '        starts = [] if start_pos is None else [int(s) for s in start_pos]\n'
        '        if not any(s > 0 for s in starts):\n'
        '            logger.info("[M1] prefill path: one-shot prefill_paged_slots")\n'
        '            host_logits = model.prefill_paged_slots(token_ids_list, pt, empty_slots, valid_lens=plens)\n'
        '            if starts:\n'
        '                # Where the range path has to pick up. Chunk one runs here by\n'
        '                # design and prefill_paged_slots never touches the cursor, so\n'
        '                # without this the FIRST continuation sees None and refuses -\n'
        '                # run 35692388798, resumed at 2048 with the scratch at None.\n'
        '                # plens[0] is the end of the range just covered, and on an\n'
        '                # unchunked prompt no continuation ever reads it.\n'
        '                model._qwen_lever_n_next_start = int(plens[0])\n'
        '        else:\n'
        '            # A continuation. model_runner.submit_prefill always supplies start_pos,\n'
        '            # so its presence says nothing; a nonzero start is what marks a resumed\n'
        '            # prompt. The first chunk stays on the one-shot path, where it is\n'
        '            # equivalent: its end is chunk-aligned so tail_real is 0 and no tail runs.\n'
        '            logger.info(\n'
        '                f"[M1] prefill path: resumable prefill_paged_slots_range "\n'
        '                f"starts={starts} ends={list(plens)}"\n'
        '            )\n'
        '            host_logits = model.prefill_paged_slots_range(\n'
        '                token_ids_list, pt, empty_slots, starts, plens, valid_lens=plens\n'
        '            )',
        'batched dispatch')
    return ''.join(lines)


# Logged from INSIDE the grafted method. install()'s marker only ever proved a
# config string had been written - run 35690327326 printed it and the policy had
# never run. This one cannot be emitted unless the edited code executed.
MARKER_SCHEDULER = '[PINDIAG] m2 one-in-flight:'
MARKER_ALTERNATE = '[PINDIAG] m2 alternation:'
# The shipped prefer-prefill branch of TTScheduler.schedule, and its replacement.
# Verified unique and byte-identical in fixtures/plugin_scheduler.py and in the real
# plugin file dumped by graft run 35707860782.
_ALTERNATION_ANCHOR = '        # Default mode:\n        # Prefer prefill whenever prefill work is pending, so new requests are\n        # admitted and partial prefills advance.\n        if has_pending_prefill:\n            prefill_result = self._schedule_prefill_only()\n'
_ALTERNATION_REPLACEMENT = '        # Default mode:\n        # Prefer prefill whenever prefill work is pending, so new requests are\n        # admitted and partial prefills advance.\n        if has_pending_prefill:\n            # Lever N M2 item 1, design section 3.3, in the class that actually\n            # runs. The design named TTLaneCoordinator._negotiate_forced_mode; run\n            # 35707860782 mounted that graft and it never fired, because\n            # check_and_update_config only selects TTLaneCoordinator when\n            # uses_tt_lane_coordinator() is true and this deployment logs\n            # data_parallel_size=1 and loads vllm_tt_plugin.scheduler.TTScheduler.\n            # With no coordinator, set_forced_mode is never called, _forced_mode\n            # stays DEFAULT, and THIS branch is the prefer-prefill rule that gives\n            # decode a step only when prefill schedules zero tokens.\n            #\n            # The trigger is a partial prefill in flight, NOT has_pending_prefill:\n            # that is also true of a fresh waiting prompt, and yielding there would\n            # delay admission and worsen the TTFT staircase. Only an in-flight\n            # chunked prefill freezes a decoding user, so only that alternates.\n            import os as _qwen_os\n            _qwen_on = _qwen_os.environ.get(\n                "TT_PREFILL_DECODE_INTERLEAVE", "1") == "1"\n            _qwen_partial = any(r.is_prefill_chunk for r in self.running)\n            _qwen_streak = getattr(self, "_qwen_prefill_streak", 0)\n            _qwen_credit = getattr(self, "_qwen_decode_credit", 0)\n            if _qwen_on and _qwen_partial and has_running_decode:\n                try:\n                    _qwen_r = int(_qwen_os.environ.get(\n                        "TT_DECODE_STEPS_PER_PREFILL_CHUNK", "1"))\n                except ValueError:\n                    _qwen_r = 1\n                _qwen_yield = _qwen_streak >= 1 and _qwen_credit < _qwen_r\n                if _qwen_yield:\n                    self._qwen_decode_credit = _qwen_credit + 1\n                else:\n                    self._qwen_decode_credit = 0\n                    self._qwen_prefill_streak = _qwen_streak + 1\n                logger.info(\n                    f"[PINDIAG] m2 alternation: yield_decode={_qwen_yield} "\n                    f"streak={_qwen_streak} credit={_qwen_credit} r={_qwen_r}")\n                if _qwen_yield:\n                    result = self._schedule_decode_only()\n                    return self._finalize_scheduler_output(result)\n            else:\n                self._qwen_prefill_streak = 1\n                self._qwen_decode_credit = 0\n            prefill_result = self._schedule_prefill_only()\n'


PLATFORM_OPT_IN = """def _m1_chunked_prefill_opt_in():
    \"\"\"Lever N M1: allow chunked prefill for a generator the graft made resumable.

    _CHUNKED_PREFILL_MODEL_TYPES lists the HF model_type values whose tt-metal
    generator accepts a chunk_start_idx. The Qwen3.8-27B generator does once
    model.py and qwen36_vllm.py carry prefill_paged_slots_range, which is exactly
    what this graft adds, so the set is opened by explicit opt-in rather than by
    guessing the model_type string.

    ONLY set TT_M1_FORCE_CHUNKED_PREFILL alongside that graft. On a stock image the
    scheduler would hand the model a chunk it cannot resume, and the prompt would be
    silently re-prefilled from zero on every step.
    \"\"\"
    return os.environ.get("TT_M1_FORCE_CHUNKED_PREFILL") == "1"


"""


def patch_platform(source):
    """Let the chunked-prefill policy opt in, for the grafted generator only.

    platform._apply_chunked_prefill_policy turns enable_chunked_prefill off for every
    model_type outside a two-entry allowlist, which is why runs 35415521079 and
    35415811328 never chunked a 5,918-token prompt despite being asked to: the flag was
    overridden at config time and max_num_batched_tokens was bumped back to max_model_len.
    """
    if '_m1_chunked_prefill_opt_in' in source:
        raise ValueError('platform already carries the M1 chunked-prefill opt-in')
    lines = source.splitlines(keepends=True)
    anchor = 'def _apply_chunked_prefill_policy(vllm_config: "VllmConfig") -> None:'
    hits = [i for i, line in enumerate(lines) if line.startswith(anchor)]
    if len(hits) != 1:
        raise ValueError('platform: expected one _apply_chunked_prefill_policy, got %d'
                         % len(hits))
    # Splice as individual lines: a single multi-line element would leave list
    # indices out of step with the line numbers function_span_module returns.
    block = PLATFORM_OPT_IN.lstrip(chr(10)).splitlines(keepends=True)
    lines[hits[0]:hits[0]] = block
    span = function_span_module(''.join(lines), '_apply_chunked_prefill_policy')
    # A SEPARATE early return, placed after the allowlist branch rather than inside
    # its condition. Opening that condition ('... or _m1_chunked_prefill_opt_in()')
    # sent qwen3_5 into the branch written for gemma4, and that branch sets
    # disable_chunked_mm_input - the exact flag vLLM refuses to honour when a declared
    # multimodal item is larger than the batched-token budget. Run 35681324335 died
    # before readiness: 'max_tokens_per_mm_item (16384) is larger than
    # max_num_batched_tokens (2048)'.
    #
    # Qwen3_5ForConditionalGeneration resolves to Qwen36ForCausalLM, which is text-only
    # and cannot consume an image, so that 16384-token item is a phantom of the HF
    # config's nested vision config. gemma4's reason for the flag - a chunk boundary
    # would split a real multimodal item's embeddings from their positions - cannot
    # apply to a class that has no such items, so this path must not inherit it.
    # vLLM 0.25.1 defaults disable_chunked_mm_input to False and the raise is guarded
    # by it (probe_mm_budget_chunked_prefill, run 35681729538), so leaving the field
    # alone is both closer to this model's shipped behaviour and enough to start.
    #
    # Placed AFTER the allowlist branch so gemma4 stays byte-identical, and BEFORE the
    # block that turns enable_chunked_prefill off, which is what it has to pre-empt.
    lines = replace_once(
        lines, span,
        '    if scheduler_config.enable_chunked_prefill:',
        '    if _m1_chunked_prefill_opt_in():' + chr(10) +
        '        return' + chr(10) + chr(10) +
        '    if scheduler_config.enable_chunked_prefill:',
        'platform opt-in')
    return ''.join(lines)


def patch_scheduler(source):
    """One prefill in flight, inside TTScheduler._schedule_prefill_only.

    The shipped body is:

        saved_max = self.max_num_running_reqs
        self.running = cast(list[Request], partial_prefills)
        self.max_num_running_reqs = max(0, saved_max - len(pure_decodes))
        try:
            result = super().schedule()
        finally:
            self.running.extend(pure_decodes)
            self.max_num_running_reqs = saved_max
        return result

    Two things are wrong with it for the fast path. With a partial in flight the
    waiting loop still admits a fresh prompt, which run 35689293766 hit
    (new=[B] cached=[A]). With nothing in flight the cap is saved_max, so it admits
    up to four, which run 35690327326 hit (new=[A, B] cached=[]). The fast path
    serves exactly one prompt per prefill: the GDN prefill scratch and the host RoPE
    table are single-occupancy.

    The hide/restore is lifted from the plugin's own _schedule_decode_only, which
    already blanks both queues with create_request_queue(self.policy) and merges
    anything the base scheduler added back with prepend_requests. Using the file's
    own idiom is the point - this is the contract's policy, not a second one.
    """
    if MARKER_SCHEDULER in source:
        raise ValueError('scheduler already carries the one-in-flight graft')
    lines = source.splitlines(keepends=True)
    span = function_span_module(source, '_schedule_prefill_only', method=True)
    lines = replace_once(
        lines, span,
        '        saved_max = self.max_num_running_reqs\n'
        '        self.running = cast(list[Request], partial_prefills)\n'
        '        self.max_num_running_reqs = max(0, saved_max - len(pure_decodes))\n'
        '        try:\n'
        '            result = super().schedule()\n'
        '        finally:\n'
        '            self.running.extend(pure_decodes)\n'
        '            self.max_num_running_reqs = saved_max\n'
        '        return result\n',
        '        saved_max = self.max_num_running_reqs\n'
        '        self.running = cast(list[Request], partial_prefills)\n'
        '        # Lever N M2: one prefill in flight. Allowed is the partials already\n'
        '        # running, or exactly one fresh prompt when there are none - the fast\n'
        '        # path cannot serve two, its GDN prefill scratch is single-occupancy.\n'
        '        _qwen_allowed = len(partial_prefills) if partial_prefills else 1\n'
        '        self.max_num_running_reqs = max(\n'
        '            0, min(saved_max - len(pure_decodes), _qwen_allowed))\n'
        '        # Hiding the queues is what actually stops a newcomer joining a partial;\n'
        '        # the cap alone does not, which run 35689293766 demonstrated. Same idiom\n'
        '        # as _schedule_decode_only below.\n'
        '        _qwen_saved_waiting = self.waiting\n'
        '        _qwen_saved_skipped = getattr(self, "skipped_waiting", None)\n'
        '        if partial_prefills:\n'
        '            self.waiting = create_request_queue(self.policy)\n'
        '            if _qwen_saved_skipped is not None:\n'
        '                self.skipped_waiting = create_request_queue(self.policy)\n'
        '        logger.info(\n'
        '            f"' + MARKER_SCHEDULER + ' partials={len(partial_prefills)} "\n'
        '            f"decodes={len(pure_decodes)} allowed={_qwen_allowed} "\n'
        '            f"hidden={bool(partial_prefills)}")\n'
        '        try:\n'
        '            result = super().schedule()\n'
        '        finally:\n'
        '            if partial_prefills:\n'
        '                if self.waiting:\n'
        '                    _qwen_saved_waiting.prepend_requests(self.waiting)\n'
        '                if _qwen_saved_skipped is not None:\n'
        '                    if self.skipped_waiting:\n'
        '                        _qwen_saved_skipped.prepend_requests(self.skipped_waiting)\n'
        '                    self.skipped_waiting = _qwen_saved_skipped\n'
        '                self.waiting = _qwen_saved_waiting\n'
        '            self.running.extend(pure_decodes)\n'
        '            self.max_num_running_reqs = saved_max\n'
        '        return result\n',
        'scheduler one-in-flight')
    return ''.join(lines)


def patch_scheduler_alternation(source):
    """Alternate prefill chunks with decode steps inside TTScheduler.schedule.

    Section 3.3's policy in the DEFAULT-mode branch. See
    move_alternation_to_ttscheduler.py for why this seat and not the coordinator:
    run 35707860782 proved TTLaneCoordinator is never constructed here.
    """
    if MARKER_ALTERNATE in source:
        raise ValueError('scheduler already carries the alternation graft')
    if source.count(_ALTERNATION_ANCHOR) != 1:
        raise ValueError('default-mode prefer-prefill anchor matched %d times'
                         % source.count(_ALTERNATION_ANCHOR))
    return source.replace(_ALTERNATION_ANCHOR, _ALTERNATION_REPLACEMENT)


def patch_scheduler_full(source):
    """Both M2 edits to scheduler.py: one prefill in flight, then alternation.

    They touch different methods - _schedule_prefill_only and schedule - so the
    order does not matter, but they must both be applied: the graft table maps one
    function per file, and mapping only patch_scheduler is how the alternation
    would silently go missing again.
    """
    return patch_scheduler_alternation(patch_scheduler(source))

def patch_lane_scheduler(source):
    """Alternate prefill chunks with decode steps, per design section 3.3.

    The shipped body is two lines:

        intent = max(self._local_prefill_intent(sched) for sched in self.lanes)
        return TTSchedulingMode.from_prefill_intent(intent)

    and _local_prefill_intent votes prefill unconditionally while a partial is in
    flight, so one chunked prompt owns every step until it finishes. The replacement is
    section 3.3's pseudo-code: when a partial prefill AND a running decode both exist,
    spend R decode steps after each prefill chunk instead of voting prefill every time.

    Everything else is unchanged by construction - no running decodes still means
    prefill, no prefill work still means decode, and the zero-token fallback is
    untouched because this only chooses the forced mode.
    """
    if MARKER_ALTERNATE in source:
        raise ValueError('lane scheduler already carries the alternation graft')
    lines = source.splitlines(keepends=True)
    span = function_span_module(source, '_negotiate_forced_mode', method=True)
    lines = replace_once(
        lines, span,
        '        intent = max(self._local_prefill_intent(sched) for sched in self.lanes)\n'
        '        return TTSchedulingMode.from_prefill_intent(intent)\n',
        '        intent = max(self._local_prefill_intent(sched) for sched in self.lanes)\n'
        '        # Lever N M2 item 1, design section 3.3. os and a logger are local: this\n'
        '        # module imports neither, and the marker is an f-string because the\n'
        '        # plugin logger is not loguru - run 35692388798 printed a brace-format\n'
        '        # marker literally and carried no numbers.\n'
        '        import os as _qwen_os\n'
        '        _qwen_on = _qwen_os.environ.get("TT_PREFILL_DECODE_INTERLEAVE", "1") == "1"\n'
        '        _qwen_streak = getattr(self, "_qwen_prefill_streak", 0)\n'
        '        _qwen_credit = getattr(self, "_qwen_decode_credit", 0)\n'
        '        if intent and _qwen_on:\n'
        '            _qwen_partial = any(any(r.is_prefill_chunk for r in s.running)\n'
        '                                for s in self.lanes)\n'
        '            _qwen_decoding = any(any(not r.is_prefill_chunk for r in s.running)\n'
        '                                 for s in self.lanes)\n'
        '            if _qwen_partial and _qwen_decoding:\n'
        '                try:\n'
        '                    _qwen_r = int(_qwen_os.environ.get(\n'
        '                        "TT_DECODE_STEPS_PER_PREFILL_CHUNK", "1"))\n'
        '                except ValueError:\n'
        '                    _qwen_r = 1\n'
        '                _qwen_yield = _qwen_streak >= 1 and _qwen_credit < _qwen_r\n'
        '                if _qwen_yield:\n'
        '                    self._qwen_decode_credit = _qwen_credit + 1\n'
        '                else:\n'
        '                    self._qwen_decode_credit = 0\n'
        '                    self._qwen_prefill_streak = _qwen_streak + 1\n'
        '                try:\n'
        '                    from loguru import logger as _qwen_logger\n'
        '                    _qwen_logger.info(\n'
        '                        f"' + MARKER_ALTERNATE + ' yield_decode={_qwen_yield} "\n'
        '                        f"streak={_qwen_streak} credit={_qwen_credit} r={_qwen_r}")\n'
        '                except BaseException:\n'
        '                    pass\n'
        '                return TTSchedulingMode.from_prefill_intent(0 if _qwen_yield else 1)\n'
        '        self._qwen_prefill_streak = 1 if intent else 0\n'
        '        self._qwen_decode_credit = 0\n'
        '        return TTSchedulingMode.from_prefill_intent(intent)\n',
        'lane scheduler alternation')
    return ''.join(lines)


def function_span_module(source, name, method=False):
    """Span of a module-level function, by the same next-sibling rule as function_span.

    With method=True, search every class body instead: the scheduler edit targets
    TTScheduler._schedule_prefill_only, and scoping to that method's own line range is
    what keeps the edit off the identically-shaped _schedule_decode_only below it.
    """
    tree = ast.parse(source)
    total = len(source.splitlines())
    if method:
        members = [n for cls in tree.body if isinstance(cls, ast.ClassDef)
                   for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    else:
        members = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for index, node in enumerate(members):
        if node.name != name:
            continue
        start = min(d.lineno for d in node.decorator_list) - 1 if node.decorator_list else node.lineno - 1
        end = members[index + 1].lineno - 1 if index + 1 < len(members) else total
        return start, end
    raise ValueError('no module-level function named %s' % name)


def patch_prefill_chunk(source, size):
    """Retune the serving prefill chunk from its default 2048 tokens.

    Total attention work does not change with chunk size: a chunk of C tokens attends to
    everything before it, and summing over N/C chunks still comes to N squared over two.
    What does change is fixed per-chunk cost, which divides by the chunk size. So this is
    a direct test of whether the per-token prefill gap is fixed overhead or real work,
    and it needs no profiler to answer.

    model.py asserts chunk_size % 128 == 0. Larger chunks also need larger activation and
    page-table buffers, so this can fail on memory rather than on merit.
    """
    if type(size) is not int or size % 128 or not 128 <= size <= 16384:
        raise ValueError('chunk must be a multiple of 128 within 128..16384, got %r' % (size,))
    old = '_PREFILL_WARMUP_CHUNK = 2048'
    if source.count(old) != 1:
        raise ValueError('expected one _PREFILL_WARMUP_CHUNK assignment, found %d'
                         % source.count(old))
    return source.replace(old, '_PREFILL_WARMUP_CHUNK = %d' % size)


def patch_prefill_chunk_fallback(source, size):
    """Retune the chunk size prefill actually uses on an untraced path.

    _PREFILL_WARMUP_CHUNK only reaches the model through the chunked-trace capture in
    warmup_model_prefill, and serving runs trace_mode='decode_only', so that capture is
    skipped and the constant is dead. prefill_traced_chunked then falls back to
    'self._chunked_chunk_size or 2048', and that literal is the value in force.

    Run 35423537994 patched the constant at 2048 and 4096 and measured 32.64 s against
    32.61 s, because both arms ran the same 2048 fallback. The lever has to be here.
    """
    if type(size) is not int or size % 128 or not 128 <= size <= 16384:
        raise ValueError('chunk must be a multiple of 128 within 128..16384, got %r' % (size,))
    old = 'chunk_size = self._chunked_chunk_size or 2048'
    if source.count(old) != 1:
        raise ValueError('expected one chunk-size fallback, found %d' % source.count(old))
    # Emit the value in force, so the experiment's control reads a marker the patch
    # itself produces rather than scraping for an incidental 'chunk=' elsewhere in the
    # log. Scraping reported 32, from an unrelated operation, in run 35423537994.
    marker = ('chunk_size = self._chunked_chunk_size or %d' % size + chr(10)
              + '        logger.info("[CHUNK] prefill chunk_size=%d" % chunk_size)')
    return source.replace(old, marker)


def patch_model(source):
    patched = patch_tp_replay(source)
    patched = patch_chunked_entry(patched)
    patched = add_slots_range(patched)
    ast.parse(patched)
    return patched


def stage(model_path, output=None):
    model_path = Path(model_path)
    source = model_path.read_text(encoding='utf-8')
    patched = patch_model(source)
    target = Path(output) if output else model_path
    target.write_text(patched, encoding='utf-8', newline='\n')
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_path')
    parser.add_argument('--output')
    options = parser.parse_args()
    print('wrote %s' % stage(options.model_path, options.output))
