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
- prefill_traced_chunked gains start / is_last and derives that range.
- prefill_paged_slots_range drives one step across N requests, writing a decode slot
  only for the requests finishing on this step.

Applying nothing on import: stage() is explicit, mirroring serving_plugin_patch.
"""

import argparse
import ast
from pathlib import Path

TP_FUNCTION = '_prefill_traced_chunked_tp'
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


def patch_tp_replay(source):
    """Make the TP chunk replay resumable: a chunk range, a conditional reset and tail."""
    lines = source.splitlines(keepends=True)
    span = function_span(source, TP_FUNCTION)

    lines = replace_once(
        lines, span,
        'self, token_ids, page_table, actual_len, num_full, chunk_size, tail_real, vision_tokens=None',
        'self, token_ids, page_table, actual_len, num_full, chunk_size, tail_real, vision_tokens=None,\n'
        '        chunk_from=0, chunk_to=None, do_reset=True, do_tail=True',
        'tp signature')

    span = function_span(''.join(lines), TP_FUNCTION)
    lines = replace_once(
        lines, span,
        '        # Re-zero GDN once; carries across replays + tail (chunk_start>0 skips reset).\n'
        '        self._reset_gdn_state_for_new_sequence()',
        '        # Re-zero GDN on the first step only; state carries across steps, replays and tail.\n'
        '        chunk_to = num_full if chunk_to is None else int(chunk_to)\n'
        '        if do_reset:\n'
        '            self._reset_gdn_state_for_new_sequence()',
        'tp reset')

    span = function_span(''.join(lines), TP_FUNCTION)
    lines = replace_once(lines, span, '        for c in range(num_full):',
                         '        for c in range(chunk_from, chunk_to):', 'tp chunk loop')

    span = function_span(''.join(lines), TP_FUNCTION)
    lines = replace_once(lines, span, '        if tail_real > 0:',
                         '        if do_tail and tail_real > 0:', 'tp tail')
    return ''.join(lines)


def patch_chunked_entry(source):
    """Thread start / is_last through the chunked entry point."""
    lines = source.splitlines(keepends=True)
    span = function_span(source, CHUNKED_FUNCTION)
    lines = replace_once(
        lines, span,
        'def prefill_traced_chunked(self, token_ids, page_table, actual_len, vision_tokens=None):',
        'def prefill_traced_chunked(self, token_ids, page_table, actual_len, vision_tokens=None,\n'
        '                               start=0, is_last=True):',
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
        '                    chunk_to=num_full, do_reset=(start == 0), do_tail=is_last,\n'
        '                )',
        'chunked tp call')
    return ''.join(lines)


SLOTS_RANGE = '''
    def prefill_paged_slots_range(self, token_ids_list, page_table, empty_slots, starts, ends,
                                  is_last, valid_lens=None):
        """One resumable prefill step across N requests (Lever N M1, design section 3.1).

        The per-step analogue of prefill_paged_slots. Each request advances through its
        own [start, end) token window; only a request whose window ends at its prompt
        length produces logits and writes its decode slot. Intermediate steps return
        zero logits for that row and leave the slot untouched, so a long prompt can be
        interleaved with decode steps for the other slots.

        One in-flight prefill per lane in v1: the GDN scratch and the host RoPE table are
        single-occupancy, so a second request's prefill between two chunk steps of the
        first would corrupt it. The scheduler enforces that; this method assumes it.
        """
        assert self.num_devices > 1, "prefill_paged_slots_range is the TP (num_devices>1) path"
        N = len(token_ids_list)
        assert len(empty_slots) == N and len(starts) == N and len(ends) == N and len(is_last) == N, (
            "one slot, start, end and is_last per request"
        )
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
                start, last = int(starts[u]), bool(is_last[u])
                lg = self.prefill_traced_chunked(
                    toks[:, :actual], pt[u : u + 1], actual_len=actual, start=start, is_last=last
                )
                if not last:
                    # Intermediate step: no logits for this row, and no slot write. The GDN
                    # scratch keeps this request's state for its next step.
                    if lg is not None:
                        ttnn.deallocate(lg)
                    host_logits[u] = torch.zeros(1, 1, self.args.vocab_size)
                    continue
                host_logits[u] = (
                    ttnn.to_torch(lg, mesh_composer=comp)
                    .reshape(-1, self.args.vocab_size)[:1]
                    .float()
                    .view(1, 1, -1)
                )
                ttnn.deallocate(lg)
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
    """Thread start_pos / is_last from the runner down to the batched prefill.

    vLLM hands the model a chunk window per step once chunked prefill is on. The entry
    passes it through unchanged; the batched path turns it into per-request starts and
    ends and calls prefill_paged_slots_range instead of prefill_paged_slots.
    """
    lines = source.splitlines(keepends=True)
    span = function_span(source, VLLM_ENTRY)
    lines = replace_once(
        lines, span,
        'return self._prefill_forward_tp_batched(model, tokens, page_table, prompt_lens, kwargs.get("empty_slots"))',
        'return self._prefill_forward_tp_batched(\n'
        '                model, tokens, page_table, prompt_lens, kwargs.get("empty_slots"),\n'
        '                start_pos=kwargs.get("start_pos"), is_last=kwargs.get("is_last"),\n'
        '            )',
        'vllm entry call')

    span = function_span(''.join(lines), VLLM_BATCHED)
    lines = replace_once(
        lines, span,
        'def _prefill_forward_tp_batched(self, model, tokens, page_table, prompt_lens, empty_slots):',
        'def _prefill_forward_tp_batched(self, model, tokens, page_table, prompt_lens, empty_slots,\n'
        '                                    start_pos=None, is_last=None):',
        'batched signature')

    span = function_span(''.join(lines), VLLM_BATCHED)
    lines = replace_once(
        lines, span,
        '        host_logits = model.prefill_paged_slots(token_ids_list, pt, empty_slots, valid_lens=plens)',
        '        if start_pos is None:\n'
        '            host_logits = model.prefill_paged_slots(token_ids_list, pt, empty_slots, valid_lens=plens)\n'
        '        else:\n'
        '            # Chunked prefill: this step covers [start_pos[u], plens[u]) of each row.\n'
        '            # prompt_lens is the chunk END in the runner terms, and is_last comes\n'
        '            # from the runner intermediate_prefill_mask rather than being re-derived.\n'
        '            starts = [int(s) for s in start_pos]\n'
        '            lasts = [bool(v) for v in is_last] if is_last is not None else [True] * N\n'
        '            host_logits = model.prefill_paged_slots_range(\n'
        '                token_ids_list, pt, empty_slots, starts, plens, lasts, valid_lens=plens\n'
        '            )',
        'batched dispatch')
    return ''.join(lines)

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
