"""A correctness-first device step for a packed block: one user at a time.

`serving_packed_bridge.execute_packed_decode` takes the device step as a
parameter. This is the simplest implementation that is CORRECT: it steps each
packed request in turn, through exactly the single-user machinery that already
works, and returns the outputs in the scheduler's order.

**This is not the batched verifier and does not pretend to be.** It spends one
full pass over the 19.92 GB of dense projections per user per round, so per-user
throughput is the single-user rate divided by the number of users - the very thing
`docs/batch-spec-tasks-2026-09-19.md` rules out as a path to 200 tok/s per user.
What it buys is a working two-user serving path today, and a place for the real
packed step to drop in behind the same interface tomorrow.

Its one genuine job beyond looping is ORDER. Probe 35436807668 measured the
scheduler presenting the pair as `cached=['B','A']`, not in creation order, and
`execute_packed_decode` checks that each output's request id matches the entry it
was produced for. Stepping in entry order is what keeps that true.
"""

import os

# Diagnostic for the two-user selector divergence (runs 35478872085 and
# 35479238722): after each user's step, every OTHER user's replicated draft
# buffers must still agree across the two chips. Read once at import and off by
# default, so it can stay in.
SHARD_CHECK = os.environ.get('QWEN_FAST_SHARD_CHECK') == '1'


def sequential_packed_step(entries, *, cancelled):
    """Step every packed request in the scheduler's order, one at a time."""
    if not entries:
        raise ValueError('A packed step needs at least one admitted request')
    outputs = []
    for index, entry in enumerate(entries):
        request, ticket = entry['request'], entry['ticket']
        if ticket.request_id != entry['request_id']:
            raise ValueError('Each packed entry must carry its own prepared ticket')
        output = request.step(ticket.request_id, cancelled=cancelled)
        if output is None or output.request_id != entry['request_id']:
            raise ValueError('A packed request must commit its own output')
        outputs.append(output)
        if SHARD_CHECK and len(entries) > 1:
            check_shards(entries, index)
    return outputs


def replicated_buffers(device):
    """Every buffer a DFlashDevice keeps replicated across the two chips, by name."""
    buffers = [('history', device.history), ('spare_history', device.spare_history)]
    if device.kv_history is not None:
        buffers.extend(('kv_history[%d].%s' % (layer, name), cache[name])
                       for layer, cache in enumerate(device.kv_history.active) for name in ('k', 'v'))
    return [(name, value) for name, value in buffers if value is not None]


def check_shards(entries, stepped):
    """After `entries[stepped]` ran, every OTHER entry's replicated draft buffers
    must still be bit-identical on both chips. One that is not was written by
    something other than its own user."""
    import torch
    from loguru import logger

    actor = entries[stepped]
    checked = 0
    for index, entry in enumerate(entries):
        if index == stepped:
            continue
        # FastRequest.runtime is the DFlashRequestRuntime; its drafter is the DFlashDevice.
        device = entry['request'].runtime.drafter
        operations = device.operations
        for name, value in replicated_buffers(device):
            shards = [operations.to_torch(shard).contiguous() for shard in operations.get_device_tensors(value)]
            if len(shards) != 2:
                raise AssertionError('Both chips required')
            left, right = shards
            if torch.equal(left.view(torch.int16), right.view(torch.int16)):
                checked += 1
                continue
            difference = (left.float() - right.float()).abs()
            raise AssertionError(
                '[PINDIAG] replicated draft buffer differs between chips after step of %s (entry %d): '
                'victim=%s (entry %d) buffer=%s differing=%d of %d max_abs=%g; '
                'scheduler order=%s proposal_calls=%s'
                % (actor['request_id'], stepped, entry['request_id'], index, name,
                   int((difference > 0).sum()), difference.numel(), float(difference.max()),
                   [other['request_id'] for other in entries],
                   [other['request'].runtime.drafter.proposal_calls for other in entries]))
    logger.info('[PINDIAG] shards equal after step of {}: {} buffers', actor['request_id'], checked)


def describe():
    """What this costs, so a benchmark reading it is not mistaken for the goal."""
    return dict(name='sequential', weight_passes_per_round='one per user',
                per_user_rate='single-user rate divided by users',
                batched=False)
