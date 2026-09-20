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
# Buffer addresses of every checked tensor, by request id, taken the first time
# the step sees that request and logged once. This module cannot see
# DFlashDevice construction, but these are persistent allocations, so the
# address at first sight is the address at construction - what a trace's freed
# regions are matched against later. Pruned to the live requests each step.
RECORDED = {}


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
    """Every checked buffer a DFlashDevice keeps replicated across the two chips,
    as (category, name, tensor), weights first.

    The weights are a bounded subset: the ones uploaded FIRST at construction sit
    at the lowest addresses, the deepest holes under a later request's capture.
    The q/k/v/o and the three MLP projections are sharded across the chips
    (draft_attention_branch.py:39-41, draft_mlp_branch.py:30), so they have no
    cross-chip equality to check; the convolution kernel projection is the
    largest replicated weight in a layer. About 79 MB of readback per device:
    selector 2.6 MB, convolution 13 MB, the two histories 21 MB each, K/V 21 MB.
    """
    if device.closed:
        return []
    weights = [('weight.selector_projection', device.selector_projection), ('weight.final_norm', device.final_norm)]
    if device.layers:
        attention, mlp = device.layers[0][0], device.layers[0][1]
        weights.extend([('weight.layers[0].attention.norm', attention['norm']),
                        ('weight.layers[0].attention.convolution', attention['convolution']),
                        ('weight.layers[0].mlp.device_norm', mlp['device_norm'])])
    buffers = [('weight', name, value) for name, value in weights]
    buffers.extend([('history', 'history', device.history), ('history', 'spare_history', device.spare_history)])
    if device.kv_history is not None:
        buffers.extend(('kv', 'kv_history[%d].%s' % (layer, name), cache[name])
                       for layer, cache in enumerate(device.kv_history.active) for name in ('k', 'v'))
    return [(category, name, value) for category, name, value in buffers if value is not None]


def check_shards(entries, stepped):
    """After `entries[stepped]` ran, every OTHER entry's replicated draft buffers
    must still be bit-identical on both chips. One that is not was written by
    something other than its own user."""
    import torch
    from gdn_multitoken_conv import addresses
    from loguru import logger

    live = [entry['request_id'] for entry in entries]
    for stale in [request_id for request_id in RECORDED if request_id not in live]:
        del RECORDED[stale]
    actor = entries[stepped]
    checked = 0
    for index, entry in enumerate(entries):
        if index == stepped:
            continue
        # FastRequest.runtime is the DFlashRequestRuntime; its drafter is the DFlashDevice.
        device = entry['request'].runtime.drafter
        operations = device.operations
        buffers = replicated_buffers(device)
        recorded = RECORDED.get(entry['request_id'])
        if recorded is None:
            recorded = RECORDED[entry['request_id']] = {name: addresses(operations, value)
                                                        for category, name, value in buffers}
            logger.info('[PINDIAG] draft buffer addresses for {}: {}', entry['request_id'], recorded)
        for category, name, value in buffers:
            shards = [operations.to_torch(shard).contiguous() for shard in operations.get_device_tensors(value)]
            if len(shards) != 2:
                raise AssertionError('Both chips required')
            left, right = shards
            if torch.equal(left.view(torch.int16), right.view(torch.int16)):
                checked += 1
                continue
            difference = (left.float() - right.float()).abs()
            # The histories and K/V swap roles on commit (dflash_device.py:193-195,
            # draft_kv_history.py:118), so say which name this address was first
            # seen under; None means it was allocated after first sight.
            current = addresses(operations, value)
            origin = next((seen for seen, address in recorded.items() if address == current), None)
            raise AssertionError(
                '[PINDIAG] replicated draft %s differs between chips after step of %s (entry %d): '
                'victim=%s (entry %d) buffer=%s shape=%s address=%s first_seen_as=%s '
                'differing=%d of %d max_abs=%g; scheduler order=%s proposal_calls=%s'
                % (category, actor['request_id'], stepped, entry['request_id'], index, name,
                   tuple(value.shape), current, origin,
                   int((difference > 0).sum()), difference.numel(), float(difference.max()),
                   live, [other['request'].runtime.drafter.proposal_calls for other in entries]))
    logger.info('[PINDIAG] shards equal after step of {}: {} buffers', actor['request_id'], checked)


def describe():
    """What this costs, so a benchmark reading it is not mistaken for the goal."""
    return dict(name='sequential', weight_passes_per_round='one per user',
                per_user_rate='single-user rate divided by users',
                batched=False)
