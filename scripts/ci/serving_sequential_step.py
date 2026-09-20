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

from serving_worker_hook import phase

# Diagnostic for the two-user selector divergence (runs 35478872085 and
# 35479238722): after each user's step, every OTHER user's replicated draft
# buffers must still agree across the two chips, and its K/V banks must still
# hold what they held after its own last step. Read once at import and off by
# default, so it can stay in. '1' raises at the first divergence; 'warn' logs it
# and lets decoding continue, so a run shows whether it still proceeds.
SHARD_CHECK = os.environ.get('QWEN_FAST_SHARD_CHECK', '0')
if SHARD_CHECK not in ('0', '1', 'warn'):
    raise ValueError('QWEN_FAST_SHARD_CHECK must be unset, 0, 1 or warn')
# Buffer addresses of every checked tensor, by request id, taken the first time
# the step sees that request and logged once. This module cannot see
# DFlashDevice construction, but these are persistent allocations, so the
# address at first sight is the address at construction - what a trace's freed
# regions are matched against later. Pruned to the live requests each step.
RECORDED = {}
# (victim request id, buffer name, chip or None) already reported, so under
# 'warn' a buffer that stays diverged is reported once rather than every step.
REPORTED = set()
# Per-chip host copies of each request's committed K/V rows, taken before its
# first step and again right after each of its own steps. The banks differ
# between the chips by construction (see replicated_buffers), so what they are
# checked for is a write between their owner's steps, chip by chip.
SNAPSHOTS = {}
# Per-chip host copies of each request's page tables - the verifier fixtures'
# page table and row table, and the replay reader's per-bundle tables - taken at
# the start of every checked round (the page bindings refresh them on a block
# change just before the step, serving_packed_bridge.py) and again right after
# the owner's own step. Static otherwise, so a change between the owner's steps
# is another request's replay writing over them (attention_replay.py): from its
# next verify on, every attention layer reads the wrong KV pages.
PAGE_TABLES = {}
# The diagnostic's line headers and the tag its detail lines carry, by kind.
LABELS = dict(mismatch=('shard mismatch', 'mismatch'), drift=('kv drift', 'drift'),
              pages=('page-table drift', 'page-table drift'))


def sequential_packed_step(entries, *, cancelled):
    """Step every packed request in the scheduler's order, one at a time."""
    if not entries:
        raise ValueError('A packed step needs at least one admitted request')
    checking = SHARD_CHECK != '0' and len(entries) > 1
    if checking:
        forget_departed(entries)
        for entry in entries:
            if entry['request_id'] not in SNAPSHOTS:
                snapshot_kv(entry)
            # Every round: the page bindings may have just rewritten them for new blocks.
            snapshot_pages(entry)
    outputs = []
    for index, entry in enumerate(entries):
        request, ticket = entry['request'], entry['ticket']
        if ticket.request_id != entry['request_id']:
            raise ValueError('Each packed entry must carry its own prepared ticket')
        # Under QWEN_FAST_PHASE_LOG the same begin/end lines the hook writes around
        # each proposal, so a hang says which phase stalled and whose.
        output = phase('step', ticket.request_id, lambda: request.step(ticket.request_id, cancelled=cancelled))
        if output is None or output.request_id != entry['request_id']:
            raise ValueError('A packed request must commit its own output')
        outputs.append(output)
        if checking:
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
    largest replicated weight in a layer. About 58 MB of readback per device:
    selector 2.6 MB, convolution 13 MB, the two histories 21 MB each.

    The K/V banks are NOT here. `project_key_value` (draft_kv_projection.py:21)
    runs those same sharded k/v projections, so each chip holds its own four of
    the eight K/V heads (draft_attention.py:23-26): the banks differ per chip by
    construction. Run 35482551725 read exactly that - all ten banks differing on
    every step, ~1,047,800 of 1,048,576 elements, both directions, from the
    first step - while these buffers stayed bit-identical; the run 35481466425
    'kv_history[0].k victim' was this structure, not a scribble. The banks get a
    per-chip stability check in `check_shards` instead.
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
    return [(category, name, value) for category, name, value in buffers if value is not None]


def kv_banks(device, spare=False):
    """The committed (or, with `spare`, the standby) draft K/V banks, by name."""
    if device.closed or device.kv_history is None:
        return []
    banks = device.kv_history.spare if spare else device.kv_history.active
    return [('%s[%d].%s' % ('kv_spare' if spare else 'kv_history', layer, name), cache[name])
            for layer, cache in enumerate(banks) for name in ('k', 'v')]


def snapshot_kv(entry):
    """Own host copies of each chip's committed K/V rows: [0, history_rows), because
    the rows beyond are rewritten by the owner's own publication."""
    device = entry['request'].runtime.drafter
    operations, banks = device.operations, kv_banks(device)
    rows = device.kv_history.history_rows if banks else 0
    SNAPSHOTS[entry['request_id']] = {
        name: (rows, [operations.to_torch(shard)[..., :rows, :].contiguous().clone()
                      for shard in operations.get_device_tensors(value)])
        for name, value in banks}


def page_tables(engine):
    """Every page table a request's verifier keeps across steps, by name: per captured
    bucket, the fixture's page table and its row tables (unpacked, one singleton every
    row's cache writer and reader binds - pooled with it), and the replay reader's
    per-bundle tables. Named by bucket key, e.g. replay_pages[16,4352][1]."""
    if engine is None or getattr(engine, 'phase', 'closed') == 'closed':
        return []
    tables = []
    for key, bucket in getattr(engine, 'buckets', {}).items():
        fixture = bucket.get('fixture')
        if fixture is None:
            continue
        label = '[%s]' % ','.join(str(part) for part in (key if isinstance(key, tuple) else (key,)))
        candidates = [('pages%s' % label, fixture.pages), ('singleton_pages%s' % label, fixture.singleton_pages)]
        candidates.extend(('row_pages%s[%d]' % (label, index), table)
                          for index, table in enumerate(getattr(fixture, 'row_pages', ())))
        replay = getattr(fixture, 'replay_reader', None)
        if replay is not None:
            candidates.extend(('replay_pages%s[%d]' % (label, index), entry[1])
                              for index, entry in enumerate(replay.metadata))
        seen = set()
        for name, table in candidates:
            # The row tables alias the singleton: read each buffer once, under its first name.
            if id(table) not in seen:
                seen.add(id(table))
                tables.append((name, table))
    return tables


def snapshot_pages(entry):
    """Own host copies of each chip's page tables, whole - they are small."""
    engine = getattr(entry['request'], 'engine', None)
    tables = page_tables(engine)
    PAGE_TABLES[entry['request_id']] = {
        name: [engine.operations.to_torch(shard).contiguous().clone()
               for shard in engine.operations.get_device_tensors(value)]
        for name, value in tables}


def forget_departed(entries):
    live = {entry['request_id'] for entry in entries}
    for table in (RECORDED, SNAPSHOTS, PAGE_TABLES):
        for stale in [request_id for request_id in table if request_id not in live]:
            del table[stale]
    REPORTED.difference_update([key for key in REPORTED if key[0] not in live])


def check_shards(entries, stepped):
    """After `entries[stepped]` ran, every OTHER entry's replicated draft buffers
    must still be bit-identical on both chips, each chip's K/V banks must still
    hold what they held after that entry's own last step, and each chip's page
    tables what they held at the start of the round or after that entry's own
    last step. One that does not was written by something other than its own
    user."""
    import torch
    from gdn_multitoken_conv import addresses
    from loguru import logger

    live = [entry['request_id'] for entry in entries]
    calls = [entry['request'].runtime.drafter.proposal_calls for entry in entries]
    actor = entries[stepped]
    snapshot_kv(actor)
    snapshot_pages(actor)
    equal = diverged = 0

    def same(left, right):
        return torch.equal(left.view(torch.int16), right.view(torch.int16))

    def report(kind, operations, index, name, value, what, left, right, failure):
        # Several short lines: the log capture truncates around 250 characters,
        # and run 35481466425 lost everything after the shape.
        victim = entries[index]['request_id']
        header, tag = LABELS[kind]
        difference = (left.float() - right.float()).abs()
        # The histories swap roles on commit (DFlashDevice.commit_publication) and
        # so do the K/V banks (DraftKVHistory.commit), so say which name this
        # address was first seen under; None means allocated after first sight.
        current = addresses(operations, value)
        origin = next((seen for seen, address in RECORDED[victim].items() if address == current), None)
        logger.info('[PINDIAG] {} after step of {} (entry {}): victim={} (entry {}) {}',
                    header, actor['request_id'], stepped, victim, index, what)
        logger.info('[PINDIAG] {} buffer={} shape={} address={} first_seen_as={}',
                    tag, name, tuple(value.shape), current, origin)
        logger.info('[PINDIAG] {} differing={} of {} max_abs={:g}',
                    tag, int((difference > 0).sum()), difference.numel(), float(difference.max()))
        logger.info('[PINDIAG] {} scheduler order={} proposal_calls={}', tag, live, calls)
        if SHARD_CHECK == '1':
            raise AssertionError('%s: victim=%s buffer=%s; see the [PINDIAG] %s log lines'
                                 % (failure, victim, name, tag))

    for index, entry in enumerate(entries):
        if index == stepped:
            continue
        # FastRequest.runtime is the DFlashRequestRuntime; its drafter is the DFlashDevice;
        # FastRequest.engine is the VerifierEngine whose fixtures hold the page tables.
        device = entry['request'].runtime.drafter
        engine = getattr(entry['request'], 'engine', None)
        operations, victim = device.operations, entry['request_id']
        replicated, banks, tables = replicated_buffers(device), kv_banks(device), page_tables(engine)
        if victim not in RECORDED:
            RECORDED[victim] = {name: addresses(operations, value) for name, value in
                                [*((name, value) for category, name, value in replicated),
                                 *banks, *kv_banks(device, spare=True)]}
            RECORDED[victim].update({name: addresses(engine.operations, value) for name, value in tables})
            for name, (first, second) in RECORDED[victim].items():
                logger.info('[PINDIAG] address {} {} {} {}', victim, name, first, second)
        for category, name, value in replicated:
            shards = [operations.to_torch(shard).contiguous() for shard in operations.get_device_tensors(value)]
            if len(shards) != 2:
                raise AssertionError('Both chips required')
            if same(*shards):
                equal += 1
                continue
            diverged += 1
            if SHARD_CHECK == 'warn' and (victim, name, None) in REPORTED:
                continue
            REPORTED.add((victim, name, None))
            report('mismatch', operations, index, name, value, 'category=%s' % category, *shards,
                   'Replicated draft %s differs between chips' % category)
        saved = SNAPSHOTS.get(victim, {})
        for name, value in banks:
            if name not in saved:
                continue
            rows, kept = saved[name]
            drifted = False
            for chip, (shard, before) in enumerate(zip(operations.get_device_tensors(value), kept, strict=True)):
                current = operations.to_torch(shard)[..., :rows, :].contiguous()
                if same(current, before):
                    continue
                drifted = True
                if SHARD_CHECK == 'warn' and (victim, name, chip) in REPORTED:
                    continue
                REPORTED.add((victim, name, chip))
                report('drift', operations, index, name, value, 'chip=%d rows=%d' % (chip, rows), current, before,
                       "Draft K/V drifted on chip %d between its owner's steps" % chip)
            equal += not drifted
            diverged += drifted
        kept_tables = PAGE_TABLES.get(victim, {})
        for name, value in tables:
            if name not in kept_tables:
                continue
            drifted = False
            for chip, (shard, before) in enumerate(zip(engine.operations.get_device_tensors(value), kept_tables[name], strict=True)):
                current = engine.operations.to_torch(shard).contiguous()
                if torch.equal(current, before):
                    continue
                drifted = True
                if SHARD_CHECK == 'warn' and (victim, name, chip) in REPORTED:
                    continue
                REPORTED.add((victim, name, chip))
                report('pages', engine.operations, index, name, value, 'chip=%d rows=%d' % (chip, tuple(value.shape)[0]),
                       current, before, "Page table drifted on chip %d between its owner's steps" % chip)
            equal += not drifted
            diverged += drifted
    if diverged:
        logger.info('[PINDIAG] shards differ after step of {}: {} equal, {} diverged proposal_calls={}',
                    actor['request_id'], equal, diverged, calls)
    else:
        logger.info('[PINDIAG] shards equal after step of {}: {} buffers proposal_calls={}',
                    actor['request_id'], equal, calls)


def describe():
    """What this costs, so a benchmark reading it is not mistaken for the goal."""
    return dict(name='sequential', weight_passes_per_round='one per user',
                per_user_rate='single-user rate divided by users',
                batched=False)
