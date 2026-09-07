"""Bounded attribution replays of existing T8 traces, not throughput measurements."""

PROFILE_BASE_FLAGS = ('batch', 'coding_cost', 'serial_sdpa', 'compact_gdn', 'reuse_gdn_input',
    'skip_row_clones', 'hoist_row_layout', 'device_loop_gdn', 'compact_prologue',
    'batch_conv', 'packed_checkpoints', 'ordered_cache')
PROFILE_BEST_FLAGS = ('norm_batch', 'grouped_attention', 'attention_dma', 'attention_parallel',
    'attention_tree', 'prefix_zero_reuse')


def profile_replays(*, rows, length, traces, restore, synchronize, execute, validate, dump, signpost):
    if rows != 8 or length not in (4095, 16383):
        raise ValueError('Verifier profiling requires T8 at the matched coding contexts')
    if not all(arm in traces for arm in ('serial', 'control', 'batch')):
        raise ValueError('Native, paired control and optimized candidate traces required')
    records = []
    for repeat in range(3):
        for arm in ('serial', 'control', 'batch'):
            restore()
            synchronize()
            dump()
            label = f'qwen_verifier_t8_ctx{length}_{arm}_{repeat}'
            print('QWEN_VERIFIER_PROFILE_BEGIN ' + label, flush=True)
            signpost(label + '_begin')
            try:
                execute(traces[arm])
                synchronize()
            finally:
                signpost(label + '_end')
                dump()
            print('QWEN_VERIFIER_PROFILE_END ' + label, flush=True)
            validate(arm)
            records.append(dict(label=label, arm=arm, repeat=repeat, trace_id=int(traces[arm]), exact=True))
    return dict(rows=rows, length=length, records=records,
                scope='Instrumented existing traces; restore, profiler dump and validation outside markers; not committed tok/s')
