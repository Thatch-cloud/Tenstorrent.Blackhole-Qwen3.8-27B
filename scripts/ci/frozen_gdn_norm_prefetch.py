"""Unqualified T16 norm bridge prefetch; preserve every FP32 bit and math operation."""

from unittest.mock import patch

from gdn_multitoken import replace_once


PREFETCH = '''    CircularBuffer sticks(cb_stick);
    sticks.reserve_back(2);
    for (uint32_t token = 0; token < n_inst; ++token) {
        for (uint32_t partition = 0; partition < 4; ++partition) {
            noc.async_read(pre_acc, sticks, 128,
                {.page_id = token * 96 + bh_start * 4 + partition},
                {.offset_bytes = (token * 4 + partition) * 128});
        }
    }
    noc.async_read_barrier();
    sticks.push_back(2);
    sticks.wait_front(2);
'''

SERIAL = '''            CircularBuffer stick(cb_stick);
            stick.reserve_back(1);
            noc.async_read(pre_acc, stick, 128,
                {.page_id = token * 96 + bh_start * 4 + partition}, {.offset_bytes = 0});
            noc.async_read_barrier();
            stick.push_back(1);
            stick.wait_front(1);
            const uint32_t src = stick.get_read_ptr();'''


def reader(source):
    source = replace_once(source, '    zero(destination, 4 * 4096 / 4);\n',
        '    zero(destination, 4 * 4096 / 4);\n' + PREFETCH)
    source = replace_once(source, SERIAL,
        '            const uint32_t src = sticks.get_read_ptr() + (token * 4 + partition) * 128;')
    source = replace_once(source, '            stick.pop_front(1);\n', '')
    return replace_once(source, '    pre.push_back(4);',
        '    sticks.pop_front(2);\n    pre.push_back(4);')


def buffers(io, fp32):
    if 5 in io or fp32.get(5) != 1:
        raise ValueError('Expected original one-page FP32 norm staging buffer')
    return dict(io), dict(fp32) | {5: 2}


def build(operations, mesh, tensors, *, root):
    import gdn_shared_qk_pipeline as pipeline
    import gdn_vsplit as split

    original_kernels, original_buffers = pipeline.load_kernels, split.cb_plan

    def kernels(requested_root):
        result = {stage: dict(parts) for stage, parts in original_kernels(requested_root).items()}
        result['norm_gate']['reader'] = reader(result['norm_gate']['reader'])
        return result

    def plan(stage, *, prefetch_inputs=False):
        original = original_buffers(stage, prefetch_inputs=prefetch_inputs)
        return buffers(*original) if stage == 'norm_gate' else original

    with patch.object(pipeline, 'load_kernels', kernels), patch.object(split, 'cb_plan', plan):
        return pipeline.build(operations, mesh, tensors, root=root)
