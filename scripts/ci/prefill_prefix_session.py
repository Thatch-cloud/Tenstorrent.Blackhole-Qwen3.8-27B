"""Explicit offline cache lifetime inside an externally held exclusive page lease."""

from contextlib import ExitStack, contextmanager

from model_batch import instance_overrides
from prefill_checkpoint_allocation import prepare
from prefill_prefix_controller import PrefixController, full_request_factory
from prefill_prefix_residency import OfflinePrefixResidency


@contextmanager
def offline_prefix_session(operations, generator, model, identity, pages, *,
        prefix_position, reserved_pages, inactive_pages):
    import torch

    if (getattr(model, 'num_devices', None) != 2
            or hasattr(model, '_qwen_prefix_session')
            or not isinstance(pages, torch.Tensor) or pages.device.type != 'cpu'
            or pages.ndim != 2 or pages.shape[0] != 1
            or pages.dtype not in (torch.int32, torch.int64)
            or type(prefix_position) is not int or prefix_position <= 0
            or prefix_position % 2048 or not prefix_position < pages.shape[1] * 64 <= 262144):
        raise ValueError('Exclusive single-stream TP2 page table and aligned prefix required')
    page_table = pages.clone()
    inactive = tuple(inactive_pages)
    residency = OfflinePrefixResidency(operations, model, identity, reserved_pages)
    with ExitStack() as stack:
        stack.callback(residency.invalidate)
        residency(identity, page_table[0].tolist(), inactive)
        stack.enter_context(instance_overrides([(model, '_qwen_prefix_session', identity)]))
        allocation = prepare(operations, generator, model)
        stack.callback(allocation.close)
        controller = PrefixController(operations, model, allocation.checkpoint, residency)
        stack.callback(controller.invalidate)
        yield full_request_factory(controller, identity, page_table,
            prefix_position=prefix_position, inactive_pages=inactive)
