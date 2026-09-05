"""Experimental single-lane scheduler and runner contract for Qwen continuation."""

import os
from contextlib import contextmanager


def enabled():
    flags = [os.environ.get(name, "0") for name in
             ("QWEN_PREFILL_CONTINUATION", "TT_PREFILL_DECODE_INTERLEAVE")]
    if any(value not in ("0", "1") for value in flags) or len(set(flags)) != 1:
        raise ValueError("Continuation and interleaving flags must both be 0 or both be 1")
    return flags[0] == "1"


def validate_config(config):
    if not enabled():
        return False
    scheduler = config.scheduler_config
    parallel = config.parallel_config
    from vllm_tt_plugin.config import get_tt_data_parallel_size

    if parallel.data_parallel_size != 1 or get_tt_data_parallel_size(config) != 1:
        raise ValueError("Continuation v1 requires a single TP2 model, no DP lanes")
    if config.model_config.model != "Qwen/Qwen3.8-27B":
        raise ValueError("Continuation is restricted to the reviewed Qwen/Qwen3.8-27B model")
    if config.speculative_config or config.cache_config.enable_prefix_caching:
        raise ValueError("Continuation v1 requires speculation and prefix reuse disabled")
    if not scheduler.enable_chunked_prefill or scheduler.max_num_batched_tokens != 2048:
        raise ValueError("Continuation requires chunked prefill with a 2048-token budget")
    if scheduler.max_num_seqs <= 1 or scheduler.long_prefill_token_threshold not in (0, 2048):
        raise ValueError("Continuation requires batched slots and no sub-chunk long-prefill threshold")
    scheduler.disable_chunked_mm_input = True
    return True


def choose_prefill(owner, wants_prefill, has_decode):
    ratio = int(os.environ.get("TT_DECODE_STEPS_PER_PREFILL_CHUNK", "1"))
    if ratio not in (1, 2, 4):
        raise ValueError("Decode steps per prefill chunk must be 1, 2 or 4")
    remaining = getattr(owner, "_qwen_decode_credit", 0)
    if not wants_prefill:
        owner._qwen_decode_credit = 0
        return False
    if has_decode and remaining:
        owner._qwen_decode_credit = remaining - 1
        return False
    owner._qwen_decode_credit = ratio if has_decode else 0
    return True


@contextmanager
def single_prefill(scheduler):
    if not enabled():
        yield False
        return
    from vllm.v1.core.sched.request_queue import create_request_queue

    decodes = [request for request in scheduler.running if not request.is_prefill_chunk]
    partial = [request for request in scheduler.running if request.is_prefill_chunk]
    if len(partial) > 1:
        raise RuntimeError("Multiple partial prefills would share one scratch owner")
    has_partial = bool(partial)
    capacity = scheduler.max_num_running_reqs
    waiting, skipped = scheduler.waiting, scheduler.skipped_waiting
    scheduler.running = partial
    scheduler.max_num_running_reqs = min(1, max(0, capacity - len(decodes)))
    if has_partial:
        scheduler.waiting = create_request_queue(scheduler.policy)
        scheduler.skipped_waiting = create_request_queue(scheduler.policy)
    try:
        yield True
    finally:
        if has_partial:
            if scheduler.waiting:
                waiting.prepend_requests(scheduler.waiting)
            if scheduler.skipped_waiting:
                skipped.prepend_requests(scheduler.skipped_waiting)
            scheduler.waiting, scheduler.skipped_waiting = waiting, skipped
        scheduler.running.extend(decodes)
        scheduler.max_num_running_reqs = capacity


def ledger(runner):
    if not hasattr(runner, "_qwen_generations"):
        runner._qwen_generations = {}
        runner._qwen_next_generation = 0
    return runner._qwen_generations


def identities(runner, request_ids):
    if not enabled():
        return None
    if len(request_ids) != 1:
        raise ValueError("Continuation prefill must contain exactly one request")
    generations = ledger(runner)
    result = []
    for request_id in request_ids:
        if request_id not in generations:
            runner._qwen_next_generation += 1
            generations[request_id] = [runner._qwen_next_generation, False]
        result.append((request_id, generations[request_id][0]))
    return result


def release_requests(runner, scheduler_output):
    if not enabled():
        return
    dead = set(scheduler_output.finished_req_ids) | set(scheduler_output.preempted_req_ids or ())
    controller = getattr(runner.model, "_prefill_continuation_controller", None)
    if controller is not None and controller.active is not None:
        request_id, epoch = controller.active["owner"]
        if request_id in dead:
            runner.async_decode.wait_for_all_pending_async_steps()
            controller.cancel(request_id, epoch)
    generations = ledger(runner)
    for request_id in dead:
        generations.pop(request_id, None)


def submit(runner, model_input, kwargs):
    identity = model_input.prefill_request_identity
    if identity is None or len(identity) != 1:
        raise ValueError("Missing scheduler-captured prefill identity")
    request_id, epoch = identity[0]
    generation = ledger(runner).get(request_id)
    if generation != [epoch, False]:
        raise ValueError("Stale or already completed prefill submission")
    if model_input.perform_device_sampling or model_input.block_tables_per_layer is not None:
        raise ValueError("Continuation v1 requires host sampling and a uniform page table")
    marker = model_input.intermediate_prefill_mask
    if marker is None or len(marker) != 1:
        raise ValueError("Missing explicit intermediate-prefill marker")
    if runner.scheduler_config.max_num_batched_tokens != runner.model.model[0]._chunked_chunk_size:
        raise ValueError("Scheduler token budget does not match captured model chunk")
    kwargs.update(request_ids=[request_id], request_epochs=[epoch], intermediate_prefill_mask=marker)
    runner.async_decode.wait_for_all_pending_async_steps()
    output = runner.model.prefill_forward(**kwargs)
    if not bool(marker[0]):
        generation[1] = True
    if runner.request_specific_rope:
        logits, deltas = output
        runner.requests[request_id].mrope_position_delta = deltas[0].item()
        return logits
    return output
