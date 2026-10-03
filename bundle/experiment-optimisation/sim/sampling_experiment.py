"""Opt-in TP2 greedy-only sampling experiment with unchanged host fallbacks."""

import json
import os
from pathlib import Path

CONTEXT = None
RECORDED = set()


def enabled():
    return os.environ.get("QWEN_TP2_SAMPLING_EXPERIMENT") == "1"


def enable_tp2(mesh_shape, vocab):
    if not enabled():
        return False
    if tuple(mesh_shape) != (1, 2) or vocab != 248320:
        raise ValueError("Sampling experiment requires the reviewed TP2 Qwen vocabulary")
    return True


def select_sampling(runner, original, is_decode):
    global CONTEXT
    CONTEXT = None
    if not enabled():
        return original
    ids = [request_id for request_id in runner.input_batch.req_ids if request_id is not None]
    if len(ids) != 1:
        return False
    request_id = ids[0]
    if "qwen-sampling-host-" in request_id:
        arm = "host"
    elif "qwen-sampling-device-" in request_id:
        arm = "device"
    else:
        return False
    temperatures = runner.input_batch.sampling.temperature[:1]
    greedy = bool((temperatures == 0).all().item())
    selected = bool(original and is_decode and arm == "device" and greedy and runner.input_batch.no_penalties)
    CONTEXT = dict(request_id=request_id, arm=arm, is_decode=is_decode,
                   original_eligible=bool(original), selected=selected)
    return selected


def require_greedy(force_argmax):
    if enabled() and not force_argmax:
        raise RuntimeError("TP2 experiment cannot execute generic Top-K sampling")


def record_execution(runner, model_input):
    if not enabled() or CONTEXT is None or not CONTEXT["is_decode"]:
        return
    actual = bool(model_input.perform_device_sampling)
    if actual != CONTEXT["selected"]:
        raise AssertionError("Sampling decision changed before execution")
    sampler = runner.model.model[0].sampling
    force_argmax = sampler.tt_sampling.force_argmax_sampling if sampler is not None else None
    if actual:
        require_greedy(force_argmax)
    request_id = CONTEXT["request_id"]
    if request_id not in RECORDED:
        with Path("/experiment/results/sampling-engagement.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(**CONTEXT, force_argmax=force_argmax, trace_mode=runner.trace_mode)) + "\n")
        RECORDED.add(request_id)
