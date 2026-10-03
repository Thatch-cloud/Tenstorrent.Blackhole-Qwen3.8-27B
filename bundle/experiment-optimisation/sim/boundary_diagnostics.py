"""Opt-in host-only fingerprints; diagnostic runs are not performance evidence."""

import hashlib
import json
import os
from pathlib import Path

import torch

CONTEXT = None
DECODE_COUNTS = {}


def enabled():
    return os.environ.get("QWEN_BOUNDARY_DIAGNOSTICS") == "1"


def fingerprint(tensor):
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
        raise ValueError("Diagnostics require an already materialized host tensor")
    data = tensor.detach().contiguous()
    return dict(shape=list(data.shape), dtype=str(data.dtype),
                sha256=hashlib.sha256(data.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


def emit(kind, **payload):
    if CONTEXT is not None:
        path = Path("/experiment/results/boundary-diagnostics.jsonl")
        with path.open("a") as stream:
            stream.write(json.dumps(dict(kind=kind, **CONTEXT, **payload)) + "\n")


def record_input(model_input, request_ids):
    global CONTEXT
    CONTEXT = None
    if not enabled():
        return
    ids = [request_id for request_id in request_ids if request_id is not None]
    if len(ids) != 1:
        raise ValueError("Boundary diagnostics require one active request")
    request_id = ids[0]
    decode = model_input.prompt_lens is None
    step = DECODE_COUNTS.get(request_id, 0)
    if decode:
        DECODE_COUNTS[request_id] = step + 1
        if step >= 3:
            return
    CONTEXT = dict(request_id=request_id, phase="decode" if decode else "prefill", decode_step=step)
    positions = model_input.input_positions.tolist()
    lengths = None if decode else [int(value) for value in model_input.prompt_lens]
    tokens = model_input.input_tokens
    emit("input", positions=positions, prompt_lens=lengths, tokens=fingerprint(tokens),
         first_tokens=tokens.reshape(-1)[:8].tolist())


def record_slot(slot, recurrent, convolution):
    if enabled() and CONTEXT is not None:
        emit("slot", slot=slot, recurrent=[fingerprint(state) for state in recurrent],
             convolution=[[fingerprint(state) for state in states] for states in convolution])


def record_output(logits):
    if not enabled() or CONTEXT is None:
        return
    if not isinstance(logits, torch.Tensor):
        raise ValueError("Expected host logits in diagnostic host-sampling mode")
    row = logits.reshape(-1, logits.shape[-1])[0].float()
    values, indices = torch.topk(row, min(8, row.numel()))
    emit("output", logits=fingerprint(logits), top_ids=indices.tolist(), top_values=values.tolist())
