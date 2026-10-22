"""Test support for the sampled-verify tests (docs/tp4-sampled-verify.md): a fake four-chip mesh that implements the handful of ttnn calls the sampled path makes (slice, pad,
topk, deallocate, the per-shard and the batched reads), and a toy language model embedded in the real 248,320-token vocabulary so every chip and every 32,768-column chunk of the
shard holds candidates. Not imported by any served module (test only).

The fake topk is torch.topk on each chip's chunk: ties are returned in whatever order torch picks, which is the point - the host merge must make the result independent of the
device's tie order (up to the cut, which the tests state)."""

import os
import random

import numpy as np
import torch

import sampled_accept as sa

VOCAB = 248320
SHARD = 62080
CHIPS = 4
FILLER = -30.0

# --- the toy language model: 100 tokens spread over the whole vocabulary --------------------------------------------------------------------------------------------------
TOY = 100
STRIDE = 2483                        # 99 * 2483 + 7 = 245,824 < 248,320: toy tokens land on every chip and in both chunks of every shard
OFFSET = 7
EOS = OFFSET + (TOY - 1) * STRIDE
SPECS = (sa.SamplingSpec(), sa.SamplingSpec(0.7, 0.9, 8), sa.SamplingSpec(1.0, 1.0, 0), sa.SamplingSpec(0.0, 1.0, 0))


def phi(token):
    return OFFSET + token * STRIDE


def toy_of(identifier):
    return (identifier - OFFSET) // STRIDE


def toy_scores(context):
    """The 100 toy tokens' logits after `context` (vocabulary ids): a function of the last two tokens, on a grid bf16 holds exactly (so ties occur)."""
    a, b = (toy_of(context[-2]), toy_of(context[-1])) if len(context) > 1 else (0, toy_of(context[-1]))
    rng = np.random.default_rng((a * 1000003 + b * 7919 + 12345) & 0xFFFFFFFF)
    scores = rng.normal(size=TOY) * 2.5
    scores[rng.integers(0, TOY, 3)] += 4.0
    return np.round(scores * 8) / 8


def full_row(context):
    """The model's logits after `context` over the whole vocabulary: the toy tokens' scores, FILLER everywhere else."""
    row = np.full(VOCAB, FILLER, dtype=np.float32)
    row[[phi(token) for token in range(TOY)]] = toy_scores(context)
    return row


def toy_sample(context, spec, seed):
    return sa.sample_logits(full_row(context), spec, seed, len(context))


def plain_run(prompt, count, seed, spec):
    """One-token-at-a-time sampling with the same seed: the text a speculative run must commit."""
    context, out = list(prompt), []
    while len(out) < count:
        token = toy_sample(context, spec, seed)
        out.append(token)
        context.append(token)
        if token == EOS:
            break
    return out


def approx_drafter(history, count):
    """The drafter's top choice under a perturbed copy of the model: usually right, sometimes not. Deterministic in the history (a function of the committed prefix)."""
    context, drafts = list(history), []
    for _ in range(count):
        noise = np.random.default_rng((toy_of(context[-1]) * 31 + len(context)) & 0xFFFF).normal(size=TOY) * 0.9
        token = phi(int(np.argmax(toy_scores(context) + noise)))
        drafts.append(token)
        context.append(token)
    return drafts


def peaked_drafter(history, count):
    context, drafts = list(history), []
    for _ in range(count):
        token = phi(int(np.argmax(toy_scores(context))))
        drafts.append(token)
        context.append(token)
    return drafts


def hash_drafter(history, count):
    rng = random.Random(len(history) * 7 + history[-1])
    return [phi(rng.randrange(TOY)) for _ in range(count)]


def copy_drafter(history, count):
    key = tuple(history[-2:])
    for start in range(len(history) - 3, -1, -1):
        if tuple(history[start:start + 2]) == key:
            return list(history[start + 2:start + 2 + count])
    return []


DRAFTERS = dict(approx=approx_drafter, peaked=peaked_drafter, hash=hash_drafter, copy=copy_drafter)


# --- the fake mesh --------------------------------------------------------------------------------------------------------------------------------------------------------

class Tensor(object):
    """A mesh tensor: one (1, 1, rows, cols) torch tensor per chip."""

    def __init__(self, chips, dtype='bf16', layout='tile'):
        self.chips = [chip.clone() for chip in chips]
        self.shape, self.dtype, self.layout = tuple(self.chips[0].shape), dtype, layout
        self.freed = False


class Shard(object):
    def __init__(self, values):
        self.values = values


class Host(object):
    def __init__(self, shape, dtype, layout):
        self.shape, self.dtype, self.layout = shape, dtype, layout
        self.chips = None


class RefusedTopk(RuntimeError):
    pass


class Ops(object):
    """The ttnn calls the sampled path and its readbacks make, counting what the host asks of the mesh."""
    bfloat16, TILE_LAYOUT = 'bf16', 'tile'

    def __init__(self, k_limit=None):
        self.k_limit = k_limit
        self.single_reads = self.composed_reads = self.copies = self.fences = 0
        self.slices, self.pads, self.topks, self.freed = [], [], [], []
        self.live = []

    def _new(self, chips, dtype='bf16', layout='tile'):
        tensor = Tensor(chips, dtype, layout)
        self.live.append(tensor)
        return tensor

    def slice(self, tensor, start, end, step):
        self.slices.append((tuple(start), tuple(end)))
        return self._new([chip[:, :, start[2]:end[2], start[3]:end[3]] for chip in tensor.chips], tensor.dtype, tensor.layout)

    def pad(self, tensor, padding, value):
        self.pads.append((tuple(map(tuple, padding)), value))
        (_, _), (_, _), (_, _), (before, after) = [tuple(p) for p in padding]
        return self._new([torch.nn.functional.pad(chip.float(), (before, after), value=value).bfloat16() for chip in tensor.chips], tensor.dtype, tensor.layout)

    def topk(self, tensor, k, dim=-1, largest=True, sorted=True):
        self.topks.append((tensor.shape, k))
        if self.k_limit is not None and k > self.k_limit:
            raise RefusedTopk('TT_FATAL: k=%d is not supported (at most %d)' % (k, self.k_limit))
        values, indices = [], []
        for chip in tensor.chips:
            top = torch.topk(chip.float(), k, dim=-1, largest=largest, sorted=sorted)
            values.append(top.values.bfloat16())
            indices.append(top.indices.to(torch.int32))
        return self._new(values, 'bf16'), self._new(indices, 'uint16')

    def deallocate(self, tensor):
        tensor.freed = True
        self.freed.append(tensor)

    def get_device_tensors(self, tensor):
        return [Shard(chip) for chip in tensor.chips]

    def to_torch(self, value, mesh_composer=None):
        if isinstance(value, Shard):
            self.single_reads += 1
            return value.values.clone()
        if isinstance(value, Host):
            return torch.cat(value.chips, dim=mesh_composer[1])
        self.composed_reads += 1
        return torch.cat([chip.clone() for chip in value.chips], dim=mesh_composer[1])

    def ConcatMeshToTensor(self, mesh, dim=0):
        return ('composer', dim)

    def allocate_tensor_on_host(self, shape, dtype, layout, mesh):
        return Host(tuple(shape), dtype, layout)

    def copy_device_to_host_tensor(self, tensor, host, blocking=True, cq_id=None):
        assert blocking is False
        self.copies += 1
        host.chips = [chip.clone() for chip in tensor.chips]

    def synchronize_device(self, mesh):
        self.fences += 1


def logits_tensor(rows_by_chip_free_array):
    """The pre-gather logits tensor of a block from a (rows, 248,320) float array: chip c holds columns [c * 62080, (c + 1) * 62080) as bf16."""
    array = np.asarray(rows_by_chip_free_array, dtype=np.float32)
    rows = array.shape[0]
    chips = [torch.from_numpy(np.ascontiguousarray(array[:, chip * SHARD:(chip + 1) * SHARD])).reshape(1, 1, rows, SHARD).bfloat16() for chip in range(CHIPS)]
    return Tensor(chips, 'bf16', 'tile')


def clean_environment(extra=None):
    """The process environment without any QWEN_FAST_ name, plus `extra` (for patch.dict(clear=True))."""
    environ = {name: value for name, value in os.environ.items() if not name.startswith('QWEN_FAST_')}
    environ.update(extra or {})
    return environ
