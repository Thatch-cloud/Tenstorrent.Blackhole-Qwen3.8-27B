"""The A0 screen's drafters as walkers: DFlash2 (the control) and DSpark v2 behind tf_pair_walk's `propose(sequence, start, count)`.

A walker owns three things and nothing else: a BACKBONE (the draft network with its context K/V cache), the target's
embedding and LM head (the drafters share the target's), and the proposal rule on top of the block's hidden rows:

  DFlash2   block = anchor row + `count` mask rows; the anchor row is dropped; the selector's greedy path over the strict top-k
            of the unary logits gives `count` proposals.   (T16: count 15, T8: count 7)
  DSpark    block = anchor row + (count - 1) mask rows, `count` rows in all, every row's logits kept; the MARKOV greedy chain
            adds `predecessor[prev] @ successor.T` to each row's logits, takes the argmax and feeds it back, starting from the
            anchor. (T16: 15 rows, T8: 7 rows.) NEVER the checkpoint's own spec_generate: it skips the Markov head.

Backbones (one interface: reset(), block_hidden(features, noise_ids, start, window) -> [rows, H]):
  PortBackbone       dflash2_torch's plain-torch model with an incremental context-K/V cache; every context row is projected once
                     (rows older than the window are dropped); a `window` slices the context to rows at or after start - window
                     (exact: the keys were rotated at absolute positions). CPU-testable.
  UpstreamBackbone   the pinned upstream model (z-lab DFlash2DraftModel, DSpark's dflash.py DFlashDraftModel) with its own cache,
                     driven the way upstream's dflash_generate drives it. GPU host only; its V-gates run in the canary.

`features` is anything with `rows(a, b) -> [b - a, taps * H]` (a0_target.FeatureView).
"""
import torch
from torch.nn import functional as F

import dflash2_torch as d2

INGEST_CHUNK = 4096


def markov_chain(base_logits, anchor_id, predecessor, successor):
    """Greedy Markov chain, any device. base_logits [rows, V]; anchor_id int (the first predecessor); predecessor and successor
    [V, rank]. Row r: argmax(base_logits[r] + predecessor[prev] @ successor.T), then prev = that token. -> [rows] int64."""
    previous = torch.tensor([anchor_id], dtype=torch.int64, device=base_logits.device)
    out = []
    for row in range(base_logits.shape[0]):
        corrected = base_logits[row][None] + predecessor[previous] @ successor.T
        previous = torch.argmax(corrected, dim=-1)
        out.append(previous)
    return torch.cat(out)


class PortBackbone(object):
    """dflash2_torch.Dflash2 with an incremental context cache. `window`: None keeps every row (DSpark: full attention)."""

    def __init__(self, model, embed_weight, chunk=INGEST_CHUNK):
        self.model, self.embed, self.chunk = model, embed_weight, chunk
        self.reset()

    def reset(self):
        self.kv = None
        self.low = self.high = 0           # the cached context rows are [low, high)

    @property
    def device(self):
        return self.embed.device

    def _ingest(self, features, lo, hi):
        """Project rows [lo, hi) once and append them (positions are absolute)."""
        for a in range(lo, hi, self.chunk):
            b = min(hi, a + self.chunk)
            rows = features.rows(a, b)[None].to(self.embed.dtype)
            part = self.model.context_kv(rows, (a + torch.arange(b - a, device=rows.device))[None])
            if self.kv is None:
                self.kv = part
            else:
                self.kv = [(torch.cat([k0, k1], dim=2), torch.cat([v0, v1], dim=2)) for (k0, v0), (k1, v1) in zip(self.kv, part)]
            self.high = b

    def _trim(self, lo):
        if self.kv is not None and lo > self.low:
            cut = lo - self.low
            self.kv = [(k[:, :, cut:], v[:, :, cut:]) for k, v in self.kv]
            self.low = lo

    def block_hidden(self, features, noise_ids, start, window):
        lo = max(0, start - window) if window is not None else 0
        if self.kv is not None and (start < self.high or lo < self.low):
            self.reset()                              # a walk only moves forward; anything else starts over (exact, slower)
        if self.kv is not None and lo >= self.high:
            self.reset()                              # the window moved past every cached row
        if self.kv is None:
            self.low = self.high = lo
        else:
            self._trim(lo)
        self._ingest(features, self.high, start)
        ids = torch.as_tensor(noise_ids, dtype=torch.int64, device=self.device)[None]
        noise = F.embedding(ids, self.embed)
        positions = (start + torch.arange(ids.shape[1], device=self.device))[None]
        ctx_positions = (self.low + torch.arange(self.high - self.low, device=self.device))[None]
        with torch.no_grad():
            return self.model(noise, positions, ctx_positions=ctx_positions, ctx_kv=self.kv)[0]

    def release(self):
        self.kv = None


class UpstreamBackbone(object):
    """The pinned upstream draft model, driven like upstream's dflash_generate: context rows are passed once, as `target_hidden`
    with absolute position ids, into the model's own cache, which is cropped back to the context after each block.
    `model(position_ids=, noise_embedding=, target_hidden=, past_key_values=, use_cache=)`; `make_cache()` and `crop(cache, length)`
    are upstream's (_make_cache, _crop_to). Verified only on the GPU host, by V1 / V3 in the canary."""

    def __init__(self, model, embed_weight, make_cache, crop):
        self.model, self.embed, self.make_cache, self.crop = model, embed_weight, make_cache, crop
        self.reset()

    def reset(self):
        self.cache = self.make_cache()
        self.high = None
        self.length = 0

    def block_hidden(self, features, noise_ids, start, window):
        lo = max(0, start - window) if window is not None else 0
        if self.high is None:
            self.high = lo
        if start < self.high or lo > self.high:
            self.reset()                              # backward, or the window moved past everything cached: start over
            self.high = lo
        ids = torch.as_tensor(noise_ids, dtype=torch.int64, device=self.embed.device)[None]
        target = features.rows(self.high, start)[None].to(self.embed.dtype)
        positions = torch.arange(self.high, start + ids.shape[1], device=self.embed.device)[None]
        with torch.no_grad():
            hidden = self.model(position_ids=positions, noise_embedding=F.embedding(ids, self.embed), target_hidden=target,
                                past_key_values=self.cache, use_cache=True)[0]
        self.length += start - self.high
        self.crop(self.cache, self.length)
        self.high = start
        return hidden

    def release(self):
        self.cache = None


class Dflash2Walker(object):
    """The control: proposals by the selector's greedy path. `count` = proposals (15 for T16, 7 for T8)."""
    name = 'dflash2'

    def __init__(self, backbone, lm_head_weight, model, mask_token, count, window=2048):
        self.backbone, self.head, self.model, self.mask, self.count, self.window = backbone, lm_head_weight, model, mask_token, count, window
        self.features = None

    def begin(self, features):
        self.features = features
        self.backbone.reset()

    def propose(self, sequence, start, count):
        if count != self.count:
            raise ValueError('this walker was built for %d proposals' % self.count)
        anchor = int(sequence[start])
        noise = [anchor] + [self.mask] * count
        hidden = self.backbone.block_hidden(self.features, noise, start, self.window)[1:]
        with torch.no_grad():
            path = self.model.propose(hidden[None], torch.tensor([anchor], device=hidden.device), self.head)
        return path[0].tolist()

    def end(self):
        self.features = None
        self.backbone.reset()


class DSparkWalker(object):
    """The challenger: block rows = `count` (anchor + count - 1 masks), Markov greedy chain over every row's logits."""
    name = 'dspark'

    def __init__(self, backbone, lm_head_weight, predecessor, successor, mask_token, count, window=None):
        self.backbone, self.head, self.pred, self.succ = backbone, lm_head_weight, predecessor, successor
        self.mask, self.count, self.window = mask_token, count, window
        self.features = None

    def begin(self, features):
        self.features = features
        self.backbone.reset()

    def propose(self, sequence, start, count):
        if count != self.count:
            raise ValueError('this walker was built for %d proposals' % self.count)
        anchor = int(sequence[start])
        noise = [anchor] + [self.mask] * (count - 1)
        hidden = self.backbone.block_hidden(self.features, noise, start, self.window)
        with torch.no_grad():
            logits = hidden @ self.head.T
            return markov_chain(logits, anchor, self.pred, self.succ).tolist()

    def end(self):
        self.features = None
        self.backbone.reset()


# -- weights into the port, and the V4 comparison -----------------------------------------------------------------------------

def load_port_state(model, tensors, ignore=()):
    """Copy checkpoint tensors {name: tensor} into the port, refusing a mismatch in either direction (a missing or an extra
    tensor, or a shape that differs). `ignore`: checkpoint names the port does not hold (a confidence head)."""
    own = model.state_dict()
    given = dict((name, value) for name, value in tensors.items() if name not in ignore)
    if set(own) != set(given):
        raise ValueError('checkpoint and port tensor names differ: %d missing, %d extra' % (len(set(own) - set(given)),
                                                                                           len(set(given) - set(own))))
    for name, value in given.items():
        if tuple(own[name].shape) != tuple(value.shape):
            raise ValueError('shape of %s differs' % name)
    with torch.no_grad():
        for name, value in given.items():
            own[name].copy_(value.to(own[name].dtype))
    return model


def v4_round(gpu_hidden, cpu_hidden, head, predecessor, successor, anchor, answer_next, accepted):
    """One V4 round. `gpu_hidden` / `cpu_hidden` [rows, H]: the two devices' backbone rows for the same inputs; the chain rule is the
    Markov greedy chain. Each row is compared conditioned on the GPU's OWN previous token: the CPU row's corrected logits are formed
    with that token, and the row agrees when the CPU argmax equals the GPU token. -> ([(agree, top-2 margin)], (gpu accepted, cpu
    accepted)) with the accepted lengths against `answer_next` (the logged tokens after the anchor); `accepted(proposed, answer)`
    is the matching-prefix count."""
    gpu_logits = gpu_hidden.float() @ head.float().T
    gpu_tokens = markov_chain(gpu_logits, anchor, predecessor.float(), successor.float()).tolist()
    cpu_base = cpu_hidden.float() @ head.float().T
    cpu_chain = markov_chain(cpu_base, anchor, predecessor.float(), successor.float()).tolist()
    rows, previous = [], anchor
    for at, token in enumerate(gpu_tokens):
        corrected = cpu_base[at] + predecessor.float()[previous] @ successor.float().T
        top = torch.topk(corrected, 2)
        rows.append((int(top.indices[0]) == token, float(top.values[0] - top.values[1])))
        previous = token
    return rows, (accepted(gpu_tokens, answer_next), accepted(cpu_chain, answer_next))
